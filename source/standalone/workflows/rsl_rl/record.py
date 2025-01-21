# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from datetime import datetime
import pickle
from scipy.io import savemat
import matplotlib.pyplot as plt


def quat2yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return yaw


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # Generate unique data tag
    date_str, time_str = datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime('%H-%M-%S')
    output_path = os.path.join("outputs", "planner_tracker", date_str, time_str)
    os.makedirs(output_path, exist_ok=True)
    for i in range(env_cfg.num_iterations):
        # reset environment
        with torch.inference_mode():
            env.reset()
        obs, info = env.get_observations()
        timestep = 0

        # data structures to save data
        N = int(env_cfg.iter_duration_s / env.unwrapped.step_dt)
        done_t = torch.zeros((N - 1, obs.shape[0]), dtype=torch.bool, device=env.device)
        obs_t = torch.zeros((N, *obs.shape), device=env.device)
        action_t = torch.zeros((N - 1, *env.unwrapped.action_space.shape))
        z = torch.zeros((N - 1, obs.shape[0], 3), device=env.device)
        v = torch.zeros((N - 1, obs.shape[0], 3), device=env.device)
        p_x = torch.zeros((N - 1, obs.shape[0], 3), device=env.device)
        obs_t[0] = obs

        # simulate environment
        while simulation_app.is_running() and timestep < N - 1:
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, rew, dones, info = env.step(actions)

            obs_t[timestep + 1] = obs.detach().clone()
            action_t[timestep] = actions.detach().clone()
            z[timestep] = env.unwrapped.command_manager._terms['base_velocity'].trajectory[:, 0].detach().clone()
            v[timestep] = env.unwrapped.command_manager._terms['base_velocity'].v_trajectory[:, 0].detach().clone()
            p_x[timestep, :, :2] = info['observations']['tracking'][:, :2].detach().clone()
            p_x[timestep, :, 2] = quat2yaw(info['observations']['tracking'][:, 3:]).detach().clone()
            done_t[timestep] = dones.detach().clone()

            timestep += 1
            if args_cli.video:
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

        fn_pickle = os.path.join(output_path, f"data_{i}.pickle")
        fn_mat = os.path.join(output_path, f"data_{i}.mat")

        data_dict = {"obs": obs_t, "act": action_t, "p_x": p_x, "z": z, "v": v, "done": done_t}
        with open(fn_pickle, 'wb') as f:
            pickle.dump(data_dict, f)

        savemat(
            fn_mat,
            {key: value.cpu().numpy() for key, value in data_dict.items()}
        )

        rbt_ind = 5
        fig, ax = plt.subplots(1, 2)
        z_plt = z[:, rbt_ind, :].cpu().numpy()
        px_plt = p_x[:, rbt_ind, :].cpu().numpy()
        ax[0].plot(z_plt[:, 0], z_plt[:, 1])
        ax[0].plot(px_plt[:, 0], px_plt[:, 1])
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].legend(['Planner', 'Tracker'])
        ax[1].plot(px_plt[:, 0] - z_plt[:, 0])
        ax[1].plot(px_plt[:, 1] - z_plt[:, 1])
        yaw_err = px_plt[:, 2] - z_plt[:, 2]
        ax[1].plot((yaw_err + torch.pi) % (2 * torch.pi) - torch.pi)
        ax[1].legend(['e_x', 'e_y', 'e_heading'])
        plt.show()
        plt.close('all')

    # close the simulator
    print(f"Data saved in: {output_path}")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
