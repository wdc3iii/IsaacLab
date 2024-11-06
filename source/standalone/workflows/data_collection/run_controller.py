# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--controller", type=str, default=None, help="Name of the controller.")
parser.add_argument("--no_data", action="store_true", default=False, help="Disable data recording.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of environment steps to simulate.")

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
from datetime import datetime

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg, parse_controller_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    controller_cfg = parse_controller_cfg(
        args_cli.controller
    )

    log_root_path = os.path.join("logs", "data_collection", controller_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if controller_cfg.run_name:
        log_dir += f"_{controller_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Generate controller
    controller = controller_cfg.create_controller(args_cli.task, env.unwrapped.num_envs, env.unwrapped.device)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "controller.yaml"), controller_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "controller.pkl"), controller_cfg)

    # reset environment
    obs, _ = env.reset()
    robot_data = {name: torch.zeros(obs[name].shape[0], args_cli.num_steps + 1, obs[name].shape[1]) for name in obs.keys()}
    save_robot_data(0, obs, robot_data)
    timestep = 0
    env_step = 0
    # simulate environment
    while simulation_app.is_running() and env_step < args_cli.num_steps:
        env_step += 1
        print(env_step / args_cli.num_steps)
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = controller(obs)
            # env stepping
            obs, _, _, _, _ = env.step(actions)

        if not args_cli.no_data:
            save_robot_data(env_step, obs, robot_data)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # save the data
    import scipy.io as sio
    sio.savemat(os.path.join(log_dir, "data.mat"), {k: v.cpu().numpy() for k, v in robot_data.items()})
    # close the simulator
    env.close()

def save_robot_data(i, obs, robot_data):
    for name in obs.keys():
        robot_data[name][:, i, :] = obs[name]

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
