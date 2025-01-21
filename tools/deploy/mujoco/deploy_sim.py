import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

ROOT_DIR = "/home/wcompton/repos/IsaacLab"
UNITREE_MUJOCO_DIR = "/home/wcompton/repos/unitree_mujoco"
def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{ROOT_DIR}/tools/deploy/mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{ROOT_DIR}", ROOT_DIR)
        xml_path = config["xml_path"].replace("{UNITREE_MUJOCO_DIR}", UNITREE_MUJOCO_DIR)
        task = config["task"]
        isaac2mujoco = config["isaac2mujoco"]
        mujoco2isaac = list(range(len(isaac2mujoco)))
        for n, i in enumerate(isaac2mujoco):
            mujoco2isaac[i] = n

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32) * 1
        kds = np.array(config["kds"], dtype=np.float32) * 1

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    print([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(12)])
    print([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(12)])

    d.qpos[7:] = default_angles
    d.qpos[2] = 0.4

    # load policy
    policy = torch.load(policy_path)

    N = 2000
    q = np.zeros((N, d.qpos.shape[0]))
    dq = np.zeros((N, d.qvel.shape[0]))
    targ_q = np.zeros((N, 12))
    tau_t = np.zeros((N, 12))

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration and counter < N:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau

            q[counter] = d.qpos
            dq[counter] = d.qvel
            targ_q[counter] = target_dof_pos
            tau_t[counter] = tau

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                v = d.qvel[:3]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                obs[:3] = v
                obs[3:6] = omega
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12: 12 + num_actions] = qj[mujoco2isaac]
                obs[12 + num_actions: 12 + 2 * num_actions] = dqj[mujoco2isaac]
                obs[12 + 2 * num_actions: 12 + 3 * num_actions] = action
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                action = action[isaac2mujoco]

                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    import matplotlib.pyplot as plt
    for i in range(12):
        plt.figure()
        plt.plot(np.arange(N) * simulation_dt, q[:, 7 + i])
        # plt.plot(np.arange(N) * simulation_dt, q[:, i])
        plt.plot(np.arange(N) * simulation_dt, targ_q[:, i])
        plt.legend(['q', 'targ'])
        plt.show()