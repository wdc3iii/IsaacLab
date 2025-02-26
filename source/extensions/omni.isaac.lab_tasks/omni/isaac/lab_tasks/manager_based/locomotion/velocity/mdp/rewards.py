# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from typing import Callable


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float, stand_threshold: float=0.1
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > stand_threshold
    return reward


def feet_air_time_positive_biped(
        env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg, stand_threshold: float=0.1
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > stand_threshold
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def foot_height(env, command_name: str, asset_cfg: SceneEntityCfg, max_height: float, stand_threshold: float) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_h = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    reward = torch.sum(torch.clamp(foot_h, 0, max_height), dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > stand_threshold
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

# Phase/Tracking based rewards
def get_time_phase(t, period, offset):
    return 2 * torch.pi * t / period + offset

def get_phase(t, period, offset):
    return torch.sin(get_time_phase(t, period, offset))

def is_stance_phase(t, period, offset):
    phase = get_phase(t, period, offset)
    return phase < 0

# Incentivize contact to align with desired gait
def phase_based_contact(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, period: float,
    stand_threshold: float, get_gait_offset: Callable, get_gait_args: dict
) -> torch.Tensor:
    # Get contacts
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    # Get commands
    gait_phase_offset = get_gait_offset(env, **get_gait_args)
    vel_cmd = env.command_manager.get_command(command_name)

    # Compute desired contacts
    desired_contacts = is_stance_phase(env.sim.current_time, period, gait_phase_offset)
    desired_contacts[torch.norm(vel_cmd, dim=-1) < stand_threshold, :] = True

    correct_contacts = desired_contacts == contacts
    return torch.mean(correct_contacts.float(), dim=-1)

# Incentivize tracking a particular swing foot trajectory (quadruped)
def swing_foot_height(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, swing_height: float,
    period: float, stand_threshold: float,  std: float, get_gait_offset: Callable, get_gait_args: dict
) -> torch.Tensor:
    # Get commands
    gait_phase_offset = get_gait_offset(env, **get_gait_args)
    vel_cmd = env.command_manager.get_command(command_name)

    # Get foot height
    asset = env.scene[asset_cfg.name]
    foot_h = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # Get desired foot height
    def cubic_bezier_interpolation(z_start, z_end, t):
        t = torch.clamp(t, 0, 1)
        z_diff = z_end - z_start
        bezier = t ** 3 + 3 * (t ** 2 * (1 - t))
        return z_start + z_diff * bezier

    t_phase = get_time_phase(env.sim.current_time, period, gait_phase_offset) % (2 * torch.pi)
    t_phase = t_phase / torch.pi
    des_foot_h = torch.where(
        t_phase <= 0.5,
        cubic_bezier_interpolation(0, swing_height, 2 * t_phase),       # Upward portion of swing
        cubic_bezier_interpolation(swing_height, 0, 2 * t_phase - 1)     # downward portion of swing
    )
    # If foot should be in stance, zero reward signal
    desired_contacts = is_stance_phase(env.sim.current_time, period, gait_phase_offset)
    desired_contacts[torch.norm(vel_cmd, dim=-1) < stand_threshold, :] = True
    des_foot_h[desired_contacts] = 0

    # Compute foot height rewards
    height_error = torch.square(foot_h - des_foot_h)
    rew = torch.exp(-height_error / std**2)

    return torch.mean(rew, dim=-1)

def single_joint_trajectory(env, command_name: str, asset_cfg: SceneEntityCfg, period: float,
                       joint_times: torch.Tensor, joint_trajectory: torch.Tensor, stand_threshold: float,
                       std: float, get_gait_offset: Callable, get_gait_args: dict) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_cmd = env.command_manager.get_command(command_name)
    gait_phase_offset = get_gait_offset(env, **get_gait_args)

    q_pos = asset.data.joint_pos.detach().clone()

    # interpolate joint positions
    t_phase = get_time_phase(env.sim.current_time, period, gait_phase_offset) % (2 * torch.pi)
    t_phase = t_phase / torch.pi
    t_phase = t_phase.repeat(1, 3)
    idxs = torch.searchsorted(joint_times, t_phase.clamp(joint_times[0], joint_times[-1])) - 1
    col_indices = torch.arange(idxs.shape[1]).expand(idxs.shape[0], -1)
    # Linear interpolation
    t0, t1 = joint_times[idxs], joint_times[torch.clamp(idxs + 1, 0, joint_times.shape[0] - 1)]
    w = (t_phase - t0) / (t1 - t0)
    q_des = torch.where(
        t_phase < 1,
        (1 - w) * joint_trajectory[idxs, col_indices] + w * joint_trajectory[idxs + 1, col_indices],
        joint_trajectory[-1]
    )  # - asset.data.default_joint_pos
    q_des[torch.norm(vel_cmd, dim=-1) < stand_threshold, :] = joint_trajectory[0]
    error = q_pos - q_des
    return torch.mean(torch.exp(-torch.square(error)/ std**2), dim=-1)

# TODO: check interpolation functions
def interp_joint_trajectory(
        env, command_name: str, asset_cfg: SceneEntityCfg, ts: torch.Tensor,
        q_refs: torch.Tensor, v_xs: torch.Tensor, v_ys: torch.Tensor, w_zs: torch.Tensor,
        std: float, get_gait_offset: Callable, get_gait_args: dict) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_cmd = env.command_manager.get_command(command_name)
    gait_phase_offset = get_gait_offset(env, **get_gait_args)
    period = torch.max(ts)

    q_pos = asset.data.joint_pos.detach().clone()
    q_des = torch.zeros_like(q_pos)

    # Get interpolation stuff
    dv_x = v_xs[1] - v_xs[0]
    vx_low = torch.floor((vel_cmd[:, 0] - v_xs[0]) / dv_x).int()
    vx_high = torch.clip(vx_low + 1, 0, v_xs.shape[0] - 1)
    vx_low = torch.clip(vx_low, 0, v_xs.shape[0] - 1)

    dv_y = v_ys[1] - v_ys[0]
    vy_low = torch.floor((vel_cmd[:, 0] - v_ys[0]) / dv_y).int()
    vy_high = torch.clip(vy_low + 1, 0, v_ys.shape[0] - 1)
    vy_low = torch.clip(vy_low, 0, v_ys.shape[0] - 1)

    dw_z = w_zs[1] - w_zs[0]
    wz_low = torch.floor((vel_cmd[:, 0] - w_zs[0]) / dw_z).int()
    wz_high = torch.clip(wz_low + 1, 0, w_zs.shape[0] - 1)
    wz_low = torch.clip(wz_low, 0, w_zs.shape[0] - 1)

    vx_inds = torch.cat((
        vx_low[:, None].repeat(1, 8),
        vx_high[:, None].repeat(1, 8)
    ), dim=-1)
    vy_inds = torch.cat((
        vy_low[:, None].repeat(1, 4),
        vy_high[:, None].repeat(1, 4)
    ), dim=-1).repeat(1, 2)
    wz_inds = torch.cat((
        wz_low[:, None].repeat(1, 2),
        wz_high[:, None].repeat(1, 2)
    ), dim=-1).repeat(1, 4)

    dt = ts[1] - ts[0]

    # interpolate joint positions
    for foot_id in range(4):
        t = (env.sim.current_time + gait_phase_offset[:, foot_id]) % period
        ts_low = torch.floor((t - ts[0]) / dt).int()
        ts_high = torch.clip(ts_low + 1, 0, ts.shape[0] - 1)
        ts_low = torch.clip(ts_low, 0, ts.shape[0] - 1)

        t_inds = torch.cat((
            ts_low[:, None],
            ts_high[:, None]
        ), dim=-1).repeat(1, 8)

        vecs = torch.dstack([v_xs[vx_inds], v_ys[vy_inds], w_zs[wz_inds], ts[t_inds]])
        des_vec = torch.hstack((vel_cmd, torch.ones((vel_cmd.shape[0], 1), device=vel_cmd.device) * t[:, None]))[:, None, :]
        dist = torch.linalg.norm(vecs - des_vec, axis=-1) + 1e-4
        weights = 1 / dist
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)

        # TODO: index q_refs properly
        q_des[:, 3 * foot_id: 3 * (foot_id + 1)] = torch.sum(
            weights[:, :, None] * q_refs[vx_inds, vy_inds, wz_inds, t_inds, 3 * foot_id:3 * (foot_id + 1)]
        , dim=1)
    error = q_pos - q_des
    return torch.mean(torch.exp(-torch.square(error) / std ** 2), dim=-1)


    # Incentivize foot to be under the hip.
def single_foot_pos_xy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, foot_xy_des: torch.tensor, std: float
):
    # Get foot height
    asset = env.scene[asset_cfg.name]
    foot_xy_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :2] - asset.data.root_pos_w[:, :2][:, None, :]
    heading = asset.data.heading_w[:, None]
    foot_xy_b = torch.stack([
        foot_xy_w[:, :, 0] * torch.cos(heading) + foot_xy_w[:, :, 1] * torch.sin(heading),
        -foot_xy_w[:, :, 0] * torch.sin(heading) + foot_xy_w[:, :, 1] * torch.cos(heading)
    ], dim=-1)
    foot_error = foot_xy_b - foot_xy_des.to(env.device)
    foot_error = torch.sum(foot_error * foot_error, dim=-1)

    return torch.mean(torch.exp(-foot_error / std**2), dim=-1)

def interp_foot_pos(
        env, command_name: str, asset_cfg: SceneEntityCfg,
        ts: torch.Tensor, foot_refs: torch.Tensor, v_xs: torch.Tensor, v_ys: torch.Tensor,
        w_zs: torch.Tensor, std: float, get_gait_offset: Callable, get_gait_args: dict) -> torch.Tensor:
    # Get foot height
    asset = env.scene[asset_cfg.name]
    foot_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_w[:, :, :2] -= asset.data.root_pos_w[:, :2][:, None, :]
    vel_cmd = env.command_manager.get_command(command_name)
    gait_phase_offset = get_gait_offset(env, **get_gait_args)
    period = torch.max(ts)

    # Get interpolation stuff
    # Assume v_x is linearly spaced
    dv_x = v_xs[1] - v_xs[0]
    vx_low = torch.floor((vel_cmd[:, 0] - v_xs[0]) / dv_x).int()
    vx_high = torch.clip(vx_low + 1, 0, v_xs.shape[0] - 1)
    vx_low = torch.clip(vx_low, 0, v_xs.shape[0] - 1)

    dv_y = v_ys[1] - v_ys[0]
    vy_low = torch.floor((vel_cmd[:, 0] - v_ys[0]) / dv_y).int()
    vy_high = torch.clip(vy_low + 1, 0, v_ys.shape[0] - 1)
    vy_low = torch.clip(vy_low, 0, v_ys.shape[0] - 1)

    dw_z = w_zs[1] - w_zs[0]
    wz_low = torch.floor((vel_cmd[:, 0] - w_zs[0]) / dw_z).int()
    wz_high = torch.clip(wz_low + 1, 0, w_zs.shape[0] - 1)
    wz_low = torch.clip(wz_low, 0, w_zs.shape[0] - 1)

    vx_inds = torch.cat((
        vx_low[:, None].repeat(1, 8),
        vx_high[:, None].repeat(1, 8)
    ), dim=-1)
    vy_inds = torch.cat((
        vy_low[:, None].repeat(1, 4),
        vy_high[:, None].repeat(1, 4)
    ), dim=-1).repeat(1, 2)
    wz_inds = torch.cat((
        wz_low[:, None].repeat(1, 2),
        wz_high[:, None].repeat(1, 2)
    ), dim=-1).repeat(1, 4)

    foot_des = torch.zeros_like(foot_w)

    dt = ts[1] - ts[0]

    # interpolate joint positions
    for foot_id in range(4):
        t = (env.sim.current_time + gait_phase_offset[:, foot_id]) % period
        ts_low = torch.floor((t - ts[0]) / dt).int()
        ts_high = torch.clip(ts_low + 1, 0, ts.shape[0] - 1)
        ts_low = torch.clip(ts_low, 0, ts.shape[0] - 1)

        t_inds = torch.cat((
            ts_low[:, None],
            ts_high[:, None]
        ), dim=-1).repeat(1, 8)

        vecs = torch.dstack([v_xs[vx_inds], v_ys[vy_inds], w_zs[wz_inds], ts[t_inds]])
        des_vec = torch.hstack((vel_cmd, torch.ones((vel_cmd.shape[0], 1), device=vel_cmd.device) * t[:, None]))[:, None, :]
        dist = torch.linalg.norm(vecs - des_vec, axis=-1) + 1e-4
        weights = 1 / dist
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)

        foot_des[:, foot_id, :] = torch.sum(weights[:, :, None] * foot_refs[vx_inds, vy_inds, wz_inds, t_inds, foot_id, :], dim=1)
    heading = asset.data.heading_w[:, None]
    foot_b = torch.stack([
        foot_w[:, :, 0] * torch.cos(heading) + foot_w[:, :, 1] * torch.sin(heading),
        -foot_w[:, :, 0] * torch.sin(heading) + foot_w[:, :, 1] * torch.cos(heading),
        foot_w[:, :, 2]
    ], dim=-1)
    foot_error = foot_b - foot_des
    foot_error = torch.sum(foot_error * foot_error, dim=-1)

    return torch.mean(torch.exp(-foot_error / std**2), dim=-1)