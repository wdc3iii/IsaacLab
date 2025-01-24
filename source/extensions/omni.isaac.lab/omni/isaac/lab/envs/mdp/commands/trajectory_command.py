# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import TrajectoryCommandCfg


class TrajectoryCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: TrajectoryCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: TrajectoryCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)
        self.env_origins = env.scene.env_origins
        self.v_min = torch.tensor([
            cfg.planning_ranges.lin_vel_x[0],
            cfg.planning_ranges.lin_vel_y[0],
            cfg.planning_ranges.ang_vel_z[0]
        ], device=self.device)
        self.v_max = torch.tensor([
            cfg.planning_ranges.lin_vel_x[1],
            cfg.planning_ranges.lin_vel_y[1],
            cfg.planning_ranges.ang_vel_z[1]
        ], device=self.device)
        self.fb_v_min = torch.tensor([cfg.ranges.lin_vel_x[0], cfg.ranges.lin_vel_y[0], cfg.ranges.ang_vel_z[0]],
                                  device=self.device)
        self.fb_v_max = torch.tensor([cfg.ranges.lin_vel_x[1], cfg.ranges.lin_vel_y[1], cfg.ranges.ang_vel_z[1]],
                                  device=self.device)
        self.K = torch.tensor(
            [cfg.parallel_control_stiffness, cfg.perpendicular_control_stiffness, cfg.heading_control_stiffness],
            device=self.device
        )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.trajectory = torch.zeros(self.num_envs, self.cfg.N + 1, 3, device=self.device)
        self.v_trajectory = torch.zeros(self.num_envs, self.cfg.N, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.t = torch.zeros(self.num_envs, device=self.device)
        self.last_step_t = torch.zeros(self.num_envs, device=self.device)
        self.k = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        # Weights perform linear combination of various commands
        self.weights = torch.zeros((self.num_envs, 3, 5), device=self.device)
        self.prop_v_max = torch.zeros((self.num_envs, 3), device=self.device)
        self.sample_hold_input = torch.zeros((self.num_envs, 3), device=self.device)
        self.extreme_input = torch.zeros((self.num_envs, 3), device=self.device)
        self.ramp_t_start = torch.zeros((self.num_envs, 3), device=self.device)
        self.ramp_v_start = torch.zeros((self.num_envs, 3), device=self.device)
        self.ramp_v_end = self.rnd_vec(self.num_envs * 3).reshape(self.num_envs, 3)
        self.sin_mag = torch.zeros((self.num_envs, 3), device=self.device)
        self.sin_freq = torch.zeros((self.num_envs, 3), device=self.device)
        self.sin_off = torch.zeros((self.num_envs, 3), device=self.device)
        self.sin_mean = torch.zeros((self.num_envs, 3), device=self.device)
        self.exp_c = torch.zeros((self.num_envs, 3), device=self.device)
        self.exp_alpha = torch.zeros((self.num_envs, 3), device=self.device)
        self.exp_t_start = torch.zeros((self.num_envs, 3), device=self.device)
        self.rnd_mag = torch.zeros((self.num_envs, 3), device=self.device)
        self.rnd_inds = torch.zeros((self.num_envs, 3), dtype=torch.bool, device=self.device)
        self.noise_std = torch.zeros((self.num_envs, 3), device=self.device)
        self.t_rem_weights = torch.zeros((self.num_envs, 3), device=self.device)
        self.t_rem_prop_v_max = torch.zeros((self.num_envs, 3), device=self.device)
        self.t_rem_sample_hold = torch.zeros((self.num_envs, 3), device=self.device)
        self.t_rem_extreme = torch.zeros((self.num_envs, 3), device=self.device)
        self.t_rem_ramp = torch.zeros((self.num_envs, 3), device=self.device)
        self.t_ramp_start = torch.zeros((self.num_envs, 3), device=self.device)
        self.t_rem_sin = torch.zeros((self.num_envs, 3), device=self.device)
        self.t_rem_exp = torch.zeros((self.num_envs, 3), device=self.device)
        self.t_exp_start = torch.zeros((self.num_envs, 3), device=self.device)
        self.t_rem_rnd = torch.zeros((self.num_envs, 3), device=self.device)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "TrajectoryCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Overrides
    """
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command generator and log metrics.

        This function resets the command counter and resamples the command. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "{name}" key.
        """
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)

        # set the command counter to zero
        self.command_counter[env_ids] = 0
        # resample the command
        self.trajectory[env_ids, :, :] = torch.zeros((len(env_ids), self.cfg.N + 1, 3),
                                                     device=self.device)
        self.v_trajectory[env_ids, :, :] = torch.zeros((len(env_ids), self.cfg.N, 3), device=self.device)

        self.trajectory[env_ids, -1, :2] = self.robot.data.root_pos_w[env_ids, :2].detach().clone() - self.env_origins[env_ids, :2]
        self.trajectory[env_ids, -1, 2] = self._quat2yaw(self.robot.data.root_quat_w[env_ids].detach().clone())
        # Add randomization to IC
        self.trajectory[env_ids, -1, :2] += self.rnd_vec((len(env_ids), 2), -self.cfg.init_pos_rnd, self.cfg.init_pos_rnd)
        self.trajectory[env_ids, -1, 2] += self.rnd_vec((len(env_ids),), -self.cfg.init_heading_rnd, self.cfg.init_heading_rnd)

        self.k[env_ids] = -self.cfg.N
        self.t[env_ids] = self.k[env_ids] * self.cfg.dt * self.cfg.rel_dt
        self.last_step_t[env_ids] = 0
        self.t_rem_weights[env_ids] = -1
        self.t_rem_prop_v_max[env_ids] = -1
        self.t_rem_sample_hold[env_ids] = -1
        self.t_rem_extreme[env_ids] = -1
        self.t_rem_ramp[env_ids] = -1
        self.t_rem_sin[env_ids] = -1
        self.t_rem_exp[env_ids] = -1
        self.t_rem_rnd[env_ids] = -1
        self._resample_command(env_ids)
        for _ in range(self.cfg.N):
            self._step_rom_idx(env_ids)
            self.t[env_ids] += self.cfg.dt * self.cfg.rel_dt
        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0
        return extras

    """
    Implementation specific functions.
    """
    def rnd_vec(self, sz, low=-1., high=1.):
        return torch.rand(sz, device=self.device) * (high - low) + low

    @staticmethod
    def _quat2yaw(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return yaw

    def _update_metrics(self):
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) * self._env.step_dt
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) * self._env.step_dt
        )

    def _resample_command(self, env_ids):
        env_mask = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.bool)
        env_mask[env_ids, :] = True
        self._resample_sample_hold_input(env_mask)
        self._resample_extreme_input(env_mask)
        self._resample_ramp_input(env_mask)
        self._resample_sinusoid_input(env_mask)
        self._resample_exp_input(env_mask)
        self._resample_rnd_input(env_mask)
        self._resample_weight(env_mask)
        self._resample_prop_v_max(env_mask)

    def _resample_sample_hold_input(self, env_mask):
        idx = torch.logical_and(env_mask, self.t_rem_sample_hold < 0)
        if torch.any(idx):
            self.sample_hold_input[idx] = self.rnd_vec(torch.sum(idx))
            self.t_rem_sample_hold[idx] = self.rnd_vec(torch.sum(idx), self.cfg.t_min, self.cfg.t_max)

    def _resample_extreme_input(self, env_mask):
        idx = torch.logical_and(env_mask, self.t_rem_extreme < 0)
        if torch.any(idx):
            self.extreme_input[idx] = torch.round(self.rnd_vec(torch.sum(idx), -1.5, 1.5)) # round (-1.5, 1.5) gives uniform over [-1, 0, 1]
            self.t_rem_extreme[idx] = self.rnd_vec(torch.sum(idx), self.cfg.t_min, self.cfg.t_max)

    def _resample_ramp_input(self, env_mask):
        idx = torch.logical_and(env_mask, self.t_rem_ramp < 0)
        if torch.any(idx):
            self.ramp_v_start[idx] = self.ramp_v_end[idx]
            self.ramp_v_end[idx] = self.rnd_vec(torch.sum(idx))
            self.t_ramp_start[idx] = self.t[:, None].repeat((1, 3))[idx]
            self.t_rem_ramp[idx] = self.rnd_vec(torch.sum(idx), self.cfg.t_min, self.cfg.t_max)
            mask = torch.rand(idx.shape, device=self.device) >= 0.1
            idx[mask] = False
            self.ramp_v_end[idx] = torch.round(self.ramp_v_end[idx] * 1.5) # round (-1.5, 1.5) gives uniform over [-1, 0, 1]

    def _resample_sinusoid_input(self, env_mask):
        idx = torch.logical_and(env_mask, self.t_rem_sin < 0)
        if torch.any(idx):
            self.sin_mag[idx] = self.rnd_vec(torch.sum(idx))
            self.sin_mean[idx] = self.rnd_vec(torch.sum(idx))
            self.sin_freq[idx] = self.rnd_vec(torch.sum(idx), self.cfg.freq_min, self.cfg.freq_max)
            self.sin_off[idx] = self.rnd_vec(torch.sum(idx), -torch.pi, torch.pi)
            self.t_rem_sin[idx] = self.rnd_vec(torch.sum(idx), self.cfg.t_min, self.cfg.t_max)

    def _resample_exp_input(self, env_mask):
        idx = torch.logical_and(env_mask, self.t_rem_exp < 0)
        if torch.any(idx):
            self.exp_c[idx] = self.rnd_vec(torch.sum(idx))
            self.exp_alpha[idx] = self.rnd_vec(torch.sum(idx), self.cfg.alpha_min, self.cfg.alpha_max)
            self.t_exp_start[idx] = self.t[:, None].repeat((1, 3))[idx]
            self.t_rem_exp[idx] = self.rnd_vec(torch.sum(idx), self.cfg.t_min, self.cfg.t_max)

    def _resample_rnd_input(self, env_mask):
        idx = torch.logical_and(env_mask, self.t_rem_rnd < 0)
        if torch.any(idx):
            self.rnd_mag[idx] = torch.clamp(torch.randn(size=(torch.sum(idx).item(),), device=self.device) * self.cfg.rng_mag_std, -1, 1)
            self.rnd_inds[idx] = (self.rnd_vec(torch.sum(idx)) / 2 + 0.5) < self.cfg.rnd_prob
            self.t_rem_rnd[idx] = self.rnd_vec(torch.sum(idx), self.cfg.t_min, self.cfg.t_max)

    def _resample_weight(self, env_mask):
        idx = torch.logical_and(env_mask, self.t_rem_weights < 0)
        if torch.any(idx):
            weight_mask = idx[:, :, None].repeat((1, 1, self.weights.shape[-1]))
            self.weights[weight_mask] = self.rnd_vec(torch.sum(idx) * self.weights.shape[-1], 0., 1.)
            self.weights /= torch.sum(self.weights, dim=-1, keepdim=True)
            soft_max_inds = torch.rand(self.weights.shape[:2]) <= self.cfg.prop_softmax
            self.weights[soft_max_inds] = torch.softmax(self.weights[soft_max_inds] / self.cfg.softmax_temp, dim=-1)
            # TODO: alter weight distribution?
            self.t_rem_weights[idx] = self.rnd_vec(torch.sum(idx), self.cfg.t_min, self.cfg.t_max)

    def _resample_prop_v_max(self, env_mask):
        idx = torch.logical_and(env_mask, self.t_rem_prop_v_max < 0)
        if torch.any(idx):
            a = self.cfg.prop_v_max
            self.prop_v_max[idx] = torch.clamp(self.rnd_vec(torch.sum(idx), 0, 1 + a / (1 - a)), 0., 1.)
            self.t_rem_prop_v_max[idx] = self.rnd_vec(torch.sum(idx), self.cfg.t_min, self.cfg.t_max)

    def _const_input(self):
        return self.sample_hold_input

    def _ramp_input_t(self, t):
        return torch.clamp(
            self.ramp_v_start + \
            (self.ramp_v_end - self.ramp_v_start) * (t[:, None] - self.ramp_t_start) / (self.t_rem_ramp + t[:, None] - self.ramp_t_start),
            -1, 1)

    def _extreme_input(self):
        return self.extreme_input

    def _sinusoid_input_t(self, t):
        return torch.clamp(self.sin_mag * torch.sin(self.sin_freq * t[:, None] + self.sin_off) + self.sin_mean, -1, 1)

    def _exp_input_t(self, t):
        return torch.clamp(self.exp_c * torch.exp(self.exp_alpha * (t[:, None] - self.exp_t_start)), -1, 1)

    def get_input_t(self, t):
        const_input = self._const_input()
        ramp_input = self._ramp_input_t(t)
        extreme_input = self._extreme_input()
        sin_input = self._sinusoid_input_t(t)
        exp_input = self._exp_input_t(t)
        v = self.weights[:, :, 0] * const_input + self.weights[:, :, 1] * ramp_input + \
            self.weights[:, :, 2] * extreme_input +  self.weights[:, :, 3] * sin_input + \
            self.weights[:, :, 4] * exp_input

        # Add noise
        v[self.rnd_inds] += self.rnd_vec(torch.sum(self.rnd_inds)) * self.rnd_mag[self.rnd_inds]
        v[self.rnd_inds] = torch.clamp(v[self.rnd_inds], -1, 1)

        return ((self.v_max - self.v_min) * (v / 2 + 0.5) + self.v_min) * self.prop_v_max

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        # Step the RoM
        step_inds = self.t - self.last_step_t >= self.cfg.dt * self.cfg.rel_dt
        if torch.any(step_inds):
            self._step_rom_idx(step_inds)
            self.last_step_t[step_inds] = self.t[step_inds]
        self.t += self.cfg.dt
        self.t_rem_sample_hold -= self.cfg.dt
        self.t_rem_extreme -= self.cfg.dt
        self.t_rem_sin -= self.cfg.dt
        self.t_rem_rnd -= self.cfg.dt
        self.t_rem_exp -= self.cfg.dt
        self.t_rem_ramp -= self.cfg.dt
        self.t_rem_weights -= self.cfg.dt

        # do PD control to the trajectory
        # compute position error
        interp = (self.k % self.cfg.rel_dt)[:, None] / self.cfg.rel_dt
        traj = (self.trajectory[:, 1] - self.trajectory[:, 0]) * interp + self.trajectory[:, 0]
        pos_err = (self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]) - traj[:, :2]
        robot_yaw = self._quat2yaw(self.robot.data.root_quat_w)
        heading_err = ((robot_yaw - traj[:, 2]) + torch.pi) % (2 * torch.pi) - torch.pi
        # convert position error into local frame
        pos_err = torch.vstack([
            torch.cos(robot_yaw) * pos_err[:, 0] + torch.sin(robot_yaw) * pos_err[:, 1],
            -torch.sin(robot_yaw) * pos_err[:, 0] + torch.cos(robot_yaw) * pos_err[:, 1]
        ]).T
        # Put feedforward velocity into local frame
        v_ff = torch.vstack([
            torch.cos(heading_err) * self.v_trajectory[:, 0, 0] + torch.sin(heading_err) * self.v_trajectory[:, 0, 1],
            -torch.sin(heading_err) * self.v_trajectory[:, 0, 0] + torch.cos(heading_err) * self.v_trajectory[:, 0, 1],
            self.v_trajectory[:, 0, 2]
        ]).T
        self.vel_command_b = torch.clamp(
            v_ff - self.K * torch.hstack((pos_err, heading_err[:, None])),
            self.fb_v_min,
            self.fb_v_max
        )

    def _step_rom_idx(self, idx):
        self._resample_command(idx)
        # Get input to apply for trajectory
        v = self.get_input_t(self.t)
        # Enforce standing
        v[self.is_standing_env, :] = 0

        # Update trajectory
        z = self.trajectory[idx, -1, :]
        s_theta = torch.sin(z[:, 2])
        c_theta = torch.cos(z[:, 2])
        z_next = z + (self.cfg.dt * self.cfg.rel_dt) * torch.vstack([
            c_theta * v[idx, 0] - s_theta * v[idx, 1],
            s_theta * v[idx, 0] + c_theta * v[idx, 1],
            v[idx, 2]
        ]).T
        self.trajectory[idx, :-1, :] = self.trajectory[idx, 1:, :]
        self.trajectory[idx, -1, :] = z_next
        self.v_trajectory[idx, :-1, :] = self.v_trajectory[idx, 1:, :]
        self.v_trajectory[idx, -1, :] = v[idx]
        self.k[idx] += 1

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
