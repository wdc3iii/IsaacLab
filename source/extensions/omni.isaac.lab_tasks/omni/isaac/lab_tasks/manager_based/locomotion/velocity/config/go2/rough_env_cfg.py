# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from typing import TYPE_CHECKING

import numpy as np
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, TrackingCommandsCfg, TrackingObservationsCfg, RewardsCfg
from omni.isaac.lab.terrains.config.rough import EASY_ROUGH_TERRAINS_CFG
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
import math
from omni.isaac.lab.managers import SceneEntityCfg
import torch

##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip

GAIT_PHASES = {
    # trot (diagonals together).
    0: torch.tensor([0, torch.pi, torch.pi, 0]),
    # walk (staggered diagonals).
    1: torch.tensor([0, 0.5 * torch.pi, torch.pi, 1.5 * torch.pi]),
    # pace (same side legs together).
    2: torch.tensor([0, torch.pi, 0, torch.pi]),
    # bound (front and back legs together).
    3: torch.tensor([0, 0, torch.pi, torch.pi]),
    # pronk (all legs together).
    4: torch.tensor([0, 0, 0, 0]),
}

def get_gait(env, gait):
    return torch.ones((env.num_envs, 1), device=env.device) * GAIT_PHASES[gait][None, :].to(env.device)

joint_traj = torch.tensor(np.loadtxt('/home/wcompton/repos/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/go2/joint_traj/joint_trajectory.txt'))
joint_times = torch.arange(joint_traj.shape[0]) / (joint_traj.shape[0] - 1)

v_xs = torch.tensor(np.load('/home/wcompton/repos/IsaacLab/tools/generate_references/go2/references/go2_vxs.npy'))
v_ys = torch.tensor(np.load('/home/wcompton/repos/IsaacLab/tools/generate_references/go2/references/go2_vys.npy'))
w_zs = torch.tensor(np.load('/home/wcompton/repos/IsaacLab/tools/generate_references/go2/references/go2_wzs.npy'))
ts = torch.tensor(np.load('/home/wcompton/repos/IsaacLab/tools/generate_references/go2/references/go2_reference_ts.npy'))
q_refs = torch.tensor(np.load('/home/wcompton/repos/IsaacLab/tools/generate_references/go2/references/go2_reference_qs_isaac.npy'))
foot_refs = torch.tensor(np.load('/home/wcompton/repos/IsaacLab/tools/generate_references/go2/references/go2_reference_foot_refs.npy'))

@configclass
class QuadrupedRewardsCfg(RewardsCfg):

    # Track Base height
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.33},
    )

    # Track contact
    phase_based_contact = RewTerm(
        func=mdp.phase_based_contact,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "period": 0.3,
            "stand_threshold": 0.05,
            "get_gait_offset": get_gait,
            "get_gait_args": {"gait": 0}
        }
    )

    foot_pos = RewTerm(
        func=mdp.interp_foot_pos,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "v_xs": v_xs,
            "v_ys": v_ys,
            "w_zs": w_zs,
            "ts": ts,
            "foot_refs": foot_refs,
            "std": math.sqrt(0.001),  # Maximum error is about stand_threshold
            "get_gait_offset": get_gait,
            "get_gait_args": {"gait": 0}
        }
    )

    joint_traj = RewTerm(
        func=mdp.interp_joint_trajectory,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "v_xs": v_xs,
            "v_ys": v_ys,
            "w_zs": w_zs,
            "ts": ts,
            "q_refs": q_refs,
            "std": math.sqrt(0.1),  # Maximum error is about stand_threshold
            "get_gait_offset": get_gait,
            "get_gait_args": {"gait": 0}
        }
    )
    # # Track swing foot height
    # swing_foot_height = RewTerm(
    #     func=mdp.swing_foot_height,
    #     weight=0.5,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
    #         "swing_height": 0.08,
    #         "period": 0.3,
    #         "stand_threshold": 0.05,
    #         "std": math.sqrt(0.001),  # Maximum error is about stand_threshold
    #         "get_gait_offset": get_gait,
    #         "get_gait_args": {"gait": 0}
    #     }
    # )
    #
    # # Track Swing foot xy
    # foot_pos_xy = RewTerm(
    #     func=mdp.foot_pos_xy,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
    #         "foot_xy_des": torch.tensor([
    #             [0.177822, 0.110444],
    #             [0.177822, -0.110444],
    #             [-0.270516, 0.111372],
    #             [-0.270516, -0.111372]
    #         ]),
    #         "std": math.sqrt(0.0025),  # wider peak than standing
    #     }
    # )

    # Track joint trajectory
    # joint_trajectories = RewTerm(
    #     func=mdp.joint_trajectories,
    #     weight=0.5,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "period": 0.3,
    #         "joint_times": joint_times,
    #         "joint_trajectory": joint_traj,
    #         "stand_threshold": 0.05,
    #         "std": math.sqrt(0.1),  # Maximum error is about stand_threshold
    #         "get_gait_offset": get_gait,
    #         "get_gait_args": {"gait": 0}
    #     }
    # )

@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    rewards: QuadrupedRewardsCfg = QuadrupedRewardsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set observation period to match reward period
        self.observations.policy.phase.params["period"] = self.rewards.phase_based_contact.params["period"]

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_slide = None
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.foot_height = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.foot_pos.params["v_xs"] = self.rewards.foot_pos.params["v_xs"].to(self.sim.device)
        self.rewards.foot_pos.params["v_ys"] = self.rewards.foot_pos.params["v_ys"].to(self.sim.device)
        self.rewards.foot_pos.params["w_zs"] = self.rewards.foot_pos.params["w_zs"].to(self.sim.device)
        self.rewards.foot_pos.params["ts"] = self.rewards.foot_pos.params["ts"].to(self.sim.device)
        self.rewards.foot_pos.params["foot_refs"] = self.rewards.foot_pos.params["foot_refs"].to(self.sim.device)
        self.rewards.joint_traj.params["v_xs"] = self.rewards.joint_traj.params["v_xs"].to(self.sim.device)
        self.rewards.joint_traj.params["v_ys"] = self.rewards.joint_traj.params["v_ys"].to(self.sim.device)
        self.rewards.joint_traj.params["w_zs"] = self.rewards.joint_traj.params["w_zs"].to(self.sim.device)
        self.rewards.joint_traj.params["ts"] = self.rewards.joint_traj.params["ts"].to(self.sim.device)
        self.rewards.joint_traj.params["q_refs"] = self.rewards.joint_traj.params["q_refs"].to(self.sim.device)


        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RoughBlindEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_generator = EASY_ROUGH_TERRAINS_CFG
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None


@configclass
class UnitreeGo2RoughTrackingEnvCfg(UnitreeGo2RoughEnvCfg):
    commands: TrackingCommandsCfg = TrackingCommandsCfg()
    observations: TrackingObservationsCfg = TrackingObservationsCfg()

    num_iterations: int = 100
    iter_duration_s: float = 20


@configclass
class UnitreeGo2RoughBlindTrackingEnvCfg(UnitreeGo2RoughBlindEnvCfg):
    commands: TrackingCommandsCfg = TrackingCommandsCfg()
    observations: TrackingObservationsCfg = TrackingObservationsCfg()

    num_iterations: int = 100
    iter_duration_s: float = 20


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class UnitreeGo2RoughBlindEnvCfg_PLAY(UnitreeGo2RoughBlindEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class UnitreeGo2RoughTrackingEnvCfg_RECORD(UnitreeGo2RoughTrackingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class UnitreeGo2RoughBlindTrackingEnvCfg_Record(UnitreeGo2RoughBlindTrackingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
