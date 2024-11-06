import math
import torch
from dataclasses import MISSING
from collections.abc import Sequence
from pytorch3d.transforms import quaternion_invert, quaternion_multiply, so3_log_map, quaternion_to_matrix, Rotate

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs.mdp.actions.joint_actions import JointAction
from omni.isaac.lab.envs.mdp.actions.actions_cfg import JointActionCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from omni.isaac.lab_assets.hopper import HOPPER_CFG  # isort: skip

##
# Scene definition
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = HOPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    foot_contact_force = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/foot", history_length=3, track_air_time=True)
    torso_contact_force = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/torso", history_length=3, track_air_time=False)
    # wheel_contact_force = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/wheel.*", history_length=3, track_air_time=False)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # ranges=mdp.UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x=(-0., 0.), lin_vel_y=(-0., 0.), ang_vel_z=(-0., 0.), heading=(-math.pi, math.pi)
        # ),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.0, 0.0), heading=(-math.pi, math.pi)
        ),
    )



class HopperGeometricPD(JointAction):
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._raw_actions[:, 0] = 1.
        self._processed_actions = torch.clone(self._raw_actions.detach())

        self.actuator_transform = Rotate(torch.tensor([
            [-0.8165, 0.2511, 0.2511],
            [-0, -0.7643, 0.7643],
            [-0.5773, -0.5939, -0.5939]
        ]), device=torch.device("cuda"))

        # self.actuator_transform = Rotate(torch.tensor([
        #     [-1., 0., 0.],
        #     [0., 1., 0.],
        #     [0., 0., -1.]
        # ]), device=torch.device("cuda"))

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions / torch.linalg.norm(self._raw_actions, axis=-1, keepdims=True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids, 1:] = 0.0
        self._raw_actions[env_ids, 0] = 1.

    def apply_actions(self):
        quat_d = self.processed_actions
        quat = self._asset.data.root_quat_w
        contact = torch.greater(torch.linalg.norm(self._env.scene.sensors['foot_contact_force'].data.net_forces_w, axis=-1), self.cfg.contact_threshold).squeeze()
        omega = self._asset.data.root_ang_vel_b
        wheel_vel = self._asset.data.joint_vel[:, 1:]

        not_contact = torch.logical_not(contact)
        torques = torch.zeros((quat.shape[0], 3), device=quat.device)
        # Spindown, when in contact
        if torch.any(contact):
            torques[contact, :] = -self.cfg.Kspindown * wheel_vel[contact, :]
        # Orientation Tracking
        if torch.any(not_contact):
            quat_d = quat_d[not_contact, :] / torch.linalg.norm(quat_d[not_contact, :], dim=-1, keepdim=True)
            quat = quat[not_contact, :] / torch.linalg.norm(quat[not_contact, :], dim=-1, keepdim=True)
            omega = omega[not_contact, :]
            err = quaternion_multiply(quaternion_invert(quat_d), quat)
            log_err = so3_log_map(quaternion_to_matrix(err))
            local_tau = -self.cfg.Kp * log_err - self.cfg.Kd * omega
            tau = self.actuator_transform.transform_points(local_tau)
            torques[not_contact, :] = tau

        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)


@configclass
class HopperGeometricPDCfg(JointActionCfg):

    class_type: type[ActionTerm] = HopperGeometricPD

    Kp: float = MISSING
    Kd: float = MISSING
    Kspindown: float = MISSING
    contact_threshold: float = MISSING


class HopperFoot(JointAction):
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 0

    def apply_actions(self):
        foot_pos = self._asset.data.joint_pos[:, self._joint_ids]
        foot_vel = self._asset.data.joint_vel[:, self._joint_ids]
        not_contact = torch.less(torch.linalg.norm(self._env.scene.sensors['foot_contact_force'].data.net_forces_w, axis=-1), self.cfg.contact_threshold).squeeze()

        torques = torch.zeros_like(foot_pos, device=foot_pos.device)
        if torch.any(not_contact):
            torques[not_contact] = self.cfg.spring_stiffness * self.cfg.foot_pos_des - self.cfg.Kp * (
                    foot_pos[not_contact] - self.cfg.foot_pos_des) - self.cfg.Kd * foot_vel[not_contact]
        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)


@configclass
class HopperFootCfg(JointActionCfg):

    class_type: type[ActionTerm] = HopperFoot
    Kp: float = MISSING
    Kd: float = MISSING
    spring_stiffness: float = MISSING
    foot_pos_des: float = MISSING
    contact_threshold: float = MISSING


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    wheel_eff = HopperGeometricPDCfg(
        asset_name="robot", joint_names=["wheel.*"],
        Kp=90., Kd=8., Kspindown=0.1,
        contact_threshold=0.1
    )
    foot_eff = HopperFootCfg(
        asset_name="robot", joint_names=["foot_slide"],
        Kp=25, Kd=10,
        spring_stiffness=HOPPER_CFG.actuators['foot'].stiffness,
        foot_pos_des=0.02, contact_threshold=0.1
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        root_pos_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.0, n_max=0.0))  # [:1]
        root_quat = ObsTerm(func=mdp.root_quat_w, params={"make_quat_unique": True}, noise=Unoise(n_min=-0.0, n_max=0.0))  # [1:5] (w, x, y, z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.0, n_max=0.0))  # [5:8]
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0., n_max=0.))  # [8:11]
        wheel_vel = ObsTerm(func=mdp.joint_vel_rel, params={'asset_cfg': SceneEntityCfg("robot", joint_names="wheel.*")}, noise=Unoise(n_min=-1.5, n_max=1.5))

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0., 0.)}, # (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    reset_foot = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",

        params={
            "position_range": (-0.01, 0.01),
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["foot.*"])
        },
    )
    reset_flywheels = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (-100, 100),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel.*"]),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.1)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("torso_contact_force", body_names=".*"), "threshold": 0.1},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.1)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("torso_contact_force", body_names=".*"), "threshold": 0.1},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

##
# Environment configuration
##


@configclass
class HopperVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=1.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.torso_contact_force is not None:
            self.scene.torso_contact_force.update_period = self.sim.dt
        if self.scene.foot_contact_force is not None:
            self.scene.foot_contact_force.update_period = self.sim.dt


@configclass
class HopperVelocityEnvCfg_PLAY(HopperVelocityEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
