import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
import os

HOPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.dirname(__file__)}/rsc/hopper.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=100000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=6
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "wheel.*": 0,
            "foot_slide": 0.02,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "wheels": DCMotorCfg(
            joint_names_expr=["wheel.*"],
            saturation_effort=12.6,
            effort_limit=2.1,
            velocity_limit=600.0,
            stiffness=0,
            damping=0.001,
            armature=0.01,
        ),
        # "wheels": IdealPDActuatorCfg(
        #     effort_limit=25000,
        #     joint_names_expr=["wheel.*"],
        #     stiffness=0.,
        #     damping=0.,
        #     armature=0.01
        # ),
        "foot": IdealPDActuatorCfg(
            effort_limit=25000,
            joint_names_expr=["foot_slide"],
            stiffness=11732.,
            damping=40.0,
            armature=0.01,
        )
    },
)