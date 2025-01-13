import torch



class VelocityRHUnitreeGo2Cfg:
    """Configuration for the Velocity Tracking Raibert Heuristic Controller on the Hopper"""
    # Raibert gains
    Kp: float = 0.2
    Kd: float = 0.4
    Kff: float = 0.05
    # Clipping Values
    clip_pos: float = 0.5
    clip_vel: float = 1.
    clip_ang: float = 1.
    clip_ff: float = 0.2
    # Command maximums for tracking
    v_max: float = 1.0
    yaw_rate_max: float = 1.5
    # z height at impact (upright) for approximating time to impact
    z_impact: float = 0.35
    # Gravitational constant
    g: float = 9.81
    # whether to track yaw
    track_yaw: int = 0  # 0: track zero yaw, 1: track yaw rate, -1: track initial yaw
    use_vel_control: bool = False  # Whether to include velocity terms in the PD controller for joints.
    # Experiment name
    experiment_name: str = "hopper_velocity_rh"
    run_name: str = None

    def create_controller(self, task, num_envs, device):
        return VelocityRHUnitreeGo2Wrapper(task, self, num_envs, device)


class VelocityRHUnitreeGo2Wrapper:

    def __init__(self, task, cfg, num_envs, device):
        self.controller = VelocityRHUnitreeGo2(cfg, num_envs, device)
        if task == "Isaac-Velocity-Unitree-Go2-Flat-v0" or task == "Isaac-Velocity-Hopper-Unitree-Go2-Flat-Play-v0":
            self.vel_inds  = []
            self.ang_vel_inds = []
            self.z_ind = 0
            self.quat_inds = []
            self.command_inds = []
        elif task == "Isaac-Velocity-Unitree-Go2-Rough-v0" or task == "Isaac-Velocity-Hopper-Unitree-Go2-Rough-Play-v0":
            self.vel_inds  = []
            self.ang_vel_inds = []
            self.z_ind = 0
            self.quat_inds = []
            self.command_inds = []
        else:
            raise RuntimeError(f"Environment type {task} not supported for Velocity Raibert Heuristic on the Go2.")

    def __call__(self, obs):
        self.controller.set_command(obs['policy'][:, self.command_inds])
        return self.controller.compute(
            obs['policy'][:, self.vel_inds],
            obs['policy'][:, self.ang_vel_inds],
            obs['policy'][:, self.z_ind],
            obs['policy'][:, self.quat_inds]
        )


class VelocityRHUnitreeGo2:

    def __init__(self, cfg: VelocityRHUnitreeGo2Cfg, num_robots: int, device: str):
        # store inputs
        self.cfg = cfg
        self.num_robots = num_robots
        self._device = device

        # create buffers
        # -- commands
        self.v_des = torch.zeros(num_robots, 2, device=self._device)
        self.yaw_rate_des = torch.zeros(num_robots, 1, device=self._device)
        # yaw_buffer
        self.init_yaw = torch.zeros((num_robots,), device=self._device)

    @property
    def num_actions(self) -> int:
        # Action is a (v_x, v_y, yaw rate) tuple
        if self.cfg.use_vel_control:
            return 24
        else:
            return 12

    def initialize(self):
        """Initialize the internals"""
        pass

    def reset_idx(self, robot_ids: torch.Tensor=None):
        """Reset the internals"""
        pass

    def set_command(self, command: torch.Tensor):
        """Set the target velocity command.

        Args:
            command: The command to set. This is a tensor of shape (num_robots, 3) where
            the actions are x and y velocities and yaw rate."""

        self.v_des = torch.clip(command[:, :2], -self.cfg.v_max, self.cfg.v_max)
        self.yaw_rate_des = torch.clip(command[:, 2], -self.cfg.yaw_rate_max, self.cfg.yaw_rate_max)

    def compute(self, v: torch.Tensor, w: torch.Tensor, z: torch.Tensor, q: torch.Tensor):
        """Computes the desired impact quaternion to track the desired velocity command.

        Args:
            v: The current center of mass velocity in shape (N, 3).
            w: The current body angular velocity in shape (N, 3).
            z: The current center of mass height in shape (N, 1).
            q: body orientation quaternion in shape (N, 4)."""
        # Extract variables
        vz = v[:, 2]

        # Compute impact time
        t_impact = (vz + torch.sqrt(torch.square(vz) + 2 * self.cfg.g * torch.clamp(z - self.cfg.z_impact, min=0))) / self.cfg.g

        # Compute the desired impact position (relative to current position).
        p_des = self.v_des * t_impact[:, None]
        yaw_des = self.yaw_rate_des * t_impact

        # Predict the next impact location (relative to current position).
        p_pred = v[:, :2] * t_impact[:, None]
        yaw_pred = w[:, 2] * t_impact

        # Implement Raibert Heuristic on these positions
        quat_des = self._compute_raibert_heuristic(p_des - p_pred, yaw_des - yaw_pred, v, w, q)
        return quat_des

    def _compute_raibert_heuristic(self, p_com, v_com, t_since_contact):
        # predict where p, v are at next contact
        return foot_pos
