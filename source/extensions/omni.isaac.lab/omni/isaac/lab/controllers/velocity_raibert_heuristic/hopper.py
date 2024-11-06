import torch



class VelocityRHHopperCfg:
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
    v_max: float = 0.3
    yaw_rate_max: float = 1.
    # z height at impact (upright) for approximating time to impact
    z_impact: float = 0.35
    # Gravitational constant
    g: float = 9.81
    # whether to track yaw
    track_yaw: int = 0  # 0: track zero yaw, 1: track yaw rate, -1: track initial yaw

    # Experiment name
    experiment_name: str = "hopper_velocity_rh"
    run_name: str = None

    def create_controller(self, task, num_envs, device):
        return VelocityRHHopperWrapper(task, self, num_envs, device)


class VelocityRHHopperWrapper:

    def __init__(self, task, cfg, num_envs, device):
        self.controller = VelocityRHHopper(cfg, num_envs, device)
        if task == "Isaac-Velocity-Hopper-v0" or task == "Isaac-Velocity-Hopper-Play-v0":
            self.vel_inds  = [5, 6, 7]
            self.ang_vel_inds = [8, 9, 10]
            self.z_ind = 0
            self.quat_inds = [1, 2, 3, 4]
            self.command_inds = [14, 15, 16]
        else:
            raise RuntimeError(f"Environment type {task} not supported for Velocity Raibert Heuristic on the hopper.")

    def __call__(self, obs):
        self.controller.set_command(obs['policy'][:, self.command_inds])
        return self.controller.compute(
            obs['policy'][:, self.vel_inds],
            obs['policy'][:, self.ang_vel_inds],
            obs['policy'][:, self.z_ind],
            obs['policy'][:, self.quat_inds]
        )


class VelocityRHHopper:

    def __init__(self, cfg: VelocityRHHopperCfg, num_robots: int, device: str):
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
        return 3

    def initialize(self):
        """Initialize the internals"""
        pass

    def reset_idx(self, robot_ids: torch.Tensor=None):
        """Reset the internals"""
        pass

    def set_command(self, command: torch.Tensor):
        """Set the target velocity command.

        Args:
            command: The commmand to set. This is a tensor of shape (num_robots, 3) where
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

    def _compute_raibert_heuristic(self, delta_p, delta_yaw, v, w, q):
        pitch_d = torch.clip(
            - self.cfg.Kp * torch.clip(-delta_p[:, 0], -self.cfg.clip_pos, self.cfg.clip_pos) \
            - self.cfg.Kd * torch.clip(v[:, 0], -self.cfg.clip_vel, self.cfg.clip_vel) \
            + self.cfg.Kff * torch.clip(self.v_des[:, 0], -self.cfg.clip_ff, self.cfg.clip_ff),
            -self.cfg.clip_ang, self.cfg.clip_ang
        )
        roll_d = torch.clip(
            - self.cfg.Kp * torch.clip(delta_p[:, 1], -self.cfg.clip_pos, self.cfg.clip_pos) \
            - self.cfg.Kd * torch.clip(-v[:, 1], -self.cfg.clip_vel, self.cfg.clip_vel) \
            + self.cfg.Kff * torch.clip(-self.v_des[:, 0], -self.cfg.clip_ff, self.cfg.clip_ff),
            -self.cfg.clip_ang, self.cfg.clip_ang
        )
        # TODO: proper track_yaw = -1.
        if self.cfg.track_yaw == 1:
            yaw_d = torch.clip(delta_yaw, -self.cfg.clip_ang, self.cfg.clip_ang) + self._quat2yaw(q)
        else:
            yaw_d = self.init_yaw
        return self._rpy2quat(roll_d, pitch_d, yaw_d)

    @staticmethod
    def _quat2yaw(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return yaw

    @staticmethod
    def _rpy2quat(r: torch.Tensor, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cy = torch.cos(y * 0.5)
        sy = torch.sin(y * 0.5)
        cp = torch.cos(p * 0.5)
        sp = torch.sin(p * 0.5)
        cr = torch.cos(r * 0.5)
        sr = torch.sin(r * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return torch.stack((w, x, y, z), dim=-1)
