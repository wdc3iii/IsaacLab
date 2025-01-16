import torch



class VelocityRHUnitreeGo2Cfg:
    """Configuration for the Velocity Tracking Raibert Heuristic Controller on the Go2"""
    # Swing trajectory
    t_swing: float = 0.25
    swing_apex_proportion: float = 0.55
    swing_apex_height: float = 0.1

    @staticmethod
    def coeffs(T, z):
        return (torch.linalg.inv(
            torch.tensor([[T ** 2, T ** 3], [2 * T, 3 * (T**2)]])
        ) @ torch.tensor([[z], [0]])).to("cuda")

    up_coeffs = coeffs(swing_apex_proportion, swing_apex_height)
    down_coeffs = coeffs(1 - swing_apex_proportion, swing_apex_height)

    swing_lateral_apex_proportion: float = 0.5
    lat_coeffs = coeffs(1., 1.)
    stance_inds = torch.tensor([1, 0, 0, 1], dtype=torch.bool) # (LF, RF, LH, RH)
    foot_offsets = torch.tensor([
        [0.385 / 2, 0.27 / 2],
        [0.385 / 2, -0.27 / 2],
        [-0.385 / 2, 0.27 / 2],
        [-0.385 / 2, -0.27 / 2],
    ]).to("cuda")

    # Raibert Heuristic
    z_des: float = 0.33             # Desired walking height
    g: float = 9.81                 # Gravitational constant
    K_raibert = (z_des / g) ** 0.5  # Raibert Gain

    # Command maximums for tracking
    v_max: float = 1.0
    yaw_rate_max: float = 1.5

    # whether to track yaw
    track_yaw: int = 0  # 0: track zero yaw, 1: track yaw rate, -1: track initial yaw
    use_vel_control: bool = False  # Whether to include velocity terms in the PD controller for joints.
    # Experiment name
    experiment_name: str = "go2_velocity_rh"
    run_name: str = None

    def create_controller(self, task, num_envs, device):
        return VelocityRHUnitreeGo2Wrapper(task, self, num_envs, device)


class VelocityRHUnitreeGo2Wrapper:

    def __init__(self, task, cfg, num_envs, device):
        self.controller = VelocityRHUnitreeGo2(cfg, num_envs, device)
        if task == "Isaac-Velocity-Flat-Unitree-Go2-v0" or task == "Isaac-Velocity-Flat-Unitree-Go2-Play-v0" \
                or task == "Isaac-Velocity-FlatWPos-Unitree-Go2-v0" or task == "Isaac-Velocity-FlatWPos-Unitree-Go2-Play-v0":
            self.p_inds  = [0, 1, 2]
            self.q_inds = [3, 4, 5, 6]
            self.v_inds = [7, 8, 9]
            self.w_inds = [10, 11, 12]
            self.command_inds = [13, 14, 15]
        elif task == "Isaac-Velocity-Rough-Unitree-Go2-v0" or task == "Isaac-Velocity-Rough-Hopper-Unitree-Go2-Play-v0" \
                or task == "Isaac-Velocity-RoughWPos-Unitree-Go2-v0" or task == "Isaac-Velocity-RoughWPos-Hopper-Unitree-Go2-Play-v0":
            self.p_inds = [0, 1, 2]
            self.q_inds = [3, 4, 5, 6]
            self.v_inds = [7, 8, 9]
            self.w_inds = [10, 11, 12]
            self.command_inds = [13, 14, 15]
        else:
            raise RuntimeError(f"Environment type {task} not supported for Velocity Raibert Heuristic on the Go2.")

    def __call__(self, t, obs):
        self.controller.set_command(obs['policy'][:, self.command_inds])
        return self.controller.compute(
            t,
            obs['policy'][:, self.p_inds],
            obs['policy'][:, self.q_inds],
            obs['policy'][:, self.v_inds],
            obs['policy'][:, self.w_inds]
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
        self.t_prev_impact = -2 * self.cfg.t_swing
        self.p_swing_takeoff_prev = torch.repeat_interleave(self.cfg.foot_offsets[None, self.cfg.stance_inds, :], num_robots, dim=0)

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

    def compute(self, t: torch.Tensor, p: torch.Tensor, q: torch.Tensor, v: torch.Tensor, w: torch.Tensor):
        """Computes the desired impact quaternion to track the desired velocity command.

        Args:
            t: The current time
            p: The current position (x, y, theta)
            q: The current orientation quaternion (w, x, y, z)
            v: The current center of mass velocity in shape (N, 3) (vx, vy, wz).
            w: The current body angular velocity in shape (N, 3).
            """
        # Compute impact time
        t_gait = t % (2 * self.cfg.t_swing)                        # Time into gait sequence
        stance_inds = self.cfg.stance_inds if t_gait < self.cfg.t_swing else not self.cfg.stance_inds  # Stance indices
        swing_inds = torch.logical_not(stance_inds)                # Swing indices
        t_impact = self.cfg.t_swing - (t_gait % self.cfg.t_swing)  # Time until next impact
        t_next_mid = t_impact + self.cfg.t_swing / 2               # Time until next mid-stance
        t_phase = (t_gait % self.cfg.t_swing) / self.cfg.t_swing
        yaw = self._quat2yaw(q)
        if t - self.t_prev_impact > self.cfg.t_swing:
            self.t_prev_impact = t
            self.p_swing_takeoff_prev = torch.vstack((
                torch.cos(yaw) * p[:, 0] - torch.sin(yaw) * p[:, 1],
                torch.sin(yaw) * p[:, 0] + torch.cos(yaw) * p[:, 1]
            )).T[:, None, :] + self.cfg.foot_offsets[None, stance_inds]

        # Compute desired swing foot position
        p_mid_stance = torch.where(
            torch.abs(self.yaw_rate_des)[:, None] >= 1e-3,
            1 / self.yaw_rate_des[:, None] * torch.vstack((
                self.v_des[:, 0] * torch.sin(self.yaw_rate_des * t_next_mid) + self.v_des[:, 1] * (torch.cos(self.yaw_rate_des * t_next_mid) - 1),
                -self.v_des[:, 0] * (torch.cos(self.yaw_rate_des * t_next_mid) - 1) + self.v_des[:, 1] * torch.sin(self.yaw_rate_des * t_next_mid)
            )).T,
            t_next_mid * self.v_des
        )  # Compute CoM position at next mid-stance, relative to current position.
        p_swing_impact_nom = p_mid_stance[:, None, :] + self.cfg.foot_offsets[None, swing_inds]  # Offset to feet under hips
        p_swing_impact = p_swing_impact_nom + self.cfg.K_raibert * (v[:, :2] - self.v_des)[:, None, :]  # modify via Raibert Heuristic
        p_swing_takeoff_rel = p[:, :2, None] - self.p_swing_takeoff_prev
        p_swing_takeoff = torch.dstack((
            torch.cos(yaw[:, None]) * p_swing_takeoff_rel[:, :, 0] + torch.sin(yaw[:, None]) * p_swing_takeoff_rel[:, :, 1],
            -torch.sin(yaw[:, None]) * p_swing_takeoff_rel[:, :, 0] + torch.cos(yaw[:, None]) * p_swing_takeoff_rel[:, :, 1]
        ))                                                             # Compute swing foot takeoff position in current body frame
        p_swing_des = self._compute_swing_traj(t_phase, p_swing_takeoff, p_swing_impact)  # Compute desired foot position along spline trajectory

        # Now implement controller. Decide pos/vel setpoints, feedforward torques.

        return

    @staticmethod
    def _quat2yaw(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return yaw

    def _compute_swing_traj(self, t, p_prev, p_next):
        t_square = t ** 2
        dp = p_next - p_prev
        x = p_prev[:, :, 0] + (self.cfg.lat_coeffs[0] * t_square + self.cfg.lat_coeffs[1] * t_square * t) * dp[:, :, 0]  # x (one spline)
        y = p_prev[:, :, 1] + (self.cfg.lat_coeffs[0] * t_square + self.cfg.lat_coeffs[1] * t_square * t) * dp[:, :, 1]  # y (one spline)
        if t < self.cfg.swing_apex_proportion:
            return torch.dstack((x, y, torch.ones_like(x) * (self.cfg.up_coeffs[0] * t_square + self.cfg.up_coeffs[1] * t_square * t)))
        else:
            # predict where p, v are at next contact
            t_r = 1 - t
            t_r_sq = torch.square(t_r)
            return torch.dstack((x, y, torch.ones_like(self.cfg.down_coeffs[0] * t_r_sq + self.cfg.down_coeffs[1] * t_r_sq * t_r)))
