import torch
from math import sqrt, tanh, cosh


class VelocityHLIPUnitreeG1Cfg:
    """Configuration for the Velocity Tracking Raibert Heuristic Controller on the Hopper"""
    # HLIP gains TODO: tune these gains
    t_ssp: float = 0.5
    t_dsp: float = 0.
    z_ref: float = 0.5
    u_y_nom_offset: float = 0.1

    # Set spline parameters
    z_pos_swf_imp: float = -0.0     # Height of swing foot desired at impact
    z_vel_swf_imp: float = -0.05    # Velocity of swing foot desired at impact
    z_vel_swf_tof: float = 0.05     # Velocity of swing foot desired at takeoff
    z_pos_swf_max: float = 0.07     # Height of swing foot desired at peak of swing
    prop_swf_peak: float = 0.75     # Proportion of swing duration at which desired peak height occurs.
    z_pos_swf_thresh: float = 0.01  # Height at which to trigger contact logic (if not using contact forces
    z_force_swf_thresh: float = 0.1 # Force at which to trigger contact logic (if using contact forces)

    # Command maximums for tracking
    v_max: float = 1.0
    yaw_rate_max: float = 1.5
    # Gravitational constant
    g: float = 9.81
    # whether to track yaw
    track_yaw: int = 0  # 0: track zero yaw, 1: track yaw rate, -1: track initial yaw
    # Whether to include velocity terms in the PD controller for joints.
    use_vel_control: bool = False
    # How to trigger contacts to switch gait phases
    contact_handling: str = "time"  # Options are "time" and "force"
    # HLIP Controller
    hlip_control: str = "deadbeat"
    # Experiment name
    experiment_name: str = "unitree_g1_velocity_hlip"
    run_name: str = None

    def create_controller(self, task, num_envs, device):
        return VelocityHLIPUnitreeG1Wrapper(task, self, num_envs, device)


class VelocityHLIPUnitreeG1Wrapper:

    def __init__(self, task, cfg, num_envs, device):
        self.controller = VelocityHLIPUnitreeG1(cfg, num_envs, device)
        if task == "Isaac-Velocity-FlatHLIP-G1-v0" or task == "Isaac-Velocity-FlatHLIP-G1-Play-v0":
            self.vel_inds  = [0, 1, 2]
            self.ang_vel_inds = []
            self.z_ind = 0
            self.quat_inds = []
            self.command_inds = [9, 10, 11]
        elif task == "Isaac-Velocity-RoughHLIP-G1-v0" or task == "Isaac-Velocity-RoughHLIP-G1-v0":
            self.vel_inds  = []
            self.ang_vel_inds = []
            self.z_ind = 0
            self.quat_inds = []
            self.command_inds = []
        else:
            raise RuntimeError(f"Environment type {task} not supported for Velocity Raibert Heuristic on the Go2.")

    def __call__(self, t, obs):
        self.controller.set_command(obs['policy'][:, self.command_inds])
        right_foot_pos = obs['foot_pos'][:, 0, :]
        left_foot_pos = obs['foot_pos'][:, 1, :]
        com_vel = obs['policy'][:, self.vel_inds]
        p_com = torch.where(self.controller.current_stance_foot[:, None], -right_foot_pos, -left_foot_pos)
        p_swf = torch.where(self.controller.current_stance_foot[:, None], left_foot_pos - right_foot_pos, right_foot_pos - left_foot_pos)
        f_swf = torch.where(self.controller.current_stance_foot[:, None], obs['foot_force'][:, 0, :], obs['foot_force'][:, 1, :])
        return self.controller.compute(
            t, p_com, com_vel, p_swf, f_swf
        )


class VelocityHLIPUnitreeG1:

    def __init__(self, cfg: VelocityHLIPUnitreeG1Cfg, num_robots: int, device: str):
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
        self.t_phase = torch.zeros((num_robots,), device=self._device)
        self.t_scaled = torch.zeros((num_robots,), device=self._device)
        self.t_last_impact = torch.zeros((num_robots,), device=self._device)
        self.current_stance_foot = torch.zeros((num_robots,), dtype=torch.bool, device=self._device)
        self.p_swf = torch.zeros((num_robots, 3), device=self._device)
        self.p_swf_tof = torch.zeros((num_robots, 3), device=self._device)
        self.p_com = torch.zeros((num_robots, 3), device=self._device)

        # Create control variables
        self.lam = sqrt(self.cfg.g / self.cfg.z_ref)
        self.sigma_1 = self.lam / tanh(self.cfg.t_ssp * self.lam / 2)
        self.sigma_2 = self.lam * tanh(self.cfg.t_ssp * self.lam / 2)
        self.V = torch.tensor([[1., 1.], [self.lam, -self.lam]])
        self.V_inv = torch.inverse(self.V)
        self.S = torch.zeros(num_robots, 2, 2, device=self._device)

        t1 = self.cfg.prop_swf_peak
        M1 = torch.tensor([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [t1**3, t1**2, t1, 1],
            [3*t1**2, 2*t1, 1, 0]
        ])
        b1 = torch.tensor([[0], [self.cfg.z_vel_swf_tof], [self.cfg.z_pos_swf_max], [0]])
        M2 = torch.tensor([
            [t1 ** 3, t1 ** 2, t1, 1],
            [3 * t1 ** 2, 2 * t1, 1, 0],
            [1, 1, 1, 1],
            [3, 2, 1, 0]
        ])
        b2 = torch.tensor([[self.cfg.z_pos_swf_max], [0], [0], [self.cfg.z_vel_swf_imp]])
        self.c1 = (torch.linalg.inv(M1) @ b1).to(self._device)
        self.c2 = (torch.linalg.inv(M2) @ b2).to(self._device)

        Mxy = torch.tensor([[1, 1], [1/3, 1/2]])
        self.Mxy_inv = torch.linalg.inv(Mxy).to(self._device)
        self.bxy = torch.zeros((num_robots, 2), device=self._device)

        if self.cfg.contact_handling == 'time':
            self.handle_impact = self._handle_impact_time
        elif self.cfg.contact_handling == 'force':
            self.handle_impact = self._handle_impact_force
        else:
            raise RuntimeError(f"HLIP contact handling method {self.cfg.contact_handling} not supported.")

        if self.cfg.hlip_control == "deadbeat":
            self.K = torch.tensor([1.,  self.cfg.t_dsp + 1. / self.lam / tanh(self.cfg.t_ssp * self.lam)], device=self._device)
        else:
            raise RuntimeError(f"HLIP control method {self.cfg.hlip_control} not supported.")

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

    def reset_idx(self, robot_ids: torch.tensor=None):
        """Reset the internals"""
        pass

    def set_command(self, command: torch.tensor):
        """Set the target velocity command.

        Args:
            command: The command to set. This is a tensor of shape (num_robots, 3) where
            the actions are x and y velocities and yaw rate."""

        self.v_des = torch.clip(command[:, :2], -self.cfg.v_max, self.cfg.v_max)
        self.yaw_rate_des = torch.clip(command[:, 2], -self.cfg.yaw_rate_max, self.cfg.yaw_rate_max)

    def compute(self, t, p_com, v_com, p_swf, f_swf):
        """Computes the desired impact quaternion to track the desired velocity command.

        Args:
            t: The current time in shape (N,).
            p_com: The current center of mass position relative to the stance foot in shape (N, 3).
            v_com: The current center of mass linear velocity in shape (N, 1).
            p_swf: The current position of the swing foot relative to the stance foot in shape (N, 3).
            f_swf: The current forces acting on the swing foot in shape (N, 3)"""
        # Compute time-based phasing
        self.t_phase = t - self.t_last_impact
        self.t_scaled = self.t_phase / self.cfg.t_ssp
        self.p_swf = p_swf
        self.p_com = p_com

        # Handle impact events
        self.handle_impact(t, p_swf, f_swf)

        # Compute nominal step length
        u_x_nom = self.v_des[:, 0] * self.cfg.t_ssp
        u_y_nom = self.v_des[:, 1] * self.cfg.t_ssp + 2 * self._get_sign_offset() * self.cfg.u_y_nom_offset

        # Compute desired preimpact state
        p_x_pre_des = self.v_des[:, 0] * (self.cfg.t_ssp + self.cfg.t_dsp) / (2.0 + self.cfg.t_dsp * self.sigma_1)
        v_x_pre_des = self.sigma_1 * p_x_pre_des
        x_imp_des = torch.hstack((p_x_pre_des, v_x_pre_des))
        p_y_pre_des = -self._get_sign_offset() * self.cfg.u_y_nom_offset + \
                self.v_des[:, 1] * (self.cfg.t_ssp + self.cfg.t_dsp) / (2.0 + self.cfg.t_dsp * self.sigma_2)
        d2 = (self.lam**2 / cosh(self.lam * self.cfg.t_ssp / 2.0) * (self.cfg.t_ssp + self.cfg.t_dsp) * self.v_des[:, 1]) / (self.lam**2 * self.cfg.t_dsp + 2.0 * self.sigma_2)
        v_y_pre_des = self.sigma_2 * p_y_pre_des + d2
        y_imp_des = torch.hstack((p_y_pre_des, v_y_pre_des))

        # Compute the predicted preimpact state
        x_curr = torch.hstack((self.p_com[:, 0], v_com[:, 0]))
        lt = self.lam * (self.cfg.t_ssp - self.t_phase)
        self.S[:, 0, 0] = torch.exp(lt)
        self.S[:, 1, 1] = torch.exp(-lt)
        x_imp_pred = (self.V @ self.S @ self.V_inv) @ x_curr
        y_curr = torch.hstack((self.p_com[:, 1], v_com[:, 1]))
        y_imp_pred = (self.V @ self.S @ self.V_inv) @ y_curr

        # The error between the desired and predicted impact states
        x_err_impact = x_imp_pred - x_imp_des
        y_err_impact = y_imp_pred - y_imp_des
        # Feedback on this error signal
        u_x = u_x_nom + self.K @ x_err_impact
        u_y = u_y_nom + self.K @ y_err_impact

        # Compute spline trajectories for foot positions
        # x_swf, dx_swf = p_swf[:, 0] * (1 - bht) + u_x * bht
        # y_swf, dy_swf = p_swf[:, 1] * (1 - bht) + u_y * bht
        x_swf, dx_swf, y_swf, dy_swf = self.evaluate_xy_traj(u_x, u_y)
        z_swf, dz_swf = self.evaluate_z_traj()

        # Inverse kinematics to get desired joint positions (and maybe velocities)
        # TODO: IK
        q_des, qdot_des = self.inv_kinematics(x_swf, dx_swf, y_swf, dy_swf, z_swf, dz_swf)

        # Return q_des, qdot_des
        return q_des, qdot_des

    def inv_kinematics(self, x, dx, y, dy, z, dz):
        return 0, 0

    def _handle_impact_time(self, t, p_swf, f_swf):
        switch_idx = torch.logical_or(
            self.t_scaled >= 1,
            torch.logical_and(self.t_scaled > 0.5, p_swf[:, 2] < self.cfg.z_pos_swf_thresh)
        )
        self.switch_stance_foot_idx(switch_idx, t)

    def _handle_impact_force(self, t, p_swf, f_swf):
        switch_idx = torch.linalg.norm(f_swf, dim=-1) > self.cfg.z_force_swf_thresh
        self.switch_stance_foot_idx(switch_idx, t)

    def switch_stance_foot_idx(self, switch_idx, t):
        # Set phase times to zero
        self.t_scaled[switch_idx] = 0
        self.t_phase[switch_idx] = 0

        # Set time of last impact to current time
        self.t_last_impact[switch_idx] = t
        # Switch which foot is considered stance foot
        self.current_stance_foot[switch_idx] = torch.logical_not(self.current_stance_foot[switch_idx])

        # Update the com and swf positions
        self.p_com[switch_idx] -= self.p_swf[switch_idx]
        self.p_swf[switch_idx] = -self.p_swf[switch_idx]

        self.p_swf_tof[switch_idx] = torch.clone(self.p_swf[switch_idx])

    def _get_sign_offset(self):
        return torch.where(self.current_stance_foot, -1, 1)

    def evaluate_z_traj(self):
        cap_t_scaled = torch.clamp(self.t_scaled, min=0, max=1)
        t_square = torch.square(cap_t_scaled)
        t = torch.concatenate((cap_t_scaled * t_square, t_square, cap_t_scaled, torch.ones_like(cap_t_scaled)), dim=1)
        dt = torch.concatenate((3 * t_square, 2 * cap_t_scaled, torch.ones_like(cap_t_scaled), torch.zeros_like(cap_t_scaled)), dim=1)
        return torch.where(
            self.t_scaled >= self.cfg.prop_swf_peak,
            t @ self.c2,
            t @ self.c1
        ), torch.where(
            self.t_scaled >= self.cfg.prop_swf_peak,
            dt @ self.c2,
            dt @ self.c1
        )

    def evaluate_xy_traj(self, xf, yf):
        p = torch.sqrt(torch.square(xf - self.p_swf_tof[:, 0]) + torch.square(yf - self.p_swf_tof[:, 1]))
        self.bxy[:, 1] = p
        c = self.Mxy_inv @ self.bxy
        cap_t_scaled = torch.clamp(self.t_scaled, min=0, max=1)
        t_square = torch.square(cap_t_scaled)
        t = torch.concatenate((cap_t_scaled * t_square, t_square), dim=1)
        dt = torch.concatenate((3 * t_square, 2 * cap_t_scaled), dim=1)
        return (xf - self.p_swf_tof[:, 0]) * t @ c / p + self.p_swf_tof[:, 0], (xf - self.p_swf_tof[:, 0]) * dt @ c / p,\
            (yf - self.p_swf_tof[:, 1]) * t @ c / p + self.p_swf_tof[:, 1], (yf - self.p_swf_tof[:, 1]) * dt @ c / p
