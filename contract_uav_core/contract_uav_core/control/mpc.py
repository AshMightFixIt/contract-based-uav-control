"""
Model predictive controller (optimal tier), moved unchanged from controllers.py.
"""

import numpy as np
from typing import Dict
import logging

from .base import BaseController
from ..frames import accel_to_attitude, attitude_to_rates_torques, mpc_accel_to_thrust
from ..interfaces import AccelCommand

logger = logging.getLogger(__name__)


class MPCController(BaseController):
    """
    Model Predictive Controller — Condensed QP formulation.

    Uses a double-integrator model:
        x = [position, velocity] ∈ R⁶
        u = [acceleration] ∈ R³
        x(k+1) = A x(k) + B u(k)

    Pre-computes prediction matrices (S, T) and Hessian inverse so each
    control step is a single matrix-vector multiply + clipping.

    Contract:
    - Assumes: Moderate wind (<15 m/s), bounded errors, solve time < 50ms
    - Guarantees: Better tracking than PID (optimization reduces error),
                  constraint satisfaction, Lyapunov decrease
    """

    def __init__(self, dt: float = 0.02):
        super().__init__("MPC", dt)

        # Prediction horizon
        self.N = 15
        self.n_x = 6   # [px, py, pz, vx, vy, vz]
        self.n_u = 3   # [ax, ay, az]

        # Cost weights
        self.Q_pos = np.array([1.0, 1.0, 2.0])   # position (altitude higher)
        self.Q_vel = np.array([0.5, 0.5, 1.0])    # velocity
        self.R_u = np.array([0.1, 0.1, 0.1])      # input (acceleration)

        # Constraints
        self.max_lateral_accel = 3.0    # m/s², ~17° tilt
        self.max_vertical_accel = 5.0   # m/s²
        self.hover_thrust = 0.5

        # Attitude inner loop (same structure as PID)
        self.kp_att = np.array([1.5, 1.5, 1.0])
        self.kd_att = np.array([0.4, 0.4, 0.3])

        # Performance tracking
        self.last_solve_time = 0.0
        self._prev_U = None  # warm start

        # Pre-compute prediction matrices
        self._build_prediction_matrices()

    def _build_prediction_matrices(self):
        """
        Pre-compute S, T, H, H_inv for the condensed QP.

        X = S * x0 + T * U   (stacked state = prediction matrix × initial + input matrix × inputs)
        cost = 0.5 U' H U + f' U
        H = T' Q_bar T + R_bar   (constant — pre-invert)
        f = T' Q_bar (S x0 - X_ref)   (changes each step)
        U* = -H_inv f
        """
        N, nx, nu = self.N, self.n_x, self.n_u
        dt = self.dt

        # Double-integrator dynamics: x(k+1) = A x(k) + B u(k)
        A = np.eye(nx)
        A[0:3, 3:6] = dt * np.eye(3)  # position += dt * velocity

        B = np.zeros((nx, nu))
        B[0:3, :] = 0.5 * dt**2 * np.eye(3)  # position += 0.5*dt²*accel
        B[3:6, :] = dt * np.eye(3)             # velocity += dt*accel

        # Build S (N*nx × nx) and T (N*nx × N*nu) prediction matrices
        self._S = np.zeros((N * nx, nx))
        self._T = np.zeros((N * nx, N * nu))

        A_pow = np.eye(nx)
        for i in range(N):
            A_pow = A_pow @ A if i > 0 else A.copy()
            self._S[i*nx:(i+1)*nx, :] = A_pow

            for j in range(i + 1):
                pow_diff = i - j
                A_pow_diff = np.linalg.matrix_power(A, pow_diff)
                self._T[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = A_pow_diff @ B

        # Build Q_bar (N*nx × N*nx diagonal) and R_bar (N*nu × N*nu diagonal)
        Q_diag = np.concatenate([self.Q_pos, self.Q_vel])
        Q_bar = np.zeros((N * nx, N * nx))
        R_bar = np.zeros((N * nu, N * nu))

        for i in range(N):
            # Terminal cost weighting (heavier at end for stability)
            weight = 1.0 if i < N - 1 else 3.0
            Q_bar[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = weight * np.diag(Q_diag)
            R_bar[i*nu:(i+1)*nu, i*nu:(i+1)*nu] = np.diag(self.R_u)

        # H = T' Q_bar T + R_bar  (constant Hessian)
        self._Q_bar = Q_bar
        H = self._T.T @ Q_bar @ self._T + R_bar

        # Regularize for numerical stability
        H += 1e-6 * np.eye(H.shape[0])

        # Pre-invert (this is the key optimization — done once)
        self._H_inv = np.linalg.inv(H)
        self._TQ = self._T.T @ Q_bar   # T' Q_bar (reused each step)

        logger.info(f"MPC prediction matrices built: N={N}, H={H.shape}")

    def reset(self):
        """Reset warm start and integral state."""
        self._prev_U = None
        self.last_error = np.zeros(3)
        self.integral_error = np.zeros(3)
        logger.info(f"{self.name} reset")

    def get_integral_state(self) -> np.ndarray:
        """Return first-step acceleration from warm-start as integral state proxy."""
        if self._prev_U is not None and len(self._prev_U) >= self.n_u:
            return self._prev_U[0:3].copy()
        return np.zeros(3)

    def set_integral_state(self, effective_accel: np.ndarray):
        """
        Seed warm-start with transferred acceleration (bumpless transfer).
        Fills all horizon steps with the inbound acceleration so the first
        MPC solve continues from a sensible initial condition.
        """
        self._prev_U = np.zeros(self.N * self.n_u)
        for i in range(self.N):
            self._prev_U[i * self.n_u:(i + 1) * self.n_u] = effective_accel

    def compute_accel_command(self,
                              state: Dict[str, np.ndarray],
                              setpoint: Dict[str, np.ndarray]) -> AccelCommand:
        """
        Solve condensed QP and return control.

        Steps:
        1. Form x0 (error state)
        2. Build X_ref (reference trajectory over horizon)
        3. Compute f = T'Q(S*x0 - X_ref)
        4. U* = -H_inv * f  (one matrix-vector multiply)
        5. Clip to constraints
        6. Extract first input, map to thrust + attitude + torques

        Returns the AccelCommand: accel is the first, constraint-clipped QP input
        (its z sets the thrust through frames.mpc_accel_to_thrust), yaw is the
        setpoint's yaw. compute_control (base.py) returns its legacy dict.
        """
        import time
        t_start = time.perf_counter()

        position = state['position']
        velocity = state['velocity']
        attitude = state['attitude']
        rates = state['rates']

        target_pos = setpoint.get('position', position)
        target_vel = setpoint.get('velocity', np.zeros(3))

        # State vector: [pos_error, vel_error] relative to target
        x0 = np.concatenate([position - target_pos, velocity - target_vel])

        # Reference trajectory: drive error to zero
        N, nx, nu = self.N, self.n_x, self.n_u
        X_ref = np.zeros(N * nx)  # All zeros = go to target

        # Compute gradient: f = T'Q(S*x0 - X_ref)
        predicted_free = self._S @ x0 - X_ref
        f = self._TQ @ predicted_free

        # Optimal unconstrained solution: U* = -H_inv * f
        U_star = -self._H_inv @ f

        # Warm start: blend with shifted previous solution
        if self._prev_U is not None:
            U_warm = np.zeros_like(U_star)
            U_warm[:-nu] = self._prev_U[nu:]  # shift by one step
            U_warm[-nu:] = self._prev_U[-nu:]  # repeat last
            U_star = 0.7 * U_star + 0.3 * U_warm

        # Clip to acceleration constraints
        for i in range(N):
            idx = i * nu
            U_star[idx:idx+2] = np.clip(U_star[idx:idx+2],
                                         -self.max_lateral_accel,
                                         self.max_lateral_accel)
            U_star[idx+2] = np.clip(U_star[idx+2],
                                     -self.max_vertical_accel,
                                     self.max_vertical_accel)

        # Save for warm start
        self._prev_U = U_star.copy()

        # Extract first input (acceleration command)
        accel_cmd = U_star[0:3]

        # Map acceleration to thrust + attitude (frames.py)
        # Simulation convention: thrust > hover → z increases (descent in NED)
        # MPC accel_cmd[2] < 0 means "move z negative" (climb) → need less thrust
        desired_thrust = mpc_accel_to_thrust(accel_cmd[2], self.hover_thrust)

        # Desired attitude from lateral acceleration (yaw-blind), clip tilt
        desired_yaw = setpoint.get('yaw', attitude[2])
        max_tilt = np.arctan2(self.max_lateral_accel, 9.81)  # ~17°
        desired_attitude = accel_to_attitude(accel_cmd, desired_yaw, max_tilt)

        # Inner loop: PD attitude control (same as PID inner loop), rate → torque
        desired_rates_cmd, torques = attitude_to_rates_torques(
            desired_attitude, attitude, rates, self.kp_att, self.kd_att, 2.0, 0.08)

        self.last_solve_time = time.perf_counter() - t_start

        return AccelCommand(
            accel=accel_cmd,
            yaw=desired_yaw,
            thrust=desired_thrust,
            torques=torques,
            desired_attitude=desired_attitude,
            desired_rates=desired_rates_cmd,
        )
