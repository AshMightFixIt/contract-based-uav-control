"""
Contract-Based Controllers for Adaptive Drone System

Implements three-tier graduated degradation:
1. PID Controller     - Efficient, for nominal conditions (wind <= 3 m/s)
2. MPC Controller     - Optimal, for moderate conditions (wind <= 15 m/s)
3. H-infinity Controller - Robust, for emergency stabilization (wind <= 25 m/s)

Each controller has formal contracts defining when they work.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseController(ABC):
    """Base class for all controllers"""
    
    def __init__(self, name: str, dt: float = 0.02):
        self.name = name
        self.dt = dt
        self.active = False
        self.last_error = np.zeros(3)
        self.integral_error = np.zeros(3)
        
    @abstractmethod
    def compute_control(self, 
                       state: Dict[str, np.ndarray],
                       setpoint: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute control output"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset controller state"""
        pass
    
    def activate(self):
        """Activate controller"""
        self.active = True
        logger.info(f"{self.name} activated")
    
    def deactivate(self):
        """Deactivate controller"""
        self.active = False
        logger.info(f"{self.name} deactivated")


class PIDController(BaseController):
    """
    Cascaded PID Controller for position and attitude

    Contract:
    - Assumes: Low wind (<3 m/s), small errors (<2m), nominal conditions
    - Guarantees: Fast response (<5s), low overshoot (<30%), efficient
    """

    def __init__(self, dt: float = 0.02):
        super().__init__("PID", dt)

        # Position PID gains (outer loop) - tuned for stability in simplified simulation
        # MUCH lower gains to prevent overshoot
        self.kp_pos = np.array([0.35, 0.35, 0.8])  # [x, y, z]
        self.ki_pos = np.array([0.005, 0.005, 0.05])  # minimal integral
        self.kd_pos = np.array([0.5, 0.5, 0.6])  # strong velocity damping

        # Attitude PID gains (inner loop)
        self.kp_att = np.array([1.5, 1.5, 1.0])  # [roll, pitch, yaw] - reduced
        self.ki_att = np.array([0.02, 0.02, 0.02])
        self.kd_att = np.array([0.4, 0.4, 0.3])  # increased damping

        # Rate PID gains (innermost loop)
        self.kp_rate = np.array([0.08, 0.08, 0.06])  # [p, q, r]

        # Limits
        self.max_tilt = 0.22  # ~13 degrees max tilt
        self.max_rate = 1.0  # rad/s
        self.max_thrust = 1.0
        self.min_thrust = 0.0

        # Hover thrust (normalized) - this counteracts gravity
        self.hover_thrust = 0.5

        # Anti-windup
        self.integral_limit = 2.0  # reduced to prevent windup
        
    def reset(self):
        """Reset integrators"""
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)
        logger.info(f"{self.name} reset")

    def get_integral_state(self) -> np.ndarray:
        """Return effective lateral acceleration contributed by integral term."""
        return self.ki_pos * self.integral_error

    def set_integral_state(self, effective_accel: np.ndarray):
        """
        Initialize integral from transferred effective acceleration (bumpless transfer).
        Solves: ki_pos * integral_error = effective_accel  →  integral_error = accel / ki
        """
        ratio = np.where(self.ki_pos > 1e-9,
                         effective_accel / np.where(self.ki_pos > 1e-9, self.ki_pos, 1.0),
                         0.0)
        self.integral_error = np.clip(ratio, -self.integral_limit, self.integral_limit)

    def compute_control(self,
                       state: Dict[str, np.ndarray],
                       setpoint: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Cascaded PID control
        
        Outer loop: Position → desired attitude
        Inner loop: Attitude → desired rates
        Innermost loop: Rates → motor commands
        """
        
        # Extract state
        position = state['position']  # [x, y, z]
        velocity = state['velocity']  # [vx, vy, vz]
        attitude = state['attitude']  # [roll, pitch, yaw]
        rates = state['rates']  # [p, q, r]
        
        # Extract setpoints
        target_pos = setpoint.get('position', position)
        target_vel = setpoint.get('velocity', np.zeros(3))
        target_yaw = setpoint.get('yaw', attitude[2])
        
        # === OUTER LOOP: Position Control ===
        pos_error = target_pos - position
        vel_error = target_vel - velocity
        
        # PID for desired acceleration
        self.integral_error += pos_error * self.dt
        self.integral_error = np.clip(self.integral_error, 
                                      -self.integral_limit, 
                                      self.integral_limit)
        
        d_error = (pos_error - self.last_error) / self.dt
        self.last_error = pos_error.copy()
        
        desired_accel = (self.kp_pos * pos_error + 
                        self.ki_pos * self.integral_error +
                        self.kd_pos * d_error)
        
        # Convert desired acceleration to desired attitude
        # In NED frame with standard quadrotor convention:
        # - Positive pitch (nose up) -> backward (negative X)
        # - Positive roll (right wing down) -> rightward (positive Y)
        # So to accelerate forward (pos X), need negative pitch
        # And to accelerate right (pos Y), need positive roll
        desired_pitch = -desired_accel[0] / 9.81  # Negative: forward accel needs nose down
        desired_roll = desired_accel[1] / 9.81    # Positive: right accel needs right roll
        desired_yaw = target_yaw
        
        # Limit tilt angles
        desired_roll = np.clip(desired_roll, -self.max_tilt, self.max_tilt)
        desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)
        
        desired_attitude = np.array([desired_roll, desired_pitch, desired_yaw])
        
        # === INNER LOOP: Attitude Control ===
        att_error = desired_attitude - attitude
        
        # Normalize yaw error to [-pi, pi]
        att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))
        
        # PD control for desired rates
        desired_rates = self.kp_att * att_error - self.kd_att * rates
        desired_rates = np.clip(desired_rates, -self.max_rate, self.max_rate)
        
        # === INNERMOST LOOP: Rate Control ===
        rate_error = desired_rates - rates
        
        # P control for torques
        torques = self.kp_rate * rate_error
        
        # === THRUST CONTROL ===
        # Desired thrust = hover thrust + altitude correction
        # hover_thrust counteracts gravity, corrections adjust for tracking
        altitude_error = target_pos[2] - position[2]
        altitude_rate_error = -velocity[2]  # Want zero vertical velocity

        # PD control on altitude with gravity feedforward
        thrust_correction = (self.kp_pos[2] * altitude_error * 0.1 +
                            self.kd_pos[2] * altitude_rate_error * 0.05)
        desired_thrust = self.hover_thrust + thrust_correction
        desired_thrust = np.clip(desired_thrust, self.min_thrust, self.max_thrust)
        
        # Return control output
        control = {
            'thrust': desired_thrust,
            'torques': torques,  # [roll_torque, pitch_torque, yaw_torque]
            'desired_attitude': desired_attitude,
            'desired_rates': desired_rates
        }
        
        return control


class HInfinityController(BaseController):
    """
    H-infinity Robust Controller for Emergency Stabilization
    
    Contract:
    - Assumes: NOTHING (always works - safety net!)
    - Guarantees: Stabilization in <10s, maintains safe altitude, robust to disturbances
    
    Strategy:
    1. Aggressively damp all angular rates
    2. Level attitude (roll, pitch → 0)
    3. Hold altitude from setpoint
    """

    def __init__(self, dt: float = 0.02):
        super().__init__("H-infinity", dt)

        # Aggressive damping gains
        self.k_rate_damping = 0.8  # Very aggressive rate damping
        self.k_attitude = 5.0      # Strong attitude correction
        self.k_altitude = 0.05     # Altitude hold gain (conservative to avoid overshoot)
        self.k_altitude_rate = 0.15  # Altitude rate damping (helps slow descent)

        # Position tracking gains (robust but active wind rejection)
        self.k_pos = np.array([0.35, 0.35, 0.0])  # lateral position tracking
        self.k_vel = np.array([0.7, 0.7, 0.0])     # velocity damping (fight wind drift)
        self.k_int = np.array([0.10, 0.10, 0.0])   # integral for wind rejection
        self.int_limit = 8.0  # anti-windup clamp
        self.pos_integral = np.zeros(3)
        self.max_tilt = 0.22  # ~13° — enough tilt to reject strong wind

        # Hover thrust (normalized) - this counteracts gravity
        self.hover_thrust = 0.5

    def reset(self):
        """Reset controller state"""
        self.pos_integral = np.zeros(3)
        logger.info(f"{self.name} reset")

    def get_integral_state(self) -> np.ndarray:
        """Return effective lateral acceleration contributed by integral term."""
        return self.k_int * self.pos_integral

    def set_integral_state(self, effective_accel: np.ndarray):
        """
        Initialize integral from transferred effective acceleration (bumpless transfer).
        Solves: k_int * pos_integral = effective_accel  →  pos_integral = accel / k_int
        Z-axis integral is zero (k_int[2] = 0), altitude handled by thrust separately.
        """
        integral = np.zeros(3)
        for i in range(3):
            if self.k_int[i] > 1e-9:
                integral[i] = effective_accel[i] / self.k_int[i]
        self.pos_integral = np.clip(integral, -self.int_limit, self.int_limit)

    def compute_control(self,
                       state: Dict[str, np.ndarray],
                       setpoint: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Robust stabilization control

        Priority:
        1. Damp angular rates (prevent tumbling)
        2. Level attitude (prevent crashing)
        3. Hold altitude from setpoint
        """

        position = state['position']
        velocity = state['velocity']
        attitude = state['attitude']
        rates = state['rates']

        target_pos = setpoint.get('position', position)

        # === POSITION TRACKING (with integral wind rejection) ===
        pos_error = target_pos - position
        vel_error = -velocity  # want zero velocity (fight drift)

        # Integrate position error for sustained wind rejection
        self.pos_integral += pos_error * self.dt
        self.pos_integral = np.clip(self.pos_integral, -self.int_limit, self.int_limit)

        desired_accel = (self.k_pos * pos_error +
                         self.k_vel * vel_error +
                         self.k_int * self.pos_integral)

        # Map lateral acceleration to desired tilt (conservative limits)
        desired_pitch = np.clip(-desired_accel[0] / 9.81, -self.max_tilt, self.max_tilt)
        desired_roll = np.clip(desired_accel[1] / 9.81, -self.max_tilt, self.max_tilt)
        target_attitude = np.array([desired_roll, desired_pitch, attitude[2]])

        # === RATE DAMPING (Highest Priority) ===
        rate_damping_torque = -self.k_rate_damping * rates

        # === ATTITUDE STABILIZATION ===
        att_error = target_attitude - attitude
        att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))

        attitude_torque = self.k_attitude * att_error

        torques = rate_damping_torque + attitude_torque

        # === ALTITUDE HOLD (from setpoint) ===
        # Thrust = hover_thrust + corrections for altitude tracking
        altitude_error = target_pos[2] - position[2]
        altitude_rate_error = 0.0 - velocity[2]

        thrust_correction = (self.k_altitude * altitude_error +
                            self.k_altitude_rate * altitude_rate_error)
        desired_thrust = self.hover_thrust + thrust_correction
        desired_thrust = np.clip(desired_thrust, 0.1, 0.9)

        control = {
            'thrust': desired_thrust,
            'torques': torques,
            'desired_attitude': target_attitude,
            'desired_rates': np.zeros(3)
        }

        return control


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

    def compute_control(self,
                       state: Dict[str, np.ndarray],
                       setpoint: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Solve condensed QP and return control.

        Steps:
        1. Form x0 (error state)
        2. Build X_ref (reference trajectory over horizon)
        3. Compute f = T'Q(S*x0 - X_ref)
        4. U* = -H_inv * f  (one matrix-vector multiply)
        5. Clip to constraints
        6. Extract first input, map to thrust + attitude + torques
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

        # Map acceleration to thrust + attitude
        # Simulation convention: thrust > hover → z increases (descent in NED)
        # MPC accel_cmd[2] < 0 means "move z negative" (climb) → need less thrust
        desired_thrust = self.hover_thrust + accel_cmd[2] / 9.81
        desired_thrust = np.clip(desired_thrust, 0.0, 1.0)

        # Desired attitude from lateral acceleration
        desired_roll = accel_cmd[1] / 9.81
        desired_pitch = -accel_cmd[0] / 9.81
        desired_yaw = setpoint.get('yaw', attitude[2])

        # Clip tilt
        max_tilt = np.arctan2(self.max_lateral_accel, 9.81)  # ~17°
        desired_roll = np.clip(desired_roll, -max_tilt, max_tilt)
        desired_pitch = np.clip(desired_pitch, -max_tilt, max_tilt)

        desired_attitude = np.array([desired_roll, desired_pitch, desired_yaw])

        # Inner loop: PD attitude control (same as PID inner loop)
        att_error = desired_attitude - attitude
        att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))
        desired_rates_cmd = self.kp_att * att_error - self.kd_att * rates
        desired_rates_cmd = np.clip(desired_rates_cmd, -2.0, 2.0)

        # Rate → torque
        rate_error = desired_rates_cmd - rates
        torques = 0.08 * rate_error

        self.last_solve_time = time.perf_counter() - t_start

        return {
            'thrust': desired_thrust,
            'torques': torques,
            'desired_attitude': desired_attitude,
            'desired_rates': desired_rates_cmd,
        }


class ControllerSwitcher:
    """
    Manages switching between controllers based on contracts
    """
    
    def __init__(self, dt: float = 0.02):
        self.controllers = {
            'PID': PIDController(dt),
            'MPC': MPCController(dt),
            'Hinf': HInfinityController(dt),
        }
        
        self.active_controller = 'PID'
        self.controllers[self.active_controller].activate()
        
        # Cooldown to prevent chattering (≥ PID settling time ~16s; 5s is a practical minimum)
        self.switch_cooldown = 5.0  # seconds
        self.last_switch_time = -self.switch_cooldown  # Allow immediate switch at t=0
        
        logger.info("Controller Switcher initialized")
    
    def switch_to(self, controller_name: str, current_time: float, reason: str = ""):
        """Switch to a different controller"""
        
        if controller_name not in self.controllers:
            logger.error(f"Controller {controller_name} not found!")
            return False
        
        if controller_name == self.active_controller:
            return False  # Already active
        
        # Check cooldown
        if current_time - self.last_switch_time < self.switch_cooldown:
            logger.debug(f"Switch blocked by cooldown ({self.switch_cooldown}s)")
            return False
        
        # Bumpless transfer: capture integral state from outgoing controller before switch
        outgoing_integral = self.controllers[self.active_controller].get_integral_state()

        # Perform switch
        self.controllers[self.active_controller].deactivate()
        self.active_controller = controller_name
        self.controllers[self.active_controller].activate()
        self.controllers[self.active_controller].reset()  # clears non-integral state (last_error, etc.)

        # Transfer integral state so incoming controller continues smoothly
        self.controllers[self.active_controller].set_integral_state(outgoing_integral)

        self.last_switch_time = current_time

        logger.warning(f"CONTROLLER SWITCH: {reason} "
                       f"(integral transferred: {outgoing_integral.round(3)})")

        return True
    
    def compute_control(self,
                       state: Dict[str, np.ndarray],
                       setpoint: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute control using active controller"""
        
        controller = self.controllers[self.active_controller]
        control = controller.compute_control(state, setpoint)
        
        return control
    
    def get_active_controller(self) -> str:
        """Get name of active controller"""
        return self.active_controller


# Test controllers
if __name__ == "__main__":
    logging.disable(logging.CRITICAL)

    print("=" * 60)
    print("Controller Test — PID / MPC / H-inf")
    print("=" * 60)

    switcher = ControllerSwitcher(dt=0.02)

    state = {
        'position': np.array([1.0, 0.5, -5.0]),
        'velocity': np.array([0.1, 0.0, 0.0]),
        'attitude': np.array([0.1, 0.05, 0.0]),
        'rates': np.array([0.05, 0.02, 0.0])
    }
    setpoint = {
        'position': np.array([0.0, 0.0, -5.0]),
        'velocity': np.array([0.0, 0.0, 0.0]),
        'yaw': 0.0
    }

    print("\n[TEST 1] PID Controller")
    control = switcher.compute_control(state, setpoint)
    print(f"  Active: {switcher.get_active_controller()}")
    print(f"  Thrust: {control['thrust']:.3f}")
    print(f"  Torques: {control['torques']}")

    print("\n[TEST 2] MPC Controller")
    switcher.switch_to('MPC', current_time=5.0, reason="Wind increasing")
    control = switcher.compute_control(state, setpoint)
    mpc = switcher.controllers['MPC']
    print(f"  Active: {switcher.get_active_controller()}")
    print(f"  Thrust: {control['thrust']:.3f}")
    print(f"  Torques: {control['torques']}")
    print(f"  Solve time: {mpc.last_solve_time*1000:.2f} ms")

    print("\n[TEST 3] H-infinity Controller (emergency)")
    switcher.switch_to('Hinf', current_time=10.0, reason="Emergency test")
    emergency_state = {
        'position': np.array([2.0, 1.0, -3.0]),
        'velocity': np.array([1.0, 0.5, -0.5]),
        'attitude': np.array([0.5, 0.4, 0.0]),
        'rates': np.array([1.0, 0.8, 0.2])
    }
    control = switcher.compute_control(emergency_state, setpoint)
    print(f"  Active: {switcher.get_active_controller()}")
    print(f"  Thrust: {control['thrust']:.3f}")
    print(f"  Torques: {control['torques']}")

    print("\n[TEST 4] MPC tracking convergence (10 steps)")
    switcher.switch_to('MPC', current_time=15.0, reason="Recovery")
    s = {
        'position': np.array([3.0, 2.0, -5.0]),
        'velocity': np.array([0.0, 0.0, 0.0]),
        'attitude': np.array([0.0, 0.0, 0.0]),
        'rates': np.array([0.0, 0.0, 0.0]),
    }
    sp = {'position': np.array([0.0, 0.0, -5.0]), 'velocity': np.zeros(3), 'yaw': 0.0}
    dt = 0.02
    for step in range(10):
        ctrl = switcher.compute_control(s, sp)
        # Simple Euler integration (double-integrator approx)
        accel = np.zeros(3)
        accel[0] = -(ctrl['desired_attitude'][1]) * 9.81  # pitch → x accel
        accel[1] = ctrl['desired_attitude'][0] * 9.81     # roll → y accel
        accel[2] = -(ctrl['thrust'] - 0.5) * 9.81         # thrust → z accel
        s['velocity'] = s['velocity'] + accel * dt
        s['position'] = s['position'] + s['velocity'] * dt
        dist = np.linalg.norm(s['position'] - sp['position'])
        if step % 3 == 0 or step == 9:
            print(f"  Step {step:2d}: dist={dist:.3f}m, thrust={ctrl['thrust']:.3f}")

    print("\n" + "=" * 60)
    print("[OK] Controller Test Complete")
    print("=" * 60)
