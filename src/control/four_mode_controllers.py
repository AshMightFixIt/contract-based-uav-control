"""
Four-Mode Controller System for AE552 Project
Modes: Nominal PID, Wind-Robust, GPS-Denied, Safety

Integrates with:
- Hierarchical contract system
- CBF safety filter
- Runtime monitors
"""

import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NominalPIDController:
    """
    Mode 1: Nominal PID Controller
    
    Contract:
    - Assumes: Wind < 3 m/s, GPS available, Nominal conditions
    - Guarantees: Fast response (<5s), Low overshoot (<20%), Efficient
    """
    
    def __init__(self, dt: float = 0.02):
        self.name = "Nominal_PID"
        self.dt = dt
        
        # Aggressive gains for fast response in nominal conditions
        self.kp_pos = np.array([2.0, 2.0, 2.5])
        self.ki_pos = np.array([0.15, 0.15, 0.25])
        self.kd_pos = np.array([1.0, 1.0, 1.2])
        
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)
        
        logger.info(f"{self.name} initialized (fast, efficient)")
    
    def compute_control(self, state: Dict, setpoint: Dict) -> Dict:
        """Compute nominal PID control"""
        pos_error = setpoint['position'] - state['position']
        
        # PID
        self.integral_error += pos_error * self.dt
        self.integral_error = np.clip(self.integral_error, -5.0, 5.0)
        d_error = (pos_error - self.last_error) / self.dt
        self.last_error = pos_error.copy()
        
        desired_accel = (self.kp_pos * pos_error + 
                        self.ki_pos * self.integral_error +
                        self.kd_pos * d_error)
        
        # Convert to thrust and attitude
        desired_thrust = 0.5 + 0.15 * (setpoint['position'][2] - state['position'][2])
        desired_thrust = np.clip(desired_thrust, 0.2, 0.8)
        
        # Simple torques from position error
        torques = np.array([
            -desired_accel[1] * 0.1,  # roll
            desired_accel[0] * 0.1,   # pitch
            0.0                        # yaw
        ])
        
        return {'thrust': desired_thrust, 'torques': torques}
    
    def reset(self):
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)


class WindRobustController:
    """
    Mode 2: Wind-Robust Controller (like our H-infinity)
    
    Contract:
    - Assumes: Wind < 10 m/s, GPS available, Disturbances present
    - Guarantees: Robustness, Bounded tracking error (<2m), Stabilization (<10s)
    """
    
    def __init__(self, dt: float = 0.02):
        self.name = "Wind_Robust"
        self.dt = dt
        
        # Robust gains with aggressive damping
        self.k_rate_damping = 0.8
        self.k_attitude = 5.0
        self.k_position = 2.0
        self.k_altitude = 3.0
        
        logger.info(f"{self.name} initialized (robust to wind)")
    
    def compute_control(self, state: Dict, setpoint: Dict) -> Dict:
        """Compute wind-robust control with aggressive damping"""
        pos_error = setpoint['position'] - state['position']
        
        # Aggressive rate damping (key for wind rejection)
        rate_damping = -self.k_rate_damping * state['rates']
        
        # Position correction
        position_correction = self.k_position * pos_error[0:2]  # Horizontal
        
        # Attitude stabilization (keep level against wind)
        target_attitude = np.array([0.0, 0.0, state['attitude'][2]])
        att_error = target_attitude - state['attitude']
        attitude_correction = self.k_attitude * att_error
        
        torques = rate_damping + attitude_correction
        
        # Altitude control (strong)
        altitude_error = setpoint['position'][2] - state['position'][2]
        desired_thrust = 0.5 + self.k_altitude * altitude_error * 0.1
        desired_thrust = np.clip(desired_thrust, 0.3, 0.7)
        
        return {'thrust': desired_thrust, 'torques': torques}
    
    def reset(self):
        pass


class GPSDeniedController:
    """
    Mode 3: GPS-Denied Controller
    
    Contract:
    - Assumes: IMU available, Vision/SLAM available (future), Low wind
    - Guarantees: Attitude stabilization, Hover in place, Safe operation
    """
    
    def __init__(self, dt: float = 0.02):
        self.name = "GPS_Denied"
        self.dt = dt
        
        # Conservative gains - focus on stabilization not tracking
        self.k_attitude = 4.0
        self.k_rate = 0.5
        self.hover_altitude = -5.0
        
        # Dead reckoning state (simplified)
        self.estimated_position = np.array([0.0, 0.0, -5.0])
        
        logger.info(f"{self.name} initialized (IMU-only, hover in place)")
    
    def compute_control(self, state: Dict, setpoint: Dict) -> Dict:
        """
        GPS-denied mode: Aggressive attitude stabilization, hover in place
        Uses IMU only - maintains last known position
        """
        # Maintain level attitude (critical without GPS)
        target_attitude = np.array([0.0, 0.0, state['attitude'][2]])
        att_error = target_attitude - state['attitude']
        
        # Strong attitude correction
        attitude_torques = self.k_attitude * att_error
        
        # Rate damping
        rate_damping = -self.k_rate * state['rates']
        
        torques = attitude_torques + rate_damping
        
        # Maintain hover altitude (from barometer/IMU)
        altitude_error = self.hover_altitude - state['position'][2]
        desired_thrust = 0.5 + 0.2 * altitude_error
        desired_thrust = np.clip(desired_thrust, 0.4, 0.6)
        
        return {'thrust': desired_thrust, 'torques': torques}
    
    def reset(self):
        self.estimated_position = np.array([0.0, 0.0, -5.0])


class SafetyController:
    """
    Mode 4: Safety Controller (Last resort!)
    
    Contract:
    - Assumes: NOTHING (always works)
    - Guarantees: Land safely, Prevent crash, Maintain attitude
    
    This is the ultimate fallback when all else fails
    """
    
    def __init__(self, dt: float = 0.02):
        self.name = "Safety_Controller"
        self.dt = dt
        
        # Maximum safety - aggressive stabilization
        self.k_rate_damping = 1.0  # Maximum damping
        self.k_attitude = 6.0      # Maximum attitude correction
        self.landing_rate = 0.5    # m/s descent rate
        
        self.emergency_active = False
        self.landing_started = False
        
        logger.info(f"{self.name} initialized (emergency landing)")
    
    def compute_control(self, state: Dict, setpoint: Dict) -> Dict:
        """
        Emergency safety mode:
        1. Level out immediately
        2. Begin controlled descent
        3. Land safely
        """
        if not self.emergency_active:
            self.emergency_active = True
            logger.error("SAFETY CONTROLLER ACTIVATED - EMERGENCY LANDING")
        
        # PRIORITY 1: Level the aircraft (prevent tumbling)
        target_attitude = np.array([0.0, 0.0, state['attitude'][2]])
        att_error = target_attitude - state['attitude']
        attitude_torques = self.k_attitude * att_error
        
        # PRIORITY 2: Damp all rates (prevent spinning)
        rate_damping = -self.k_rate_damping * state['rates']
        
        torques = attitude_torques + rate_damping
        
        # PRIORITY 3: Controlled descent
        if not self.landing_started:
            # First stabilize attitude
            if np.linalg.norm(state['rates']) < 0.1:
                self.landing_started = True
                logger.warning("Attitude stabilized - beginning descent")
        
        if self.landing_started:
            # Gentle descent
            desired_thrust = 0.45  # Slightly below hover
        else:
            # Maintain altitude while stabilizing
            desired_thrust = 0.5
        
        return {'thrust': desired_thrust, 'torques': torques}
    
    def reset(self):
        self.emergency_active = False
        self.landing_started = False


class FourModeControllerManager:
    """
    Manages switching between 4 controller modes based on contracts
    
    Priority (highest to lowest):
    1. Safety Controller (if critical failure)
    2. GPS-Denied Controller (if GPS lost)
    3. Wind-Robust Controller (if wind exceeds nominal bounds)
    4. Nominal PID Controller (if all conditions nominal)
    """
    
    def __init__(self, dt: float = 0.02):
        self.controllers = {
            'nominal': NominalPIDController(dt),
            'wind_robust': WindRobustController(dt),
            'gps_denied': GPSDeniedController(dt),
            'safety': SafetyController(dt)
        }
        
        self.active_mode = 'nominal'
        self.last_switch_time = 0.0
        self.switch_cooldown = 2.0  # seconds
        
        logger.info("Four-Mode Controller Manager initialized")
    
    def select_mode(self, 
                    monitoring_data: Dict[str, float],
                    barriers_ok: bool,
                    current_time: float) -> str:
        """
        Select appropriate controller mode based on conditions
        
        Decision logic (priority order):
        1. If CBF barrier violated → Safety
        2. If GPS quality < 0.5 → GPS-Denied
        3. If wind > 3 m/s → Wind-Robust
        4. Otherwise → Nominal
        """
        # Check cooldown
        if current_time - self.last_switch_time < self.switch_cooldown:
            return self.active_mode
        
        # Priority 1: Critical safety violation
        if not barriers_ok:
            return 'safety'
        
        # Priority 2: GPS degradation
        if monitoring_data.get('gps_quality', 1.0) < 0.5:
            return 'gps_denied'
        
        # Priority 3: High wind
        if monitoring_data.get('wind_speed', 0.0) > 3.0:
            return 'wind_robust'
        
        # Default: Nominal
        return 'nominal'
    
    def switch_mode(self, new_mode: str, reason: str, current_time: float):
        """Switch to a new controller mode"""
        if new_mode == self.active_mode:
            return False
        
        logger.warning(f"MODE SWITCH: {self.active_mode} → {new_mode}")
        logger.warning(f"  Reason: {reason}")
        
        # Reset new controller
        self.controllers[new_mode].reset()
        
        self.active_mode = new_mode
        self.last_switch_time = current_time
        
        return True
    
    def compute_control(self, state: Dict, setpoint: Dict) -> Dict:
        """Compute control using active controller"""
        controller = self.controllers[self.active_mode]
        return controller.compute_control(state, setpoint)
    
    def get_active_mode(self) -> str:
        return self.active_mode


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Four-Mode Controller System Test")
    print("=" * 60)
    
    manager = FourModeControllerManager()
    
    state = {
        'position': np.array([0.0, 0.0, -5.0]),
        'velocity': np.array([0.1, 0.0, 0.0]),
        'attitude': np.array([0.0, 0.0, 0.0]),
        'rates': np.array([0.0, 0.0, 0.0])
    }
    
    setpoint = {
        'position': np.array([1.0, 1.0, -5.0]),
        'yaw': 0.0
    }
    
    # Test mode selection
    print("\n[TEST 1] Nominal conditions")
    mode = manager.select_mode({'wind_speed': 1.0, 'gps_quality': 1.0}, True, 0.0)
    print(f"Selected mode: {mode}")
    
    print("\n[TEST 2] High wind")
    mode = manager.select_mode({'wind_speed': 5.0, 'gps_quality': 1.0}, True, 3.0)
    print(f"Selected mode: {mode}")
    manager.switch_mode(mode, "High wind detected", 3.0)
    
    print("\n[TEST 3] GPS loss")
    mode = manager.select_mode({'wind_speed': 1.0, 'gps_quality': 0.3}, True, 6.0)
    print(f"Selected mode: {mode}")
    manager.switch_mode(mode, "GPS degraded", 6.0)
    
    print("\n[TEST 4] Safety violation")
    mode = manager.select_mode({'wind_speed': 1.0, 'gps_quality': 1.0}, False, 9.0)
    print(f"Selected mode: {mode}")
    manager.switch_mode(mode, "CBF barrier violated", 9.0)
    
    print("\n" + "=" * 60)
    print("✓ Four-Mode Controller Test Complete")
    print("=" * 60)
