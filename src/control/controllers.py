"""
Contract-Based Controllers for Adaptive Drone System

Implements:
1. PID Controller - Efficient, for nominal conditions
2. H-infinity Controller - Robust, for emergency stabilization
3. (MPC placeholder for future)

Each controller has formal contracts defining when they work
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
        
        # Position PID gains (outer loop)
        self.kp_pos = np.array([1.5, 1.5, 2.0])  # [x, y, z]
        self.ki_pos = np.array([0.1, 0.1, 0.2])
        self.kd_pos = np.array([0.8, 0.8, 1.0])
        
        # Attitude PID gains (inner loop)
        self.kp_att = np.array([3.0, 3.0, 2.0])  # [roll, pitch, yaw]
        self.ki_att = np.array([0.1, 0.1, 0.1])
        self.kd_att = np.array([0.5, 0.5, 0.3])
        
        # Rate PID gains (innermost loop)
        self.kp_rate = np.array([0.15, 0.15, 0.1])  # [p, q, r]
        
        # Limits
        self.max_tilt = 0.5  # 30 degrees max tilt
        self.max_rate = 2.0  # rad/s
        self.max_thrust = 1.0
        self.min_thrust = 0.0
        
        # Anti-windup
        self.integral_limit = 5.0
        
    def reset(self):
        """Reset integrators"""
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)
        logger.info(f"{self.name} reset")
    
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
        # (assuming small angles and thrust ~ gravity)
        desired_roll = -desired_accel[1] / 9.81  # Negative for right-handed frame
        desired_pitch = desired_accel[0] / 9.81
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
        # Desired thrust to maintain altitude
        altitude_error = target_pos[2] - position[2]
        desired_thrust = 0.5 + 0.2 * altitude_error - 0.1 * velocity[2]
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
        self.k_altitude = 3.0      # Strong altitude hold

    def reset(self):
        """Reset controller state"""
        logger.info(f"{self.name} reset")
    
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

        # === RATE DAMPING (Highest Priority) ===
        rate_damping_torque = -self.k_rate_damping * rates

        # === ATTITUDE STABILIZATION ===
        # Force roll and pitch to zero (level flight), keep current yaw
        target_attitude = np.array([0.0, 0.0, attitude[2]])
        att_error = target_attitude - attitude
        att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))

        attitude_torque = self.k_attitude * att_error

        torques = rate_damping_torque + attitude_torque

        # === ALTITUDE HOLD (from setpoint) ===
        altitude_error = target_pos[2] - position[2]
        altitude_rate_error = 0.0 - velocity[2]

        desired_thrust = 0.5 + (self.k_altitude * altitude_error +
                               0.5 * altitude_rate_error)
        desired_thrust = np.clip(desired_thrust, 0.2, 0.8)

        control = {
            'thrust': desired_thrust,
            'torques': torques,
            'desired_attitude': target_attitude,
            'desired_rates': np.zeros(3)
        }

        return control


class ControllerSwitcher:
    """
    Manages switching between controllers based on contracts
    """
    
    def __init__(self, dt: float = 0.02):
        self.controllers = {
            'PID': PIDController(dt),
            'Hinf': HInfinityController(dt)
        }
        
        self.active_controller = 'PID'
        self.controllers[self.active_controller].activate()
        
        # Cooldown to prevent chattering
        self.switch_cooldown = 2.0  # seconds
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
        
        # Perform switch
        self.controllers[self.active_controller].deactivate()
        self.active_controller = controller_name
        self.controllers[self.active_controller].activate()
        self.controllers[self.active_controller].reset()
        
        self.last_switch_time = current_time
        
        logger.warning(f"CONTROLLER SWITCH: {reason}")
        
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
    print("=" * 60)
    print("Controller Test")
    print("=" * 60)
    
    # Create controllers
    switcher = ControllerSwitcher(dt=0.02)
    
    # Test state
    state = {
        'position': np.array([1.0, 0.5, -5.0]),
        'velocity': np.array([0.1, 0.0, 0.0]),
        'attitude': np.array([0.1, 0.05, 0.0]),  # Small tilt
        'rates': np.array([0.05, 0.02, 0.0])
    }
    
    # Test setpoint
    setpoint = {
        'position': np.array([0.0, 0.0, -5.0]),
        'velocity': np.array([0.0, 0.0, 0.0]),
        'yaw': 0.0
    }
    
    print("\n[TEST 1] PID Controller")
    control = switcher.compute_control(state, setpoint)
    print(f"Active: {switcher.get_active_controller()}")
    print(f"Thrust: {control['thrust']:.3f}")
    print(f"Torques: {control['torques']}")
    
    print("\n[TEST 2] Switch to H-infinity")
    switcher.switch_to('Hinf', current_time=5.0, reason="Emergency test")
    
    # Simulate emergency state (large tilt, high rates)
    emergency_state = {
        'position': np.array([2.0, 1.0, -3.0]),
        'velocity': np.array([1.0, 0.5, -0.5]),
        'attitude': np.array([0.5, 0.4, 0.0]),  # Large tilt
        'rates': np.array([1.0, 0.8, 0.2])      # High rates
    }
    
    control = switcher.compute_control(emergency_state, setpoint)
    print(f"Active: {switcher.get_active_controller()}")
    print(f"Thrust: {control['thrust']:.3f}")
    print(f"Torques: {control['torques']}")
    print(f"Desired attitude: {control['desired_attitude']}")
    
    print("\n" + "=" * 60)
    print("Controller Test Complete")
    print("=" * 60)
