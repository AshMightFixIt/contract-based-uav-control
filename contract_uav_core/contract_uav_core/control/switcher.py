"""
ControllerSwitcher: holds the PID, MPC and H-infinity controllers and switches
between them (moved unchanged from controllers.py, with its self-test).

Self-test: python -m contract_uav_core.control.switcher
"""

import numpy as np
from typing import Dict
import logging

from .pid import PIDController
from .mpc import MPCController
from .hinf import HInfinityController
from ..interfaces import AccelCommand

logger = logging.getLogger(__name__)


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

    def compute_accel_command(self,
                              state: Dict[str, np.ndarray],
                              setpoint: Dict[str, np.ndarray]) -> AccelCommand:
        """The active controller's AccelCommand (instead of compute_control, not in addition)."""
        return self.controllers[self.active_controller].compute_accel_command(state, setpoint)
    
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
