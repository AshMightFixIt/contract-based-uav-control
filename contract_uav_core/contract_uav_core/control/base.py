"""
BaseController: the common base class of the PID, MPC and H-infinity controllers.

Moved from controllers.py; see control/pid.py, mpc.py, hinf.py and
switcher.py. Each tier implements compute_accel_command (an AccelCommand,
interfaces.py); compute_control returns its legacy dict, unchanged in keys,
order and values.
"""

import numpy as np
from typing import Dict
import logging
from abc import ABC, abstractmethod

from ..interfaces import AccelCommand

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
    def compute_accel_command(self,
                              state: Dict[str, np.ndarray],
                              setpoint: Dict[str, np.ndarray]) -> AccelCommand:
        """Compute this step's command (advances the controller's internal state)."""
        pass

    def compute_control(self,
                       state: Dict[str, np.ndarray],
                       setpoint: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute control output: the legacy dict of compute_accel_command.

        Keys, in order: thrust, torques, desired_attitude, desired_rates. Call
        either this or compute_accel_command once per step, not both: each
        call advances the integrators and the MPC warm start.
        """
        return self.compute_accel_command(state, setpoint).to_legacy_dict()
    
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
