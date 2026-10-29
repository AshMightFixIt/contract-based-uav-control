"""
BaseController: the common base class of the PID, MPC and H-infinity controllers.

Moved from controllers.py; see control/pid.py, mpc.py, hinf.py and
switcher.py. Each tier implements compute_accel_command (an AccelCommand,
interfaces.py); compute_control returns its legacy dict, unchanged in keys,
order and types.

Airframe (P1.2, W4-05): each tier takes an optional airframe, a
config.airframe.Airframe (anything with hover_thrust, min_thrust and
max_thrust works). Its hover thrust normalises the collective thrust
(frames.accel_to_thrust), and the thrust is kept inside its
[min_thrust, max_thrust] (PX4's MPC_THR_MIN and MPC_THR_MAX). Without one the
hover thrust is frames.DEFAULT_HOVER_THRUST (0.5) and the limits are [0, 1],
the values every tier used before.
"""

import math

import numpy as np
from typing import Dict, Optional
import logging
from abc import ABC, abstractmethod

from ..frames import DEFAULT_HOVER_THRUST
from ..interfaces import AccelCommand

logger = logging.getLogger(__name__)


class BaseController(ABC):
    """Base class for all controllers"""

    def __init__(self, name: str, dt: float = 0.02, airframe: Optional[object] = None):
        self.name = name
        self.dt = dt
        self.active = False
        self.last_error = np.zeros(3)
        self.integral_error = np.zeros(3)

        # Collective thrust normalisation and limits (P1.2): see the module docstring.
        self.airframe = airframe
        if airframe is None:
            self.hover_thrust, self.min_thrust, self.max_thrust = DEFAULT_HOVER_THRUST, 0.0, 1.0
        else:
            self.hover_thrust = float(airframe.hover_thrust)
            self.min_thrust = float(airframe.min_thrust)
            self.max_thrust = float(airframe.max_thrust)
        if not (all(math.isfinite(v) for v in (self.hover_thrust, self.min_thrust, self.max_thrust))
                and 0.0 <= self.min_thrust <= self.hover_thrust <= self.max_thrust <= 1.0
                and self.hover_thrust > 0.0):
            raise ValueError(f"{name}: need 0 <= min_thrust <= hover_thrust <= max_thrust <= 1 and "
                             f"hover_thrust > 0, got {self.min_thrust}, {self.hover_thrust}, {self.max_thrust}")

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
