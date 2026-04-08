"""
ActuatorInterface — converts AdaptiveDroneController output to Betaflight RC channels.

Architecture
------------
Our code runs the *outer* (position) control loop on the companion computer and
sends attitude + throttle setpoints to Betaflight, which closes the fast inner
(attitude/rate → motor) loop internally at 1–8 kHz.

This is the standard "companion computer offboard control" pattern for Betaflight.
The FC must be in ANGLE mode so that RC roll/pitch channels are interpreted as
angle setpoints, not rate setpoints.

Mapping
-------
    control['desired_attitude'][0]  (roll  rad)  → RC channel 1 (roll)
    control['desired_attitude'][1]  (pitch rad)  → RC channel 2 (pitch)
    control['thrust']               ([0, 1])     → RC channel 3 (throttle)
    control['desired_rates'][2]     (yaw  rad/s) → RC channel 4 (yaw)

RC values are in microseconds: 1000 (min) … 1500 (centre) … 2000 (max).

Safety interlocks
-----------------
- If armed=False the throttle channel is forced to THROTTLE_IDLE.
- All channels are clamped to [RC_MIN, RC_MAX] before transmission.
- An emergency disarm sends RC_ARM channel = RC_DISARMED.
"""

import math
import logging
from typing import Dict, List, Optional

import numpy as np

from . import hardware_config as cfg

logger = logging.getLogger(__name__)

RAD2DEG = 180.0 / math.pi


class ActuatorInterface:
    """
    Converts a control dict → RC channel list and sends it via MSPInterface.

    Typical usage in the flight loop:
        ok = actuator.send(control_output, armed=True)
    """

    def __init__(self, msp):
        """
        Parameters
        ----------
        msp : MSPInterface
            A connected MSPInterface instance.
        """
        self._msp   = msp
        self._armed = False

        # Build a safe neutral RC frame (used for pre-arm and fallback)
        self._neutral = self._build_neutral()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def arm(self) -> bool:
        """
        Send the ARM signal on AUX1.
        Betaflight requires the throttle to be at the bottom (≤ 1100) while arming.
        Returns True if the frame was sent.
        """
        channels = self._neutral.copy()
        channels[cfg.RC_THROTTLE] = cfg.THROTTLE_IDLE
        channels[cfg.RC_ARM]      = cfg.RC_ARMED
        channels[cfg.RC_MODE]     = cfg.RC_ANGLE_MODE_ON
        ok = self._msp.set_rc_channels(channels)
        if ok:
            self._armed = True
            logger.info("ARM signal sent")
        return ok

    def disarm(self) -> bool:
        """
        Send the DISARM signal. Throttle is forced to THROTTLE_IDLE first.
        Returns True if the frame was sent.
        """
        channels = self._neutral.copy()
        channels[cfg.RC_THROTTLE] = cfg.THROTTLE_IDLE
        channels[cfg.RC_ARM]      = cfg.RC_DISARMED
        ok = self._msp.set_rc_channels(channels)
        if ok:
            self._armed = False
            logger.warning("DISARM signal sent")
        return ok

    def send(self, control: Dict[str, np.ndarray], armed: bool = True) -> bool:
        """
        Convert a control dict to RC channels and transmit.

        Parameters
        ----------
        control : dict
            Output from AdaptiveDroneController.control_step() — must contain
            at minimum 'thrust', 'desired_attitude', 'desired_rates'.
        armed : bool
            Pass False to force throttle to idle regardless of thrust command.

        Returns True if transmission succeeded.
        """
        channels = self._control_to_rc(control, armed=armed)
        return self._msp.set_rc_channels(channels)

    def send_hover(self) -> bool:
        """Send a zero-movement hover command (all axes centred, hover throttle)."""
        channels = self._neutral.copy()
        channels[cfg.RC_THROTTLE] = cfg.THROTTLE_HOVER
        channels[cfg.RC_ARM]      = cfg.RC_ARMED if self._armed else cfg.RC_DISARMED
        channels[cfg.RC_MODE]     = cfg.RC_ANGLE_MODE_ON
        return self._msp.set_rc_channels(channels)

    def send_emergency_disarm(self) -> bool:
        """
        Cut throttle and disarm immediately.
        Use only when the drone is on the ground — in-flight disarm will crash.
        """
        channels = [cfg.RC_MIN] * cfg.RC_NUM_CHANNELS
        channels[cfg.RC_ARM]  = cfg.RC_DISARMED
        channels[cfg.RC_MODE] = cfg.RC_ANGLE_MODE_ON
        ok = self._msp.set_rc_channels(channels)
        self._armed = False
        return ok

    # ------------------------------------------------------------------
    # Conversion logic
    # ------------------------------------------------------------------

    def _control_to_rc(self, control: Dict, armed: bool) -> List[int]:
        """
        Map controller output to a list of RC pulse widths.

        Attitude convention in this codebase (NED frame):
            roll  > 0 → right wing down  → drone moves East
            pitch > 0 → nose up          → drone moves South  (decelerates North)
        Betaflight Angle mode convention (same sense for roll; pitch sign varies
        by build — verify in Betaflight configurator and flip if needed).
        """
        channels = self._neutral.copy()

        # ---- Throttle ----
        thrust = float(control.get('thrust', 0.5))
        if not armed:
            channels[cfg.RC_THROTTLE] = cfg.THROTTLE_IDLE
        else:
            # Map thrust [0, 1] → [THROTTLE_IDLE, THROTTLE_MAX]
            throttle_range = cfg.THROTTLE_MAX - cfg.THROTTLE_IDLE
            rc_throttle = cfg.THROTTLE_IDLE + int(thrust * throttle_range)
            channels[cfg.RC_THROTTLE] = self._clamp(rc_throttle)

        # ---- Roll / Pitch from desired attitude ----
        desired_att = control.get('desired_attitude', np.zeros(3))
        if desired_att is not None and len(desired_att) >= 2:
            roll_rad  = float(desired_att[0])
            pitch_rad = float(desired_att[1])

            roll_deg  = roll_rad  * RAD2DEG
            pitch_deg = pitch_rad * RAD2DEG

            # Clamp to configured limits
            roll_deg  = max(-cfg.MAX_ROLL_DEG,  min(cfg.MAX_ROLL_DEG,  roll_deg))
            pitch_deg = max(-cfg.MAX_PITCH_DEG, min(cfg.MAX_PITCH_DEG, pitch_deg))

            # Map degrees to RC: 0° → 1500, ±MAX → 1500 ± 500
            channels[cfg.RC_ROLL]  = self._angle_to_rc(roll_deg,  cfg.MAX_ROLL_DEG)
            channels[cfg.RC_PITCH] = self._angle_to_rc(pitch_deg, cfg.MAX_PITCH_DEG)

        # ---- Yaw rate from desired rates ----
        desired_rates = control.get('desired_rates', np.zeros(3))
        if desired_rates is not None and len(desired_rates) >= 3:
            yaw_rate_rads = float(desired_rates[2])
            yaw_rate_dps  = yaw_rate_rads * RAD2DEG
            yaw_rate_dps  = max(-cfg.MAX_YAW_RATE_DPS,
                                 min(cfg.MAX_YAW_RATE_DPS, yaw_rate_dps))
            channels[cfg.RC_YAW] = self._rate_to_rc(yaw_rate_dps, cfg.MAX_YAW_RATE_DPS)

        # ---- Arm / Mode ----
        channels[cfg.RC_ARM]  = cfg.RC_ARMED if armed else cfg.RC_DISARMED
        channels[cfg.RC_MODE] = cfg.RC_ANGLE_MODE_ON

        return channels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_neutral() -> List[int]:
        """Return a neutral (centred, disarmed, idle) RC frame."""
        channels = [cfg.RC_MID] * cfg.RC_NUM_CHANNELS
        channels[cfg.RC_THROTTLE] = cfg.THROTTLE_IDLE
        channels[cfg.RC_ARM]      = cfg.RC_DISARMED
        channels[cfg.RC_MODE]     = cfg.RC_ANGLE_MODE_ON
        return channels

    @staticmethod
    def _angle_to_rc(angle_deg: float, max_deg: float) -> int:
        """Map an angle in degrees to an RC pulse width centred at 1500."""
        fraction = angle_deg / max_deg          # [-1, 1]
        rc = cfg.RC_MID + int(fraction * 500)   # 1500 ± 500
        return max(cfg.RC_MIN, min(cfg.RC_MAX, rc))

    @staticmethod
    def _rate_to_rc(rate_dps: float, max_dps: float) -> int:
        """Map a rate in degrees/s to an RC pulse width centred at 1500."""
        fraction = rate_dps / max_dps
        rc = cfg.RC_MID + int(fraction * 500)
        return max(cfg.RC_MIN, min(cfg.RC_MAX, rc))

    @staticmethod
    def _clamp(value: int) -> int:
        return max(cfg.RC_MIN, min(cfg.RC_MAX, value))
