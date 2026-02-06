"""
Sensor Fault Injection Module
Configurable fault models for testing contract-based degradation handling.

Supports GPS and IMU faults with time-scheduled injection.
Works as a pure wrapper on sensor data dicts - no modification to existing code needed.
"""

import numpy as np
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


class FaultType:
    """Supported sensor fault types."""
    # GPS faults
    GPS_SATELLITE_LOSS = 'gps_satellite_loss'
    GPS_HDOP_INCREASE = 'gps_hdop_increase'
    GPS_POSITION_DRIFT = 'gps_position_drift'
    GPS_COMPLETE_LOSS = 'gps_complete_loss'
    GPS_INTERMITTENT = 'gps_intermittent'

    # IMU faults
    IMU_NOISE_INCREASE = 'imu_noise_increase'
    IMU_TEMPERATURE_DRIFT = 'imu_temperature_drift'
    IMU_BIAS_ACCUMULATION = 'imu_bias_accumulation'
    IMU_CALIBRATION_LOSS = 'imu_calibration_loss'
    IMU_SPIKE = 'imu_spike'


@dataclass
class SensorFault:
    """Definition of a single sensor fault event."""
    name: str
    sensor: str           # 'gps' or 'imu'
    fault_type: str       # FaultType constant
    t_start: float        # Start time (seconds)
    t_end: float          # End time (seconds)
    severity: float = 1.0
    params: Dict = field(default_factory=dict)


class SensorFaultInjector:
    """
    Applies time-scheduled sensor faults to sensor data dictionaries.

    Usage:
        injector = SensorFaultInjector(fault_schedule)
        degraded = injector.inject(clean_sensor_data, current_time)
    """

    def __init__(self, faults: List[SensorFault]):
        self.faults = faults
        self._bias_state: Dict[str, np.ndarray] = {}
        self._drift_state: Dict[str, np.ndarray] = {}
        self._last_time: Dict[str, float] = {}

    def inject(self, sensor_data: Dict, t: float) -> Dict:
        """Apply all active faults at time t. Returns a deep copy."""
        data = copy.deepcopy(sensor_data)
        for fault in self.faults:
            if fault.t_start <= t < fault.t_end:
                progress = (t - fault.t_start) / max(fault.t_end - fault.t_start, 1e-6)
                data = self._apply_fault(data, fault, t, progress)
        return data

    def get_active_faults(self, t: float) -> List[SensorFault]:
        """Return faults active at time t."""
        return [f for f in self.faults if f.t_start <= t < f.t_end]

    def get_sensor_health(self, t: float) -> Dict[str, float]:
        """Return health score (0.0-1.0) per sensor at time t."""
        gps_health = 1.0
        imu_health = 1.0

        for fault in self.get_active_faults(t):
            progress = (t - fault.t_start) / max(fault.t_end - fault.t_start, 1e-6)
            impact = progress * fault.severity

            if fault.sensor == 'gps':
                if fault.fault_type == FaultType.GPS_COMPLETE_LOSS:
                    gps_health = 0.0
                elif fault.fault_type == FaultType.GPS_INTERMITTENT:
                    period = fault.params.get('period', 2.0)
                    duty = fault.params.get('duty', 0.5)
                    gps_on = (t % period) < (period * duty)
                    gps_health = min(gps_health, 0.5 if gps_on else 0.0)
                else:
                    gps_health = min(gps_health, max(1.0 - impact, 0.0))

            elif fault.sensor == 'imu':
                if fault.fault_type == FaultType.IMU_CALIBRATION_LOSS:
                    imu_health = 0.0
                else:
                    imu_health = min(imu_health, max(1.0 - impact * 0.8, 0.0))

        return {'gps': gps_health, 'imu': imu_health}

    def get_phase_name(self, t: float) -> str:
        """Return descriptive phase name for the current time."""
        active = self.get_active_faults(t)
        if not active:
            return "NOMINAL"

        sensors = set(f.sensor.upper() for f in active)
        if len(sensors) > 1:
            return "COMBINED"

        types = [f.fault_type for f in active]
        if FaultType.GPS_COMPLETE_LOSS in types or FaultType.IMU_CALIBRATION_LOSS in types:
            return f"{list(sensors)[0]} FAILURE"
        if FaultType.GPS_INTERMITTENT in types:
            return "GPS INTERMITTENT"
        return f"{list(sensors)[0]} DEGRADED"

    # --- Fault dispatch ---

    def _apply_fault(self, data: Dict, fault: SensorFault, t: float, progress: float) -> Dict:
        handlers = {
            FaultType.GPS_SATELLITE_LOSS: self._apply_gps_satellite_loss,
            FaultType.GPS_HDOP_INCREASE: self._apply_gps_hdop_increase,
            FaultType.GPS_POSITION_DRIFT: self._apply_gps_position_drift,
            FaultType.GPS_COMPLETE_LOSS: self._apply_gps_complete_loss,
            FaultType.GPS_INTERMITTENT: self._apply_gps_intermittent,
            FaultType.IMU_NOISE_INCREASE: self._apply_imu_noise_increase,
            FaultType.IMU_TEMPERATURE_DRIFT: self._apply_imu_temperature_drift,
            FaultType.IMU_BIAS_ACCUMULATION: self._apply_imu_bias_accumulation,
            FaultType.IMU_CALIBRATION_LOSS: self._apply_imu_calibration_loss,
            FaultType.IMU_SPIKE: self._apply_imu_spike,
        }
        handler = handlers.get(fault.fault_type)
        if handler:
            data = handler(data, fault, t, progress)
        return data

    # --- GPS fault implementations ---

    def _apply_gps_satellite_loss(self, data: Dict, fault: SensorFault, t: float, progress: float) -> Dict:
        min_sats = fault.params.get('min_satellites', 2)
        current = data['gps']['satellites']
        data['gps']['satellites'] = int(current - progress * fault.severity * (current - min_sats))
        return data

    def _apply_gps_hdop_increase(self, data: Dict, fault: SensorFault, t: float, progress: float) -> Dict:
        max_hdop = fault.params.get('max_hdop', 10.0)
        nominal_hdop = 0.8
        data['gps']['hdop'] = nominal_hdop + progress * fault.severity * (max_hdop - nominal_hdop)
        return data

    def _apply_gps_position_drift(self, data: Dict, fault: SensorFault, t: float, progress: float) -> Dict:
        drift_rate = fault.params.get('drift_rate', 0.5)  # m/s
        key = fault.name

        if key not in self._drift_state:
            direction = np.random.randn(3)
            direction[2] = 0  # horizontal drift only
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self._drift_state[key] = direction
            self._last_time[key] = t

        dt = t - self._last_time.get(key, t)
        self._last_time[key] = t

        direction = self._drift_state[key]
        elapsed = t - fault.t_start
        bias = direction * drift_rate * elapsed * fault.severity

        data['gps']['position'] = data['gps']['position'] + bias
        data['gps']['velocity'] = data['gps']['velocity'] + direction * drift_rate * fault.severity * 0.1
        return data

    def _apply_gps_complete_loss(self, data: Dict, fault: SensorFault, t: float, progress: float) -> Dict:
        data['gps']['valid'] = False
        data['gps']['satellites'] = 0
        data['gps']['hdop'] = 99.9
        return data

    def _apply_gps_intermittent(self, data: Dict, fault: SensorFault, t: float, progress: float) -> Dict:
        period = fault.params.get('period', 2.0)
        duty = fault.params.get('duty', 0.5)
        gps_on = (t % period) < (period * duty)

        if not gps_on:
            data['gps']['valid'] = False
            data['gps']['satellites'] = 0
            data['gps']['hdop'] = 99.9
        return data

    # --- IMU fault implementations ---

    def _apply_imu_noise_increase(self, data: Dict, fault: SensorFault, t: float, progress: float) -> Dict:
        max_mult = fault.params.get('max_multiplier', 10.0)
        multiplier = 1.0 + progress * fault.severity * (max_mult - 1.0)

        data['imu']['attitude'] = data['imu']['attitude'] + np.random.randn(3) * 0.001 * multiplier
        data['imu']['rates'] = data['imu']['rates'] + np.random.randn(3) * 0.001 * multiplier
        return data

    def _apply_imu_temperature_drift(self, data: Dict, fault: SensorFault, t: float, progress: float) -> Dict:
        target_temp = fault.params.get('target_temp', 50.0)
        nominal_temp = 25.0
        data['imu']['temperature'] = nominal_temp + progress * fault.severity * (target_temp - nominal_temp)
        return data

    def _apply_imu_bias_accumulation(self, data: Dict, fault: SensorFault, t: float, progress: float) -> Dict:
        bias_rate = fault.params.get('bias_rate', 0.01)  # rad/s growth
        key = fault.name

        if key not in self._bias_state:
            direction = np.random.randn(3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self._bias_state[key] = direction

        direction = self._bias_state[key]
        elapsed = t - fault.t_start
        magnitude = bias_rate * elapsed * fault.severity

        data['imu']['attitude'] = data['imu']['attitude'] + direction * magnitude
        data['imu']['rates'] = data['imu']['rates'] + direction * bias_rate * fault.severity
        return data

    def _apply_imu_calibration_loss(self, data: Dict, fault: SensorFault, t: float, progress: float) -> Dict:
        data['imu']['calibrated'] = False
        return data

    def _apply_imu_spike(self, data: Dict, fault: SensorFault, t: float, progress: float) -> Dict:
        spike_prob = fault.params.get('spike_probability', 0.05)
        spike_mag = fault.params.get('spike_magnitude', 1.0)

        if np.random.random() < spike_prob * fault.severity:
            spike = np.random.randn(3) * spike_mag
            data['imu']['attitude'] = data['imu']['attitude'] + spike
            data['imu']['rates'] = data['imu']['rates'] + spike * 0.5
        return data


def create_standard_degradation_schedule() -> List[SensorFault]:
    """
    Create the standard 7-phase degradation schedule.

    Phases:
      1. Nominal (0-20s)
      2. GPS Degradation (20-35s)
      3. GPS Recovery (35-45s)
      4. IMU Degradation (45-60s)
      5. Combined Failure (60-72s)
      6. Partial Recovery (72-85s)
      7. Full Recovery (85-120s)
    """
    return [
        # Phase 2: GPS Degradation
        SensorFault('GPS sat loss', 'gps', FaultType.GPS_SATELLITE_LOSS,
                    t_start=20.0, t_end=35.0, params={'min_satellites': 3}),
        SensorFault('GPS HDOP rise', 'gps', FaultType.GPS_HDOP_INCREASE,
                    t_start=20.0, t_end=35.0, params={'max_hdop': 6.0}),

        # Phase 4: IMU Degradation
        SensorFault('IMU noise', 'imu', FaultType.IMU_NOISE_INCREASE,
                    t_start=45.0, t_end=60.0, params={'max_multiplier': 10.0}),
        SensorFault('IMU temp drift', 'imu', FaultType.IMU_TEMPERATURE_DRIFT,
                    t_start=45.0, t_end=60.0, params={'target_temp': 42.0}),
        SensorFault('IMU bias', 'imu', FaultType.IMU_BIAS_ACCUMULATION,
                    t_start=45.0, t_end=60.0, params={'bias_rate': 0.005}),

        # Phase 5: Combined Failure
        SensorFault('GPS total loss', 'gps', FaultType.GPS_COMPLETE_LOSS,
                    t_start=60.0, t_end=72.0),
        SensorFault('IMU cal loss', 'imu', FaultType.IMU_CALIBRATION_LOSS,
                    t_start=60.0, t_end=72.0),

        # Phase 6: Partial Recovery
        SensorFault('GPS flicker', 'gps', FaultType.GPS_INTERMITTENT,
                    t_start=72.0, t_end=85.0, params={'period': 3.0, 'duty': 0.5}),
        SensorFault('IMU noise (mild)', 'imu', FaultType.IMU_NOISE_INCREASE,
                    t_start=72.0, t_end=85.0, params={'max_multiplier': 5.0}),
    ]
