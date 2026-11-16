"""
Read-only getters: the sensor limits behind the estimator's gps_ok / imu_ok (W4-08, T6).

No limit values of its own: everything is read from the sensor contract that the
HierarchicalContractMonitor holds (monitor.sensor_contract, "GPS_IMU_Sensors",
defined in contracts/library.py). Its assumption ranges today:
  gps_satellites (6.0, 30.0), gps_hdop (0.5, 5.0),
  imu_temperature (5.0, 55.0), imu_calibrated (1.0, 1.0).

How ContractAwareEKF.check_sensor_contracts uses them:
- GPS ok: the fix is flagged valid, satellites >= min_satellites and
  hdop <= max_hdop. Only the "worse" side of each range is applied: more
  satellites than 30, or an HDOP below 0.5, is a better fix, not a broken one
  (a multi-constellation receiver can report more than 30 satellites).
- IMU ok: the IMU is flagged valid and calibrated, and its temperature is
  inside [min_imu_temperature, max_imu_temperature] (both sides matter).
The bounds are inclusive, as in SimpleContract.check_assumptions.

Before W4-08 the estimator used its own literals instead: 6 satellites and
HDOP < 2.0, and "calibrated" alone for the IMU.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorLimits:
    """The sensor contract's limits, as gps_ok / imu_ok apply them (see the module docstring)."""
    min_satellites: float
    max_hdop: float
    min_imu_temperature: float
    max_imu_temperature: float
    imu_calibrated: tuple  # (low, high): the required range of the 0/1 calibrated flag


def sensor_limits(contract_monitor) -> SensorLimits:
    """The limits in contract_monitor.sensor_contract's assumptions.

    Raises ValueError if the monitor has no sensor contract yet (call its
    define_contracts() first) or the contract lacks one of the four variables.
    """
    contract = getattr(contract_monitor, 'sensor_contract', None)
    if contract is None:
        raise ValueError("the contract monitor has no sensor contract: call define_contracts() first")
    assumptions = contract.assumptions
    missing = [v for v in ('gps_satellites', 'gps_hdop', 'imu_temperature', 'imu_calibrated')
               if v not in assumptions]
    if missing:
        raise ValueError(f"sensor contract {contract.name!r} has no assumption on {', '.join(missing)}")
    t_lo, t_hi = assumptions['imu_temperature']
    c_lo, c_hi = assumptions['imu_calibrated']
    return SensorLimits(
        min_satellites=float(assumptions['gps_satellites'][0]),
        max_hdop=float(assumptions['gps_hdop'][1]),
        min_imu_temperature=float(t_lo),
        max_imu_temperature=float(t_hi),
        imu_calibrated=(float(c_lo), float(c_hi)),
    )
