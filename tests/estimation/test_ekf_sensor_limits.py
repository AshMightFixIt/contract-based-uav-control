"""estimation/sensor_limits.py and ContractAwareEKF.check_sensor_contracts (W4-08, T6).

gps_ok / imu_ok come from the sensor contract the monitor holds
(contracts/library.py, GPS_IMU_Sensors), not from literals. Before W4-08 the
estimator used 6 satellites and HDOP < 2.0, and "calibrated" alone for the IMU;
it also kept a sensor's last quality fields when its dict disappeared, and it
passed the monitor a variable the contract does not have.
"""
from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.usefixtures("core_package")


def make_ekf():
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.estimation.ekf import ContractAwareEKF
    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    return ContractAwareEKF(monitor)


def sensors(satellites=12, hdop=0.8, gps_valid=True, calibrated=True, temperature=25.0, imu_valid=True):
    return {'gps': {'satellites': satellites, 'hdop': hdop, 'valid': gps_valid},
            'imu': {'calibrated': calibrated, 'temperature': temperature, 'valid': imu_valid}}


def check(ekf, raw):
    ekf.update_sensor_quality(raw)
    return ekf.check_sensor_contracts(1.0)


def test_limits_are_the_sensor_contract_assumptions():
    from contract_uav_core.estimation.sensor_limits import sensor_limits
    ekf = make_ekf()
    contract = ekf.contract_monitor.sensor_contract
    limits = sensor_limits(ekf.contract_monitor)
    assert limits.min_satellites == contract.assumptions['gps_satellites'][0] == 6.0
    assert limits.max_hdop == contract.assumptions['gps_hdop'][1] == 5.0
    assert (limits.min_imu_temperature, limits.max_imu_temperature) == contract.assumptions['imu_temperature'] \
        == (5.0, 55.0)
    assert limits.imu_calibrated == contract.assumptions['imu_calibrated'] == (1.0, 1.0)


def test_monitor_without_contracts_raises():
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.estimation.sensor_limits import sensor_limits
    with pytest.raises(ValueError, match="define_contracts"):
        sensor_limits(HierarchicalContractMonitor())


@pytest.mark.parametrize("satellites, hdop, valid, ok", [
    (12, 0.8, True, True),
    (6, 5.0, True, True),        # both limits are inclusive
    (5, 0.8, True, False),       # below the contract's 6 satellites
    (12, 5.01, True, False),     # above the contract's HDOP 5.0
    (12, 2.5, True, True),       # rejected by the old literal (HDOP < 2.0), allowed by the contract
    (40, 0.3, True, True),       # better than the ideal end of each range is not a failure
    (12, 0.8, False, False),     # flagged invalid
    (12, math.nan, True, False),
    (math.nan, 0.8, True, False),
])
def test_gps_ok_follows_the_contract(satellites, hdop, valid, ok):
    gps_ok, imu_ok = check(make_ekf(), sensors(satellites=satellites, hdop=hdop, gps_valid=valid))
    assert gps_ok is ok
    assert imu_ok is True


@pytest.mark.parametrize("calibrated, temperature, valid, ok", [
    (True, 25.0, True, True),
    (True, 5.0, True, True),
    (True, 55.0, True, True),
    (True, 4.9, True, False),
    (True, 55.1, True, False),
    (False, 25.0, True, False),
    (True, 25.0, False, False),
    (True, math.nan, True, False),
])
def test_imu_ok_follows_the_contract(calibrated, temperature, valid, ok):
    gps_ok, imu_ok = check(make_ekf(), sensors(calibrated=calibrated, temperature=temperature, imu_valid=valid))
    assert imu_ok is ok
    assert gps_ok is True


def test_limits_are_read_from_the_monitor_not_copied():
    """Changing the monitor's contract changes gps_ok / imu_ok: no literal is left in the estimator."""
    ekf = make_ekf()
    assumptions = ekf.contract_monitor.sensor_contract.assumptions
    raw = sensors(satellites=7, hdop=3.0, temperature=40.0)
    assert check(ekf, raw) == (True, True)
    assumptions['gps_hdop'] = (0.5, 2.0)
    assert check(ekf, raw) == (False, True)
    assumptions['gps_hdop'] = (0.5, 5.0)
    assumptions['gps_satellites'] = (8.0, 30.0)
    assert check(ekf, raw) == (False, True)
    assumptions['imu_temperature'] = (5.0, 35.0)
    assert check(ekf, raw) == (False, False)


def test_missing_sensor_dict_counts_as_invalid():
    """A GPS or IMU dict that disappears is not ok any more (before, its last fields were kept)."""
    ekf = make_ekf()
    assert check(ekf, sensors()) == (True, True)
    raw = sensors()
    del raw['gps']
    assert check(ekf, raw) == (False, True)
    raw = sensors()
    del raw['imu']
    assert check(ekf, raw) == (True, False)


def test_monitor_gets_the_contract_variables():
    """The sensor contract's runtime record uses its own variables, so nominal sensors satisfy it."""
    ekf = make_ekf()
    check(ekf, sensors())
    metrics = ekf.sensor_metrics
    assert metrics is ekf.contract_monitor.history[-1]
    assert metrics.component_name == 'sensors'
    assert metrics.assumption_values == {'gps_satellites': 12.0, 'gps_hdop': 0.8,
                                         'imu_temperature': 25.0, 'imu_calibrated': 1.0}
    assert metrics.assumptions_met is True
    assert metrics.predicted_bounds['position_meas_error_max'] == pytest.approx(1.5 * 0.8)
    check(ekf, sensors(calibrated=False))
    assert ekf.sensor_metrics.assumptions_met is False
