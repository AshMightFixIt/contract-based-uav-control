"""Pre-flight on real telemetry and an airframe (W4-09, T4; A-F11, core half).

AdaptiveDroneController.pre_flight_check(telemetry, mission, airframe=None),
contracts/preflight.py. Before W4-09 (base 36a06ec, same calls):
- a NaN battery voltage or wind speed passed (feasible, PID);
- a missing input made a controller "infeasible" without naming it, and a
  missing disturbance silently gave MPC;
- a DEXI battery at 22 V (6 cells) or motors at 15 degC made every controller
  infeasible: the contracts assume a 3-cell pack (11.0-12.6 V) and a motor
  temperature of 20-80 degC, which PX4 does not report.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

GOOD = {
    'gps_satellites': 12.0, 'gps_hdop': 1.0, 'imu_temperature': 25.0, 'imu_calibrated': 1.0,
    'battery_voltage': 12.4, 'motor_temperature': 25.0, 'wind_speed': 1.5, 'disturbance': 0.5,
}
TARGET = np.array([10.0, 0.0, -5.0])
MISSION = {'target_position': TARGET, 'waypoints': [TARGET]}


def preflight(airframe=None, drop=(), **values):
    from contract_uav_core.core import AdaptiveDroneController

    telemetry = {k: v for k, v in dict(GOOD, **values).items() if k not in drop}
    ctrl = AdaptiveDroneController(dt=0.02, use_horizon_planner=False)
    if airframe is None:
        ok, msg = ctrl.pre_flight_check(telemetry, MISSION)
    else:
        ok, msg = ctrl.pre_flight_check(telemetry, MISSION, airframe=airframe)
    assert ctrl.mission_feasible is ok
    return ok, msg


def test_good_telemetry_unchanged():
    assert preflight() == (True, "Mission feasible with efficient PID")


def test_pipeline_inputs():
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor

    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    assert monitor.pipeline_inputs() == (
        'battery_voltage', 'computation_time', 'disturbance', 'gps_hdop', 'gps_satellites',
        'imu_calibrated', 'imu_temperature', 'motor_temperature', 'wind_speed')


def test_missing_inputs_are_named():
    assert preflight(drop=('gps_hdop', 'imu_temperature')) == (
        False, "Pre-flight inputs rejected: missing: gps_hdop, imu_temperature")
    # None is "no reading"; a missing disturbance no longer falls through to MPC.
    assert preflight(disturbance=None) == (False, "Pre-flight inputs rejected: missing: disturbance")
    # computation_time is optional (the MPC check uses 0.01 s, as before).
    assert preflight(computation_time=None)[0] is True


def test_unknown_keys_warn_and_are_ignored(caplog):
    with caplog.at_level(logging.WARNING, logger="contract_uav_core.contracts.preflight"):
        ok, msg = preflight(imu_temperature_stable=1.0, position_meas_error=99.0)
    assert ok is True
    assert msg == ("Mission feasible with efficient PID; "
                   "ignored unknown inputs: imu_temperature_stable, position_meas_error")
    assert "ignored unknown inputs: imu_temperature_stable, position_meas_error" in caplog.text


@pytest.mark.parametrize('name, value, why', [
    ('battery_voltage', float('nan'), 'NaN'),
    ('wind_speed', float('nan'), 'NaN'),
    ('gps_hdop', float('inf'), 'infinite'),
    ('disturbance', -float('inf'), 'infinite'),
    ('gps_satellites', '12', 'not a number: str'),
])
def test_nan_infinite_and_non_numbers_are_rejected(name, value, why):
    assert preflight(**{name: value}) == (False, f"Pre-flight inputs rejected: invalid: {name} ({why})")


def test_missing_and_invalid_reported_together():
    ok, msg = preflight(drop=('gps_hdop',), battery_voltage=float('nan'), extra=1)
    assert (ok, msg) == (False, "Pre-flight inputs rejected: missing: gps_hdop; invalid: battery_voltage (NaN); "
                                "ignored unknown inputs: extra")


def test_numpy_scalars_and_bools_are_numbers():
    assert preflight(gps_satellites=np.int64(12), battery_voltage=np.float32(12.4), imu_calibrated=True)[0]
    assert preflight(imu_calibrated=np.bool_(True))[0]
    assert not preflight(imu_calibrated=False)[0]       # the contracts require a calibrated IMU


@pytest.mark.parametrize('volts, ok', [(22.0, True), (11.5, False), (19.8, True), (25.2, True),
                                       (19.7, False), (25.3, False)])
def test_dexi_battery_checked_against_its_pack_range(volts, ok):
    """DEXI: 6 cells, 3.3-4.2 V per cell -> 19.8-25.2 V (config/airframes/dexi.yaml, W4-05)."""
    result, msg = preflight(airframe='dexi', drop=('motor_temperature',), battery_voltage=volts)
    assert result is ok, msg
    assert msg.endswith("not checked (unavailable on dexi): motor_temperature; "
                        "battery_voltage checked against dexi's range 19.8-25.2")
    if not ok:
        assert msg.startswith("Mission infeasible - all controller contracts violated (PID: Assumptions violated: "
                              "['battery_voltage']; MPC: Assumptions violated: ['battery_voltage']; Hinf: ")


def test_without_airframe_the_3_cell_range_applies():
    assert preflight(battery_voltage=22.0)[0] is False      # A-F11: the DEXI's pack fails the 3-cell contract
    assert preflight(battery_voltage=11.5)[0] is True


def test_unavailable_input_is_not_checked():
    """motor_temperature: PX4 reports none (dexi.yaml, unavailable). Not required, not checked, never zero."""
    assert preflight(airframe='dexi', drop=('motor_temperature',), battery_voltage=22.0)[0] is True
    # A value given anyway (the node fills in 25 degC today) cannot come from the vehicle: ignored, with a note.
    ok, msg = preflight(airframe='dexi', battery_voltage=22.0, motor_temperature=15.0)
    assert ok is True
    assert "not checked (unavailable on dexi): motor_temperature (value given and ignored: motor_temperature)" in msg
    # Without an airframe it is checked as before: 15 degC is outside 20-80 degC, and absent is missing.
    assert preflight(motor_temperature=15.0)[0] is False
    assert preflight(drop=('motor_temperature',)) == (
        False, "Pre-flight inputs rejected: missing: motor_temperature")


def test_airframe_object_and_x500():
    from contract_uav_core.config.airframe import load_airframe

    x500 = load_airframe('x500_sitl')                       # 4 cells: 14.4-16.8 V
    assert preflight(airframe=x500, drop=('motor_temperature',), battery_voltage=15.2)[0] is True
    assert preflight(airframe=x500, drop=('motor_temperature',), battery_voltage=12.4)[0] is False


def test_unknown_airframe_is_an_error():
    from contract_uav_core.config.airframe import AirframeError

    with pytest.raises(AirframeError, match="unknown airframe 'nope'"):
        preflight(airframe='nope')


def test_check_telemetry_result():
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.contracts.preflight import check_telemetry

    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    check = check_telemetry(dict(GOOD, battery_voltage=22, gps_hdop=None, foo='x'),
                            monitor.pipeline_inputs(), airframe='dexi')
    assert not check.ok
    assert check.missing == ('gps_hdop',) and check.invalid == () and check.unknown == ('foo',)
    assert check.not_checked == ('motor_temperature',) and check.ignored == ('motor_temperature',)
    assert dict(check.assumption_overrides) == {'battery_voltage': (19.8, 25.2)}
    assert check.airframe == 'dexi'
    assert 'motor_temperature' not in check.conditions and check.conditions['battery_voltage'] == 22.0
    with pytest.raises(TypeError):
        check_telemetry([('gps_hdop', 1.0)], monitor.pipeline_inputs())


def test_predictions_leave_out_what_the_airframe_overrides():
    """The actuator guarantees' voltage terms are for a 3-cell pack: no prediction from a 6-cell voltage."""
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor

    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    base = dict(GOOD)
    ok, msg = monitor.verify_mission_feasibility('PID', base)
    assert ok and 'thrust_accuracy' in msg
    dexi = {k: v for k, v in dict(GOOD, battery_voltage=22.0).items() if k != 'motor_temperature'}
    ok, msg = monitor.verify_mission_feasibility(
        'PID', dexi, assumption_overrides={'battery_voltage': (19.8, 25.2)}, not_checked=('motor_temperature',))
    assert ok and 'thrust_accuracy' not in msg and 'tracking_error_max' in msg


def test_node_style_telemetry():
    """The dict the ROS node builds today (controller_node.py, _run_preflight): fixed values."""
    node = {'gps_satellites': 12, 'gps_hdop': 1.0, 'imu_temperature': 25.0, 'imu_calibrated': True,
            'battery_voltage': 12.0, 'motor_temperature': 25.0, 'wind_speed': 0.0, 'disturbance': 0.0,
            'computation_time': 0.01}
    from contract_uav_core.core import AdaptiveDroneController

    assert AdaptiveDroneController(dt=0.02, use_horizon_planner=False).pre_flight_check(node, MISSION)[0]
    # Its fixed 12.0 V is not a DEXI pack voltage: with the airframe, pre-flight fails (W4-13 must send the real one).
    ok, msg = AdaptiveDroneController(dt=0.02, use_horizon_planner=False).pre_flight_check(
        node, MISSION, airframe='dexi')
    assert ok is False and "['battery_voltage']" in msg
