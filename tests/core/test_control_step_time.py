"""control_step(raw_sensors, t=None): the legacy clock and the caller-time mode.

See AdaptiveDroneController.control_step in contract_uav_core/core.py. These tests
import the installed core in-process; the ``core_package`` fixture (conftest.py)
checks that it is this checkout's.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.02

GOOD_CONDITIONS = {
    'gps_satellites': 12.0, 'gps_hdop': 1.0, 'imu_temperature': 25.0, 'imu_calibrated': 1.0,
    'battery_voltage': 12.4, 'motor_temperature': 25.0, 'wind_speed': 1.5, 'disturbance': 0.5,
}


def make_controller():
    from contract_uav_core.core import AdaptiveDroneController

    ctrl = AdaptiveDroneController(dt=DT, use_horizon_planner=False)
    target = np.array([10.0, 5.0, -8.0])
    feasible, msg = ctrl.pre_flight_check(GOOD_CONDITIONS, {'target_position': target, 'waypoints': [target]})
    assert feasible, msg
    assert ctrl.start_mission()
    return ctrl


def sensors(k=0):
    return {
        'gps': {'position': np.array([0.01 * k, 0.0, -5.0]), 'velocity': np.array([0.5, 0.0, 0.0]),
                'valid': True, 'satellites': 12, 'hdop': 1.0},
        'imu': {'attitude': np.zeros(3), 'rates': np.zeros(3), 'valid': True,
                'calibrated': True, 'temperature': 25.0},
        'battery': {'voltage': 12.4},
        'motors': {'temperature': 30.0},
    }


@pytest.fixture
def ctrl_and_dts():
    """A ready controller whose EKF records the dt of every predict call."""
    ctrl = make_controller()
    dts = []
    original = ctrl.ekf.predict

    def recording_predict(dt=None):
        dts.append(dt)
        return original(dt)

    ctrl.ekf.predict = recording_predict
    return ctrl, dts


def timing(telemetry):
    return telemetry['timing']


def test_first_call_uses_the_nominal_dt(ctrl_and_dts):
    ctrl, dts = ctrl_and_dts
    _, tel = ctrl.control_step(sensors(), t=100.0)
    assert tel['time'] == 100.0
    assert ctrl.time == 100.0  # not advanced at the end in caller-time mode
    assert dts == [DT]
    assert timing(tel) == {
        'time_source': 'caller', 'step_count': 1, 'overrun_count': 0,
        'nonincreasing_count': 0, 'dt_raw': DT, 'dt_clamped': DT,
    }


def test_normal_step_passes_the_measured_dt(ctrl_and_dts):
    ctrl, dts = ctrl_and_dts
    ctrl.control_step(sensors(0), t=100.0)
    _, tel = ctrl.control_step(sensors(1), t=100.021)
    raw = 100.021 - 100.0
    assert dts == [DT, raw]
    assert tel['time'] == 100.021 and ctrl.time == 100.021
    assert timing(tel)['dt_raw'] == raw and timing(tel)['dt_clamped'] == raw
    assert timing(tel)['step_count'] == 2
    assert timing(tel)['overrun_count'] == 0 and timing(tel)['nonincreasing_count'] == 0


def test_jitter_is_clamped_to_the_band(ctrl_and_dts):
    ctrl, dts = ctrl_and_dts
    ctrl.control_step(sensors(0), t=10.0)
    _, short = ctrl.control_step(sensors(1), t=10.002)   # 0.1 x nominal -> 0.5 x
    _, long_ = ctrl.control_step(sensors(2), t=10.102)   # 5 x nominal -> 2 x
    assert dts == [DT, 0.5 * DT, 2.0 * DT]
    assert timing(short)['dt_raw'] == 10.002 - 10.0
    assert timing(short)['dt_clamped'] == 0.5 * DT
    assert timing(short)['overrun_count'] == 0
    assert timing(long_)['dt_raw'] == 10.102 - 10.002
    assert timing(long_)['dt_clamped'] == 2.0 * DT
    assert timing(long_)['overrun_count'] == 1


def test_overrun_is_counted_without_clamping_inside_the_band(ctrl_and_dts):
    ctrl, dts = ctrl_and_dts
    ctrl.control_step(sensors(0), t=1.0)
    _, tel = ctrl.control_step(sensors(1), t=1.035)       # 1.75 x nominal: overrun, inside [0.5, 2]
    assert timing(tel)['overrun_count'] == 1
    assert dts[-1] == 1.035 - 1.0 == timing(tel)['dt_clamped']
    _, tel = ctrl.control_step(sensors(2), t=1.055)       # 1.0 x nominal: no overrun
    assert timing(tel)['overrun_count'] == 1
    assert timing(tel)['step_count'] == 3


def test_non_increasing_t_is_clamped_and_counted(ctrl_and_dts):
    ctrl, dts = ctrl_and_dts
    ctrl.control_step(sensors(0), t=5.0)
    _, repeat = ctrl.control_step(sensors(1), t=5.0)      # repeated timestamp: raw dt 0
    _, back = ctrl.control_step(sensors(2), t=4.9)        # older timestamp: raw dt < 0
    assert dts == [DT, 0.5 * DT, 0.5 * DT]
    assert timing(repeat)['dt_raw'] == 0.0
    assert timing(repeat)['nonincreasing_count'] == 1
    assert timing(back)['dt_raw'] == 4.9 - 5.0
    assert timing(back)['nonincreasing_count'] == 2
    assert timing(back)['overrun_count'] == 0
    # self.time follows the caller, backwards included (documented)
    assert back['time'] == 4.9 and ctrl.time == 4.9
    _, fwd = ctrl.control_step(sensors(3), t=4.92)        # the next step is measured from 4.9
    assert timing(fwd)['dt_raw'] == 4.92 - 4.9
    assert timing(fwd)['nonincreasing_count'] == 2


@pytest.mark.parametrize("bad", [float('nan'), float('inf'), -float('inf')])
def test_non_finite_t_is_rejected(ctrl_and_dts, bad):
    ctrl, dts = ctrl_and_dts
    with pytest.raises(ValueError, match="finite"):
        ctrl.control_step(sensors(), t=bad)
    assert dts == [] and ctrl.step_count == 0


def test_t_none_reproduces_the_legacy_clock_bit_for_bit(ctrl_and_dts):
    """The legacy path: time accumulates self.dt; the EKF always gets self.dt."""
    ctrl, dts = ctrl_and_dts
    steps = 600
    expected = 0.0
    differs_from_product = 0
    for k in range(steps):
        _, tel = ctrl.control_step(sensors(k))
        assert tel['time'] == expected, (k, tel['time'], expected)
        assert timing(tel) == {
            'time_source': 'accumulated', 'step_count': k + 1, 'overrun_count': 0,
            'nonincreasing_count': 0, 'dt_raw': DT, 'dt_clamped': DT,
        }
        differs_from_product += expected != k * DT
        expected += DT
    assert ctrl.time == expected
    assert len(dts) == steps and all(dt is ctrl.dt for dt in dts)
    # The accumulated clock differs from step * dt in the last bits, so this test
    # would notice a switch to the scripts' own clock.
    assert differs_from_product > 0
