"""The core when the planner has no plan: H-inf and the emergency flag (S2, review R-09).

W4-09 (T2) made two inputs give no plan, where the planner used to return one:
a NaN input (before: a FEASIBLE plan with an infinite margin) and wind above
H-inf's 15 m/s limit (before: an H-inf plan checked at wind - 1 m/s). These
tests drive AdaptiveDroneController, with the planner on, through 10 control
steps and check what it then does: it flies H-inf from the first step
(IntegratedPlanner.update returns 'hinf' for an unsafe plan and the switching
policy escalates to it), every step's plan is infeasible (with no plan the
margin is -1, so the planner replans on every step), and the planner's
emergency flag is set from the third consecutive failed replan on.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.02
STEPS = 10
GOOD_CONDITIONS = {
    'gps_satellites': 12.0, 'gps_hdop': 1.0, 'imu_temperature': 25.0, 'imu_calibrated': 1.0,
    'battery_voltage': 12.4, 'motor_temperature': 25.0, 'wind_speed': 1.0, 'disturbance': 0.2,
}


def fly(wind_sensor):
    """10 steps of the core with the planner on; the telemetry of each step."""
    from contract_uav_core.core import AdaptiveDroneController

    ctrl = AdaptiveDroneController(dt=DT, use_horizon_planner=True)
    assert ctrl.horizon_planner is not None
    target = np.array([10.0, 0.0, -5.0])
    ok, msg = ctrl.pre_flight_check(GOOD_CONDITIONS, {'target_position': target, 'waypoints': [target]})
    assert ok, msg
    assert ctrl.start_mission()
    telemetry = []
    for k in range(STEPS):
        raw = {'gps': {'position': np.array([0.01 * k, 0.0, -5.0]), 'velocity': np.array([0.5, 0.0, 0.0]),
                       'valid': True, 'satellites': 12, 'hdop': 1.0},
               'imu': {'attitude': np.zeros(3), 'rates': np.zeros(3), 'valid': True, 'calibrated': True,
                       'temperature': 25.0},
               'battery': {'voltage': 12.4}, 'motors': {'temperature': 30.0},
               'wind': {'speed': wind_sensor, 'direction': 0.0}}
        control, tel = ctrl.control_step(raw)
        assert np.isfinite(control['thrust'])
        telemetry.append(tel)
    return telemetry


def assert_no_plan_response(telemetry):
    assert [t['active_controller'] for t in telemetry] == ['Hinf'] * STEPS
    assert [t['planner']['plan_status'] for t in telemetry] == ['infeasible'] * STEPS
    assert [t['planner']['replanned'] for t in telemetry] == [True] * STEPS
    assert [t['planner']['emergency'] for t in telemetry] == [False, False] + [True] * (STEPS - 2)


def test_nan_input_flies_hinf_with_the_emergency_flag(monkeypatch):
    """A NaN estimate reaching the planner (its tracking error). Before W4-09: MPC on a FEASIBLE plan."""
    from contract_uav_core.planning.integrated_planner import IntegratedPlanner

    convert = IntegratedPlanner._convert_state

    def nan_tracking_error(self, drone_state, target):
        return dict(convert(self, drone_state, target), tracking_error=float('nan'))

    monkeypatch.setattr(IntegratedPlanner, '_convert_state', nan_tracking_error)
    assert_no_plan_response(fly(wind_sensor=1.0))


def test_wind_above_hinf_limit_flies_hinf_with_the_emergency_flag():
    """Wind 16 m/s (sensor). Before W4-09 the replanner accepted H-inf, checked at 15 m/s, with no emergency."""
    telemetry = fly(wind_sensor=16.0)
    assert_no_plan_response(telemetry)
    assert {t['planner']['safety_margin'] for t in telemetry} == {-1.0}


def test_nominal_wind_has_a_plan():
    """The same flight at 1 m/s: a feasible plan, no emergency, and no H-inf.

    The core flies MPC here, not PID: the switching policy's error-trend signal
    switches PID to MPC at t = 0 (K13, W4-06's), not the planner.
    """
    telemetry = fly(wind_sensor=1.0)
    assert {t['planner']['plan_status'] for t in telemetry} == {'feasible'}
    assert not any(t['planner']['emergency'] for t in telemetry)
    assert 'Hinf' not in {t['active_controller'] for t in telemetry}
