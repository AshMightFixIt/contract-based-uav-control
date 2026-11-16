"""W4-07 T3 (A-R4): AdaptiveDroneController(flight_log_len=...) bounds or turns off the flight log.

The flight log gets one entry per control_step (copies of the state, the
control, the conditions and the planner info: about 2 KB). The default,
flight_log_len=None, keeps today's behaviour: a list of every step, which the
scripts keep (demos/simulation_demo.py saves it). flight_log_len=N keeps the
newest N entries in a deque; 0 keeps none, and the entry is not built.

The option changes nothing else: a controller with a bounded or no flight log
returns the same control and telemetry as one with the default, step by step,
on the same sensor data (the two run in lockstep on the patrol of patrol.py),
and a bounded log holds exactly the default log's newest entries.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

STEPS = 1500        # 30 s of the patrol: calm, moderate and strong wind


def same(a, b):
    """Equal values of the same types, recursively: dicts with keys in the same order, arrays bit for bit."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, float) and np.isnan(a):
        return np.isnan(b)
    return a == b


def lockstep(patrol, steps, **kwargs):
    """Fly the patrol with the default controller; feed a second one, built with kwargs, the same sensors."""
    from contract_uav_core.core import AdaptiveDroneController

    run = patrol.Patrol()
    other = AdaptiveDroneController(dt=patrol.DT, **kwargs)
    mission = {'target_position': patrol.WAYPOINTS[0], 'waypoints': list(patrol.WAYPOINTS)}
    assert other.pre_flight_check(patrol.PREFLIGHT, mission)[0] and other.start_mission()
    original = run.controller.control_step
    mismatches, compared = [], []

    def both(raw):
        control, telemetry = original(raw)
        other_control, other_telemetry = other.control_step(raw)
        compared.append(run.k)
        if not (same(control, other_control) and same(telemetry, other_telemetry)):
            mismatches.append(run.k)
        return control, telemetry

    run.controller.control_step = both
    reload = run.controller.set_waypoint_mission

    def reload_both(waypoints):
        reload(waypoints)
        other.set_waypoint_mission(waypoints)

    run.controller.set_waypoint_mission = reload_both
    for _ in range(steps):
        run.step()
    assert compared == list(range(steps))
    return run.controller, other, mismatches


def test_default_keeps_every_step_in_a_list(patrol):
    run = patrol.Patrol()
    assert type(run.controller.flight_log) is list
    for _ in range(300):
        run.step()
    log = run.controller.flight_log
    assert type(log) is list and len(log) == 300
    assert list(log[-1]) == ['time', 'state', 'control', 'controller', 'flight_mode', 'cbf_intervened',
                             'system_conditions', 'estimation_ok', 'planner_info']
    assert np.isclose(log[-1]['time'], 299 * patrol.DT)


@pytest.mark.parametrize('length', [1, 250])
def test_bounded_log_holds_the_newest_entries_and_changes_nothing_else(patrol, length):
    default, bounded, mismatches = lockstep(patrol, STEPS, flight_log_len=length)
    assert mismatches == []
    assert isinstance(bounded.flight_log, deque) and bounded.flight_log.maxlen == length
    assert len(default.flight_log) == STEPS and len(bounded.flight_log) == length
    assert all(same(a, b) for a, b in zip(bounded.flight_log, default.flight_log[-length:]))


def test_zero_turns_the_log_off_and_changes_nothing_else(patrol):
    _, off, mismatches = lockstep(patrol, STEPS, flight_log_len=0)
    assert mismatches == []
    assert len(off.flight_log) == 0


class NoCopy(dict):
    def copy(self):
        raise AssertionError("copied for a flight-log entry")


@pytest.mark.parametrize('length', [None, 5, 0])
def test_zero_does_not_build_the_entry(patrol, length):
    """With flight_log_len=0, _record_step skips the entry and its copies."""
    from contract_uav_core.core import AdaptiveDroneController

    ctrl = AdaptiveDroneController(dt=patrol.DT, flight_log_len=length)
    run = patrol.Patrol()
    args = (run.sensors(0.0), ctrl.ekf.get_state(), {'position': np.zeros(3), 'velocity': np.zeros(3)},
            {'thrust': 0.5, 'torques': np.zeros(3)}, False, True, NoCopy(replanned=False))
    if length == 0:
        ctrl._record_step(*args)
    else:
        with pytest.raises(AssertionError, match='flight-log entry'):
            ctrl._record_step(*args)


@pytest.mark.parametrize('value', [np.int64(7), 7])
def test_integer_lengths(value):
    from contract_uav_core.core import AdaptiveDroneController

    log = AdaptiveDroneController(use_horizon_planner=False, flight_log_len=value).flight_log
    assert isinstance(log, deque) and log.maxlen == 7 and type(log.maxlen) is int


@pytest.mark.parametrize('value', [-1, 2.5, True, '10', [3]])
def test_invalid_lengths_are_rejected(value):
    from contract_uav_core.core import AdaptiveDroneController

    with pytest.raises(ValueError, match='flight_log_len'):
        AdaptiveDroneController(use_horizon_planner=False, flight_log_len=value)
