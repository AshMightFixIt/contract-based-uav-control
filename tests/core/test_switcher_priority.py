"""ControllerSwitcher.switch_to(..., priority): the default reproduces today's decisions.

The priority is only recorded (switch_history); its semantics arrive in P1.5.
EXPECTED holds the decisions of the switcher at commit 79d0287 (before the
priority existed) on the scripted sequence SEQ, recorded by running it there:
(return value, active controller, last_switch_time) after each request.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

SEQ = [('PID', 0.0), ('MPC', 0.0), ('Hinf', 4.99), ('Hinf', 5.0), ('Nope', 20.0),
       ('Hinf', 20.0), ('PID', 9.99), ('PID', 10.0), ('MPC', 12.0), ('MPC', 15.0), ('PID', 1e9)]

EXPECTED = [
    (False, 'PID', -5.0),    # already active
    (True, 'MPC', 0.0),      # the initial last_switch_time allows a switch at t = 0
    (False, 'MPC', 0.0),     # cooldown: 4.99 s < 5 s
    (True, 'Hinf', 5.0),     # cooldown over
    (False, 'Hinf', 5.0),    # unknown controller
    (False, 'Hinf', 5.0),    # already active
    (False, 'Hinf', 5.0),    # cooldown
    (True, 'PID', 10.0),
    (False, 'PID', 10.0),    # cooldown
    (True, 'MPC', 15.0),
    (True, 'PID', 1e9),
]

STATE = {'position': np.array([1.0, 0.5, -5.0]), 'velocity': np.array([0.1, 0.0, 0.0]),
         'attitude': np.array([0.1, 0.05, 0.0]), 'rates': np.array([0.05, 0.02, 0.0])}
SETPOINT = {'position': np.array([0.0, 0.0, -5.0]), 'velocity': np.zeros(3), 'yaw': 0.0}


def run(**priority_kw):
    from contract_uav_core.control.switcher import ControllerSwitcher

    switcher = ControllerSwitcher(dt=0.02)
    decisions, integrals = [], []
    for name, t in SEQ:
        switcher.compute_control(STATE, SETPOINT)  # gives the bumpless transfer something to move
        result = switcher.switch_to(name, t, 'scripted', **priority_kw)
        decisions.append((result, switcher.get_active_controller(), switcher.last_switch_time))
        integrals.append(switcher.controllers[switcher.get_active_controller()]
                         .get_integral_state().tobytes())
    return switcher, decisions, integrals


def test_default_priority_reproduces_the_legacy_decisions():
    from contract_uav_core.interfaces import SwitchPriority

    switcher, decisions, integrals = run()
    assert decisions == EXPECTED
    # Passing NORMAL explicitly is the same call, integral transfer included.
    _, decisions_normal, integrals_normal = run(priority=SwitchPriority.NORMAL)
    assert decisions_normal == EXPECTED and integrals_normal == integrals
    # Performed switches only, in order, all recorded as NORMAL.
    history = list(switcher.switch_history)
    assert [(r.time, r.from_controller, r.to_controller) for r in history] == [
        (0.0, 'PID', 'MPC'), (5.0, 'MPC', 'Hinf'), (10.0, 'Hinf', 'PID'),
        (15.0, 'PID', 'MPC'), (1e9, 'MPC', 'PID')]
    assert all(r.priority is SwitchPriority.NORMAL and r.reason == 'scripted' for r in history)


def test_emergency_is_recorded_but_switches_like_normal_until_p1_5():
    """Update this test when P1.5 gives EMERGENCY its own semantics (K14)."""
    from contract_uav_core.interfaces import SwitchPriority

    _, _, integrals = run()
    switcher, decisions, integrals_emergency = run(priority=SwitchPriority.EMERGENCY)
    assert decisions == EXPECTED and integrals_emergency == integrals
    assert {r.priority for r in switcher.switch_history} == {SwitchPriority.EMERGENCY}


def test_priority_is_validated_and_history_is_bounded():
    from contract_uav_core.control.switcher import ControllerSwitcher
    from contract_uav_core.interfaces import SwitchPriority

    switcher = ControllerSwitcher(dt=0.02)
    with pytest.raises(ValueError):
        switcher.switch_to('MPC', 0.0, priority='urgent')
    assert switcher.get_active_controller() == 'PID' and not switcher.switch_history
    assert switcher.switch_to('MPC', 0.0, priority='emergency')  # the enum's value is accepted
    assert switcher.switch_history[-1].priority is SwitchPriority.EMERGENCY

    names = ['PID', 'MPC']
    for k in range(ControllerSwitcher.SWITCH_HISTORY_LEN + 5):
        switcher.switch_to(names[k % 2], 10.0 * (k + 1))
    assert len(switcher.switch_history) == ControllerSwitcher.SWITCH_HISTORY_LEN


def test_benchmark_style_monkeypatch_still_absorbs_the_priority():
    """The scripts replace switch_to with ``lambda *a, **kw: False``; a priority keyword is harmless."""
    from contract_uav_core.control.switcher import ControllerSwitcher
    from contract_uav_core.interfaces import SwitchPriority

    switcher = ControllerSwitcher(dt=0.02)
    switcher.switch_to = lambda *a, **kw: False
    assert switcher.switch_to('Hinf', 99.0, 'x', priority=SwitchPriority.EMERGENCY) is False
