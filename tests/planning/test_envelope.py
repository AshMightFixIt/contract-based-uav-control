"""contracts/envelope.py: read-only getters for the wind limits (W4-09, T5).

The getters read config/contracts.json and change no limit. These tests pin the
values and check that they equal the other places that hold the same limits
today: the planner's wind_limits, the framework's MPC limit and the
supervisor's EMERGENCY trigger.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")


@pytest.fixture(scope="module")
def envelope():
    from contract_uav_core.contracts import envelope as module
    return module


def test_wind_envelope_is_the_planner_contract_set(envelope):
    assert envelope.wind_envelope() == {'pid': 3.0, 'mpc': 8.0, 'hinf': 15.0}


def test_emergency_wind_threshold(envelope):
    assert envelope.emergency_wind_threshold() == 15.0
    assert isinstance(envelope.emergency_wind_threshold(), float)


def test_contract_bounds_is_a_copy(envelope):
    first = envelope.contract_bounds()
    first['contracts']['pid']['input_bounds']['wind_speed'][1] = 99.0
    first.clear()
    assert envelope.contract_bounds()['contracts']['pid']['input_bounds']['wind_speed'] == [0.0, 3.0]
    assert envelope.wind_envelope()['pid'] == 3.0


def test_framework_mpc_limit_is_the_emergency_threshold(envelope):
    """The framework set (contracts/library.py) lets MPC run up to the emergency threshold (K03)."""
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor

    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    assert monitor.controller_contracts['MPC'].assumptions['wind_speed'][1] == envelope.emergency_wind_threshold()


def test_supervisor_emergency_trigger_is_the_threshold(envelope):
    """FlightModeSupervisor goes to EMERGENCY just above the threshold, not at it."""
    from contract_uav_core.control.supervisor import FlightMode, FlightModeSupervisor

    threshold = envelope.emergency_wind_threshold()
    state = {'position': np.array([0.0, 0.0, -5.0]), 'velocity': np.zeros(3),
             'attitude': np.zeros(3), 'rates': np.zeros(3)}
    for wind, expected in ((threshold, False), (threshold + 0.01, True)):
        supervisor = FlightModeSupervisor(0.02)
        supervisor.set_waypoints([np.array([10.0, 0.0, -5.0])])
        supervisor.update(state, {'wind_speed': wind, 'disturbance': 0.0}, 'PID', True, 2.0)
        assert (supervisor.get_mode() == FlightMode.EMERGENCY) is expected, wind

