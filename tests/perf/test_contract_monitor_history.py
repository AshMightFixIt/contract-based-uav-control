"""W4-07 T1, T2 (A-R4): the contract monitor's bounded history and its O(1) status.

Before W4-07, HierarchicalContractMonitor.history was a list that grew by 5
ContractMetrics per control_step (the sensors twice, the estimator, the active
controller, the actuators), about 1.4 KB each, and get_contract_status scanned
all of it, 4 times per step: the step time grew with the flight time (A-R4).
Now history is a deque of the newest HISTORY_LEN metrics, and
get_contract_status reads the newest metrics per component from a dict that
monitor_runtime updates.

The fixture ``legacy`` holds get_contract_status and export_metrics as they
were at 291fa13, verbatim, run on a recorder that keeps the whole, unbounded
history. The new status must equal the legacy one at every step of a recorded
closed-loop run (the patrol, patrol.py), in which the bounded history
overflows and every status value occurs.
"""
from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

COMPONENTS = ('sensors', 'estimator', 'controller_PID', 'controller_MPC', 'controller_Hinf',
              'actuators', 'no_such_component')
PATROL_STEPS = 3000      # one minute: one whole cycle of the patrol's wind and GPS outage


@pytest.fixture(scope="module")
def legacy():
    """get_contract_status and export_metrics at 291fa13, verbatim (indented into this fixture)."""
    import logging
    from contract_uav_core.contracts.spec import ContractStatus
    logger = logging.getLogger(__name__)

    def get_contract_status(self, component: str) -> ContractStatus:
        """Get current contract status for a component"""
        if not self.history:
            return ContractStatus.UNKNOWN

        # Get latest metrics for this component
        recent = [m for m in self.history if m.component_name == component]
        if not recent:
            return ContractStatus.UNKNOWN

        latest = recent[-1]

        if latest.assumptions_met and latest.guarantees_met:
            return ContractStatus.SATISFIED
        elif not latest.assumptions_met and latest.guarantees_met:
            return ContractStatus.ASSUMPTIONS_VIOLATED
        elif latest.assumptions_met and not latest.guarantees_met:
            return ContractStatus.GUARANTEES_VIOLATED
        else:
            return ContractStatus.BOTH_VIOLATED

    def export_metrics(self, filename: str):
        """Export contract monitoring history for analysis"""
        import json

        data = {
            "history": [
                {
                    "timestamp": m.timestamp,
                    "component": m.component_name,
                    "assumptions_met": m.assumptions_met,
                    "guarantees_met": m.guarantees_met,
                    "margin": m.margin,
                    "values": m.assumption_values,
                    "predicted_bounds": m.predicted_bounds
                }
                for m in self.history
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self.history)} metrics to {filename}")

    return SimpleNamespace(get_contract_status=get_contract_status, export_metrics=export_metrics)


def record_history(monitor):
    """Wrap monitor.monitor_runtime; return the list of every metrics it appended to the history."""
    full, original = [], monitor.monitor_runtime

    def recording(component, values, timestamp):
        metrics = original(component, values, timestamp)
        if monitor.history and monitor.history[-1] is metrics:
            full.append(metrics)
        return metrics

    monitor.monitor_runtime = recording
    return full


def new_monitor():
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor

    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    return monitor


SAMPLES = {
    'sensors': {'gps_satellites': 12.0, 'gps_hdop': 0.8, 'imu_temperature': 25.0, 'imu_calibrated': 1.0},
    'controller_PID': {'position_error': 0.5, 'velocity_error': 0.1, 'wind_speed': 1.0, 'disturbance': 0.2,
                       'control_effort': 0.5},
    'actuators': {'control_effort': 0.5, 'battery_voltage': 12.4, 'motor_temperature': 30.0},
}


def test_history_is_a_deque_of_the_newest_history_len_metrics():
    """T1: history keeps exactly the newest HISTORY_LEN metrics, oldest first."""
    monitor = new_monitor()
    full = record_history(monitor)
    n = monitor.HISTORY_LEN + 123
    names = list(SAMPLES)
    for k in range(n):
        name = names[k % len(names)]
        metrics = monitor.monitor_runtime(name, SAMPLES[name], 0.02 * k)
        assert monitor.history[-1] is metrics        # tests/estimation relies on this
    assert isinstance(monitor.history, deque) and monitor.history.maxlen == monitor.HISTORY_LEN
    assert len(full) == n and len(monitor.history) == monitor.HISTORY_LEN
    assert all(a is b for a, b in zip(monitor.history, full[-monitor.HISTORY_LEN:]))


def test_history_len_holds_a_whole_simulation_demo_run(patrol):
    """T1: 1100 steps (demos/simulation_demo.py's run) leave the whole history in the monitor."""
    run = patrol.Patrol()
    full = record_history(run.controller.contract_monitor)
    for _ in range(1100):
        run.step()
    assert len(full) == 5 * 1100 <= run.controller.contract_monitor.HISTORY_LEN
    assert list(run.controller.contract_monitor.history) == full


def test_an_unknown_component_is_not_recorded(legacy):
    monitor = new_monitor()
    metrics = monitor.monitor_runtime('no_such_component', {'x': 1.0}, 0.0)
    assert not metrics.assumptions_met and not metrics.guarantees_met
    assert len(monitor.history) == 0
    assert monitor.get_contract_status('no_such_component') == legacy.get_contract_status(monitor, 'no_such_component')


def test_status_equals_the_legacy_scan_at_every_step_of_a_patrol(patrol, legacy):
    """T2: at every step, get_contract_status and the telemetry's contract_status equal the old scan."""
    from contract_uav_core.contracts.spec import ContractStatus

    run = patrol.Patrol()
    monitor = run.controller.contract_monitor
    recorder = SimpleNamespace(history=record_history(monitor))
    seen = set()
    for k in range(PATROL_STEPS):
        _, telemetry, _ = run.step()
        expected = {c: legacy.get_contract_status(recorder, c) for c in COMPONENTS}
        assert {c: monitor.get_contract_status(c) for c in COMPONENTS} == expected, k
        reported = telemetry['contract_status']
        controller = f"controller_{telemetry['active_controller']}"
        assert reported == {'sensors': expected['sensors'], 'estimator': expected['estimator'],
                            'controller': expected[controller], 'actuators': expected['actuators']}, k
        seen.update(expected.values())
    assert len(recorder.history) > monitor.HISTORY_LEN           # the bounded history overflowed
    assert seen == set(ContractStatus)                          # and every status value occurred


def test_status_of_a_component_that_left_the_bounded_history(legacy):
    """T2: the newest metrics of a component count even after they left the bounded history."""
    from contract_uav_core.contracts.spec import ContractStatus

    monitor = new_monitor()
    recorder = SimpleNamespace(history=record_history(monitor))
    monitor.monitor_runtime('controller_PID', SAMPLES['controller_PID'], 0.0)
    for k in range(monitor.HISTORY_LEN):
        monitor.monitor_runtime('actuators', SAMPLES['actuators'], 0.02 * (k + 1))
    assert all(m.component_name == 'actuators' for m in monitor.history)
    # A scan of the bounded history would no longer find it ...
    assert legacy.get_contract_status(monitor, 'controller_PID') == ContractStatus.UNKNOWN
    # ... but the status is the one of the unbounded history, as before W4-07
    expected = legacy.get_contract_status(recorder, 'controller_PID')
    assert expected != ContractStatus.UNKNOWN
    assert monitor.get_contract_status('controller_PID') == expected


def test_status_does_not_read_the_history():
    """T2: O(1): get_contract_status never iterates the history."""
    class NoScan(deque):
        def __iter__(self):
            raise AssertionError("get_contract_status iterated the history")

    monitor = new_monitor()
    for name, values in SAMPLES.items():
        monitor.monitor_runtime(name, values, 0.0)
    monitor.history = NoScan(monitor.history, maxlen=monitor.HISTORY_LEN)
    for c in COMPONENTS:
        monitor.get_contract_status(c)


@pytest.mark.parametrize('extra', [0, 77], ids=['fits', 'overflowed'])
def test_export_metrics_writes_the_bounded_history_as_before(legacy, tmp_path, extra):
    """export_metrics writes what the old one wrote for the newest HISTORY_LEN metrics."""
    monitor = new_monitor()
    full = record_history(monitor)
    n = (monitor.HISTORY_LEN if extra else 1234) + extra
    names = list(SAMPLES)
    for k in range(n):
        name = names[k % len(names)]
        monitor.monitor_runtime(name, SAMPLES[name], 0.02 * k)
    monitor.export_metrics(str(tmp_path / 'new.json'))
    legacy.export_metrics(SimpleNamespace(history=full[-monitor.HISTORY_LEN:]), str(tmp_path / 'old.json'))
    assert (tmp_path / 'new.json').read_bytes() == (tmp_path / 'old.json').read_bytes()
    exported = json.loads((tmp_path / 'new.json').read_text())['history']
    assert len(exported) == min(n, monitor.HISTORY_LEN)
    assert np.isclose(exported[-1]['timestamp'], 0.02 * (n - 1))
