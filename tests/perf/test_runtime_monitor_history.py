"""W4-07 T1 (A-R4): RuntimeMonitor's histories are bounded.

Before W4-07, RuntimeMonitor.history['wind' | 'gps' | 'compute'] were lists that
control_step appended to every step (wind and GPS; compute is never updated
by the core). Nothing reads them. Now each is a deque of the newest
HISTORY_LEN values; the current values (wind_estimate, gps_quality,
compute_delay, get_monitoring_data) are computed as before.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")


def test_histories_keep_the_newest_history_len_values():
    from contract_uav_core.safety.runtime_monitor import RuntimeMonitor

    monitor = RuntimeMonitor()
    n = monitor.HISTORY_LEN + 250
    wind, gps, compute = [], [], []
    for k in range(n):
        velocity = np.array([0.01 * k, -0.02 * k, 0.3])
        monitor.update_wind_estimate(velocity, np.array([0.1, 0.0, 0.0]))
        wind.append(np.linalg.norm((velocity - np.array([0.1, 0.0, 0.0]))[0:2]))
        monitor.update_gps_quality(k % 15, 0.1 * (k % 60))
        gps.append(0.5 * min((k % 15) / 12.0, 1.0) + 0.5 * max(0.0, 1.0 - 0.1 * (k % 60) / 5.0))
        monitor.update_compute_delay(0.02, 0.02 + 1e-5 * k)
        compute.append((0.02 + 1e-5 * k) - 0.02)
        assert monitor.get_monitoring_data() == {'wind_speed': wind[-1], 'gps_quality': gps[-1],
                                                 'compute_delay': compute[-1]}
    for key, expected in (('wind', wind), ('gps', gps), ('compute', compute)):
        history = monitor.history[key]
        assert isinstance(history, deque) and history.maxlen == monitor.HISTORY_LEN
        assert list(history) == expected[-monitor.HISTORY_LEN:], key


def test_control_step_keeps_the_runtime_histories_bounded(patrol):
    """In closed loop, the wind and GPS histories stop at HISTORY_LEN (one value per step)."""
    run = patrol.Patrol()
    monitor = run.controller.runtime_monitor
    for _ in range(monitor.HISTORY_LEN + 100):
        run.step()
    assert len(monitor.history['wind']) == len(monitor.history['gps']) == monitor.HISTORY_LEN
    assert len(monitor.history['compute']) == 0
