"""G1.5 (gate 1, docs/ROADMAP.md): the planner meets its speed budget with no Pacti at run time.

W4-09 (T3; K10). Budget: one replan within 1 ms. A replan is one call of
AdaptiveRePlanner.find_feasible_plan, the planning work IntegratedPlanner.update
does when it replans (every 0.5 s, or at once when conditions call for it).

Measured on the benchmark missions: the Adaptive run of
benchmarks/benchmark_controllers.py (120 s, wind up to 9.8 m/s) and of
benchmarks/benchmark_sensor_degradation.py (180 s, GPS and IMU faults), each in
a child process with Pacti blocked (sys.modules['pacti'] = None), from a copy of
the scripts. A wrapper records every replan's inputs and its wall-clock time.

Two checks, because CI runners are noisy and shared:
- CPU time, p99 < 1 ms. CPU time (time.process_time) does not count the time
  the process waits for the CPU, so a busy runner does not inflate it. Its
  clock is coarse on some systems (Windows updates it about every 15.6 ms), so
  each recorded replan is replayed REPEATS times with the same inputs (a replan
  is deterministic and keeps no state) and its CPU time is the batch's CPU time
  divided by REPEATS.
- Wall clock, median < 1 ms, over the replans as they ran in the mission
  (time.perf_counter). The median ignores the few replans a busy runner delays.

Before W4-09 a replan made Pacti calls (get_variable_bounds, one LP each) and
took about 120 to 160 ms (K10); without Pacti the planner was off. Measured
for this change (Linux, CPython 3.11): 118 and 122 replans, CPU p99 about
0.02 ms, wall-clock median about 0.05 ms.
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.usefixtures("core_package")

BUDGET_S = 1e-3
REPEATS = 200

MEASURE = r"""
import json, statistics, sys, time
sys.modules['pacti'] = None          # as if Pacti were not installed
import numpy as np
from contract_uav_core.planning.horizon_planner import AdaptiveRePlanner

SCRIPT, REPEATS = sys.argv[1], int(sys.argv[2])
calls = []
original = AdaptiveRePlanner.find_feasible_plan

def timed(self, current_state, target, environment):
    start = time.perf_counter()
    result = original(self, current_state, target, environment)
    calls.append((time.perf_counter() - start, self, dict(current_state), np.array(target, dtype=float),
                  dict(environment)))
    return result

AdaptiveRePlanner.find_feasible_plan = timed
module = __import__(SCRIPT)
module.run_scenario('adaptive')

cpu = []
for _, replanner, state, target, env in calls:
    start = time.process_time()
    for _ in range(REPEATS):
        original(replanner, state, target, env)
    cpu.append((time.process_time() - start) / REPEATS)
wall = [c[0] for c in calls]
cpu.sort()
print(json.dumps({
    'replans': len(calls),
    'cpu_p99': cpu[min(len(cpu) - 1, int(0.99 * len(cpu)))] if cpu else None,
    'cpu_max': max(cpu) if cpu else None,
    'wall_median': statistics.median(wall) if wall else None,
    'wall_max': max(wall) if wall else None,
    'pacti_loaded': [m for m in sys.modules if m.split('.')[0] == 'pacti' and sys.modules[m] is not None],
}))
"""


@pytest.mark.parametrize("script", ["benchmark_controllers", "benchmark_sensor_degradation"])
def test_g1_5_replan_within_1ms_without_pacti(script, repo_copy, run_python):
    """G1.5: every benchmark mission's replans: CPU p99 < 1 ms and wall-clock median < 1 ms, no Pacti."""
    proc = run_python(["-c", MEASURE, script, REPEATS], cwd=repo_copy / "benchmarks", timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"G1.5 {script}: {out}")
    assert out['pacti_loaded'] == []
    assert out['replans'] >= 100, out          # the planner ran: at least one replan per 0.5 s for 50 s
    assert out['cpu_p99'] < BUDGET_S, out
    assert out['wall_median'] < BUDGET_S, out
