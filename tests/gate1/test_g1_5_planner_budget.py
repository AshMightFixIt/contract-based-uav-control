"""G1.5 (gate 1, docs/ROADMAP.md): the planner meets its speed budget with no Pacti at run time.

W4-09 (T3; K10). Budget: one replan within 1 ms. A replan is one call of
AdaptiveRePlanner.find_feasible_plan, the planning work IntegratedPlanner.update
does when it replans (every 0.5 s, or at once when conditions call for it).

Measured on the benchmark missions: the Adaptive run of
benchmarks/benchmark_controllers.py (120 s, wind up to 9.8 m/s) and of
benchmarks/benchmark_sensor_degradation.py (180 s, GPS and IMU faults), each in
a child process with Pacti blocked (sys.modules['pacti'] = None), from a copy of
the scripts. A wrapper records every replan's inputs and its wall-clock time.
Each test needs at least MIN_REPLANS samples for a meaningful p99. A mission
that yields fewer is topped up, to exactly MIN_REPLANS, with replans from the
other mission, recorded the same way in the same process (W4-08: the sensor
mission now ends at 30 s, through the benchmark's "mission complete" break,
K07, and yields 68 replans).

Two checks, because CI runners are noisy and shared:
- CPU time, p99 < 1 ms. CPU time (time.process_time) does not count the time
  the process waits for the CPU, so a busy runner does not inflate it. Its
  clock is coarse on some systems (Windows updates it about every 15.6 ms), so
  each recorded replan is replayed REPEATS times with the same inputs (a replan
  is deterministic and keeps no state) and its CPU time is the batch's CPU time
  divided by REPEATS.
- Wall clock, median < 1 ms, over the replans as they ran in the mission
  (time.perf_counter). The median ignores the few replans a busy runner delays.

The missions never take the slowest path, so a third test times it directly
(review R-09, N3): every controller passes its assumption check but none has a
safe margin (tracking error 4.99 m, wind 2.9 m/s), so plan() computes all 17
tries (PID, MPC, H-inf at the full horizon, MPC and H-inf at each shorter one)
before the emergency plan; the planner's warnings and its error are formatted
into an in-memory log, as a node that logs them would pay for. Same budget and
the same two checks, except that its CPU time is the calling thread's
(time.thread_time): right after numpy is imported, its OpenBLAS worker threads
spin for a few tenths of a second, and process_time counts them (observed: 0.7
instead of 0.17 ms per replan in the first two batches; with
OPENBLAS_NUM_THREADS=1 the two clocks agree). The mission replays run after the
mission, when those threads are idle.

Before W4-09 a replan made Pacti calls (get_variable_bounds, one LP each) and
took about 120 to 160 ms (K10); without Pacti the planner was off. Measured
for this change (Linux, CPython 3.11): 118 and 122 replans, CPU p99 about
0.02 ms, wall-clock median about 0.05 ms; the slowest path about 0.12 ms
without logging and 0.2 to 0.35 ms with it.
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.usefixtures("core_package")

BUDGET_S = 1e-3
REPEATS = 200
MIN_REPLANS = 100
OTHER_MISSION = {'benchmark_controllers': 'benchmark_sensor_degradation',
                 'benchmark_sensor_degradation': 'benchmark_controllers'}

MEASURE = r"""
import json, statistics, sys, time
sys.modules['pacti'] = None          # as if Pacti were not installed
import numpy as np
from contract_uav_core.planning.horizon_planner import AdaptiveRePlanner

SCRIPT, REPEATS, MIN_REPLANS, OTHER = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
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
mission_replans = len(calls)
if len(calls) < MIN_REPLANS:             # top up from the other mission, to exactly MIN_REPLANS
    __import__(OTHER).run_scenario('adaptive')
    del calls[MIN_REPLANS:]

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
    'mission_replans': mission_replans,
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
    proc = run_python(["-c", MEASURE, script, REPEATS, MIN_REPLANS, OTHER_MISSION[script]],
                      cwd=repo_copy / "benchmarks", timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"G1.5 {script}: {out}")
    assert out['pacti_loaded'] == []
    assert out['replans'] >= MIN_REPLANS, out  # enough samples (topped up from the other mission if needed)
    assert out['mission_replans'] > 0, out     # and the planner ran in this mission
    assert out['cpu_p99'] < BUDGET_S, out
    assert out['wall_median'] < BUDGET_S, out


SLOWEST_PATH = r"""
import io, json, logging, statistics, sys, time
sys.modules['pacti'] = None          # as if Pacti were not installed
logging.basicConfig(level=logging.WARNING, stream=io.StringIO())
import numpy as np
from contract_uav_core.planning.horizon_planner import AdaptiveRePlanner, HorizonPlanner

BATCHES, REPEATS = int(sys.argv[1]), int(sys.argv[2])
state = {'tracking_error': 4.99, 'position_est_error': 0.5, 'velocity_est_error': 0.1}
target, env = np.zeros(3), {'wind_speed': 2.9}
replanner = AdaptiveRePlanner(HorizonPlanner())

tries, original = [], HorizonPlanner._try_plan
def counted(self, *args):
    plan = original(self, *args)
    tries.append((plan.recommended_controller, plan.horizon_length, len(plan.steps)))
    return plan
HorizonPlanner._try_plan = counted
plan, mods = replanner.find_feasible_plan(state, target, env)
HorizonPlanner._try_plan = original

cpu = []
for _ in range(BATCHES):
    start = time.thread_time()
    for _ in range(REPEATS):
        replanner.find_feasible_plan(state, target, env)
    cpu.append((time.thread_time() - start) / REPEATS)
wall = []
for _ in range(REPEATS):
    start = time.perf_counter()
    replanner.find_feasible_plan(state, target, env)
    wall.append(time.perf_counter() - start)
print(json.dumps({
    'status': plan.status.value, 'controller': plan.recommended_controller, 'mods': mods,
    'tries': tries, 'cpu_max': max(cpu), 'wall_median': statistics.median(wall), 'wall_max': max(wall),
    'pacti_loaded': [m for m in sys.modules if m.split('.')[0] == 'pacti' and sys.modules[m] is not None],
}))
"""


def test_g1_5_slowest_replan_within_1ms_without_pacti(run_python, tmp_path):
    """G1.5, slowest path: 17 full tries, then the emergency plan; CPU p99 and wall median < 1 ms, no Pacti."""
    proc = run_python(["-c", SLOWEST_PATH, 10, REPEATS], cwd=tmp_path, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"G1.5 slowest path: { {k: v for k, v in out.items() if k != 'tries'} }")
    assert out['pacti_loaded'] == []
    assert (out['status'], out['controller'], out['mods']) == ('infeasible', 'hinf', {'emergency_mode': True})
    # Every try computed its steps: 3 controllers at horizon 10, then MPC and H-inf at 9 ... 3.
    expected = [(c, 10, 10) for c in ('pid', 'mpc', 'hinf')]
    expected += [(c, h, h) for h in range(9, 2, -1) for c in ('mpc', 'hinf')]
    assert [tuple(t) for t in out['tries']] == expected
    assert out['cpu_max'] < BUDGET_S, out       # the p99 of 10 batch averages is their maximum
    assert out['wall_median'] < BUDGET_S, out
