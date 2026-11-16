"""G1.4 (gate 1, docs/ROADMAP.md): a 30-minute run stays in budget.

W4-07 (A-R4). The whole core, AdaptiveDroneController with the horizon planner
and flight_log_len=3000 (the newest 60 s, as a long run or the ROS node should
use it), flies the patrol of tests/perf/patrol.py: the reference plant
sim/quadrotor.py with the x500_sitl airframe, the benchmark's four waypoints
again and again, every minute the benchmark's wind schedule (PID, MPC and
H-infinity all run) and a 2 s GPS outage. 30 simulated minutes are 90 000
steps at 50 Hz. Each measurement runs in a fresh process (run_python).

Step time: the CPU time of each control_step on the calling thread's clock
(time.thread_time: neither other processes nor NumPy's BLAS threads count,
W4-09; for the wall clock, see below). Checks: p99 < 2 ms, and the median does
not grow. The medians compared are those of 5-minute windows, adjusted for the
controller mix (each tier's median weighted by its share of the run; an MPC
step costs about 0.30 ms, a PID or H-infinity step about 0.22 ms, and the mix
of a window depends on the path flown):
- the last 5 minutes within 10 % of minutes 5 to 10;
- the last 5 minutes within 20 % of the first 5.
The first 5 minutes are a warm-up, as for the memory: they are 2 to 6 % faster
than every later window, because the bounded buffers (the contract history,
the flight log) fill in the first 40 to 60 s, and the heap they churn then
settles (with buffers of 20 to 100 entries the step is flat from the start;
over 60 minutes it stays flat after the first window). Measured with the
first 5 minutes as the reference, the plain median ratio reached 1.10 in 11
runs and the mix-adjusted one 1.08, too close to a 10 % bound.

The wall-clock fallback (Windows): where the thread clock is coarse (Windows
updates it every 15.6 ms), patrol.py times each step with the wall clock
(time.perf_counter) and reports clock 'perf_counter'. The wall clock charges a
step that the OS preempts with all the time the process waits, so under CPU
load its tail measures the machine, not the core: with the fallback forced on
Linux and 7 busy processes on 4 cores, 3-minute runs gave p95 4.4 ms and p99
8.5 ms, while the medians stayed at 0.31 ms and within 8 % of each other
(review R-07, S2). So on the wall clock the p99 check is skipped, and the
median step is held to the same bound instead (2 ms; 4 ms in the 3-minute
run); the median-growth checks are the same on both clocks.

Memory: tracemalloc traces the whole run (a separate run, as tracing slows each
step); the memory traced at minute 30 minus that at minute 5 < 1 MB, and the
growth of sys.getallocatedblocks() (the number of live objects) over the same
minutes < 10 000 blocks. The block count catches a leak of a small object per
step, which a byte bound alone can miss (review R-07, S1).

The 30-minute run is marked ``perf``, so ``make test`` leaves it out; run it
with ``make test-perf`` (``make test-all`` includes it). A 3-minute run, in
``make test``, checks the same with 1-minute windows: p99 < 4 ms (on the wall
clock: the median < 4 ms), the last minute within 25 % of the first, and from
minute 1 to minute 3 (every bounded buffer is full after 60 s) under 64 kB and
under 1000 blocks of growth.
Measured at this change, 3 minutes, 3.11 and 3.10: -7.7 to +1.5 kB and -143
to -37 blocks; with one runtime history unbounded again (a leak of one float
per step, R-07's RM6) +190 kB and +5900 blocks. 30 minutes: +1495 blocks (the
switch_history filling); one object per step would add about 75 000.

Before W4-07 (291fa13) the bookkeeping grew without limit (A-R4). Measured on
this patrol at 291fa13 (its default, unbounded flight log): over 3 minutes the
median step rose from 1.65 to 3.19 to 4.98 ms, minute by minute (p99 9.1 ms),
and the traced memory from 28.5 to 56.8 to 85.1 MB; over 10 minutes the
median was 7.4 ms in the first 5 minutes and 20.4 ms in the next 5 (p99
85 ms). Measured for this change
(Linux, CPython 3.11 and 3.10, an otherwise idle machine), 30 minutes: p99
0.47 to 0.73 ms, window medians about 0.25 to 0.30 ms, last 5 minutes / minutes
5 to 10 between 1.01 and 1.04 (mix-adjusted), growth from minute 5 to 30
about 0.09 MB (the switcher's switch_history, bounded at 1000 switches, still
filling at about 10 switches per minute).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("core_package")

PATROL = Path(__file__).resolve().parent.parent / "perf" / "patrol.py"
FLIGHT_LOG_LEN = 3000

RUNS = [
    pytest.param(dict(minutes=30, window=5, p99_ms=2.0, after_warmup=0.10, from_start=0.20,
                      growth_bytes=1_000_000, growth_blocks=10_000),
                 marks=pytest.mark.perf, id='30min'),
    pytest.param(dict(minutes=3, window=1, p99_ms=4.0, after_warmup=None, from_start=0.25,
                      growth_bytes=64_000, growth_blocks=1_000),
                 id='3min'),
]


def patrol(run_python, tmp_path, *args):
    proc = run_python([PATROL, *map(str, args), '--flight-log-len', FLIGHT_LOG_LEN], cwd=tmp_path, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize('run', RUNS)
def test_g1_4_step_time_stays_in_budget(run, run_python, tmp_path):
    """G1.4: p99 of the step's CPU time under budget (wall clock: the median), and no median growth."""
    out = patrol(run_python, tmp_path, '--minutes', run['minutes'], '--measure', 'time', '--window', run['window'])
    print(f"G1.4 step time, {run['minutes']} min: {json.dumps(out)}")
    assert out['steps'] == run['minutes'] * 3000
    # the run is the intended one: all three tiers, the GPS hover, the planner's replans (every 0.5 s)
    assert set(out['controllers']) == {'PID', 'MPC', 'Hinf'}, out['controllers']
    assert out['modes'].get('hover', 0) > 0 and out['replans'] >= 0.9 * run['minutes'] * 60 / 0.5, out
    if out['clock'] == 'thread_time':
        assert out['p99_ms'] < run['p99_ms'], out
    else:                   # the wall clock: its tail is the machine's load (see the module docstring)
        assert out['clock'] == 'perf_counter' and out['median_ms'] < run['p99_ms'], out
    medians = [w['adjusted_ms'] for w in out['windows']]
    if run['after_warmup'] is not None:
        assert abs(medians[-1] / medians[1] - 1.0) <= run['after_warmup'], medians
    assert abs(medians[-1] / medians[0] - 1.0) <= run['from_start'], medians


@pytest.mark.parametrize('run', RUNS)
def test_g1_4_memory_stays_flat(run, run_python, tmp_path):
    """G1.4: neither the memory traced nor the number of live blocks grows after the warm-up."""
    out = patrol(run_python, tmp_path, '--minutes', run['minutes'], '--measure', 'memory', '--warmup', run['window'])
    print(f"G1.4 memory, {run['minutes']} min: {json.dumps(out)}")
    assert out['lengths']['flight_log'] == FLIGHT_LOG_LEN, out
    assert out['growth_bytes'] < run['growth_bytes'], out
    assert out['growth_blocks'] < run['growth_blocks'], out
