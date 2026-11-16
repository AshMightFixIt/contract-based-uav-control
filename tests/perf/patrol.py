"""The patrol: a long closed-loop flight of the whole core, for the W4-07 tests (A-R4, G1.4).

AdaptiveDroneController (dt = 0.02 s, horizon planner on) flies the reference
plant sim/quadrotor.py with the x500_sitl airframe, from 5 m up (z = -5, NED),
around the four waypoints of benchmarks/benchmark_controllers.py, again and
again: whenever the supervisor hovers with GPS valid (the patrol is complete,
or GPS came back after an outage) the mission is loaded again with
set_waypoint_mission. Every minute the wind runs through benchmark_controllers.py's
schedule, compressed from 120 s into 60 s (calm, moderate, strong, easing,
calm: PID, MPC and H-infinity all run), and GPS is lost for 2 s (the
estimator's no-GPS path and the supervisor's HOVER run). The cycle is one
minute long so that windows of whole minutes, which the step-time checks
compare, hold the same mix of conditions. The sensors carry the benchmark's noise, from a seeded
generator, so a run is deterministic. The core builds its controllers without
the airframe (hover thrust 0.5, docs/KNOWN_ISSUES.md), so on the x500 (hover
0.6) it flies about 2 m below its setpoints; that does not matter here.

Imported by the tests (see Patrol), or run as a script, in a fresh process, to
measure (G1.4):

    python patrol.py --minutes 30 --measure time   [--window 5]
    python patrol.py --minutes 30 --measure memory [--warmup 5]

Both take --flight-log-len N (AdaptiveDroneController(flight_log_len=N); the
default, -1, leaves the constructor's default) and print one JSON line.
- time: the CPU time of every control_step, from the calling thread's clock
  (time.thread_time), so neither other processes nor NumPy's BLAS threads count
  (W4-09). Where that clock is coarse (Windows updates it every 15.6 ms) the
  wall clock (time.perf_counter) is used instead, and 'clock' says so. Reports
  p99 and max, and for each window of --window minutes the median: plain,
  and adjusted for the controller mix (adjusted_ms: each tier's median in the
  window, weighted by the tier's share of the whole run). The mix matters: an
  MPC step takes about 0.30 ms, a PID or H-infinity step about 0.22 ms, and
  the share of each tier in a window varies with the path flown.
- memory: tracemalloc traces the whole run; the growth is the memory traced
  at the end minus that traced after --warmup minutes (whole minutes; also
  reported for each later minute). Tracing from the first step, not from the
  warm-up, matters: a bounded buffer that is full when tracing starts would
  count as growing while its untraced entries are replaced by traced ones.
  Also the growth of sys.getallocatedblocks() over the same minutes: the
  number of live objects, which a leak of one small object per step raises by
  one per step whatever its size. The plant and the patrol's own bookkeeping
  are constant in size.
The core's warnings (each controller switch and mode change) are formatted and
dropped, so a run pays for its logging but keeps none of it.
"""
import argparse
import json
import logging
import math
import sys
import time
import tracemalloc

import numpy as np

from contract_uav_core.config.airframe import load_airframe
from contract_uav_core.core import AdaptiveDroneController
from contract_uav_core.interfaces import AttitudeThrustCommand
from contract_uav_core.sim.quadrotor import QuadrotorParams, QuadrotorPlant

DT = 0.02
STEPS_PER_MINUTE = int(round(60 / DT))
START = (0.0, 0.0, -5.0)
WAYPOINTS = [np.array([10.0, 0.0, -5.0]), np.array([10.0, 10.0, -8.0]),
             np.array([0.0, 10.0, -6.0]), np.array([0.0, 0.0, -5.0])]
PREFLIGHT = {'gps_satellites': 12.0, 'gps_hdop': 0.8, 'imu_temperature': 25.0, 'imu_calibrated': 1.0,
             'battery_voltage': 12.4, 'motor_temperature': 30.0, 'wind_speed': 0.5, 'disturbance': 0.2}
CYCLE = 60.0                                   # s: the wind schedule and the GPS outage repeat
# (start, end, wind NED m/s) in each cycle: benchmark_controllers.py's WIND_SCHEDULE, its times halved
WIND = [(0.0, 12.5, (0.5, 0.0, 0.0)), (12.5, 22.5, (4.5, 2.0, 0.0)), (22.5, 30.0, (9.0, 4.0, 0.0)),
        (30.0, 40.0, (5.0, 2.0, 0.0)), (40.0, 60.0, (1.0, 0.5, 0.0))]
GPS_OUTAGE = (55.0, 57.0)                      # s in each cycle: GPS invalid
SEED = 42
FINE_CLOCK = 1e-5                              # s: a clock that ticks coarser than this is not used per step


def wind_at(t):
    """The wind (NED, m/s) at time t."""
    phase = t % CYCLE
    for start, end, wind in WIND:
        if start <= phase < end:
            return np.array(wind)
    return np.zeros(3)


def gps_valid_at(t):
    return not (GPS_OUTAGE[0] <= t % CYCLE < GPS_OUTAGE[1])


class Patrol:
    """The core and the plant in closed loop; step() runs one control_step and one plant step."""

    def __init__(self, **controller_kwargs):
        self.controller = AdaptiveDroneController(dt=DT, **controller_kwargs)
        mission = {'target_position': WAYPOINTS[0], 'waypoints': list(WAYPOINTS)}
        feasible, msg = self.controller.pre_flight_check(PREFLIGHT, mission)
        if not feasible or not self.controller.start_mission():
            raise RuntimeError(f"patrol: pre-flight failed: {msg}")
        self.plant = QuadrotorPlant(QuadrotorParams.from_airframe(load_airframe('x500_sitl')), position=START)
        self.state = self.plant.state
        self.rng = np.random.default_rng(SEED)
        self.k = 0
        self.reloads = 0

    def sensors(self, t):
        s, rng, gps_ok = self.state, self.rng, gps_valid_at(t)
        wind = wind_at(t)
        return {
            'gps': {'position': s.position + rng.normal(0.0, 0.01, 3),
                    'velocity': s.velocity + rng.normal(0.0, 0.01, 3),
                    'valid': gps_ok, 'satellites': 12 if gps_ok else 2, 'hdop': 0.8 if gps_ok else 10.0},
            'imu': {'attitude': s.attitude + rng.normal(0.0, 0.001, 3),
                    'rates': s.rates + rng.normal(0.0, 0.001, 3),
                    'valid': True, 'calibrated': True, 'temperature': 25.0},
            'battery': {'voltage': 12.4},
            'motors': {'temperature': 30.0},
            'wind': {'speed': float(np.linalg.norm(wind[0:2])), 'direction': math.atan2(wind[1], wind[0])},
        }

    def step(self, clock=None):
        """One step. Returns (control, telemetry, seconds of clock() in control_step, or None)."""
        t = self.k * DT
        raw = self.sensors(t)
        if clock is None:
            control, telemetry = self.controller.control_step(raw)
            elapsed = None
        else:
            start = clock()
            control, telemetry = self.controller.control_step(raw)
            elapsed = clock() - start
        self.state = self.plant.step(AttitudeThrustCommand.from_legacy(control), DT, wind=wind_at(t))
        if telemetry.get('flight_mode') == 'hover' and raw['gps']['valid']:
            self.controller.set_waypoint_mission(WAYPOINTS)
            self.reloads += 1
        self.k += 1
        return control, telemetry, elapsed


def clock_tick(clock, samples=20):
    """The smallest step seen between two different readings of clock, in seconds."""
    ticks = []
    for _ in range(samples):
        t0 = t1 = clock()
        while t1 == t0:
            t1 = clock()
        ticks.append(t1 - t0)
    return min(ticks)


def mix_adjusted_median(ms, tiers, share):
    """The median step time per tier, weighted by each tier's share of the whole run (share)."""
    medians = {tier: float(np.median(ms[tiers == tier])) for tier in share if np.any(tiers == tier)}
    return sum(share[t] * medians[t] for t in medians) / sum(share[t] for t in medians), medians


def measure_time(minutes, window, controller_kwargs):
    tick = clock_tick(time.thread_time)
    clock, name = (time.thread_time, 'thread_time') if tick <= FINE_CLOCK else (time.perf_counter, 'perf_counter')
    patrol = Patrol(**controller_kwargs)
    steps = int(round(minutes * STEPS_PER_MINUTE))
    per_step = np.empty(steps)
    tiers = np.empty(steps, dtype=object)
    modes, replans = {}, 0
    for k in range(steps):
        _, telemetry, per_step[k] = patrol.step(clock)
        tiers[k] = telemetry['active_controller']
        modes[telemetry['flight_mode']] = modes.get(telemetry['flight_mode'], 0) + 1
        replans += bool(telemetry['planner']['replanned'])
    n = int(round(window * STEPS_PER_MINUTE))
    ms = 1e3 * per_step
    controllers = {tier: int(np.sum(tiers == tier)) for tier in sorted(set(tiers))}
    share = {tier: count / steps for tier, count in controllers.items()}
    windows = []
    for i in range(0, steps - n + 1, n):
        adjusted, tier_medians = mix_adjusted_median(ms[i:i + n], tiers[i:i + n], share)
        windows.append({'adjusted_ms': adjusted, 'median_ms': float(np.median(ms[i:i + n])),
                        'tier_medians_ms': tier_medians})
    return {
        'measure': 'time', 'minutes': minutes, 'steps': steps, 'clock': name, 'thread_time_tick_s': tick,
        'p99_ms': float(np.percentile(ms, 99)), 'max_ms': float(ms.max()), 'median_ms': float(np.median(ms)),
        'window_minutes': window, 'windows': windows,
        'minute_medians_ms': [round(float(np.median(ms[i:i + STEPS_PER_MINUTE])), 4)
                              for i in range(0, steps, STEPS_PER_MINUTE)],
        'controllers': controllers, 'modes': modes, 'replans': replans, 'reloads': patrol.reloads,
    }


def measure_memory(minutes, warmup, controller_kwargs):
    patrol = Patrol(**controller_kwargs)
    steps = int(round(minutes * STEPS_PER_MINUTE))
    traced, blocks = {}, {}                    # whole minute -> bytes traced since the first step, blocks
    tracemalloc.start()
    for k in range(steps):
        patrol.step()
        if (k + 1) % STEPS_PER_MINUTE == 0:
            traced[(k + 1) // STEPS_PER_MINUTE] = tracemalloc.get_traced_memory()[0]
            blocks[(k + 1) // STEPS_PER_MINUTE] = sys.getallocatedblocks()
    tracemalloc.stop()
    warm, end = int(round(warmup)), int(round(minutes))
    c = patrol.controller
    return {
        'measure': 'memory', 'minutes': minutes, 'warmup_minutes': warmup, 'steps': steps,
        'growth_bytes': traced[end] - traced[warm],
        'growth_blocks': blocks[end] - blocks[warm],
        'growth_by_minute_bytes': {str(m): b - traced[warm] for m, b in traced.items() if m > warm},
        'growth_by_minute_blocks': {str(m): b - blocks[warm] for m, b in blocks.items() if m > warm},
        'traced_by_minute_bytes': {str(m): b for m, b in traced.items()},
        'lengths': {'flight_log': len(c.flight_log), 'contract_history': len(c.contract_monitor.history),
                    **{f'runtime_{k}': len(v) for k, v in c.runtime_monitor.history.items()},
                    'switch_history': len(c.controller_switcher.switch_history)},
    }


class FormatAndDrop(logging.Handler):
    """Formats each record, as a node that logs would, and keeps nothing."""

    def emit(self, record):
        self.format(record)


def main(argv=None):
    # The core warns at each controller switch and mode change: pay for the formatting,
    # but keep no output, so the log neither floods the pipe nor grows in memory.
    logging.basicConfig(level=logging.WARNING, handlers=[FormatAndDrop()])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--minutes', type=float, required=True)
    parser.add_argument('--measure', choices=('time', 'memory'), required=True)
    parser.add_argument('--window', type=float, default=5.0, help='time: minutes at the start and the end')
    parser.add_argument('--warmup', type=float, default=5.0, help='memory: minutes before tracing starts')
    parser.add_argument('--flight-log-len', type=int, default=-1)
    args = parser.parse_args(argv)
    kwargs = {} if args.flight_log_len < 0 else {'flight_log_len': args.flight_log_len}
    if args.measure == 'time':
        result = measure_time(args.minutes, args.window, kwargs)
    else:
        result = measure_memory(args.minutes, args.warmup, kwargs)
    result['flight_log_len'] = kwargs.get('flight_log_len', 'default')
    result['python'] = sys.version.split()[0]
    print(json.dumps(result))


if __name__ == '__main__':
    main()
