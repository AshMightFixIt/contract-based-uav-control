# Known issues

This is the list of confirmed problems in the project, and where each one stands.

**Where the list comes from:** an analysis in September 2026 checked the code, the thesis and the PX4 / ROS 2 / Gazebo integration audit. Every issue below was confirmed by reading the code or by running it.

**About the evidence:** the analysis looked at the code before the files moved into `contract_uav_core/`, so its file references use the old names:

| Old file | New file |
|---|---|
| `controllers.py` | `control/pid.py`, `mpc.py`, `hinf.py`, `switcher.py` |
| `adaptive_control_system.py` | `core.py`, `conditions.py`, `switching_policy.py`, `telemetry.py` |
| `contract_framework.py` | `contracts/` |
| `contract_ekf.py` | `estimation/ekf.py` |
| `cbf_filter.py` | `safety/cbf.py` |
| `flight_mode_supervisor.py` | `control/supervisor.py` |

The code IDs (K01, A-R2 and so on) are kept so each issue can be traced back to the analysis.

The order of the fix batches is in [`ROADMAP.md`](ROADMAP.md).

**Status words used below:**

| Status | Meaning |
|---|---|
| Fixed | The fix is merged into the `HILSILCode` branch. |
| Batch 3 | Being fixed now (core correctness). |
| Batch 3b | Next: the scripts, plus one contract source (this one needs a decision first). |
| Later | A later batch: the ROS 2 bridge, the environment, Gazebo, or research and documents. |

## Flight core

| ID | Problem | Status |
|---|---|---|
| K01 | The thrust sign is backwards. All three controllers, and all seven simple drone models in the scripts, treat "more thrust" as "go down". In a correct model every altitude loop fails: starting at 4 m with a target of 5 m, the drone hits the ground in 1.1 to 1.6 s. | Batch 3 |
| A-R2 | Turning an acceleration into a tilt ignores the heading. The drone only moves the right way when it faces north. At 90 or 180 degrees the error grows to hundreds of metres. | Batch 3 |
| A-G6 | The code assumes a hover thrust of 0.5 for every drone, a 3-cell battery, and motors above 20 °C. The DEXI 5 defaults are hover thrust 0.22, maximum thrust 0.5 and a 6-cell battery. | Batch 3 |
| A-R4 | Bookkeeping grows without limit. Over 5 simulated minutes the mean step time grows from 1.5 to 10.7 ms, and memory from 35 to 172 MB. | Batch 3 |
| K17 | The estimator is a simple constant-velocity filter. It fuses missing measurements as zeros, does not wrap the heading (errors up to 137 degrees near ±180), freezes velocity without GPS (the position uncertainty grows from 0.3 to 116 m in 20 s), then snaps back. Its contract result is always False, and its GPS limits are hard-coded. | Batch 3 |
| A-F7 | When PX4 marks the position invalid, the bridge sends zeros, and the estimator fuses them. The estimate runs to the origin. The core half is in batch 3; the bridge half is later. | Batch 3 and later |
| K10 | The planner runs 10 or more Pacti compositions per replan, about 120 to 160 ms each, then throws them away. On the ground it replans on every step. It also approves H-infinity at wind 16 m/s, because it checks at wind − 1. | Batch 3 |
| A-F11 | Pre-flight cannot read real telemetry. The node uses a fixed set of values, so the check always passes. Real values (a 6-cell battery at about 22 V, or motors at 15 °C) make every controller "infeasible", and then the thrust is 0. | Batch 3 (core) and later (node) |
| K07 | "Mission complete" means "hovering while aiming at the last waypoint". The counts stop at N − 1, a hover after GPS loss counts as complete, and the benchmark's 5-second break means its last phases never run. | Batch 3 |
| K08 | There is no way back from HOVER to the mission. The sensor demo and benchmark work around it by writing the supervisor's mode directly. | Batch 3 |
| K13 | The switching signals are artefacts. One compares a fixed thrust of 0.5 with limits that sit below 0.5 at low wind. Another differentiates a jumpy error measure from a zero start, so the code switches PID to MPC at t = 0. The disturbance input is always 0. | Batch 3 |
| K14 | The 5-second wait between controller switches also blocks safety switches. A switch to H-infinity after GPS loss was blocked for 6.9 s. | Batch 3 |
| A-R3 | Everything assumes the drone starts in the air. On the ground the safety filter commands thrust 0.95, and below 2 m it adds +0.4. The landing floor sits inside that band. | Batch 3 |
| K06 | The safety filter is a set of fixed rules, not an optimisation. Its tilt and spin rules change only the torque values, which no drone model and not PX4 receive. | Batch 3 (behaviour); later (a real optimisation) |
| A-F13 | The time step is fixed, and the core's clock counts steps. Batch 2 added an optional real time (`control_step(raw, t)`), but only the estimator uses the measured step so far. The node has no simulated-time support. | Partly fixed; rest later |
| K03 | The wind limits disagree. The framework allows 3/15/25 m/s for PID/MPC/H-infinity, the Pacti planner 3/8/15, the supervisor's EMERGENCY starts above 15, and the old README said MPC up to 8. | Batch 3b; needs the owner's choice of the official contract set |
| K04 | Pacti's role was overstated. The runtime monitor and the pre-flight check never use it, and the two contract sets disagree on 40 of 55 limits. | Batch 3b (same decision) |
| K05 | Pre-flight checks one point over a composition that drops one implication. It accepts GPS quality (HDOP) up to 5, while PID's own assumption fails above 2.76. | Batch 3b (same decision) |
| K16 | The scripts' drone models are simple kinematic copies: tilt times 8 gives acceleration, speed is capped, and wind nudges the velocity. They ignore torques, yet the smoothness and effort numbers are mostly torque. | Batch 3b |

## Scripts and demos

| ID | Problem | Status |
|---|---|---|
| K18 | Outdated pre-flight inputs make `demos/simulation_demo.py` stop at its pre-flight check, so it never flies. | Batch 3b |
| K12 | `demos/demo_racing_live.py` crashes on NumPy 2 (`ndarray.ptp` was removed). | Batch 3b |
| K11 | The published results need Pacti. Without it the planner turned itself off, and the numbers in sections 1 and 3 changed. The requirements now list Pacti separately (`requirements/pacti.txt`), and batch 3 removes the runtime need for it. | Partly fixed; rest in batch 3 |

## ROS 2 bridge (`contract_uav_control/`)

| ID | Problem | Status |
|---|---|---|
| A-F9 | The switch into offboard mode is broken. OFFBOARD+ARM is sent once, at the 21st setpoint (0.4 s, below PX4's "more than 1 s" rule), with no retry, and the vehicle status is never read. With the planner on, the heartbeat drops below PX4's 2 Hz rule. | Later |
| A-F10 | The core runs while the drone is disarmed. The PID height integral winds up to its limit, the switch wait is used up, and the supervisor is already in TRACK before engage. | Later |
| A9-N2 | While disarmed, the node streams a near-full thrust (0.95), north-facing attitude setpoint, and PX4 receives exactly that when offboard engages. | Later |
| A9-N3 | A failed pre-flight only logs an error. The node still sends OFFBOARD+ARM if `auto_arm` is set. | Later |
| A9-N4 | The bridge ignores PX4's velocity-valid flags, so an invalid velocity is used whenever the position is valid. | Later |
| A-G3 | With no GPS message, the bridge invents 12 satellites and HDOP 1.0. The DEXI flies indoors on optical flow. | Later |
| A-E5, A-N1 | Topic names differ between PX4 releases. In v1.16 and v1.17 some topics carry a `_v1` suffix, so the node's unsuffixed names miss them. On v1.17 PX4 would ignore the node's attitude setpoints. | Later |
| A7-N1 | The ROS install notes say "numpy only", which leaves the planner silently off. | Later |
| A9-N1 | The pre-arm health check in `docs/HARDWARE_TESTING.md` cannot pass with the shipped defaults. | Later |
| — | Since batch 2 the core no longer sets up Python logging, so its INFO messages do not show in the node's output (warnings and errors do). | Later |

## Environment and setup

| ID | Problem | Status |
|---|---|---|
| A-E1, A-E2 | Pacti needs NumPy 2.2.6 or newer and Python 3.10 or newer. That clashes with ROS 2 Humble's NumPy 1.21.5, and the manual told users to pip-install Pacti into the ROS Python. | Later (after batch 3 removes Pacti from the runtime, this mostly goes away) |
| A-E4 | `px4_msgs` is cloned from `main` with no fixed version. | Later |
| A-N2 | The DEXI's current image runs Raspberry Pi OS Bookworm with ROS 2 Jazzy and PX4 v1.16.x, not Ubuntu with Humble. | Needs the owner's decision (D1) |
| — | The colcon (ROS) build is not tested in CI. | Later |
| — | One of Pacti's dependencies, pyparsing, is not pinned, and Pacti 0.3.1 already shows deprecation warnings. | Later |

## Gazebo

| ID | Problem | Status |
|---|---|---|
| A-G1 | Gazebo does not move the stock x500 with wind. In v1.16 and v1.17 PX4's server setup does not load the wind plugin, and a custom world with any plugin drops PX4's sensors. Wind should be applied as a force (ApplyLinkWrench). | Later |
| A-G2 | The bridge sends no wind. Only a drift estimate remains, and it is capped at 10 m/s, so it can never trigger H-infinity or EMERGENCY. | Later |

## Research claims and documents

| ID | Problem | Status |
|---|---|---|
| K15 | "H-infinity" uses hand-set gains with no synthesis. "MPC" is an unconstrained gain plus clipping, blended with the previous plan. The claimed stability and feasibility guarantees are not there. | Later |
| K02 | H-infinity runs 27 to 36 % of the time with good sensors and moderate wind, not "emergencies only". | Later |
| K09 | The speed claims (2.5 / 4.1 ms per step, 12.5 % CPU, a 5 ms planner) were never measured. | Later |
| K19 | The text in `docs/RESULTS.md` contradicts its own tables (winners, percentages, units, a single seed). Its numbers come from the old, backwards-thrust simulation; a note at the top says so. | Later (re-measured after batch 3) |
| K20 | The thesis in `thesis/` describes an older PID / H-infinity system. Its chapter 9 results have no source in the repository. | Later |
| A7-N2 | One troubleshooting tip in the manuals names a topic (`sensor_gps`) that PX4 v1.14 to v1.17 does not publish. | Later |

## Smaller issues found during the fixes

| Problem | Status |
|---|---|
| `save_flight_log` turns in-memory arrays into lists after saving, and a file name without `.json` is overwritten by the contract metrics. | Later |
| The MPC's measured solve time (wall clock) feeds a contract input, so a slow computer can change which controller is chosen. | Later |
| If the caller's time goes backwards, the core's clock follows it, and the switch and mode waits see negative elapsed time. | Later |
| "Pacti not available" is printed to standard output, not logged. | Later |
| The golden benchmark files rely on the MPC finishing each step within 50 ms; the slowest Windows CI runner is the most likely place for this to fail. | Watch |
| The DEXI's `MPC_THR_MIN` (0.025) is below PX4's documented minimum (0.05). | Check on the vehicle |
| The DEXI full-charge voltage of 4.2 V per cell is a datasheet value, not a PX4 parameter. Confirm it on the real batteries. | Check on the vehicle |

## Already fixed (merged into `HILSILCode`)

| ID | What was wrong |
|---|---|
| K21 | `test_all.sh` could not run and could not report a failure; the setup scripts installed nothing; `.gitignore` hid every output; `docs/DEVELOPMENT.md` showed functions that do not exist. |
| A-E6, A-E10 | The core was found only through `CONTRACT_UAV_ROOT` or `sys.path` hacks. It is now an installable package. |
| A-E8 | There was no `.gitattributes`. |
| A-F14 | Five library modules set up logging when imported, and the estimator logged 50 lines per second. |
| K11 (part) | Pacti was missing from the requirements. |
| — | The old README's unmeasured performance numbers, dated roadmap and "QP" safety-filter description were removed. |
