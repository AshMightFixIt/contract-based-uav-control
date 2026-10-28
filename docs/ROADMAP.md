# Roadmap

This is the plan for fixing the problems in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md), and the order the work happens in.

**How the work is done:**
- Work happens in **batches**. Each batch goes into the `HILSILCode` branch through a pull request, merged with "Rebase and merge".
- Each change is made by a builder and checked by a separate reviewer.
- Behaviour changes are proven with tests. The saved benchmark outputs in `tests/golden/` change only in a commit that explains every changed number.
- A batch ends with a **gate**: a fresh check that the batch did what it promised. The next batch starts only after the gate passes.

## Done

| Batch | What it did | Pull request |
|---|---|---|
| 1 | Automatic tests on Linux and Windows (`make test`), saved benchmark outputs, and the offline contracts tool (`tools/contracts_offline/`) | #1 |
| — | The battery model and planner speed-up from the owner | #2 |
| 2, step A | The core became an installable package (`contract_uav_core/`), and the ROS 2 packages moved to the top level | #3 |
| 2, step B | Fixed interfaces (`control_step(raw, t)`, `AccelCommand`, the switch priority), a reference drone model (`sim/quadrotor.py`), and the x500 and DEXI airframe files | #4 |
| Gate 0a | Passed: a fresh install works, the benchmarks were unchanged, and the node loads the core with no path settings | — |
| Cleanup | The folders `benchmarks/`, `demos/`, `requirements/`, `thesis/` and `outputs/`, a plain-language README, and the license year | #5 |

## Batch 3: core correctness (in progress)

Five fixes. Each owns its own files, so they do not overlap:

| Fix | What it does | Issues |
|---|---|---|
| W4-05 | Thrust with the correct sign, tilt that follows the heading, thrust scaled by each airframe's hover thrust, more airframe fields | K01, A-R2, A-G6 |
| W4-06 | Flight phases (take-off to landing), a way back from HOVER to the mission, switching signals that make sense, safety switches that skip the 5 s wait, a safety filter that understands the ground | K07, K08, K13, K14, A-R3, K06 (behaviour) |
| W4-07 | Bounded memory and step time: a 30-minute run stays under 2 ms per step at p99 | A-R4 |
| W4-08 | A correct estimator: fuse only valid data, wrap the heading, a bounded no-GPS mode, smooth GPS recovery, and a PX4 pass-through mode | K17, A-F7 (core) |
| W4-09 | A pre-flight check that reads real data, a planner with no Pacti at run time and under 1 ms, and a small getter for the wind limits | K10, A-F11 (core) |

**Order:** W4-05 goes in first, because W4-06's tests need correct thrust. Then W4-09, W4-08 and W4-07. W4-06 goes in last.

**Batch 3b** comes next:
- **W4-10** moves the scripts onto the reference drone model, and fixes `simulation_demo.py` and the NumPy 2 crash (K16, K18, K12).
- **W4-11** makes one official contract source (K03, K04, K05). It needs the owner's decision **DC**, below.

**Gate 1** checks:

| Check | What must hold |
|---|---|
| G1.1 | Height steps settle for every controller |
| G1.2 | Sideways steps settle at any heading |
| G1.3 | Ground start and landing work |
| G1.4 | A 30-minute run stays in budget |
| G1.5 | The planner meets its speed budget with no Pacti at run time |
| G1.6 | The estimator handles its edge cases |
| G1.7 | Safety switches are never blocked |
| G1.8, G1.9 | The contract checks; these need decision DC |
| G1.10 | The README commands work |

## After gate 1

| Phase | What it covers |
|---|---|
| 0b, environment (W4-12) | Fixed versions for PX4, `px4_msgs`, the DDS agent and ROS; a ROS build job in CI; a check that `px4_msgs` matches PX4 |
| 2, ROS 2 bridge (W4-13) | A correct offboard engage with a heartbeat and retries, no control while disarmed, real pre-flight data, validity flags, versioned topic names, simulated time, and node logging |
| 2D, DEXI bring-up | First flights on the real DEXI, after the owner's on-vehicle checks |
| 3, Gazebo (W4-15) | The same scenarios in PX4 simulation, with wind applied as a force |
| 4, presentation (W4-16) | Every figure made from recorded flight data |
| 5, research validity and documents (W4-14, W4-17) | Make the MPC, H-infinity, safety filter and switching claims true, or rename them; measure the speed for real; regenerate `docs/RESULTS.md`; update the thesis |

## Decisions the owner needs to make

| ID | Question | Recommendation |
|---|---|---|
| DC | Which contract set is official? The two sets disagree; for example MPC is allowed up to 15 m/s of wind in one and 8 m/s in the other. | One contract file, composed offline by Pacti and read by every part of the code. The limit values are the owner's call. |
| D1 | Where does the DEXI's software run? | Keep Humble on the workstation, and also support Jazzy on the DEXI's own image. |
| D4 | Which PX4 release to pin | v1.16.x, with `px4_msgs` `release/1.16` at a fixed commit |
| D6 | Where the wind measurement comes from in simulation | An emulated wind sensor, with the drift estimate reported for comparison |
| D7 | Which position estimate the bridge uses by default: the core's estimator or PX4's (pass-through) | Decide during Phase 2 |

## Checks only the owner can run

| Check | What to do |
|---|---|
| Check 5 (on the DEXI) | Report the PX4 version (`ver all`), the parameters `MPC_THR_HOVER`, `BAT1_N_CELLS`, `BAT1_V_EMPTY` and `MPC_THR_MAX`, the OS version, the ROS distribution, the Python and NumPy versions, and the list of PX4 topics. These replace the PX4 defaults in `dexi.yaml`. |
| Open question 4 | Does `MPC_THR_MAX` limit thrust in offboard attitude mode? This decides whether the DEXI's 0.5 cap applies to the node's commands. |
| Check 7 | Confirm that a force applied with ApplyLinkWrench moves the x500 in Gazebo. |
