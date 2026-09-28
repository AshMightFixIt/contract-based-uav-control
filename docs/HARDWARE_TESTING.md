# DEXI Drone — Hardware Test Procedure

Step-by-step procedure for flight-testing the contract-based controller on the
**real DEXI drone**. Written for the whole team — follow it exactly, in order.

> **Prerequisite:** the controller must already pass in the simulator (SITL) before
> any hardware test. This document covers **hardware only**. If you have not flown it
> in `gz_x500` SITL yet, stop and do that first (see `TESTING_MANUAL.md`).

---

## Roles for every hardware test

Assign these **before** powering anything on. Never test alone.

| Role | Responsibility |
|------|----------------|
| **Safety Pilot** | Holds the RC transmitter the entire flight. Can take over instantly. Has final say to abort. |
| **Operator** | Runs the laptop / QGroundControl. Arms, switches modes, watches data. |
| **Observer** | Watches the drone (not a screen). Calls out anything abnormal. Keeps bystanders clear. |

---

## Where everything runs

Everything flies **on the drone's onboard computer**. The laptop only monitors.

```
DEXI Drone (onboard computer)              Laptop
  - contract_uav_core/ (the algorithm)       - QGroundControl (monitor + arm)
  - contract_uav_control/ (ROS 2 bridge)     - SSH terminal (to start controller)
  - MicroXRCEAgent (DDS bridge)
        | wired link inside the drone
  PX4 Flight Controller (Pixhawk)
        |
  Motors / IMU / GPS

RC Transmitter --> PX4 directly (always highest priority)
```

---

## A. Pre-flight safety checklist (every single flight)

Do not arm until **every** box is ticked.

- [ ] Test site is an open area, or drone is tethered / props removed for first checks
- [ ] Bystanders are clear of the flight zone
- [ ] Battery fully charged and secured
- [ ] Propellers checked — correct, tight, undamaged
- [ ] Firmware on PX4 matches the version validated in SITL
- [ ] **`auto_arm` is set to `false`** in `config/params.yaml`
- [ ] RC transmitter on, bound, and Safety Pilot is holding it
- [ ] RC mode switch is set to a **manual** mode (Position / Stabilize), NOT Offboard
- [ ] QGroundControl connected and drone shows healthy (green)
- [ ] Everyone knows the abort call and who makes it

---

## B. Start the system (3 terminals on the onboard computer)

SSH into the drone from the laptop, then open three terminals (or use `tmux`).
In each terminal first run:
```bash
source /opt/ros/humble/setup.bash
source ~/contract-based-uav-control/install/setup.bash
```

**Terminal 1 — DDS bridge** (connects ROS 2 to PX4 over the internal serial link)
```bash
MicroXRCEAgent serial --dev /dev/ttyAMA0 -b 921600
```
> The port/baud may differ on your DEXI — check the DEXI docs. You should see
> client/topic activity once PX4 is powered.

**Terminal 2 — the controller**
```bash
ros2 launch contract_uav_control contract_control.launch.py
```
You should see log lines like `[track] ctrl=PID cbf=False thrust=0.50`.

**Terminal 3 — start recording BEFORE arming**
```bash
ros2 bag record /contract_uav/state -o flight_$(date +%Y%m%d_%H%M%S)
```

---

## C. Verify the controller is healthy (do this BEFORE arming)

In a fourth terminal:
```bash
ros2 topic hz /contract_uav/state
```
- **Must show ~50 Hz.** If it shows nothing or an unstable rate — **STOP. Do not arm.**
  Fix the link first (see Troubleshooting).

Quick sanity check of the live values:
```bash
ros2 topic echo /contract_uav/state --field cmd_thrust
```
- At rest this should sit near **0.5** (hover thrust). Wildly different → stop and check.

---

## D. Arm and engage (Operator + Safety Pilot together)

1. Operator confirms with the Safety Pilot: "Arming now."
2. In QGroundControl: **Arm** the drone.
3. Safety Pilot moves the RC mode switch to **Offboard**.
4. The drone is now flown by the algorithm.

**At any moment**, the Safety Pilot flicks the mode switch back to Position/Stabilize
to take full manual control. The algorithm keeps running but PX4 ignores it.

---

## E. What to watch during the flight

Keep these on screen. The Observer watches the drone, not the screen.

| Field | Watch for | Meaning |
|-------|-----------|---------|
| `position_error_norm` | should be small / shrinking | distance (m) from target — the main metric |
| `active_controller` | `PID` nominal; `Hinf` = stress | which controller is flying right now |
| `cbf_intervened` | occasional OK; constant = problem | safety filter is overriding commands |
| `flight_mode` | matches what you expect | track / hover / land / emergency |
| `cmd_thrust` | near 0.5 at hover | throttle command |

Live view of the key number:
```bash
ros2 topic echo /contract_uav/state --field position_error_norm
```

---

## F. Abort procedure (everyone must know this)

Call **"ABORT"** loudly if the drone does anything unexpected. On abort:

1. **Safety Pilot** immediately flicks the RC mode switch to **Position/Stabilize**
   and flies the drone manually (land or hover).
2. If the drone is unrecoverable, Safety Pilot disarms via RC (throttle down + disarm).
3. Operator stops the controller (Ctrl+C in Terminal 2) — but only after the Safety
   Pilot has control.
4. Leave the bag recording running; the data is valuable for diagnosing what happened.

---

## G. Land and collect data

1. Safety Pilot switches to Position mode and lands manually (or commands LAND).
2. Disarm.
3. Stop recording: `Ctrl+C` in Terminal 3.
4. Note in the log sheet: what you tested, what happened, any abort.

Review the flight:
```bash
ros2 run plotjuggler plotjuggler
# File -> Data -> load the bag -> drag position_error_norm onto a plot
```
Compare against the SITL run and against previous hardware runs.

---

## H. Staged hardware progression — DO NOT SKIP A STAGE

Run the **same A–G procedure** at each stage. Only move to the next stage once the
current one is rock-solid. Record a bag every time.

| Stage | Setup | Goal / pass criteria |
|-------|-------|----------------------|
| **1. HITL** | Real PX4 FMU in the loop, **no props**, drone clamped | Topics at 50 Hz, `cmd_thrust ≈ 0.5`, no errors, controller logic sane |
| **2. Tethered hover** | Props on, drone tethered, low altitude | Stable hover, `position_error_norm` small, minimal `cbf_intervened` |
| **3. Free hover** | Open area, untethered, ~2 m | Holds position; smooth thrust; no drift |
| **4. Free waypoints** | Open area, the default mission | Reaches each waypoint; watch controller switching as it moves |
| **5. Wind / stress** | Windy day or moderate disturbance | Watch `active_controller` escalate PID → MPC → H-inf |

---

## I. Troubleshooting (hardware-specific)

| Symptom | Check |
|---------|-------|
| No `/contract_uav/state` topic | Is the DDS agent (Terminal 1) running? Is PX4 powered? `ros2 topic list` |
| Topic exists but 0 Hz | DDS link down — check serial port/baud, cable, PX4 uXRCE-DDS client enabled |
| Drone won't arm | QGroundControl message panel; PX4 needs the offboard heartbeat streaming first (Terminal 2 must be running) |
| Won't enter Offboard | Heartbeat must stream >2 Hz before switching; confirm Terminal 2 is publishing |
| GPS topic missing | Some PX4 versions use `/fmu/out/sensor_gps` vs `/fmu/out/vehicle_gps_position` — update the subscription name in `controller_node.py` |
| Drifts / oscillates unlike SITL | Gains were tuned in sim — re-tune on hardware (see `TESTING_MANUAL.md` §6) |
| `cbf_intervened` constantly true | Limits may be too tight for the real airframe — review `contract_uav_core/contract_uav_core/safety/cbf.py` |

---

## J. Per-flight log sheet (copy one per flight)

```
Date / time:            ____________________
Stage (HITL/hover/...): ____________________
Safety Pilot:           ____________________
Operator / Observer:    ____________________
Bag filename:           ____________________
Battery start / end:    ______ V  /  ______ V
Wind conditions:        ____________________

Result (circle):   PASS  /  PASS w/ notes  /  ABORT

position_error_norm (typical): ______ m
Controllers seen:    PID  /  MPC  /  H-inf
CBF interventions:   none / few / many
Abort? when & why:   ____________________________________
Changes for next run:____________________________________
```

---

*Controller internals and gain-tuning recipe: see `TESTING_MANUAL.md` and the main
project `README.md`. This document covers the hardware flight procedure only.*
