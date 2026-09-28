# DEXI Drone — Testing & Tuning Manual

A step-by-step guide for testing the contract-based controller on the DEXI drone.
Written so anyone on the team can follow it, even without deep ROS 2 experience.

**Golden rule: never skip the simulation step. Always fly in the simulator (SITL)
first, then hardware-in-the-loop, then a tethered drone, then free flight.**

---

## 0. What you are testing

The drone is flown by a "contract-based" controller. It automatically picks one of
three controllers depending on conditions:

| Controller | When it is used | Think of it as |
|-----------|-----------------|----------------|
| **PID**  | Calm air (wind under ~3 m/s) | Efficient, smooth |
| **MPC**  | Moderate wind (up to ~8 m/s) | Optimal, predictive |
| **H-inf**| Strong wind / emergencies (up to ~25 m/s) | Robust, safe |

A safety filter (**CBF**) sits on top and blocks any command that would break a hard
limit (too low, too fast, too tilted). Your job in testing is to watch these pieces
behave and tune them.

---

## 1. One-time setup

Do this once on the computer that will run the controller (your laptop for SITL, or
the DEXI onboard computer for hardware).

### 1.1 Install the basics
- Ubuntu 22.04
- ROS 2 Humble — follow the official ROS 2 install guide
- Python packages: `pip install numpy` (and `pip install pacti` if you want the
  horizon planner; it works without it)

### 1.2 Get the code
```bash
# the controller repo (this project)
git clone <your-repo-url> contract-based-uav-control

# PX4 message definitions (needed to talk to the flight controller). The repo
# root is also the colcon workspace; px4_msgs/ is gitignored there.
cd contract-based-uav-control
git clone https://github.com/PX4/px4_msgs.git
```

### 1.3 Install the DDS bridge (lets ROS 2 talk to PX4)
Follow PX4's "uXRCE-DDS" guide to install `Micro-XRCE-DDS-Agent`. Test that the
`MicroXRCEAgent` command exists:
```bash
MicroXRCEAgent --help
```

### 1.4 Build the workspace
```bash
cd contract-based-uav-control

# contract_uav_core is the core Python code; the node imports it once it is built
colcon build --packages-select px4_msgs contract_uav_msgs contract_uav_core contract_uav_control
source install/setup.bash
```
You should see "Finished" with no errors. If `px4_msgs` fails, build it alone first,
then build the others.

> **Tip:** add this line to your `~/.bashrc` so you don't retype it:
> ```bash
> source /full/path/to/contract-based-uav-control/install/setup.bash
> ```

---

## 2. Test in the simulator (SITL) — DO THIS FIRST

You need **four terminals**. In every terminal, first run:
```bash
source /opt/ros/humble/setup.bash
source ~/contract-based-uav-control/install/setup.bash
```

**Terminal 1 — the DDS bridge**
```bash
MicroXRCEAgent udp4 -p 8888
```

**Terminal 2 — PX4 simulator**
```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500
```
Wait until you see `Ready for takeoff!`.

**Terminal 3 — the controller**
```bash
ros2 launch contract_uav_control contract_control.launch.py
```
You should see log lines like `[track] ctrl=PID cbf=False thrust=0.50`.

**Terminal 4 — watch the data** (see Section 4).

### Arming in the simulator
For SITL you may let the controller arm itself. Edit
`contract_uav_control/config/params.yaml` and set `auto_arm: true`,
rebuild, and relaunch. The drone should arm, switch to OFFBOARD, and start flying
the waypoints. **Keep `auto_arm: false` for hardware (Section 5).**

---

## 3. The data you can monitor

The controller publishes everything on one ROS 2 topic: **`/contract_uav/state`**.
This is your main tuning tool. Key fields:

| Field | What it tells you | Use it to tune |
|-------|-------------------|----------------|
| `position`, `velocity`, `attitude` | where the drone actually is | sanity / drift |
| `setpoint_position` | where it is being told to go | compare to position |
| `position_error_norm` | how far off it is (metres) | **main tracking metric** |
| `attitude_error` | desired vs actual tilt | inner-loop PID gains |
| `cmd_thrust` | throttle command (0.5 = hover) | altitude / hover thrust |
| `cmd_desired_attitude` | commanded tilt | aggressiveness |
| `active_controller` | PID / MPC / H-inf right now | switching behaviour |
| `flight_mode` | track / hover / land / emergency | mission logic |
| `cbf_intervened` | safety filter stepped in (true/false) | limits too tight? |
| `planner_safety_margin` | predicted safety headroom | planner thresholds |
| `contract_*` | each component's contract status | spot violations |

---

## 4. How to look at the data

Pick whichever you like. All three read the same `/contract_uav/state` topic.

### A. Quick text view (no setup)
```bash
ros2 topic echo /contract_uav/state
```
Good for a fast "is it alive and sane?" check.

### B. Live graphs with PlotJuggler (best for tuning)
```bash
sudo apt install ros-humble-plotjuggler-ros   # one time
ros2 run plotjuggler plotjuggler
```
In PlotJuggler: *Start* → ROS 2 Topic Subscriber → tick `/contract_uav/state`.
Drag `position_error_norm` onto a plot. Drag `cmd_thrust`, `attitude_error`, etc.
onto more plots. Watch them update live while the drone flies.

### C. Simple graphs with rqt_plot
```bash
ros2 run rqt_plot rqt_plot /contract_uav/state/position_error_norm
```

### Record a flight for later analysis (do this on EVERY test)
```bash
ros2 bag record /contract_uav/state -o flight_$(date +%Y%m%d_%H%M%S)
```
Stop with Ctrl+C after the flight. Replay later with:
```bash
ros2 bag play flight_20260622_1530
```
and open it in PlotJuggler (File → Data → load the bag). This lets you compare
before/after tuning runs side by side.

---

## 5. Hardware testing on the DEXI

Only after the simulator flight looks good.

### 5.1 Safety first — every single time
- [ ] Fly outdoors in an open area, or indoors only with props off / drone tethered.
- [ ] A human pilot holds the RC transmitter, ready to take over.
- [ ] `auto_arm` is set to **false** in `params.yaml`.
- [ ] Battery charged; props checked; firmware matches the simulator version.
- [ ] QGroundControl is connected and shows the drone is healthy.

### 5.2 Connect the controller to the real drone
On the DEXI onboard computer, the DDS bridge runs over the serial link to the flight
controller instead of UDP (check your DEXI docs for the exact port/baud):
```bash
MicroXRCEAgent serial --dev /dev/ttyAMA0 -b 921600
```
Then start the controller exactly as in Terminal 3 above.

### 5.3 Arm manually
1. Confirm `/contract_uav/state` is publishing (`ros2 topic hz /contract_uav/state`
   should show ~50 Hz).
2. In QGroundControl, arm the drone and switch the flight mode to **Offboard**.
3. The drone now follows the controller. The RC pilot can switch out of Offboard at
   any moment to take manual control.

### 5.4 First flight checklist
- [ ] Start `ros2 bag record /contract_uav/state` BEFORE arming.
- [ ] Begin with a simple hover (one waypoint at the start position, a few metres up).
- [ ] Watch `position_error_norm` and `cmd_thrust` live.
- [ ] If anything looks wrong, the RC pilot takes over immediately.

---

## 6. How to tune (simple recipe)

Change **one thing at a time**, re-fly the same mission, and compare the recorded
data. The controller gains live in the core code, not in ROS:
`contract_uav_core/contract_uav_core/control/` (`pid.py`, `mpc.py`, `hinf.py`).

| Symptom you see in the data | What to try |
|-----------------------------|-------------|
| Drone slowly drifts, never reaches target (`position_error_norm` stays high) | Increase position P gain `kp_pos` |
| Drone overshoots / wobbles around the target | Decrease `kp_pos`, or increase damping `kd_pos` |
| Slowly creeps to target then sits with small steady error | Increase integral gain `ki_pos` a little |
| Altitude sags or climbs (`cmd_thrust` not near 0.5 at hover) | Adjust `hover_thrust` |
| Tilt is jerky (`attitude_error` noisy) | Lower attitude gains `kp_att` |
| `cbf_intervened` is true a lot in normal flight | Limits may be too tight — review `contract_uav_core/contract_uav_core/safety/cbf.py` |
| Switches controllers too often (`active_controller` flickers) | Increase the hysteresis margin in `contract_uav_core/contract_uav_core/switching_policy.py` |

After each change, rebuild the core (`colcon build --packages-select contract_uav_core`)
and restart the controller node (Terminal 3). If you built with
`colcon build --symlink-install`, edits to the core Python apply without a rebuild;
just restart the node.

---

## 7. If something goes wrong

| Problem | Check this |
|---------|-----------|
| Controller prints nothing / no `/contract_uav/state` | Is the DDS agent (Terminal 1) running? Is PX4 publishing? `ros2 topic list` |
| `No module named 'contract_uav_core'` error | The core is not built or not sourced: `colcon build --packages-select contract_uav_core`, then `source install/setup.bash` |
| Topics exist but no data | QoS mismatch — confirm you built the latest node; PX4 uses best-effort QoS |
| Drone won't arm | Look at QGroundControl messages; PX4 needs the offboard heartbeat streaming first |
| GPS topic missing | Some PX4 versions use `/fmu/out/sensor_gps` instead of `/fmu/out/vehicle_gps_position` — update the subscription name in `controller_node.py` |
| Drone behaves differently than sim | Gains were tuned in a simple sim; re-tune on hardware (Section 6) |

---

## 8. Recommended test progression

1. **SITL hover** — one waypoint, calm air. Confirm stable hover, `thrust ≈ 0.5`.
2. **SITL waypoints** — the default 3-waypoint mission. Confirm it reaches each.
3. **SITL wind** — add wind in the simulator; watch `active_controller` switch
   PID → MPC → H-inf as wind rises.
4. **HITL** — same tests with the real flight controller in the loop.
5. **Tethered hover** — real drone, tethered, props on, manual arm.
6. **Free hover** — open area, RC pilot ready.
7. **Free waypoints** — only after hover is rock-solid.

Record data at every step. Keep the bags. Compare runs.

---

*Questions about the controller internals are in the main project `README.md` and
`DEVELOPMENT.md`. This manual only covers running and tuning it.*
