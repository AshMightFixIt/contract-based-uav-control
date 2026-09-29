# Contract-Based UAV Control

## What this project is

This project is a flight controller for a small drone with four rotors.
Such a drone is called a quadrotor.
UAV stands for "unmanned aerial vehicle", which is another word for a drone.
A controller is code that decides the thrust and the tilt of the drone.
Thrust is how hard the rotors push the drone up.
This project has three controllers, called PID, MPC and H-infinity.
The section "How it works" explains each one.
It uses one of them at a time, and it switches when things change.
For example, it can switch when the wind gets stronger.
Each part of the system has written rules, called contracts.
A contract says what the part may assume and what it must deliver.
The code checks these rules before the flight and at every step of it.
Today the project runs in a simulation, which is a program that pretends to be the drone.
It also has a bridge to PX4, the autopilot software used on many real drones.
The bridge uses ROS 2, a system that lets robot programs send messages to each other.

**Do not fly a real drone with this code yet.**
The controllers have a sign error in the thrust, so a real drone would fall.
See "Known problems" below.

## What is in each folder

| Folder | What is in it |
|---|---|
| `benchmarks/` | Scripts that fly one test course with each controller and compare the results. |
| `demos/` | Scripts that each show one part of the system, such as flying to a list of points (waypoints) or coping with broken sensors. |
| `requirements/` | Lists of Python packages to install. See "Install". |
| `contract_uav_core/` | The main Python package. It holds the controllers, the contracts, the other flight parts (see "How it works") and a simple drone model. Most of the code is here. |
| `contract_uav_control/` | The ROS 2 bridge. It connects the core package to PX4. |
| `contract_uav_msgs/` | The ROS 2 message type that the bridge uses to report its state. |
| `docs/` | Longer guides: testing with PX4 ([`docs/TESTING_MANUAL.md`](docs/TESTING_MANUAL.md)), tests on the real drone ([`docs/HARDWARE_TESTING.md`](docs/HARDWARE_TESTING.md)), notes for developers ([`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)) and benchmark results ([`docs/RESULTS.md`](docs/RESULTS.md)). |
| `thesis/` | The source files of the written thesis about this project, in LaTeX (a language for writing documents). Parts of it describe an older version of the code. |
| `tests/` | The automatic tests. `make test` runs them. |
| `tools/` | The offline contracts tool, in `tools/contracts_offline/`. It uses Pacti to combine the contracts and to compare the two sets of contract values in the code. It never runs on the drone. |
| `outputs/` | Plots and flight logs. The scripts create this folder when you run them. Git ignores it. |

Important files at the top level:

- `Makefile` holds short test commands, such as `make test`.
- [`LICENSE`](LICENSE) is the license text (MIT).
- `pytest.ini` holds the test settings.
- `.github/workflows/` holds the CI setup. CI means that the tests run by themselves each time someone pushes a change to GitHub.

## Install

You need Python 3.10 or 3.11.
CI tests the code with these two versions, on Linux and on Windows.
Other versions are not tested.

1. Get the code and go into its folder.

   ```bash
   git clone https://github.com/AshMightFixIt/contract-based-uav-control.git
   cd contract-based-uav-control
   ```

2. Make a virtual environment and turn it on.
   A virtual environment is a private set of Python packages for one project.
   It keeps this project's packages away from the rest of your computer.
   Git ignores the `.venv` folder that it makes.

   On Linux or macOS:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, in Command Prompt:

   ```bat
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```

   In Windows PowerShell, use `.venv\Scripts\Activate.ps1` for the second line.

3. Install the core package.

   ```bash
   python -m pip install -e ./contract_uav_core
   ```

   The `-e` means "editable".
   Python then uses the code in this folder, so your changes work at once.

4. Install the packages that the demos need.
   These are matplotlib, for plots, and pygame, for the 3D window.

   ```bash
   python -m pip install -r requirements/demos.txt
   ```

The other files in `requirements/`, and when you need them:

- `requirements/core.txt` lists only what the core package needs: NumPy (a math library) and PyYAML (it reads settings files). Step 3 already installs these.
- `requirements/test.txt` adds pytest, the tool that runs the tests. You need it to run the tests.
- `requirements/pacti.txt` adds Pacti 0.3.1, a Python library for contracts. You need it to turn on the planner (see "How it works") and to get the numbers in `docs/RESULTS.md`. It needs Python 3.10 or newer, and it always installs NumPy 2.
- `requirements/contracts_tool.txt` sets exact versions for the offline contracts tool. It fixes the NumPy version too, so give the tool its own virtual environment. See [`tools/contracts_offline/README.md`](tools/contracts_offline/README.md).

Without Pacti, the planner is off, and the rest of the code still runs.

## Run something

Run these commands from the top folder of the project, with the virtual environment turned on.

A demo:

```bash
python demos/demo_waypoint_mission.py
```

The drone flies to four waypoints while the wind changes.
A waypoint is a point in the air that the drone must reach.
The printed table shows which controller is in use.
A 3D window shows the flight.
If you close the window or press Esc, the flight ends early.
This demo takes about 2 minutes.

A benchmark:

```bash
python benchmarks/benchmark_controllers.py
```

This flies the same mission four times.
In the first run, the code switches controllers by itself.
Each of the other three runs uses one controller only.
The script prints a table that compares the four runs.
This takes about 30 seconds.

The tests:

```bash
python -m pip install -r requirements/test.txt
make test
```

If you do not have `make` (common on Windows), run `python -m pytest` instead.

Five of the scripts save a plot in the `outputs/` folder at the top of the project.
They create this folder the first time.
Each one prints the full path of the file it saved.

On a normal desktop, some scripts open a window.
On a computer with no screen, turn the windows off.
On Linux or macOS, put three settings in front of the command, like this:

```bash
MPLBACKEND=Agg SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python demos/demo_waypoint_mission.py
```

To see the windows again, leave these settings out.

Here is every demo and benchmark.
The times were measured on one Linux computer, with the windows turned off and without Pacti.
Your times will differ.

| Script | What it shows | Time |
|---|---|---|
| `demos/demo_waypoint_mission.py` | Four waypoints in changing wind. The controller switches. Opens a 3D window. Saves a plot. | about 2 min |
| `demos/demo_sensor_degradation.py` | GPS and IMU faults during a mission, and how the estimator copes. Opens a 3D window. Saves a plot. | about 30 s |
| `demos/demo_battery.py` | The battery model: voltage, power use and flight time left. Prints tables only. | under 1 s |
| `demos/demo_horizon_planner.py` | The planner looking ahead. Needs Pacti. Without Pacti it still runs, but the planner is off. Prints tables only. | under 1 s without Pacti, about 1.5 min with Pacti |
| `demos/demo_racing_live.py` | Four drones race live in a window, one per controller setup. Does not work with NumPy 2 (see "Known problems"). | fails at once |
| `demos/simulation_demo.py` | Meant to show all four flight modes. Today it stops at its pre-flight check (see "Known problems"). | under 1 s |
| `benchmarks/benchmark_controllers.py` | One waypoint mission: automatic switching against each controller alone. Saves a plot. | about 30 s |
| `benchmarks/benchmark_sensor_degradation.py` | The sensor-fault mission: automatic switching against each controller alone. Saves a plot. | about 20 s |
| `benchmarks/benchmark_racing.py` | A ten-waypoint race course: automatic switching against each controller alone. Saves a plot. | about 1.5 min |

## How it works

At every step (50 steps per second by default), data flows through these parts in order:

```
  sensors            GPS position, IMU tilt and turn rates, battery
     |
     v
  estimator          best guess of where the drone is and how it moves
     |
     v
  planner            optional (needs Pacti): may ask for a stronger controller
     |
     v
  supervisor         picks the flight mode, the target point and the controller
     |
     v
  controller         PID, MPC or H-infinity: works out thrust and tilt
     |
     v
  safety filter      changes the command if it would break a hard limit
     |
     v
  commands           thrust and tilt (attitude) sent to the drone
```

Positions follow the NED rule: x points north, y points east and z points down.
So a height of 5 m is z = -5.
All paths below are inside `contract_uav_core/contract_uav_core/`.
Each step is one call of `control_step` in `core.py`.
It runs the parts in the order shown above.

**Sensors.**
In the simulation, each script makes up sensor readings from its own simple drone model.
With PX4, the bridge reads them from the drone.
`core.py` turns the readings into measurements.

**Estimator** (`estimation/ekf.py`).
Sensor readings are noisy.
The estimator mixes them into one best guess of position, speed, tilt and turn rate.
The code calls it an EKF (extended Kalman filter).
In fact it is a simpler filter that assumes the drone keeps a steady speed.
It uses two sensors.
GPS gives the position from satellites.
The IMU measures tilt and turning.
The estimator has three modes.
With good GPS and a good IMU, it uses both.
With bad GPS, it uses only the IMU.
With a bad IMU, it uses no sensor at all and just moves its last guess forward in time.

**Planner** (`planning/`).
The planner is optional.
It runs only when Pacti is installed.
It looks a short time ahead.
Its advice can only raise the choice of controller, never lower it.

**Supervisor** (`control/supervisor.py`).
The supervisor picks the flight mode.
There are four modes.
TRACK flies to the waypoints in order.
HOVER holds the drone still in one place.
LAND goes down slowly, at 0.5 m/s.
EMERGENCY holds the drone at a safe height.
The mode sets the target point for the controller.
Some mode changes happen by themselves.
When the wind is over 15 m/s, the mode becomes EMERGENCY.
When GPS is lost during TRACK, or the last waypoint is reached, the mode becomes HOVER.
The supervisor can also overrule the choice of controller.
In LAND and EMERGENCY, and in HOVER without GPS, it always picks H-infinity.

**Controllers** (`control/pid.py`, `control/mpc.py`, `control/hinf.py`).
Each controller compares where the drone is with where it should be.
From that it works out the thrust and the tilt.
PID is the simplest.
It reacts to the error now, to the error built up over time, and to how fast the error changes.
MPC looks 15 steps ahead with a simple model, and picks the moves that fit best.
H-infinity is named after a math method for building controllers that cope with the worst gusts.
The version here is simpler.
It uses fixed, hand-picked settings, and it strongly slows down any spinning.
It is the fallback, because its contract allows the widest range of conditions.
`control/switcher.py` holds all three and makes the switch.

**Safety filter** (`safety/cbf.py`).
The code calls it a CBF (control barrier function) filter.
The safety filter watches four limits.
The height must stay between 2 m and 50 m.
The speed must stay under 15 m/s.
The tilt must stay under 30 degrees.
The turn rate must stay under 3 radians (about 170 degrees) per second.
Near a limit, the filter changes the command with a few fixed rules, one after another.
For example, when the drone is too low, it adds thrust.
It does not solve an optimization problem (a math search for the best command).

**Contracts** (`contracts/`).
`contracts/library.py` writes down the contract of each part.
`contracts/spec.py` checks a contract against numbers.
`contracts/monitor.py` checks the contracts during the flight and keeps a record.
`contracts/preflight.py` runs the pre-flight check.
This check tries PID first, then MPC, then H-infinity.
It starts with the first controller whose contracts fit the conditions.
If none fits, the flight does not start.

### How the switching works

The rules are in `switching_policy.py`.
At every step, the code checks the PID and MPC contracts against the current conditions.
These conditions include the estimated wind speed.
It prefers PID if the PID contract holds and PID should keep the error under 3 m.
If not, it picks MPC in the same way.
If neither fits, it picks H-infinity.
It moves to a stronger controller if the error grows fast.
It also does so if the controller in use breaks its promise.
Going back to a weaker controller needs a clear margin: a predicted error under 2.1 m.
This stops the choice from flipping back and forth.
If GPS and IMU both fail, it picks H-infinity.
Then the planner and the supervisor may raise the choice, as described above.
Last, `control/switcher.py` makes the switch.
It waits at least 5 seconds between two switches.
When it switches, it passes the built-up correction to the new controller, to make the change smooth.

### What a contract is

A contract has two parts.
The assumptions say what must be true for the part to work.
The guarantees say what the part promises when the assumptions are true.

Here is a real example from `contracts/library.py`.
The PID contract assumes that the wind speed is between 0 and 3 m/s.
It promises that the tracking error stays under a limit.
The tracking error is how far the drone is from where it should be.
That limit is the estimator's error, plus half the wind speed, plus 0.2 m.
Now say the estimated wind reaches 3.1 m/s.
The PID assumption is broken, so the switching rules stop choosing PID.
The MPC contract allows wind up to 15 m/s.
Take a wind of 4 m/s and an estimator error of 0.5 m.
Then the MPC promise works out to 0.85 m.
That is under 3 m, so the rules ask for MPC instead.

## Using it with PX4 and ROS 2

The package `contract_uav_control/` has one ROS 2 program, called a node.
While it runs, it repeats these steps 50 times per second by default:

1. It reads the drone's position, tilt, GPS and battery from PX4.
2. It runs the core controller on them.
3. It sends PX4 a target tilt and thrust.
4. It sends its own state on the ROS 2 topic `/contract_uav/state`, so you can watch it. A topic is a named message channel.

The node arms the drone (makes it ready to spin the motors) by itself only if you set `auto_arm: true` in `contract_uav_control/config/params.yaml`.
The default is `false`.

You cannot run this part without ROS 2.
You also need PX4 and the `px4_msgs` package, which holds the PX4 message types for ROS 2.
The steps are in [`docs/TESTING_MANUAL.md`](docs/TESTING_MANUAL.md) and [`docs/HARDWARE_TESTING.md`](docs/HARDWARE_TESTING.md).
The manual starts with SITL, where PX4 runs on your computer with a simulated drone.
Only then does it move on to the real drone.
For example, this is how the node starts (needs ROS 2):

```bash
ros2 launch contract_uav_control contract_control.launch.py
```

CI does not install ROS 2.
It only checks the node's Python code with fake ROS modules.
This bridge has not been tested on a real drone in this repository's CI.

## Known problems (being fixed next)

- The controllers use the wrong sign for thrust. They add thrust to go down, so a real drone would fall. The demos still look fine, because their simple drone models have the same wrong sign.
- The tilt direction ignores the drone's heading (the way its nose points). The drone only moves the right way when it faces north.
- A switch between controllers can be delayed by the 5-second wait, even in an emergency.
- Memory use grows during long runs, and each step gets slower.
- Without GPS, the position estimate drifts badly, because it keeps the last known speed.
- The pre-flight check in the ROS 2 bridge does not read real drone data. It uses fixed values, so it always passes.
- `demos/demo_racing_live.py` crashes with NumPy 2. It works with NumPy 1.26.
- `demos/simulation_demo.py` stops at its pre-flight check, so it never flies.
- With Pacti installed, the planner can be very slow. The planner demo took about 80 seconds to simulate 14 seconds of flight.
- The numbers in `docs/RESULTS.md` were measured with the old simulation, before these fixes.

## Results

The benchmark tables are in [`docs/RESULTS.md`](docs/RESULTS.md).
Read the note at its top first.
It says that the numbers come from the old simulation, and that they will be measured again after the fixes.

## Testing and contributing

Run the tests with `make test`, as shown in "Run something".
They run the self-tests built into the core modules.
They check that two benchmarks still print the same tables as the saved copies in `tests/golden/`.
They also run small tests of single parts, including a check of the ROS 2 node with fake ROS modules.
This took about 1 minute on our test computer.

With Pacti installed, `make test-pacti` checks that the controller benchmark still gives the numbers in section 1 of `docs/RESULTS.md`:

```bash
python -m pip install -r requirements/pacti.txt
make test-pacti
```

For more on how the code is laid out and how to add a controller, see [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).
Some parts of that guide are out of date.

## License

This project uses the MIT license. See [`LICENSE`](LICENSE).
Anyone may use, copy, change and share the code, even for commercial work.
They must keep the copyright line and the license text with it.
The code comes with no warranty.

## Contact

Aswatth Sunil, University of Michigan (EECS), aswatths@umich.edu, GitHub [@AshMightFixIt](https://github.com/AshMightFixIt).
Advisor: Prof. Iñigo Incer.

## Glossary

| Word | Meaning |
|---|---|
| Assumption | The part of a contract that says what must be true for a part to work. |
| Attitude | Which way the drone points: its tilt to the side (roll), its tilt forward or back (pitch) and its heading (yaw). |
| Benchmark | A script that runs the same test with each controller and compares the results. |
| CBF | Control barrier function. A math idea for keeping a system inside safe limits. The safety filter here is named after it, but it uses simple fixed rules. |
| CI | Continuous integration. The tests run by themselves on GitHub for every change. |
| Contract | Written rules for one part of the system: its assumptions and its guarantees. |
| Controller | Code that decides the thrust and the tilt, so the drone reaches its target. |
| Cooldown | The 5-second wait between two controller switches. |
| Demo | A script that shows one feature of the system. |
| EKF | Extended Kalman filter. A common way to turn noisy sensor readings into one best guess. The one here is a simplified version. |
| Estimator | The part that guesses where the drone is and how it moves. Here it is the EKF. |
| Flight mode | What the drone is doing: TRACK, HOVER, LAND or EMERGENCY. |
| GPS | Satellite positioning. It tells the drone where it is. |
| Guarantee | The part of a contract that says what a part promises when its assumptions hold. |
| H-infinity | The name of a robust control method. Here, the fallback controller: its contract allows the widest range of conditions. |
| HIL | Hardware in the loop (the docs also write HITL). PX4 runs on the real flight board, but the drone and the world are simulated. |
| IMU | Inertial measurement unit. A sensor that measures tilt and turn rates. |
| MPC | Model predictive control. A controller that predicts a few steps ahead and picks the best moves. |
| NED | North, East, Down. The direction rule for positions: x points north, y east and z down. So z = -5 means 5 m above the start. |
| Node | One program in ROS 2. |
| Pacti | A Python library for working with contracts. It is optional here. |
| PID | Proportional, integral, derivative. A simple controller that reacts to the error now, the error built up over time, and how fast the error changes. |
| Planner | An optional part that looks ahead and can ask for a stronger controller. It needs Pacti. |
| PX4 | Open-source autopilot software that runs on many real drones. |
| Quadrotor | A drone with four rotors. |
| ROS 2 | Robot Operating System 2. A system that lets programs on a robot send messages to each other. |
| Safety filter | The last part before the commands leave. It changes a command that would break a hard limit. Here it is the CBF. |
| Simulation | A program that pretends to be the drone, so you can test without flying. |
| SITL | Software in the loop. PX4 runs on your computer, together with a simulated drone. |
| Supervisor | The part that picks the flight mode, the target point and, in some modes, the controller. |
| Thrust | The upward push of the rotors. |
| UAV | Unmanned aerial vehicle: a drone. |
| Virtual environment | A private set of Python packages for one project. |
| Waypoint | A point in the air that the drone must reach. |
| Yaw (heading) | The direction the drone's nose points, such as north or east. |
