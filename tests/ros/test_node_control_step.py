"""The ROS 2 node drives control_step with the PX4 time; the legacy call still works.

Same mechanism as test_node_import.py: no ROS here. A child process replaces
rclpy, px4_msgs and contract_uav_msgs with stub modules injected into
``sys.modules`` (no stub files on disk, no ``sys.path`` edits) and loads
contract_uav_control by file path. The stubs are richer than there: the Node
stub stores parameters and records publications, so ContractControllerNode can
be built and its timer callback ``_control_loop`` called by hand with stub PX4
messages.

Both call styles against the installed core:
- the node's: ``control_step(raw_sensors, t=<VehicleLocalPosition.timestamp in s>)``;
- the scripts': ``control_step(raw_sensors)`` on a controller from the same import.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_PACKAGE_DIR = REPO_ROOT / "contract_uav_control" / "contract_uav_control"

CHILD = r'''
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


def message(name, **consts):
    """A message class: instances accept any attribute; class constants as given."""
    return type(name, (), dict(consts))


class Publisher:
    def __init__(self, topic):
        self.topic = topic
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


class Logger:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def error(self, *a, **k): pass


class Clock:
    def now(self):
        return types.SimpleNamespace(nanoseconds=123_000_000_000, to_msg=lambda: "stamp")


class Node:  # the parts of rclpy.node.Node that ContractControllerNode uses
    OVERRIDES = {"use_horizon_planner": False}

    def __init__(self, name):
        self._params = {}
        self.publishers = {}

    def declare_parameter(self, name, value):
        self._params[name] = self.OVERRIDES.get(name, value)

    def get_parameter(self, name):
        return types.SimpleNamespace(value=self._params[name])

    def get_logger(self):
        return Logger()

    def get_clock(self):
        return Clock()

    def create_subscription(self, *args):
        return None

    def create_publisher(self, msg_type, topic, qos):
        self.publishers[topic] = Publisher(topic)
        return self.publishers[topic]

    def create_timer(self, period, callback):
        return None


class ContractState:
    def __init__(self):
        self.header = types.SimpleNamespace()


stub("rclpy")
stub("rclpy.node", Node=Node)
stub("rclpy.qos",
     QoSProfile=lambda **kwargs: types.SimpleNamespace(**kwargs),
     QoSReliabilityPolicy=types.SimpleNamespace(BEST_EFFORT="best_effort"),
     QoSHistoryPolicy=types.SimpleNamespace(KEEP_LAST="keep_last"),
     QoSDurabilityPolicy=types.SimpleNamespace(TRANSIENT_LOCAL="transient_local"))
stub("px4_msgs")
stub("px4_msgs.msg", **{n: message(n) for n in (
    "OffboardControlMode", "VehicleAttitudeSetpoint", "VehicleLocalPosition",
    "VehicleAttitude", "VehicleAngularVelocity", "VehicleStatus", "SensorGps", "BatteryStatus")},
    VehicleCommand=message("VehicleCommand", VEHICLE_CMD_DO_SET_MODE=176,
                           VEHICLE_CMD_COMPONENT_ARM_DISARM=400))
stub("contract_uav_msgs")
stub("contract_uav_msgs.msg", ContractState=ContractState)

pkg_dir = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "contract_uav_control", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
package = importlib.util.module_from_spec(spec)
sys.modules["contract_uav_control"] = package
spec.loader.exec_module(package)
node_module = importlib.import_module("contract_uav_control.controller_node")

node = node_module.ContractControllerNode()
ctrl = node.ctrl
calls = []
original_step = ctrl.control_step


def recording_step(*args, **kwargs):
    calls.append({"nargs": len(args), "t": kwargs.get("t")})
    control, telemetry = original_step(*args, **kwargs)
    calls[-1]["telemetry_time"] = telemetry["time"]
    calls[-1]["timing"] = telemetry["timing"]
    return control, telemetry


ctrl.control_step = recording_step

# PX4 timestamps in microseconds: nominal 20 ms steps, then a repeated message
# (the timer ticked before a new VehicleLocalPosition arrived) and a late one.
stamps_us = [5_000_000, 5_020_000, 5_040_000, 5_040_000, 5_100_000, 5_120_000]
node.attitude = types.SimpleNamespace(q=[1.0, 0.0, 0.0, 0.0])
for k, stamp in enumerate(stamps_us):
    node.local_pos = types.SimpleNamespace(
        timestamp=stamp, xy_valid=True, z_valid=True,
        x=0.01 * k, y=0.0, z=-5.0, vx=0.5, vy=0.0, vz=0.0)
    node._control_loop()

# The scripts' call style, on a controller from the same import path.
cls, _ = node_module._import_core_controller()
legacy = cls(dt=node.dt, use_horizon_planner=False)
target = np.array([0.0, 0.0, -5.0])
ok, msg = legacy.pre_flight_check(
    {"gps_satellites": 12, "gps_hdop": 1.0, "imu_temperature": 25.0, "imu_calibrated": True,
     "battery_voltage": 12.0, "motor_temperature": 25.0, "wind_speed": 0.0, "disturbance": 0.0,
     "computation_time": 0.01}, {"target_position": target, "waypoints": [target]})
assert ok, msg
legacy.start_mission()
raw = node_module.build_raw_sensors(node.local_pos, node.attitude, None, None, None)
legacy_times = []
for _ in range(5):
    _, tel = legacy.control_step(raw)
    legacy_times.append([tel["time"], tel["timing"]["time_source"]])

print(json.dumps({
    "dt": node.dt,
    "calls": calls,
    "ctrl_time": ctrl.time,
    "attitude_setpoints": len(node.publishers["/fmu/in/vehicle_attitude_setpoint"].sent),
    "state_msgs": len(node.publishers["/contract_uav/state"].sent),
    "controller_time_field": node.publishers["/contract_uav/state"].sent[-1].controller_time,
    "legacy_times": legacy_times,
    "legacy_final_time": legacy.time,
}))
'''


def test_node_passes_px4_time_and_legacy_call_still_works(core_package, tmp_path, run_python, monkeypatch):
    for name in [k for k in os.environ if k.startswith("CONTRACT_UAV")]:
        monkeypatch.delenv(name)
    proc = run_python(["-c", CHILD, NODE_PACKAGE_DIR], cwd=tmp_path, timeout=300)
    assert proc.returncode == 0, (
        f"--- stdout (tail) ---\n{proc.stdout[-3000:]}\n--- stderr (tail) ---\n{proc.stderr[-3000:]}"
    )
    # The core prints a pacti warning on import when pacti is absent; the JSON is the last line.
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    dt = info["dt"]
    assert dt == 1.0 / 50.0  # the node's default control_frequency

    # The node's style: one positional argument plus t = timestamp in seconds.
    stamps_s = [5.0, 5.02, 5.04, 5.04, 5.1, 5.12]
    calls = info["calls"]
    assert [c["nargs"] for c in calls] == [1] * len(stamps_s)
    assert [c["t"] for c in calls] == stamps_s
    assert [c["telemetry_time"] for c in calls] == stamps_s
    assert info["ctrl_time"] == stamps_s[-1]
    assert info["controller_time_field"] == stamps_s[-1]
    assert info["attitude_setpoints"] == info["state_msgs"] == len(stamps_s)
    last = calls[-1]["timing"]
    assert last["time_source"] == "caller"
    assert last["step_count"] == len(stamps_s)
    assert last["nonincreasing_count"] == 1   # the repeated 5.04
    assert last["overrun_count"] == 1         # 5.04 -> 5.1 is 3 x dt
    assert calls[4]["timing"]["dt_clamped"] == 2.0 * dt

    # The scripts' style: no t, accumulated clock.
    expected, t = [], 0.0
    for _ in range(5):
        expected.append([t, "accumulated"])
        t += dt
    assert info["legacy_times"] == expected
    assert info["legacy_final_time"] == t
