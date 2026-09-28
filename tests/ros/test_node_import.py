"""The ROS 2 node imports the core from the installed contract_uav_core package.

There is no ROS here. A child process replaces rclpy, px4_msgs and
contract_uav_msgs with stub modules injected into ``sys.modules`` (no stub
files on disk, no ``sys.path`` edits). contract_uav_control is not installed:
the child loads it by file path with ``importlib.util.spec_from_file_location``.
The child runs from pytest's temp dir with CONTRACT_UAV_ROOT and PYTHONPATH
unset, so only the installed contract_uav_core can satisfy the node's import.
Building with colcon is not covered.
"""
from __future__ import annotations

import json
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


def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


def placeholder(name):
    return type(name, (), {})


class Node:  # ContractControllerNode derives from rclpy.node.Node
    pass


stub("rclpy")
stub("rclpy.node", Node=Node)
stub("rclpy.qos", **{n: placeholder(n) for n in (
    "QoSProfile", "QoSReliabilityPolicy", "QoSHistoryPolicy", "QoSDurabilityPolicy")})
stub("px4_msgs")
stub("px4_msgs.msg", **{n: placeholder(n) for n in (
    "OffboardControlMode", "VehicleAttitudeSetpoint", "VehicleCommand", "VehicleLocalPosition",
    "VehicleAttitude", "VehicleAngularVelocity", "VehicleStatus", "SensorGps", "BatteryStatus")})
stub("contract_uav_msgs")
stub("contract_uav_msgs.msg", ContractState=placeholder("ContractState"))

pkg_dir = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "contract_uav_control", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
package = importlib.util.module_from_spec(spec)
sys.modules["contract_uav_control"] = package
spec.loader.exec_module(package)

path_before = list(sys.path)
node = importlib.import_module("contract_uav_control.controller_node")
cls, location = node._import_core_controller()
print(json.dumps({
    "node_file": node.__file__,
    "class": cls.__module__ + "." + cls.__qualname__,
    "class_file": sys.modules[cls.__module__].__file__,
    "installed_core_file": importlib.util.find_spec("contract_uav_core.core").origin,
    "location": location,
    "sys_path_changed": sys.path != path_before,
}))
'''


def test_node_resolves_core_from_installed_package(core_package, tmp_path, run_python, monkeypatch):
    monkeypatch.delenv("CONTRACT_UAV_ROOT", raising=False)
    proc = run_python(["-c", CHILD, NODE_PACKAGE_DIR], cwd=tmp_path, timeout=300)
    assert proc.returncode == 0, (
        f"--- stdout (tail) ---\n{proc.stdout[-3000:]}\n--- stderr (tail) ---\n{proc.stderr[-3000:]}"
    )
    # The core prints a pacti warning on import when pacti is absent; the JSON is the last line.
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    assert Path(info["node_file"]).samefile(NODE_PACKAGE_DIR / "controller_node.py")
    assert info["class"] == "contract_uav_core.core.AdaptiveDroneController"
    assert Path(info["class_file"]).samefile(info["installed_core_file"])
    assert Path(info["class_file"]).samefile(core_package)
    assert Path(info["location"]).samefile(core_package.parent)
    assert info["sys_path_changed"] is False
