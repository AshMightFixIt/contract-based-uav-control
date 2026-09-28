"""PX4 offboard ROS 2 node wrapping the contract-based three-tier adaptive controller.

Pipeline per cycle (default 50 Hz):
    PX4 estimator topics -> raw_sensors dict -> AdaptiveDroneController.control_step()
    -> VehicleAttitudeSetpoint (+ OffboardControlMode heartbeat) -> PX4.

The contract-based stack runs the position/attitude outer loops and the CBF safety
filter; PX4's onboard rate controller + mixer run underneath the attitude setpoint.

Targets DEXI / PX4 over micro-XRCE-DDS (px4_msgs). Validate in PX4 SITL before flight.
"""

import os
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    VehicleAttitudeSetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleAttitude,
    VehicleAngularVelocity,
    VehicleStatus,
    SensorGps,
    BatteryStatus,
)
from contract_uav_msgs.msg import ContractState

from contract_uav_control.px4_conversions import build_raw_sensors, euler_to_quaternion


def _f3(v):
    """numpy/list -> [float,float,float] for ROS message arrays."""
    return [float(v[0]), float(v[1]), float(v[2])]


def _import_core_controller():
    """Import AdaptiveDroneController from the installed contract_uav_core package.

    The core controller is the plain-Python package contract_uav_core, installed
    next to this one (colcon build, or `pip install -e ./contract_uav_core`), so a
    normal import finds it. Returns the class and the directory it was loaded from.
    """
    from contract_uav_core.core import AdaptiveDroneController
    core_file = sys.modules[AdaptiveDroneController.__module__].__file__
    return AdaptiveDroneController, os.path.dirname(os.path.abspath(core_file))


class ContractControllerNode(Node):
    def __init__(self):
        super().__init__('contract_controller_node')

        # ---- parameters ----
        self.declare_parameter('control_frequency', 50.0)
        self.declare_parameter('use_horizon_planner', True)
        self.declare_parameter('takeoff_altitude', 5.0)   # metres above start (NED z = -alt)
        self.declare_parameter('auto_arm', False)         # safety: off by default
        self.declare_parameter('waypoints', [0.0, 0.0, -5.0])  # flat [x,y,z,...] NED

        freq = self.get_parameter('control_frequency').value
        self.dt = 1.0 / float(freq)
        use_planner = self.get_parameter('use_horizon_planner').value
        self.auto_arm = self.get_parameter('auto_arm').value
        self.takeoff_alt = self.get_parameter('takeoff_altitude').value

        # ---- core controller ----
        AdaptiveDroneController, root = _import_core_controller()
        self.get_logger().info(f"Loaded contract-based core from: {root}")
        self.ctrl = AdaptiveDroneController(dt=self.dt, use_horizon_planner=use_planner)

        waypoints = self._parse_waypoints(self.get_parameter('waypoints').value)
        self._run_preflight(waypoints)

        # ---- QoS: PX4 publishes best-effort, keep-last-1 ----
        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---- subscribers (PX4 outputs) ----
        self.local_pos = None
        self.attitude = None
        self.ang_vel = None
        self.sensor_gps = None
        self.battery = None
        self.vehicle_status = None

        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position',
                                 self._cb_local_pos, px4_qos)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude',
                                 self._cb_attitude, px4_qos)
        self.create_subscription(VehicleAngularVelocity, '/fmu/out/vehicle_angular_velocity',
                                 self._cb_ang_vel, px4_qos)
        self.create_subscription(SensorGps, '/fmu/out/vehicle_gps_position',
                                 self._cb_gps, px4_qos)
        self.create_subscription(BatteryStatus, '/fmu/out/battery_status',
                                 self._cb_battery, px4_qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status',
                                 self._cb_status, px4_qos)

        # ---- publishers (PX4 inputs) ----
        self.pub_offboard = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.pub_att_sp = self.create_publisher(
            VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', px4_qos)
        self.pub_cmd = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', px4_qos)

        # ---- telemetry publisher (monitoring + tuning) ----
        # Reliable QoS so PlotJuggler / ros2 bag capture every sample.
        self.pub_state = self.create_publisher(
            ContractState, '/contract_uav/state', 10)

        self._setpoint_counter = 0
        self.timer = self.create_timer(self.dt, self._control_loop)
        self.get_logger().info(
            f"Contract controller node up @ {freq:.0f} Hz "
            f"(auto_arm={self.auto_arm}). Streaming offboard heartbeat.")

    # ---------- helpers ----------
    def _parse_waypoints(self, flat):
        arr = list(flat)
        if len(arr) % 3 != 0:
            self.get_logger().warn("waypoints length not divisible by 3; using default.")
            return [np.array([0.0, 0.0, -self.takeoff_alt])]
        return [np.array(arr[i:i + 3]) for i in range(0, len(arr), 3)]

    def _run_preflight(self, waypoints):
        initial_conditions = {
            'gps_satellites': 12, 'gps_hdop': 1.0,
            'imu_temperature': 25.0, 'imu_calibrated': True,
            'battery_voltage': 12.0, 'motor_temperature': 25.0,
            'wind_speed': 0.0, 'disturbance': 0.0, 'computation_time': 0.01,
        }
        mission = {'target_position': waypoints[-1], 'waypoints': waypoints}
        feasible, msg = self.ctrl.pre_flight_check(initial_conditions, mission)
        if feasible:
            self.ctrl.start_mission()
            self.get_logger().info(f"Pre-flight OK: {msg}")
        else:
            self.get_logger().error(f"Pre-flight FAILED: {msg}")

    def _now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    # ---------- subscription callbacks ----------
    def _cb_local_pos(self, msg): self.local_pos = msg
    def _cb_attitude(self, msg): self.attitude = msg
    def _cb_ang_vel(self, msg): self.ang_vel = msg
    def _cb_gps(self, msg): self.sensor_gps = msg
    def _cb_battery(self, msg): self.battery = msg
    def _cb_status(self, msg): self.vehicle_status = msg

    # ---------- PX4 commands ----------
    def _publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = self._now_us()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.pub_cmd.publish(msg)

    def _engage_offboard_and_arm(self):
        # 1 = custom main mode, 6 = PX4 OFFBOARD sub-mode
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("Sent OFFBOARD + ARM commands.")

    def _publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.timestamp = self._now_us()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True   # we command attitude + collective thrust
        msg.body_rate = False
        self.pub_offboard.publish(msg)

    def _publish_attitude_setpoint(self, control):
        att = control.get('desired_attitude', np.zeros(3))
        thrust = float(np.clip(control.get('thrust', 0.0), 0.0, 1.0))

        msg = VehicleAttitudeSetpoint()
        msg.timestamp = self._now_us()
        q = euler_to_quaternion(att[0], att[1], att[2])
        msg.q_d = q
        # multicopter collective acts along body -Z (up)
        msg.thrust_body = [0.0, 0.0, -thrust]
        self.pub_att_sp.publish(msg)

    def _publish_state(self, control, telemetry):
        msg = ContractState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ned'
        msg.controller_time = float(telemetry.get('time', 0.0))

        msg.flight_mode = str(telemetry.get('flight_mode', ''))
        msg.active_controller = str(telemetry.get('active_controller', ''))
        msg.cbf_intervened = bool(telemetry.get('cbf_intervened', False))
        msg.cbf_intervention_count = int(
            telemetry.get('cbf_stats', {}).get('intervention_count', 0))

        pos = telemetry.get('position', np.zeros(3))
        vel = telemetry.get('velocity', np.zeros(3))
        att = telemetry.get('attitude', np.zeros(3))
        msg.position = _f3(pos)
        msg.velocity = _f3(vel)
        msg.attitude = _f3(att)

        sp = telemetry.get('setpoint', {}).get('position', np.zeros(3))
        pos_err = np.asarray(sp) - np.asarray(pos)
        msg.setpoint_position = _f3(sp)
        msg.position_error = _f3(pos_err)
        msg.position_error_norm = float(np.linalg.norm(pos_err))

        des_att = control.get('desired_attitude', np.zeros(3))
        msg.cmd_thrust = float(control.get('thrust', 0.0))
        msg.cmd_desired_attitude = _f3(des_att)
        msg.cmd_desired_rates = _f3(control.get('desired_rates', np.zeros(3)))
        msg.cmd_torques = _f3(control.get('torques', np.zeros(3)))
        msg.attitude_error = _f3(np.asarray(des_att) - np.asarray(att))

        planner = telemetry.get('planner', {})
        msg.planner_active = bool(planner.get('active', False))
        msg.planner_emergency = bool(planner.get('emergency', False))
        msg.planner_safety_margin = float(planner.get('safety_margin', -1.0))
        msg.planner_horizon = int(planner.get('horizon', 0))

        cs = telemetry.get('contract_status', {})
        msg.contract_sensors = self._status_str(cs.get('sensors'))
        msg.contract_estimator = self._status_str(cs.get('estimator'))
        msg.contract_controller = self._status_str(cs.get('controller'))
        msg.contract_actuators = self._status_str(cs.get('actuators'))

        self.pub_state.publish(msg)

    @staticmethod
    def _status_str(status):
        """ContractStatus enum -> its string value (robust to plain strings)."""
        if status is None:
            return 'unknown'
        return getattr(status, 'value', str(status))

    # ---------- main loop ----------
    def _control_loop(self):
        # Offboard heartbeat must stream >2 Hz before/while armed.
        self._publish_offboard_heartbeat()

        if self.local_pos is None or self.attitude is None:
            return  # wait for estimator before commanding

        raw_sensors = build_raw_sensors(
            self.local_pos, self.attitude, self.ang_vel,
            self.sensor_gps, self.battery)

        control, telemetry = self.ctrl.control_step(raw_sensors)
        self._publish_attitude_setpoint(control)
        self._publish_state(control, telemetry)

        # Arm + switch to offboard after enough setpoints have streamed.
        if self._setpoint_counter == 20 and self.auto_arm:
            self._engage_offboard_and_arm()
        self._setpoint_counter += 1

        if self._setpoint_counter % 50 == 0:
            self.get_logger().info(
                f"[{telemetry.get('flight_mode','?')}] "
                f"ctrl={telemetry.get('active_controller','?')} "
                f"cbf={telemetry.get('cbf_intervened', False)} "
                f"thrust={control.get('thrust', 0.0):.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = ContractControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
