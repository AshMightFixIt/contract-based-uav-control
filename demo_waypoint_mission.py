"""
Demonstration: Multi-Waypoint Mission with Three-Tier Controller Switching

Shows:
1. Loading a multi-waypoint mission
2. Autonomous waypoint tracking
3. Automatic waypoint advancement
4. Three-tier contract-based controller switching: PID → MPC → H-inf → MPC → PID
5. Mission completion with hover at final waypoint
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.ERROR, format='%(message)s')

from contract_uav_core.core import AdaptiveDroneController

try:
    from contract_uav_core.viz.live_visualizer import LiveVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


class DroneSimulation:
    """Simple drone dynamics for waypoint mission testing"""

    def __init__(self):
        self.position = np.array([0.0, 0.0, -5.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.attitude = np.array([0.0, 0.0, 0.0])
        self.rates = np.array([0.0, 0.0, 0.0])
        self.wind = np.array([0.0, 0.0, 0.0])
        self.gps_available = True
        self.dt = 0.02

    def set_wind(self, wind_velocity: np.ndarray):
        self.wind = wind_velocity

    def update(self, control: dict):
        """Update dynamics - velocity-based model for stable simulation"""
        thrust = control.get('thrust', 0.5)
        desired_att = control.get('desired_attitude', np.zeros(3))

        # Vertical dynamics - thrust=0.5 is hover
        # Reduced gain for gentler altitude control
        vertical_accel = (thrust - 0.5) * 8.0
        self.velocity[2] += vertical_accel * self.dt

        # Horizontal dynamics - attitude commands velocity
        # Small angle approximation: tilt angle -> horizontal acceleration
        # Reduced acceleration for gentler movement
        max_accel = 2.0  # m/s^2 - reduced from 3.0
        horizontal_accel = np.array([
            np.clip(-np.sin(desired_att[1]) * 8.0, -max_accel, max_accel),
            np.clip(np.sin(desired_att[0]) * 8.0, -max_accel, max_accel),
            0.0
        ])
        self.velocity[0:2] += horizontal_accel[0:2] * self.dt

        # Wind affects velocity (drag-like disturbance)
        self.velocity[0:2] += self.wind[0:2] * 0.06 * self.dt

        # Limit maximum velocity
        max_vel = 3.0  # m/s - reduced from 5.0 for gentler flight
        vel_magnitude = np.linalg.norm(self.velocity[0:2])
        if vel_magnitude > max_vel:
            self.velocity[0:2] *= max_vel / vel_magnitude

        # Limit vertical velocity too
        self.velocity[2] = np.clip(self.velocity[2], -2.0, 2.0)

        # Integrate position
        self.position += self.velocity * self.dt

        # Clamp altitude to prevent going underground (NED: negative Z = altitude)
        if self.position[2] > -0.5:
            self.position[2] = -0.5
            self.velocity[2] = min(self.velocity[2], 0.0)  # Stop descent

        # Attitude follows command with delay (slower response)
        self.attitude = 0.95 * self.attitude + 0.05 * desired_att

        # Velocity damping (drag model)
        self.velocity *= 0.97
        self.rates *= 0.85

    def get_sensor_data(self):
        # Minimal noise for clean waypoint tracking demo
        noise_pos = np.random.randn(3) * 0.01
        noise_vel = np.random.randn(3) * 0.01

        return {
            'gps': {
                'position': self.position + noise_pos,
                'velocity': self.velocity + noise_vel,
                'valid': self.gps_available,
                'satellites': 12 if self.gps_available else 2,
                'hdop': 0.8 if self.gps_available else 10.0
            },
            'imu': {
                'attitude': self.attitude + np.random.randn(3) * 0.001,
                'rates': self.rates + np.random.randn(3) * 0.001,
                'valid': True,
                'calibrated': True,
                'temperature': 25.0
            },
            'battery': {'voltage': 12.4},
            'motors': {'temperature': 30.0},
            # Wind sensor (e.g., pitot tube or anemometer)
            'wind': {
                'speed': np.linalg.norm(self.wind[0:2]),
                'direction': np.arctan2(self.wind[1], self.wind[0]) if np.linalg.norm(self.wind[0:2]) > 0.1 else 0.0
            }
        }


def run_waypoint_mission():
    print("=" * 70)
    print("MULTI-WAYPOINT MISSION DEMONSTRATION")
    print("=" * 70)

    # Initialize with horizon planner enabled
    controller = AdaptiveDroneController(dt=0.02, use_horizon_planner=True)
    drone = DroneSimulation()

    # Define a square mission pattern at varying altitudes
    waypoints = [
        np.array([10.0, 0.0, -5.0]),    # WP1: East
        np.array([10.0, 10.0, -8.0]),   # WP2: Northeast, higher
        np.array([0.0, 10.0, -6.0]),    # WP3: North, mid altitude
        np.array([0.0, 0.0, -5.0]),     # WP4: Return to start
    ]

    # Optional live visualizer
    viz = None
    if HAS_VISUALIZER:
        try:
            viz = LiveVisualizer(waypoints)
            print("\n[LIVE VISUALIZER] Window opened (drag=orbit, scroll=zoom, F=follow, Esc=exit)")
        except Exception as e:
            print(f"\n[LIVE VISUALIZER] Could not start: {e}")
            viz = None

    print(f"\nMission: Square pattern with {len(waypoints)} waypoints")
    print("Waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i+1}: [{wp[0]:5.1f}, {wp[1]:5.1f}, {wp[2]:5.1f}]")

    # Pre-flight check
    print("\n[1] PRE-FLIGHT CHECK")
    print("-" * 70)

    initial_conditions = {
        'gps_satellites': 12.0,
        'gps_hdop': 0.8,
        'imu_temperature': 25.0,
        'imu_calibrated': 1.0,
        'battery_voltage': 12.4,
        'motor_temperature': 30.0,
        'wind_speed': 0.5,
        'disturbance': 0.2,
    }

    mission = {
        'target_position': waypoints[0],
        'waypoints': waypoints
    }

    feasible, msg = controller.pre_flight_check(initial_conditions, mission)
    print(f"Result: {msg}")

    if not feasible:
        print("Mission aborted!")
        return

    controller.start_mission()

    # Wind schedule — designed to demonstrate three-tier graduated degradation:
    #   PID → MPC → H-inf → MPC → PID
    wind_schedule = [
        (0.0,  25.0, np.array([0.5, 0.0, 0.0])),    # Calm       → PID
        (25.0, 45.0, np.array([4.5, 2.0, 0.0])),    # Moderate   → MPC
        (45.0, 60.0, np.array([9.0, 4.0, 0.0])),    # Strong     → H-inf
        (60.0, 80.0, np.array([5.0, 2.0, 0.0])),    # Easing     → MPC
        (80.0, 120.0, np.array([1.0, 0.5, 0.0])),   # Calm       → PID
    ]

    # Data recording
    time_history = []
    position_history = []
    velocity_history = []
    controller_history = []
    waypoint_idx_history = []
    target_history = []

    print("\n[2] MISSION EXECUTION")
    print("-" * 70)
    print(f"{'Time':>6} | {'WP':>3} | {'Dist':>6} | {'Ctrl':>5} | {'Position (X, Y, Z)':^25} | Event")
    print("-" * 70)

    last_wp_idx = -1
    mission_complete = False
    mission_complete_time = None

    # Run for up to 120 seconds (longer mission)
    for step in range(6000):
        t = step * 0.02
        time_history.append(t)

        # Get current wind
        current_wind = np.array([0.5, 0.0, 0.0])
        for t_start, t_end, wind in wind_schedule:
            if t_start <= t < t_end:
                current_wind = wind
                break

        drone.set_wind(current_wind)

        # Get sensor data and run control
        sensors = drone.get_sensor_data()
        control, telemetry = controller.control_step(sensors)
        drone.update(control)

        # Record data
        position_history.append(drone.position.copy())
        velocity_history.append(drone.velocity.copy())
        controller_history.append(telemetry['active_controller'])

        # Get waypoint info from supervisor
        sup_status = controller.supervisor.get_status()
        wp_idx = sup_status['waypoint_idx']
        waypoint_idx_history.append(wp_idx)

        # Update live visualizer
        if viz is not None:
            viz_state = {
                'time': t,
                'position': telemetry['position'],
                'velocity': drone.velocity.copy(),
                'attitude': drone.attitude.copy(),
                'active_controller': telemetry['active_controller'],
                'flight_mode': telemetry['flight_mode'],
                'cbf_intervened': telemetry['cbf_intervened'],
                'wind_speed': np.linalg.norm(current_wind[0:2]),
                'waypoint_idx': wp_idx,
            }
            if not viz.update(viz_state):
                print("\n[LIVE VISUALIZER] Window closed by user")
                break

        if wp_idx < len(waypoints):
            target_history.append(waypoints[wp_idx].copy())
        else:
            target_history.append(waypoints[-1].copy())

        # Calculate distance to current waypoint (using telemetry/EKF position, not true position)
        ekf_pos = telemetry['position']
        if wp_idx < len(waypoints):
            dist_to_wp = np.linalg.norm(waypoints[wp_idx] - ekf_pos)
        else:
            dist_to_wp = 0.0

        # Print status on waypoint change or every 2 seconds
        event = ""
        if wp_idx != last_wp_idx:
            if wp_idx > last_wp_idx and last_wp_idx >= 0:
                event = f"WP{last_wp_idx+1} REACHED!"
            last_wp_idx = wp_idx

        if telemetry['flight_mode'] == 'hover' and wp_idx >= len(waypoints) - 1 and not mission_complete:
            event = "MISSION COMPLETE"
            mission_complete = True
            mission_complete_time = t

        if step % 100 == 0 or event:
            # Show EKF position (what controller sees) - this matches the distance calculation
            pos = ekf_pos
            print(f"{t:6.2f} | {wp_idx+1:>3} | {dist_to_wp:6.2f} | {telemetry['active_controller']:>5} | "
                  f"[{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] | {event}")

        # End 5 seconds after mission complete
        if mission_complete and mission_complete_time and t > mission_complete_time + 5:
            break

    if viz is not None:
        viz.close()

    controller.stop_mission()

    # Convert to arrays
    position_history = np.array(position_history)
    velocity_history = np.array(velocity_history)
    target_history = np.array(target_history)
    time_history = np.array(time_history)

    # Summary
    print("\n" + "=" * 70)
    print("[3] MISSION SUMMARY")
    print("=" * 70)

    print(f"\nFlight duration: {time_history[-1]:.1f} seconds")
    print(f"Waypoints completed: {max(waypoint_idx_history)}/{len(waypoints)}")
    print(f"Final position: [{drone.position[0]:.2f}, {drone.position[1]:.2f}, {drone.position[2]:.2f}]")

    # Controller usage
    total_time = time_history[-1]
    pid_time = sum(1 for c in controller_history if c == 'PID') * 0.02
    mpc_time = sum(1 for c in controller_history if c == 'MPC') * 0.02
    hinf_time = sum(1 for c in controller_history if c == 'Hinf') * 0.02
    print(f"\nController usage:")
    print(f"  PID:   {pid_time:6.1f}s ({100*pid_time/total_time:5.1f}%)")
    print(f"  MPC:   {mpc_time:6.1f}s ({100*mpc_time/total_time:5.1f}%)")
    print(f"  H-inf: {hinf_time:6.1f}s ({100*hinf_time/total_time:5.1f}%)")

    # Plot results - comprehensive time series
    print("\nGenerating plots...")

    fig = plt.figure(figsize=(16, 14))

    # 1. 3D trajectory (top left)
    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    ax1.plot(position_history[:, 0], position_history[:, 1], -position_history[:, 2],
             'b-', linewidth=1.5, label='Trajectory')
    wp_array = np.array(waypoints)
    ax1.scatter(wp_array[:, 0], wp_array[:, 1], -wp_array[:, 2],
                c='red', s=100, marker='^', label='Waypoints')
    for i, wp in enumerate(waypoints):
        ax1.text(wp[0], wp[1], -wp[2] + 0.5, f'WP{i+1}', fontsize=9)
    ax1.scatter([0], [0], [5], c='green', s=150, marker='o', label='Start')
    ax1.scatter([position_history[-1, 0]], [position_history[-1, 1]], [-position_history[-1, 2]],
                c='purple', s=150, marker='s', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Flight Trajectory')
    ax1.legend(loc='upper left', fontsize=8)

    # 2. Top-down view (top right)
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(position_history[:, 0], position_history[:, 1], 'b-', linewidth=1.5, label='Trajectory')
    ax2.scatter(wp_array[:, 0], wp_array[:, 1], c='red', s=100, marker='^', label='Waypoints')
    for i, wp in enumerate(waypoints):
        ax2.annotate(f'WP{i+1}', (wp[0], wp[1]), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)
    ax2.scatter([0], [0], c='green', s=150, marker='o', label='Start')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (X-Y Plane)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 3. Position vs time (middle left)
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(time_history, position_history[:, 0], 'b-', linewidth=1.5, label='X')
    ax3.plot(time_history, position_history[:, 1], 'g-', linewidth=1.5, label='Y')
    ax3.plot(time_history, -position_history[:, 2], 'r-', linewidth=1.5, label='Altitude')
    # Mark controller switches
    for i in range(1, len(controller_history)):
        if controller_history[i] != controller_history[i-1]:
            ax3.axvline(x=time_history[i], color='orange', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position vs Time')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Velocity vs time (middle right)
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(time_history, velocity_history[:, 0], 'b-', linewidth=1.5, label='Vx')
    ax4.plot(time_history, velocity_history[:, 1], 'g-', linewidth=1.5, label='Vy')
    ax4.plot(time_history, velocity_history[:, 2], 'r-', linewidth=1.5, label='Vz')
    speed = np.linalg.norm(velocity_history, axis=1)
    ax4.plot(time_history, speed, 'k--', linewidth=1, label='Speed', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity vs Time')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Active Controller (bottom left)
    ax5 = fig.add_subplot(3, 2, 5)
    # Map controller names to numeric: PID=1, MPC=2, H-inf=3
    ctrl_map = {'PID': 1, 'MPC': 2, 'Hinf': 3}
    ctrl_numeric = [ctrl_map.get(c, 3) for c in controller_history]
    ax5.fill_between(time_history, 0, ctrl_numeric, step='post', alpha=0.6,
                     color='steelblue')
    # Color-code segments
    for i in range(len(time_history) - 1):
        color = {'PID': 'blue', 'MPC': 'green', 'Hinf': 'red'}.get(controller_history[i], 'gray')
        ax5.fill_between(time_history[i:i+2], 0, ctrl_numeric[i:i+2],
                         step='post', alpha=0.4, color=color)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Controller')
    ax5.set_title('Active Controller (PID→MPC→H-inf)')
    ax5.set_yticks([1, 2, 3])
    ax5.set_yticklabels(['PID', 'MPC', 'H-inf'])
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0.5, 3.5)

    # 6. Flight Mode & Waypoint Progress (bottom right)
    ax6 = fig.add_subplot(3, 2, 6)
    # Distance to current waypoint
    distances = []
    for i, (pos, wp_idx) in enumerate(zip(position_history, waypoint_idx_history)):
        if wp_idx < len(waypoints):
            distances.append(np.linalg.norm(waypoints[wp_idx] - pos))
        else:
            distances.append(0.0)
    ax6.plot(time_history, distances, 'b-', linewidth=1.5, label='Distance to WP')
    ax6.axhline(y=1.5, color='r', linestyle='--', label='WP tolerance', alpha=0.7)
    # Mark waypoint reaches
    for i in range(1, len(waypoint_idx_history)):
        if waypoint_idx_history[i] != waypoint_idx_history[i-1]:
            ax6.axvline(x=time_history[i], color='g', linestyle='-', alpha=0.7, linewidth=2)
            ax6.text(time_history[i]+0.5, max(distances)*0.85, f'WP{waypoint_idx_history[i-1]+1}',
                    fontsize=9, fontweight='bold')
    # Secondary axis for waypoint index
    ax6_twin = ax6.twinx()
    ax6_twin.plot(time_history, waypoint_idx_history, 'k-', linewidth=1, alpha=0.5, label='WP Index')
    ax6_twin.set_ylabel('Waypoint Index', color='gray')
    ax6_twin.tick_params(axis='y', labelcolor='gray')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Distance (m)')
    ax6.set_title('Waypoint Progress & Flight Mode')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), 'waypoint_mission_results.png')
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    print("\n" + "=" * 70)
    print("WAYPOINT MISSION DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_waypoint_mission()
