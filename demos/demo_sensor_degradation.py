"""
Demonstration: Sensor Degradation Testing

Shows contract-based graceful degradation under sensor faults:
1. GPS degradation (satellite loss, HDOP increase)
2. GPS recovery
3. IMU degradation (noise, temperature drift, bias)
4. Combined GPS + IMU failure (dead reckoning)
5. Intermittent GPS recovery
6. Full system recovery

Validates that contracts, EKF fusion modes, controller switching,
and CBF safety work together to maintain safe flight.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import logging

logging.basicConfig(level=logging.ERROR, format='%(message)s')

from contract_uav_core.core import AdaptiveDroneController
from contract_uav_core.control.supervisor import FlightMode
from contract_uav_core.sim.sensor_faults import SensorFaultInjector, create_standard_degradation_schedule

try:
    from contract_uav_core.viz.live_visualizer import LiveVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


class DroneSimulation:
    """Simple drone dynamics for testing (identical to demo_waypoint_mission.py)"""

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
        thrust = control.get('thrust', 0.5)
        desired_att = control.get('desired_attitude', np.zeros(3))

        vertical_accel = (thrust - 0.5) * 8.0
        self.velocity[2] += vertical_accel * self.dt

        max_accel = 2.0
        horizontal_accel = np.array([
            np.clip(-np.sin(desired_att[1]) * 8.0, -max_accel, max_accel),
            np.clip(np.sin(desired_att[0]) * 8.0, -max_accel, max_accel),
            0.0
        ])
        self.velocity[0:2] += horizontal_accel[0:2] * self.dt
        self.velocity[0:2] += self.wind[0:2] * 0.06 * self.dt

        max_vel = 3.0
        vel_magnitude = np.linalg.norm(self.velocity[0:2])
        if vel_magnitude > max_vel:
            self.velocity[0:2] *= max_vel / vel_magnitude

        self.velocity[2] = np.clip(self.velocity[2], -2.0, 2.0)
        self.position += self.velocity * self.dt

        if self.position[2] > -0.5:
            self.position[2] = -0.5
            self.velocity[2] = min(self.velocity[2], 0.0)

        self.attitude = 0.95 * self.attitude + 0.05 * desired_att
        self.velocity *= 0.97
        self.rates *= 0.85

    def get_sensor_data(self):
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
            'wind': {
                'speed': np.linalg.norm(self.wind[0:2]),
                'direction': np.arctan2(self.wind[1], self.wind[0]) if np.linalg.norm(self.wind[0:2]) > 0.1 else 0.0
            }
        }


def run_sensor_degradation_demo():
    print("=" * 80)
    print("SENSOR DEGRADATION TESTING DEMONSTRATION")
    print("=" * 80)

    # Initialize
    controller = AdaptiveDroneController(dt=0.02, use_horizon_planner=True)
    drone = DroneSimulation()
    fault_injector = SensorFaultInjector(create_standard_degradation_schedule())

    # Compact square mission - smaller legs so drone completes all 4 WPs
    # despite degradation-induced hover pauses
    waypoints = [
        np.array([5.0, 0.0, -5.0]),     # WP1: East
        np.array([5.0, 5.0, -7.0]),     # WP2: Northeast, higher
        np.array([0.0, 5.0, -6.0]),     # WP3: North, mid altitude
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

    print(f"\nMission: {len(waypoints)}-waypoint square pattern (5m legs)")
    print("Wind: Mild [1.0, 0.5, 0.0] m/s (constant, isolating sensor effects)")
    print("\nDegradation Schedule:")
    print("  Phase 1:  0-20s    NOMINAL          - Full fusion, PID control")
    print("  Phase 2: 20-35s    GPS DEGRADATION  - Satellite loss, HDOP rise")
    print("  Phase 3: 35-45s    GPS RECOVERY     - Sensors restored, resume mission")
    print("  Phase 4: 45-60s    IMU DEGRADATION  - Noise, temp drift, bias")
    print("  Phase 5: 60-72s    COMBINED FAILURE  - GPS loss + IMU cal loss")
    print("  Phase 6: 72-85s    PARTIAL RECOVERY - GPS intermittent, mild IMU noise")
    print("  Phase 7: 85-120s   FULL RECOVERY    - All nominal")

    # Pre-flight check
    print("\n[1] PRE-FLIGHT CHECK")
    print("-" * 80)

    initial_conditions = {
        'gps_satellites': 12.0,
        'gps_hdop': 0.8,
        'imu_temperature': 25.0,
        'imu_calibrated': 1.0,
        'battery_voltage': 12.4,
        'motor_temperature': 30.0,
        'wind_speed': 1.0,
        'disturbance': 0.2,
        'computation_time': 0.001,
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

    # Mild constant wind to isolate sensor effects
    constant_wind = np.array([1.0, 0.5, 0.0])

    # Data recording
    time_history = []
    position_history = []
    velocity_history = []
    controller_history = []
    flight_mode_history = []
    waypoint_idx_history = []
    fusion_mode_history = []
    gps_health_history = []
    imu_health_history = []
    gps_sats_history = []
    gps_hdop_history = []
    imu_temp_history = []
    pos_covariance_history = []
    att_covariance_history = []
    cbf_intervention_history = []
    phase_history = []

    print("\n[2] MISSION EXECUTION WITH SENSOR DEGRADATION")
    print("-" * 80)
    print(f"{'Time':>6} | {'Phase':<16} | {'WP':>2} | {'Dist':>5} | {'Ctrl':>4} | "
          f"{'Mode':<9} | {'GPS':>4} | {'IMU':>4} | {'Fusion':<12} | Event")
    print("-" * 80)

    last_wp_idx = -1
    last_phase = ""
    mission_complete = False
    mission_complete_time = None
    was_degraded = False

    for step in range(9000):  # 180 seconds max (mission complete triggers early exit)
        t = step * 0.02
        time_history.append(t)

        drone.set_wind(constant_wind)

        # Get clean sensor data, then inject faults
        sensors_clean = drone.get_sensor_data()
        sensors = fault_injector.inject(sensors_clean, t)

        # Run control with degraded sensors
        control, telemetry = controller.control_step(sensors)
        drone.update(control)

        # Determine fusion mode from EKF sensor contracts
        gps_ok, imu_ok = controller.ekf.check_sensor_contracts(controller.time - controller.dt)
        if gps_ok and imu_ok:
            fusion_mode = 'full'
        elif imu_ok:
            fusion_mode = 'imu_only'
        else:
            fusion_mode = 'dead_reckoning'

        # Resume TRACK mode when GPS recovers and waypoints remain
        sup_status = controller.supervisor.get_status()
        wp_idx = sup_status['waypoint_idx']
        if (telemetry['flight_mode'] == 'hover' and
                gps_ok and imu_ok and
                was_degraded and
                wp_idx < len(waypoints) and
                not mission_complete):
            controller.supervisor.mode = FlightMode.TRACK
            was_degraded = False

        if not gps_ok or not imu_ok:
            was_degraded = True

        # Get sensor health
        health = fault_injector.get_sensor_health(t)
        phase = fault_injector.get_phase_name(t)

        # EKF covariance
        P = controller.ekf.get_covariance()
        pos_cov = np.trace(P[0:3, 0:3])
        att_cov = np.trace(P[6:9, 6:9])

        # Record data
        position_history.append(drone.position.copy())
        velocity_history.append(drone.velocity.copy())
        controller_history.append(telemetry['active_controller'])
        flight_mode_history.append(telemetry['flight_mode'])
        fusion_mode_history.append(fusion_mode)
        gps_health_history.append(health['gps'])
        imu_health_history.append(health['imu'])
        gps_sats_history.append(sensors['gps']['satellites'])
        gps_hdop_history.append(sensors['gps']['hdop'])
        imu_temp_history.append(sensors['imu']['temperature'])
        pos_covariance_history.append(pos_cov)
        att_covariance_history.append(att_cov)
        cbf_intervention_history.append(telemetry['cbf_intervened'])
        phase_history.append(phase)

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
                'wind_speed': np.linalg.norm(constant_wind[0:2]),
                'waypoint_idx': sup_status['waypoint_idx'],
            }
            if not viz.update(viz_state):
                print("\n[LIVE VISUALIZER] Window closed by user")
                break

        # Waypoint tracking
        sup_status = controller.supervisor.get_status()
        wp_idx = sup_status['waypoint_idx']
        waypoint_idx_history.append(wp_idx)

        ekf_pos = telemetry['position']
        if wp_idx < len(waypoints):
            dist_to_wp = np.linalg.norm(waypoints[wp_idx] - ekf_pos)
        else:
            dist_to_wp = 0.0

        # Events
        event = ""
        if wp_idx != last_wp_idx:
            if wp_idx > last_wp_idx and last_wp_idx >= 0:
                event = f"WP{last_wp_idx+1} REACHED"
            last_wp_idx = wp_idx

        if phase != last_phase:
            if not event:
                event = f"-> {phase}"
            last_phase = phase

        if telemetry['flight_mode'] == 'hover' and wp_idx >= len(waypoints) - 1 and not mission_complete:
            event = "MISSION COMPLETE"
            mission_complete = True
            mission_complete_time = t

        # Print status every 2 seconds or on events
        if step % 100 == 0 or event:
            print(f"{t:6.1f} | {phase:<16} | {wp_idx+1:>2} | {dist_to_wp:5.1f} | "
                  f"{telemetry['active_controller']:>4} | {telemetry['flight_mode']:<9} | "
                  f"{health['gps']:4.2f} | {health['imu']:4.2f} | {fusion_mode:<12} | {event}")

        if mission_complete and mission_complete_time and t > mission_complete_time + 5:
            break

    if viz is not None:
        viz.close()

    controller.stop_mission()

    # Convert to arrays
    time_arr = np.array(time_history)
    pos_arr = np.array(position_history)
    vel_arr = np.array(velocity_history)

    # Summary
    print("\n" + "=" * 80)
    print("[3] DEGRADATION TEST SUMMARY")
    print("=" * 80)

    total_time = time_arr[-1]
    wps_completed = len(waypoints) if mission_complete else max(waypoint_idx_history)
    print(f"\nFlight duration: {total_time:.1f} seconds")
    print(f"Waypoints completed: {wps_completed}/{len(waypoints)}")
    print(f"Mission status: {'COMPLETE' if mission_complete else 'INCOMPLETE'}")

    pid_time = sum(1 for c in controller_history if c == 'PID') * 0.02
    mpc_time = sum(1 for c in controller_history if c == 'MPC') * 0.02
    hinf_time = sum(1 for c in controller_history if c == 'Hinf') * 0.02
    print(f"\nController usage:")
    print(f"  PID:   {pid_time:5.1f}s ({100*pid_time/total_time:4.1f}%)")
    print(f"  MPC:   {mpc_time:5.1f}s ({100*mpc_time/total_time:4.1f}%)")
    print(f"  H-inf: {hinf_time:5.1f}s ({100*hinf_time/total_time:4.1f}%)")

    full_time = sum(1 for f in fusion_mode_history if f == 'full') * 0.02
    imu_time = sum(1 for f in fusion_mode_history if f == 'imu_only') * 0.02
    dr_time = sum(1 for f in fusion_mode_history if f == 'dead_reckoning') * 0.02
    print(f"\nEKF fusion modes:")
    print(f"  Full fusion:    {full_time:5.1f}s ({100*full_time/total_time:4.1f}%)")
    print(f"  IMU-only:       {imu_time:5.1f}s ({100*imu_time/total_time:4.1f}%)")
    print(f"  Dead reckoning: {dr_time:5.1f}s ({100*dr_time/total_time:4.1f}%)")

    cbf_count = sum(cbf_intervention_history)
    print(f"\nCBF interventions: {cbf_count}")

    mode_counts = {}
    for m in flight_mode_history:
        mode_counts[m] = mode_counts.get(m, 0) + 1
    print(f"\nFlight modes:")
    for mode, count in sorted(mode_counts.items()):
        t_mode = count * 0.02
        print(f"  {mode:<12} {t_mode:5.1f}s ({100*t_mode/total_time:4.1f}%)")

    # --- Visualization ---
    print("\nGenerating plots...")

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('Sensor Degradation Testing Results', fontsize=14, fontweight='bold', y=0.98)

    # Phase boundaries for vertical lines
    phase_boundaries = [0, 20, 35, 45, 60, 72, 85, total_time]
    phase_labels = ['Nominal', 'GPS Degrade', 'GPS Recovery', 'IMU Degrade',
                    'Combined Fail', 'Partial Recovery', 'Full Recovery']
    phase_colors = ['#e8f5e9', '#fff3e0', '#e8f5e9', '#fff3e0',
                    '#ffebee', '#fff3e0', '#e8f5e9']

    def add_phase_shading(ax):
        """Add phase background shading to a time-series axis."""
        for i in range(len(phase_labels)):
            t_s = phase_boundaries[i]
            t_e = phase_boundaries[i + 1]
            if t_e > time_arr[-1]:
                t_e = time_arr[-1]
            if t_s < time_arr[-1]:
                ax.axvspan(t_s, t_e, alpha=0.15, color=phase_colors[i])

    # 1. 3D Trajectory colored by fusion mode
    ax1 = fig.add_subplot(4, 2, 1, projection='3d')
    color_map = {'full': 'green', 'imu_only': '#FFA500', 'dead_reckoning': 'red'}
    for i in range(1, len(pos_arr)):
        ax1.plot(pos_arr[i-1:i+1, 0], pos_arr[i-1:i+1, 1], -pos_arr[i-1:i+1, 2],
                 color=color_map[fusion_mode_history[i]], linewidth=1.2, alpha=0.8)
    wp_array = np.array(waypoints)
    ax1.scatter(wp_array[:, 0], wp_array[:, 1], -wp_array[:, 2],
                c='red', s=100, marker='^', zorder=5)
    for i, wp in enumerate(waypoints):
        ax1.text(wp[0], wp[1], -wp[2] + 0.5, f'WP{i+1}', fontsize=8)
    ax1.scatter([0], [0], [5], c='blue', s=100, marker='o', zorder=5)
    legend_elements = [Line2D([0], [0], color='green', label='Full Fusion'),
                       Line2D([0], [0], color='#FFA500', label='IMU-Only'),
                       Line2D([0], [0], color='red', label='Dead Reckoning')]
    ax1.legend(handles=legend_elements, fontsize=7, loc='upper left')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Trajectory (colored by fusion mode)')

    # 2. Top-down view colored by fusion mode
    ax2 = fig.add_subplot(4, 2, 2)
    for i in range(1, len(pos_arr)):
        ax2.plot(pos_arr[i-1:i+1, 0], pos_arr[i-1:i+1, 1],
                 color=color_map[fusion_mode_history[i]], linewidth=1.5, alpha=0.8)
    ax2.scatter(wp_array[:, 0], wp_array[:, 1], c='red', s=100, marker='^', zorder=5)
    for i, wp in enumerate(waypoints):
        ax2.annotate(f'WP{i+1}', (wp[0], wp[1]), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)
    ax2.scatter([0], [0], c='blue', s=100, marker='o', zorder=5, label='Start')
    ax2.legend(handles=legend_elements + [Line2D([0], [0], marker='o', color='blue',
               linestyle='', label='Start')], fontsize=7)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (colored by fusion mode)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 3. Sensor Health Timeline
    ax3 = fig.add_subplot(4, 2, 3)
    add_phase_shading(ax3)
    ax3.plot(time_arr, gps_health_history, 'b-', linewidth=2, label='GPS Health')
    ax3.plot(time_arr, imu_health_history, 'r-', linewidth=2, label='IMU Health')
    ax3.fill_between(time_arr, 0, gps_health_history, alpha=0.2, color='blue')
    ax3.fill_between(time_arr, 0, imu_health_history, alpha=0.2, color='red')
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    # Phase labels at top
    for i in range(len(phase_labels)):
        mid = (phase_boundaries[i] + min(phase_boundaries[i+1], time_arr[-1])) / 2
        if mid <= time_arr[-1]:
            ax3.text(mid, 1.08, phase_labels[i], ha='center', va='bottom',
                     fontsize=6, fontweight='bold', rotation=0)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Health (0-1)')
    ax3.set_title('Sensor Health')
    ax3.legend(fontsize=8, loc='center left')
    ax3.set_ylim(-0.05, 1.15)
    ax3.grid(True, alpha=0.3)

    # 4. EKF Fusion Mode
    ax4 = fig.add_subplot(4, 2, 4)
    add_phase_shading(ax4)
    mode_numeric = [{'full': 2, 'imu_only': 1, 'dead_reckoning': 0}[m] for m in fusion_mode_history]
    ax4.fill_between(time_arr, 0, [1 if m == 'full' else 0 for m in fusion_mode_history],
                     alpha=0.6, color='green', step='post', label='Full Fusion')
    ax4.fill_between(time_arr, 0, [1 if m == 'imu_only' else 0 for m in fusion_mode_history],
                     alpha=0.6, color='#FFA500', step='post', label='IMU-Only')
    ax4.fill_between(time_arr, 0, [1 if m == 'dead_reckoning' else 0 for m in fusion_mode_history],
                     alpha=0.6, color='red', step='post', label='Dead Reckoning')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Active')
    ax4.set_title('EKF Fusion Mode')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Off', 'On'])
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_ylim(-0.1, 1.3)
    ax4.grid(True, alpha=0.3)

    # 5. Position + Altitude vs time
    ax5 = fig.add_subplot(4, 2, 5)
    add_phase_shading(ax5)
    ax5.plot(time_arr, pos_arr[:, 0], 'b-', linewidth=1.5, label='X')
    ax5.plot(time_arr, pos_arr[:, 1], 'g-', linewidth=1.5, label='Y')
    ax5.plot(time_arr, -pos_arr[:, 2], 'r-', linewidth=1.5, label='Altitude')
    # Mark controller switches
    for i in range(1, len(controller_history)):
        if controller_history[i] != controller_history[i-1]:
            ax5.axvline(x=time_arr[i], color='orange', linestyle='--', alpha=0.4, linewidth=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position (m)')
    ax5.set_title('Position & Altitude')
    ax5.legend(fontsize=8, loc='upper left')
    ax5.grid(True, alpha=0.3)

    # 6. Active Controller + Flight Mode
    ax6 = fig.add_subplot(4, 2, 6)
    add_phase_shading(ax6)
    # Controller lane (upper)
    ax6.fill_between(time_arr, 1.1, [1.9 if c == 'PID' else 1.1 for c in controller_history],
                     alpha=0.6, color='blue', step='post')
    ax6.fill_between(time_arr, 1.1, [1.9 if c == 'MPC' else 1.1 for c in controller_history],
                     alpha=0.6, color='green', step='post')
    ax6.fill_between(time_arr, 1.1, [1.9 if c == 'Hinf' else 1.1 for c in controller_history],
                     alpha=0.6, color='red', step='post')
    # Flight mode lane (lower)
    mode_colors = {'track': 'green', 'hover': '#FFA500', 'emergency': 'red', 'land': 'blue'}
    for mode_name, color in mode_colors.items():
        ax6.fill_between(time_arr, 0, [0.8 if m == mode_name else 0 for m in flight_mode_history],
                         alpha=0.6, color=color, step='post')
    ax6.set_xlabel('Time (s)')
    ax6.set_yticks([0.4, 1.5])
    ax6.set_yticklabels(['Flight Mode', 'Controller'])
    ax6.set_title('Controller & Flight Mode')
    legend_items = [Patch(facecolor='blue', alpha=0.6, label='PID'),
                    Patch(facecolor='green', alpha=0.6, label='MPC'),
                    Patch(facecolor='red', alpha=0.6, label='H-inf'),
                    Patch(facecolor='green', alpha=0.3, label='TRACK'),
                    Patch(facecolor='#FFA500', alpha=0.6, label='HOVER'),
                    Patch(facecolor='red', alpha=0.3, label='EMERGENCY')]
    ax6.legend(handles=legend_items, fontsize=7, loc='upper right', ncol=2)
    ax6.set_ylim(-0.1, 2.1)
    ax6.grid(True, alpha=0.3)

    # 7. EKF Covariance Trace
    ax7 = fig.add_subplot(4, 2, 7)
    add_phase_shading(ax7)
    ax7.semilogy(time_arr, pos_covariance_history, 'b-', linewidth=1.5, label='Position Uncertainty')
    ax7.semilogy(time_arr, att_covariance_history, 'r-', linewidth=1.5, label='Attitude Uncertainty')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Covariance Trace (log)')
    ax7.set_title('EKF Estimation Uncertainty')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 8. GPS Satellites + HDOP
    ax8 = fig.add_subplot(4, 2, 8)
    add_phase_shading(ax8)
    ax8.plot(time_arr, gps_sats_history, 'b-', linewidth=1.5, label='Satellites')
    ax8.axhline(y=6, color='blue', linestyle=':', alpha=0.5, label='Sat threshold (6)')
    ax8_twin = ax8.twinx()
    ax8_twin.plot(time_arr, gps_hdop_history, 'r-', linewidth=1.5, label='HDOP')
    ax8_twin.axhline(y=2.0, color='red', linestyle=':', alpha=0.5, label='HDOP threshold (2.0)')
    # Clip HDOP display for readability
    ax8_twin.set_ylim(0, max(15, max(h for h in gps_hdop_history if h < 50) + 2))
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Satellites', color='blue')
    ax8_twin.set_ylabel('HDOP', color='red')
    ax8.tick_params(axis='y', labelcolor='blue')
    ax8_twin.tick_params(axis='y', labelcolor='red')
    ax8.set_title('GPS Quality Indicators')
    # Combined legend
    lines_1, labels_1 = ax8.get_legend_handles_labels()
    lines_2, labels_2 = ax8_twin.get_legend_handles_labels()
    ax8.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=7, loc='center right')
    ax8.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(os.path.dirname(__file__), 'sensor_degradation_results.png')
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    print("\n" + "=" * 80)
    print("SENSOR DEGRADATION TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_sensor_degradation_demo()
