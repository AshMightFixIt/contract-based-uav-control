"""
Benchmark: Contract-Based Adaptive Architecture vs Standalone Controllers

Runs the same wind/waypoint scenario with 4 configurations:
  1. Adaptive (full contract-based PID→MPC→H-inf switching)
  2. PID-only (forced)
  3. MPC-only (forced)
  4. H-inf-only (forced)

Produces comparison metrics table + benchmark_results.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import copy

logging.basicConfig(level=logging.ERROR, format='%(message)s')

from adaptive_control_system import AdaptiveDroneController


# ---------------------------------------------------------------------------
# DroneSimulation (copied from demo_waypoint_mission.py for self-containment)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Shared scenario parameters
# ---------------------------------------------------------------------------
WAYPOINTS = [
    np.array([10.0, 0.0, -5.0]),
    np.array([10.0, 10.0, -8.0]),
    np.array([0.0, 10.0, -6.0]),
    np.array([0.0, 0.0, -5.0]),
]

WIND_SCHEDULE = [
    (0.0,  25.0, np.array([0.5,  0.0, 0.0])),   # Calm       → PID
    (25.0, 45.0, np.array([4.5,  2.0, 0.0])),   # Moderate   → MPC
    (45.0, 60.0, np.array([9.0,  4.0, 0.0])),   # Strong     → H-inf
    (60.0, 80.0, np.array([5.0,  2.0, 0.0])),   # Easing     → MPC
    (80.0, 120.0, np.array([1.0, 0.5, 0.0])),   # Calm       → PID
]

INITIAL_CONDITIONS = {
    'gps_satellites': 12.0,
    'gps_hdop': 0.8,
    'imu_temperature': 25.0,
    'imu_calibrated': 1.0,
    'battery_voltage': 12.4,
    'motor_temperature': 30.0,
    'wind_speed': 0.5,
    'disturbance': 0.2,
}

MAX_STEPS = 6000  # 120 s at 0.02 dt
DT = 0.02


# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------
def run_scenario(mode: str):
    """
    Run the full mission under a given controller strategy.

    mode: 'adaptive' | 'PID' | 'MPC' | 'Hinf'
    Returns dict of time-series data.
    """
    np.random.seed(42)  # reproducible noise

    drone = DroneSimulation()
    controller = AdaptiveDroneController(dt=DT, use_horizon_planner=(mode == 'adaptive'))

    mission = {'target_position': WAYPOINTS[0], 'waypoints': list(WAYPOINTS)}
    feasible, msg = controller.pre_flight_check(INITIAL_CONDITIONS, mission)
    if not feasible:
        print(f"  [!] Pre-flight failed for {mode}: {msg}")
        return None

    controller.start_mission()

    # --- Force standalone controller ------------------------------------------
    if mode != 'adaptive':
        target_ctrl = mode  # 'PID', 'MPC', or 'Hinf'
        # Force-switch immediately (bypass cooldown)
        controller.controller_switcher.last_switch_time = -10.0
        controller.controller_switcher.switch_to(target_ctrl, 0.0, f"Benchmark: force {target_ctrl}")
        # Monkey-patch switch_to to be a no-op so adaptive logic can't switch away
        controller.controller_switcher.switch_to = lambda *a, **kw: False

    # --- Time-series storage -------------------------------------------------
    times = []
    positions = []
    velocities = []
    controllers_used = []
    thrusts = []
    torques_list = []
    attitudes = []
    cbf_flags = []
    waypoint_indices = []
    targets = []

    mission_complete = False
    mission_complete_time = None
    last_control = None

    for step in range(MAX_STEPS):
        t = step * DT
        times.append(t)

        # Current wind
        current_wind = np.array([0.5, 0.0, 0.0])
        for t_start, t_end, wind in WIND_SCHEDULE:
            if t_start <= t < t_end:
                current_wind = wind
                break

        drone.set_wind(current_wind)
        sensors = drone.get_sensor_data()
        control, telemetry = controller.control_step(sensors)
        drone.update(control)

        # Record
        positions.append(drone.position.copy())
        velocities.append(drone.velocity.copy())
        controllers_used.append(telemetry['active_controller'])
        thrusts.append(control['thrust'])
        torques_list.append(control.get('torques', np.zeros(3)).copy())
        attitudes.append(drone.attitude.copy())
        cbf_flags.append(telemetry['cbf_intervened'])

        sup_status = controller.supervisor.get_status()
        wp_idx = sup_status['waypoint_idx']
        waypoint_indices.append(wp_idx)

        if wp_idx < len(WAYPOINTS):
            targets.append(WAYPOINTS[wp_idx].copy())
        else:
            targets.append(WAYPOINTS[-1].copy())

        # Mission complete detection
        if telemetry['flight_mode'] == 'hover' and wp_idx >= len(WAYPOINTS) - 1 and not mission_complete:
            mission_complete = True
            mission_complete_time = t

        if mission_complete and mission_complete_time and t > mission_complete_time + 5:
            break

        last_control = control

    controller.stop_mission()

    return {
        'mode': mode,
        'times': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'controllers': controllers_used,
        'thrusts': np.array(thrusts),
        'torques': np.array(torques_list),
        'attitudes': np.array(attitudes),
        'cbf_flags': cbf_flags,
        'waypoint_indices': waypoint_indices,
        'targets': np.array(targets),
        'mission_complete': mission_complete,
        'mission_complete_time': mission_complete_time,
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(data):
    """Compute summary metrics from one scenario run."""
    pos = data['positions']
    vel = data['velocities']
    tgt = data['targets']
    thrusts = data['thrusts']
    torques = data['torques']
    att = data['attitudes']
    cbf = data['cbf_flags']
    wp_idx = data['waypoint_indices']
    times = data['times']

    # Tracking error (distance to current target at each step)
    tracking_errors = np.linalg.norm(pos - tgt, axis=1)

    # Speed
    speeds = np.linalg.norm(vel, axis=1)

    # Control smoothness (sum of |u(k) - u(k-1)| for thrust)
    thrust_diffs = np.abs(np.diff(thrusts))
    torque_diffs = np.linalg.norm(np.diff(torques, axis=0), axis=1)
    smoothness = np.sum(thrust_diffs) + np.sum(torque_diffs)

    # Energy proxy: sum of thrust
    total_energy = np.sum(thrusts) * DT

    # Control effort: sum of |thrust - hover| + |torques|
    control_effort = np.sum(np.abs(thrusts - 0.5) + np.linalg.norm(torques, axis=1)) * DT

    # Max tilt (roll/pitch)
    tilt_angles = np.sqrt(att[:, 0]**2 + att[:, 1]**2)

    # Waypoints completed
    max_wp = max(wp_idx)
    waypoints_completed = min(max_wp, len(WAYPOINTS))

    # Mission time
    if data['mission_complete'] and data['mission_complete_time'] is not None:
        mission_time = data['mission_complete_time']
    else:
        mission_time = times[-1]  # didn't complete — full duration

    # CBF interventions
    cbf_count = sum(1 for f in cbf if f)

    return {
        'RMS Tracking Error (m)': float(np.sqrt(np.mean(tracking_errors**2))),
        'Max Tracking Error (m)': float(np.max(tracking_errors)),
        'Waypoints Completed': waypoints_completed,
        'Mission Time (s)': mission_time,
        'Total Energy (N·s)': total_energy,
        'Control Effort': control_effort,
        'CBF Interventions': cbf_count,
        'Max Tilt (rad)': float(np.max(tilt_angles)),
        'Control Smoothness': smoothness,
        'Max Speed (m/s)': float(np.max(speeds)),
    }


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------
def print_comparison_table(all_metrics):
    """Print a formatted comparison table with best/worst highlighting."""
    modes = list(all_metrics.keys())
    metric_names = list(next(iter(all_metrics.values())).keys())

    # Determine best/worst for each metric
    # Lower is better for most metrics; higher is better for Waypoints Completed
    higher_is_better = {'Waypoints Completed'}
    lower_is_better_set = set(metric_names) - higher_is_better

    col_width = 18
    header = f"{'Metric':<28}" + "".join(f"{m:>{col_width}}" for m in modes)
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("BENCHMARK RESULTS: Adaptive vs Standalone Controllers")
    print("=" * len(header))
    print(header)
    print(sep)

    for metric in metric_names:
        values = {m: all_metrics[m][metric] for m in modes}
        vals_list = list(values.values())

        if metric in higher_is_better:
            best_val = max(vals_list)
            worst_val = min(vals_list)
        else:
            best_val = min(vals_list)
            worst_val = max(vals_list)

        row = f"{metric:<28}"
        for m in modes:
            v = values[m]
            if isinstance(v, int) or (isinstance(v, float) and v == int(v) and abs(v) < 1000):
                cell = f"{int(v)}"
            else:
                cell = f"{v:.3f}"

            # Mark best (*) and worst (!)
            if v == best_val and best_val != worst_val:
                cell += " *"
            elif v == worst_val and best_val != worst_val:
                cell += " !"
            row += f"{cell:>{col_width}}"
        print(row)

    print(sep)
    print("  * = best   ! = worst")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def generate_plots(all_data, all_metrics):
    """Generate 8-panel comparison figure (4x2 layout)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle('Controller Benchmark: Adaptive vs Standalone', fontsize=14, fontweight='bold', y=0.98)

    colors = {
        'adaptive': '#2196F3',
        'PID': '#4CAF50',
        'MPC': '#FF9800',
        'Hinf': '#F44336',
    }
    labels = {
        'adaptive': 'Adaptive',
        'PID': 'PID-only',
        'MPC': 'MPC-only',
        'Hinf': 'H-inf-only',
    }
    modes = list(all_data.keys())
    wind_colors = ['#E3F2FD', '#FFF3E0', '#FFEBEE', '#FFF3E0', '#E3F2FD']
    wind_labels_text = ['Calm', 'Moderate', 'Strong', 'Moderate', 'Calm']

    def add_wind_shading(ax):
        for i, (t0, t1, _w) in enumerate(WIND_SCHEDULE):
            ax.axvspan(t0, t1, alpha=0.15, color=wind_colors[i])
            ax.text((t0 + t1) / 2, ax.get_ylim()[1] * 0.95, wind_labels_text[i],
                    ha='center', va='top', fontsize=6, alpha=0.6)

    # --- Panel 1 (0,0): Tracking error vs time with wind shading + switch markers ---
    ax = axes[0, 0]
    for mode in modes:
        d = all_data[mode]
        errs = np.linalg.norm(d['positions'] - d['targets'], axis=1)
        ax.plot(d['times'], errs, color=colors[mode], label=labels[mode], linewidth=1.2, alpha=0.85)
    # Wind regime shading
    for i, (t0, t1, _w) in enumerate(WIND_SCHEDULE):
        ax.axvspan(t0, t1, alpha=0.15, color=wind_colors[i])
        ax.text((t0 + t1) / 2, 0.02, wind_labels_text[i], ha='center', va='bottom',
                fontsize=6, alpha=0.6, transform=ax.get_xaxis_transform())
    # Controller switch markers for adaptive
    if 'adaptive' in all_data:
        d = all_data['adaptive']
        ctrls = d['controllers']
        for j in range(1, len(ctrls)):
            if ctrls[j] != ctrls[j-1]:
                ax.axvline(x=d['times'][j], color=colors['adaptive'], linestyle=':', alpha=0.4, linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error (m)')
    ax.set_title('Tracking Error Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2 (0,1): XY trajectory with dashed waypoint connections ---
    ax = axes[0, 1]
    wp_arr = np.array(WAYPOINTS)
    # Dashed waypoint connections
    all_wp_x = [0] + [wp[0] for wp in WAYPOINTS]
    all_wp_y = [0] + [wp[1] for wp in WAYPOINTS]
    ax.plot(all_wp_x, all_wp_y, 'k--', alpha=0.3, linewidth=1.0, zorder=1)
    for mode in modes:
        d = all_data[mode]
        ax.plot(d['positions'][:, 0], d['positions'][:, 1],
                color=colors[mode], label=labels[mode], linewidth=1.2, alpha=0.85)
        # Direction arrowheads (every 20% of trajectory)
        n = len(d['positions'])
        for frac in [0.2, 0.4, 0.6, 0.8]:
            idx = int(frac * n)
            if idx < n - 1:
                dx = d['positions'][idx+1, 0] - d['positions'][idx, 0]
                dy = d['positions'][idx+1, 1] - d['positions'][idx, 1]
                ax.annotate('', xy=(d['positions'][idx, 0] + dx, d['positions'][idx, 1] + dy),
                           xytext=(d['positions'][idx, 0], d['positions'][idx, 1]),
                           arrowprops=dict(arrowstyle='->', color=colors[mode], lw=1.5))
    ax.scatter(wp_arr[:, 0], wp_arr[:, 1], c='red', s=100, marker='^', zorder=5, label='Waypoints')
    for i, wp in enumerate(WAYPOINTS):
        ax.annotate(f'WP{i+1}', (wp[0], wp[1]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9, fontweight='bold')
    ax.scatter([0], [0], c='black', s=80, marker='o', zorder=5, label='Start')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Trajectory Comparison')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # --- Panel 3 (1,0): 3D trajectory ---
    axes[1, 0].remove()
    ax3d = fig.add_subplot(4, 2, 3, projection='3d')
    for mode in modes:
        d = all_data[mode]
        ax3d.plot(d['positions'][:, 0], d['positions'][:, 1], -d['positions'][:, 2],
                  color=colors[mode], label=labels[mode], linewidth=1.2, alpha=0.85)
    ax3d.scatter(wp_arr[:, 0], wp_arr[:, 1], -wp_arr[:, 2],
                 c='red', s=80, marker='^', zorder=5, label='Waypoints')
    for i, wp in enumerate(WAYPOINTS):
        ax3d.text(wp[0], wp[1], -wp[2], f'  WP{i+1}', fontsize=7)
    ax3d.set_xlabel('X (m)', fontsize=8)
    ax3d.set_ylabel('Y (m)', fontsize=8)
    ax3d.set_zlabel('Altitude (m)', fontsize=8)
    ax3d.set_title('3D Trajectory Comparison', fontsize=10)
    ax3d.legend(fontsize=7, loc='upper left')

    # --- Panel 4 (1,1): Altitude vs time ---
    ax = axes[1, 1]
    for mode in modes:
        d = all_data[mode]
        ax.plot(d['times'], -d['positions'][:, 2], color=colors[mode],
                label=labels[mode], linewidth=1.2, alpha=0.85)
    # Horizontal lines at waypoint altitudes
    wp_alts = sorted(set(-wp[2] for wp in WAYPOINTS))
    for alt in wp_alts:
        ax.axhline(y=alt, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
        ax.text(0.5, alt, f'{alt:.0f}m', fontsize=7, alpha=0.5, va='bottom')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 5 (2,0): Key metrics bar chart ---
    ax = axes[2, 0]
    bar_metrics = ['RMS Tracking Error (m)', 'Total Energy (N·s)', 'CBF Interventions', 'Mission Time (s)']
    bar_labels = ['RMS Error\n(m)', 'Energy\n(N·s)', 'CBF\nInterventions', 'Mission Time\n(s)']
    x = np.arange(len(bar_metrics))
    width = 0.18
    for i, mode in enumerate(modes):
        vals = [all_metrics[mode][m] for m in bar_metrics]
        ax.bar(x + i * width, vals, width, label=labels[mode], color=colors[mode], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_title('Key Metrics Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 6 (2,1): Thrust over time ---
    ax = axes[2, 1]
    for mode in modes:
        d = all_data[mode]
        ax.plot(d['times'], d['thrusts'], color=colors[mode], label=labels[mode],
                linewidth=0.8, alpha=0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Hover')
    for i, (t0, t1, _w) in enumerate(WIND_SCHEDULE):
        ax.axvspan(t0, t1, alpha=0.1, color=wind_colors[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust')
    ax.set_title('Thrust Command Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 7 (3,0): Speed over time with controller-switch markers ---
    ax = axes[3, 0]
    for mode in modes:
        d = all_data[mode]
        speeds = np.linalg.norm(d['velocities'], axis=1)
        ax.plot(d['times'], speeds, color=colors[mode], label=labels[mode],
                linewidth=1.0, alpha=0.8)
    # Controller switch markers for adaptive
    if 'adaptive' in all_data:
        d = all_data['adaptive']
        ctrls = d['controllers']
        for j in range(1, len(ctrls)):
            if ctrls[j] != ctrls[j-1]:
                ax.axvline(x=d['times'][j], color=colors['adaptive'], linestyle=':', alpha=0.4, linewidth=0.8)
    for i, (t0, t1, _w) in enumerate(WIND_SCHEDULE):
        ax.axvspan(t0, t1, alpha=0.1, color=wind_colors[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 8 (3,1): Radar/spider chart ---
    axes[3, 1].remove()
    ax_radar = fig.add_subplot(4, 2, 8, projection='polar')
    radar_metrics = ['RMS Tracking Error (m)', 'Control Effort', 'Control Smoothness',
                     'Max Tilt (rad)', 'Max Speed (m/s)', 'Total Energy (N·s)']
    radar_labels_text = ['RMS Error', 'Ctrl Effort', 'Smoothness', 'Max Tilt', 'Max Speed', 'Energy']
    # All these are "lower is better" — invert so bigger polygon = better
    normalized = {}
    for mode in modes:
        normalized[mode] = []
    for metric in radar_metrics:
        vals = [all_metrics[m][metric] for m in modes]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax != vmin else 1.0
        for mode in modes:
            # Invert: 1.0 = best (lowest), 0.0 = worst (highest)
            normalized[mode].append(1.0 - (all_metrics[mode][metric] - vmin) / rng)

    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close polygon

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_rlabel_position(30)
    plt.setp(ax_radar.get_yticklabels(), fontsize=6)

    for mode in modes:
        values = normalized[mode] + normalized[mode][:1]  # Close polygon
        ax_radar.plot(angles, values, color=colors[mode], linewidth=1.5, label=labels[mode])
        ax_radar.fill(angles, values, color=colors[mode], alpha=0.1)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels_text, fontsize=7)
    ax_radar.set_ylim(0, 1.1)
    ax_radar.set_title('Performance Radar (bigger = better)', fontsize=10, pad=20)
    ax_radar.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BENCHMARK: Contract-Based Adaptive vs Standalone Controllers")
    print("=" * 70)

    print("\nWind profile:")
    print("  0-25s:  Calm     (0.5 m/s)  - PID comfort zone")
    print("  25-45s: Moderate (5.0 m/s)  - MPC comfort zone")
    print("  45-60s: Strong   (10.0 m/s) - H-inf territory")
    print("  60-80s: Easing   (5.0 m/s)  - MPC comfort zone")
    print("  80-120s: Calm    (1.0 m/s)  - PID comfort zone")

    print(f"\nWaypoints: {len(WAYPOINTS)} (square pattern)")
    print(f"Max duration: {MAX_STEPS * DT:.0f}s\n")

    modes = ['adaptive', 'PID', 'MPC', 'Hinf']
    mode_labels = {'adaptive': 'Adaptive', 'PID': 'PID-only', 'MPC': 'MPC-only', 'Hinf': 'H-inf-only'}

    all_data = {}
    all_metrics = {}

    for mode in modes:
        label = mode_labels[mode]
        print(f"Running {label}...", end=" ", flush=True)
        data = run_scenario(mode)
        if data is None:
            print("FAILED")
            continue
        metrics = compute_metrics(data)
        all_data[mode] = data
        all_metrics[mode] = metrics
        status = "COMPLETE" if data['mission_complete'] else f"TIMEOUT ({data['times'][-1]:.0f}s)"
        print(f"{status}  (WP: {metrics['Waypoints Completed']}/{len(WAYPOINTS)}, "
              f"RMS: {metrics['RMS Tracking Error (m)']:.2f}m)")

    if len(all_metrics) < 2:
        print("\nNot enough scenarios completed for comparison.")
        return

    # Print comparison table
    print_comparison_table(all_metrics)

    # Qualitative summary
    print("\n" + "=" * 70)
    print("QUALITATIVE SUMMARY")
    print("=" * 70)

    # Find best RMS error
    rms_vals = {m: all_metrics[m]['RMS Tracking Error (m)'] for m in all_metrics}
    best_rms = min(rms_vals, key=rms_vals.get)

    # Find best energy
    energy_vals = {m: all_metrics[m]['Total Energy (N·s)'] for m in all_metrics}
    best_energy = min(energy_vals, key=energy_vals.get)

    # Find most waypoints / fastest
    wp_vals = {m: all_metrics[m]['Waypoints Completed'] for m in all_metrics}
    best_wp = max(wp_vals, key=wp_vals.get)

    time_vals = {m: all_metrics[m]['Mission Time (s)'] for m in all_metrics}
    best_time = min(time_vals, key=time_vals.get)

    print(f"  Best tracking accuracy : {mode_labels.get(best_rms, best_rms)} "
          f"(RMS {rms_vals[best_rms]:.3f}m)")
    print(f"  Most energy efficient  : {mode_labels.get(best_energy, best_energy)} "
          f"({energy_vals[best_energy]:.1f} N·s)")
    print(f"  Most waypoints reached : {mode_labels.get(best_wp, best_wp)} "
          f"({wp_vals[best_wp]}/{len(WAYPOINTS)})")
    print(f"  Fastest mission time   : {mode_labels.get(best_time, best_time)} "
          f"({time_vals[best_time]:.1f}s)")

    # Overall winner (simple: count how many categories each wins)
    wins = {}
    for m in all_metrics:
        wins[m] = 0
    for cat, best in [(rms_vals, best_rms), (energy_vals, best_energy),
                      (wp_vals, best_wp), (time_vals, best_time)]:
        wins[best] += 1
    overall = max(wins, key=wins.get)
    print(f"\n  Overall winner: {mode_labels.get(overall, overall)} ({wins[overall]}/4 categories)")

    # Generate plots
    print("\nGenerating comparison plots...")
    generate_plots(all_data, all_metrics)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
