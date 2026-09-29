"""
Benchmark: Racing Course — Contract-Based Adaptive vs Standalone Controllers

Runs a 10-waypoint figure-8 racing course with faster drone physics.
Tests high-speed maneuvering performance with 4 configurations:
  1. Adaptive (full contract-based PID->MPC->H-inf switching)
  2. PID-only (forced)
  3. MPC-only (forced)
  4. H-inf-only (forced)

Produces comparison metrics table + benchmark_racing.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.ERROR, format='%(message)s')

from contract_uav_core.core import AdaptiveDroneController


# ---------------------------------------------------------------------------
# Racing DroneSimulation (faster physics)
# ---------------------------------------------------------------------------
class RacingDroneSimulation:
    """Faster drone dynamics for racing benchmark."""

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

        max_accel = 4.0  # Was 2.0
        horizontal_accel = np.array([
            np.clip(-np.sin(desired_att[1]) * 8.0, -max_accel, max_accel),
            np.clip(np.sin(desired_att[0]) * 8.0, -max_accel, max_accel),
            0.0
        ])
        self.velocity[0:2] += horizontal_accel[0:2] * self.dt

        self.velocity[0:2] += self.wind[0:2] * 0.06 * self.dt

        max_vel = 5.0  # Was 3.0
        vel_magnitude = np.linalg.norm(self.velocity[0:2])
        if vel_magnitude > max_vel:
            self.velocity[0:2] *= max_vel / vel_magnitude

        self.velocity[2] = np.clip(self.velocity[2], -3.0, 3.0)  # Was ±2.0
        self.position += self.velocity * self.dt

        if self.position[2] > -0.5:
            self.position[2] = -0.5
            self.velocity[2] = min(self.velocity[2], 0.0)

        self.attitude = 0.95 * self.attitude + 0.05 * desired_att
        self.velocity *= 0.98  # Was 0.97
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
# Racing course (10 waypoints, figure-8 with altitude changes)
# ---------------------------------------------------------------------------
WAYPOINTS = [
    np.array([0.0,   0.0,  -5.0]),    # WP1: Start
    np.array([15.0,  5.0,  -8.0]),    # WP2: Accelerate NE + climb
    np.array([30.0,  0.0,  -10.0]),   # WP3: Fast straight E + peak altitude
    np.array([30.0, -10.0, -7.0]),    # WP4: Turn south + descend
    np.array([20.0, -15.0, -5.0]),    # WP5: Sweep SW
    np.array([10.0, -10.0, -3.0]),    # WP6: Low altitude run NW
    np.array([0.0,  -5.0,  -5.0]),    # WP7: Back toward start
    np.array([5.0,   5.0,  -8.0]),    # WP8: Climb NE
    np.array([15.0,  10.0, -10.0]),   # WP9: High-speed cruise
    np.array([0.0,   0.0,  -5.0]),    # WP10: Return to start
]

# Wind profile sweeps through all three controller tiers during the race:
#   0–25 s  : calm  (0.5→2.5 m/s)  — PID territory    (wind < 3)
#  25–55 s  : moderate (3→7 m/s)   — MPC territory     (3 ≤ wind < 8)
#  55–80 s  : strong  (8→11 m/s)   — H-inf territory   (wind ≥ 8)
#  80–100 s : recovery (11→1.5 m/s)— back to PID/MPC
# Direction is constant NE (~60°) so the profile is a scalar ramp.
WIND_DIRECTION = np.array([0.857, 0.515, 0.0])   # unit vector, NE

def get_wind_at_time(t: float) -> np.ndarray:
    """Return wind vector for elapsed race time t (seconds)."""
    if t < 25.0:
        mag = 0.5 + t * 0.08           # 0.5 → 2.5 m/s
    elif t < 55.0:
        mag = 2.5 + (t - 25.0) * 0.15  # 2.5 → 7.0 m/s
    elif t < 80.0:
        mag = 7.0 + (t - 55.0) * 0.16  # 7.0 → 11.0 m/s
    elif t < 100.0:
        mag = 11.0 - (t - 80.0) * 0.475  # 11.0 → 1.5 m/s
    else:
        mag = 1.5
    return WIND_DIRECTION * mag

INITIAL_CONDITIONS = {
    'gps_satellites': 12.0,
    'gps_hdop': 0.8,
    'imu_temperature': 25.0,
    'imu_calibrated': 1.0,
    'battery_voltage': 12.4,
    'motor_temperature': 30.0,
    'wind_speed': 0.5,
    'disturbance': 0.2,
    'computation_time': 0.001,
}

MAX_STEPS = 10000  # 200s at 0.02 dt
DT = 0.02


# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------
def run_scenario(mode: str):
    """
    mode: 'adaptive' | 'PID' | 'MPC' | 'Hinf'
    """
    np.random.seed(42)

    drone = RacingDroneSimulation()
    controller = AdaptiveDroneController(dt=DT, use_horizon_planner=(mode == 'adaptive'))

    mission = {'target_position': WAYPOINTS[0], 'waypoints': list(WAYPOINTS)}
    feasible, msg = controller.pre_flight_check(INITIAL_CONDITIONS, mission)
    if not feasible:
        print(f"  [!] Pre-flight failed for {mode}: {msg}")
        return None

    controller.start_mission()

    # Force standalone controller
    if mode != 'adaptive':
        controller.controller_switcher.last_switch_time = -10.0
        controller.controller_switcher.switch_to(mode, 0.0, f"Benchmark: force {mode}")
        controller.controller_switcher.switch_to = lambda *a, **kw: False

    # Storage
    times, positions, velocities = [], [], []
    controllers_used, thrusts, torques_list = [], [], []
    attitudes, cbf_flags, waypoint_indices = [], [], []
    targets = []

    # Per-waypoint timing
    wp_completion_times = {}
    last_wp_idx = 0

    mission_complete = False
    mission_complete_time = None

    for step in range(MAX_STEPS):
        t = step * DT
        times.append(t)

        drone.set_wind(get_wind_at_time(t))
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

        # Track per-waypoint completion
        if wp_idx > last_wp_idx:
            wp_completion_times[last_wp_idx] = t
            last_wp_idx = wp_idx

        # Mission complete detection
        if telemetry['flight_mode'] == 'hover' and wp_idx >= len(WAYPOINTS) - 1 and not mission_complete:
            mission_complete = True
            mission_complete_time = t
            if last_wp_idx not in wp_completion_times:
                wp_completion_times[last_wp_idx] = t

        if mission_complete and mission_complete_time and t > mission_complete_time + 5:
            break

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
        'wp_completion_times': wp_completion_times,
        'mission_complete': mission_complete,
        'mission_complete_time': mission_complete_time,
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(data):
    pos = data['positions']
    vel = data['velocities']
    tgt = data['targets']
    thrusts = data['thrusts']
    torques = data['torques']
    cbf = data['cbf_flags']
    wp_idx = data['waypoint_indices']
    times = data['times']

    tracking_errors = np.linalg.norm(pos - tgt, axis=1)
    speeds = np.linalg.norm(vel, axis=1)

    # Control smoothness
    thrust_diffs = np.abs(np.diff(thrusts))
    torque_diffs = np.linalg.norm(np.diff(torques, axis=0), axis=1)
    smoothness = np.sum(thrust_diffs) + np.sum(torque_diffs)

    # Energy
    total_energy = np.sum(thrusts) * DT

    # Waypoints completed
    max_wp = max(wp_idx)
    waypoints_completed = min(max_wp, len(WAYPOINTS))

    # Lap time (mission completion time)
    if data['mission_complete'] and data['mission_complete_time'] is not None:
        lap_time = data['mission_complete_time']
    else:
        lap_time = times[-1]

    # CBF interventions
    cbf_count = sum(1 for f in cbf if f)

    # Per-waypoint completion times (segments)
    wp_segment_times = []
    wpc = data['wp_completion_times']
    sorted_wps = sorted(wpc.keys())
    prev_t = 0.0
    for wp_i in sorted_wps:
        wp_segment_times.append(wpc[wp_i] - prev_t)
        prev_t = wpc[wp_i]

    return {
        'Lap Time (s)': lap_time,
        'Avg Speed (m/s)': float(np.mean(speeds)),
        'Max Speed (m/s)': float(np.max(speeds)),
        'RMS Tracking Error (m)': float(np.sqrt(np.mean(tracking_errors**2))),
        'Max Tracking Error (m)': float(np.max(tracking_errors)),
        'Waypoints Completed': waypoints_completed,
        'Control Smoothness': smoothness,
        'Total Energy (N·s)': total_energy,
        'CBF Interventions': cbf_count,
        'wp_segment_times': wp_segment_times,
    }


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------
def print_comparison_table(all_metrics):
    modes = list(all_metrics.keys())
    metric_names = [k for k in next(iter(all_metrics.values())).keys() if k != 'wp_segment_times']

    higher_is_better = {'Waypoints Completed', 'Avg Speed (m/s)', 'Max Speed (m/s)'}

    col_width = 18
    header = f"{'Metric':<28}" + "".join(f"{m:>{col_width}}" for m in modes)
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("RACING BENCHMARK RESULTS")
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
    """Generate 8-panel racing benchmark figure (4x2 layout)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle('Racing Benchmark: Adaptive vs Standalone Controllers', fontsize=14, fontweight='bold', y=0.98)

    colors = {
        'adaptive': '#2196F3',
        'PID': '#4CAF50',
        'MPC': '#FF9800',
        'Hinf': '#F44336',
    }
    plot_labels = {
        'adaptive': 'Adaptive',
        'PID': 'PID-only',
        'MPC': 'MPC-only',
        'Hinf': 'H-inf-only',
    }
    modes = list(all_data.keys())
    wp_arr = np.array(WAYPOINTS)

    # --- Panel 1 (0,0): 3D trajectory (all modes) ---
    axes[0, 0].remove()
    ax3d = fig.add_subplot(4, 2, 1, projection='3d')
    for mode in modes:
        d = all_data[mode]
        ax3d.plot(d['positions'][:, 0], d['positions'][:, 1], -d['positions'][:, 2],
                  color=colors[mode], label=plot_labels[mode], linewidth=1.2, alpha=0.85)
    ax3d.scatter(wp_arr[:, 0], wp_arr[:, 1], -wp_arr[:, 2],
                 c='red', s=60, marker='^', zorder=5)
    for i, wp in enumerate(WAYPOINTS):
        ax3d.text(wp[0], wp[1], -wp[2], f' {i+1}', fontsize=7, fontweight='bold')
    # Course outline
    course_x = [wp[0] for wp in WAYPOINTS] + [WAYPOINTS[0][0]]
    course_y = [wp[1] for wp in WAYPOINTS] + [WAYPOINTS[0][1]]
    course_z = [-wp[2] for wp in WAYPOINTS] + [-WAYPOINTS[0][2]]
    ax3d.plot(course_x, course_y, course_z, 'k--', alpha=0.25, linewidth=0.8)
    ax3d.set_xlabel('X (m)', fontsize=8)
    ax3d.set_ylabel('Y (m)', fontsize=8)
    ax3d.set_zlabel('Altitude (m)', fontsize=8)
    ax3d.set_title('3D Racing Trajectory', fontsize=10)
    ax3d.legend(fontsize=7, loc='upper left')

    # --- Panel 2 (0,1): XY trajectory with numbered waypoints + dashed course ---
    ax = axes[0, 1]
    # Dashed course outline
    course_x_2d = [wp[0] for wp in WAYPOINTS] + [WAYPOINTS[0][0]]
    course_y_2d = [wp[1] for wp in WAYPOINTS] + [WAYPOINTS[0][1]]
    ax.plot(course_x_2d, course_y_2d, 'k--', alpha=0.3, linewidth=1.0, zorder=1, label='Course')
    for mode in modes:
        d = all_data[mode]
        ax.plot(d['positions'][:, 0], d['positions'][:, 1],
                color=colors[mode], label=plot_labels[mode], linewidth=1.2, alpha=0.85)
    ax.scatter(wp_arr[:, 0], wp_arr[:, 1], c='red', s=80, marker='^', zorder=5)
    for i, wp in enumerate(WAYPOINTS):
        ax.annotate(f'{i+1}', (wp[0], wp[1]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    ax.scatter([0], [0], c='black', s=80, marker='o', zorder=5, label='Start')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Racing Trajectory')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # --- Panel 3 (1,0): Altitude vs time ---
    ax = axes[1, 0]
    for mode in modes:
        d = all_data[mode]
        ax.plot(d['times'], -d['positions'][:, 2], color=colors[mode],
                label=plot_labels[mode], linewidth=1.2, alpha=0.85)
    wp_alts = sorted(set(-wp[2] for wp in WAYPOINTS))
    for alt in wp_alts:
        ax.axhline(y=alt, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
        ax.text(0.5, alt, f'{alt:.0f}m', fontsize=7, alpha=0.5, va='bottom')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4 (1,1): Speed vs time ---
    ax = axes[1, 1]
    for mode in modes:
        d = all_data[mode]
        speeds = np.linalg.norm(d['velocities'], axis=1)
        ax.plot(d['times'], speeds, color=colors[mode], label=plot_labels[mode],
                linewidth=1.0, alpha=0.85)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 5 (2,0): Tracking error vs time ---
    ax = axes[2, 0]
    for mode in modes:
        d = all_data[mode]
        errs = np.linalg.norm(d['positions'] - d['targets'], axis=1)
        ax.plot(d['times'], errs, color=colors[mode], label=plot_labels[mode],
                linewidth=1.0, alpha=0.85)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error (m)')
    ax.set_title('Tracking Error Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 6 (2,1): Per-waypoint completion time (grouped bars) ---
    ax = axes[2, 1]
    max_segments = max(len(all_metrics[m]['wp_segment_times']) for m in modes)
    x = np.arange(max_segments)
    width = 0.18
    for i, mode in enumerate(modes):
        segs = all_metrics[mode]['wp_segment_times']
        padded = segs + [0] * (max_segments - len(segs))
        ax.bar(x + i * width, padded, width, label=plot_labels[mode],
               color=colors[mode], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'WP{i+1}' for i in range(max_segments)], fontsize=7, rotation=45, ha='right')
    ax.set_xlabel('Waypoint')
    ax.set_ylabel('Segment Time (s)')
    ax.set_title('Per-Waypoint Completion Time')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 7 (3,0): Normalised performance-score bar chart ---
    # Each metric is rescaled so that 100 = best performer, 0 = worst.
    # This avoids the mixed-unit problem where Control Smoothness (~50-200)
    # completely dwarfs speed values (~1-5) on a shared y-axis.
    ax = axes[3, 0]
    perf_metrics = ['Lap Time (s)', 'Avg Speed (m/s)', 'Max Speed (m/s)',
                    'Control Smoothness', 'RMS Tracking Error (m)']
    perf_labels  = ['Lap Time', 'Avg Speed', 'Max Speed', 'Smoothness', 'RMS Error']
    higher_better_bar = {'Avg Speed (m/s)', 'Max Speed (m/s)'}

    # Compute 0-100 performance scores per metric
    perf_scores = {mode: [] for mode in modes}
    for metric in perf_metrics:
        vals = [all_metrics[m][metric] for m in modes]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax != vmin else 1.0
        for mode in modes:
            raw = (all_metrics[mode][metric] - vmin) / rng   # 0..1
            score = raw if metric in higher_better_bar else 1.0 - raw
            perf_scores[mode].append(round(score * 100, 1))

    x = np.arange(len(perf_metrics))
    width = 0.16
    for i, mode in enumerate(modes):
        bars = ax.bar(x + i * width, perf_scores[mode], width,
                      label=plot_labels[mode], color=colors[mode], alpha=0.85)
        # Annotate bars with their score
        for bar, score in zip(bars, perf_scores[mode]):
            if score > 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f'{score:.0f}', ha='center', va='bottom',
                        fontsize=5.5, color=colors[mode])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(perf_labels, fontsize=8)
    ax.set_ylabel('Performance Score (0=worst, 100=best)', fontsize=7)
    ax.set_ylim(0, 115)
    ax.set_title('Performance Scores (all metrics on same scale)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 8 (3,1): Radar/spider chart ---
    axes[3, 1].remove()
    ax_radar = fig.add_subplot(4, 2, 8, projection='polar')
    radar_metrics = ['Lap Time (s)', 'RMS Tracking Error (m)', 'Control Smoothness',
                     'Total Energy (N·s)', 'Avg Speed (m/s)', 'Max Speed (m/s)']
    radar_labels_text = ['Lap\nTime', 'RMS\nError', 'Smooth-\nness', 'Energy', 'Avg\nSpeed', 'Max\nSpeed']
    # "higher is better" for speed metrics; "lower is better" for the rest
    higher_better_radar = {'Avg Speed (m/s)', 'Max Speed (m/s)'}

    normalized = {mode: [] for mode in modes}
    for metric in radar_metrics:
        vals = [all_metrics[m][metric] for m in modes]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax != vmin else 1.0
        for mode in modes:
            raw = (all_metrics[mode][metric] - vmin) / rng
            if metric not in higher_better_radar:
                raw = 1.0 - raw   # invert: lower is better → larger polygon = better
            normalized[mode].append(raw)

    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)

    # Explicit radial grid with readable labels; position labels away from spokes
    ax_radar.set_rgrids([0.25, 0.5, 0.75, 1.0],
                        labels=['0.25', '0.50', '0.75', '1.00'],
                        angle=15, fontsize=7, color='#666677')
    ax_radar.set_ylim(0, 1.1)

    for mode in modes:
        values = normalized[mode] + normalized[mode][:1]
        ax_radar.plot(angles, values, color=colors[mode], linewidth=1.8,
                      label=plot_labels[mode])
        ax_radar.fill(angles, values, color=colors[mode], alpha=0.12)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels_text, fontsize=7.5)

    ax_radar.set_title('Performance Radar\n(outer = better)', fontsize=10, pad=14)

    # Legend placed below the polar axes so it never clips outside the figure
    ax_radar.legend(
        fontsize=7.5, loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=2, framealpha=0.7,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(os.path.dirname(__file__), 'benchmark_racing.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BENCHMARK: Racing Course — Adaptive vs Standalone Controllers")
    print("=" * 70)

    print(f"\nRacing Course: {len(WAYPOINTS)} waypoints (figure-8 with altitude changes)")
    print("  Faster physics: max_vel=5.0, max_accel=4.0, vert_clip=±3.0, damping=0.98")
    print("  Wind: time-varying profile (0->2.5->7->11->1.5 m/s, sweeps PID/MPC/H-inf tiers)")
    print(f"  Max duration: {MAX_STEPS * DT:.0f}s\n")

    print("  Waypoints:")
    wp_names = ['Start', 'Accelerate NE+climb', 'Peak altitude', 'Turn south+descend',
                'Sweep SW', 'Low altitude run', 'Back toward start', 'Climb NE',
                'High-speed cruise', 'Return to start']
    for i, (wp, name) in enumerate(zip(WAYPOINTS, wp_names)):
        print(f"    WP{i+1:>2}: [{wp[0]:>5.0f}, {wp[1]:>5.0f}, {wp[2]:>5.0f}]  {name}")
    print()

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
              f"Lap: {metrics['Lap Time (s)']:.1f}s, "
              f"Avg: {metrics['Avg Speed (m/s)']:.2f} m/s)")

    if len(all_metrics) < 2:
        print("\nNot enough scenarios completed for comparison.")
        return

    print_comparison_table(all_metrics)

    # Qualitative summary
    print("\n" + "=" * 70)
    print("QUALITATIVE SUMMARY")
    print("=" * 70)

    lap_vals = {m: all_metrics[m]['Lap Time (s)'] for m in all_metrics}
    speed_vals = {m: all_metrics[m]['Avg Speed (m/s)'] for m in all_metrics}
    rms_vals = {m: all_metrics[m]['RMS Tracking Error (m)'] for m in all_metrics}
    wp_vals = {m: all_metrics[m]['Waypoints Completed'] for m in all_metrics}

    best_lap = min(lap_vals, key=lap_vals.get)
    best_speed = max(speed_vals, key=speed_vals.get)
    best_rms = min(rms_vals, key=rms_vals.get)
    best_wp = max(wp_vals, key=wp_vals.get)

    print(f"  Fastest lap time       : {mode_labels[best_lap]} ({lap_vals[best_lap]:.1f}s)")
    print(f"  Highest avg speed      : {mode_labels[best_speed]} ({speed_vals[best_speed]:.2f} m/s)")
    print(f"  Best tracking accuracy : {mode_labels[best_rms]} (RMS {rms_vals[best_rms]:.3f}m)")
    print(f"  Most waypoints reached : {mode_labels[best_wp]} ({wp_vals[best_wp]}/{len(WAYPOINTS)})")

    wins = {m: 0 for m in all_metrics}
    for best in [best_lap, best_speed, best_rms, best_wp]:
        wins[best] += 1
    overall = max(wins, key=wins.get)
    print(f"\n  Overall winner: {mode_labels[overall]} ({wins[overall]}/4 categories)")

    print("\nGenerating racing benchmark plots...")
    generate_plots(all_data, all_metrics)

    print("\n" + "=" * 70)
    print("RACING BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
