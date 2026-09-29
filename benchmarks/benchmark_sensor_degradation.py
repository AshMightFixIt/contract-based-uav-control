"""
Benchmark: Sensor Degradation — Contract-Based Adaptive vs Standalone Controllers

Runs the same 7-phase sensor degradation scenario with 4 configurations:
  1. Adaptive (full contract-based switching + EKF fusion mode changes)
  2. PID-only (forced)
  3. MPC-only (forced)
  4. H-inf-only (forced)

All configurations use the same EKF + CBF safety pipeline; only the controller
selection is locked. This isolates the controller's resilience to degraded
sensor input (noisy/biased/missing measurements).

Produces comparison metrics table + outputs/benchmark_sensor_degradation.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import logging

logging.basicConfig(level=logging.ERROR, format='%(message)s')

from contract_uav_core.core import AdaptiveDroneController
from contract_uav_core.control.supervisor import FlightMode
from contract_uav_core.sim.sensor_faults import SensorFaultInjector, create_standard_degradation_schedule
from contract_uav_core.sim.outputs import output_file


# ---------------------------------------------------------------------------
# DroneSimulation (same as demos)
# ---------------------------------------------------------------------------
class DroneSimulation:
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
    np.array([5.0, 0.0, -5.0]),
    np.array([5.0, 5.0, -7.0]),
    np.array([0.0, 5.0, -6.0]),
    np.array([0.0, 0.0, -5.0]),
]

CONSTANT_WIND = np.array([1.0, 0.5, 0.0])  # Mild, to isolate sensor effects

INITIAL_CONDITIONS = {
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

MAX_STEPS = 9000  # 180 s
DT = 0.02

# Phase info for plotting
PHASE_BOUNDARIES = [0, 20, 35, 45, 60, 72, 85, 180]
PHASE_LABELS = ['Nominal', 'GPS Degrade', 'GPS Recovery', 'IMU Degrade',
                'Combined Fail', 'Partial Recov', 'Full Recovery']
PHASE_COLORS = ['#e8f5e9', '#fff3e0', '#e8f5e9', '#fff3e0',
                '#ffebee', '#fff3e0', '#e8f5e9']


# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------
def run_scenario(mode: str):
    """
    mode: 'adaptive' | 'PID' | 'MPC' | 'Hinf'
    """
    np.random.seed(42)

    drone = DroneSimulation()
    controller = AdaptiveDroneController(dt=DT, use_horizon_planner=(mode == 'adaptive'))
    fault_injector = SensorFaultInjector(create_standard_degradation_schedule())

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
    targets, fusion_modes, phases = [], [], []
    gps_health_hist, imu_health_hist = [], []

    mission_complete = False
    mission_complete_time = None
    was_degraded = False

    for step in range(MAX_STEPS):
        t = step * DT
        times.append(t)

        drone.set_wind(CONSTANT_WIND)
        sensors_clean = drone.get_sensor_data()
        sensors = fault_injector.inject(sensors_clean, t)

        control, telemetry = controller.control_step(sensors)
        drone.update(control)

        # EKF fusion mode
        gps_ok, imu_ok = controller.ekf.check_sensor_contracts(controller.time - DT)
        if gps_ok and imu_ok:
            fusion = 'full'
        elif imu_ok:
            fusion = 'imu_only'
        else:
            fusion = 'dead_reckoning'

        # Resume TRACK after recovery (same logic as demo)
        sup_status = controller.supervisor.get_status()
        wp_idx = sup_status['waypoint_idx']
        if (telemetry['flight_mode'] == 'hover' and gps_ok and imu_ok and
                was_degraded and wp_idx < len(WAYPOINTS) and not mission_complete):
            controller.supervisor.mode = FlightMode.TRACK
            was_degraded = False
        if not gps_ok or not imu_ok:
            was_degraded = True

        health = fault_injector.get_sensor_health(t)
        phase = fault_injector.get_phase_name(t)

        # Record
        positions.append(drone.position.copy())
        velocities.append(drone.velocity.copy())
        controllers_used.append(telemetry['active_controller'])
        thrusts.append(control['thrust'])
        torques_list.append(control.get('torques', np.zeros(3)).copy())
        attitudes.append(drone.attitude.copy())
        cbf_flags.append(telemetry['cbf_intervened'])
        waypoint_indices.append(wp_idx)
        fusion_modes.append(fusion)
        phases.append(phase)
        gps_health_hist.append(health['gps'])
        imu_health_hist.append(health['imu'])

        if wp_idx < len(WAYPOINTS):
            targets.append(WAYPOINTS[wp_idx].copy())
        else:
            targets.append(WAYPOINTS[-1].copy())

        if telemetry['flight_mode'] == 'hover' and wp_idx >= len(WAYPOINTS) - 1 and not mission_complete:
            mission_complete = True
            mission_complete_time = t

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
        'fusion_modes': fusion_modes,
        'phases': phases,
        'gps_health': np.array(gps_health_hist),
        'imu_health': np.array(imu_health_hist),
        'mission_complete': mission_complete,
        'mission_complete_time': mission_complete_time,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(data):
    pos = data['positions']
    vel = data['velocities']
    tgt = data['targets']
    thrusts = data['thrusts']
    torques = data['torques']
    att = data['attitudes']
    cbf = data['cbf_flags']
    wp_idx = data['waypoint_indices']
    times = data['times']
    fusions = data['fusion_modes']

    tracking_errors = np.linalg.norm(pos - tgt, axis=1)
    speeds = np.linalg.norm(vel, axis=1)

    # Per-phase RMS errors
    phase_rms = {}
    for i, (t0, t1) in enumerate(zip(PHASE_BOUNDARIES[:-1], PHASE_BOUNDARIES[1:])):
        mask = (times >= t0) & (times < t1) & (times <= times[-1])
        if mask.any():
            phase_rms[PHASE_LABELS[i]] = float(np.sqrt(np.mean(tracking_errors[mask]**2)))

    thrust_diffs = np.abs(np.diff(thrusts))
    torque_diffs = np.linalg.norm(np.diff(torques, axis=0), axis=1)
    smoothness = np.sum(thrust_diffs) + np.sum(torque_diffs)

    total_energy = np.sum(thrusts) * DT
    control_effort = np.sum(np.abs(thrusts - 0.5) + np.linalg.norm(torques, axis=1)) * DT
    tilt_angles = np.sqrt(att[:, 0]**2 + att[:, 1]**2)

    max_wp = max(wp_idx)
    waypoints_completed = min(max_wp, len(WAYPOINTS))
    if data['mission_complete'] and data['mission_complete_time'] is not None:
        mission_time = data['mission_complete_time']
    else:
        mission_time = times[-1]

    cbf_count = sum(1 for f in cbf if f)

    # Time spent in degraded fusion modes
    imu_only_pct = 100.0 * sum(1 for f in fusions if f == 'imu_only') / len(fusions)
    dead_reckoning_pct = 100.0 * sum(1 for f in fusions if f == 'dead_reckoning') / len(fusions)

    # Max position drift during combined failure phase (60-72s)
    mask_combined = (times >= 60) & (times < 72) & (times <= times[-1])
    if mask_combined.any():
        max_drift_combined = float(np.max(tracking_errors[mask_combined]))
    else:
        max_drift_combined = 0.0

    metrics = {
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
        'Max Drift (Combined)': max_drift_combined,
        'IMU-Only Time (%)': imu_only_pct,
        'Dead Reckoning (%)': dead_reckoning_pct,
    }
    metrics['phase_rms'] = phase_rms
    return metrics


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def print_comparison_table(all_metrics):
    modes = list(all_metrics.keys())
    # Exclude phase_rms from main table
    metric_names = [k for k in next(iter(all_metrics.values())).keys() if k != 'phase_rms']

    higher_is_better = {'Waypoints Completed'}

    col_width = 18
    header = f"{'Metric':<28}" + "".join(f"{m:>{col_width}}" for m in modes)
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("SENSOR DEGRADATION BENCHMARK RESULTS")
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
    print("  * = best   ! = worst\n")

    # Per-phase RMS table
    print("PER-PHASE RMS TRACKING ERROR (m):")
    phase_header = f"{'Phase':<20}" + "".join(f"{m:>{col_width}}" for m in modes)
    print(phase_header)
    print("-" * len(phase_header))
    for phase_name in PHASE_LABELS:
        row = f"{phase_name:<20}"
        vals = {}
        for m in modes:
            pr = all_metrics[m].get('phase_rms', {})
            vals[m] = pr.get(phase_name, float('nan'))
        valid_vals = [v for v in vals.values() if not np.isnan(v)]
        best_v = min(valid_vals) if valid_vals else float('nan')
        worst_v = max(valid_vals) if valid_vals else float('nan')
        for m in modes:
            v = vals[m]
            if np.isnan(v):
                cell = "N/A"
            else:
                cell = f"{v:.3f}"
                if v == best_v and best_v != worst_v:
                    cell += " *"
                elif v == worst_v and best_v != worst_v:
                    cell += " !"
            row += f"{cell:>{col_width}}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def generate_plots(all_data, all_metrics):
    """Generate 8-panel comparison figure (4x2 layout)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle('Sensor Degradation Benchmark: Adaptive vs Standalone', fontsize=14, fontweight='bold', y=0.98)

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

    def add_phase_shading(ax):
        for i in range(len(PHASE_LABELS)):
            t0 = PHASE_BOUNDARIES[i]
            t1 = PHASE_BOUNDARIES[i + 1]
            ax.axvspan(t0, t1, alpha=0.12, color=PHASE_COLORS[i])
            mid = (t0 + t1) / 2
            ax.text(mid, 0.98, PHASE_LABELS[i], ha='center', va='top', fontsize=5.5,
                    alpha=0.7, transform=ax.get_xaxis_transform())

    # --- Panel 1 (0,0): Tracking error + phase shading ---
    ax = axes[0, 0]
    for mode in modes:
        d = all_data[mode]
        errs = np.linalg.norm(d['positions'] - d['targets'], axis=1)
        ax.plot(d['times'], errs, color=colors[mode], label=labels[mode], linewidth=1.0, alpha=0.85)
    add_phase_shading(ax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error (m)')
    ax.set_title('Tracking Error Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2 (0,1): XY trajectory with dashed WP connections ---
    ax = axes[0, 1]
    wp_arr = np.array(WAYPOINTS)
    all_wp_x = [0] + [wp[0] for wp in WAYPOINTS]
    all_wp_y = [0] + [wp[1] for wp in WAYPOINTS]
    ax.plot(all_wp_x, all_wp_y, 'k--', alpha=0.3, linewidth=1.0, zorder=1)
    for mode in modes:
        d = all_data[mode]
        ax.plot(d['positions'][:, 0], d['positions'][:, 1],
                color=colors[mode], label=labels[mode], linewidth=1.2, alpha=0.85)
        # Direction arrowheads
        n = len(d['positions'])
        for frac in [0.25, 0.5, 0.75]:
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
    wp_alts = sorted(set(-wp[2] for wp in WAYPOINTS))
    for alt in wp_alts:
        ax.axhline(y=alt, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
        ax.text(0.5, alt, f'{alt:.0f}m', fontsize=7, alpha=0.5, va='bottom')
    add_phase_shading(ax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 5 (2,0): Per-phase RMS bar chart ---
    ax = axes[2, 0]
    phase_names_short = ['Nominal', 'GPS Deg', 'GPS Rec', 'IMU Deg', 'Combined', 'Partial', 'Full Rec']
    x = np.arange(len(PHASE_LABELS))
    width = 0.18
    for i, mode in enumerate(modes):
        pr = all_metrics[mode].get('phase_rms', {})
        vals = [pr.get(p, 0) for p in PHASE_LABELS]
        ax.bar(x + i * width, vals, width, label=labels[mode], color=colors[mode], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(phase_names_short, fontsize=7, rotation=20, ha='right')
    ax.set_ylabel('RMS Tracking Error (m)')
    ax.set_title('Per-Phase RMS Tracking Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 6 (2,1): Key metrics bar chart ---
    ax = axes[2, 1]
    bar_metrics = ['RMS Tracking Error (m)', 'Max Drift (Combined)', 'CBF Interventions', 'Control Smoothness']
    bar_labels_short = ['RMS Error\n(m)', 'Max Drift\n(Combined)', 'CBF\nInterventions', 'Control\nSmoothness']
    x = np.arange(len(bar_metrics))
    width = 0.18
    for i, mode in enumerate(modes):
        vals = [all_metrics[mode][m] for m in bar_metrics]
        ax.bar(x + i * width, vals, width, label=labels[mode], color=colors[mode], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(bar_labels_short, fontsize=8)
    ax.set_title('Key Metrics Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 7 (3,0): Sensor health timeline ---
    ax = axes[3, 0]
    d = all_data[modes[0]]
    ax.plot(d['times'], d['gps_health'], 'b-', linewidth=2, label='GPS Health')
    ax.plot(d['times'], d['imu_health'], 'r-', linewidth=2, label='IMU Health')
    ax.fill_between(d['times'], 0, d['gps_health'], alpha=0.15, color='blue')
    ax.fill_between(d['times'], 0, d['imu_health'], alpha=0.15, color='red')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    for i in range(len(PHASE_LABELS)):
        t0 = PHASE_BOUNDARIES[i]
        t1 = min(PHASE_BOUNDARIES[i+1], d['times'][-1])
        if t0 < d['times'][-1]:
            mid = (t0 + t1) / 2
            ax.text(mid, 1.08, PHASE_LABELS[i], ha='center', va='bottom', fontsize=5.5, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Health (0-1)')
    ax.set_title('Sensor Health (shared degradation profile)')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.2)
    ax.grid(True, alpha=0.3)

    # --- Panel 8 (3,1): Radar/spider chart ---
    axes[3, 1].remove()
    ax_radar = fig.add_subplot(4, 2, 8, projection='polar')
    radar_metrics = ['RMS Tracking Error (m)', 'Max Drift (Combined)', 'Control Smoothness',
                     'Max Tilt (rad)', 'Control Effort', 'Total Energy (N·s)']
    radar_labels_text = ['RMS Error', 'Max Drift', 'Smoothness', 'Max Tilt', 'Ctrl Effort', 'Energy']
    # All "lower is better" — invert so bigger polygon = better
    normalized = {}
    for mode in modes:
        normalized[mode] = []
    for metric in radar_metrics:
        vals = [all_metrics[m][metric] for m in modes]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax != vmin else 1.0
        for mode in modes:
            normalized[mode].append(1.0 - (all_metrics[mode][metric] - vmin) / rng)

    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_rlabel_position(30)
    plt.setp(ax_radar.get_yticklabels(), fontsize=6)

    for mode in modes:
        values = normalized[mode] + normalized[mode][:1]
        ax_radar.plot(angles, values, color=colors[mode], linewidth=1.5, label=labels[mode])
        ax_radar.fill(angles, values, color=colors[mode], alpha=0.1)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels_text, fontsize=7)
    ax_radar.set_ylim(0, 1.1)
    ax_radar.set_title('Performance Radar (bigger = better)', fontsize=10, pad=20)
    ax_radar.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_file(__file__, 'benchmark_sensor_degradation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("BENCHMARK: Sensor Degradation — Adaptive vs Standalone Controllers")
    print("=" * 72)

    print("\nDegradation Schedule (7 phases):")
    for i, label in enumerate(PHASE_LABELS):
        print(f"  Phase {i+1}: {PHASE_BOUNDARIES[i]:>3}-{PHASE_BOUNDARIES[i+1]:>3}s  {label}")

    print(f"\nWaypoints: {len(WAYPOINTS)} (5m square pattern)")
    print(f"Wind: Mild constant [1.0, 0.5, 0.0] m/s (isolating sensor effects)")
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
              f"RMS: {metrics['RMS Tracking Error (m)']:.2f}m, "
              f"DR: {metrics['Dead Reckoning (%)']:.1f}%)")

    if len(all_metrics) < 2:
        print("\nNot enough scenarios completed for comparison.")
        return

    print_comparison_table(all_metrics)

    # Qualitative summary
    print("=" * 72)
    print("QUALITATIVE SUMMARY")
    print("=" * 72)

    rms_vals = {m: all_metrics[m]['RMS Tracking Error (m)'] for m in all_metrics}
    best_rms = min(rms_vals, key=rms_vals.get)
    drift_vals = {m: all_metrics[m]['Max Drift (Combined)'] for m in all_metrics}
    best_drift = min(drift_vals, key=drift_vals.get)
    wp_vals = {m: all_metrics[m]['Waypoints Completed'] for m in all_metrics}
    best_wp = max(wp_vals, key=wp_vals.get)
    smooth_vals = {m: all_metrics[m]['Control Smoothness'] for m in all_metrics}
    best_smooth = min(smooth_vals, key=smooth_vals.get)

    print(f"  Best tracking accuracy    : {mode_labels[best_rms]} (RMS {rms_vals[best_rms]:.3f}m)")
    print(f"  Best drift resilience     : {mode_labels[best_drift]} (max drift {drift_vals[best_drift]:.2f}m during combined failure)")
    print(f"  Most waypoints completed  : {mode_labels[best_wp]} ({wp_vals[best_wp]}/{len(WAYPOINTS)})")
    print(f"  Smoothest control         : {mode_labels[best_smooth]} (smoothness {smooth_vals[best_smooth]:.1f})")

    wins = {m: 0 for m in all_metrics}
    for _vals, best in [(rms_vals, best_rms), (drift_vals, best_drift),
                        (wp_vals, best_wp), (smooth_vals, best_smooth)]:
        wins[best] += 1
    overall = max(wins, key=wins.get)
    print(f"\n  Overall winner: {mode_labels[overall]} ({wins[overall]}/4 categories)")

    print("\nGenerating comparison plots...")
    generate_plots(all_data, all_metrics)

    print("\n" + "=" * 72)
    print("SENSOR DEGRADATION BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
