"""
Live Racing Visualization — 2×2 Speed Panels

Runs all 4 drone configurations simultaneously and shows a live 2×2 grid:

  Adaptive (top-left)   |  PID-only  (top-right)
  MPC-only (bottom-left)|  H-inf-only (bottom-right)

Each panel:
  • Racing course outline + numbered waypoint markers
  • Speed-heatmap trail   (plasma colormap: dark-purple = slow → yellow = fast)
  • Drone position dot    (coloured per controller)
  • Live speed readout    (top-left badge)
  • Speedometer bar gauge (bottom strip, fills left→right as speed increases)
  • Waypoint progress     (top-right)
  • Elapsed time          (bottom-right)

Wind profile (same for all drones — fair comparison):
   0–25 s  calm   0.5→2.5 m/s  PID territory
  25–55 s  moderate 2.5→7 m/s  MPC territory
  55–80 s  strong  7→11 m/s    H-inf territory
  80–100 s recovery 11→1.5 m/s back to PID/MPC
Adaptive panel switches controller automatically; forced panels show the cost of mismatch.

Controls:  close the window or press Esc to exit.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
logging.disable(logging.CRITICAL)

import numpy as np

# ── Import simulation modules BEFORE touching matplotlib.
# benchmark_racing.py calls matplotlib.use('Agg') at module level; we let it
# run first, then switch to an interactive backend before creating any figure.
from adaptive_control_system import AdaptiveDroneController
from benchmark_racing import (
    RacingDroneSimulation, WAYPOINTS,
    INITIAL_CONDITIONS, DT, MAX_STEPS,
    get_wind_at_time,
)

import matplotlib
import matplotlib.pyplot as plt

# Switch away from the Agg backend that benchmark_racing forced.
for _backend in ('TkAgg', 'Qt5Agg', 'WXAgg', 'GTK3Agg'):
    try:
        plt.switch_backend(_backend)
        break
    except Exception:
        continue

import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from collections import deque

# ── Configuration ────────────────────────────────────────────────────────────
MODES  = ['adaptive', 'PID', 'MPC', 'Hinf']
LABELS = {
    'adaptive': 'Adaptive',
    'PID':      'PID-only',
    'MPC':      'MPC-only',
    'Hinf':     'H-inf-only',
}
COLORS = {
    'adaptive': '#2196F3',
    'PID':      '#4CAF50',
    'MPC':      '#FF9800',
    'Hinf':     '#F44336',
}

TRAIL_LEN     = 350   # past positions kept in trail scatter
SIM_PER_FRAME = 8     # simulation steps between rendered frames
MAX_SPEED     = 5.0   # m/s — top of colour scale and gauge

# ── Build one simulation per mode ────────────────────────────────────────────
def _make_sim(mode):
    np.random.seed(42)
    drone = RacingDroneSimulation()
    ctrl  = AdaptiveDroneController(dt=DT, use_horizon_planner=(mode == 'adaptive'))
    mission = {'target_position': WAYPOINTS[0], 'waypoints': list(WAYPOINTS)}
    ctrl.pre_flight_check(INITIAL_CONDITIONS, mission)
    ctrl.start_mission()
    if mode != 'adaptive':
        ctrl.controller_switcher.last_switch_time = -10.0
        ctrl.controller_switcher.switch_to(mode, 0.0, f"force {mode}")
        orig_switch = ctrl.controller_switcher.switch_to
        ctrl.controller_switcher.switch_to = lambda *a, **kw: False
    return drone, ctrl

drones      = {}
controllers = {}
for m in MODES:
    drones[m], controllers[m] = _make_sim(m)

# ── Shared simulation state ───────────────────────────────────────────────────
trail  = {m: deque(maxlen=TRAIL_LEN) for m in MODES}   # (x, y, speed)
done   = {m: False for m in MODES}
step_n = [0]
sim_t  = [0.0]

# ── Figure geometry ───────────────────────────────────────────────────────────
wp_arr = np.array(WAYPOINTS)
_xr    = wp_arr[:, 0].ptp()
_yr    = wp_arr[:, 1].ptp()
_xpad  = max(_xr * 0.14, 4.0)
_ypad  = max(_yr * 0.14, 4.0)
X_LIM  = (wp_arr[:, 0].min() - _xpad, wp_arr[:, 0].max() + _xpad)
Y_LIM  = (wp_arr[:, 1].min() - _ypad, wp_arr[:, 1].max() + _ypad)

PANEL_BG = '#0e0e1e'
TRACK_C  = '#2a3a55'
WP_C     = '#ff5555'

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.patch.set_facecolor('#07070f')
fig.suptitle(
    'Live Racing  —  Speed Heatmap  (plasma: dark = slow  ▶  bright = fast)',
    color='#ccddee', fontsize=12, fontweight='bold', y=0.99,
)

norm = Normalize(vmin=0, vmax=MAX_SPEED)
cmap = plt.cm.plasma

# Gauge layout (axes-fraction units)
GAUGE_X0 = 0.04
GAUGE_Y0 = 0.025
GAUGE_W  = 0.92
GAUGE_H  = 0.042

# Course path arrays (closed loop)
course_xs = [wp[0] for wp in WAYPOINTS] + [WAYPOINTS[0][0]]
course_ys = [wp[1] for wp in WAYPOINTS] + [WAYPOINTS[0][1]]

ax_of = {m: axes[i // 2, i % 2] for i, m in enumerate(MODES)}

# Per-panel artists
scatter_art = {}
drone_dot   = {}
speed_txt   = {}
wp_txt      = {}
time_txt    = {}
gauge_fill  = {}
gauge_spd_txt = {}

for m in MODES:
    ax = ax_of[m]
    c  = COLORS[m]

    # Axes cosmetics
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(colors='#3a4a5a', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#1a2a3a')

    # Course outline
    ax.plot(course_xs, course_ys, '--', color=TRACK_C, lw=1.2, alpha=0.9, zorder=1)

    # Waypoint markers
    ax.scatter(wp_arr[:, 0], wp_arr[:, 1],
               c=WP_C, s=22, marker='^', zorder=4, alpha=0.85)
    for i, wp in enumerate(WAYPOINTS):
        ax.text(wp[0] + 0.6, wp[1] + 0.6, str(i + 1),
                fontsize=6, color='#bb8888', zorder=5)

    # Title
    ax.set_title(LABELS[m], color=c, fontsize=11, fontweight='bold', pad=5)

    # ── Speed-heatmap trail scatter ──
    sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm,
                    s=10, alpha=0.88, zorder=3, linewidths=0)
    scatter_art[m] = sc

    # ── Drone position dot ──
    dot, = ax.plot([], [], 'o',
                   color=c, markersize=10,
                   markeredgecolor='white', markeredgewidth=1.3,
                   zorder=10)
    drone_dot[m] = dot

    # ── Speed badge (top-left) ──
    stxt = ax.text(
        0.04, 0.96, '0.00 m/s',
        transform=ax.transAxes,
        color='white', fontsize=11, fontweight='bold',
        va='top', zorder=11,
        bbox=dict(boxstyle='round,pad=0.28',
                  facecolor='#00000099', edgecolor=c, linewidth=1.4),
    )
    speed_txt[m] = stxt

    # ── Waypoint progress (top-right) ──
    wt = ax.text(
        0.96, 0.96, 'WP  0/10',
        transform=ax.transAxes,
        color='#9aabbb', fontsize=8, va='top', ha='right', zorder=11,
    )
    wp_txt[m] = wt

    # ── Elapsed time (lower-right, above gauge) ──
    tt = ax.text(
        0.96, 0.09, 't = 0.0 s',
        transform=ax.transAxes,
        color='#556677', fontsize=8, va='bottom', ha='right', zorder=11,
    )
    time_txt[m] = tt

    # ── Gauge background (full-width grey bar) ──
    bg = mpatches.Rectangle(
        (GAUGE_X0, GAUGE_Y0), GAUGE_W, GAUGE_H,
        transform=ax.transAxes,
        facecolor='#1a2233', edgecolor='#2a3a4a', linewidth=0.8,
        zorder=8, clip_on=False,
    )
    ax.add_patch(bg)

    # ── Gauge fill (starts at minimum width) ──
    gf = mpatches.Rectangle(
        (GAUGE_X0, GAUGE_Y0), 1e-4, GAUGE_H,
        transform=ax.transAxes,
        facecolor=c, alpha=0.88,
        zorder=9, clip_on=False,
    )
    ax.add_patch(gf)
    gauge_fill[m] = gf

    # ── Speed value on gauge ──
    gspd = ax.text(
        GAUGE_X0 + GAUGE_W / 2, GAUGE_Y0 + GAUGE_H / 2,
        '0.00 / 5.00 m/s',
        transform=ax.transAxes,
        color='white', fontsize=7, va='center', ha='center',
        zorder=10, alpha=0.9,
    )
    gauge_spd_txt[m] = gspd

# ── Shared colourbar on right margin ─────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                    orientation='vertical', fraction=0.018, pad=0.02)
cbar.set_label('Speed (m/s)', color='#9aabbb', fontsize=9, labelpad=6)
cbar.ax.yaxis.set_tick_params(color='#9aabbb', labelsize=8)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#9aabbb')
cbar.outline.set_edgecolor('#334455')

fig.subplots_adjust(left=0.05, right=0.93, top=0.94, bottom=0.06, hspace=0.28, wspace=0.22)

# ── Simulation step ───────────────────────────────────────────────────────────
def _advance():
    """Advance all unfinished simulations by one time step."""
    t = step_n[0] * DT
    step_n[0] += 1
    sim_t[0] = t

    for m in MODES:
        if done[m]:
            continue
        drone = drones[m]
        ctrl  = controllers[m]
        drone.set_wind(get_wind_at_time(t))
        sensors          = drone.get_sensor_data()
        control, telemetry = ctrl.control_step(sensors)
        drone.update(control)

        speed = float(np.linalg.norm(drone.velocity))
        trail[m].append((drone.position[0], drone.position[1], speed))

        sup    = ctrl.supervisor.get_status()
        wp_idx = sup['waypoint_idx']
        if telemetry['flight_mode'] == 'hover' and wp_idx >= len(WAYPOINTS) - 1:
            done[m] = True
        if step_n[0] >= MAX_STEPS:
            done[m] = True


# ── Animation callback ────────────────────────────────────────────────────────
def _animate(frame):
    for _ in range(SIM_PER_FRAME):
        _advance()
        if all(done.values()):
            break

    artists = []
    for m in MODES:
        tr = trail[m]
        if not tr:
            artists += [scatter_art[m], drone_dot[m],
                        speed_txt[m], wp_txt[m], time_txt[m],
                        gauge_fill[m], gauge_spd_txt[m]]
            continue

        pts           = np.array(tr)          # (n, 3): x, y, speed
        xs, ys, spds  = pts[:, 0], pts[:, 1], pts[:, 2]

        # Trail
        scatter_art[m].set_offsets(np.c_[xs, ys])
        scatter_art[m].set_array(spds)

        # Drone dot
        drone_dot[m].set_data([xs[-1]], [ys[-1]])

        # Speed badge
        cur_spd = float(spds[-1])
        speed_txt[m].set_text(f'{cur_spd:.2f} m/s')

        # Speedometer gauge
        frac = min(cur_spd / MAX_SPEED, 1.0)
        gauge_fill[m].set_width(max(frac * GAUGE_W, 1e-4))
        gauge_spd_txt[m].set_text(f'{cur_spd:.2f} / {MAX_SPEED:.1f} m/s')

        # WP counter
        sup    = controllers[m].supervisor.get_status()
        wp_idx = sup['waypoint_idx']
        wp_txt[m].set_text(f'WP {min(wp_idx + 1, len(WAYPOINTS)):2d}/{len(WAYPOINTS)}')

        # Time
        time_txt[m].set_text(f't = {sim_t[0]:.1f} s')

        artists += [scatter_art[m], drone_dot[m],
                    speed_txt[m], wp_txt[m], time_txt[m],
                    gauge_fill[m], gauge_spd_txt[m]]

    return artists


# ── Key handler ───────────────────────────────────────────────────────────────
def _on_key(event):
    if event.key == 'escape':
        plt.close('all')


fig.canvas.mpl_connect('key_press_event', _on_key)

# ── Run ───────────────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig, _animate,
    interval=33,            # ~30 fps target
    blit=True,
    cache_frame_data=False,
)

print("=" * 60)
print("Live Racing Visualization -- 2x2 Speed Panels")
print("=" * 60)
print(f"  4 drones racing {len(WAYPOINTS)}-waypoint figure-8 course simultaneously")
print(f"  Colour scale:  0 m/s (dark purple) -> {MAX_SPEED:.1f} m/s (bright yellow)")
print(f"  Trail length:  {TRAIL_LEN} steps  |  {SIM_PER_FRAME} sim steps / frame")
print()
print("  Close window or press Esc to exit.")
print()

plt.show()

for m in MODES:
    try:
        controllers[m].stop_mission()
    except Exception:
        pass
