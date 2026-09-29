"""The old racing plant: the simplified drone, course, wind schedule and constants
of the racing benchmark.

It is kept here so that the racing benchmark (benchmarks/benchmark_racing.py)
and the live racing demo (demos/demo_racing_live.py) share one copy. The code
below was moved verbatim out of benchmark_racing.py, which the demo used to
import. Batch 3's P1.12 will replace this plant with sim/quadrotor.py.
"""

import numpy as np


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
