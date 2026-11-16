"""ContractAwareEKF predicts with the command it was given (W4-08b T1).

Before W4-08b a velocity without GPS kept its last value whatever the controller
commanded, so in closed loop the vehicle ran away (46.6 m up in an 8 s outage on
sim/quadrotor.py). Now, without GPS velocity, the velocity follows the acceleration
of the command just sent plus a bias learned while GPS was there; with GPS
velocity the constant-velocity model is unchanged. The closed-loop checks are in
tests/gate1/test_g1_6_closed_loop_gps_outage.py.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

DT = 0.02
G = 9.81
GPS = {'satellites': 12, 'hdop': 0.8, 'valid': True}
NO_GPS = {'satellites': 0, 'hdop': 99.9, 'valid': False}
IMU = {'calibrated': True, 'temperature': 25.0, 'valid': True}


def make_ekf(**kw):
    from contract_uav_core.contracts.monitor import HierarchicalContractMonitor
    from contract_uav_core.estimation.ekf import ContractAwareEKF
    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()
    return ContractAwareEKF(monitor, **kw)


class Truth:
    """A point mass whose acceleration is the command plus a hidden bias, with measurements."""

    def __init__(self, bias=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), seed=0):
        self.ekf = make_ekf()
        self.pos = np.array([0.0, 0.0, -5.0])
        self.vel = np.array(velocity, dtype=float)
        self.bias = np.array(bias, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

    def step(self, command, gps=True, accel=None):
        """The truth moves under command (+ bias, or `accel` if given); then predict + update."""
        a = command + self.bias if accel is None else accel
        self.vel = self.vel + a * DT
        self.pos = self.pos + self.vel * DT
        self.t += DT
        meas = {'imu_attitude': self.rng.standard_normal(3) * 0.001,
                'imu_rates': self.rng.standard_normal(3) * 0.001}
        if gps:
            meas['gps_position'] = self.pos + self.rng.standard_normal(3) * 0.01
            meas['gps_velocity'] = self.vel + self.rng.standard_normal(3) * 0.01
        self.ekf.predict(DT, accel_cmd=command)
        self.ekf.update(meas, {'gps': GPS if gps else NO_GPS, 'imu': IMU}, self.t)

    def error(self):
        return self.ekf.get_state()['position'] - self.pos


# --- the inverse of the command ------------------------------------------------------------------

def test_attitude_thrust_to_accel_is_the_reference_plants_thrust_model():
    """frames.attitude_thrust_to_accel equals sim/quadrotor.py's acceleration (no drag, no lag)."""
    from contract_uav_core.frames import attitude_thrust_to_accel
    from contract_uav_core.interfaces import AttitudeThrustCommand
    from contract_uav_core.sim.quadrotor import QuadrotorParams, QuadrotorPlant
    rng = np.random.default_rng(3)
    for hover in (0.5, 0.35):
        params = QuadrotorParams(mass=2.0, hover_thrust=hover, max_thrust=1.0,
                                 attitude_time_constant=DT, drag_coefficient=0.0)   # no lag
        for _ in range(20):
            attitude = np.r_[rng.uniform(-0.3, 0.3, 2), rng.uniform(-np.pi, np.pi)]
            thrust = rng.uniform(0.0, 1.0)
            plant = QuadrotorPlant(params, position=(0.0, 0.0, -50.0))
            v0 = plant.state.velocity
            v1 = plant.step(AttitudeThrustCommand(attitude=attitude, thrust=thrust), DT).velocity
            assert np.allclose((v1 - v0) / DT, attitude_thrust_to_accel(attitude, thrust, hover), atol=1e-9)


def test_attitude_thrust_to_accel_signs():
    from contract_uav_core.frames import attitude_thrust_to_accel
    assert np.allclose(attitude_thrust_to_accel([0, 0, 0], 0.5, 0.5), 0.0)            # hover
    assert np.allclose(attitude_thrust_to_accel([0, 0, 0], 0.0, 0.5), [0, 0, G])       # free fall
    assert np.allclose(attitude_thrust_to_accel([0, 0, 0], -0.2, 0.5), [0, 0, G])      # no pulling
    assert attitude_thrust_to_accel([0, 0, 0], 0.6, 0.5)[2] < 0                        # climbs (NED)
    nose_down = attitude_thrust_to_accel([0, -0.1, 0], 0.5, 0.5)
    assert nose_down[0] > 0.9 and abs(nose_down[1]) < 1e-12                           # forward = north
    facing_east = attitude_thrust_to_accel([0, -0.1, np.pi / 2], 0.5, 0.5)
    assert facing_east[1] > 0.9 and abs(facing_east[0]) < 1e-12                       # forward = east
    assert attitude_thrust_to_accel([0.1, 0, 0], 0.5, 0.5)[1] > 0.9                    # right roll: east


def test_attitude_thrust_to_accel_inverts_the_tiers_mapping_for_small_demands():
    """The tiers' level-flight law (accel_to_attitude, accel_to_thrust) is inverted to first order."""
    from contract_uav_core.frames import accel_to_attitude, accel_to_thrust, attitude_thrust_to_accel
    for demand, yaw in [((0.5, -0.3, -0.4), 0.0), ((-0.2, 0.4, 0.3), 1.2)]:
        attitude = accel_to_attitude(np.array(demand), yaw, 0.22)
        thrust = accel_to_thrust(demand[2], 0.5)
        assert np.allclose(attitude_thrust_to_accel(attitude, thrust, 0.5), demand, atol=0.02)


def test_attitude_thrust_to_accel_clips_at_max_thrust():
    """N-2 (review R-08b): above the vehicle's max_thrust the plant gives max_thrust; so does the
    inverse when max_thrust is known (e.g. the CBF's 0.95 with an airframe of 0.8)."""
    from contract_uav_core.frames import attitude_thrust_to_accel
    from contract_uav_core.interfaces import AttitudeThrustCommand
    from contract_uav_core.sim.quadrotor import QuadrotorParams, QuadrotorPlant
    params = QuadrotorParams(mass=2.0, hover_thrust=0.5, max_thrust=0.8, attitude_time_constant=DT,
                             drag_coefficient=0.0)
    plant = QuadrotorPlant(params, position=(0.0, 0.0, -50.0))
    attitude = np.array([0.05, -0.1, 0.4])
    v1 = plant.step(AttitudeThrustCommand(attitude=attitude, thrust=0.95), DT).velocity
    assert np.allclose(v1 / DT, attitude_thrust_to_accel(attitude, 0.95, 0.5, 0.8), atol=1e-9)
    assert np.array_equal(attitude_thrust_to_accel(attitude, 0.95, 0.5, 0.8), attitude_thrust_to_accel(attitude, 0.8, 0.5))
    assert np.array_equal(attitude_thrust_to_accel(attitude, 0.95, 0.5), attitude_thrust_to_accel(attitude, 0.95, 0.5, None))


def test_commanded_acceleration_uses_the_final_thrust_at_the_estimated_attitude():
    from contract_uav_core.estimation.command import commanded_acceleration
    from contract_uav_core.frames import attitude_thrust_to_accel
    control = {'thrust': 0.55, 'desired_attitude': np.array([0.0, -0.2, 0.0])}
    estimated = np.array([0.0, -0.05, 0.3])
    assert np.allclose(commanded_acceleration(control, estimated, True, 0.5),
                       attitude_thrust_to_accel(estimated, 0.55, 0.5))
    # without a usable IMU, the commanded attitude
    assert np.allclose(commanded_acceleration(control, estimated, False, 0.5),
                       attitude_thrust_to_accel(control['desired_attitude'], 0.55, 0.5))
    assert np.allclose(commanded_acceleration({'thrust': 0.95}, estimated, True, 0.5, 0.8),
                       attitude_thrust_to_accel(estimated, 0.8, 0.5))
    assert commanded_acceleration({}, estimated, True, 0.5) is None
    assert commanded_acceleration({'thrust': 0.5}, estimated, False, 0.5) is None


# --- predict with a command ----------------------------------------------------------------------

def lose_gps(ekf, steps=1):
    for k in range(steps):
        ekf.predict(DT)
        ekf.update({'imu_attitude': np.zeros(3), 'imu_rates': np.zeros(3)}, {'gps': NO_GPS, 'imu': IMU}, k * DT)
    assert not ekf.velocity_observed.any()


def test_without_gps_velocity_the_command_drives_the_velocity_and_position():
    ekf = make_ekf()
    lose_gps(ekf)
    x0 = ekf.x.copy()
    a = np.array([0.4, -0.2, -1.0])
    ekf.predict(DT, accel_cmd=a)
    assert np.allclose(ekf.x[3:6], x0[3:6] + a * DT, rtol=0, atol=1e-15)
    assert np.allclose(ekf.x[0:3], x0[0:3] + x0[3:6] * DT + 0.5 * a * DT ** 2, rtol=0, atol=1e-15)
    assert np.array_equal(ekf.x[6:12], x0[6:12] + np.r_[x0[9:12] * DT, np.zeros(3)])
    assert ekf.command_available and ekf.get_status()['command']


def test_with_gps_velocity_the_command_changes_nothing():
    """Nominal flight is the constant-velocity filter of before, bit for bit (goldens unchanged)."""
    a, b = make_ekf(), make_ekf()
    rng = np.random.default_rng(1)
    for k in range(200):
        meas = {'gps_position': rng.standard_normal(3), 'gps_velocity': rng.standard_normal(3) * 0.1,
                'imu_attitude': rng.standard_normal(3) * 0.01, 'imu_rates': rng.standard_normal(3) * 0.01}
        a.predict(DT)
        b.predict(DT, accel_cmd=rng.standard_normal(3))
        a.update(meas, {'gps': GPS, 'imu': IMU}, k * DT)
        b.update(meas, {'gps': GPS, 'imu': IMU}, k * DT)
        assert np.array_equal(a.x, b.x) and np.array_equal(a.P, b.P)


def test_set_command_is_used_by_the_next_predict_once():
    ekf = make_ekf()
    lose_gps(ekf)
    a = np.array([0.0, 0.0, -2.0])
    ekf.set_command(a)
    v0 = ekf.x[5]
    ekf.predict(DT)
    assert ekf.x[5] == pytest.approx(v0 - 2.0 * DT) and ekf.command_available
    v1 = ekf.x[5]
    ekf.predict(DT)                                # no new command: the velocity is kept
    assert ekf.x[5] == v1 and not ekf.command_available
    ekf.set_command(a)
    ekf.predict(DT, accel_cmd=np.zeros(3))         # an explicit accel_cmd wins
    assert ekf.x[5] == v1


def test_a_non_finite_command_is_ignored_and_a_wrong_shape_raises():
    ekf = make_ekf()
    lose_gps(ekf)
    v0 = ekf.x[3:6].copy()
    ekf.predict(DT, accel_cmd=[np.nan, 0.0, 1.0])
    assert np.array_equal(ekf.x[3:6], v0) and not ekf.command_available
    with pytest.raises(ValueError):
        ekf.predict(DT, accel_cmd=[1.0, 2.0])


def test_commanded_velocity_variance_grows_by_the_mismatch_noise_capped_by_the_envelope():
    """With a command the velocity variance is not raised to the speed envelope at the loss: it
    grows by COMMAND_MISMATCH_PSD per second, and the envelope caps it, keeping the correlation."""
    truth = Truth()
    for _ in range(250):
        truth.step(np.zeros(3))
    ekf = truth.ekf
    truth.step(np.zeros(3), gps=False)              # the loss: nominal Q once (the step order)
    p_loss = np.diag(ekf.P)[3:6].copy()
    assert np.all(p_loss < 0.2)
    for k in range(1, 126):                          # 2.5 s
        truth.step(np.zeros(3), gps=False)
    hold = np.array([ekf._bias_hold_sigma(i) for i in range(3)]) ** 2 * 2.5 ** 2   # fix round: ~1e-5
    expected = p_loss + ekf.COMMAND_MISMATCH_PSD * 2.5 + hold
    assert np.diag(ekf.P)[3:6] == pytest.approx(expected)
    for _ in range(1000):                            # 20 s more: the envelope caps it
        truth.step(np.zeros(3), gps=False)
    v = ekf.x[3:6]
    cap = [((abs(v[i]) + ekf.NO_GPS_SPEED_LIMIT[i]) / 3.0) ** 2 for i in range(3)]
    assert np.diag(ekf.P)[3:6] == pytest.approx(cap)
    P = ekf.P
    for i in range(3):                               # still a covariance
        assert P[i, 3 + i] ** 2 <= P[i, i] * P[3 + i, 3 + i]


def test_without_a_command_the_w4_08_envelope_is_kept():
    """No command: the velocity is kept, its variance raised to the envelope at once (W4-08 S-1)."""
    truth = Truth(velocity=(1.0, -0.5, 0.2))
    for _ in range(250):
        truth.step(None, accel=np.zeros(3))
    ekf = truth.ekf
    for _ in range(3):
        truth.step(None, gps=False, accel=np.zeros(3))
    v = ekf.x[3:6]
    expected = [((abs(v[i]) + ekf.NO_GPS_SPEED_LIMIT[i]) / 3.0) ** 2 for i in range(3)]
    assert np.diag(ekf.P)[3:6] == pytest.approx(expected)


def test_the_command_of_the_step_gps_is_lost_is_not_dropped():
    """That step's predict still kept the velocity (it was measured until then); the update that
    finds it unmeasured adds the command's v += a dt, p += a dt^2 / 2."""
    truth = Truth(velocity=(1.0, 0.0, 0.0))
    for _ in range(250):
        truth.step(np.zeros(3))
    ekf = truth.ekf
    a = np.array([-2.0, 0.5, 1.0])
    ekf.predict(DT, accel_cmd=a)
    x_pred = ekf.x.copy()
    ekf.update({'imu_attitude': np.zeros(3), 'imu_rates': np.zeros(3)}, {'gps': NO_GPS, 'imu': IMU}, 6.0)
    assert np.allclose(ekf.x[3:6], x_pred[3:6] + (a + ekf.command_bias) * DT, rtol=0, atol=1e-15)
    assert np.allclose(ekf.x[0:3], x_pred[0:3] + 0.5 * (a + ekf.command_bias) * DT ** 2, rtol=0, atol=1e-15)


# --- the bias ------------------------------------------------------------------------------------

def test_bias_is_learned_with_gps_and_held_without():
    """A constant acceleration beyond the command (e.g. wind drag) is learned within 5 s (five
    time constants) while GPS velocity is fused, held through an outage, and not changed by the
    correction at the return."""
    bias = np.array([0.3, -0.15, 0.2])
    truth = Truth(bias=bias)
    command = np.array([-0.3, 0.15, -0.2])           # the controller balances it: the truth hovers
    for _ in range(250):
        truth.step(command)
    assert truth.ekf.command_bias == pytest.approx(bias, abs=0.03)
    for _ in range(250):
        truth.step(command)
    learned = truth.ekf.command_bias.copy()
    assert learned == pytest.approx(bias, abs=0.01)
    for _ in range(400):                              # 8 s without GPS: held
        truth.step(command, gps=False)
        assert np.array_equal(truth.ekf.command_bias, learned)
    assert np.linalg.norm(truth.error()) < 0.2        # the estimate stayed with the hovering truth
    truth.step(command)                               # the return: no lesson from the correction
    assert np.array_equal(truth.ekf.command_bias, learned)
    assert truth.ekf.get_status()['command_bias'] == pytest.approx(learned)


def test_bias_is_not_learned_without_a_command():
    truth = Truth(bias=(0.3, 0.0, 0.0))
    for _ in range(200):
        truth.step(None, accel=np.array([0.3, 0.0, 0.0]))
    assert np.array_equal(truth.ekf.command_bias, np.zeros(3))


def test_outage_while_braking_follows_the_commanded_braking():
    """The supervisor answers a GPS loss with HOVER, and the controller brakes: the estimate brakes
    with it. Without the command (W4-08) the kept 3 m/s ran the estimate 28 m ahead in 10 s."""
    def run(with_command):
        truth = Truth(velocity=(3.0, 0.0, 0.0))
        for _ in range(250):
            truth.step(np.zeros(3) if with_command else None, accel=np.zeros(3))
        worst = 0.0
        for _ in range(500):                          # brake at 2 m/s^2 to hover, then hold
            brake = np.array([-min(2.0, truth.vel[0] / DT), 0.0, 0.0])
            truth.step(brake if with_command else None, gps=False, accel=brake)
            P = truth.ekf.get_covariance()
            worst = max(worst, max(abs(truth.error()[i]) / (3 * np.sqrt(P[i, i])) for i in range(3)))
        return np.linalg.norm(truth.error()), worst
    error, worst = run(True)
    assert error < 0.5 and worst < 1.0     # 0.37 m: the velocity and bias noise at the loss
    assert run(False)[0] > 25.0


# --- the core ------------------------------------------------------------------------------------

def test_the_core_passes_the_command_it_sent_to_the_next_predict():
    """control_step hands the final thrust, at the estimated attitude and the tier's hover thrust,
    to the estimator (set_command), and the next step's predict uses it."""
    from contract_uav_core.core import AdaptiveDroneController
    from contract_uav_core.frames import attitude_thrust_to_accel
    ctrl = AdaptiveDroneController(dt=DT, use_horizon_planner=False)
    target = np.array([0.0, 0.0, -5.0])
    conditions = {'gps_satellites': 12.0, 'gps_hdop': 0.8, 'imu_temperature': 25.0, 'imu_calibrated': 1.0,
                  'battery_voltage': 12.4, 'motor_temperature': 25.0, 'wind_speed': 1.0, 'disturbance': 0.2}
    assert ctrl.pre_flight_check(conditions, {'target_position': target, 'waypoints': [target]})[0]
    assert ctrl.start_mission()
    raw = {'gps': {'position': np.array([0.3, -0.2, -4.0]), 'velocity': np.zeros(3), 'valid': True,
                   'satellites': 12, 'hdop': 0.8},
           'imu': {'attitude': np.array([0.01, -0.02, 0.3]), 'rates': np.zeros(3), 'valid': True,
                   'calibrated': True, 'temperature': 25.0},
           'battery': {'voltage': 12.4}, 'motors': {'temperature': 25.0}}
    sent = []
    original = ctrl.ekf.set_command
    ctrl.ekf.set_command = lambda accel: (sent.append(accel), original(accel))
    control, telemetry = ctrl.control_step(raw)
    tier = ctrl.controller_switcher.controllers[telemetry['active_controller']]
    assert tier.hover_thrust == 0.5 and tier.max_thrust == 1.0
    expected = attitude_thrust_to_accel(telemetry['attitude'], control['thrust'], 0.5, 1.0)
    assert len(sent) == 1 and np.array_equal(sent[0], expected)
    assert not ctrl.ekf.command_available            # the first predict had none
    ctrl.control_step(raw)
    assert ctrl.ekf.command_available                # the second used the first step's command


def test_the_core_passes_the_command_after_the_safety_filter():
    """The command handed over is the one sent, after the CBF filter: below 0.5 m the filter
    replaces the tier's thrust with 0.95, and that is what the estimator gets."""
    from contract_uav_core.core import AdaptiveDroneController
    from contract_uav_core.frames import attitude_thrust_to_accel
    ctrl = AdaptiveDroneController(dt=DT, use_horizon_planner=False)
    target = np.array([0.0, 0.0, -5.0])
    conditions = {'gps_satellites': 12.0, 'gps_hdop': 0.8, 'imu_temperature': 25.0, 'imu_calibrated': 1.0,
                  'battery_voltage': 12.4, 'motor_temperature': 25.0, 'wind_speed': 1.0, 'disturbance': 0.2}
    assert ctrl.pre_flight_check(conditions, {'target_position': target, 'waypoints': [target]})[0]
    assert ctrl.start_mission()
    raw = {'gps': {'position': np.array([0.0, 0.0, -0.2]), 'velocity': np.zeros(3), 'valid': True,
                   'satellites': 12, 'hdop': 0.8},
           'imu': {'attitude': np.zeros(3), 'rates': np.zeros(3), 'valid': True,
                   'calibrated': True, 'temperature': 25.0},
           'battery': {'voltage': 12.4}, 'motors': {'temperature': 25.0}}
    sent, nominal = [], []
    original_set, original_compute = ctrl.ekf.set_command, ctrl.controller_switcher.compute_control
    ctrl.ekf.set_command = lambda accel: (sent.append(accel), original_set(accel))
    ctrl.controller_switcher.compute_control = lambda *a: nominal.append(original_compute(*a)) or nominal[-1]
    control, telemetry = ctrl.control_step(raw)
    assert telemetry['cbf_intervened'] and control['thrust'] == 0.95
    assert nominal[0]['thrust'] != control['thrust']
    assert np.array_equal(sent[0], attitude_thrust_to_accel(telemetry['attitude'], 0.95, 0.5))


# --- the bias's bounds (fix round, review R-08b S-2) ---------------------------------------------

def test_bias_is_clipped_to_a_plausible_size():
    """Horizontally to BIAS_LIMIT_XY (the largest horizontal acceleration a tier commands), as a
    norm; vertically to BIAS_LIMIT_Z (a +-50 % hover-thrust error). Unbounded, ground contact
    taught it +8.8 m/s^2."""
    truth = Truth(bias=(4.0, 3.0, 6.0))
    command = np.array([-4.0, -3.0, -6.0])          # the truth hovers: the bias is "learned" as 5, 6
    for _ in range(500):
        truth.step(command)
    b = truth.ekf.command_bias
    assert np.hypot(b[0], b[1]) == pytest.approx(truth.ekf.BIAS_LIMIT_XY)
    assert b[0] / b[1] == pytest.approx(4.0 / 3.0, rel=0.05)      # the direction is kept
    assert b[2] == pytest.approx(truth.ekf.BIAS_LIMIT_Z)
    assert truth.ekf.BIAS_LIMIT_Z == pytest.approx(0.5 * G)


def test_bias_is_not_learned_near_the_ground():
    """Within GROUND_CLEARANCE of z = 0 the ground's reaction would be learned as a bias."""
    truth = Truth(bias=(0.3, 0.0, -2.0))
    truth.pos = np.array([0.0, 0.0, -0.3])
    command = np.array([-0.3, 0.0, 2.0])             # the truth stays put (on the ground, say)
    for _ in range(250):
        truth.step(command)
    assert np.array_equal(truth.ekf.command_bias, np.zeros(3))


def test_bias_is_not_learned_with_less_than_half_the_hover_thrust():
    """A command below half the hover's specific force (a_z > g / 2: idle, landing, not flying
    yet) teaches nothing either."""
    truth = Truth(bias=(0.3, 0.0, -0.6 * G))
    command = np.array([-0.3, 0.0, 0.6 * G])         # the truth hovers (held up by the ground)
    for _ in range(250):
        truth.step(command)
    assert np.array_equal(truth.ekf.command_bias, np.zeros(3))
    command = np.array([-0.3, 0.0, 0.4 * G])         # above half the hover: learned again
    truth.bias = np.array([0.3, 0.0, -0.4 * G])
    for _ in range(250):
        truth.step(command)
    assert truth.ekf.command_bias[0] == pytest.approx(0.3, abs=0.03)


# --- the held bias as the speed changes (fix round, review R-08b B-1) ----------------------------

def drag_truth(wind=(1.0, 0.0), k=0.025, cruise=8.0):
    """A truth with quadratic drag k |v - w| (v - w) horizontally (as sim/quadrotor.py) and a hover-
    thrust error of 0.4 m/s^2 vertically, under a controller that balances it: 5 s hover, then
    a 2 m/s^2 acceleration to `cruise` m/s along x and 6 s of cruise, with GPS. Returns (truth,
    drag function)."""
    w = np.array([wind[0], wind[1], 0.0])

    def accel_extra(v):
        rel = v - w
        rel[2] = 0.0
        return -k * np.linalg.norm(rel) * rel + np.array([0.0, 0.0, 0.4])

    truth = Truth()
    for _ in range(250):
        command = -accel_extra(truth.vel)
        truth.step(command, accel=command + accel_extra(truth.vel))
    for _ in range(int(round(cruise / 2.0 / DT)) + 300):
        push = np.array([min(2.0, (cruise - truth.vel[0]) / DT), 0.0, 0.0])
        command = push - accel_extra(truth.vel)
        truth.step(command, accel=command + accel_extra(truth.vel))
    return truth, accel_extra


def test_held_bias_goes_back_to_the_low_speed_anchor_as_the_vehicle_stops():
    """GPS lost while cruising, then the HOVER brake: the cruise drag held as it was (3de3b82) ran
    the estimate tens of metres off; scheduled by the speed between the learned bias and the one
    learned in hover (the anchor: wind drag, hover-thrust error), the estimate follows."""
    truth, extra = drag_truth()
    ekf = truth.ekf
    assert ekf.command_bias[0] == pytest.approx(extra(np.array([8.0, 0.0, 0.0]))[0], abs=0.05)
    assert ekf.command_bias_anchor[0] == pytest.approx(extra(np.zeros(3))[0], abs=0.02)
    assert ekf.command_bias_anchor[2] == pytest.approx(0.4, abs=0.02)
    for _ in range(500):                             # 10 s: brake at 2 m/s^2, then hold
        brake = np.array([-min(2.0, truth.vel[0] / DT), 0.0, 0.0])
        command = brake - extra(truth.vel)           # the controller's model is right here
        truth.step(command, gps=False, accel=command + extra(truth.vel))
    assert abs(truth.vel[0]) < 0.05
    assert np.linalg.norm(truth.error()) < 1.0       # held as learned: ~40 m (see below)
    held = ekf._held_bias(np.zeros(3))
    assert held == pytest.approx(ekf.command_bias_anchor.tolist())    # stopped: the anchor
    assert ekf._held_bias(np.array([8.0, 0.0, 0.0]))[0] == pytest.approx(ekf.command_bias[0])
    assert ekf._held_bias(np.array([8.0, 0.0, 0.0]))[2] == pytest.approx(0.4, abs=0.02)


def test_without_the_schedule_the_cruise_drag_runs_the_estimate_off():
    """The mechanism of review R-08b B-1, open loop: the same flight with the anchor forced to the
    cruise bias (i.e. the bias held as learned) ends tens of metres off."""
    truth, extra = drag_truth()
    ekf = truth.ekf
    ekf.command_bias_anchor = ekf.command_bias.copy()
    for _ in range(500):
        brake = np.array([-min(2.0, truth.vel[0] / DT), 0.0, 0.0])
        command = brake - extra(truth.vel)
        truth.step(command, gps=False, accel=command + extra(truth.vel))
    assert np.linalg.norm(truth.error()) > 30.0


def test_held_bias_variance_grows_with_the_bias_and_the_envelope_widens_to_the_speed_flown():
    """The held bias may be wrong by up to its own size: the velocity variance also grows by
    (max(|bias|, |anchor|) / 3)^2 (2 tau dt + dt^2) per step, not capped by the envelope below that
    term; and the envelope's limit is at least the largest speed flown with GPS."""
    truth, extra = drag_truth()
    ekf = truth.ekf
    assert ekf._speed_limit(0) == pytest.approx(8.0, abs=0.05)       # flown at 8 m/s
    assert ekf._speed_limit(1) == ekf.NO_GPS_SPEED_LIMIT[1]
    command = -extra(truth.vel)
    truth.step(command, gps=False, accel=command + extra(truth.vel))  # the loss step
    p0 = np.diag(ekf.P)[3:6].copy()
    n = 150
    for _ in range(n):
        command = -extra(truth.vel)
        truth.step(command, gps=False, accel=command + extra(truth.vel))
    tau = DT * n                                     # the outage clock starts after the loss step
    for i in range(3):
        sigma_b = max(abs(ekf.command_bias[i]), abs(ekf.command_bias_anchor[i])) / ekf.BIAS_HOLD_SIGMAS
        grown = p0[i] + ekf.COMMAND_MISMATCH_PSD * tau + sigma_b ** 2 * tau ** 2
        cap = max((abs(ekf.x[3 + i]) + ekf._speed_limit(i)) / 3.0, sigma_b * tau)
        assert ekf.P[3 + i, 3 + i] == pytest.approx(min(grown, cap ** 2), rel=1e-6), i
    assert np.diag(ekf.P)[5] == pytest.approx(((abs(ekf.x[5]) + 2.0) / 3.0) ** 2, rel=1e-6)
    assert np.diag(ekf.P)[3] > p0[0] + ekf.COMMAND_MISMATCH_PSD * tau + 0.9 * (abs(ekf.command_bias[0]) / 3 * tau) ** 2


def test_a_plausible_hover_thrust_error_is_learned_in_full():
    """The vertical bound must not cut a real hover-thrust error: a plant hover of 0.65 against the
    tiers' 0.5 (+30 %) is a bias of +0.3 g, learned in full (with a 0.25 g bound the vehicle sank
    into the ground in a 20 s outage, closed loop)."""
    truth = Truth(bias=(0.0, 0.0, 0.3 * G))
    command = np.array([0.0, 0.0, -0.3 * G])
    for _ in range(500):
        truth.step(command)
    assert truth.ekf.command_bias[2] == pytest.approx(0.3 * G, abs=0.02)


def test_covariance_covers_a_held_bias_that_vanishes():
    """The held bias's own term is not capped by the speed envelope: a wind that drops at the loss
    (a learned 1 m/s^2 of drag that vanishes) drives the velocity error to 8 m/s in 8 s, beyond the
    3 m/s envelope, and the 3 sigma bound still covers it (review R-08b: the wind 6 -> 0 case was
    1.5 x 3 sigma in velocity)."""
    truth = Truth(bias=(1.0, 0.0, 0.0))
    command = np.array([-1.0, 0.0, 0.0])
    for _ in range(500):
        truth.step(command)
    assert truth.ekf._speed_limit(0) == truth.ekf.NO_GPS_SPEED_LIMIT[0]
    worst = 0.0
    for _ in range(400):                              # the wind drops with the GPS loss
        truth.step(command, gps=False, accel=command)
        P = truth.ekf.get_covariance()
        verr = truth.ekf.get_state()['velocity'] - truth.vel
        worst = max(worst, max(abs(verr[i]) / (3 * np.sqrt(P[3 + i, 3 + i])) for i in range(3)),
                    max(abs(truth.error()[i]) / (3 * np.sqrt(P[i, i])) for i in range(3)))
    assert np.linalg.norm(truth.ekf.get_state()['velocity'] - truth.vel) > 7.0
    assert worst <= 1.0
