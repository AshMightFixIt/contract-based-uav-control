"""The tiers' legacy control dict, computed through compute_accel_command, is unchanged.

The three functions legacy_*_compute_control below are the compute_control
methods of PIDController, MPCController and HInfinityController as they were at
commit 79d0287, before the acceleration-to-attitude mapping moved to frames.py
and the tiers gained compute_accel_command. They are copied verbatim (trailing
whitespace removed) and run on a controller instance as ``self``.

For each tier, one controller runs the frozen legacy method, one runs
compute_control and one runs compute_accel_command().to_legacy_dict(), on the
same sequence of states and setpoints: states recorded from a closed-loop run
of the core (benchmark_racing's simulation and wind schedule, the tier forced as
the benchmarks force it) plus hand-made edge cases. After every step the three
dicts must match exactly: same keys in the same order, same value types, and
arrays equal byte for byte (so -0.0 and 0.0 differ); the controllers' internal
state must match too.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")


# ---------------------------------------------------------------------------
# Frozen reference: compute_control at 79d0287 (verbatim)
# ---------------------------------------------------------------------------
def legacy_pid_compute_control(self,
                   state: Dict[str, np.ndarray],
                   setpoint: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Cascaded PID control

    Outer loop: Position → desired attitude
    Inner loop: Attitude → desired rates
    Innermost loop: Rates → motor commands
    """

    # Extract state
    position = state['position']  # [x, y, z]
    velocity = state['velocity']  # [vx, vy, vz]
    attitude = state['attitude']  # [roll, pitch, yaw]
    rates = state['rates']  # [p, q, r]

    # Extract setpoints
    target_pos = setpoint.get('position', position)
    target_vel = setpoint.get('velocity', np.zeros(3))
    target_yaw = setpoint.get('yaw', attitude[2])

    # === OUTER LOOP: Position Control ===
    pos_error = target_pos - position
    vel_error = target_vel - velocity

    # PID for desired acceleration
    self.integral_error += pos_error * self.dt
    self.integral_error = np.clip(self.integral_error,
                                  -self.integral_limit,
                                  self.integral_limit)

    d_error = (pos_error - self.last_error) / self.dt
    self.last_error = pos_error.copy()

    desired_accel = (self.kp_pos * pos_error +
                    self.ki_pos * self.integral_error +
                    self.kd_pos * d_error)

    # Convert desired acceleration to desired attitude
    # In NED frame with standard quadrotor convention:
    # - Positive pitch (nose up) -> backward (negative X)
    # - Positive roll (right wing down) -> rightward (positive Y)
    # So to accelerate forward (pos X), need negative pitch
    # And to accelerate right (pos Y), need positive roll
    desired_pitch = -desired_accel[0] / 9.81  # Negative: forward accel needs nose down
    desired_roll = desired_accel[1] / 9.81    # Positive: right accel needs right roll
    desired_yaw = target_yaw

    # Limit tilt angles
    desired_roll = np.clip(desired_roll, -self.max_tilt, self.max_tilt)
    desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)

    desired_attitude = np.array([desired_roll, desired_pitch, desired_yaw])

    # === INNER LOOP: Attitude Control ===
    att_error = desired_attitude - attitude

    # Normalize yaw error to [-pi, pi]
    att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))

    # PD control for desired rates
    desired_rates = self.kp_att * att_error - self.kd_att * rates
    desired_rates = np.clip(desired_rates, -self.max_rate, self.max_rate)

    # === INNERMOST LOOP: Rate Control ===
    rate_error = desired_rates - rates

    # P control for torques
    torques = self.kp_rate * rate_error

    # === THRUST CONTROL ===
    # Desired thrust = hover thrust + altitude correction
    # hover_thrust counteracts gravity, corrections adjust for tracking
    altitude_error = target_pos[2] - position[2]
    altitude_rate_error = -velocity[2]  # Want zero vertical velocity

    # PD control on altitude with gravity feedforward
    thrust_correction = (self.kp_pos[2] * altitude_error * 0.1 +
                        self.kd_pos[2] * altitude_rate_error * 0.05)
    desired_thrust = self.hover_thrust + thrust_correction
    desired_thrust = np.clip(desired_thrust, self.min_thrust, self.max_thrust)

    # Return control output
    control = {
        'thrust': desired_thrust,
        'torques': torques,  # [roll_torque, pitch_torque, yaw_torque]
        'desired_attitude': desired_attitude,
        'desired_rates': desired_rates
    }

    return control


def legacy_mpc_compute_control(self,
                   state: Dict[str, np.ndarray],
                   setpoint: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Solve condensed QP and return control.

    Steps:
    1. Form x0 (error state)
    2. Build X_ref (reference trajectory over horizon)
    3. Compute f = T'Q(S*x0 - X_ref)
    4. U* = -H_inv * f  (one matrix-vector multiply)
    5. Clip to constraints
    6. Extract first input, map to thrust + attitude + torques
    """
    import time
    t_start = time.perf_counter()

    position = state['position']
    velocity = state['velocity']
    attitude = state['attitude']
    rates = state['rates']

    target_pos = setpoint.get('position', position)
    target_vel = setpoint.get('velocity', np.zeros(3))

    # State vector: [pos_error, vel_error] relative to target
    x0 = np.concatenate([position - target_pos, velocity - target_vel])

    # Reference trajectory: drive error to zero
    N, nx, nu = self.N, self.n_x, self.n_u
    X_ref = np.zeros(N * nx)  # All zeros = go to target

    # Compute gradient: f = T'Q(S*x0 - X_ref)
    predicted_free = self._S @ x0 - X_ref
    f = self._TQ @ predicted_free

    # Optimal unconstrained solution: U* = -H_inv * f
    U_star = -self._H_inv @ f

    # Warm start: blend with shifted previous solution
    if self._prev_U is not None:
        U_warm = np.zeros_like(U_star)
        U_warm[:-nu] = self._prev_U[nu:]  # shift by one step
        U_warm[-nu:] = self._prev_U[-nu:]  # repeat last
        U_star = 0.7 * U_star + 0.3 * U_warm

    # Clip to acceleration constraints
    for i in range(N):
        idx = i * nu
        U_star[idx:idx+2] = np.clip(U_star[idx:idx+2],
                                     -self.max_lateral_accel,
                                     self.max_lateral_accel)
        U_star[idx+2] = np.clip(U_star[idx+2],
                                 -self.max_vertical_accel,
                                 self.max_vertical_accel)

    # Save for warm start
    self._prev_U = U_star.copy()

    # Extract first input (acceleration command)
    accel_cmd = U_star[0:3]

    # Map acceleration to thrust + attitude
    # Simulation convention: thrust > hover → z increases (descent in NED)
    # MPC accel_cmd[2] < 0 means "move z negative" (climb) → need less thrust
    desired_thrust = self.hover_thrust + accel_cmd[2] / 9.81
    desired_thrust = np.clip(desired_thrust, 0.0, 1.0)

    # Desired attitude from lateral acceleration
    desired_roll = accel_cmd[1] / 9.81
    desired_pitch = -accel_cmd[0] / 9.81
    desired_yaw = setpoint.get('yaw', attitude[2])

    # Clip tilt
    max_tilt = np.arctan2(self.max_lateral_accel, 9.81)  # ~17°
    desired_roll = np.clip(desired_roll, -max_tilt, max_tilt)
    desired_pitch = np.clip(desired_pitch, -max_tilt, max_tilt)

    desired_attitude = np.array([desired_roll, desired_pitch, desired_yaw])

    # Inner loop: PD attitude control (same as PID inner loop)
    att_error = desired_attitude - attitude
    att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))
    desired_rates_cmd = self.kp_att * att_error - self.kd_att * rates
    desired_rates_cmd = np.clip(desired_rates_cmd, -2.0, 2.0)

    # Rate → torque
    rate_error = desired_rates_cmd - rates
    torques = 0.08 * rate_error

    self.last_solve_time = time.perf_counter() - t_start

    return {
        'thrust': desired_thrust,
        'torques': torques,
        'desired_attitude': desired_attitude,
        'desired_rates': desired_rates_cmd,
    }


def legacy_hinf_compute_control(self,
                   state: Dict[str, np.ndarray],
                   setpoint: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Robust stabilization control

    Priority:
    1. Damp angular rates (prevent tumbling)
    2. Level attitude (prevent crashing)
    3. Hold altitude from setpoint
    """

    position = state['position']
    velocity = state['velocity']
    attitude = state['attitude']
    rates = state['rates']

    target_pos = setpoint.get('position', position)

    # === POSITION TRACKING (with integral wind rejection) ===
    pos_error = target_pos - position
    vel_error = -velocity  # want zero velocity (fight drift)

    # Integrate position error for sustained wind rejection
    self.pos_integral += pos_error * self.dt
    self.pos_integral = np.clip(self.pos_integral, -self.int_limit, self.int_limit)

    desired_accel = (self.k_pos * pos_error +
                     self.k_vel * vel_error +
                     self.k_int * self.pos_integral)

    # Map lateral acceleration to desired tilt (conservative limits)
    desired_pitch = np.clip(-desired_accel[0] / 9.81, -self.max_tilt, self.max_tilt)
    desired_roll = np.clip(desired_accel[1] / 9.81, -self.max_tilt, self.max_tilt)
    target_attitude = np.array([desired_roll, desired_pitch, attitude[2]])

    # === RATE DAMPING (Highest Priority) ===
    rate_damping_torque = -self.k_rate_damping * rates

    # === ATTITUDE STABILIZATION ===
    att_error = target_attitude - attitude
    att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))

    attitude_torque = self.k_attitude * att_error

    torques = rate_damping_torque + attitude_torque

    # === ALTITUDE HOLD (from setpoint) ===
    # Thrust = hover_thrust + corrections for altitude tracking
    altitude_error = target_pos[2] - position[2]
    altitude_rate_error = 0.0 - velocity[2]

    thrust_correction = (self.k_altitude * altitude_error +
                        self.k_altitude_rate * altitude_rate_error)
    desired_thrust = self.hover_thrust + thrust_correction
    desired_thrust = np.clip(desired_thrust, 0.1, 0.9)

    control = {
        'thrust': desired_thrust,
        'torques': torques,
        'desired_attitude': target_attitude,
        'desired_rates': np.zeros(3)
    }

    return control


# ---------------------------------------------------------------------------
# Inputs: recorded from a closed-loop run (see the module docstring), then edge cases
# ---------------------------------------------------------------------------
# Recorded with the core at 79d0287: AdaptiveDroneController(dt=0.02,
# use_horizon_planner=False) on benchmark_racing.RacingDroneSimulation with its
# wind schedule, np.random.seed(42), the tier forced; the state and setpoint the
# switcher passed to the controller at steps 1, 150, 700, 1400, 2600 and 3900.
RECORDED = {'Hinf': [({'attitude': [-7.356577119591143e-05, -0.0006520682145910997, 0.000800270459437968],
            'position': [0.002624365267549603, -0.007027889014093005, -3.3594226509066267],
            'rates': [-5.733980688311046e-05, -7.862069686929675e-05, -0.001161677271920455],
            'velocity': [0.00131159522276242, -0.007185645100800259, -0.01433905780952355]},
           {'position': [0.0, 0.0, -5.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
          ({'attitude': [0.11635503206175654, -0.219589998680003, -0.000982535908527847],
            'position': [3.535380685675588, 2.4264115691410497, -6.194111658782511],
            'rates': [0.0004225490031365041, -0.00017069194797670402, 3.6276119373723054e-05],
            'velocity': [1.642836954155501, 0.9572705702969567, -0.36687290816676715]},
           {'position': [15.0, 5.0, -8.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
          ({'attitude': [-0.1410935247097351, -0.22071109951437715, 0.0004218737167060001],
            'position': [18.991841843733013, 3.508779910921018, -8.630630052489382],
            'rates': [-7.407457571280643e-05, 8.305958070595773e-05, 0.0007348497505087895],
            'velocity': [1.7412867323749377, -1.07743880098533, -0.2867282670647065]},
           {'position': [30.0, 0.0, -10.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
          ({'attitude': [-0.07871673875288339, 0.02611788139564013, 0.001199716680783712],
            'position': [31.499140030425178, -8.938638178243172, -7.664303748906887],
            'rates': [-0.00113995976053689, 0.0002875434785229125, 0.000735421704383008],
            'velocity': [-0.007080699487881226, -0.6259049849323814, 0.1293515063122742]},
           {'position': [30.0, -10.0, -7.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
          ({'attitude': [-0.04140281845868844, 0.06277955257235648, -0.0005728312926542217],
            'position': [9.854562552326108, -8.430683667457675, -3.1804762924008294],
            'rates': [0.00016586109668483375, -0.0014315981609855479, 0.00046744564086709706],
            'velocity': [-0.2157880300198264, -0.06490520537478173, 0.041043825830851116]},
           {'position': [10.0, -10.0, -3.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
          ({'attitude': [-0.12249939690266799, 0.11423719070101603, 0.0019814862852933833],
            'position': [14.284442685921448, 10.364467298687646, -8.924343310827199],
            'rates': [-0.0010663216139208, -0.00043349930572093186, -0.0005004669072204045],
            'velocity': [0.372786619709181, 0.02728607547039208, 0.306883574596375]},
           {'position': [0.0, 0.0, -5.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0})],
 'MPC': [({'attitude': [-7.099043340740648e-05, -0.0006352552000685879, 0.0008157472680760257],
           'position': [0.0026233395845394166, -0.007027731904696644, -3.359484871022118],
           'rates': [-5.730826803720826e-05, -7.841479647186352e-05, -0.0011614877353536895],
           'velocity': [0.0012595500380974296, -0.007177673059277604, -0.01749622953651163]},
          {'position': [0.0, 0.0, -5.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
         ({'attitude': [0.06576835873888794, -0.2319767309249235, -0.0010522398200514625],
           'position': [4.590064926009083, 1.8585278514554957, -6.426365735161277],
           'rates': [0.0004280312578710114, -0.0001464382298435828, 3.632508981868003e-05],
           'velocity': [1.900821730060995, 0.5826300855830145, -0.3840426243169351]},
          {'position': [15.0, 5.0, -8.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
         ({'attitude': [-0.1175626504696274, -0.2769752687739307, -0.00013779212736447918],
           'position': [14.703104413955794, 4.18452259760483, -8.193273845443168],
           'rates': [-7.210653715919853e-05, 6.277896581526369e-05, 0.0007344590272427965],
           'velocity': [1.6243429445183342, -0.6257736840811368, -0.42175053818764924]},
          {'position': [30.0, 0.0, -10.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
         ({'attitude': [-0.20559011397941337, -0.007099905963392913, 0.00014392348294713737],
           'position': [29.152121199123357, -1.5603633586168273, -9.02961509146851],
           'rates': [-0.0011204804975231686, 0.0002857993562999768, 0.0007363572904681187],
           'velocity': [0.22865237676856767, -1.4565550625642876, 0.47773831534845146]},
          {'position': [30.0, -10.0, -7.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
         ({'attitude': [-0.03542191169843286, 0.05639248755647938, -0.0007707899000779117],
           'position': [21.63437599249571, -14.023146005705854, -5.087015019696298],
           'rates': [0.00016913295750770507, -0.001430607290587696, 0.00046717479193008574],
           'velocity': [-0.15330824223560813, -0.07747886090583879, 0.02582487147733083]},
          {'position': [20.0, -15.0, -5.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
         ({'attitude': [-0.038206461251316846, 0.0687032618647807, 0.0009357553561926413],
           'position': [11.508275756012308, -9.277496710779438, -3.0133225747320016],
           'rates': [-0.0009705101725437548, -0.0005372734935846215, -0.0005001583080630628],
           'velocity': [-0.010041994189619256, 0.03316943100642221, -0.00160929075961414]},
          {'position': [10.0, -10.0, -3.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0})],
 'PID': [({'attitude': [-1.808634934806529e-05, -0.0004628083089472865, 0.0008157472680760257],
           'position': [0.002612819504388078, -0.0070245044743055575, -3.3595759183363616],
           'rates': [-5.666037875363711e-05, -7.630292751639077e-05, -0.0011614877353536895],
           'velocity': [0.0007257403632473861, -0.007013906857605892, -0.022116150346330236]},
          {'position': [0.0, 0.0, -5.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
         ({'attitude': [0.08051745732266004, -0.21958999103270602, -0.0010522398200514625],
           'position': [3.53525906970309, 1.9989463578459645, -7.427694949940225],
           'rates': [0.0004241305948250885, -0.0001706919555367394, 3.632508981868003e-05],
           'velocity': [1.64283221883505, 0.6951727407903527, -0.584275665717406]},
          {'position': [15.0, 5.0, -8.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
         ({'attitude': [-0.09308414309447333, -0.22035563982850195, -0.00013779212736447918],
           'position': [16.666356120739255, 3.359894889093754, -9.332077076536397],
           'rates': [-7.426255313149204e-05, 8.270817709509599e-05, 0.0007344590272427965],
           'velocity': [1.6652919806721669, -0.6902044945964652, -0.5428974347285364]},
          {'position': [30.0, 0.0, -10.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
         ({'attitude': [-0.13219192486064602, 0.0019906117911925513, 0.00014392348294713737],
           'position': [29.882957732016063, -5.030699567832366, -7.034421757502074],
           'rates': [-0.0011337080810754911, 0.00028493000912738556, 0.0007363572904681187],
           'velocity': [0.15327970514786438, -1.0949512692168772, 0.15465054044749166]},
          {'position': [30.0, -10.0, -7.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
         ({'attitude': [0.05792760730087201, 0.21916782067622162, -0.0007707899000779117],
           'position': [18.543820582393167, -12.394268130396414, -3.459745188131839],
           'rates': [0.00016023855874051137, -0.0014294561691152157, 0.00046717479193008574],
           'velocity': [-1.3221697695148709, 0.713864629508714, 0.4322984179668001]},
          {'position': [10.0, -10.0, -3.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
         ({'attitude': [-0.03929566165691921, 0.0652700051551931, 0.0009357553561926413],
           'position': [11.773052509060014, -8.963807180234024, -3.00323800379105],
           'rates': [-0.000969785102294129, -0.000535382449027668, -0.0005001583080630628],
           'velocity': [0.02403683778260391, 0.02321571846844643, -0.004941126676262078]},
          {'position': [10.0, -10.0, -3.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0})]}


EDGE_CASES = [
    # far target: the tilt limits and the thrust clip saturate
    ({'position': [0.0, 0.0, -2.0], 'velocity': [0.0, 0.0, 0.0],
      'attitude': [0.0, 0.0, 0.0], 'rates': [0.0, 0.0, 0.0]},
     {'position': [60.0, -45.0, -30.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
    # yaw error across +-pi (wrap), fast rates (rate clip)
    ({'position': [1.0, 1.0, -5.0], 'velocity': [2.0, -1.0, 0.5],
      'attitude': [0.3, -0.2, 3.1], 'rates': [4.0, -3.0, 2.5]},
     {'position': [0.0, 0.0, -5.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': -3.1}),
    # setpoint without yaw or velocity: the defaults (current yaw, zero velocity)
    ({'position': [-3.0, 2.0, -7.0], 'velocity': [0.1, 0.2, -0.3],
      'attitude': [-0.05, 0.04, -1.2], 'rates': [0.01, -0.02, 0.03]},
     {'position': [0.0, 0.0, -5.0]}),
    # at the target and at rest
    ({'position': [0.0, 0.0, -5.0], 'velocity': [0.0, 0.0, 0.0],
      'attitude': [0.0, 0.0, 0.0], 'rates': [0.0, 0.0, 0.0]},
     {'position': [0.0, 0.0, -5.0], 'velocity': [0.0, 0.0, 0.0], 'yaw': 0.0}),
    # empty setpoint: hold the current position
    ({'position': [4.0, -4.0, -6.0], 'velocity': [-1.0, 1.0, 0.0],
      'attitude': [0.1, 0.1, 0.7], 'rates': [0.0, 0.0, 0.0]},
     {}),
]


def as_arrays(state_sp):
    state, sp = state_sp
    return ({k: np.array(v, dtype=float) for k, v in state.items()},
            {k: (np.array(v, dtype=float) if isinstance(v, list) else v) for k, v in sp.items()})


def sequence(tier):
    return [as_arrays(s) for s in RECORDED[tier] + EDGE_CASES]


def assert_identical(old, new, where):
    assert list(new.keys()) == list(old.keys()), where
    for key in old:
        a, b = old[key], new[key]
        assert type(b) is type(a), (where, key, type(a), type(b))
        if isinstance(a, np.ndarray):
            assert b.dtype == a.dtype and b.shape == a.shape, (where, key)
            assert b.tobytes() == a.tobytes(), (where, key, a, b)
        else:
            assert b == a and repr(b) == repr(a), (where, key, a, b)


STATE_FIELDS = {
    'PID': ('integral_error', 'last_error'),
    'MPC': ('_prev_U', 'last_error', 'integral_error'),
    'Hinf': ('pos_integral',),
}


def controller(tier):
    from contract_uav_core.control.hinf import HInfinityController
    from contract_uav_core.control.mpc import MPCController
    from contract_uav_core.control.pid import PIDController

    return {'PID': PIDController, 'MPC': MPCController, 'Hinf': HInfinityController}[tier](0.02)


LEGACY = {
    'PID': legacy_pid_compute_control,
    'MPC': legacy_mpc_compute_control,
    'Hinf': legacy_hinf_compute_control,
}


@pytest.mark.parametrize("tier", ['PID', 'MPC', 'Hinf'])
def test_legacy_dict_is_unchanged(tier):
    old, new, via_cmd = controller(tier), controller(tier), controller(tier)
    for step, (state, sp) in enumerate(sequence(tier)):
        where = f"{tier} step {step}"
        expected = LEGACY[tier](old, state, sp)
        assert_identical(expected, new.compute_control(state, sp), where)
        assert_identical(expected, via_cmd.compute_accel_command(state, sp).to_legacy_dict(), where)
        for field in STATE_FIELDS[tier]:
            ref = getattr(old, field)
            for other in (new, via_cmd):
                got = getattr(other, field)
                assert (got is None) == (ref is None), (where, field)
                if ref is not None:
                    assert got.tobytes() == ref.tobytes(), (where, field)


@pytest.mark.parametrize("tier", ['PID', 'MPC', 'Hinf'])
def test_accel_command_fields_feed_the_mapping(tier):
    """accel and yaw are the inputs of frames.accel_to_attitude (and of the MPC thrust)."""
    from contract_uav_core import frames
    from contract_uav_core.interfaces import AccelCommand

    ctrl = controller(tier)
    max_tilt = np.arctan2(ctrl.max_lateral_accel, 9.81) if tier == 'MPC' else ctrl.max_tilt
    for state, sp in sequence(tier):
        cmd = ctrl.compute_accel_command(state, sp)
        assert isinstance(cmd, AccelCommand)
        assert cmd.accel.shape == (3,)
        expected_yaw = state['attitude'][2] if tier == 'Hinf' else sp.get('yaw', state['attitude'][2])
        assert cmd.yaw == expected_yaw
        again = frames.accel_to_attitude(cmd.accel, cmd.yaw, max_tilt)
        assert again.tobytes() == cmd.desired_attitude.tobytes()
        if tier == 'MPC':
            assert frames.mpc_accel_to_thrust(cmd.accel[2], ctrl.hover_thrust) == cmd.thrust
        if tier == 'Hinf':
            assert cmd.accel[2] == 0.0


def test_switcher_accel_command_uses_the_active_controller():
    from contract_uav_core.control.switcher import ControllerSwitcher

    old, new = ControllerSwitcher(0.02), ControllerSwitcher(0.02)
    for name, t in [('PID', 0.0), ('MPC', 5.0), ('Hinf', 10.0)]:
        old.switch_to(name, t)
        new.switch_to(name, t)
        state, sp = as_arrays(RECORDED[name][1])
        assert_identical(old.compute_control(state, sp),
                         new.compute_accel_command(state, sp).to_legacy_dict(), name)
