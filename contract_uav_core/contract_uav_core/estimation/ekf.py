"""
Contract-Aware Extended Kalman Filter
State estimation with formal contract-based sensor fusion

Key Innovation:
- EKF checks sensor quality contracts before fusion
- Degrades gracefully when sensors violate contracts
- Provides formal guarantees on estimation accuracy

A constant-velocity, constant-rate model of [position, velocity, attitude,
rates] (NED, see interfaces.py), at the step predict() is given.

What W4-08 changed (K17, the core half of A-F7):
- Only measurements that are present, finite and not flagged invalid are fused
  (per-axis flags as in PX4's vehicle_local_position); before, a missing one was
  fused as zeros. Each is gated on its normalised innovation; 10 rejects in a row
  reset its rows to it (T1, T2).
- The heading is wrapped to (-pi, pi], in the innovation and the state (T3).
- The process noise is per second. Without GPS the velocity is kept, with a
  variance that covers any velocity within NO_GPS_SPEED_LIMIT, and the position
  is flagged untrusted once eph > POSITION_UNTRUSTED_EPH (T4).
- When GPS returns, the published position glides back at <= 2 m/s (T5).
- gps_ok / imu_ok come from the sensor contract (sensor_limits.py, T6).
- mode='passthrough' takes PX4's own estimate, with its eph / epv covariance (T7).
- The estimator contract is checked with its own quantities and inputs (T8).
- The filter works on six independent 2-state blocks (see PAIRS): the same
  numbers as the 12x12 form, up to rounding, at a fraction of the cost.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from ..contracts.monitor import HierarchicalContractMonitor
from ..contracts.spec import ContractMetrics
from .sensor_limits import sensor_limits

logger = logging.getLogger(__name__)


def wrap_angle(angle: float) -> float:
    """angle wrapped to (-pi, pi]; an angle already in (-pi, pi] is returned unchanged."""
    if -np.pi < angle <= np.pi:
        return angle
    wrapped = angle - 2.0 * np.pi * np.ceil((angle - np.pi) / (2.0 * np.pi))
    return wrapped + 2.0 * np.pi if wrapped <= -np.pi else wrapped


class ContractAwareEKF:
    """
    Extended Kalman Filter with contract-based sensor selection

    State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
    - Position: x, y, z (m)
    - Velocity: vx, vy, vz (m/s)
    - Attitude: roll, pitch, yaw (rad)
    - Angular rates: p, q, r (rad/s)

    The model, the process and measurement noise and the measurements never couple
    two of the six pairs (a quantity and its rate) in PAIRS, so the covariance is
    block diagonal in them, and it is kept as six 2x2 blocks [P_aa, P_ab, P_bb]:
    predict and update are closed forms per pair, with no 12x12 product or inverse.
    The P property assembles the 12x12 matrix (and takes one back, if it has that
    structure); assign a whole matrix to change it, not single elements.
    """

    # The measurements update() reads (keys of its `measurements` dict, each shape (3,))
    # and the state rows each one observes (H is the identity on those rows).
    MEASUREMENT_ROWS = {
        'gps_position': (0, 1, 2),
        'gps_velocity': (3, 4, 5),
        'imu_attitude': (6, 7, 8),
        'imu_rates': (9, 10, 11),
    }
    YAW = 8  # state row of the heading, kept in (-pi, pi]
    # The six independent pairs (quantity row, rate row), and each row's (pair, slot 0 or 1)
    PAIRS = ((0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11))
    _FIRST = np.array([a for a, _ in PAIRS])
    _SECOND = np.array([b for _, b in PAIRS])
    _SLOT = {row: (j, k) for j, pair in enumerate(PAIRS) for k, row in enumerate(pair)}
    # Per-axis validity flags in sensors['gps'] (PX4 vehicle_local_position names; see
    # interfaces.py): measurement -> {flag: components it covers}. An absent flag counts as valid.
    GPS_AXIS_FLAGS = {
        'gps_position': {'xy_valid': (0, 1), 'z_valid': (2,)},
        'gps_velocity': {'v_xy_valid': (0, 1), 'v_z_valid': (2,)},
    }

    # Innovation gate (T2): a measurement whose normalised innovation squared
    # y' S^-1 y exceeds the chi-square 99.9 % quantile for its number of components is rejected.
    # The components of one measurement lie in different pairs, so S is diagonal for it.
    CHI2_GATE = {1: 10.828, 2: 13.816, 3: 16.266}
    # After this many rejects in a row, the measurement's rows are reset to it ...
    GATE_RESET_COUNT = 10
    # ... with a wide covariance: this factor times the measurement noise R.
    RESET_VARIANCE_FACTOR = 10.0

    # Process noise (T4). self.Q holds the noise per Q_REFERENCE_DT, the 50 Hz step it was
    # tuned for; predict(dt) scales it by dt / Q_REFERENCE_DT, so the covariance grows per
    # second, not per call.
    Q_REFERENCE_DT = 0.02  # s
    # Without GPS velocity a velocity component keeps its last value, v_i. The vehicle may
    # then brake (the supervisor answers a GPS loss with HOVER), accelerate or turn, but its
    # speed stays within NO_GPS_SPEED_LIMIT per axis, so the error of the kept value is at
    # most |v_i| + limit_i. Its variance is therefore raised to ((|v_i| + limit_i) / 3)^2,
    # so that 3 sigma covers any velocity in that envelope, and is held there (no further
    # velocity process noise). The position error, the integral of the velocity error, then
    # stays inside 3 sigma for any manoeuvre within the envelope (review R-08, S-1).
    # The limits are the speeds the benchmark vehicle can fly (its simulation clips the
    # horizontal speed at 3 m/s and the vertical at 2 m/s); set them from the airframe.
    NO_GPS_SPEED_LIMIT = (3.0, 3.0, 2.0)  # m/s, x y z
    # When nothing is fused at all (both sensor contracts violated) the covariance grows by
    # this many more times the step's process noise (the legacy 2 Q).
    DEAD_RECKONING_EXTRA_Q = 2.0
    # The position is flagged untrusted once its horizontal uncertainty
    # eph = sqrt(P_xx + P_yy) (PX4's eph convention) exceeds this (T4).
    POSITION_UNTRUSTED_EPH = 5.0  # m

    # mode='passthrough' (T7, decision D7): the accuracy keys in sensors['gps'] (PX4
    # vehicle_local_position: 1-sigma, eph and evh horizontal as sqrt(var_N + var_E), epv and evv
    # vertical) that give the covariance of PX4's position and velocity.
    MODES = ('fusion', 'passthrough')
    PASSTHROUGH_ACCURACY = {'gps_position': ('eph', 'epv'), 'gps_velocity': ('evh', 'evv')}

    # Smooth recovery (T5): while the GPS recovers (from any GPS row coming back after an
    # outage, or a gate reset, until the GPS position is fused again and the published
    # position has caught up), the published position moves at no more than this speed beyond
    # the estimated velocity, whatever moves the estimate, instead of jumping.
    MAX_RECOVERY_SPEED = 2.0  # m/s
    GPS_ROWS = frozenset(range(6))
    POSITION_ROWS = frozenset(range(3))

    def __init__(self, contract_monitor: HierarchicalContractMonitor, *, mode: str = 'fusion'):
        """mode: 'fusion' (default) fuses GPS and IMU measurements; 'passthrough' (T7) takes
        PX4's own position and velocity estimate as the state (see update())."""
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, not {mode!r}")
        self.mode = mode
        self.contract_monitor = contract_monitor

        # State dimension
        self.n = 12

        # Initialize state
        self.x = np.zeros(self.n)  # State estimate
        self.P = np.eye(self.n) * 1.0  # Covariance (kept as the pairs' 2x2 blocks)

        # Process noise (motion model uncertainty)
        self.Q = np.diag([
            0.01, 0.01, 0.01,  # Position process noise
            0.1, 0.1, 0.1,     # Velocity process noise
            0.01, 0.01, 0.01,  # Attitude process noise
            0.1, 0.1, 0.1      # Angular rate process noise
        ])

        # Measurement noise (sensor quality)
        self.R_nominal = np.diag([
            1.0, 1.0, 1.0,     # GPS position noise
            0.1, 0.1, 0.1,     # GPS velocity noise
            0.01, 0.01, 0.01,  # IMU attitude noise
            0.05, 0.05, 0.05   # IMU rate noise
        ])

        self.R_degraded = self.R_nominal * 5.0  # Higher uncertainty when GPS bad

        self.dt = 0.02  # 50 Hz update rate
        self.last_update_time = 0.0

        # Sensor quality tracking
        self.sensor_quality = {
            'gps': {'satellites': 0, 'hdop': 100.0, 'valid': False},
            'imu': {'calibrated': False, 'temperature': 25.0, 'valid': False}
        }
        # The latest sensor-contract check (ContractMetrics), set by check_sensor_contracts,
        # and the latest estimator-contract check, set by update (T8)
        self.sensor_metrics: Optional[ContractMetrics] = None
        self.estimator_metrics: Optional[ContractMetrics] = None

        # What the last update() did: 'nominal' | 'degraded' | 'emergency', and per
        # measurement the state rows fused, rejected by the gate, or reset to it
        self.estimation_quality: Optional[str] = None
        self.last_fused: Dict[str, Tuple[int, ...]] = {}
        self.last_rejected: Dict[str, Tuple[int, ...]] = {}
        self.last_reset: Dict[str, Tuple[int, ...]] = {}
        # Rejects in a row per measurement (reset to 0 when it is accepted or absent)
        self.reject_count = {key: 0 for key in self.MEASUREMENT_ROWS}
        # Whether each measurement's innovation has been inside the gate once. Until then
        # the estimate is still converging from the initial guess (zeros, variance 1), so
        # the measurement is fused without gating, as before W4-08.
        self.initialized = {key: False for key in self.MEASUREMENT_ROWS}

        # T4: which velocity components the last update measured (the process noise of the
        # next predict depends on it), the step and process noise of the last predict, and
        # whether the published position is trusted
        self.velocity_observed = np.ones(3, dtype=bool)
        self.last_dt = self.dt
        self._last_q = self.Q.diagonal().tolist()
        self.position_trusted = True
        # T5: the published position is self.x[0:3] + glide_offset while it glides back
        # (None otherwise); whether the GPS is recovering; the GPS state rows (0-5) used in
        # the last update, and ever
        self.glide_offset: Optional[np.ndarray] = None
        self.recovering = False
        self._gps_rows_last = frozenset()
        self._gps_rows_ever = set()

        logger.info("Contract-Aware EKF initialized")

    # --- the covariance, as six 2x2 blocks ---------------------------------------------------

    @property
    def P(self) -> np.ndarray:
        """The 12x12 covariance, assembled from the pairs' blocks (a new array each time)."""
        blocks = np.array(self._cov)
        P = np.zeros((self.n, self.n))
        P[self._FIRST, self._FIRST] = blocks[:, 0]
        P[self._FIRST, self._SECOND] = blocks[:, 1]
        P[self._SECOND, self._FIRST] = blocks[:, 1]
        P[self._SECOND, self._SECOND] = blocks[:, 2]
        return P

    @P.setter
    def P(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != (self.n, self.n):
            raise ValueError(f"P must be {self.n}x{self.n}, not {value.shape}")
        blocks = [[float(value[a, a]), float(value[a, b]), float(value[b, b])] for a, b in self.PAIRS]
        previous, self._cov = getattr(self, '_cov', None), blocks
        if not np.array_equal(self.P, value):
            self._cov = previous
            raise ValueError("P must be symmetric and couple only the rows of a pair in PAIRS "
                             "(a quantity and its rate)")

    def predict(self, dt: Optional[float] = None):
        """
        Prediction step: propagate state forward using motion model

        Constant velocity and constant rates: each quantity += its rate * dt, and per pair
        P <- F P F' + Q with F = [[1, dt], [0, 1]].
        """
        if dt is None:
            dt = self.dt
        dt = float(dt)

        x = self.x.tolist()
        for a, b in self.PAIRS:
            x[a] += dt * x[b]
        self.x = np.array(x)
        self.x[self.YAW] = wrap_angle(self.x[self.YAW])

        # Predict covariance, with the process noise of this step (T4)
        self.last_dt = dt
        q = self._process_noise(dt)
        self._last_q = q
        for j, (a, b) in enumerate(self.PAIRS):
            paa, pab, pbb = self._cov[j]
            self._cov[j] = [paa + dt * (2.0 * pab + dt * pbb) + q[a], pab + dt * pbb, pbb + q[b]]

    def _process_noise(self, dt: float) -> List[float]:
        """The diagonal of process_noise(dt), as a list."""
        scale = dt / self.Q_REFERENCE_DT
        q = [v * scale for v in self.Q.diagonal().tolist()]
        for i in range(3):
            if not self.velocity_observed[i]:
                q[3 + i] = 0.0      # held at the speed envelope instead (_after_measurements)
        return q

    def process_noise(self, dt: float) -> np.ndarray:
        """The process noise of a step of dt seconds (T4).

        self.Q scaled by dt / Q_REFERENCE_DT (before W4-08: self.Q whatever dt was), with no
        noise on a velocity component the last update did not measure: its variance is held at
        the speed envelope (NO_GPS_SPEED_LIMIT). The first predict after a GPS loss cannot know
        of it yet and adds the nominal velocity noise once (0.1 m^2/s^2 at 50 Hz); the envelope
        that the same update then sets is at least (limit / 3)^2 = 1 m^2/s^2, so it dominates.
        """
        return np.diag(self._process_noise(dt))

    @property
    def last_Q(self) -> np.ndarray:
        """The process noise the last predict added (12x12, diagonal)."""
        return np.diag(self._last_q)

    def update_sensor_quality(self, sensors: Dict[str, any]):
        """
        Record the sensors' quality fields (the raw sensor dict keys are in interfaces.py).

        A missing 'gps' or 'imu' dict counts as an invalid sensor (W4-08): before,
        the last values were kept, so a GPS that disappeared from the dict stayed ok.
        """
        gps = sensors.get('gps') or {}
        self.sensor_quality['gps'] = {
            'satellites': gps.get('satellites', 0),
            'hdop': gps.get('hdop', 100.0),
            'valid': gps.get('valid', False)
        }

        imu = sensors.get('imu') or {}
        self.sensor_quality['imu'] = {
            'calibrated': imu.get('calibrated', False),
            'temperature': imu.get('temperature', 25.0),
            'valid': imu.get('valid', False)
        }

    def check_sensor_contracts(self, timestamp: float) -> Tuple[bool, bool]:
        """
        Check the sensors against the sensor contract the monitor holds.

        Returns (gps_ok, imu_ok), from the contract's limits (estimation/sensor_limits.py,
        W4-08; before, the literals 6 satellites and HDOP < 2.0, and "calibrated" alone):
        - gps_ok: the GPS is flagged valid, satellites >= the contract's minimum (6)
          and HDOP <= its maximum (5.0);
        - imu_ok: the IMU is flagged valid and calibrated, and its temperature is in
          the contract's range (5 to 55 degrees C).
        It also records the sensor contract's runtime check with the monitor, now with
        the contract's own variables (gps_satellites, gps_hdop, imu_temperature,
        imu_calibrated; before, an 'imu_temperature_stable' the contract does not
        have, so its assumptions never held).
        """
        limits = sensor_limits(self.contract_monitor)
        gps = self.sensor_quality['gps']
        imu = self.sensor_quality['imu']
        sensor_state = {
            'gps_satellites': float(gps['satellites']),
            'gps_hdop': float(gps['hdop']),
            'imu_temperature': float(imu['temperature']),
            'imu_calibrated': 1.0 if imu['calibrated'] else 0.0,
        }

        self.sensor_metrics = self.contract_monitor.monitor_runtime(
            component="sensors",
            values=sensor_state,
            timestamp=timestamp
        )

        # NaN fails every comparison, so a NaN field counts as not ok
        gps_ok = (bool(gps['valid'])
                  and sensor_state['gps_satellites'] >= limits.min_satellites
                  and sensor_state['gps_hdop'] <= limits.max_hdop)
        imu_ok = (bool(imu['valid'])
                  and limits.imu_calibrated[0] <= sensor_state['imu_calibrated'] <= limits.imu_calibrated[1]
                  and limits.min_imu_temperature <= sensor_state['imu_temperature']
                  <= limits.max_imu_temperature)

        return gps_ok, imu_ok

    def _components(self, key: str, measurements: Dict[str, np.ndarray],
                    gps: Dict[str, any]) -> Optional[List[Tuple[int, float]]]:
        """[(state row, value)] of one measurement's usable components, or None.

        A component is usable if it is present and finite and no validity flag marks it
        invalid (GPS_AXIS_FLAGS). Before W4-08 a missing measurement was fused as zeros.
        """
        value = measurements.get(key)
        if value is None:
            return None
        z = np.asarray(value, dtype=float)
        if z.shape != (3,):
            raise ValueError(f"measurement {key!r} has shape {z.shape}, expected (3,)")
        dropped = set()
        for flag, components in self.GPS_AXIS_FLAGS.get(key, {}).items():
            if not gps.get(flag, True):
                dropped.update(components)
        rows = self.MEASUREMENT_ROWS[key]
        found = [(rows[i], v) for i, v in enumerate(z.tolist()) if i not in dropped and math.isfinite(v)]
        return found or None

    def _variance(self, row: int) -> float:
        j, k = self._SLOT[row]
        return self._cov[j][2 * k]

    def _set_row(self, row: int, value: float, variance: float):
        """Set one state row to a value, with a variance and no correlation (reset, pass-through)."""
        j, k = self._SLOT[row]
        self.x[row] = value
        self._cov[j][2 * k] = variance
        self._cov[j][1] = 0.0

    def _reset(self, key: str, components, r: Dict[int, float], nis: float):
        """Set a measurement's rows to it (T2): variance RESET_VARIANCE_FACTOR * R, no correlation."""
        for row, value in components:
            self._set_row(row, value, r[row] * self.RESET_VARIANCE_FACTOR)
        self.x[self.YAW] = wrap_angle(self.x[self.YAW])
        logger.warning(f"EKF: {key} rejected {self.GATE_RESET_COUNT} times in a row "
                       f"(NIS {nis:.1f} > {self.CHI2_GATE[len(components)]:.1f}); "
                       f"state reset to the measurement")

    def _fuse(self, measurements: Dict[str, np.ndarray], sensors: Dict[str, any],
              keys: Tuple[str, ...], R: np.ndarray):
        """One Kalman update with the usable, gated measurements among `keys` (T1-T3).

        Only the components that are present, finite and not flagged invalid are used
        (_components); nothing is fused for the others. Each measurement is then gated on
        its normalised innovation (CHI2_GATE): a rejected one is not fused, and after
        GATE_RESET_COUNT rejects in a row its rows are reset to it (_reset), since then the
        measurement is right and the estimate is wrong. A measurement is gated only once its
        innovation has been inside the gate: before that the estimate is still converging
        from the initial guess.
        The accepted components are fused one at a time (scalar updates, which equal the
        joint update since R is diagonal), each in its pair's 2x2 block.
        """
        gps = sensors.get('gps') or {}
        r_all = R.diagonal().tolist()
        if 'gps_position' in keys:
            r_all[0:6] = self.gps_noise(gps, r_all[0:6])
        x = self.x.tolist()
        accepted = []
        for key in self.MEASUREMENT_ROWS:
            components = self._components(key, measurements, gps) if key in keys else None
            if components is None:
                self.reject_count[key] = 0
                continue
            rows = tuple(row for row, _ in components)
            # Normalised innovation squared, heading wrapped (T3); S is diagonal here
            nis = 0.0
            for row, value in components:
                y = value - x[row]
                if row == self.YAW:
                    y = wrap_angle(y)
                nis += y * y / (self._variance(row) + r_all[row])
            inside = nis <= self.CHI2_GATE[len(components)]
            if inside or not self.initialized[key]:
                self.reject_count[key] = 0
                self.initialized[key] = self.initialized[key] or inside
                accepted.extend((row, value, r_all[row]) for row, value in components)
                self.last_fused[key] = rows
            elif self.reject_count[key] + 1 < self.GATE_RESET_COUNT:
                self.reject_count[key] += 1
                self.last_rejected[key] = rows
            else:
                self.reject_count[key] = 0
                self._reset(key, components, {row: r_all[row] for row in rows}, nis)
                x = self.x.tolist()
                self.last_reset[key] = rows

        # Scalar Kalman updates, in measurement order (a pair's quantity before its rate)
        for row, value, r in accepted:
            j, k = self._SLOT[row]
            a, b = self.PAIRS[j]
            paa, pab, pbb = self._cov[j]
            if k == 0:                      # the quantity is measured: H = [1, 0]
                y = value - x[a]
                if a == self.YAW:
                    y = wrap_angle(y)
                s = paa + r
                ka, kb = paa / s, pab / s
                self._cov[j] = [paa - ka * paa, pab - ka * pab, pbb - kb * pab]
            else:                           # its rate is measured: H = [0, 1]
                y = value - x[b]
                s = pbb + r
                ka, kb = pab / s, pbb / s
                self._cov[j] = [paa - ka * pab, pab - kb * pab, pbb - kb * pbb]
            x[a] += ka * y
            x[b] += kb * y
        if accepted:
            self.x = np.array(x)
            self.x[self.YAW] = wrap_angle(self.x[self.YAW])

    def gps_noise(self, gps: Dict[str, any], nominal: List[float]) -> List[float]:
        """The GPS measurement noise of this update: variances of x, y, z, vx, vy, vz (S-3).

        Each is at least the nominal value, and grows with the reported accuracy, so that a
        fix the sensor contract accepts is gated on the noise it actually has:
        - position: eph^2 / 2 for x and y and epv^2 for z when the GPS reports eph / epv
          (PX4's convention, as in pass-through); otherwise (HDOP x UERE)^2 / 2 per axis,
          the sensor contract's predicted position_meas_error (HDOP x UERE is the horizontal
          RMS error, so half its square per axis; z gets the same, as there is no VDOP);
        - velocity: evh^2 / 2 and evv^2 when reported, otherwise the contract's
          velocity_meas_error squared.
        With HDOP 0.8 (UERE 1.5 m: 0.72 m^2) the nominal 1 m^2 is kept, so nominal flight is
        unchanged; at HDOP 5, 28 m^2. Before, R was 1 m^2 whatever the HDOP, and with 3 m of
        noise 58 % of the fixes were rejected.
        """
        bounds = self.sensor_metrics.predicted_bounds if self.sensor_metrics is not None else {}
        pme = bounds.get('position_meas_error_max', 0.0)
        vme = bounds.get('velocity_meas_error_max', 0.0)
        position_var = pme * pme / 2.0
        velocity_var = vme * vme
        reported = [self._accuracy_variance(gps, 'eph', 2.0, position_var),
                    self._accuracy_variance(gps, 'eph', 2.0, position_var),
                    self._accuracy_variance(gps, 'epv', 1.0, position_var),
                    self._accuracy_variance(gps, 'evh', 2.0, velocity_var),
                    self._accuracy_variance(gps, 'evh', 2.0, velocity_var),
                    self._accuracy_variance(gps, 'evv', 1.0, velocity_var)]
        return [max(n, r) for n, r in zip(nominal, reported)]

    def _pass_through(self, measurements: Dict[str, np.ndarray], sensors: Dict[str, any]):
        """mode='passthrough' (T7): PX4's position and velocity become the state.

        measurements['gps_position'] / ['gps_velocity'] are then PX4's own estimate (the node
        sends vehicle_local_position). A component is taken if sensors['gps']['valid'] is set
        and the component is usable as in fusion mode (present, finite, its per-axis flag not
        False): its state row is set to it, its correlations are cleared, and its variance
        comes from PX4's accuracy keys: eph^2 / 2 for x and y, epv^2 for z (evh, evv for the
        velocity). A missing, non-finite or negative accuracy falls back to R_nominal's
        variance. The sensor contract (satellites, HDOP) does not decide this: PX4's estimate
        may be valid without GPS, and its own flags and accuracies say how good it is. An
        invalid component is not used: its row is predicted on, as without GPS in fusion mode.
        """
        gps = sensors.get('gps') or {}
        if not gps.get('valid', False):
            return
        r_nominal = self.R_nominal.diagonal().tolist()
        for key, (horizontal, vertical) in self.PASSTHROUGH_ACCURACY.items():
            components = self._components(key, measurements, gps)
            if components is None:
                continue
            first = self.MEASUREMENT_ROWS[key][0]
            variance = [self._accuracy_variance(gps, horizontal, 2.0, r_nominal[first]),
                        self._accuracy_variance(gps, horizontal, 2.0, r_nominal[first + 1]),
                        self._accuracy_variance(gps, vertical, 1.0, r_nominal[first + 2])]
            for row, value in components:
                self._set_row(row, value, variance[row - first])
            self.last_fused[key] = tuple(row for row, _ in components)

    @staticmethod
    def _accuracy_variance(gps: Dict[str, any], name: str, axes: float, fallback: float) -> float:
        """Per-axis variance from a 1-sigma accuracy over `axes` axes (sigma^2 / axes), or fallback."""
        sigma = gps.get(name)
        try:
            sigma = float(sigma)
        except (TypeError, ValueError):
            return float(fallback)
        if not np.isfinite(sigma) or sigma < 0.0:
            return float(fallback)
        return sigma * sigma / axes

    def update(self,
               measurements: Dict[str, np.ndarray],
               sensors: Dict[str, any],
               timestamp: float) -> Tuple[np.ndarray, bool]:
        """
        Main update function with contract-based sensor fusion

        The sensor contracts choose which sensors may be fused (unchanged):
        - both ok ("nominal"): GPS position and velocity, IMU attitude and rates, R_nominal
          (the GPS part raised to the reported accuracy, gps_noise);
        - IMU ok only ("degraded"): IMU attitude and rates, R_degraded;
        - otherwise ("emergency"): nothing, and the covariance grows by 2 Q more.
        Of those, only the components that are present, finite, not flagged invalid
        and inside the innovation gate are fused (_fuse). In mode='passthrough' the GPS
        rows are not fused: PX4's position and velocity are taken as they are, when
        flagged valid, with their eph / epv covariance (_pass_through, T7). Then (_after_measurements):
        an unmeasured velocity keeps its value with the speed-envelope variance (T4), the
        position is flagged untrusted once eph > POSITION_UNTRUSTED_EPH (T4), and a
        GPS that comes back (any row, either mode) glides in at <= MAX_RECOVERY_SPEED (T5).

        Returns: (state_estimate, estimation_contract_satisfied); the state is the
        published one, as get_state()
        """
        # Update sensor quality
        self.update_sensor_quality(sensors)

        # Check sensor contracts
        gps_ok, imu_ok = self.check_sensor_contracts(timestamp)

        # Contract-based fusion strategy
        self.last_fused, self.last_rejected, self.last_reset = {}, {}, {}
        published_before = self._published_position()
        if gps_ok and imu_ok:
            keys, R, estimation_quality = tuple(self.MEASUREMENT_ROWS), self.R_nominal, "nominal"
        elif imu_ok:
            keys, R, estimation_quality = ('imu_attitude', 'imu_rates'), self.R_degraded, "degraded"
        else:
            keys, R, estimation_quality = (), None, "emergency"
        if self.mode == 'passthrough':
            keys = tuple(key for key in keys if key not in self.PASSTHROUGH_ACCURACY)
        if keys:
            self._fuse(measurements, sensors, keys, R)
        else:
            for key in self.reject_count:
                self.reject_count[key] = 0
            # Dead reckoning: prediction already done in predict step; increase uncertainty
            extra = self.DEAD_RECKONING_EXTRA_Q
            for j, (a, b) in enumerate(self.PAIRS):
                self._cov[j][0] += self._last_q[a] * extra
                self._cov[j][2] += self._last_q[b] * extra
        if self.mode == 'passthrough':
            self._pass_through(measurements, sensors)
        self._after_measurements(published_before)
        if estimation_quality != self.estimation_quality:
            log = logger.warning if estimation_quality == "emergency" else logger.info
            log(f"EKF fusion: {self.estimation_quality} -> {estimation_quality}")
            self.estimation_quality = estimation_quality

        # Check estimator contract (T8)
        metrics = self.contract_monitor.monitor_runtime(
            component="estimator",
            values=self.estimator_contract_values(),
            timestamp=timestamp
        )
        self.estimator_metrics = metrics

        logger.debug(f"EKF update: {estimation_quality}, contract: {metrics.assumptions_met}")

        return self._published_state(), metrics.guarantees_met

    # The estimator contract's inputs, from the sensor contract's predicted bounds (T8)
    MEAS_ERROR_INPUTS = ('position_meas_error', 'velocity_meas_error', 'rate_meas_error', 'attitude_meas_error')

    def estimator_contract_values(self) -> Dict[str, float]:
        """The values update() checks the estimator contract with (contracts/library.py, EKF_Estimator).

        - Outputs: position_error, velocity_error, attitude_error, the RMS error magnitude
          the published covariance implies, sqrt(trace(P_block)) in m, m/s and rad: the
          same position_error the switching policy feeds the controller contracts
          (switching_policy.py). Before W4-08: the Frobenius norm of each covariance block,
          i.e. a variance (m^2, ...), against bounds in m.
        - Inputs: position_meas_error, velocity_meas_error, rate_meas_error and
          attitude_meas_error, the sensor contract's predicted bounds for the latest sensor
          check (HDOP x UERE, ...). Before W4-08 none was passed, so the contract's
          assumptions never held and its relational bounds lost their input term.
        """
        cov = self._cov
        position_var = cov[0][0] + cov[1][0] + cov[2][0]
        if self.glide_offset is not None:       # get_covariance's P + d d' (T5)
            position_var += float(self.glide_offset @ self.glide_offset)
        values = {
            'position_error': math.sqrt(position_var),
            'velocity_error': math.sqrt(cov[0][2] + cov[1][2] + cov[2][2]),
            'attitude_error': math.sqrt(cov[3][0] + cov[4][0] + cov[5][0]),
        }
        bounds = self.sensor_metrics.predicted_bounds if self.sensor_metrics is not None else {}
        for name in self.MEAS_ERROR_INPUTS:
            if f'{name}_max' in bounds:
                values[name] = float(bounds[f'{name}_max'])
        return values

    def _after_measurements(self, published_before: np.ndarray):
        """Book-keeping after an update's measurements (T4, T5).

        - The velocity components measured in this update set the next step's process noise,
          and an unmeasured one gets the speed-envelope variance (NO_GPS_SPEED_LIMIT).
        - Glide (T5): the GPS starts recovering when any GPS state row (position or velocity,
          in either mode) is used in this update after an update without it, having been used
          before (not the very first fix), or when the GPS position is reset by the gate. Per
          row, so a horizontal return while z stayed valid (PX4's xy_valid) counts, and so
          does a velocity that comes back while the position is still gated. While recovering,
          the published position moves from where the prediction put it toward the estimate by
          at most MAX_RECOVERY_SPEED * dt per step, whatever moved the estimate. Recovery ends
          when the published position has caught up and the GPS position was fused (not
          rejected) in this update. The filter itself takes the measurements in full: only
          what is published glides, and get_covariance() adds the remaining offset to the
          position covariance, so the covariance stays honest.
        - The position is trusted while the published eph <= POSITION_UNTRUSTED_EPH.
        """
        used = set()
        for key in ('gps_position', 'gps_velocity'):
            used.update(self.last_fused.get(key, ()), self.last_reset.get(key, ()))
        self.velocity_observed = np.array([row in used for row in (3, 4, 5)])
        for i in range(3):          # T4, S-1: an unmeasured velocity's error is bounded by the envelope
            if not self.velocity_observed[i]:
                bound = (abs(float(self.x[3 + i])) + self.NO_GPS_SPEED_LIMIT[i]) / 3.0
                self._cov[i][2] = max(self._cov[i][2], bound * bound)

        returned = (used - self._gps_rows_last) & self._gps_rows_ever
        if returned or 'gps_position' in self.last_reset:
            if not self.recovering:
                logger.info(f"EKF: GPS back on state rows {sorted(returned) or 'reset'}; the published "
                            f"position glides at <= {self.MAX_RECOVERY_SPEED} m/s")
            self.recovering = True
        if self.recovering:
            offset = published_before - self.x[0:3]
            limit = self.MAX_RECOVERY_SPEED * self.last_dt
            distance = float(np.linalg.norm(offset))
            self.glide_offset = offset * (1.0 - limit / distance) if distance > limit else None
            if (self.glide_offset is None and used & self.POSITION_ROWS
                    and 'gps_position' not in self.last_rejected):
                self.recovering = False
        self._gps_rows_last = frozenset(used)
        self._gps_rows_ever |= used

        self.position_trusted = bool(self._eph() <= self.POSITION_UNTRUSTED_EPH)

    def _eph(self) -> float:
        """sqrt(P_xx + P_yy) of the published position (get_covariance), in m."""
        var = self._cov[0][0] + self._cov[1][0]
        if self.glide_offset is not None:
            var += self.glide_offset[0] ** 2 + self.glide_offset[1] ** 2
        return float(np.sqrt(var))

    def _published_position(self) -> np.ndarray:
        """The position get_state() publishes: the estimate, plus the glide offset while gliding."""
        if self.glide_offset is None:
            return self.x[0:3].copy()
        return self.x[0:3] + self.glide_offset

    def _published_state(self) -> np.ndarray:
        state = self.x.copy()
        state[0:3] = self._published_position()
        return state

    def get_state(self) -> Dict[str, np.ndarray]:
        """Return state as dictionary (copies, so later updates do not change them).

        'position' is the published position: while it glides back after a GPS outage
        (T5) it differs from the filter's own estimate by glide_offset.
        """
        return {
            'position': self._published_position(),
            'velocity': self.x[3:6].copy(),
            'attitude': self.x[6:9].copy(),
            'rates': self.x[9:12].copy()
        }

    def get_covariance(self) -> np.ndarray:
        """Return state covariance matrix, of the published state.

        While the position glides back (T5), the position block also holds the offset
        between the published position and the filter's estimate (P + d d').
        """
        P = self.P
        if self.glide_offset is not None:
            P[0:3, 0:3] += np.outer(self.glide_offset, self.glide_offset)
        return P

    def get_status(self) -> Dict[str, any]:
        """How the estimate stands after the last update (W4-08).

        quality: 'nominal' | 'degraded' | 'emergency' (None before the first update);
        position_trusted: False once eph > POSITION_UNTRUSTED_EPH (T4); eph, epv: the
        published horizontal and vertical position uncertainty, sqrt(P_xx + P_yy) and
        sqrt(P_zz), in m; gliding: the position is gliding back (T5) and glide_offset is
        the published position minus the estimate; fused / rejected / reset: per
        measurement, the state rows fused, rejected by the gate, or reset to it.
        """
        epv_var = self._cov[2][0]
        if self.glide_offset is not None:
            epv_var += self.glide_offset[2] ** 2
        return {
            'quality': self.estimation_quality,
            'position_trusted': self.position_trusted,
            'eph': self._eph(),
            'epv': float(np.sqrt(epv_var)),
            'gliding': self.glide_offset is not None,
            'glide_offset': np.zeros(3) if self.glide_offset is None else self.glide_offset.copy(),
            'fused': dict(self.last_fused),
            'rejected': dict(self.last_rejected),
            'reset': dict(self.last_reset),
        }


# Test the contract-aware EKF
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Contract-Aware EKF Test")
    print("=" * 60)

    # Create contract monitor
    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()

    # Create EKF
    ekf = ContractAwareEKF(monitor)

    # Simulate some sensor data
    print("\n[TEST 1] Nominal sensors (GPS + IMU good)")

    measurements = {
        'gps_position': np.array([1.0, 2.0, -5.0]),
        'gps_velocity': np.array([0.1, 0.0, 0.0]),
        'imu_attitude': np.array([0.0, 0.0, 0.0]),
        'imu_rates': np.array([0.0, 0.0, 0.0])
    }

    sensors = {
        'gps': {'satellites': 12, 'hdop': 1.0, 'valid': True},
        'imu': {'calibrated': True, 'temperature': 25.0, 'valid': True}
    }

    # Predict
    ekf.predict()

    # Update
    state, contract_ok = ekf.update(measurements, sensors, timestamp=1.0)

    print(f"State estimate: {ekf.get_state()['position']}")
    print(f"Contract satisfied: {contract_ok}")

    print("\n[TEST 2] GPS degraded (few satellites)")

    sensors['gps']['satellites'] = 4  # Below threshold

    ekf.predict()
    state, contract_ok = ekf.update(measurements, sensors, timestamp=2.0)

    print(f"State estimate: {ekf.get_state()['position']}")
    print(f"Contract satisfied: {contract_ok}")

    print("\n" + "=" * 60)
    print("[OK] Contract-Aware EKF Test Complete")
    print("=" * 60)
