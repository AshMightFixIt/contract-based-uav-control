"""
Contract-Aware Extended Kalman Filter
State estimation with formal contract-based sensor fusion

Key Innovation:
- EKF checks sensor quality contracts before fusion
- Degrades gracefully when sensors violate contracts
- Provides formal guarantees on estimation accuracy
"""

import numpy as np
from typing import Dict, Tuple, Optional
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
    # Per-axis validity flags in sensors['gps'] (PX4 vehicle_local_position names; see
    # interfaces.py): measurement -> {flag: components it covers}. An absent flag counts as valid.
    GPS_AXIS_FLAGS = {
        'gps_position': {'xy_valid': (0, 1), 'z_valid': (2,)},
        'gps_velocity': {'v_xy_valid': (0, 1), 'v_z_valid': (2,)},
    }

    # Innovation gate (T2): a measurement whose normalised innovation squared
    # y' S^-1 y exceeds the chi-square 99.9 % quantile for its number of components is rejected.
    CHI2_GATE = {1: 10.828, 2: 13.816, 3: 16.266}
    # After this many rejects in a row, the measurement's rows are reset to it ...
    GATE_RESET_COUNT = 10
    # ... with a wide covariance: this factor times the measurement noise R.
    RESET_VARIANCE_FACTOR = 10.0

    def __init__(self, contract_monitor: HierarchicalContractMonitor):
        self.contract_monitor = contract_monitor
        
        # State dimension
        self.n = 12
        
        # Initialize state
        self.x = np.zeros(self.n)  # State estimate
        self.P = np.eye(self.n) * 1.0  # Covariance
        
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
        # The latest sensor-contract check (ContractMetrics), set by check_sensor_contracts
        self.sensor_metrics: Optional[ContractMetrics] = None

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
        self._rows = {key: np.array(rows) for key, rows in self.MEASUREMENT_ROWS.items()}

        logger.info("Contract-Aware EKF initialized")
    
    def predict(self, dt: Optional[float] = None):
        """
        Prediction step: propagate state forward using motion model
        """
        if dt is None:
            dt = self.dt
        
        # Simple motion model: constant velocity + angular rates
        F = np.eye(self.n)
        
        # Position updates from velocity
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        
        # Attitude updates from angular rates (simplified)
        F[6, 9] = dt   # roll += p * dt
        F[7, 10] = dt  # pitch += q * dt
        F[8, 11] = dt  # yaw += r * dt
        
        # Predict state
        self.x = F @ self.x
        self.x[self.YAW] = wrap_angle(self.x[self.YAW])

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
    
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
    
    def _measurement(self, key: str, measurements: Dict[str, np.ndarray],
                     gps: Dict[str, any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(state rows, values) of one measurement's usable components, or None.

        A component is usable if it is present and finite and no validity flag marks it
        invalid (GPS_AXIS_FLAGS). Before W4-08 a missing measurement was fused as zeros.
        """
        value = measurements.get(key)
        if value is None:
            return None
        z = np.asarray(value, dtype=float)
        if z.shape != (3,):
            raise ValueError(f"measurement {key!r} has shape {z.shape}, expected (3,)")
        flags = self.GPS_AXIS_FLAGS.get(key, {})
        if all(gps.get(flag, True) for flag in flags) and np.isfinite(z).all():
            return self._rows[key], z
        keep = np.isfinite(z)
        for flag, components in flags.items():
            if not gps.get(flag, True):
                keep[list(components)] = False
        if not keep.any():
            return None
        return self._rows[key][keep], z[keep]

    def _nis(self, S: np.ndarray, y: np.ndarray, sizes) -> np.ndarray:
        """Normalised innovation squared y_i' S_ii^-1 y_i of each measurement i, from its
        diagonal block S_ii of S (one batched solve when every measurement has 3 rows)."""
        if all(size == 3 for size in sizes):
            g = np.arange(len(sizes))
            blocks = S.reshape(len(sizes), 3, len(sizes), 3)[g, :, g, :]  # [k] = S[3k:3k+3, 3k:3k+3]
            ys = y.reshape(-1, 3)
            return np.einsum('ij,ij->i', ys, np.linalg.solve(blocks, ys[..., None])[..., 0])
        nis, start = [], 0
        for size in sizes:
            block, yi = S[start:start + size, start:start + size], y[start:start + size]
            nis.append(float(yi @ np.linalg.solve(block, yi)))
            start += size
        return np.array(nis)

    def _reset(self, key: str, rows: np.ndarray, z: np.ndarray, r: np.ndarray, nis: float):
        """Set a measurement's rows to it (T2): variance RESET_VARIANCE_FACTOR * R, no correlation."""
        self.x[rows] = z
        self.x[self.YAW] = wrap_angle(self.x[self.YAW])
        self.P[rows, :] = 0.0
        self.P[:, rows] = 0.0
        self.P[rows, rows] = r * self.RESET_VARIANCE_FACTOR
        logger.warning(f"EKF: {key} rejected {self.GATE_RESET_COUNT} times in a row "
                       f"(NIS {nis:.1f} > {self.CHI2_GATE[len(rows)]:.1f}); state reset to the measurement")

    def _fuse(self, measurements: Dict[str, np.ndarray], sensors: Dict[str, any],
              keys: Tuple[str, ...], R: np.ndarray):
        """One Kalman update with the usable, gated measurements among `keys` (T1-T3).

        z, H and R hold only the rows of the measurements that are present, finite and
        not flagged invalid (_measurement); nothing is fused for the others. Each of
        those is then gated on its normalised innovation (CHI2_GATE): a rejected one is
        not fused, and after GATE_RESET_COUNT rejects in a row its rows are reset to it
        (_reset), since then the measurement is right and the estimate is wrong. A
        measurement is gated only once its innovation has been inside the gate: before
        that the estimate is still converging from the initial guess.
        H selects rows, so H x = x[idx], H P H' = P[idx, idx] and P H' = P[:, idx]
        exactly: with all four measurements this is, operation for operation, the
        update the estimator always did, and it gives the same numbers.
        """
        gps = sensors.get('gps') or {}
        groups = []
        for key in self.MEASUREMENT_ROWS:
            found = self._measurement(key, measurements, gps) if key in keys else None
            if found is None:
                self.reject_count[key] = 0
            else:
                groups.append((key,) + found)
        if not groups:
            return

        idx = np.concatenate([rows for _, rows, _ in groups])
        z = np.concatenate([values for _, _, values in groups])
        full = len(idx) == self.n  # all 12 rows, in order: H is the identity
        r = np.diag(R) if full else np.diag(R)[idx]

        # Innovation, with the heading wrapped to (-pi, pi] (T3)
        y = z - (self.x if full else self.x[idx])
        for i in ((self.YAW,) if full else np.flatnonzero(idx == self.YAW)):
            y[i] = wrap_angle(y[i])

        # Innovation covariance
        S = (self.P if full else self.P[np.ix_(idx, idx)]) + np.diag(r)

        # Innovation gate, one verdict per measurement (T2)
        nis = self._nis(S, y, [len(rows) for _, rows, _ in groups])
        accepted, start = [], 0
        for (key, rows, values), q in zip(groups, nis):
            stop = start + len(rows)
            inside = q <= self.CHI2_GATE[len(rows)]
            if inside or not self.initialized[key]:
                self.reject_count[key] = 0
                self.initialized[key] = self.initialized[key] or bool(inside)
                accepted.extend(range(start, stop))
                self.last_fused[key] = tuple(rows.tolist())
            elif self.reject_count[key] + 1 < self.GATE_RESET_COUNT:
                self.reject_count[key] += 1
                self.last_rejected[key] = tuple(rows.tolist())
            else:
                self.reject_count[key] = 0
                self._reset(key, rows, values, r[start:stop], float(q))
                self.last_reset[key] = tuple(rows.tolist())
            start = stop
        if not accepted:
            return
        if len(accepted) < len(idx):
            keep = np.array(accepted)
            idx, y, S = idx[keep], y[keep], S[np.ix_(keep, keep)]
            full = False

        # Kalman gain
        K = (self.P if full else self.P[:, idx]) @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y
        self.x[self.YAW] = wrap_angle(self.x[self.YAW])

        # Update covariance
        if full:
            KH = K
        else:
            KH = np.zeros((self.n, self.n))
            KH[:, idx] = K
        self.P = (np.eye(self.n) - KH) @ self.P

    def update(self,
               measurements: Dict[str, np.ndarray],
               sensors: Dict[str, any],
               timestamp: float) -> Tuple[np.ndarray, bool]:
        """
        Main update function with contract-based sensor fusion

        The sensor contracts choose which sensors may be fused (unchanged):
        - both ok ("nominal"): GPS position and velocity, IMU attitude and rates, R_nominal;
        - IMU ok only ("degraded"): IMU attitude and rates, R_degraded;
        - otherwise ("emergency"): nothing, and the covariance grows by 2 Q more.
        Of those, only the components that are present, finite, not flagged invalid
        and inside the innovation gate are fused (_fuse).

        Returns: (state_estimate, estimation_contract_satisfied)
        """
        # Update sensor quality
        self.update_sensor_quality(sensors)

        # Check sensor contracts
        gps_ok, imu_ok = self.check_sensor_contracts(timestamp)

        # Contract-based fusion strategy
        self.last_fused, self.last_rejected, self.last_reset = {}, {}, {}
        if gps_ok and imu_ok:
            keys, R, estimation_quality = tuple(self.MEASUREMENT_ROWS), self.R_nominal, "nominal"
        elif imu_ok:
            keys, R, estimation_quality = ('imu_attitude', 'imu_rates'), self.R_degraded, "degraded"
        else:
            keys, R, estimation_quality = (), None, "emergency"
        if keys:
            self._fuse(measurements, sensors, keys, R)
        else:
            for key in self.reject_count:
                self.reject_count[key] = 0
            # Dead reckoning: prediction already done in predict step; increase uncertainty
            self.P += self.Q * 2.0
        if estimation_quality != self.estimation_quality:
            log = logger.warning if estimation_quality == "emergency" else logger.info
            log(f"EKF fusion: {self.estimation_quality} -> {estimation_quality}")
            self.estimation_quality = estimation_quality

        # Check estimator contract
        estimation_state = {
            'position_error': np.linalg.norm(self.P[0:3, 0:3]),  # Position uncertainty
            'velocity_error': np.linalg.norm(self.P[3:6, 3:6]),  # Velocity uncertainty
            'attitude_error': np.linalg.norm(self.P[6:9, 6:9]),  # Attitude uncertainty
        }

        metrics = self.contract_monitor.monitor_runtime(
            component="estimator",
            values=estimation_state,
            timestamp=timestamp
        )

        logger.debug(f"EKF update: {estimation_quality}, contract: {metrics.assumptions_met}")

        return self.x.copy(), metrics.guarantees_met

    def get_state(self) -> Dict[str, np.ndarray]:
        """Return state as dictionary (copies, so later updates do not change them)"""
        return {
            'position': self.x[0:3].copy(),
            'velocity': self.x[3:6].copy(),
            'attitude': self.x[6:9].copy(),
            'rates': self.x[9:12].copy()
        }
    
    def get_covariance(self) -> np.ndarray:
        """Return state covariance matrix"""
        return self.P.copy()


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
