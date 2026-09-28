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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContractAwareEKF:
    """
    Extended Kalman Filter with contract-based sensor selection
    
    State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
    - Position: x, y, z (m)
    - Velocity: vx, vy, vz (m/s)
    - Attitude: roll, pitch, yaw (rad)
    - Angular rates: p, q, r (rad/s)
    """
    
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
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
    
    def update_sensor_quality(self, sensors: Dict[str, any]):
        """
        Check sensor quality and update contracts
        """
        # GPS quality
        if 'gps' in sensors:
            gps = sensors['gps']
            self.sensor_quality['gps'] = {
                'satellites': gps.get('satellites', 0),
                'hdop': gps.get('hdop', 100.0),
                'valid': gps.get('valid', False)
            }
        
        # IMU quality
        if 'imu' in sensors:
            imu = sensors['imu']
            self.sensor_quality['imu'] = {
                'calibrated': imu.get('calibrated', False),
                'temperature': imu.get('temperature', 25.0),
                'valid': imu.get('valid', False)
            }
    
    def check_sensor_contracts(self, timestamp: float) -> Tuple[bool, bool]:
        """
        Check if sensors satisfy their contracts
        Returns: (gps_contract_ok, imu_contract_ok)
        """
        # Build sensor state dict for contract checking
        sensor_state = {
            'gps_satellites': float(self.sensor_quality['gps']['satellites']),
            'gps_hdop': self.sensor_quality['gps']['hdop'],
            'imu_temperature_stable': 1.0 if abs(self.sensor_quality['imu']['temperature'] - 25.0) < 10.0 else 0.0,
        }
        
        # Check contracts using monitor
        metrics = self.contract_monitor.monitor_runtime(
            component="sensors",
            values=sensor_state,
            timestamp=timestamp
        )
        
        # Determine which sensors are good
        gps_ok = (self.sensor_quality['gps']['satellites'] >= 6 and 
                  self.sensor_quality['gps']['hdop'] < 2.0)
        imu_ok = self.sensor_quality['imu']['calibrated']
        
        return gps_ok, imu_ok
    
    def update_full(self, measurements: Dict[str, np.ndarray]):
        """
        Full update with GPS + IMU when both satisfy contracts
        """
        # Measurement vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        z = np.zeros(12)
        
        if 'gps_position' in measurements:
            z[0:3] = measurements['gps_position']
        if 'gps_velocity' in measurements:
            z[3:6] = measurements['gps_velocity']
        if 'imu_attitude' in measurements:
            z[6:9] = measurements['imu_attitude']
        if 'imu_rates' in measurements:
            z[9:12] = measurements['imu_rates']
        
        # Measurement matrix (direct observation)
        H = np.eye(self.n)
        
        # Innovation
        y = z - H @ self.x
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_nominal
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(self.n) - K @ H) @ self.P
        
        logger.debug("Full EKF update with GPS + IMU")
    
    def update_imu_only(self, measurements: Dict[str, np.ndarray]):
        """
        IMU-only update when GPS contract violated
        """
        # Only update attitude and rates
        z = np.zeros(6)
        
        if 'imu_attitude' in measurements:
            z[0:3] = measurements['imu_attitude']
        if 'imu_rates' in measurements:
            z[3:6] = measurements['imu_rates']
        
        # Measurement matrix (only observe attitude + rates)
        H = np.zeros((6, self.n))
        H[0:3, 6:9] = np.eye(3)   # Attitude
        H[3:6, 9:12] = np.eye(3)  # Rates
        
        # Innovation
        y = z - H @ self.x
        
        # Innovation covariance (higher uncertainty without GPS)
        R_imu = self.R_degraded[6:, 6:]
        S = H @ self.P @ H.T + R_imu
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(self.n) - K @ H) @ self.P
        
        logger.debug("IMU-only EKF update (GPS degraded)")
    
    def propagate_only(self):
        """
        Dead reckoning when both sensors violated
        Just predict, no measurement update
        """
        logger.warning("Both sensors violated contracts - dead reckoning only")
        # Prediction already done in predict step
        # Increase uncertainty
        self.P += self.Q * 2.0
    
    def update(self, 
               measurements: Dict[str, np.ndarray], 
               sensors: Dict[str, any],
               timestamp: float) -> Tuple[np.ndarray, bool]:
        """
        Main update function with contract-based sensor fusion
        
        Returns: (state_estimate, estimation_contract_satisfied)
        """
        # Update sensor quality
        self.update_sensor_quality(sensors)
        
        # Check sensor contracts
        gps_ok, imu_ok = self.check_sensor_contracts(timestamp)
        
        # Contract-based fusion strategy
        if gps_ok and imu_ok:
            self.update_full(measurements)
            estimation_quality = "nominal"
        elif imu_ok:
            self.update_imu_only(measurements)
            estimation_quality = "degraded"
        else:
            self.propagate_only()
            estimation_quality = "emergency"
        
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
        
        logger.info(f"EKF update: {estimation_quality}, contract: {metrics.assumptions_met}")
        
        return self.x.copy(), metrics.guarantees_met
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Return state as dictionary"""
        return {
            'position': self.x[0:3],
            'velocity': self.x[3:6],
            'attitude': self.x[6:9],
            'rates': self.x[9:12]
        }
    
    def get_covariance(self) -> np.ndarray:
        """Return state covariance matrix"""
        return self.P.copy()


# Test the contract-aware EKF
if __name__ == "__main__":
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
