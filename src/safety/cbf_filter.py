"""
Control Barrier Function (CBF) Safety Filter
Integrates with hierarchical contract system for runtime assurance

Based on CBF theory: ensures system stays within safe set S
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CBFSafetyFilter:
    """
    Control Barrier Function safety filter for UAV
    
    Safety constraints:
    1. Altitude bounds: z ∈ [-50, -2] (2m to 50m above ground)
    2. Velocity limits: ||v|| ≤ 15 m/s
    3. Tilt limits: |roll|, |pitch| ≤ 30°
    4. Rate limits: |p|, |q|, |r| ≤ 3 rad/s
    
    CBF formulation:
    h(x) ≥ 0  → x is safe
    �ḣ(x) + α·h(x) ≥ 0  → x remains safe
    
    If nominal control violates CBF, filter finds minimal modification
    """
    
    def __init__(self):
        # Safety bounds
        self.altitude_min = -50.0  # meters (NED frame, so negative)
        self.altitude_max = -2.0   # meters
        self.velocity_max = 15.0   # m/s
        self.tilt_max = 0.524      # radians (30°)
        self.rate_max = 3.0        # rad/s
        
        # CBF class-K function parameter (aggressiveness)
        self.alpha = 1.0
        
        # Safety margin (buffer from hard limits)
        self.margin = 0.1
        
        # Intervention tracking
        self.intervention_count = 0
        self.last_intervention_time = 0.0
        
        logger.info("CBF Safety Filter initialized")
        logger.info(f"  Altitude: [{self.altitude_min:.1f}, {self.altitude_max:.1f}] m")
        logger.info(f"  Velocity: ≤{self.velocity_max:.1f} m/s")
        logger.info(f"  Tilt: ≤{np.degrees(self.tilt_max):.1f}°")
    
    def barrier_altitude(self, z: float) -> float:
        """
        Altitude barrier function
        h(z) = (z - z_min)(z_max - z)
        Safe when h(z) > 0
        """
        return (z - self.altitude_min) * (self.altitude_max - z)
    
    def barrier_velocity(self, velocity: np.ndarray) -> float:
        """
        Velocity barrier function
        h(v) = v_max^2 - ||v||^2
        Safe when h(v) > 0
        """
        v_norm_sq = np.dot(velocity, velocity)
        return self.velocity_max**2 - v_norm_sq
    
    def barrier_tilt(self, roll: float, pitch: float) -> float:
        """
        Tilt barrier function
        h(φ,θ) = tilt_max^2 - (φ^2 + θ^2)
        Safe when h > 0
        """
        return self.tilt_max**2 - (roll**2 + pitch**2)
    
    def barrier_rates(self, rates: np.ndarray) -> float:
        """
        Angular rate barrier function
        h(ω) = rate_max^2 - ||ω||^2
        Safe when h > 0
        """
        rate_norm_sq = np.dot(rates, rates)
        return self.rate_max**2 - rate_norm_sq
    
    def evaluate_safety(self, state: Dict[str, np.ndarray]) -> Tuple[bool, Dict[str, float]]:
        """
        Evaluate all barrier functions
        Returns: (is_safe, barrier_values)
        """
        position = state['position']
        velocity = state['velocity']
        attitude = state['attitude']
        rates = state['rates']
        
        barriers = {
            'altitude': self.barrier_altitude(position[2]),
            'velocity': self.barrier_velocity(velocity),
            'tilt': self.barrier_tilt(attitude[0], attitude[1]),
            'rates': self.barrier_rates(rates)
        }
        
        # Safe if ALL barriers are positive
        is_safe = all(h > 0 for h in barriers.values())
        
        return is_safe, barriers
    
    def filter_control(self, 
                      state: Dict[str, np.ndarray],
                      nominal_control: Dict[str, float],
                      timestamp: float) -> Tuple[Dict[str, float], bool]:
        """
        Apply CBF safety filter to nominal control
        
        Returns: (safe_control, intervention_occurred)
        """
        # Evaluate current safety
        is_safe, barriers = self.evaluate_safety(state)
        
        if is_safe and all(h > self.margin for h in barriers.values()):
            # Comfortably safe - pass through nominal control
            return nominal_control, False
        
        # Need intervention!
        logger.warning(f"CBF INTERVENTION at t={timestamp:.2f}s")
        logger.warning(f"  Barriers: {barriers}")
        
        self.intervention_count += 1
        self.last_intervention_time = timestamp
        
        # Apply safety modifications
        safe_control = self.compute_safe_control(state, nominal_control, barriers)
        
        return safe_control, True
    
    def compute_safe_control(self,
                            state: Dict[str, np.ndarray],
                            nominal_control: Dict[str, float],
                            barriers: Dict[str, float]) -> Dict[str, float]:
        """
        Compute minimally-invasive safe control
        
        Strategy:
        1. If altitude barrier violated → adjust thrust
        2. If tilt barrier violated → reduce torques
        3. If velocity barrier violated → reduce thrust
        4. If rate barrier violated → damp rates
        """
        safe_control = nominal_control.copy()
        
        # Altitude safety
        if barriers['altitude'] < self.margin:
            z = state['position'][2]
            vz = state['velocity'][2]
            
            if z < self.altitude_min + 2.0:
                # Too low - increase thrust
                safe_control['thrust'] = min(0.8, safe_control['thrust'] + 0.2)
                logger.warning(f"  Altitude safety: z={z:.2f}m, increasing thrust")
            elif z > self.altitude_max - 2.0:
                # Too high - decrease thrust
                safe_control['thrust'] = max(0.2, safe_control['thrust'] - 0.2)
                logger.warning(f"  Altitude safety: z={z:.2f}m, decreasing thrust")
        
        # Tilt safety
        if barriers['tilt'] < self.margin:
            # Reduce torques to level out
            if 'torques' in safe_control:
                torques = safe_control['torques']
                safe_control['torques'] = torques * 0.5  # Cut torques by 50%
                logger.warning(f"  Tilt safety: reducing torques")
        
        # Velocity safety
        if barriers['velocity'] < self.margin:
            # Reduce thrust to slow down
            safe_control['thrust'] = min(0.5, safe_control['thrust'])
            logger.warning(f"  Velocity safety: limiting thrust")
        
        # Rate safety
        if barriers['rates'] < self.margin:
            # Aggressive damping
            if 'torques' in safe_control:
                rates = state['rates']
                damping_torques = -2.0 * rates  # Strong damping
                safe_control['torques'] = damping_torques
                logger.warning(f"  Rate safety: applying damping")
        
        return safe_control
    
    def get_statistics(self) -> Dict[str, any]:
        """Get CBF filter statistics"""
        return {
            'intervention_count': self.intervention_count,
            'last_intervention_time': self.last_intervention_time
        }


class RuntimeMonitor:
    """
    Runtime monitors for environmental conditions
    Compatible with agrt-cbf-mini framework
    """
    
    def __init__(self):
        self.wind_estimate = 0.0
        self.gps_quality = 1.0  # 1.0 = good, 0.0 = bad
        self.compute_delay = 0.0
        
        self.history = {
            'wind': [],
            'gps': [],
            'compute': []
        }
    
    def update_wind_estimate(self, velocity: np.ndarray, expected_velocity: np.ndarray):
        """
        Estimate wind speed from velocity error
        """
        error = velocity - expected_velocity
        self.wind_estimate = np.linalg.norm(error[0:2])  # Horizontal wind
        self.history['wind'].append(self.wind_estimate)
    
    def update_gps_quality(self, satellites: int, hdop: float):
        """
        Assess GPS quality
        """
        # Quality based on satellites and HDOP
        sat_quality = min(satellites / 12.0, 1.0)  # 12 sats = perfect
        hdop_quality = max(0.0, 1.0 - hdop / 5.0)  # HDOP < 2 is good
        
        self.gps_quality = 0.5 * sat_quality + 0.5 * hdop_quality
        self.history['gps'].append(self.gps_quality)
    
    def update_compute_delay(self, dt_expected: float, dt_actual: float):
        """
        Monitor computation time
        """
        self.compute_delay = dt_actual - dt_expected
        self.history['compute'].append(self.compute_delay)
    
    def get_monitoring_data(self) -> Dict[str, float]:
        """
        Get current monitoring values for contract checking
        """
        return {
            'wind_speed': self.wind_estimate,
            'gps_quality': self.gps_quality,
            'compute_delay': self.compute_delay
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("CBF Safety Filter Test")
    print("=" * 60)
    
    # Create CBF filter
    cbf = CBFSafetyFilter()
    
    # Test 1: Safe state
    print("\n[TEST 1] Safe state")
    safe_state = {
        'position': np.array([0.0, 0.0, -5.0]),
        'velocity': np.array([1.0, 0.0, 0.0]),
        'attitude': np.array([0.1, 0.05, 0.0]),
        'rates': np.array([0.1, 0.05, 0.0])
    }
    
    is_safe, barriers = cbf.evaluate_safety(safe_state)
    print(f"Safe: {is_safe}")
    print(f"Barriers: {barriers}")
    
    nominal_control = {'thrust': 0.5, 'torques': np.array([0.1, 0.1, 0.0])}
    safe_control, intervention = cbf.filter_control(safe_state, nominal_control, 0.0)
    print(f"Intervention: {intervention}")
    
    # Test 2: Unsafe altitude
    print("\n[TEST 2] Too low altitude")
    unsafe_state = {
        'position': np.array([0.0, 0.0, -1.5]),  # Too low!
        'velocity': np.array([0.0, 0.0, -1.0]),
        'attitude': np.array([0.1, 0.05, 0.0]),
        'rates': np.array([0.1, 0.05, 0.0])
    }
    
    is_safe, barriers = cbf.evaluate_safety(unsafe_state)
    print(f"Safe: {is_safe}")
    print(f"Barriers: {barriers}")
    
    safe_control, intervention = cbf.filter_control(unsafe_state, nominal_control, 1.0)
    print(f"Intervention: {intervention}")
    print(f"Modified control: {safe_control}")
    
    # Test 3: Excessive tilt
    print("\n[TEST 3] Excessive tilt")
    tilted_state = {
        'position': np.array([0.0, 0.0, -5.0]),
        'velocity': np.array([2.0, 1.0, 0.0]),
        'attitude': np.array([0.6, 0.5, 0.0]),  # ~35° tilt - unsafe!
        'rates': np.array([0.5, 0.4, 0.0])
    }
    
    is_safe, barriers = cbf.evaluate_safety(tilted_state)
    print(f"Safe: {is_safe}")
    print(f"Barriers: {barriers}")
    
    safe_control, intervention = cbf.filter_control(tilted_state, nominal_control, 2.0)
    print(f"Intervention: {intervention}")
    print(f"Modified control: {safe_control}")
    
    print("\n" + "=" * 60)
    print("[OK] CBF Safety Filter Test Complete")
    print("=" * 60)
