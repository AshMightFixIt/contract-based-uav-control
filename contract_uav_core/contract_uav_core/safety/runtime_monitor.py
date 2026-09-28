"""
RuntimeMonitor: runtime monitors for environmental conditions (wind, GPS
quality, compute delay). Moved unchanged from cbf_filter.py.
"""

import numpy as np
from typing import Dict


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
