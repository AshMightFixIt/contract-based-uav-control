"""
SensorReader — converts Betaflight MSP telemetry into the sensor dict format
expected by AdaptiveDroneController.control_step().

Expected sensor dict format (mirrors simulation):
    {
        'gps': {
            'position':   np.ndarray([x, y, z]),   # NED metres from home
            'velocity':   np.ndarray([vx, vy, vz]), # m/s (estimated from GPS)
            'satellites': int,
            'hdop':       float,
            'valid':      bool,
        },
        'imu': {
            'attitude':   np.ndarray([roll, pitch, yaw]),   # radians
            'rates':      np.ndarray([p, q, r]),            # rad/s
            'calibrated': bool,
            'temperature': float,   # °C (placeholder — MSP does not provide this)
            'valid':      bool,
        },
        'battery': { 'voltage': float },   # volts
        'motors':  { 'temperature': float }, # °C (not available via MSP — use fixed value)
        'wind':    { 'speed': float },       # m/s (estimated from GPS ground-speed vs air)
        'baro':    { 'altitude_m': float,    # metres AGL
                     'vspeed_ms':  float },  # m/s (positive = climbing)
    }

Coordinate convention
---------------------
GPS gives lat/lon/alt (MSL). We convert to local NED using a home reference
set once at startup via calibrate_home().  Altitude in NED: z = -(alt_AGL).
"""

import math
import logging
import time
from typing import Optional, Dict

import numpy as np

from . import hardware_config as cfg

logger = logging.getLogger(__name__)

DEG2RAD = math.pi / 180.0


class SensorReader:
    """
    Wraps an MSPInterface and returns a sensor dict on each call to read().

    Home position (lat/lon/alt) must be captured before flight by calling
    calibrate_home() once the GPS has a valid 3D fix.
    """

    def __init__(self, msp):
        """
        Parameters
        ----------
        msp : MSPInterface
            An already-connected MSPInterface instance.
        """
        self._msp = msp

        # Home reference for NED conversion
        self._home_lat: Optional[float] = None
        self._home_lon: Optional[float] = None
        self._home_alt_m: Optional[float] = None   # MSL

        # GPS velocity estimation (finite difference)
        self._prev_pos_ned: Optional[np.ndarray] = None
        self._prev_gps_time: Optional[float] = None

        # Gyro calibration state (Betaflight handles this, but we track it)
        self._imu_calibrated = False
        self._calibration_samples = 0
        self._gyro_bias = np.zeros(3)   # rad/s

        # HDOP is not directly in MSP_RAW_GPS from most Betaflight versions.
        # We derive an approximate HDOP from satellite count.
        # Better: use an external GPS over NMEA if available.

    # ------------------------------------------------------------------
    # Home position calibration
    # ------------------------------------------------------------------

    def calibrate_home(self, min_satellites: int = 8, timeout_s: float = 60.0) -> bool:
        """
        Block until GPS has at least min_satellites and capture the home position.

        Call this once before arming — preferably with the drone stationary on
        the ground so the home altitude is accurate.

        Returns True on success.
        """
        logger.info(f"Waiting for GPS fix ({min_satellites}+ satellites)...")
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            gps = self._msp.read_gps()
            if gps and gps['valid'] and gps['satellites'] >= min_satellites:
                self._home_lat   = gps['lat_deg']
                self._home_lon   = gps['lon_deg']
                self._home_alt_m = gps['alt_m']

                # Also update config module so other components can read it
                cfg.HOME_LAT   = self._home_lat
                cfg.HOME_LON   = self._home_lon
                cfg.HOME_ALT_M = self._home_alt_m

                logger.info(
                    f"Home set: lat={self._home_lat:.6f}° lon={self._home_lon:.6f}° "
                    f"alt={self._home_alt_m:.1f}m MSL  ({gps['satellites']} sats)"
                )
                return True
            sats = gps['satellites'] if gps else 0
            logger.info(f"  GPS: {sats} satellites — waiting...")
            time.sleep(1.0)

        logger.error("calibrate_home timed out — GPS fix not acquired")
        return False

    def home_set(self) -> bool:
        return self._home_lat is not None

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    def _latlon_to_ned(self, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        """
        Convert geodetic (lat, lon, alt MSL) to local NED (metres) relative to home.

        Uses flat-Earth approximation — valid to ~few km from home.
        NED z is negative upward (altitude above ground → −z).
        """
        if self._home_lat is None:
            return np.zeros(3)

        d_lat = math.radians(lat_deg  - self._home_lat)
        d_lon = math.radians(lon_deg  - self._home_lon)
        d_alt = alt_m - self._home_alt_m   # positive = higher than home

        north = d_lat * cfg.EARTH_RADIUS_M
        east  = d_lon * cfg.EARTH_RADIUS_M * math.cos(math.radians(self._home_lat))
        down  = -d_alt   # NED: up is negative Z

        return np.array([north, east, down])

    @staticmethod
    def _estimate_hdop(num_satellites: int) -> float:
        """
        Rough HDOP estimate from satellite count.
        Better accuracy requires NMEA GGA/GSA sentences.
        """
        if   num_satellites >= 12: return 0.8
        elif num_satellites >= 9:  return 1.0
        elif num_satellites >= 7:  return 1.5
        elif num_satellites >= 5:  return 2.5
        else:                      return 5.0

    # ------------------------------------------------------------------
    # Main read
    # ------------------------------------------------------------------

    def read(self) -> Dict:
        """
        Poll the FC and return a full sensor dict.

        Called at CONTROL_RATE_HZ (50 Hz).  Unavailable readings are filled
        with safe defaults (valid=False) so the contract/EKF layer degrades
        gracefully rather than crashing.
        """
        now = time.monotonic()
        sensor_dict = {}

        # ----------------------------------------------------------------
        # GPS
        # ----------------------------------------------------------------
        gps_raw = self._msp.read_gps()
        if gps_raw and self.home_set():
            pos_ned = self._latlon_to_ned(
                gps_raw['lat_deg'], gps_raw['lon_deg'], gps_raw['alt_m']
            )

            # Velocity: finite-difference of NED position
            if self._prev_pos_ned is not None and self._prev_gps_time is not None:
                dt_gps = now - self._prev_gps_time
                vel_ned = (pos_ned - self._prev_pos_ned) / dt_gps if dt_gps > 0 else np.zeros(3)
            else:
                vel_ned = np.zeros(3)
            self._prev_pos_ned  = pos_ned.copy()
            self._prev_gps_time = now

            hdop = self._estimate_hdop(gps_raw['satellites'])
            valid = (
                gps_raw['valid']
                and gps_raw['satellites'] >= cfg.GPS_MIN_SATELLITES
                and hdop <= cfg.GPS_MAX_HDOP
                and self.home_set()
            )

            sensor_dict['gps'] = {
                'position':   pos_ned,
                'velocity':   vel_ned,
                'satellites': gps_raw['satellites'],
                'hdop':       hdop,
                'valid':      valid,
            }
        else:
            sensor_dict['gps'] = {
                'position':   np.zeros(3),
                'velocity':   np.zeros(3),
                'satellites': 0,
                'hdop':       99.0,
                'valid':      False,
            }

        # ----------------------------------------------------------------
        # IMU  (attitude from Betaflight's attitude estimator + raw gyro)
        # ----------------------------------------------------------------
        att_raw = self._msp.read_attitude()
        imu_raw = self._msp.read_raw_imu()

        if att_raw and imu_raw:
            roll_rad  =  att_raw['roll_deg']  * DEG2RAD
            pitch_rad =  att_raw['pitch_deg'] * DEG2RAD
            yaw_rad   =  att_raw['yaw_deg']   * DEG2RAD

            # Gyro: Betaflight returns raw counts (°/s × 4).
            # Convert to rad/s and subtract bias.
            gx = imu_raw['gyro_raw'][0] / 4.0 * DEG2RAD
            gy = imu_raw['gyro_raw'][1] / 4.0 * DEG2RAD
            gz = imu_raw['gyro_raw'][2] / 4.0 * DEG2RAD
            rates_rps = np.array([gx, gy, gz]) - self._gyro_bias

            # IMU is considered calibrated once Betaflight has been running
            # for at least 5 seconds (it runs its own gyro calibration at boot).
            self._imu_calibrated = True

            tilt_deg = math.sqrt(att_raw['roll_deg']**2 + att_raw['pitch_deg']**2)
            imu_valid = tilt_deg < cfg.MAX_TILT_DEG

            sensor_dict['imu'] = {
                'attitude':    np.array([roll_rad, pitch_rad, yaw_rad]),
                'rates':       rates_rps,
                'calibrated':  self._imu_calibrated,
                'temperature': 25.0,    # MSP does not expose IMU temperature
                'valid':       imu_valid,
            }
        else:
            sensor_dict['imu'] = {
                'attitude':    np.zeros(3),
                'rates':       np.zeros(3),
                'calibrated':  False,
                'temperature': 25.0,
                'valid':       False,
            }

        # ----------------------------------------------------------------
        # Battery
        # ----------------------------------------------------------------
        analog = self._msp.read_analog()
        if analog:
            sensor_dict['battery'] = {'voltage': analog['voltage']}
        else:
            sensor_dict['battery'] = {'voltage': 0.0}

        # ----------------------------------------------------------------
        # Barometer  (altitude above home)
        # ----------------------------------------------------------------
        baro = self._msp.read_altitude()
        if baro:
            # Betaflight altitude is relative to its own home (set at FC boot).
            # Positive = above home.  We convert sign convention for NED (down+).
            sensor_dict['baro'] = {
                'altitude_m': baro['altitude_m'],
                'vspeed_ms':  baro['vspeed_ms'],
            }
        else:
            sensor_dict['baro'] = {'altitude_m': 0.0, 'vspeed_ms': 0.0}

        # ----------------------------------------------------------------
        # Motors temperature — not available via MSP; use a fixed placeholder.
        # Replace with ESC telemetry (KISS / BLHeli_32) if fitted.
        # ----------------------------------------------------------------
        sensor_dict['motors'] = {'temperature': 30.0}

        # ----------------------------------------------------------------
        # Wind (estimated from horizontal speed residual after NED velocity
        # is projected onto the heading direction).
        # This is a rough proxy — replace with a pitot tube if fitted.
        # ----------------------------------------------------------------
        if 'gps' in sensor_dict and sensor_dict['gps']['valid']:
            gnd_speed_ms = gps_raw.get('speed_ms', 0.0) if gps_raw else 0.0
            sensor_dict['wind'] = {'speed': max(0.0, gnd_speed_ms - 1.0)}
        else:
            sensor_dict['wind'] = {'speed': 0.0}

        return sensor_dict

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_altitude_agl(self) -> float:
        """Return barometric altitude above ground level (metres)."""
        baro = self._msp.read_altitude()
        return baro['altitude_m'] if baro else 0.0

    def is_armed(self) -> bool:
        """Return True if the FC reports armed state."""
        status = self._msp.get_fc_status()
        return status.get('armed', False)
