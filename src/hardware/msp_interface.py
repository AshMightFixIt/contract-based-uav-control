"""
MSP v1 (MultiWii Serial Protocol) interface for Betaflight

Frame format:
    Preamble : '$' 'M'
    Direction: '<' (host→FC) or '>' (FC→host)
    Length   : 1 byte  — payload byte count
    Command  : 1 byte  — MSP command code
    Payload  : <length> bytes
    Checksum : 1 byte  — XOR of length, command, and all payload bytes

Usage:
    msp = MSPInterface('/dev/ttyUSB0')
    msp.connect()
    data = msp.read_attitude()   # {'roll': x, 'pitch': y, 'yaw': z}
    msp.set_rc_channels([1500, 1500, 1200, 1500, 1000, 1800, 1500, 1500])
    msp.disconnect()
"""

import struct
import logging
import time
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MSP command codes (Betaflight-compatible subset)
# ---------------------------------------------------------------------------
MSP_API_VERSION   = 1
MSP_FC_VARIANT    = 2
MSP_FC_VERSION    = 3
MSP_STATUS        = 101   # cycle time, i2c errors, sensor flags, flight modes
MSP_RAW_IMU       = 102   # accelerometer + gyroscope raw counts
MSP_ATTITUDE      = 108   # roll, pitch (×10 deg), yaw (deg)
MSP_ALTITUDE      = 109   # altitude cm, vario cm/s
MSP_RAW_GPS       = 106   # fix, num_sat, lat, lon, alt, speed, course
MSP_ANALOG        = 110   # vbat, current, rssi, amperage
MSP_SET_RAW_RC    = 200   # override RC channels (requires MSP receiver mode)
MSP_SET_ATTITUDE  = 212   # not standard — use SET_RAW_RC instead


class MSPError(Exception):
    """Raised when the FC returns an error frame or communication fails."""


class MSPInterface:
    """
    Low-level MSP v1 framing and serial communication.

    Thread safety: not thread-safe — call from a single control thread.
    """

    PREAMBLE = b'$M'

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.05):
        self.port   = port
        self.baud   = baud
        self.timeout = timeout
        self._serial = None
        self._connected = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Open the serial port. Returns True on success."""
        try:
            import serial  # deferred import so non-hardware runs don't need pyserial
        except ImportError:
            raise ImportError(
                "pyserial is required for hardware mode. Install with: pip install pyserial"
            )
        try:
            self._serial = serial.Serial(
                self.port,
                self.baud,
                timeout=self.timeout,
                write_timeout=0.1,
            )
            self._connected = True
            logger.info(f"MSP connected on {self.port} @ {self.baud} baud")

            # Flush any stale data
            self._serial.reset_input_buffer()
            time.sleep(0.1)

            # Verify we're talking to a Betaflight FC
            self._verify_fc()
            return True
        except Exception as exc:
            logger.error(f"MSP connect failed: {exc}")
            self._connected = False
            return False

    def disconnect(self):
        """Close the serial port."""
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._connected = False
        logger.info("MSP disconnected")

    @property
    def connected(self) -> bool:
        return self._connected and self._serial is not None and self._serial.is_open

    # ------------------------------------------------------------------
    # Internal framing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _checksum(length: int, cmd: int, payload: bytes) -> int:
        """XOR checksum over length + command + payload bytes."""
        crc = length ^ cmd
        for b in payload:
            crc ^= b
        return crc

    def _build_frame(self, cmd: int, payload: bytes = b'') -> bytes:
        """Build a complete MSP request frame."""
        length = len(payload)
        crc = self._checksum(length, cmd, payload)
        return self.PREAMBLE + b'<' + bytes([length, cmd]) + payload + bytes([crc])

    def _send(self, cmd: int, payload: bytes = b''):
        """Send a request frame to the FC."""
        if not self.connected:
            raise MSPError("Not connected")
        frame = self._build_frame(cmd, payload)
        self._serial.write(frame)

    def _recv(self) -> tuple[int, bytes]:
        """
        Read one MSP response frame from the FC.
        Returns (command, payload) or raises MSPError.
        """
        # Synchronise on preamble '$M'
        buf = b''
        deadline = time.monotonic() + self.timeout * 4
        while time.monotonic() < deadline:
            byte = self._serial.read(1)
            if not byte:
                continue
            buf += byte
            if buf[-2:] == self.PREAMBLE:
                break
        else:
            raise MSPError("Timeout waiting for MSP preamble")

        # Direction byte
        direction = self._serial.read(1)
        if not direction:
            raise MSPError("Timeout reading direction byte")
        if direction == b'!':
            raise MSPError("FC returned error frame")

        # Length, command, payload, checksum
        header = self._serial.read(2)
        if len(header) < 2:
            raise MSPError("Timeout reading MSP header")
        length, cmd = header[0], header[1]

        payload = self._serial.read(length) if length > 0 else b''
        if len(payload) < length:
            raise MSPError(f"Short payload: expected {length}, got {len(payload)}")

        crc_byte = self._serial.read(1)
        if not crc_byte:
            raise MSPError("Timeout reading checksum")

        expected_crc = self._checksum(length, cmd, payload)
        if crc_byte[0] != expected_crc:
            raise MSPError(
                f"CRC mismatch: got 0x{crc_byte[0]:02x}, expected 0x{expected_crc:02x}"
            )

        return cmd, payload

    def _request(self, cmd: int, payload: bytes = b'') -> bytes:
        """Send a request and return the response payload."""
        self._send(cmd, payload)
        resp_cmd, resp_payload = self._recv()
        if resp_cmd != cmd:
            raise MSPError(f"Unexpected response cmd {resp_cmd} (expected {cmd})")
        return resp_payload

    # ------------------------------------------------------------------
    # FC identification
    # ------------------------------------------------------------------

    def _verify_fc(self):
        """Check that the FC responds to MSP_API_VERSION."""
        try:
            payload = self._request(MSP_API_VERSION)
            # Betaflight returns: protocol version (1B), api_major (1B), api_minor (1B)
            if len(payload) >= 3:
                proto, major, minor = payload[0], payload[1], payload[2]
                logger.info(f"FC API version: {major}.{minor} (proto {proto})")
            else:
                logger.warning("FC responded but API version payload was short")
        except MSPError as e:
            logger.warning(f"FC verification warning (non-fatal): {e}")

    def get_fc_status(self) -> Dict:
        """Read MSP_STATUS: cycle time, sensor flags, armed state."""
        try:
            payload = self._request(MSP_STATUS)
            # cycleTime(2) + i2cError(2) + sensorFlags(2) + flightModeFlags(4) + ...
            if len(payload) >= 10:
                cycle_time    = struct.unpack_from('<H', payload, 0)[0]
                i2c_errors    = struct.unpack_from('<H', payload, 2)[0]
                sensor_flags  = struct.unpack_from('<H', payload, 4)[0]
                flight_modes  = struct.unpack_from('<I', payload, 6)[0]
                armed = bool(flight_modes & 0x01)
                return {
                    'cycle_time_us': cycle_time,
                    'i2c_errors':    i2c_errors,
                    'sensor_flags':  sensor_flags,
                    'armed':         armed,
                }
        except MSPError as e:
            logger.debug(f"MSP_STATUS failed: {e}")
        return {}

    # ------------------------------------------------------------------
    # Sensor reads
    # ------------------------------------------------------------------

    def read_attitude(self) -> Optional[Dict]:
        """
        Read MSP_ATTITUDE.
        Returns dict with roll, pitch (degrees, ×10 in wire format), yaw (degrees).
        """
        try:
            payload = self._request(MSP_ATTITUDE)
            # roll(i16) pitch(i16) yaw(i16) — roll/pitch in 0.1° units, yaw in 1° units
            if len(payload) >= 6:
                raw_roll, raw_pitch, raw_yaw = struct.unpack_from('<hhh', payload)
                return {
                    'roll_deg':  raw_roll  / 10.0,
                    'pitch_deg': raw_pitch / 10.0,
                    'yaw_deg':   float(raw_yaw),
                }
        except MSPError as e:
            logger.debug(f"MSP_ATTITUDE failed: {e}")
        return None

    def read_raw_imu(self) -> Optional[Dict]:
        """
        Read MSP_RAW_IMU.
        Returns raw accel (g × 512) and gyro (°/s × 4) counts.
        Caller is responsible for scaling.
        """
        try:
            payload = self._request(MSP_RAW_IMU)
            # ax ay az (i16 each, unit = g/512) then gx gy gz (i16 each, unit = °/s/4)
            if len(payload) >= 12:
                vals = struct.unpack_from('<6h', payload)
                return {
                    'acc_raw':  vals[0:3],   # scale by /512 for g
                    'gyro_raw': vals[3:6],   # scale by /4 for °/s
                }
        except MSPError as e:
            logger.debug(f"MSP_RAW_IMU failed: {e}")
        return None

    def read_altitude(self) -> Optional[Dict]:
        """
        Read MSP_ALTITUDE (barometer / sonar fused altitude).
        Returns altitude in metres and vertical speed in m/s.
        """
        try:
            payload = self._request(MSP_ALTITUDE)
            # alt(i32 cm) + vario(i16 cm/s)
            if len(payload) >= 6:
                alt_cm   = struct.unpack_from('<i', payload, 0)[0]
                vario_cm = struct.unpack_from('<h', payload, 4)[0]
                return {
                    'altitude_m':  alt_cm   / 100.0,
                    'vspeed_ms':   vario_cm / 100.0,
                }
        except MSPError as e:
            logger.debug(f"MSP_ALTITUDE failed: {e}")
        return None

    def read_gps(self) -> Optional[Dict]:
        """
        Read MSP_RAW_GPS.
        Returns fix type, satellite count, lat/lon (degrees × 1e7), altitude (cm),
        ground speed (cm/s), and ground course (degrees × 10).
        """
        try:
            payload = self._request(MSP_RAW_GPS)
            # fix(u8) numSat(u8) lat(i32 deg×1e7) lon(i32 deg×1e7)
            # alt(u16 cm) speed(u16 cm/s) groundCourse(u16 deg×10)
            if len(payload) >= 14:
                fix      = payload[0]
                num_sat  = payload[1]
                lat      = struct.unpack_from('<i', payload, 2)[0] / 1e7
                lon      = struct.unpack_from('<i', payload, 6)[0] / 1e7
                alt_cm   = struct.unpack_from('<H', payload, 10)[0]
                speed_cm = struct.unpack_from('<H', payload, 12)[0]
                return {
                    'fix':       fix,
                    'satellites': num_sat,
                    'lat_deg':   lat,
                    'lon_deg':   lon,
                    'alt_m':     alt_cm / 100.0,   # MSL metres
                    'speed_ms':  speed_cm / 100.0,
                    'valid':     fix >= 2 and num_sat >= 4,
                }
        except MSPError as e:
            logger.debug(f"MSP_RAW_GPS failed: {e}")
        return None

    def read_analog(self) -> Optional[Dict]:
        """
        Read MSP_ANALOG: battery voltage, current, RSSI.
        """
        try:
            payload = self._request(MSP_ANALOG)
            # vbat(u8 0.1V) mAh(u16) rssi(u16 0-1023) amperage(u16 0.01A)
            if len(payload) >= 7:
                vbat_raw  = payload[0]
                rssi_raw  = struct.unpack_from('<H', payload, 3)[0]
                amp_raw   = struct.unpack_from('<H', payload, 5)[0]
                return {
                    'voltage':   vbat_raw / 10.0,         # volts
                    'rssi':      rssi_raw / 1023.0,        # 0–1
                    'current_a': amp_raw  / 100.0,         # amperes
                }
        except MSPError as e:
            logger.debug(f"MSP_ANALOG failed: {e}")
        return None

    # ------------------------------------------------------------------
    # Actuator output
    # ------------------------------------------------------------------

    def set_rc_channels(self, channels: List[int]) -> bool:
        """
        Send MSP_SET_RAW_RC to override RC channels.

        channels : list of up to 16 PWM values in microseconds [1000, 2000].
        The FC must be configured with "MSP" as the receiver type.

        Returns True if the frame was sent without error.
        """
        if not self.connected:
            logger.error("set_rc_channels called while disconnected")
            return False

        # Clamp and pad to at least 8 channels
        vals = [max(1000, min(2000, int(c))) for c in channels]
        while len(vals) < 8:
            vals.append(1500)

        payload = struct.pack(f'<{len(vals)}H', *vals)
        try:
            # MSP_SET_RAW_RC is fire-and-forget: Betaflight does not always send
            # an ACK for set commands, and waiting for one would stall the 50 Hz
            # control loop.  Just send the frame and return immediately.
            self._send(MSP_SET_RAW_RC, payload)
            return True
        except MSPError as e:
            logger.warning(f"set_rc_channels failed: {e}")
            return False
