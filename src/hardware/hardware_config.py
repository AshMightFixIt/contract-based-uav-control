"""
DEXI-5 Hardware Configuration

Frame   : DEXI-5 (5-inch quad, kit build — not RTF)
Protocol: Betaflight MSP v1 over UART (companion computer ↔ FC)

Betaflight setup required before flight
--------------------------------------
1. Ports tab  → enable MSP on the UART connected to companion computer
2. Receiver   → set Receiver Mode to "MSP" (Betaflight 4.x+)
3. Modes      → map AUX1 to ARM
4. Modes      → map AUX2 to ANGLE mode (required for attitude setpoints)
5. GPS        → enable if an external GPS module is fitted
6. Failsafe   → configure Stage 2 to RTH or LAND (NOT drop/disarm)

Coordinate frames
-----------------
This codebase uses NED (North-East-Down):
    +X = North, +Y = East, +Z = Down
    Altitude above ground → negative Z values

Betaflight uses its own body frame internally; MSP returns attitude in
degrees (roll/pitch ×10 in MSP_ATTITUDE, yaw in degrees).
GPS altitude from Betaflight is in centimetres above sea level (MSL).
A launch-site MSL reference is captured at startup for AGL conversion.
"""

# ---------------------------------------------------------------------------
# Serial connection
# ---------------------------------------------------------------------------
SERIAL_PORT    = '/dev/ttyUSB0'   # Linux. Use 'COMx' on Windows.
BAUD_RATE      = 115200
SERIAL_TIMEOUT = 0.05             # seconds — keep short to avoid blocking the loop

# ---------------------------------------------------------------------------
# Loop rates
# ---------------------------------------------------------------------------
CONTROL_RATE_HZ     = 50    # Outer loop (position → attitude setpoints)
TELEMETRY_RATE_HZ   = 100   # How often we poll the FC for sensor data
DT                  = 1.0 / CONTROL_RATE_HZ   # 0.02 s

# ---------------------------------------------------------------------------
# Betaflight RC channel mapping  (standard AETR1234 layout)
# ---------------------------------------------------------------------------
RC_ROLL      = 0   # Channel 1 — roll angle setpoint
RC_PITCH     = 1   # Channel 2 — pitch angle setpoint
RC_THROTTLE  = 2   # Channel 3 — throttle (motor power)
RC_YAW       = 3   # Channel 4 — yaw rate setpoint
RC_ARM       = 4   # Channel 5 / AUX1 — arm switch
RC_MODE      = 5   # Channel 6 / AUX2 — flight mode (ANGLE, etc.)
RC_NUM_CHANNELS = 8

# PWM pulse-width limits (microseconds)
RC_MIN   = 1000
RC_MID   = 1500
RC_MAX   = 2000

# Arm/disarm threshold on AUX1
RC_ARMED   = 1800   # value that arms the FC
RC_DISARMED = 1000  # value that disarms the FC

# ANGLE mode active on AUX2
RC_ANGLE_MODE_ON  = 1800
RC_ANGLE_MODE_OFF = 1000

# ---------------------------------------------------------------------------
# DEXI-5 physical parameters  (update once you have your build spec)
# ---------------------------------------------------------------------------
MASS_KG          = 0.35     # approximate all-up weight with battery [kg]
MOTOR_KV         = 2306     # motor KV rating (for reference)
PROP_SIZE_INCH   = 5        # propeller size
THRUST_TO_WEIGHT = 4.0      # typical 5-inch with 2306-class motors

# ---------------------------------------------------------------------------
# Actuator output limits sent to Betaflight
# ---------------------------------------------------------------------------
# In ANGLE mode Betaflight clamps roll/pitch to its own configured limits.
# These values are what *we* ask for — set them within Betaflight's limits.
MAX_ROLL_DEG    = 30.0    # degrees — keep conservative for initial tests
MAX_PITCH_DEG   = 30.0    # degrees
MAX_YAW_RATE_DPS = 180.0  # degrees/second

# Throttle range mapping: our normalised thrust [0, 1] → RC [1000, 2000]
THROTTLE_IDLE   = 1150    # minimum armed throttle (above spin-up threshold)
THROTTLE_HOVER  = 1500    # approximate hover throttle (tune per build)
THROTTLE_MAX    = 1900    # leave headroom for stabilisation

# ---------------------------------------------------------------------------
# Sensor health thresholds  (used by contract checks)
# ---------------------------------------------------------------------------
GPS_MIN_SATELLITES = 6
GPS_MAX_HDOP       = 2.0   # horizontal dilution of precision
IMU_MAX_TEMP_C     = 85.0
BARO_VALID         = True  # set False if no barometer fitted

# ---------------------------------------------------------------------------
# Safety hard limits
# ---------------------------------------------------------------------------
MAX_ALTITUDE_AGL_M  = 30.0    # metres above ground — hard ceiling
MIN_ALTITUDE_AGL_M  = 0.3     # metres — landing threshold
MAX_VELOCITY_MS     = 8.0     # m/s total speed limit (conservative)
MAX_TILT_DEG        = 45.0    # emergency disarm if exceeded
EMERGENCY_LAND_WIND = 15.0    # m/s — force landing if wind exceeds this

# ---------------------------------------------------------------------------
# GPS reference (set at startup)
# ---------------------------------------------------------------------------
# These are populated by SensorReader.calibrate_home() at startup.
HOME_LAT  = None   # degrees
HOME_LON  = None   # degrees
HOME_ALT_M = None  # metres MSL at launch site

# Earth radius for lat/lon → local NED conversion
EARTH_RADIUS_M = 6_371_000.0
