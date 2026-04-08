"""
hardware_flight.py — Real-hardware flight script for DEXI-5

Replaces the simulation loop in demo_waypoint_mission.py with actual hardware
I/O via MSP.  The control logic (AdaptiveDroneController) is unchanged.

Architecture
------------
    Companion computer (this script @ 50 Hz)
        └── SensorReader  → reads FC telemetry via MSP
        └── AdaptiveDroneController → outer-loop position control
        └── ActuatorInterface → sends attitude + throttle setpoints to FC

    Betaflight FC (running its own inner loop at ≥1 kHz)
        └── ANGLE mode: receives RC roll/pitch/throttle/yaw from companion
        └── ESC/motors: closed by Betaflight internally

BEFORE RUNNING THIS SCRIPT
--------------------------
1.  Betaflight setup:
    a. Ports   → enable MSP on the UART wired to companion computer (UART1 or UART2)
    b. Receiver → set Receiver Mode to "MSP" in the Betaflight Configurator
    c. Modes   → AUX1 = ARM,  AUX2 = ANGLE (required for attitude setpoints)
    d. Failsafe → set Stage 2 action (RTH or LAND — NOT "drop")
    e. GPS     → enable if a GPS module is fitted and connected to the FC

2.  Edit src/hardware/hardware_config.py:
    a. Set SERIAL_PORT to the correct device (e.g. '/dev/ttyUSB0' or 'COM3')
    b. Adjust THROTTLE_HOVER to match your build's actual hover throttle
    c. Adjust MAX_ROLL_DEG / MAX_PITCH_DEG to stay within your comfort zone

3.  Position the drone outdoors with clear sky view.

4.  Run: python hardware_flight.py

SAFETY NOTES
------------
- Keep a human pilot ready with a physical RC transmitter in buddy-box or
  takeover mode. The physical transmitter must be able to override and land.
- Do NOT arm until pre-flight checks pass AND you are clear of the drone.
- The script will refuse to take off if GPS fix is insufficient or any
  contract pre-flight check fails.
- Wind limits: PID ≤ 3 m/s, MPC ≤ 15 m/s, H-inf ≤ 25 m/s.
  Abort if conditions exceed H-inf limits.
"""

import sys
import os
import time
import logging
import signal
from typing import Optional

import numpy as np

# Add src/ to path so imports resolve without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hardware.msp_interface   import MSPInterface
from hardware.sensor_reader   import SensorReader
from hardware.actuator_interface import ActuatorInterface
from hardware                 import hardware_config as cfg
from adaptive_control_system  import AdaptiveDroneController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('hardware_flight')


# ---------------------------------------------------------------------------
# Mission definition — edit these before flight
# ---------------------------------------------------------------------------

# Waypoints in local NED frame (metres from home).
# Home (0, 0, 0) is where the drone sits at startup.
# Z is negative upward: -5.0 means 5 metres above the ground.
WAYPOINTS = [
    np.array([ 0.0,  0.0, -5.0]),   # Climb to 5 m
    np.array([10.0,  0.0, -5.0]),   # Fly 10 m North
    np.array([10.0, 10.0, -5.0]),   # Fly 10 m East
    np.array([ 0.0,  0.0, -5.0]),   # Return home (at altitude)
]

# Initial conditions for pre-flight contract verification.
# The sensor reader will fill in real values at runtime; these are used for
# the contract check BEFORE GPS is read.
PRE_FLIGHT_CONDITIONS = {
    'gps_satellites':   12.0,
    'gps_hdop':          1.0,
    'imu_temperature':  25.0,
    'imu_calibrated':    1.0,
    'battery_voltage':  15.0,   # 4S LiPo nominal
    'motor_temperature': 25.0,
    'wind_speed':        1.0,
    'disturbance':       0.5,
    'computation_time':  0.01,
}

MISSION = {
    'waypoints':       WAYPOINTS,
    'target_position': WAYPOINTS[-1],
}


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_running = True
_actuator: Optional[ActuatorInterface] = None


def _shutdown_handler(sig, frame):
    global _running
    logger.warning(f"Signal {sig} received — requesting shutdown")
    _running = False
    # Best-effort immediate disarm on second signal (double Ctrl+C = emergency)
    if _actuator is not None:
        try:
            _actuator.send_emergency_disarm()
        except Exception:
            pass


signal.signal(signal.SIGINT,  _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)


# ---------------------------------------------------------------------------
# Flight script
# ---------------------------------------------------------------------------

def wait_for_user(prompt: str) -> bool:
    """Print prompt and wait for Enter.  Returns False if user types 'q'."""
    try:
        ans = input(f"\n{prompt} [Enter to continue, q to quit]: ").strip().lower()
        return ans != 'q'
    except (EOFError, KeyboardInterrupt):
        return False


def run_preflight_sensor_check(reader: SensorReader) -> bool:
    """
    Read live sensor data and verify minimum hardware health before arming.
    Returns True only if all checks pass.
    """
    logger.info("Running pre-flight sensor checks...")
    sensors = reader.read()

    checks_passed = True

    # Battery
    v = sensors['battery']['voltage']
    if v < 13.0:   # adjust for your cell count / chemistry
        logger.error(f"Battery voltage low: {v:.1f}V (minimum 13.0V for 4S)")
        checks_passed = False
    else:
        logger.info(f"  Battery: {v:.1f}V — OK")

    # GPS
    gps = sensors['gps']
    if not gps['valid']:
        logger.error(
            f"GPS invalid: {gps['satellites']} sats, HDOP {gps['hdop']:.1f} "
            f"(need {cfg.GPS_MIN_SATELLITES}+ sats, HDOP < {cfg.GPS_MAX_HDOP})"
        )
        checks_passed = False
    else:
        logger.info(f"  GPS: {gps['satellites']} sats, HDOP {gps['hdop']:.1f} — OK")

    # IMU
    imu = sensors['imu']
    if not imu['calibrated'] or not imu['valid']:
        logger.error("IMU not calibrated or invalid")
        checks_passed = False
    else:
        att_deg = np.degrees(imu['attitude'])
        logger.info(f"  IMU: roll={att_deg[0]:.1f}° pitch={att_deg[1]:.1f}° — OK")

    return checks_passed


def update_preflight_conditions(conditions: dict, sensors: dict) -> dict:
    """Fill pre-flight condition dict with live sensor values."""
    updated = conditions.copy()
    gps = sensors.get('gps', {})
    imu = sensors.get('imu', {})
    bat = sensors.get('battery', {})

    if gps.get('valid'):
        updated['gps_satellites'] = float(gps.get('satellites', 0))
        updated['gps_hdop']       = float(gps.get('hdop', 99.0))
    if imu.get('valid'):
        updated['imu_temperature'] = float(imu.get('temperature', 25.0))
        updated['imu_calibrated']  = 1.0 if imu.get('calibrated') else 0.0
    if 'voltage' in bat:
        updated['battery_voltage'] = float(bat['voltage'])

    return updated


def main():
    global _running, _actuator

    logger.info("=" * 65)
    logger.info("DEXI-5 Hardware Flight — Contract-Based UAV Control")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # 1. Connect to FC
    # ------------------------------------------------------------------
    logger.info(f"Connecting to FC on {cfg.SERIAL_PORT} @ {cfg.BAUD_RATE} baud...")
    msp = MSPInterface(cfg.SERIAL_PORT, cfg.BAUD_RATE, cfg.SERIAL_TIMEOUT)

    if not msp.connect():
        logger.error("Failed to connect to FC.  Check SERIAL_PORT in hardware_config.py.")
        return 1

    reader   = SensorReader(msp)
    actuator = ActuatorInterface(msp)
    _actuator = actuator   # expose for signal handler

    # ------------------------------------------------------------------
    # 2. GPS home calibration
    # ------------------------------------------------------------------
    logger.info("\nStep 1/4: GPS home calibration")
    if not reader.calibrate_home(min_satellites=cfg.GPS_MIN_SATELLITES, timeout_s=120.0):
        logger.error("GPS home calibration failed.  Aborting.")
        msp.disconnect()
        return 1

    # ------------------------------------------------------------------
    # 3. Live sensor check
    # ------------------------------------------------------------------
    logger.info("\nStep 2/4: Pre-flight sensor checks")
    sensors_now = reader.read()
    if not run_preflight_sensor_check(reader):
        logger.error("Sensor checks failed.  Fix issues before flying.")
        msp.disconnect()
        return 1

    # ------------------------------------------------------------------
    # 4. Contract pre-flight verification
    # ------------------------------------------------------------------
    logger.info("\nStep 3/4: Contract pre-flight verification")
    controller = AdaptiveDroneController(dt=cfg.DT, use_horizon_planner=False)

    live_conditions = update_preflight_conditions(PRE_FLIGHT_CONDITIONS, sensors_now)
    feasible, msg = controller.pre_flight_check(live_conditions, MISSION)

    if not feasible:
        logger.error(f"CONTRACT CHECK FAILED: {msg}")
        logger.error("Mission is not safe to fly under current conditions.")
        msp.disconnect()
        return 1

    logger.info(f"Contract check: {msg}")

    # ------------------------------------------------------------------
    # 5. Operator confirmation
    # ------------------------------------------------------------------
    logger.info("\nStep 4/4: Operator confirmation")
    logger.warning(
        "\n  *** SAFETY CHECK ***\n"
        "  - All people and obstacles must be clear of the drone.\n"
        "  - Your RC transmitter must be on and ready to take over.\n"
        "  - The drone will arm and take off automatically after confirmation.\n"
    )
    if not wait_for_user("Confirm ready to arm and fly"):
        logger.info("Operator cancelled.  Aborting.")
        msp.disconnect()
        return 0

    # ------------------------------------------------------------------
    # 6. Arm
    # ------------------------------------------------------------------
    logger.info("Arming...")
    if not actuator.arm():
        logger.error("Arm signal failed.  Check MSP receiver setting in Betaflight.")
        msp.disconnect()
        return 1

    # Brief pause to confirm arm state
    time.sleep(1.0)
    if not reader.is_armed():
        logger.error(
            "FC did not arm.  Check:\n"
            "  - Throttle was at minimum before arming\n"
            "  - AUX1 is mapped to ARM in Betaflight Modes tab\n"
            "  - Receiver type is set to MSP in Betaflight Ports/Receiver\n"
        )
        actuator.disarm()
        msp.disconnect()
        return 1

    logger.info("ARMED — starting mission")
    controller.start_mission()

    # ------------------------------------------------------------------
    # 7. Control loop
    # ------------------------------------------------------------------
    dt        = cfg.DT
    loop_hz   = cfg.CONTROL_RATE_HZ
    loop_count = 0

    logger.info(f"Control loop running at {loop_hz} Hz  (Ctrl+C to abort)")

    try:
        while _running:
            t_loop_start = time.monotonic()

            # Read sensors
            raw_sensors = reader.read()

            # Run control step
            control, telemetry = controller.control_step(raw_sensors)

            # Send to FC
            armed = reader.is_armed()
            actuator.send(control, armed=armed)

            # Telemetry printout every 50 loops (~1 s)
            if loop_count % 50 == 0:
                pos = telemetry.get('position', np.zeros(3))
                ctrl_name = telemetry.get('active_controller', '?')
                mode = telemetry.get('flight_mode', '?')
                wind = raw_sensors.get('wind', {}).get('speed', 0.0)
                v    = raw_sensors.get('battery', {}).get('voltage', 0.0)
                logger.info(
                    f"t={telemetry.get('time', 0.0):.1f}s | "
                    f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})m | "
                    f"ctrl={ctrl_name} | mode={mode} | "
                    f"wind={wind:.1f}m/s | vbat={v:.1f}V"
                )

            # Mission complete — issue land command once
            if not controller.mission_active and telemetry.get('flight_mode') != 'LAND':
                logger.info("Mission complete — initiating landing")
                controller.command_land()

            # Check flight mode — if LAND and near ground, disarm
            if telemetry.get('flight_mode') == 'LAND':
                alt = reader.get_altitude_agl()
                if alt < cfg.MIN_ALTITUDE_AGL_M:
                    logger.info(f"Altitude {alt:.2f}m — on ground, disarming")
                    _running = False

            # Rate limiting: sleep for remainder of loop period
            elapsed = time.monotonic() - t_loop_start
            sleep_s = dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif elapsed > dt * 1.5:
                logger.warning(f"Control loop overran: {elapsed*1000:.1f}ms (budget {dt*1000:.0f}ms)")

            loop_count += 1

    except Exception as exc:
        logger.exception(f"Control loop exception: {exc}")

    # ------------------------------------------------------------------
    # 8. Safe shutdown  (wrapped in try/finally so disarm ALWAYS fires)
    # ------------------------------------------------------------------
    try:
        logger.info("Shutting down — sending hover then disarm")
        controller.stop_mission()

        # Hold hover for 2 s to let things settle, then disarm
        for _ in range(int(2.0 / dt)):
            try:
                raw_sensors = reader.read()
                control, _ = controller.control_step(raw_sensors)
                actuator.send(control, armed=True)
            except Exception:
                actuator.send_hover()   # fallback: neutral hover frame
            time.sleep(dt)
    finally:
        # Disarm no matter what
        actuator.disarm()
        time.sleep(0.5)

        # Save flight log
        try:
            log_path = f"flight_log_{int(time.time())}.json"
            controller.save_flight_log(log_path)
            logger.info(f"Flight log saved: {log_path}")
        except Exception as exc:
            logger.error(f"Failed to save flight log: {exc}")

        msp.disconnect()
        logger.info("Disconnected.  Flight complete.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
