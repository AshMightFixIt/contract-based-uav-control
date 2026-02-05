"""
Demonstration: Horizon-Based Contract Planning with Pacti

Shows:
1. Contract cascade over planning horizon
2. Safety margin optimization
3. Automatic re-planning when margins are tight
4. Horizon reduction under extreme conditions
5. Controller switching (PID <-> H-inf)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import logging

logging.basicConfig(level=logging.WARNING, format='%(message)s')

from adaptive_control_system import AdaptiveDroneController

# Quieter logging for demo
logging.getLogger('planning').setLevel(logging.WARNING)
logging.getLogger('estimation').setLevel(logging.WARNING)
logging.getLogger('contracts').setLevel(logging.WARNING)


def run_demo():
    print("=" * 70)
    print("HORIZON-BASED CONTRACT PLANNING DEMONSTRATION")
    print("=" * 70)
    print("""
Architecture:
  Sensors -> EKF -> [HORIZON PLANNER] -> Supervisor -> [PID|Hinf] -> CBF -> Actuators
                          |
                   Pacti Contract Cascade
                          |
                   Safety Margin Monitor
                          |
                   Re-planning Trigger
    """)

    # Create controller with horizon planner
    controller = AdaptiveDroneController(dt=0.02, use_horizon_planner=True)

    # Pre-flight
    print("\n[1] PRE-FLIGHT CHECK")
    print("-" * 70)
    initial_conditions = {
        'gps_satellites': 12.0,
        'imu_temperature_stable': 1.0,
        'battery_voltage': 12.4,
        'motor_temperature': 30.0,
        'wind_speed': 0.5,
        'disturbance': 0.2,
    }

    waypoints = [
        np.array([5.0, 0.0, -5.0]),
        np.array([5.0, 5.0, -8.0]),
    ]

    mission = {
        'target_position': waypoints[0],
        'waypoints': waypoints
    }

    feasible, msg = controller.pre_flight_check(initial_conditions, mission)
    print(f"Result: {msg}")

    if not feasible:
        print("Mission aborted!")
        return

    controller.start_mission()

    # Simulation phases
    phases = [
        ("NOMINAL",   0.0,  3.0, 0.5,  True,  "Low wind, PID expected"),
        ("WIND_UP",   3.0,  6.0, 4.0,  True,  "Wind increasing, may switch to H-inf"),
        ("EXTREME",   6.0, 10.0, 10.0, True,  "Extreme wind, horizon reduction expected"),
        ("RECOVERY", 10.0, 14.0, 2.0,  True,  "Wind decreasing, may switch back to PID"),
    ]

    print("\n[2] FLIGHT SIMULATION WITH HORIZON PLANNING")
    print("-" * 70)
    print(f"{'Time':>6} | {'Phase':^10} | {'Wind':>5} | {'Controller':^6} | "
          f"{'Horizon':>7} | {'Margin':>7} | {'Status':^12} | Notes")
    print("-" * 70)

    step = 0
    for phase_name, t_start, t_end, wind_speed, gps_avail, description in phases:
        print(f"\n--- {phase_name}: {description} ---")

        while controller.time < t_end:
            # Generate sensor data
            raw_sensors = {
                'gps': {
                    'position': np.array([step * 0.02, 0.0, -5.0]),
                    'velocity': np.array([0.5 + wind_speed * 0.3, wind_speed * 0.5, 0.0]),
                    'valid': gps_avail,
                    'satellites': 12 if gps_avail else 3,
                    'hdop': 1.0 if gps_avail else 8.0
                },
                'imu': {
                    'attitude': np.array([0.05, 0.02, 0.0]),
                    'rates': np.array([0.01, 0.01, 0.0]),
                    'valid': True,
                    'calibrated': True,
                    'temperature': 25.0
                },
                'battery': {'voltage': 12.4},
                'motors': {'temperature': 30.0}
            }

            # Control step
            control, telemetry = controller.control_step(raw_sensors)

            # Print status every 0.5 seconds
            if step % 25 == 0:
                planner = telemetry.get('planner', {})
                margin = planner.get('safety_margin', -1)
                horizon = planner.get('horizon', 0)
                status = planner.get('plan_status', 'N/A')
                replanned = planner.get('replanned', False)

                notes = []
                if replanned:
                    notes.append("REPLAN")
                if planner.get('emergency', False):
                    notes.append("EMERG")
                notes_str = ','.join(notes) if notes else ""

                print(f"{controller.time:6.2f} | {phase_name:^10} | {wind_speed:5.1f} | "
                      f"{telemetry['active_controller']:^6} | {horizon:7d} | {margin:7.2f} | "
                      f"{status:^12} | {notes_str}")

            step += 1

    controller.stop_mission()

    # Summary
    print("\n" + "=" * 70)
    print("[3] DEMONSTRATION SUMMARY")
    print("=" * 70)

    if controller.horizon_planner:
        status = controller.horizon_planner.get_status()
        print(f"\nHorizon Planner Status:")
        print(f"  - Emergency mode: {status['emergency_mode']}")
        print(f"  - Consecutive failures: {status['consecutive_failures']}")
        print(f"  - Final plan status: {status['current_plan_status']}")
        print(f"  - Final safety margin: {status['safety_margin']:.2f}")

    print(f"\nCBF Statistics:")
    cbf_stats = controller.cbf_filter.get_statistics()
    print(f"  - Interventions: {cbf_stats['intervention_count']}")

    print("""
KEY OBSERVATIONS:
1. NOMINAL: PID controller with full horizon (10 steps), high safety margin
2. WIND_UP: Horizon planner detects wind, may switch to H-inf preemptively
3. EXTREME: Horizon reduced (10 -> 3) to maintain feasibility under uncertainty
4. RECOVERY: Wind subsides, safety margin improves, may switch back to PID

The horizon planner uses Pacti's formal contract composition to:
- Cascade contracts over the planning horizon
- Optimize safety margins at each step
- Trigger re-planning when margins fall below threshold
- Reduce horizon when full horizon becomes infeasible
""")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
