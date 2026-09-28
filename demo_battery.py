"""
Battery Module Demonstration

Tests:
1. Power draw model per controller across wind conditions
2. OCV curve and voltage sag under load
3. Discharge simulation over a multi-phase mission
4. Remaining flight time estimates
5. Controller energy ranking (battery-optimal vs min-time)
6. Critical/depleted detection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from utils.battery_model import BatteryModel, BatteryConfig, ControllerPowerProfile


CONTROLLERS = ['PID', 'MPC', 'Hinf']
SEP = "=" * 70


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def run_demo():
    print(SEP)
    print("  BATTERY MODULE DEMONSTRATION")
    print(SEP)

    battery = BatteryModel()
    profile = ControllerPowerProfile(battery)

    # ------------------------------------------------------------------
    # 1. OCV curve
    # ------------------------------------------------------------------
    section("1. OCV CURVE — Voltage vs State of Charge")
    print(f"  {'SoC':>6} | {'V_oc (V)':>10} | {'Bar':}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*30}")
    for soc in np.linspace(1.0, 0.0, 11):
        v = battery.ocv(soc)
        bar = "#" * int((v - 10.5) / 0.21)
        print(f"  {soc:6.0%} | {v:10.2f} | {bar}")

    # ------------------------------------------------------------------
    # 2. Power draw vs control effort
    # ------------------------------------------------------------------
    section("2. POWER DRAW MODEL — P = P_hover × (ce/0.5)^1.5 + k_wind × v²")
    print(f"  {'ce':>5} | {'P (wind=0)':>12} | {'P (wind=5)':>12} | {'P (wind=10)':>12}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for ce in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p0  = battery.power_draw(ce, 0.0)
        p5  = battery.power_draw(ce, 5.0)
        p10 = battery.power_draw(ce, 10.0)
        print(f"  {ce:5.1f} | {p0:10.1f} W | {p5:10.1f} W | {p10:10.1f} W")

    # ------------------------------------------------------------------
    # 3. Per-controller effort and power at wind speeds
    # ------------------------------------------------------------------
    section("3. CONTROLLER POWER PROFILES — effort upper bound from contracts")
    winds = [0, 2, 4, 6, 8, 10]
    print(f"  {'Controller':^8} | {'Wind':>5} | {'Effort':>7} | {'Power':>8} | {'Current':>8} | {'V_terminal':>10}")
    print(f"  {'-'*8}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    for ctrl in CONTROLLERS:
        for wind in winds:
            ce   = profile.effort_estimate(ctrl, wind)
            P    = profile.power_estimate(ctrl, wind)
            I    = battery.current_draw(ce, wind)
            V_t  = battery.terminal_voltage_under_load(ce, wind)
            print(f"  {ctrl:^8} | {wind:5.0f} | {ce:7.3f} | {P:6.1f} W | {I:6.2f} A | {V_t:8.2f} V")
        print()

    # ------------------------------------------------------------------
    # 4. Energy ranking over horizon
    # ------------------------------------------------------------------
    section("4. ENERGY RANKING — battery-optimal vs min-time (horizon=10, dt=0.1s)")
    dt = 0.1
    horizon = 10
    for wind in [1, 5, 10]:
        ranked_energy = profile.rank_by_energy(CONTROLLERS, wind, horizon, dt)
        ranked_time   = profile.rank_by_time(CONTROLLERS, wind)
        print(f"\n  Wind = {wind} m/s:")
        print(f"    Battery-optimal order: "
              + " ->".join(f"{c}({e*1000:.2f}mWh)" for c, e in ranked_energy))
        print(f"    Min-time order:        "
              + " ->".join(f"{c}({t:.0f}s)" for c, t in ranked_time))

    # ------------------------------------------------------------------
    # 5. Discharge simulation — multi-phase mission
    # ------------------------------------------------------------------
    section("5. DISCHARGE SIMULATION — multi-phase mission")

    battery_sim = BatteryModel(initial_soc=1.0)
    profile_sim = ControllerPowerProfile(battery_sim)
    dt_sim = 0.02   # 50 Hz, matches control loop

    phases = [
        ("TAKEOFF",   60,  'PID',  0.0,  "Calm, PID hover"),
        ("CRUISE",   120,  'MPC',  2.0,  "Light wind, MPC cruise"),
        ("WIND_UP",   60,  'MPC',  5.0,  "Medium wind, MPC"),
        ("EXTREME",   30,  'Hinf', 10.0, "High wind, H-inf"),
        ("RECOVERY",  90,  'MPC',  3.0,  "Wind drop, MPC"),
        ("LAND",      30,  'PID',  0.5,  "Calm, PID descent"),
    ]

    print(f"\n  {'Phase':^10} | {'t':>5} | {'Ctrl':^5} | {'Wind':>5} | "
          f"{'SoC':>6} | {'V_oc':>6} | {'Power':>7} | {'Rmg':>8} | Status")
    print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-"
          f"{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

    t = 0.0
    for phase_name, duration_s, ctrl, wind, description in phases:
        steps = int(duration_s / dt_sim)
        printed = False
        for _ in range(steps):
            ce   = profile_sim.effort_estimate(ctrl, wind)
            tele = battery_sim.update(ce, wind, dt_sim)
            t   += dt_sim

            if not printed:
                ft  = battery_sim.flight_time_remaining(ce, wind)
                status = ("CRITICAL" if battery_sim.is_critical() else
                          "DEPLETED" if battery_sim.is_depleted() else "OK")
                print(f"  {phase_name:^10} | {t:5.0f} | {ctrl:^5} | {wind:5.1f} | "
                      f"{tele['soc']:5.1%} | {battery_sim.ocv(tele['soc']):6.2f} | "
                      f"{tele['power_w']:5.1f} W | {ft:6.0f} s | {status}")
                printed = True

        # End-of-phase summary
        ft  = battery_sim.flight_time_remaining(ce, wind)
        status = ("CRITICAL" if battery_sim.is_critical() else
                  "DEPLETED" if battery_sim.is_depleted() else "OK")
        print(f"  {phase_name+' END':^10} | {t:5.0f} | {ctrl:^5} | {wind:5.1f} | "
              f"{battery_sim.soc:5.1%} | {battery_sim.ocv(battery_sim.soc):6.2f} | "
              f"{tele['power_w']:5.1f} W | {ft:6.0f} s | {status}")

    # ------------------------------------------------------------------
    # 6. Final battery status
    # ------------------------------------------------------------------
    section("6. FINAL BATTERY STATUS")
    status = battery_sim.get_status()
    for k, v in status.items():
        unit = {'soc_pct': '%', 'ocv_v': 'V', 'remaining_mah': 'mAh',
                'consumed_mah': 'mAh', 'energy_consumed_wh': 'Wh',
                'time_elapsed_s': 's'}.get(k, '')
        if isinstance(v, float):
            print(f"  {k:<25}: {v:>10.2f} {unit}")

    # ------------------------------------------------------------------
    # 7. Voltage sag under load
    # ------------------------------------------------------------------
    section("7. VOLTAGE SAG — terminal vs OCV at end-of-mission SoC")
    soc_eom = battery_sim.soc
    v_oc = battery_sim.ocv(soc_eom)
    print(f"\n  SoC at end of mission: {soc_eom:.1%}  (V_oc = {v_oc:.2f} V)\n")
    print(f"  {'Controller':^8} | {'Wind':>5} | {'Effort':>7} | {'Current':>8} | "
          f"{'V_oc':>7} | {'V_terminal':>10} | {'Sag':>6}")
    print(f"  {'-'*8}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}-+-{'-'*6}")
    for ctrl in CONTROLLERS:
        for wind in [0, 5, 10]:
            ce  = profile_sim.effort_estimate(ctrl, wind)
            I   = battery_sim.current_draw(ce, wind)
            V_t = battery_sim.terminal_voltage_under_load(ce, wind)
            sag = v_oc - V_t
            print(f"  {ctrl:^8} | {wind:5.0f} | {ce:7.3f} | {I:6.2f} A | "
                  f"{v_oc:7.2f} | {V_t:10.2f} | {sag:5.2f} V")

    print(f"\n{SEP}")
    print("  Battery module demonstration complete")
    print(SEP)


if __name__ == "__main__":
    run_demo()
