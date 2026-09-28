"""
Battery Model for UAV Control System

Physics-based 3S LiPo model that exposes:
  - Power draw as a function of control_effort and wind_speed
  - State-of-charge / terminal voltage tracking
  - Per-controller energy cost profiles (for planner optimization)

Power model (actuator disk theory):
    P_thrust = P_hover * (control_effort / hover_thrust) ^ 1.5
    P_wind   = k_wind * wind_speed ^ 2
    P_total  = P_thrust + P_wind

Discharge:
    I = P_total / V_terminal
    SoC_new = SoC - I * dt / (3600 * capacity_Ah)

Terminal voltage (one-shot sag correction):
    V_terminal = V_oc(SoC) - I_estimated * R_internal
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Controller effort model — derived directly from contract_framework.py
# control_effort <= wind_coeff * wind_speed + baseline  (upper bound)
# control_effort >= effort_min                          (lower bound)
# ---------------------------------------------------------------------------
CONTROLLER_EFFORT_PARAMS: Dict[str, Dict[str, float]] = {
    'PID':  {'wind_coeff': 0.15, 'baseline': 0.30, 'effort_min': 0.10},
    'Hinf': {'wind_coeff': 0.03, 'baseline': 0.40, 'effort_min': 0.20},
    'MPC':  {'wind_coeff': 0.08, 'baseline': 0.20, 'effort_min': 0.10},
    # Lowercase aliases used by horizon_planner
    'pid':  {'wind_coeff': 0.15, 'baseline': 0.30, 'effort_min': 0.10},
    'hinf': {'wind_coeff': 0.03, 'baseline': 0.40, 'effort_min': 0.20},
    'mpc':  {'wind_coeff': 0.08, 'baseline': 0.20, 'effort_min': 0.10},
}

# OCV (open-circuit voltage) vs SoC for a 3S LiPo (piecewise linear)
# SoC: 0.0 (empty, 10.5 V) → 1.0 (full, 12.6 V)
_OCV_SOC  = np.array([0.00, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.00])
_OCV_VOLT = np.array([10.50, 10.70, 10.90, 11.20, 11.50, 12.00, 12.30, 12.60])


@dataclass
class BatteryConfig:
    """Physical parameters for a 3S LiPo pack."""
    capacity_mah: float = 2200.0        # mAh
    nominal_voltage: float = 11.1       # V (3 × 3.7)
    max_voltage: float = 12.6           # V (3 × 4.2, fully charged)
    min_voltage: float = 10.5           # V (3 × 3.5, safe cutoff)
    internal_resistance: float = 0.10   # Ω (typical 3S 2200mAh pack)

    # Power model
    hover_power_w: float = 111.0        # W at hover thrust (11.1V × 10A)
    hover_thrust: float = 0.50          # control_effort at hover (from memory)
    wind_power_coeff: float = 0.05      # W / (m/s)², aerodynamic drag penalty

    # Alert thresholds
    critical_soc: float = 0.20          # RTL trigger level
    depleted_soc: float = 0.05          # Emergency land level


class BatteryModel:
    """
    Physics-based LiPo battery model.

    Tracks SoC via current integration and models terminal voltage
    including internal-resistance sag under load.
    """

    def __init__(self,
                 config: Optional[BatteryConfig] = None,
                 initial_soc: float = 1.0):
        self.config = config or BatteryConfig()
        self.soc = float(np.clip(initial_soc, 0.0, 1.0))
        self._capacity_ah = self.config.capacity_mah / 1000.0

        # Running totals for telemetry
        self.energy_consumed_wh: float = 0.0
        self.charge_consumed_mah: float = 0.0
        self.time_elapsed_s: float = 0.0

        logger.info(f"BatteryModel init: SoC={self.soc:.0%}, "
                    f"capacity={self.config.capacity_mah:.0f}mAh, "
                    f"V_oc={self.ocv(self.soc):.2f}V")

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    def ocv(self, soc: float) -> float:
        """Open-circuit voltage at given SoC (piecewise linear interpolation)."""
        return float(np.interp(soc, _OCV_SOC, _OCV_VOLT))

    def power_draw(self, control_effort: float, wind_speed: float) -> float:
        """
        Instantaneous power draw in Watts.

        Actuator disk theory: P_thrust ∝ T^1.5
            P_thrust = P_hover * (ce / hover_thrust)^1.5

        Aerodynamic drag adds a wind-speed-squared penalty.
        """
        ce = max(0.0, control_effort)
        thrust_ratio = ce / self.config.hover_thrust
        P_thrust = self.config.hover_power_w * (thrust_ratio ** 1.5)
        P_wind = self.config.wind_power_coeff * (wind_speed ** 2)
        return P_thrust + P_wind

    def current_draw(self, control_effort: float, wind_speed: float) -> float:
        """
        Current draw in Amps, accounting for terminal voltage sag.

        Uses one-shot approximation:
            I_0 = P / V_oc
            V_t = V_oc - I_0 * R_int
            I   = P / V_t
        """
        P = self.power_draw(control_effort, wind_speed)
        v_oc = self.ocv(self.soc)
        if v_oc <= 0:
            return 0.0
        I0 = P / v_oc
        v_t = v_oc - I0 * self.config.internal_resistance
        v_t = max(v_t, 0.1)  # prevent divide-by-zero
        return P / v_t

    @property
    def terminal_voltage(self) -> float:
        """
        Terminal voltage at current operating point (no active load assumed).
        For in-flight terminal voltage, use terminal_voltage_under_load().
        """
        return self.ocv(self.soc)

    def terminal_voltage_under_load(self,
                                    control_effort: float,
                                    wind_speed: float) -> float:
        """Terminal voltage V_t = V_oc - I * R_internal under active load."""
        I = self.current_draw(control_effort, wind_speed)
        return self.ocv(self.soc) - I * self.config.internal_resistance

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(self,
               control_effort: float,
               wind_speed: float,
               dt: float) -> Dict[str, float]:
        """
        Advance battery state by dt seconds.

        Returns a telemetry dict with the step's power/current/SoC.
        """
        I = self.current_draw(control_effort, wind_speed)
        P = self.power_draw(control_effort, wind_speed)

        delta_ah = I * dt / 3600.0
        self.soc = float(np.clip(self.soc - delta_ah / self._capacity_ah, 0.0, 1.0))

        self.energy_consumed_wh += P * dt / 3600.0
        self.charge_consumed_mah += delta_ah * 1000.0
        self.time_elapsed_s += dt

        return {
            'power_w': P,
            'current_a': I,
            'soc': self.soc,
            'terminal_voltage': self.terminal_voltage_under_load(control_effort, wind_speed),
        }

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def remaining_capacity_mah(self) -> float:
        return self.soc * self.config.capacity_mah

    @property
    def remaining_capacity_fraction(self) -> float:
        return self.soc

    def flight_time_remaining(self,
                              control_effort: float,
                              wind_speed: float) -> float:
        """
        Estimated remaining flight time in seconds at constant operating point.

        Returns inf if current draw is zero.
        """
        I = self.current_draw(control_effort, wind_speed)
        if I <= 0:
            return float('inf')
        remaining_ah = self.soc * self._capacity_ah
        return (remaining_ah / I) * 3600.0

    def is_critical(self) -> bool:
        """SoC below RTL threshold or OCV below safety floor."""
        return (self.soc < self.config.critical_soc or
                self.ocv(self.soc) < 10.8)

    def is_depleted(self) -> bool:
        """SoC below emergency-land threshold or OCV at cutoff."""
        return (self.soc < self.config.depleted_soc or
                self.ocv(self.soc) <= self.config.min_voltage)

    def get_status(self) -> Dict[str, float]:
        """Full telemetry snapshot."""
        return {
            'soc': self.soc,
            'soc_pct': self.soc * 100.0,
            'ocv_v': self.ocv(self.soc),
            'remaining_mah': self.remaining_capacity_mah,
            'consumed_mah': self.charge_consumed_mah,
            'energy_consumed_wh': self.energy_consumed_wh,
            'time_elapsed_s': self.time_elapsed_s,
            'is_critical': float(self.is_critical()),
            'is_depleted': float(self.is_depleted()),
        }


# ---------------------------------------------------------------------------
# Per-controller power cost — used by the planner for optimization
# ---------------------------------------------------------------------------

class ControllerPowerProfile:
    """
    Maps (controller, wind_speed) → expected power draw and energy cost
    over a planning horizon.

    Control effort estimates come from contract_framework.py upper bounds:
        effort_upper(wind) = wind_coeff * wind + baseline

    Used by the horizon planner to rank controllers by energy cost when
    operating in battery-optimal mode.
    """

    def __init__(self, battery: BatteryModel):
        self.battery = battery

    def effort_estimate(self, controller: str, wind_speed: float) -> float:
        """
        Upper-bound control effort for a controller at given wind speed.
        Derived from contract guarantees in contract_framework.py.
        """
        params = CONTROLLER_EFFORT_PARAMS.get(controller)
        if params is None:
            logger.warning(f"Unknown controller '{controller}', using Hinf params")
            params = CONTROLLER_EFFORT_PARAMS['hinf']
        effort = params['wind_coeff'] * wind_speed + params['baseline']
        return float(np.clip(effort, params['effort_min'], 1.0))

    def power_estimate(self, controller: str, wind_speed: float) -> float:
        """Expected power draw in Watts at the contract's upper-bound effort."""
        ce = self.effort_estimate(controller, wind_speed)
        return self.battery.power_draw(ce, wind_speed)

    def energy_over_horizon(self,
                            controller: str,
                            wind_speed: float,
                            horizon_steps: int,
                            dt: float) -> float:
        """
        Upper-bound energy cost in Wh over a planning horizon.

        Assumes constant wind and nominal effort throughout.
        """
        P = self.power_estimate(controller, wind_speed)
        return P * horizon_steps * dt / 3600.0

    def flight_time_at_effort(self, controller: str, wind_speed: float) -> float:
        """Remaining flight time in seconds if this controller runs continuously."""
        ce = self.effort_estimate(controller, wind_speed)
        return self.battery.flight_time_remaining(ce, wind_speed)

    def rank_by_energy(self,
                       controllers: list,
                       wind_speed: float,
                       horizon_steps: int,
                       dt: float) -> list:
        """
        Sort controllers by ascending energy cost over the horizon.
        Returns list of (controller_name, energy_wh) tuples.
        """
        costs = [
            (ctrl, self.energy_over_horizon(ctrl, wind_speed, horizon_steps, dt))
            for ctrl in controllers
        ]
        return sorted(costs, key=lambda x: x[1])

    def rank_by_time(self, controllers: list, wind_speed: float) -> list:
        """
        Sort controllers by ascending settling time (min-time objective).
        Settling time upper bounds from contract_framework.py:
            PID:  5s, MPC: 7s, H-inf: 10s
        """
        # From controller contract settling_time upper bounds
        settling_upper = {'pid': 5.0, 'PID': 5.0,
                          'mpc': 7.0, 'MPC': 7.0,
                          'hinf': 10.0, 'Hinf': 10.0}
        costs = [
            (ctrl, settling_upper.get(ctrl, 10.0))
            for ctrl in controllers
        ]
        return sorted(costs, key=lambda x: x[1])
