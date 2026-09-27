"""Shared constants, the normalised bound model and deterministic number formatting.

This module has no Pacti dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# --------------------------------------------------------------------------
# Input files (repo-relative, POSIX separators)
# --------------------------------------------------------------------------
FRAMEWORK_PATH = "src/contracts/contract_framework.py"
PACTI_LIB_PATH = "src/planning/pacti_contracts.py"
HORIZON_PLANNER_PATH = "src/planning/horizon_planner.py"
INTEGRATED_PLANNER_PATH = "src/planning/integrated_planner.py"
THIRD_COPY_PATHS = (HORIZON_PLANNER_PATH, INTEGRATED_PLANNER_PATH)
INPUT_PATHS = tuple(sorted((FRAMEWORK_PATH, PACTI_LIB_PATH) + THIRD_COPY_PATHS))

# Short names used in the Markdown report (the CSV keeps full paths).
SHORT_PATH = {
    FRAMEWORK_PATH: "cf",
    PACTI_LIB_PATH: "pc",
    HORIZON_PLANNER_PATH: "hp",
    INTEGRATED_PLANNER_PATH: "ip",
}

SOURCES = ("framework", "pacti_library")
SOURCE_LABEL = {"framework": "framework", "pacti_library": "Pacti library"}

# Logical components. The framework has one sensor contract for GPS and IMU;
# its rows are split by variable (SENSOR_VAR_COMPONENT).
COMPONENT_ORDER = ("GPS", "IMU", "EKF", "PID", "MPC", "HINF", "ACTUATOR", "DYNAMICS")
CONTROLLERS = ("pid", "mpc", "hinf")
CONTROLLER_COMPONENT = {"pid": "PID", "mpc": "MPC", "hinf": "HINF"}

FRAMEWORK_SENSOR_CONTRACT = "GPS_IMU_Sensors"
FRAMEWORK_CONTRACT_COMPONENT = {
    "EKF_Estimator": "EKF",
    "PID_Controller": "PID",
    "MPC_Controller": "MPC",
    "Hinf_Controller": "HINF",
    "Actuators": "ACTUATOR",
}
SENSOR_VAR_COMPONENT = {
    "gps_satellites": "GPS",
    "gps_hdop": "GPS",
    "position_meas_error": "GPS",
    "velocity_meas_error": "GPS",
    "imu_temperature": "IMU",
    "imu_calibrated": "IMU",
    "rate_meas_error": "IMU",
    "attitude_meas_error": "IMU",
}
# controller id -> key in HierarchicalContractMonitor.controller_contracts
FRAMEWORK_CONTROLLER_KEY = {"pid": "PID", "mpc": "MPC", "hinf": "Hinf"}

PACTI_CONTRACT_COMPONENT = {
    "gps": "GPS",
    "imu": "IMU",
    "ekf": "EKF",
    "pid": "PID",
    "mpc": "MPC",
    "hinf": "HINF",
    "actuator": "ACTUATOR",
    "dynamics": "DYNAMICS",
}

# Comparison-layer renaming. The repo has no renaming layer between the two
# libraries (F-A1-09 "conflicts"), so the tool needs one to line rows up.
# (component or "*", Pacti-library name, framework name, reason)
ALIASES = (
    ("*", "position_est_error", "position_error", "EKF output and controller input in both libraries"),
    ("*", "velocity_est_error", "velocity_error", "EKF output and controller input in both libraries"),
    ("*", "attitude_est_error", "attitude_error", "EKF output in both libraries"),
    ("*", "motor_temp", "motor_temperature", "actuator input in both libraries"),
    ("HINF", "settling_time", "stabilization_time",
     "ASSUMPTION carried over from A1's F-A1-09 table (row 50, 'H-inf time bound'), not established by "
     "the code: no code in the repo reads either variable, so their equivalence is unverified"),
)
# Deliberately NOT aliased: the quantities differ.
NOT_ALIASED = (
    ("IMU", "imu_temperature", "imu_temp_deviation",
     "absolute IMU temperature in degC (framework) vs deviation from the calibration temperature (Pacti library)"),
)


def canonical_var(component: str, source: str, name: str) -> str:
    """Map a native variable name to the name used in reconciliation rows (framework naming)."""
    if source != "pacti_library":
        return name
    for comp, pacti_name, fw_name, _ in ALIASES:
        if pacti_name == name and comp in ("*", component):
            return fw_name
    return name


# --------------------------------------------------------------------------
# Normalised bound
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Bound:
    """One linear bound on a single subject variable.

    kind "A": ``var >= constant`` (lower) or ``var <= constant`` (upper), no coefficients.
    kind "G": ``var <= sum(c*x) + constant`` (upper) or ``var >= ...`` (lower).
    """

    source: str
    component: str
    contract: str
    kind: str
    var: str
    direction: str
    coefficients: Tuple[Tuple[str, float], ...]
    constant: float
    loc: str
    via: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()

    def coef_dict(self):
        return dict(self.coefficients)

    def via_for(self, item: str) -> Tuple[str, ...]:
        for key, names in self.via:
            if key == item:
                return names
        return ()

    def label(self) -> str:
        op = "<=" if self.direction == "upper" else ">="
        if self.kind == "A":
            return f"{self.component}.A {self.var}{op}{fmt(self.constant)}"
        return f"{self.component}.G {self.var} {op} {linear_text(self.coefficients, self.constant)}"


# --------------------------------------------------------------------------
# Deterministic numbers
# --------------------------------------------------------------------------
SIG_DIGITS = 10


def rnd(x: Optional[float]) -> Optional[float]:
    """Round to SIG_DIGITS significant digits; None for None/NaN/inf; no negative zero."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    v = float(f"{x:.{SIG_DIGITS}g}")
    return 0.0 if v == 0 else v


def fmt(x: Optional[float]) -> str:
    v = rnd(x)
    if v is None:
        return ""
    return f"{v:.{SIG_DIGITS}g}"


def fmt_interval(lo: Optional[float], hi: Optional[float]) -> str:
    los = fmt(lo) if rnd(lo) is not None else "-inf"
    his = fmt(hi) if rnd(hi) is not None else "+inf"
    return f"[{los}, {his}]"


def close(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def linear_text(coefficients, constant: float) -> str:
    """Render ``sum(c*x) + k`` deterministically (coefficients sorted by name)."""
    parts = []
    for name, c in sorted(coefficients):
        if rnd(c) == 0:
            continue
        mag = fmt(abs(c))
        term = name if mag == "1" else f"{mag}*{name}"
        if not parts:
            parts.append(term if c > 0 else f"-{term}")
        else:
            parts.append(("+ " if c > 0 else "- ") + term)
    k = rnd(constant) or 0.0
    if not parts:
        return fmt(k) if k != 0 else "0"
    if k > 0:
        parts.append(f"+ {fmt(k)}")
    elif k < 0:
        parts.append(f"- {fmt(-k)}")
    return " ".join(parts)


def term_text(coefficients, constant: float) -> str:
    """Render a Pacti term ``sum(a*v) <= k``; single-variable terms render as bounds."""
    coefs = [(n, c) for n, c in sorted(coefficients) if rnd(c) != 0]
    if len(coefs) == 1:
        name, a = coefs[0]
        return f"{name} <= {fmt(constant / a)}" if a > 0 else f"{name} >= {fmt(constant / a)}"
    lhs = linear_text(coefs, 0.0)
    return f"{lhs} <= {fmt(constant)}"
