"""
Hierarchical Contract Framework for Adaptive Drone Control
Uses relational Assume-Guarantee contracts with linear constraints

Key Concepts:
- Every component has a contract (assumptions -> guarantees)
- Guarantees are RELATIONAL: output bounds depend on input values via equations
- Contracts compose algebraically through the pipeline
- Runtime monitoring evaluates predicted bounds
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from pacti.contracts import PolyhedralIoContract
    PACTI_AVAILABLE = True
except ImportError:
    PACTI_AVAILABLE = False
    print("WARNING: Pacti not available. Using simplified contracts.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContractStatus(Enum):
    """Status of contract checking"""
    SATISFIED = "satisfied"
    ASSUMPTIONS_VIOLATED = "assumptions_violated"
    GUARANTEES_VIOLATED = "guarantees_violated"
    BOTH_VIOLATED = "both_violated"
    UNKNOWN = "unknown"


@dataclass
class ContractMetrics:
    """Metrics for evaluating contract satisfaction"""
    timestamp: float
    component_name: str
    assumptions_met: bool
    guarantees_met: bool
    assumption_values: Dict[str, float] = field(default_factory=dict)
    guarantee_values: Dict[str, float] = field(default_factory=dict)
    predicted_bounds: Dict[str, float] = field(default_factory=dict)
    margin: Optional[float] = None  # How close to violation (0 = at boundary, >0 = safe)


@dataclass
class LinearConstraint:
    """
    A linear relational constraint of the form:
        output_var <= coeff_1 * var_1 + coeff_2 * var_2 + ... + constant   (upper bound)
        output_var >= coeff_1 * var_1 + coeff_2 * var_2 + ... + constant   (lower bound)

    This is the building block for relational contracts. Unlike range-based
    guarantees, the bound on the output depends on the actual input values.
    """
    output_var: str
    coefficients: Dict[str, float]   # {input_var_name: multiplier}
    constant: float
    is_upper_bound: bool = True      # True for <=, False for >=

    def evaluate_bound(self, values: Dict[str, float]) -> float:
        """Compute the bound given current input values."""
        bound = self.constant
        for var, coeff in self.coefficients.items():
            if var in values:
                bound += coeff * values[var]
        return bound

    def check(self, values: Dict[str, float]) -> Tuple[bool, float]:
        """
        Check if constraint is satisfied given values that include the output variable.
        Returns: (satisfied, margin) where margin > 0 means safe.
        """
        if self.output_var not in values:
            return True, float('inf')  # Can't disprove without actual value

        actual = values[self.output_var]
        bound = self.evaluate_bound(values)

        if self.is_upper_bound:
            margin = bound - actual   # positive = actual is below bound = safe
            return actual <= bound + 1e-9, margin
        else:
            margin = actual - bound   # positive = actual is above bound = safe
            return actual >= bound - 1e-9, margin

    def substitute(self, upstream_map: Dict[str, Dict[str, 'LinearConstraint']]) -> 'LinearConstraint':
        """
        Algebraic substitution for contract composition.

        If this constraint uses variable Y, and upstream guarantees Y <= a*X + b,
        substitute to express the constraint in terms of X.

        upstream_map = {
            'upper': {var_name: LinearConstraint, ...},  # upper bound constraints
            'lower': {var_name: LinearConstraint, ...},  # lower bound constraints
        }

        Soundness: For an upper bound with positive coefficient on Y, we use
        the upper bound of Y (worst case). For negative coefficient, we use
        the lower bound of Y (worst case going the other direction).
        """
        new_coefficients = {}
        new_constant = self.constant

        for var, coeff in self.coefficients.items():
            # Determine which upstream bound gives the sound worst-case
            if self.is_upper_bound:
                # Upper bound guarantee: positive coeff -> use upper bound of var
                #                        negative coeff -> use lower bound of var
                upstream = (upstream_map['upper'].get(var) if coeff >= 0
                            else upstream_map['lower'].get(var))
            else:
                # Lower bound guarantee: positive coeff -> use lower bound of var
                #                        negative coeff -> use upper bound of var
                upstream = (upstream_map['lower'].get(var) if coeff >= 0
                            else upstream_map['upper'].get(var))

            if upstream is not None:
                # Substitute: var's bound expression replaces var
                for uvar, ucoeff in upstream.coefficients.items():
                    new_coefficients[uvar] = new_coefficients.get(uvar, 0.0) + coeff * ucoeff
                new_constant += coeff * upstream.constant
            else:
                # Variable not from upstream (environment input) - keep as-is
                new_coefficients[var] = new_coefficients.get(var, 0.0) + coeff

        return LinearConstraint(
            output_var=self.output_var,
            coefficients=new_coefficients,
            constant=new_constant,
            is_upper_bound=self.is_upper_bound
        )

    def __repr__(self):
        terms = []
        for var, coeff in sorted(self.coefficients.items()):
            if coeff == 1.0:
                terms.append(var)
            elif coeff == -1.0:
                terms.append(f"-{var}")
            else:
                terms.append(f"{coeff:g}*{var}")
        if self.constant != 0.0 or not terms:
            terms.append(f"{self.constant:g}")
        expr = " + ".join(terms)
        op = "<=" if self.is_upper_bound else ">="
        return f"{self.output_var} {op} {expr}"


class SimpleContract:
    """
    Contract with relational guarantees.

    Assumptions: Range-based input bounds {var: (min, max)}
    Guarantees: Linear constraints relating outputs to inputs

    Example:
        PID contract assumes wind_speed in [0, 3] and guarantees
        tracking_error <= 1.0 * position_error + 0.3 * wind_speed + 0.2

    When contracts compose, the equations propagate algebraically through
    the pipeline, producing end-to-end performance predictions.
    """
    def __init__(self,
                 name: str,
                 assumptions: Dict[str, Tuple[float, float]],
                 guarantees: List[LinearConstraint]):
        self.name = name
        self.assumptions = assumptions
        self.guarantees = guarantees

    def check_assumptions(self, values: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """Check if assumptions are satisfied (range-based)."""
        margins = {}
        all_met = True

        for var, (min_val, max_val) in self.assumptions.items():
            if var in values:
                val = values[var]
                # Compute margin (positive = safe, negative = violated)
                margin_min = val - min_val
                margin_max = max_val - val
                margins[var] = min(margin_min, margin_max)

                if val < min_val or val > max_val:
                    all_met = False
            else:
                all_met = False
                margins[var] = -float('inf')

        return all_met, margins

    def check_guarantees(self, values: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """
        Check if guarantees are satisfied using relational equations.

        For each constraint, if the output variable is in the values dict,
        compute the relational bound and check the actual value against it.
        If the output variable is not available, the guarantee is vacuously
        met (cannot be disproved without measurement).
        """
        margins = {}
        all_met = True
        has_any_check = False

        for constraint in self.guarantees:
            output_var = constraint.output_var

            if output_var not in values:
                continue  # No actual value to check

            has_any_check = True
            satisfied, margin = constraint.check(values)

            if not satisfied:
                all_met = False

            # Keep worst margin per output variable
            if output_var in margins:
                margins[output_var] = min(margins[output_var], margin)
            else:
                margins[output_var] = margin

        return all_met, margins

    def predict_guarantees(self, values: Dict[str, float]) -> Dict[str, float]:
        """
        Compute predicted guarantee bounds from input values only.
        Does not require actual output values - uses the relational equations
        to predict what the outputs will be bounded by.

        Returns dict like {'tracking_error_max': 2.43, 'settling_time_max': 3.64, ...}
        """
        predictions = {}

        for constraint in self.guarantees:
            # Check all input variables are available
            can_evaluate = all(v in values for v in constraint.coefficients) or len(constraint.coefficients) == 0
            if not can_evaluate:
                continue

            bound = constraint.evaluate_bound(values)

            if constraint.is_upper_bound:
                key = f"{constraint.output_var}_max"
                # Keep tightest upper bound
                if key not in predictions or bound < predictions[key]:
                    predictions[key] = bound
            else:
                key = f"{constraint.output_var}_min"
                # Keep tightest lower bound
                if key not in predictions or bound > predictions[key]:
                    predictions[key] = bound

        return predictions

    def get_output_vars(self) -> set:
        """Return set of all output variable names in guarantees."""
        return {c.output_var for c in self.guarantees}

    def compose(self, other: 'SimpleContract') -> 'SimpleContract':
        """
        Compose two contracts: self >> other (self feeds into other).

        Algebraic composition:
        - If self guarantees y <= a*x + b  and  other guarantees z <= c*y + d
          then the composed contract guarantees z <= c*a*x + c*b + d

        Assumptions: self.assumptions + (other.assumptions not covered by self outputs)
        Guarantees: other.guarantees with upstream bounds substituted
        """
        # Build lookup of upstream guarantee constraints by output variable
        upper_bounds = {}
        lower_bounds = {}
        output_vars = set()

        for constraint in self.guarantees:
            output_vars.add(constraint.output_var)
            if constraint.is_upper_bound:
                # Keep tightest upper bound per variable
                if constraint.output_var not in upper_bounds:
                    upper_bounds[constraint.output_var] = constraint
            else:
                # Keep tightest lower bound per variable
                if constraint.output_var not in lower_bounds:
                    lower_bounds[constraint.output_var] = constraint

        upstream_map = {'upper': upper_bounds, 'lower': lower_bounds}

        # New assumptions = self.assumptions + downstream assumptions not covered by self outputs
        new_assumptions = self.assumptions.copy()
        for var, bounds in other.assumptions.items():
            if var not in output_vars:
                new_assumptions[var] = bounds

        # New guarantees = other.guarantees with upstream bounds substituted
        # PLUS upstream guarantees whose outputs are NOT consumed by downstream
        new_guarantees = []

        # 1. Pass through upstream guarantees not consumed by downstream
        downstream_assumed = set(other.assumptions.keys())
        for constraint in self.guarantees:
            if constraint.output_var not in downstream_assumed:
                new_guarantees.append(constraint)

        # 2. Downstream guarantees with upstream bounds substituted
        for constraint in other.guarantees:
            new_constraint = constraint.substitute(upstream_map)
            new_guarantees.append(new_constraint)

        composed = SimpleContract(
            name=f"{self.name}>>{other.name}",
            assumptions=new_assumptions,
            guarantees=new_guarantees
        )

        return composed

    def __repr__(self):
        lines = [f"Contract '{self.name}':"]
        lines.append(f"  Assumptions ({len(self.assumptions)}):")
        for var, (lo, hi) in sorted(self.assumptions.items()):
            lines.append(f"    {var} in [{lo:g}, {hi:g}]")
        lines.append(f"  Guarantees ({len(self.guarantees)}):")
        for g in self.guarantees:
            lines.append(f"    {g}")
        return "\n".join(lines)


class HierarchicalContractMonitor:
    """
    Monitors hierarchical contract composition for the entire system

    Pipeline: Sensors -> Estimator -> Controller -> Actuators -> Dynamics
    Each component has contracts, and we compose them for system-level guarantees
    """

    def __init__(self, use_pacti: bool = False):
        self.use_pacti = use_pacti and PACTI_AVAILABLE

        # Component contracts
        self.sensor_contract = None
        self.estimator_contract = None
        self.controller_contracts = {}  # Multiple controllers
        self.actuator_contract = None

        # Composed contracts
        self.pipeline_contracts = {}  # For each controller
        self.current_pipeline = None

        # Monitoring history
        self.history: List[ContractMetrics] = []

        logger.info(f"Contract Monitor initialized (Pacti: {self.use_pacti})")

    def define_contracts(self):
        """
        Define all component contracts with domain-verifiable equations.

        Each contract is grounded in the component's theory:
        - GPS: σ_pos = HDOP × UERE  (standard positioning equation)
        - IMU: σ_rate = ND × √BW + TC × T  (IEEE 952 / Allan variance)
        - EKF: P_ss from DARE  (Kalman filter steady-state covariance)
        - PID: t_s = 4/(ζωn), e_ss = d/K  (classical control theory)
        - H-inf: ‖T_zw‖∞ ≤ γ  (robust control H∞ norm bound)
        - MPC: V(k+1) ≤ V(k) - l(k)  (Lyapunov decrease / recursive feasibility)
        - Actuator: T = k_f × ω², T_max ∝ V²  (motor dynamics)

        Named parameters are traceable to hardware specs or design choices.
        A domain specialist can verify each contract against its theory.
        """

        # =============================================================
        # PHYSICAL CONSTANTS — traceable to hardware specs and theory
        # =============================================================

        # GPS: Standard Positioning Service Performance Standard
        UERE = 1.5              # m, User Equivalent Range Error (SBAS-augmented)
        SIGMA_DOPPLER = 0.1     # m/s, carrier-phase Doppler velocity accuracy

        # IMU: IEEE 952 / Allan variance (MPU-6050 class consumer MEMS)
        # σ_rate = ND_gyro × √BW,  bias_drift = TC × ΔT
        # Linear model: rate_meas_error ≤ A × temperature + B
        # At T=5°C: ~0.01 rad/s,  at T=25°C: ~0.026 rad/s,  at T=55°C: ~0.05 rad/s
        IMU_RATE_TC = 0.0008    # rad/s per °C (temperature coefficient)
        IMU_RATE_BASE = 0.006   # rad/s at T=0°C intercept
        IMU_ATT_TC = IMU_RATE_TC * 0.02   # attitude = rate × dt (integrated per step)
        IMU_ATT_BASE = IMU_RATE_BASE * 0.02

        # EKF: Derived from DARE with actual filter matrices
        # Q = diag(0.01, 0.1, 0.01, 0.1), R = diag(1.0, 0.1, 0.01, 0.05)
        # Contraction ratio α = √(P_ss / R), noise floor β = √(Q contribution)
        ALPHA_POS = 0.7         # position contraction (filtering reduces error)
        BETA_POS = 0.1          # m, process noise floor
        ALPHA_VEL = 0.5         # velocity contraction (better due to integration)
        BETA_VEL = 0.05         # m/s, velocity noise floor
        ALPHA_ATT = 0.3         # attitude contraction (strong, low R_att)
        BETA_ATT = 0.01         # rad, attitude noise floor
        # NEES 95% bound: χ²_inv(0.975, 12) / 12 = 1.95
        NEES_BOUND = 1.95
        NIS_BOUND = 1.95

        # PID: Classical control theory
        # Characteristic: s² + kd×s + kp = 0 with kp=0.2, kd=0.5
        # ωn = √0.2 ≈ 0.447 rad/s,  ζ = 0.5/(2×0.447) ≈ 0.559
        # t_s = 4/(ζ×ωn) ≈ 16s,  Mp = exp(-πζ/√(1-ζ²)) ≈ 0.12
        OMEGA_N = 0.447         # rad/s, natural frequency = √(kp)
        ZETA = 0.559            # damping ratio = kd/(2ωn)
        T_SETTLE_BASE = 4.0 / (ZETA * OMEGA_N)  # ≈ 16.0s (2% settling)
        KP_EFF = 0.2            # effective position loop gain (proportional only)
        TILT_LIMIT = 0.15       # rad, max tilt (hardware saturation)
        E_FLOOR_PID = 0.2       # m, irreducible error (discretization + delay)
        # Wind sensitivity for PID WITH integral action.
        # Pure-proportional bound: e_wind = wind / kp_pos ≈ wind / 0.35 ≈ 2.86×wind.
        # Integral action reduces steady-state wind error toward zero; the transient
        # bound depends on the integral time constant T_i ≈ kp/ki.
        # Simulation: wind coupling = 0.06 m/s² per m/s; kp_pos = 0.35.
        # With integral: e_wind_ss → 0; transient peak ≈ (0.06/0.35)×wind ≈ 0.17×wind.
        # Conservative estimate with settling transient included: γ ≈ 0.5×wind.
        GAMMA_WIND_PID = 0.5    # wind → tracking error bound (PID+integral, sim-matched)

        # H-infinity: Robust control ‖T_zw‖∞ ≤ γ
        # γ = worst-case disturbance amplification factor
        GAMMA_WIND_HINF = 0.5   # wind → tracking error amplification
        E_BASE_HINF = 1.0       # m, base error (stabilization only, no position tracking)
        K_RATE_DAMP = 0.8       # rate damping gain
        K_ATT_GAIN = 5.0        # attitude correction gain

        # MPC: Lyapunov stability / optimal cost bound
        # tracking_error ≤ √(V*/λ_min(Q)) ≈ α × est_error + β × wind + floor
        ALPHA_MPC = 0.3         # optimization contracts error (α < 1)
        BETA_MPC = 0.15         # wind sensitivity (smaller than PID: prediction helps)
        E_FLOOR_MPC = 0.1       # m, irreducible floor
        COMPUTE_TIME_MAX = 0.05 # s, maximum solve time for real-time constraint

        # Actuator: Motor dynamics T = k_f × ω², T_max ∝ V_bat²
        TAU_MOTOR = 0.015       # s, mechanical time constant
        K_V_THRUST = 0.03       # thrust accuracy coefficient per volt
        K_T_DERATE = 0.0005     # thermal derating coefficient per °C

        # =====================================================================
        # GPS SENSOR CONTRACT
        #
        # Theory: σ_position = HDOP × UERE  (GPS positioning equation)
        # Ref: GPS Standard Positioning Service Performance Standard
        #
        # Specialist verification:
        #   - Check UERE against receiver datasheet (1.5m typical for SBAS)
        #   - Check SIGMA_DOPPLER against Doppler measurement spec (~0.1 m/s)
        #   - Verify HDOP range covers operational satellite geometry
        #
        # Outputs flowing to EKF: position_meas_error, velocity_meas_error
        # Internal outputs: (none currently — all flow downstream)
        # =====================================================================
        self.sensor_contract = SimpleContract(
            name="GPS_IMU_Sensors",
            assumptions={
                "gps_satellites": (6.0, 30.0),    # ≥6 for good geometry
                "gps_hdop": (0.5, 5.0),            # 0.5=ideal, 5.0=poor
                "imu_temperature": (5.0, 55.0),    # °C, operational range
                "imu_calibrated": (1.0, 1.0),      # must be calibrated
            },
            guarantees=[
                # --- GPS position accuracy ---
                # Theory: σ_pos = HDOP × UERE
                # At HDOP=0.8: ≤ 1.2m, at HDOP=2.0: ≤ 3.0m, at HDOP=5.0: ≤ 7.5m
                LinearConstraint("position_meas_error",
                                 {"gps_hdop": UERE}, 0.0),
                LinearConstraint("position_meas_error", {}, 0.0, is_upper_bound=False),

                # --- GPS velocity accuracy ---
                # Theory: carrier-phase Doppler measurement, ~0.1 m/s 1-sigma
                LinearConstraint("velocity_meas_error", {}, SIGMA_DOPPLER),
                LinearConstraint("velocity_meas_error", {}, 0.0, is_upper_bound=False),

                # --- IMU rate measurement error ---
                # Theory: σ_rate = ND_gyro × √BW + TC_gyro × T  (IEEE 952)
                # Linear model: increases with temperature
                # At 25°C: ≤ 0.026 rad/s, at 55°C: ≤ 0.050 rad/s
                LinearConstraint("rate_meas_error",
                                 {"imu_temperature": IMU_RATE_TC}, IMU_RATE_BASE),
                LinearConstraint("rate_meas_error", {}, 0.0, is_upper_bound=False),

                # --- IMU attitude measurement error ---
                # Theory: σ_att ≈ σ_rate × dt (integrated gyro noise per step)
                # At 25°C: ≤ 0.00052 rad, at 55°C: ≤ 0.001 rad
                LinearConstraint("attitude_meas_error",
                                 {"imu_temperature": IMU_ATT_TC}, IMU_ATT_BASE),
                LinearConstraint("attitude_meas_error", {}, 0.0, is_upper_bound=False),
            ]
        )

        # =====================================================================
        # EKF ESTIMATOR CONTRACT
        #
        # Theory: Steady-state covariance from DARE
        #   P = F P F' + Q - F P H'(H P H' + R)^{-1} H P F'
        #   Contraction: est_error ≤ α × meas_error + β  where α = √(P_ss/R)
        #
        # The filter ALWAYS reduces error (α < 1) when observable (KF theory).
        # The floor β comes from process noise (irreducible even with perfect sensors).
        #
        # Specialist verification:
        #   1. Solve DARE with Q=diag(0.01,0.1,0.01,0.1), R=diag(1.0,0.1,0.01,0.05)
        #   2. Check √(P_ss[i,i]) ≤ α_i × √(R[i,i]) + β_i
        #   3. Run Monte Carlo → NEES within χ²(12) bounds 95% of time
        #
        # Outputs flowing to Controller: position_error, velocity_error
        # Internal: attitude_error, nees_bound, nis_bound (verification only)
        # =====================================================================
        self.estimator_contract = SimpleContract(
            name="EKF_Estimator",
            assumptions={
                "position_meas_error": (0.0, UERE * 5.0),   # max HDOP=5 × UERE
                "velocity_meas_error": (0.0, 0.5),           # GPS Doppler bound
                "rate_meas_error": (0.0, 0.1),               # IMU rate noise bound
                "attitude_meas_error": (0.0, 0.01),          # IMU attitude noise bound
            },
            guarantees=[
                # --- Position estimation error ---
                # Theory: P_ss_pos from DARE, α = √(P_ss/R) ≈ 0.7
                # EKF contracts GPS error by 30% + process noise floor
                LinearConstraint("position_error",
                                 {"position_meas_error": ALPHA_POS}, BETA_POS),
                LinearConstraint("position_error", {}, 0.0, is_upper_bound=False),

                # --- Velocity estimation error ---
                # Theory: P_ss_vel from DARE, α ≈ 0.5 (better contraction via integration)
                LinearConstraint("velocity_error",
                                 {"velocity_meas_error": ALPHA_VEL}, BETA_VEL),
                LinearConstraint("velocity_error", {}, 0.0, is_upper_bound=False),

                # --- Attitude estimation error ---
                # Theory: P_ss_att from DARE, α ≈ 0.3 (strong contraction, low R_att)
                LinearConstraint("attitude_error",
                                 {"attitude_meas_error": ALPHA_ATT}, BETA_ATT),
                LinearConstraint("attitude_error", {}, 0.0, is_upper_bound=False),

                # --- Filter consistency (internal, for verification) ---
                # Theory: NEES ~ χ²(n_x)/n_x, 95% bound = χ²_inv(0.975,12)/12 = 1.95
                LinearConstraint("nees_bound", {}, NEES_BOUND),
                # Theory: NIS ~ χ²(n_z)/n_z
                LinearConstraint("nis_bound", {}, NIS_BOUND),
            ]
        )

        # =====================================================================
        # PID CONTROLLER CONTRACT
        #
        # Theory: Classical control — second-order analysis
        #   Characteristic: s² + kd×s + kp = 0 → ωn, ζ
        #   Settling: t_s = 4/(ζ×ωn)
        #   Tracking: e = est_error + disturbance/loop_gain + floor
        #
        # Specialist verification:
        #   1. Compute ωn = √kp = 0.447, ζ = kd/(2ωn) = 0.559
        #   2. Verify t_s = 4/(ζωn) ≈ 16s, Mp = exp(-πζ/√(1-ζ²)) ≈ 12%
        #   3. Check phase margin ≥ 50° from loop TF
        #   4. Final value theorem: e_ss = disturbance/kp for ramp
        #
        # Outputs flowing to Actuator: control_effort
        # Internal: settling_time, overshoot (verification only)
        # =====================================================================
        self.controller_contracts['PID'] = SimpleContract(
            name="PID_Controller",
            assumptions={
                "position_error": (0.0, 3.0),    # linearization valid range
                "velocity_error": (0.0, 2.0),    # velocity feedback accuracy
                "wind_speed": (0.0, 3.0),         # m/s — tilt saturates above this
                "disturbance": (0.0, 3.0),        # Nm, bounded external torques
            },
            guarantees=[
                # --- Tracking error ---
                # Theory: e = est_error + GAMMA_WIND_PID × wind + floor
                # GAMMA_WIND_PID = 0.5 reflects PID+integral: integral action eliminates
                # steady-state wind error; the bound covers the transient peak only.
                # (Pure-proportional bound would be 1/KP_EFF = 5.0 — far too pessimistic.)
                LinearConstraint("tracking_error",
                                 {"position_error": 1.0, "wind_speed": GAMMA_WIND_PID},
                                 E_FLOOR_PID),
                LinearConstraint("tracking_error", {}, 0.0, is_upper_bound=False),

                # --- Settling time ---
                # Theory: t_s = 4/(ζωn) + wind contribution
                # Base ≈ 16s from second-order analysis, wind extends it
                LinearConstraint("settling_time",
                                 {"position_error": 1.0, "wind_speed": 2.0},
                                 T_SETTLE_BASE),
                LinearConstraint("settling_time", {}, 1.0, is_upper_bound=False),

                # --- Control effort ---
                # Theory: u = kp × e + kd × ė + hover_feedforward
                LinearConstraint("control_effort",
                                 {"wind_speed": 0.15, "disturbance": 0.05}, 0.3),
                LinearConstraint("control_effort", {}, 0.1, is_upper_bound=False),

                # --- Max tilt (hardware saturation) ---
                LinearConstraint("max_tilt", {}, TILT_LIMIT),
                LinearConstraint("max_tilt", {}, 0.0, is_upper_bound=False),
            ]
        )

        # =====================================================================
        # H-INFINITY CONTROLLER CONTRACT
        #
        # Theory: Robust control ‖T_zw‖∞ ≤ γ
        #   ‖tracking_error‖₂ ≤ ‖est_error‖₂ + γ × ‖wind‖₂
        #   Stabilization via rate damping (k=0.8) + attitude correction (k=5.0)
        #
        # Very wide assumptions → "always available" safety net.
        #
        # Specialist verification:
        #   1. Compute closed-loop TF from wind to tracking error
        #   2. Find ‖T_zw‖∞ = max σ(T(jω)) → verify ≤ γ
        #   3. Check rate damping eigenvalues of closed-loop A
        #   4. Verify robust stability for bounded model uncertainty
        #
        # Outputs flowing to Actuator: control_effort
        # Internal: stabilization_time, tilt_angle, rate_damping_ratio
        # =====================================================================
        self.controller_contracts['Hinf'] = SimpleContract(
            name="Hinf_Controller",
            assumptions={
                "position_error": (0.0, 50.0),    # works with degraded estimation
                "velocity_error": (0.0, 30.0),
                "wind_speed": (0.0, 25.0),         # handles severe wind
                "disturbance": (0.0, 20.0),
            },
            guarantees=[
                # --- Tracking error ---
                # Theory: ‖e‖ ≤ ‖est_err‖ + γ × ‖w‖ + base (H∞ norm bound)
                # Does NOT track position (stabilization only) → high base error
                LinearConstraint("tracking_error",
                                 {"position_error": 1.0,
                                  "wind_speed": GAMMA_WIND_HINF},
                                 E_BASE_HINF),
                LinearConstraint("tracking_error", {}, 0.0, is_upper_bound=False),

                # --- Stabilization time ---
                # Theory: time constant from rate damping + attitude correction
                LinearConstraint("stabilization_time",
                                 {"position_error": 0.2, "wind_speed": 0.3}, 3.0),
                LinearConstraint("stabilization_time", {}, 2.0, is_upper_bound=False),

                # --- Tilt angle ---
                # Theory: steady-state tilt = disturbance / k_attitude
                LinearConstraint("tilt_angle",
                                 {"disturbance": 1.0 / K_ATT_GAIN,
                                  "wind_speed": 0.01}, 0.15),
                LinearConstraint("tilt_angle", {}, 0.0, is_upper_bound=False),

                # --- Control effort ---
                LinearConstraint("control_effort",
                                 {"wind_speed": 0.03, "disturbance": 0.005}, 0.4),
                LinearConstraint("control_effort", {}, 0.2, is_upper_bound=False),

                # --- Rate damping ratio (internal) ---
                # Theory: directly from gain k_rate_damping
                LinearConstraint("rate_damping", {}, K_RATE_DAMP,
                                 is_upper_bound=False),
            ]
        )

        # =====================================================================
        # MPC CONTROLLER CONTRACT
        #
        # Theory: Model Predictive Control — Lyapunov stability
        #   V*(x(k+1)) ≤ V*(x(k)) - l(x(k), u(k))
        #   Tracking bound: ‖e‖ ≤ √(V*(x₀) / λ_min(Q))
        #   Recursive feasibility: feasible at t → feasible at t+1
        #
        # Intermediate envelope: wind ≤ 15 m/s (between PID 3 and H-inf emergency >15)
        # Best tracking: optimization contracts error (α < 1)
        #
        # Specialist verification:
        #   1. Verify terminal cost P_f = DARE(A,B,Q,R) (Lyapunov condition)
        #   2. Check recursive feasibility: X_f invariant under terminal controller
        #   3. Compute λ_min(Q) → verify tracking bound
        #   4. Measure computation time → verify < 50ms
        #
        # Outputs flowing to Actuator: control_effort
        # Internal: settling_time, constraint_satisfaction
        # =====================================================================
        self.controller_contracts['MPC'] = SimpleContract(
            name="MPC_Controller",
            assumptions={
                "position_error": (0.0, 5.0),     # wider than PID (model handles more)
                "velocity_error": (0.0, 2.0),
                "wind_speed": (0.0, 15.0),         # extends to H-inf threshold (emergency-only above 15)
                "computation_time": (0.0, COMPUTE_TIME_MAX),  # real-time constraint
            },
            guarantees=[
                # --- Tracking error ---
                # Theory: e ≤ √(V*/λ_min(Q)) ≈ α × est_error + β × wind + floor
                # α < 1: optimization REDUCES error vs PID passthrough
                LinearConstraint("tracking_error",
                                 {"position_error": ALPHA_MPC,
                                  "wind_speed": BETA_MPC},
                                 E_FLOOR_MPC),
                LinearConstraint("tracking_error", {}, 0.0, is_upper_bound=False),

                # --- Settling time ---
                # Theory: convergence rate from Lyapunov decrease
                LinearConstraint("settling_time",
                                 {"position_error": 0.3, "wind_speed": 0.5}, 3.0),
                LinearConstraint("settling_time", {}, 1.5, is_upper_bound=False),

                # --- Control effort ---
                # Theory: optimal effort from cost minimization (lower than PID/H-inf)
                LinearConstraint("control_effort",
                                 {"wind_speed": 0.08, "position_error": 0.05}, 0.2),
                LinearConstraint("control_effort", {}, 0.1, is_upper_bound=False),
            ]
        )

        # =====================================================================
        # ACTUATOR CONTRACT
        #
        # Theory: Motor dynamics
        #   Thrust: T = k_f × ω²  (aerodynamic thrust equation)
        #   Max thrust: T_max ∝ V_battery²  (voltage limits RPM)
        #   Response: τ = L/R + J/(K_t × K_e)  (electrical + mechanical)
        #   Derating: accuracy drops with low voltage and high temperature
        #
        # Specialist verification:
        #   - Check k_f against propeller test data
        #   - Verify τ_motor against motor step response measurement
        #   - Check voltage derating against battery discharge curve
        #
        # No downstream contract (end of pipeline)
        # =====================================================================
        self.actuator_contract = SimpleContract(
            name="Actuators",
            assumptions={
                "control_effort": (0.0, 1.0),
                "battery_voltage": (11.0, 12.6),    # 3S LiPo range
                "motor_temperature": (20.0, 80.0),   # °C operational range
            },
            guarantees=[
                # --- Thrust accuracy (lower bound) ---
                # Theory: T_max ∝ V², thermal derating
                # At 12.6V, 20°C: ≥ 0.968, at 11.0V, 80°C: ≥ 0.89
                LinearConstraint("thrust_accuracy",
                                 {"battery_voltage": K_V_THRUST,
                                  "motor_temperature": -K_T_DERATE}, 0.6,
                                 is_upper_bound=False),
                # --- Thrust accuracy (upper bound) ---
                LinearConstraint("thrust_accuracy",
                                 {"battery_voltage": -K_V_THRUST,
                                  "motor_temperature": K_T_DERATE}, 1.4),

                # --- Response time ---
                # Theory: τ_motor + load-dependent component
                LinearConstraint("response_time",
                                 {"motor_temperature": 0.0005,
                                  "control_effort": 0.01}, TAU_MOTOR),
                LinearConstraint("response_time", {}, 0.005, is_upper_bound=False),

                # --- Thrust bandwidth (internal) ---
                # Theory: f_bw = 1/(2π×τ_motor)
                LinearConstraint("thrust_bandwidth", {},
                                 1.0 / (2.0 * 3.14159 * TAU_MOTOR),
                                 is_upper_bound=False),
            ]
        )

        logger.info("All component contracts defined (domain-verifiable equations)")

    def compose_pipeline(self, controller_name: str) -> SimpleContract:
        """
        Compose sensor -> estimator -> controller -> actuator pipeline.
        Returns composed contract with end-to-end relational guarantees.

        The composed guarantees are algebraic expressions in terms of
        the top-level inputs (gps_satellites, wind_speed, etc.)
        """
        if controller_name not in self.controller_contracts:
            logger.error(f"Controller {controller_name} not defined!")
            return None

        # Compose step by step (equations propagate through substitution)
        pipeline = self.sensor_contract
        pipeline = pipeline.compose(self.estimator_contract)
        pipeline = pipeline.compose(self.controller_contracts[controller_name])
        pipeline = pipeline.compose(self.actuator_contract)

        logger.info(f"Pipeline composed for {controller_name}")
        logger.info(f"  Assumptions: {list(pipeline.assumptions.keys())}")
        logger.info(f"  Guarantee equations ({len(pipeline.guarantees)}):")
        for g in pipeline.guarantees:
            if g.coefficients:  # Only log non-trivial guarantees
                logger.info(f"    {g}")

        return pipeline

    def verify_mission_feasibility(self,
                                   controller_name: str,
                                   current_conditions: Dict[str, float]) -> Tuple[bool, str]:
        """
        BEFORE TAKEOFF: Check if mission is feasible given current conditions.
        Uses contract composition to verify mission feasibility AND predict performance.
        """
        pipeline = self.compose_pipeline(controller_name)

        if pipeline is None:
            return False, "Pipeline composition failed"

        # Check if current conditions satisfy pipeline assumptions
        assumptions_met, margins = pipeline.check_assumptions(current_conditions)

        if not assumptions_met:
            violated = [k for k, v in margins.items() if v < 0]
            return False, f"Assumptions violated: {violated}"

        min_margin = min(margins.values()) if margins else 0.0

        # Compute predicted performance from relational guarantees
        predictions = pipeline.predict_guarantees(current_conditions)
        pred_items = sorted(predictions.items())
        pred_str = ", ".join(f"{k}={v:.3f}" for k, v in pred_items)

        return True, (f"Mission feasible with {controller_name} "
                      f"(margin: {min_margin:.2f}, predicted: {pred_str})")

    def monitor_runtime(self,
                       component: str,
                       values: Dict[str, float],
                       timestamp: float) -> ContractMetrics:
        """
        DURING FLIGHT: Monitor if contracts are satisfied at runtime.
        Also computes predicted guarantee bounds from current inputs.
        """
        contract = None

        if component == "sensors":
            contract = self.sensor_contract
        elif component == "estimator":
            contract = self.estimator_contract
        elif component.startswith("controller_"):
            ctrl_name = component.split("_")[1]
            contract = self.controller_contracts.get(ctrl_name)
        elif component == "actuators":
            contract = self.actuator_contract

        if contract is None:
            logger.warning(f"Unknown component: {component}")
            return ContractMetrics(
                timestamp=timestamp,
                component_name=component,
                assumptions_met=False,
                guarantees_met=False
            )

        # Check assumptions and guarantees
        assumptions_met, assumption_margins = contract.check_assumptions(values)
        guarantees_met, guarantee_margins = contract.check_guarantees(values)

        # Compute predicted bounds from input values
        predicted_bounds = contract.predict_guarantees(values)

        # Compute overall margin (worst case across assumptions and any checked guarantees)
        all_margins = {**assumption_margins, **guarantee_margins}
        min_margin = min(all_margins.values()) if all_margins else 0.0

        metrics = ContractMetrics(
            timestamp=timestamp,
            component_name=component,
            assumptions_met=assumptions_met,
            guarantees_met=guarantees_met,
            assumption_values=values.copy(),
            guarantee_values=values.copy(),
            predicted_bounds=predicted_bounds,
            margin=min_margin
        )

        self.history.append(metrics)

        return metrics

    def get_contract_status(self, component: str) -> ContractStatus:
        """Get current contract status for a component"""
        if not self.history:
            return ContractStatus.UNKNOWN

        # Get latest metrics for this component
        recent = [m for m in self.history if m.component_name == component]
        if not recent:
            return ContractStatus.UNKNOWN

        latest = recent[-1]

        if latest.assumptions_met and latest.guarantees_met:
            return ContractStatus.SATISFIED
        elif not latest.assumptions_met and latest.guarantees_met:
            return ContractStatus.ASSUMPTIONS_VIOLATED
        elif latest.assumptions_met and not latest.guarantees_met:
            return ContractStatus.GUARANTEES_VIOLATED
        else:
            return ContractStatus.BOTH_VIOLATED

    def should_switch_controller(self,
                                current_controller: str,
                                system_state: Dict[str, float]) -> Tuple[bool, Optional[str], str]:
        """
        Determine if controller should switch based on contract violations.
        Returns: (should_switch, new_controller, reason)
        """
        # Check current controller's contract
        current_contract = self.controller_contracts.get(current_controller)
        if current_contract is None:
            return False, None, "Current controller not found"

        assumptions_met, margins = current_contract.check_assumptions(system_state)

        if assumptions_met:
            # Current controller OK, but check if more efficient option available
            # Priority: PID (efficient) > MPC (optimal) > Hinf (safe)

            if current_controller == 'Hinf':
                # Try to switch back to more efficient controllers
                pid_contract = self.controller_contracts['PID']
                pid_ok, _ = pid_contract.check_assumptions(system_state)
                if pid_ok:
                    return True, 'PID', "Conditions improved, switching back to efficient PID"

                mpc_contract = self.controller_contracts.get('MPC')
                if mpc_contract:
                    mpc_ok, _ = mpc_contract.check_assumptions(system_state)
                    if mpc_ok:
                        return True, 'MPC', "Conditions improved, switching to optimal MPC"

            elif current_controller == 'MPC':
                # Try to switch to PID if even more efficient
                pid_contract = self.controller_contracts['PID']
                pid_ok, _ = pid_contract.check_assumptions(system_state)
                if pid_ok:
                    return True, 'PID', "Conditions normal, switching to more efficient PID"

            # Current controller is the best option
            return False, None, "Current controller assumptions satisfied"

        # Current controller assumptions violated - need to switch
        violated_vars = [k for k, v in margins.items() if v < 0]

        # Try other controllers in priority order
        # Priority: PID (efficient) > MPC (optimal) > Hinf (safe)
        candidates = ['PID', 'MPC', 'Hinf']
        if current_controller in candidates:
            candidates.remove(current_controller)

        for candidate in candidates:
            candidate_contract = self.controller_contracts[candidate]
            candidate_ok, _ = candidate_contract.check_assumptions(system_state)

            if candidate_ok:
                reason = f"{current_controller} violated {violated_vars}, switching to {candidate}"
                return True, candidate, reason

        # If no controller assumptions satisfied, force Hinf (widest assumptions)
        if current_controller != 'Hinf':
            reason = f"Emergency: All controllers violated, forcing Hinf"
            return True, 'Hinf', reason

        return False, None, "Already in Hinf, no alternatives"

    def export_metrics(self, filename: str):
        """Export contract monitoring history for analysis"""
        import json

        data = {
            "history": [
                {
                    "timestamp": m.timestamp,
                    "component": m.component_name,
                    "assumptions_met": m.assumptions_met,
                    "guarantees_met": m.guarantees_met,
                    "margin": m.margin,
                    "values": m.assumption_values,
                    "predicted_bounds": m.predicted_bounds
                }
                for m in self.history
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self.history)} metrics to {filename}")


# =========================================================================
# Example usage and tests
# =========================================================================
if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)

    print("=" * 70)
    print("Contract Framework Test — Domain-Verifiable Equations")
    print("=" * 70)

    monitor = HierarchicalContractMonitor()
    monitor.define_contracts()

    # ---- Test 1: Contract structure (theory-grounded) ----
    print("\n[TEST 1] Contract Structure (Theory-Grounded)")
    print("-" * 50)
    print(monitor.sensor_contract)
    print()
    print(monitor.estimator_contract)
    print()
    print(monitor.controller_contracts['MPC'])

    # ---- Test 2: GPS sensor predictions ----
    print("\n[TEST 2] Sensor Predictions (GPS + IMU)")
    print("-" * 50)
    sensor_good = {
        "gps_satellites": 12.0, "gps_hdop": 0.8,
        "imu_temperature": 25.0, "imu_calibrated": 1.0,
    }
    preds = monitor.sensor_contract.predict_guarantees(sensor_good)
    print(f"  Good GPS (HDOP=0.8, T=25C):")
    for k, v in sorted(preds.items()):
        print(f"    {k} = {v:.4f}")

    sensor_poor = {
        "gps_satellites": 6.0, "gps_hdop": 4.0,
        "imu_temperature": 50.0, "imu_calibrated": 1.0,
    }
    preds_bad = monitor.sensor_contract.predict_guarantees(sensor_poor)
    print(f"  Poor GPS (HDOP=4.0, T=50C):")
    for k, v in sorted(preds_bad.items()):
        print(f"    {k} = {v:.4f}")

    # ---- Test 3: EKF contraction ----
    print("\n[TEST 3] EKF Contraction (DARE-derived)")
    print("-" * 50)
    ekf_inputs = {
        "position_meas_error": 1.2,   # From HDOP=0.8
        "velocity_meas_error": 0.1,
        "rate_meas_error": 0.026,
        "attitude_meas_error": 0.00052,
    }
    ekf_preds = monitor.estimator_contract.predict_guarantees(ekf_inputs)
    print(f"  Sensor errors: pos={ekf_inputs['position_meas_error']:.2f}m, "
          f"vel={ekf_inputs['velocity_meas_error']:.2f}m/s")
    print(f"  EKF contracts to:")
    for k, v in sorted(ekf_preds.items()):
        if '_max' in k:
            name = k.replace('_max', '')
            print(f"    {name} <= {v:.4f}  (contraction ratio visible)")

    # ---- Test 4: Three-tier controller comparison ----
    print("\n[TEST 4] Three-Tier Controller Predictions")
    print("-" * 50)
    ctrl_state = {
        "position_error": 1.5, "velocity_error": 0.3,
        "wind_speed": 2.0, "disturbance": 1.0,
        "computation_time": 0.01,
    }
    print(f"  Conditions: pos_err=1.5m, wind=2.0 m/s")
    for name in ['PID', 'MPC', 'Hinf']:
        contract = monitor.controller_contracts[name]
        ok, _ = contract.check_assumptions(ctrl_state)
        preds = contract.predict_guarantees(ctrl_state)
        te = preds.get('tracking_error_max', '?')
        if isinstance(te, float):
            te = f"{te:.3f}m"
        print(f"  {name:5s}: assumptions_met={ok}, tracking_error <= {te}")

    # ---- Test 5: Pipeline composition (PID) ----
    print("\n[TEST 5] Pipeline Composition (PID)")
    print("-" * 50)
    pipeline = monitor.compose_pipeline("PID")
    print(f"  Pipeline: {pipeline.name}")
    print(f"  Top-level inputs: {sorted(pipeline.assumptions.keys())}")
    print(f"  End-to-end equations:")
    for g in pipeline.guarantees:
        if g.coefficients:
            print(f"    {g}")

    # ---- Test 6: End-to-end prediction ----
    print("\n[TEST 6] End-to-End Predictions")
    print("-" * 50)
    conditions = {
        "gps_satellites": 12.0, "gps_hdop": 1.0,
        "imu_temperature": 25.0, "imu_calibrated": 1.0,
        "battery_voltage": 12.4, "motor_temperature": 25.0,
        "wind_speed": 1.5, "disturbance": 0.5,
    }
    e2e = pipeline.predict_guarantees(conditions)
    print(f"  Conditions: HDOP=1.0, T=25C, wind=1.5 m/s")
    for k, v in sorted(e2e.items()):
        if '_max' in k:
            print(f"    {k.replace('_max','')} <= {v:.3f}")

    # ---- Test 7: Pre-flight verification ----
    print("\n[TEST 7] Pre-flight Verification")
    print("-" * 50)
    feasible, msg = monitor.verify_mission_feasibility("PID", conditions)
    print(f"  PID: Feasible={feasible}")

    windy = conditions.copy()
    windy["wind_speed"] = 5.0
    windy["disturbance"] = 3.0
    windy["computation_time"] = 0.01  # MPC solve time (required for MPC pipeline)

    feasible, msg = monitor.verify_mission_feasibility("PID", windy)
    print(f"  PID (wind=5): Feasible={feasible}")
    feasible, msg = monitor.verify_mission_feasibility("MPC", windy)
    print(f"  MPC (wind=5): Feasible={feasible}")
    feasible, msg = monitor.verify_mission_feasibility("Hinf", windy)
    print(f"  Hinf (wind=5): Feasible={feasible}")

    # ---- Test 8: Three-tier switching ----
    print("\n[TEST 8] Three-Tier Controller Switching")
    print("-" * 50)
    print(f"  {'Wind':>5s}  {'Current':>8s}  {'Switch?':>7s}  {'New':>8s}  Reason")
    print(f"  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*30}")

    current = 'PID'
    for wind in [1.0, 2.5, 3.5, 5.0, 8.5, 5.0, 2.0]:
        state = {
            "position_error": 1.5, "velocity_error": 0.3,
            "wind_speed": wind, "disturbance": wind * 0.5,
            "computation_time": 0.01,
        }
        switch, new, reason = monitor.should_switch_controller(current, state)
        new_ctrl = new if switch else current
        short_reason = reason[:40]
        print(f"  {wind:5.1f}  {current:>8s}  {'YES' if switch else 'no':>7s}  "
              f"{new_ctrl:>8s}  {short_reason}")
        if switch:
            current = new

    print("\n" + "=" * 70)
    print("[OK] Contract Framework Test Complete")
    print("=" * 70)
