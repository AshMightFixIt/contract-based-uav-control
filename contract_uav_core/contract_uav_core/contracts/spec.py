"""
Contract types of the hierarchical contract framework.

ContractStatus, ContractMetrics, LinearConstraint and SimpleContract, moved
unchanged from contract_framework.py. The contract definitions are in
library.py, pipeline composition and pre-flight checks in preflight.py, and
HierarchicalContractMonitor in monitor.py.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from pacti.contracts import PolyhedralIoContract
    PACTI_AVAILABLE = True
except ImportError:
    PACTI_AVAILABLE = False
    print("WARNING: Pacti not available. Using simplified contracts.")

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
