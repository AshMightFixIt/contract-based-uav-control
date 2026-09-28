"""
Horizon-Based Contract Planner

Uses Pacti contracts to:
1. Cascade contracts over a planning horizon
2. Optimize safety margins at each step
3. Re-plan when safety is threatened

Key Innovation:
- PREDICTIVE planning using formal contract composition
- ADAPTIVE horizon: shrink when safety margins are tight
- GRACEFUL degradation: switch strategies when needed
"""

from pacti.contracts import PolyhedralIoContract
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planning.pacti_contracts import PactiContractLibrary

logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """Status of the current plan."""
    FEASIBLE = "feasible"           # Plan is safe
    MARGINAL = "marginal"           # Plan is safe but margins are tight
    REPLAN_HORIZON = "replan_horizon"  # Need shorter horizon
    REPLAN_STRATEGY = "replan_strategy"  # Need different approach
    INFEASIBLE = "infeasible"       # No safe plan found


@dataclass
class HorizonStep:
    """One step in the planning horizon."""
    timestep: int
    controller: str                 # 'pid', 'mpc', or 'hinf'
    tracking_error_bound: float     # Upper bound on tracking error
    safety_margin: float            # Distance from safety limit
    contract: Optional[PolyhedralIoContract] = None


@dataclass
class Plan:
    """Complete plan over the horizon."""
    horizon_length: int
    steps: List[HorizonStep]
    status: PlanStatus
    min_safety_margin: float
    recommended_controller: str
    replan_reason: Optional[str] = None

    def is_safe(self) -> bool:
        return self.status in [PlanStatus.FEASIBLE, PlanStatus.MARGINAL]


@dataclass
class PlannerConfig:
    """Configuration for the horizon planner."""
    max_horizon: int = 10           # Maximum planning horizon (steps)
    min_horizon: int = 3            # Minimum planning horizon
    dt: float = 0.1                 # Timestep for planning (can be coarser than control)
    safety_limit: float = 5.0       # Maximum acceptable tracking error
    margin_threshold: float = 1.0   # Minimum acceptable safety margin
    replan_threshold: float = 0.5   # Margin below this triggers re-planning


class HorizonPlanner:
    """
    Plans over a horizon using Pacti contract composition.

    Architecture:
        Current State -> Cascade Contracts -> Optimize Margins -> Plan/Replan

    At each planning step:
    1. Build horizon contract by composing dynamics over N steps
    2. For each step, compute safety margin (safety_limit - tracking_error_bound)
    3. If margin < threshold, try:
       a. Shorter horizon
       b. Different controller
       c. Modified target
    4. Return best feasible plan
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.contracts = PactiContractLibrary()
        self.current_plan: Optional[Plan] = None
        self.replan_count = 0

        logger.info(f"HorizonPlanner initialized: horizon={self.config.max_horizon}, "
                   f"safety_limit={self.config.safety_limit}")

    def plan(self,
             current_state: Dict[str, float],
             target_position: np.ndarray,
             environment: Dict[str, float]) -> Plan:
        """
        Generate a plan over the horizon.

        Args:
            current_state: Current drone state
                - position: [x, y, z]
                - velocity: [vx, vy, vz]
                - tracking_error: current tracking error
            target_position: Goal position
            environment: Environmental conditions
                - wind_speed: estimated wind
                - gps_satellites: number of satellites
                - gps_hdop: horizontal dilution of precision
                - battery_voltage: current battery voltage

        Returns:
            Plan with horizon steps and recommended controller
        """
        # Start with maximum horizon
        horizon = self.config.max_horizon

        # Try controllers in order: PID (efficient) → MPC (optimal) → H-inf (robust)
        for ctrl in ['pid', 'mpc', 'hinf']:
            plan = self._try_plan(current_state, environment, horizon, ctrl)
            if plan.is_safe():
                logger.info(f"{ctrl.upper()} plan feasible: horizon={horizon}, "
                           f"min_margin={plan.min_safety_margin:.2f}")
                return plan
            logger.warning(f"{ctrl.upper()} plan failed: {plan.replan_reason}")

        # All controllers failed at max horizon, try shrinking
        logger.warning(f"All controllers failed at horizon={horizon}, trying shorter horizons")

        for h in range(horizon - 1, self.config.min_horizon - 1, -1):
            # Try each controller at reduced horizon (MPC before H-inf)
            for ctrl in ['mpc', 'hinf']:
                plan = self._try_plan(current_state, environment, h, ctrl)
                if plan.is_safe():
                    logger.info(f"{ctrl.upper()} plan feasible at reduced horizon={h}")
                    return plan

        # All plans failed
        logger.error("No feasible plan found!")
        return Plan(
            horizon_length=self.config.min_horizon,
            steps=[],
            status=PlanStatus.INFEASIBLE,
            min_safety_margin=-1.0,
            recommended_controller='hinf',
            replan_reason="All planning attempts failed - emergency mode"
        )

    def _try_plan(self,
                  current_state: Dict[str, float],
                  environment: Dict[str, float],
                  horizon: int,
                  controller: str) -> Plan:
        """
        Try to build a plan with given horizon and controller.
        """
        steps = []
        min_margin = float('inf')

        # Get initial tracking error — kept fixed so _propagate_error computes
        # the correct closed-form 0.9^t * e_0 + (1-0.9^t) * e_ss at each step t.
        initial_tracking_error = current_state.get('tracking_error', 0.0)

        # Get environment parameters
        wind_speed = environment.get('wind_speed', 0.0)
        gps_sats = environment.get('gps_satellites', 12.0)
        gps_hdop = environment.get('gps_hdop', 1.0)
        battery = environment.get('battery_voltage', 12.0)

        # Check if controller assumptions are satisfied
        ctrl_contract = self.contracts.get_contract(controller)
        if ctrl_contract is None:
            return Plan(
                horizon_length=horizon,
                steps=[],
                status=PlanStatus.INFEASIBLE,
                min_safety_margin=-1.0,
                recommended_controller=controller,
                replan_reason=f"Controller contract not found: {controller}"
            )

        # Build assumptions dict for controller
        ctrl_assumptions = {
            'position_est_error': current_state.get('position_est_error', 0.5),
            'velocity_est_error': current_state.get('velocity_est_error', 0.1),
            'wind_speed': wind_speed,
            'initial_tracking_error': initial_tracking_error,
        }

        # Check if assumptions are satisfied
        assumptions_ok = self._check_assumptions(ctrl_contract, ctrl_assumptions)

        if not assumptions_ok:
            return Plan(
                horizon_length=horizon,
                steps=[],
                status=PlanStatus.INFEASIBLE,
                min_safety_margin=-1.0,
                recommended_controller=controller,
                replan_reason=f"{controller} assumptions violated"
            )

        # Cascade dynamics over horizon
        for t in range(horizon):
            # Build step contract by composing controller with dynamics
            step_contract = self._build_step_contract(controller, t)

            # Closed-form bound: e_t = 0.9^t * e_0 + (1 - 0.9^t) * e_ss(wind, ctrl)
            error_bound = self._propagate_error(initial_tracking_error, wind_speed, t, controller)

            # Compute safety margin
            margin = self.config.safety_limit - error_bound

            step = HorizonStep(
                timestep=t,
                controller=controller,
                tracking_error_bound=error_bound,
                safety_margin=margin,
                contract=step_contract
            )
            steps.append(step)

            min_margin = min(min_margin, margin)

        # Determine plan status
        if min_margin >= self.config.margin_threshold:
            status = PlanStatus.FEASIBLE
        elif min_margin >= self.config.replan_threshold:
            status = PlanStatus.MARGINAL
        elif min_margin > 0:
            status = PlanStatus.REPLAN_HORIZON
        else:
            status = PlanStatus.REPLAN_STRATEGY

        replan_reason = None
        if status == PlanStatus.REPLAN_HORIZON:
            replan_reason = f"Safety margin too tight: {min_margin:.2f} < {self.config.margin_threshold}"
        elif status == PlanStatus.REPLAN_STRATEGY:
            replan_reason = f"Safety violated at horizon: margin={min_margin:.2f}"

        return Plan(
            horizon_length=horizon,
            steps=steps,
            status=status,
            min_safety_margin=min_margin,
            recommended_controller=controller,
            replan_reason=replan_reason
        )

    def _check_assumptions(self,
                           contract: PolyhedralIoContract,
                           values: Dict[str, float]) -> bool:
        """
        Check if contract assumptions are satisfied by given values.
        """
        # For each assumption constraint, check if value is within bounds
        try:
            for var in contract.inputvars:
                var_name = str(var)
                if var_name in values:
                    bounds = contract.get_variable_bounds(var_name)
                    val = values[var_name]
                    if val < bounds[0] or val > bounds[1]:
                        logger.debug(f"Assumption violated: {var_name}={val} not in {bounds}")
                        return False
            return True
        except Exception as e:
            logger.warning(f"Error checking assumptions: {e}")
            return True  # Optimistic if we can't check

    def _build_step_contract(self,
                             controller: str,
                             timestep: int) -> Optional[PolyhedralIoContract]:
        """
        Return the pre-cached ctrl ∘ dynamics contract.
        Falls back to on-demand composition if the cache entry is missing.
        """
        cached = self.contracts.get_step_contract(controller)
        if cached is not None:
            return cached
        # Fallback: compose on demand (cache missed at init)
        try:
            ctrl = self.contracts.get_contract(controller)
            dyn = self.contracts.get_contract('dynamics')
            if ctrl and dyn:
                return ctrl.compose(dyn)
        except Exception as e:
            logger.warning(f"Failed to compose step contract: {e}")
        return None

    def _propagate_error(self,
                         initial_error: float,
                         wind_speed: float,
                         steps: int,
                         controller: str) -> float:
        """
        Propagate tracking error over N steps using controller-specific steady-state bounds.

        Dynamics decay: 0.9 per step (from dynamics contract).
        Steady-state target: e_ss = wind_coeff * wind + constant (from controller contract).

        Formula: e_N = 0.9^N * e_0 + (1 - 0.9^N) * e_ss(wind, controller)

        This gives different bounds per controller:
          MPC (wind_coeff=0.15) is tightest, H-inf (0.50 + 1.0) is most conservative.
        """
        decay_n = 0.9 ** steps
        e_ss = self.contracts.get_controller_ss_error(controller, wind_speed)
        return decay_n * initial_error + (1 - decay_n) * e_ss

    def replan(self,
               current_state: Dict[str, float],
               environment: Dict[str, float],
               reason: str) -> Plan:
        """
        Trigger re-planning due to changing conditions.
        """
        self.replan_count += 1
        logger.warning(f"Re-planning triggered (#{self.replan_count}): {reason}")

        # Try with current max horizon
        plan = self.plan(current_state, np.zeros(3), environment)

        if not plan.is_safe():
            # Emergency: force H-inf with minimum horizon
            logger.error("Re-planning failed, forcing emergency mode")
            plan = Plan(
                horizon_length=self.config.min_horizon,
                steps=[],
                status=PlanStatus.INFEASIBLE,
                min_safety_margin=-1.0,
                recommended_controller='hinf',
                replan_reason="Emergency: re-planning failed"
            )

        self.current_plan = plan
        return plan

    def check_and_replan(self,
                         current_state: Dict[str, float],
                         environment: Dict[str, float]) -> Tuple[bool, Optional[Plan]]:
        """
        Check if current plan is still valid, replan if needed.

        Returns: (needs_replan, new_plan)
        """
        if self.current_plan is None:
            return True, self.plan(current_state, np.zeros(3), environment)

        # Check if conditions have changed significantly
        current_error = current_state.get('tracking_error', 0.0)
        wind = environment.get('wind_speed', 0.0)

        # Recompute safety margin with current conditions
        margin = self.config.safety_limit - current_error

        if margin < self.config.replan_threshold:
            new_plan = self.replan(
                current_state, environment,
                f"Safety margin critical: {margin:.2f}"
            )
            return True, new_plan

        # Check if environment violates current plan's controller assumptions
        ctrl = self.current_plan.recommended_controller
        wind_limits = {'pid': 3.0, 'mpc': 8.0, 'hinf': 15.0}
        wind_limit = wind_limits.get(ctrl, 15.0)

        if wind > wind_limit:
            new_plan = self.replan(
                current_state, environment,
                f"Wind ({wind:.1f} m/s) exceeds {ctrl.upper()} envelope ({wind_limit} m/s)"
            )
            return True, new_plan

        return False, None

    def get_recommended_action(self) -> Tuple[str, int]:
        """
        Get recommended controller and horizon from current plan.

        Returns: (controller_name, effective_horizon)
        """
        if self.current_plan is None or not self.current_plan.is_safe():
            return 'hinf', self.config.min_horizon

        return self.current_plan.recommended_controller, self.current_plan.horizon_length


class AdaptiveRePlanner:
    """
    Higher-level replanning strategies when basic horizon planning fails.

    Strategies (in order of preference):
    1. Shrink horizon - less lookahead, more reactive
    2. Switch controller - PID -> H-inf
    3. Modify target - slow down, change waypoint
    4. Emergency hover - stop and stabilize
    """

    def __init__(self, planner: HorizonPlanner):
        self.planner = planner
        self.strategy_attempts = 0
        self.max_attempts = 5

    def find_feasible_plan(self,
                           current_state: Dict[str, float],
                           target: np.ndarray,
                           environment: Dict[str, float]) -> Tuple[Plan, Dict[str, any]]:
        """
        Try multiple strategies to find a feasible plan.

        Returns: (plan, modifications) where modifications describes
                 any changes made (horizon, controller, target)
        """
        modifications = {
            'horizon_reduced': False,
            'controller_switched': False,
            'target_modified': False,
            'emergency_mode': False
        }

        # Strategy 1: Try with PID at full horizon
        plan = self.planner.plan(current_state, target, environment)
        if plan.is_safe():
            return plan, modifications

        # Strategy 2: Already tried H-inf in plan(), check if it worked
        if plan.recommended_controller == 'hinf' and plan.is_safe():
            modifications['controller_switched'] = True
            return plan, modifications

        # Strategy 3: Shrink horizon progressively
        for h in range(self.planner.config.max_horizon - 2,
                       self.planner.config.min_horizon - 1, -2):
            plan = self.planner._try_plan(current_state, environment, h, 'hinf')
            if plan.is_safe():
                modifications['horizon_reduced'] = True
                modifications['new_horizon'] = h
                logger.info(f"Found feasible plan with reduced horizon: {h}")
                return plan, modifications

        # Strategy 4: Modify target (slow approach)
        # Reduce effective target distance
        modified_env = environment.copy()
        modified_env['wind_speed'] = max(0, environment.get('wind_speed', 0) - 1)

        plan = self.planner._try_plan(current_state, modified_env,
                                      self.planner.config.min_horizon, 'hinf')
        if plan.is_safe():
            modifications['target_modified'] = True
            logger.info("Found feasible plan with modified approach")
            return plan, modifications

        # Strategy 5: Emergency mode
        logger.error("All strategies failed - entering emergency mode")
        modifications['emergency_mode'] = True
        return Plan(
            horizon_length=1,
            steps=[],
            status=PlanStatus.INFEASIBLE,
            min_safety_margin=-1.0,
            recommended_controller='hinf',
            replan_reason="Emergency: all strategies exhausted"
        ), modifications


# Test the horizon planner
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Horizon Planner Test")
    print("=" * 70)

    planner = HorizonPlanner()

    # Test 1: Nominal conditions
    print("\n=== TEST 1: Nominal Conditions ===")
    state = {
        'tracking_error': 0.5,
        'position_est_error': 0.3,
        'velocity_est_error': 0.1,
    }
    env = {
        'wind_speed': 1.0,
        'gps_satellites': 12,
        'gps_hdop': 1.0,
        'battery_voltage': 12.4,
    }

    plan = planner.plan(state, np.array([10, 0, -5]), env)
    print(f"Status: {plan.status.value}")
    print(f"Controller: {plan.recommended_controller}")
    print(f"Horizon: {plan.horizon_length}")
    print(f"Min safety margin: {plan.min_safety_margin:.2f}")
    print(f"Steps: {len(plan.steps)}")

    if plan.steps:
        print("\nHorizon steps:")
        for step in plan.steps[:5]:  # First 5 steps
            print(f"  t={step.timestep}: error_bound={step.tracking_error_bound:.2f}, "
                  f"margin={step.safety_margin:.2f}")

    # Test 2: High wind - should switch to H-inf
    print("\n=== TEST 2: High Wind ===")
    env['wind_speed'] = 5.0
    plan = planner.plan(state, np.array([10, 0, -5]), env)
    print(f"Status: {plan.status.value}")
    print(f"Controller: {plan.recommended_controller}")
    print(f"Min safety margin: {plan.min_safety_margin:.2f}")
    print(f"Replan reason: {plan.replan_reason}")

    # Test 3: Extreme conditions - may need horizon reduction
    print("\n=== TEST 3: Extreme Conditions ===")
    state['tracking_error'] = 3.0
    env['wind_speed'] = 8.0
    plan = planner.plan(state, np.array([10, 0, -5]), env)
    print(f"Status: {plan.status.value}")
    print(f"Controller: {plan.recommended_controller}")
    print(f"Horizon: {plan.horizon_length}")
    print(f"Min safety margin: {plan.min_safety_margin:.2f}")
    print(f"Replan reason: {plan.replan_reason}")

    # Test 4: Adaptive replanner
    print("\n=== TEST 4: Adaptive Replanner ===")
    replanner = AdaptiveRePlanner(planner)
    state['tracking_error'] = 4.0
    env['wind_speed'] = 10.0

    plan, mods = replanner.find_feasible_plan(state, np.array([10, 0, -5]), env)
    print(f"Status: {plan.status.value}")
    print(f"Modifications: {mods}")

    print("\n" + "=" * 70)
    print("Horizon Planner Test Complete")
    print("=" * 70)
