"""
Pre-flight contract verification.

- ContractPreflightMixin gives HierarchicalContractMonitor (monitor.py) its
  compose_pipeline and verify_mission_feasibility, moved unchanged from
  contract_framework.py.
- ControllerPreflightMixin gives AdaptiveDroneController (core.py) its
  pre_flight_check, moved unchanged from adaptive_control_system.py.
"""

import numpy as np
from typing import Dict, Tuple
import logging

from .spec import SimpleContract

logger = logging.getLogger(__name__)


class ContractPreflightMixin:
    """Pipeline composition and mission feasibility for HierarchicalContractMonitor."""

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


class ControllerPreflightMixin:
    """Pre-flight check of AdaptiveDroneController: picks the first feasible controller."""

    def pre_flight_check(self,
                        initial_conditions: Dict[str, float],
                        mission: Dict[str, any]) -> Tuple[bool, str]:
        """
        Pre-flight contract verification.
        Uses contract composition to verify mission is feasible.
        Sets up the supervisor with the mission target.
        """
        logger.info("=" * 60)
        logger.info("PRE-FLIGHT CONTRACT VERIFICATION")
        logger.info("=" * 60)

        target = mission.get('target_position', np.array([0.0, 0.0, -5.0]))
        waypoints = mission.get('waypoints', [target])

        # Try controllers in order: PID (efficient) → MPC (optimal) → H-inf (robust)
        controllers = [
            ('PID', "efficient PID"),
            ('MPC', "optimal MPC"),
            ('Hinf', "robust H-infinity"),
        ]

        for i, (name, desc) in enumerate(controllers, 1):
            logger.info(f"\n[{i}/{len(controllers)}] Checking {desc}...")
            # MPC pipeline needs computation_time in conditions
            check_conditions = initial_conditions.copy()
            if name == 'MPC' and 'computation_time' not in check_conditions:
                check_conditions['computation_time'] = 0.01  # typical solve time

            feasible, msg = self.contract_monitor.verify_mission_feasibility(
                name, check_conditions
            )

            if feasible:
                logger.info(f"{name} feasible: {msg}")
                if name != 'PID':
                    self.controller_switcher.switch_to(
                        name, 0.0, f"Pre-flight: conditions require {desc}")
                self.supervisor.set_waypoints(waypoints)
                self.mission_feasible = True
                perf_note = "" if name == 'PID' else " (degraded performance expected)"
                return True, f"Mission feasible with {desc}{perf_note}"
            else:
                logger.warning(f"{name} infeasible: {msg}")

        logger.error("MISSION INFEASIBLE - all controller contracts violated!")
        self.mission_feasible = False
        return False, "Mission infeasible - all controller contracts violated"
