"""
Pre-flight contract verification.

ContractPreflightMixin gives HierarchicalContractMonitor (monitor.py) its
compose_pipeline and verify_mission_feasibility, moved unchanged from
contract_framework.py.
"""

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
