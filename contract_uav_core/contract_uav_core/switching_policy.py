"""
Controller-recommendation policy for AdaptiveDroneController (core.py).

The multi-factor switching signals A-E and the rule chain that turns them into
a recommended controller, moved unchanged from step 7 of control_step in
adaptive_control_system.py. control_step calls _recommend_controller and hands
the result to the flight-mode supervisor.
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SwitchingPolicyMixin:
    """Step 7 of control_step: the controller the contracts recommend."""

    def _recommend_controller(self,
                              system_conditions: Dict[str, float],
                              gps_ok: bool,
                              imu_ok: bool,
                              planner_emergency: bool,
                              planner_controller: Optional[str]) -> str:
        """Return the recommended controller: 'PID', 'MPC' or 'Hinf'.

        Reads the controller switcher, the EKF covariance and the controller
        contracts, and updates self._prev_tracking_error (signal E).
        """
        # 7. Multi-factor contract-based switching: PID > MPC > H-inf
        #
        # Four signals evaluated jointly — wind speed is ONE of them, not the only one:
        #
        #   A. Assumption feasibility  — can controller X legally operate here?
        #   B. Guarantee violation     — is the active controller meeting its promises?
        #   C. Predictive comparison   — which controller predicts the tightest tracking error?
        #   D. EKF uncertainty         — high covariance → prefer robust controller
        #   E. Error trend             — diverging error → proactive escalation
        #
        # Escalation is forced by B or E; C is used to pick between A-feasible controllers.
        # The horizon planner can further escalate (never downgrade) on top of this.

        check_conditions = system_conditions.copy()
        mpc_ctrl = self.controller_switcher.controllers.get('MPC')
        if mpc_ctrl:
            check_conditions['computation_time'] = getattr(mpc_ctrl, 'last_solve_time', 0.01)

        # D: EKF position uncertainty — use as the theory-grounded position_error contract input
        # (replaces the tracking-error proxy that was conflating input/output)
        ekf_P = self.ekf.get_covariance()
        pos_uncertainty = float(np.sqrt(np.trace(ekf_P[0:3, 0:3])))
        check_conditions['position_error'] = pos_uncertainty

        # Observed tracking performance (contract output, distinct from EKF estimation error)
        observed_tracking_error = system_conditions.get('position_error', 0.0)

        # E: Error trend — proactive escalation before conditions become unrecoverable
        error_trend = (observed_tracking_error - self._prev_tracking_error) / self.dt
        self._prev_tracking_error = observed_tracking_error
        error_rising_fast = error_trend > 3.0   # m/s equivalent — diverging, not just noisy

        # Dual-sensor failure: GPS + IMU both gone → dead-reckoning only → force H-inf
        dual_sensor_failure = not gps_ok and not imu_ok

        pid_contract = self.contract_monitor.controller_contracts.get('PID')
        mpc_contract = self.contract_monitor.controller_contracts.get('MPC')

        # A: Assumption feasibility
        pid_ok = False
        mpc_ok = False
        if pid_contract:
            pid_ok, _ = pid_contract.check_assumptions(check_conditions)
        if mpc_contract:
            mpc_ok, _ = mpc_contract.check_assumptions(check_conditions)

        # B: Guarantee violation — check if active controller exceeds its promised bound.
        # IMPORTANT: Guarantees are steady-state bounds. During waypoint approach the
        # drone is deliberately closing distance, so observed_tracking_error is naturally
        # large and transient. Checking guarantees while error_trend < 0 (approaching)
        # causes spurious H-inf escalation. Only raise the flag when error is growing.
        active_ctrl = self.controller_switcher.get_active_controller()
        active_contract = self.contract_monitor.controller_contracts.get(active_ctrl)
        guarantee_violated = False
        if active_contract and error_trend > 0.5:   # only check when error is growing
            guar_vals = check_conditions.copy()
            guar_vals['tracking_error'] = observed_tracking_error   # observed output
            g_ok, g_margins = active_contract.check_guarantees(guar_vals)
            if not g_ok:
                guarantee_violated = True
                logger.warning(
                    f"Guarantee violation: {active_ctrl} "
                    f"tracking={observed_tracking_error:.2f}m trend={error_trend:.1f}m/s "
                    f"margins={g_margins}"
                )

        # C: Predicted tracking error — choose quantitatively better controller
        pid_pred_error = float('inf')
        mpc_pred_error = float('inf')
        if pid_ok and pid_contract:
            preds = pid_contract.predict_guarantees(check_conditions)
            pid_pred_error = preds.get('tracking_error_max', float('inf'))
        if mpc_ok and mpc_contract:
            preds = mpc_contract.predict_guarantees(check_conditions)
            mpc_pred_error = preds.get('tracking_error_max', float('inf'))

        # --- Combine all signals into a recommendation ---
        # Policy: use the *most efficient* controller whose worst-case guarantee
        # still falls within the mission's acceptable tracking tolerance.
        # (Comparing raw predicted bounds always picks MPC, because MPC has
        # provably tighter guarantee constants. The right question is:
        # "can PID guarantee acceptable performance here?" — if yes, use PID.)
        ACCEPTABLE_TRACKING = 3.0   # metres — mission-level tolerance
        # Hysteresis: only downgrade when conditions drop to HYST * threshold.
        # Prevents chattering near boundaries (e.g. wind oscillating around 3 m/s).
        HYST = 0.70
        HYST_TRACKING = ACCEPTABLE_TRACKING * HYST   # 2.1m — downgrade threshold

        controller_priority = {'PID': 0, 'MPC': 1, 'Hinf': 2}
        active_priority = controller_priority.get(active_ctrl, 0)

        if guarantee_violated or error_rising_fast:
            # B/E: Active controller failing or error diverging → force escalation
            if active_priority < 1 and mpc_ok:
                recommended = 'MPC'
                logger.info(
                    f"Escalating PID→MPC: "
                    f"{'guarantee violated' if guarantee_violated else 'error trend'} "
                    f"(tracking={observed_tracking_error:.2f}m, trend={error_trend:.1f}m/s)"
                )
            else:
                recommended = 'Hinf'
        elif pid_ok and pid_pred_error < ACCEPTABLE_TRACKING:
            # A+C: PID can guarantee acceptable tracking — use it (most efficient)
            recommended = 'PID'
        elif mpc_ok and mpc_pred_error < ACCEPTABLE_TRACKING:
            # A+C: PID can't guarantee, but MPC can
            recommended = 'MPC'
            logger.info(
                f"PID→MPC: predicted PID={pid_pred_error:.2f}m exceeds tolerance "
                f"{ACCEPTABLE_TRACKING}m, MPC predicts {mpc_pred_error:.2f}m"
            )
        else:
            recommended = 'Hinf'

        # Hysteresis post-filter: suppress downgrades that are still close to the threshold.
        # Only escalation (recommended > active) passes without hysteresis check.
        if controller_priority.get(recommended, 0) < active_priority:
            if recommended == 'PID' and pid_pred_error > HYST_TRACKING:
                # PID predicted error still above hysteresis band — stay at current
                recommended = active_ctrl
            elif recommended == 'MPC' and mpc_pred_error > HYST_TRACKING:
                # MPC predicted error still above hysteresis band — stay at Hinf
                recommended = active_ctrl

        # D override: high EKF position uncertainty → prefer at least MPC
        if pos_uncertainty > 2.5 and controller_priority.get(recommended, 0) < 1:
            recommended = 'MPC'
            logger.info(f"EKF pos uncertainty {pos_uncertainty:.2f}m → escalating to MPC")

        # Emergency override: dual sensor failure → dead reckoning → must use H-inf
        if dual_sensor_failure:
            recommended = 'Hinf'
            logger.warning("Dual sensor failure (GPS + IMU) → forcing H-inf")

        # Horizon planner can further escalate (never downgrade)
        if planner_emergency:
            recommended = 'Hinf'
        elif planner_controller:
            planner_mapped = {'pid': 'PID', 'mpc': 'MPC', 'hinf': 'Hinf'}.get(
                planner_controller, planner_controller)
            if controller_priority.get(planner_mapped, 0) > controller_priority.get(recommended, 0):
                recommended = planner_mapped
                logger.info(f"Horizon planner: preemptive escalation to {recommended}")

        return recommended
