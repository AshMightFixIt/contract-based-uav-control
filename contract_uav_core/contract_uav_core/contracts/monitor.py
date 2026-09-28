"""
Runtime contract monitor: HierarchicalContractMonitor.

Moved from contract_framework.py. The runtime parts (monitor_runtime,
get_contract_status, should_switch_controller, export_metrics) are here; the
contract definitions come from library.py and compose_pipeline /
verify_mission_feasibility from preflight.py.

Self-test: python -m contract_uav_core.contracts.monitor
"""

from typing import Dict, List, Tuple, Optional
import logging

from .library import ContractLibraryMixin
from .preflight import ContractPreflightMixin
from .spec import PACTI_AVAILABLE, ContractMetrics, ContractStatus

logger = logging.getLogger(__name__)


class HierarchicalContractMonitor(ContractLibraryMixin, ContractPreflightMixin):
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
