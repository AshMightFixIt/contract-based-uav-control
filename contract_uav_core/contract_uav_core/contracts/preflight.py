"""
Pre-flight contract verification.

- ContractPreflightMixin gives HierarchicalContractMonitor (monitor.py) its
  compose_pipeline and verify_mission_feasibility, moved from
  contract_framework.py, and pipeline_inputs.
- ControllerPreflightMixin gives AdaptiveDroneController (core.py) its
  pre_flight_check, moved from adaptive_control_system.py.
- check_telemetry checks the telemetry dict pre-flight reads (W4-09, A-F11).

The telemetry dict (A-F11, core half; W4-13 builds it in the ROS node from PX4
messages) is flat, keyed by the pipelines' input names (pipeline_inputs):
gps_satellites (count), gps_hdop, imu_temperature (degC), imu_calibrated (1 or
0, or a bool), battery_voltage (V, the whole pack), motor_temperature (degC),
wind_speed (m/s), disturbance and, optionally, computation_time (s; 0.01 is
used for the MPC check when it is absent). Values are numbers; None means "no
reading".

pre_flight_check(initial_conditions, mission, airframe=None):
- reports missing inputs by name, warns about keys it does not use (and
  ignores them), and rejects NaN, infinite and non-numeric values. Any missing
  or rejected input makes pre-flight fail before any controller is tried;
- with an airframe (a name such as 'dexi', or a config.airframe.Airframe), it
  checks battery_voltage against the airframe's pack voltage range
  (Airframe.battery_pack_voltage_range, e.g. DEXI 19.8-25.2 V) instead of the
  actuator contract's 3-cell 11.0-12.6 V, and it does not check the inputs the
  airframe lists as unavailable (e.g. motor_temperature): they are not
  required, a value given for one is ignored with a warning (it cannot come
  from the vehicle), and the predictions that need one are left out. Without
  an airframe every input is checked against the contracts, as before.
"""

import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import logging

from ..config.airframe import Airframe, load_airframe
from .spec import SimpleContract

logger = logging.getLogger(__name__)

# Inputs pre-flight fills in when they are absent: name -> value. computation_time
# is the MPC's solve time, which is only known in flight.
OPTIONAL_INPUTS = {'computation_time': 0.01}


@dataclass(frozen=True)
class TelemetryCheck:
    """What check_telemetry made of a telemetry dict.

    conditions: the values pre-flight checks (known, valid, not unavailable), as floats.
    missing: required inputs that are absent or None.
    invalid: (input, why) for NaN, infinite and non-numeric values.
    unknown: keys pre-flight does not use; ignored.
    not_checked: inputs the airframe lists as unavailable.
    ignored: unavailable inputs that came with a value anyway; ignored.
    assumption_overrides: input -> (lo, hi) that replaces the contracts' range
        (battery_voltage -> the airframe's pack range).
    airframe: the airframe's name, or None.
    """
    conditions: Mapping[str, float]
    missing: Tuple[str, ...] = ()
    invalid: Tuple[Tuple[str, str], ...] = ()
    unknown: Tuple[str, ...] = ()
    not_checked: Tuple[str, ...] = ()
    ignored: Tuple[str, ...] = ()
    assumption_overrides: Mapping[str, Tuple[float, float]] = None
    airframe: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.missing and not self.invalid

    def problems(self) -> str:
        """Why the telemetry is rejected ('' if it is not)."""
        parts = []
        if self.missing:
            parts.append(f"missing: {', '.join(self.missing)}")
        if self.invalid:
            parts.append("invalid: " + ", ".join(f"{name} ({why})" for name, why in self.invalid))
        return "; ".join(parts)

    def notes(self) -> str:
        """What pre-flight left out or checked differently ('' if nothing)."""
        parts = []
        if self.unknown:
            parts.append(f"ignored unknown inputs: {', '.join(self.unknown)}")
        if self.not_checked:
            given = f" (value given and ignored: {', '.join(self.ignored)})" if self.ignored else ""
            parts.append(f"not checked (unavailable on {self.airframe}): {', '.join(self.not_checked)}{given}")
        for name, (lo, hi) in sorted((self.assumption_overrides or {}).items()):
            parts.append(f"{name} checked against {self.airframe}'s range {lo:g}-{hi:g}")
        return "; ".join(parts)


def _number(value: Any) -> Tuple[Optional[float], Optional[str]]:
    """(float value, None), or (None, why it is not a usable number)."""
    if isinstance(value, (bool, np.bool_)):
        return float(value), None
    if not isinstance(value, numbers.Real):
        return None, f"not a number: {type(value).__name__}"
    x = float(value)
    if math.isnan(x):
        return None, "NaN"
    if math.isinf(x):
        return None, "infinite"
    return x, None


def resolve_airframe(airframe: Union[None, str, Airframe]) -> Optional[Airframe]:
    """An Airframe from a packaged airframe's name (config/airframes/<name>.yaml), or as given."""
    if airframe is None or isinstance(airframe, Airframe):
        return airframe
    if isinstance(airframe, str):
        return load_airframe(airframe)
    raise TypeError(f"airframe must be None, a name or an Airframe, not {type(airframe).__name__}")


def check_telemetry(telemetry: Mapping[str, Any],
                    inputs: Iterable[str],
                    airframe: Union[None, str, Airframe] = None) -> TelemetryCheck:
    """Check a telemetry dict against the pipelines' inputs (and an optional airframe)."""
    if not isinstance(telemetry, Mapping):
        raise TypeError(f"telemetry must be a mapping of input name to value, not {type(telemetry).__name__}")
    inputs = tuple(sorted(inputs))
    frame = resolve_airframe(airframe)
    unavailable = set(frame.unavailable) if frame is not None else set()
    overrides = {}
    if frame is not None and 'battery_voltage' in inputs:
        overrides['battery_voltage'] = frame.battery_pack_voltage_range

    conditions, missing, invalid, not_checked, ignored = {}, [], [], [], []
    for name in inputs:
        value = telemetry.get(name)
        if name in unavailable:
            not_checked.append(name)
            if value is not None:
                ignored.append(name)
            continue
        if value is None:
            if name not in OPTIONAL_INPUTS:
                missing.append(name)
            continue
        number, why = _number(value)
        if why is not None:
            invalid.append((name, why))
        else:
            conditions[name] = number
    unknown = sorted(str(k) for k in telemetry if k not in inputs)
    return TelemetryCheck(
        conditions=conditions, missing=tuple(missing), invalid=tuple(invalid), unknown=tuple(unknown),
        not_checked=tuple(not_checked), ignored=tuple(ignored), assumption_overrides=overrides,
        airframe=frame.name if frame is not None else None)


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

    def pipeline_inputs(self) -> Tuple[str, ...]:
        """The inputs pre-flight reads: the assumption variables of every controller's pipeline, sorted."""
        names = set()
        for controller_name in self.controller_contracts:
            pipeline = self.compose_pipeline(controller_name)
            if pipeline is not None:
                names.update(pipeline.assumptions)
        return tuple(sorted(names))

    def verify_mission_feasibility(self,
                                   controller_name: str,
                                   current_conditions: Dict[str, float],
                                   *,
                                   assumption_overrides: Optional[Mapping[str, Tuple[float, float]]] = None,
                                   not_checked: Iterable[str] = ()) -> Tuple[bool, str]:
        """
        BEFORE TAKEOFF: Check if mission is feasible given current conditions.
        Uses contract composition to verify mission feasibility AND predict performance.

        assumption_overrides: input -> (lo, hi) checked instead of the pipeline's range
            (pre-flight: battery_voltage -> the airframe's pack range).
        not_checked: inputs whose assumptions are not checked (unavailable on the airframe).
        The predictions leave out the guarantees that need an overridden or unchecked input.
        """
        pipeline = self.compose_pipeline(controller_name)

        if pipeline is None:
            return False, "Pipeline composition failed"

        overrides = dict(assumption_overrides or {})
        skipped = set(not_checked)
        predict_conditions = current_conditions
        if overrides or skipped:
            assumptions = {var: tuple(overrides.get(var, bounds))
                           for var, bounds in pipeline.assumptions.items() if var not in skipped}
            pipeline = SimpleContract(pipeline.name, assumptions, pipeline.guarantees)
            predict_conditions = {k: v for k, v in current_conditions.items()
                                  if k not in skipped and k not in overrides}

        # Check if current conditions satisfy pipeline assumptions
        assumptions_met, margins = pipeline.check_assumptions(current_conditions)

        if not assumptions_met:
            violated = [k for k, v in margins.items() if v < 0]
            return False, f"Assumptions violated: {violated}"

        min_margin = min(margins.values()) if margins else 0.0

        # Compute predicted performance from relational guarantees
        predictions = pipeline.predict_guarantees(predict_conditions)
        pred_items = sorted(predictions.items())
        pred_str = ", ".join(f"{k}={v:.3f}" for k, v in pred_items)

        return True, (f"Mission feasible with {controller_name} "
                      f"(margin: {min_margin:.2f}, predicted: {pred_str})")


class ControllerPreflightMixin:
    """Pre-flight check of AdaptiveDroneController: picks the first feasible controller."""

    def pre_flight_check(self,
                        initial_conditions: Dict[str, float],
                        mission: Dict[str, any],
                        *,
                        airframe: Union[None, str, Airframe] = None) -> Tuple[bool, str]:
        """
        Pre-flight contract verification.
        Uses contract composition to verify mission is feasible.
        Sets up the supervisor with the mission target.

        initial_conditions: the telemetry dict (see this module's docstring).
        airframe: optional airframe name or Airframe: battery checked against its
            pack range, its unavailable inputs not checked.
        """
        logger.info("=" * 60)
        logger.info("PRE-FLIGHT CONTRACT VERIFICATION")
        logger.info("=" * 60)

        target = mission.get('target_position', np.array([0.0, 0.0, -5.0]))
        waypoints = mission.get('waypoints', [target])

        telemetry = check_telemetry(initial_conditions, self.contract_monitor.pipeline_inputs(), airframe)
        notes = telemetry.notes()
        if notes:
            logger.warning(f"Pre-flight: {notes}")
        suffix = f"; {notes}" if notes else ""
        if not telemetry.ok:
            msg = f"Pre-flight inputs rejected: {telemetry.problems()}"
            logger.error(msg)
            self.mission_feasible = False
            return False, msg + suffix

        # Try controllers in order: PID (efficient) → MPC (optimal) → H-inf (robust)
        controllers = [
            ('PID', "efficient PID"),
            ('MPC', "optimal MPC"),
            ('Hinf', "robust H-infinity"),
        ]

        failures = []
        for i, (name, desc) in enumerate(controllers, 1):
            logger.info(f"\n[{i}/{len(controllers)}] Checking {desc}...")
            # MPC pipeline needs computation_time in conditions
            check_conditions = dict(telemetry.conditions)
            if name == 'MPC' and 'computation_time' not in check_conditions:
                check_conditions['computation_time'] = OPTIONAL_INPUTS['computation_time']  # typical solve time

            feasible, msg = self.contract_monitor.verify_mission_feasibility(
                name, check_conditions,
                assumption_overrides=telemetry.assumption_overrides, not_checked=telemetry.not_checked,
            )

            if feasible:
                logger.info(f"{name} feasible: {msg}")
                if name != 'PID':
                    self.controller_switcher.switch_to(
                        name, 0.0, f"Pre-flight: conditions require {desc}")
                self.supervisor.set_waypoints(waypoints)
                self.mission_feasible = True
                perf_note = "" if name == 'PID' else " (degraded performance expected)"
                return True, f"Mission feasible with {desc}{perf_note}{suffix}"
            else:
                logger.warning(f"{name} infeasible: {msg}")
                failures.append(f"{name}: {msg}")

        logger.error("MISSION INFEASIBLE - all controller contracts violated!")
        self.mission_feasible = False
        return False, f"Mission infeasible - all controller contracts violated ({'; '.join(failures)}){suffix}"
