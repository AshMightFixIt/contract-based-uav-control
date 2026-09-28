"""
Pacti Contract Library for UAV Control System

Defines formal Assume-Guarantee contracts using Pacti's PolyhedralIoContract.
These contracts enable:
1. Formal composition across the control pipeline
2. Optimization of safety margins over horizons
3. Verification of mission feasibility
"""

from pacti.contracts import PolyhedralIoContract
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PactiContractLibrary:
    """
    Library of Pacti contracts for all system components.

    Each contract is a PolyhedralIoContract with:
    - Input variables (from environment or upstream components)
    - Output variables (to downstream components)
    - Assumptions (constraints on inputs)
    - Guarantees (promises about outputs given assumptions hold)
    """

    def __init__(self):
        self.contracts: Dict[str, PolyhedralIoContract] = {}
        # Steady-state error params (wind_coeff, constant) derived from controller guarantees.
        # e_ss(wind) = wind_coeff * wind_speed + constant
        self.controller_ss_params: Dict[str, Tuple[float, float]] = {
            'pid':  (0.30, 0.2),   # tracking_error <= ... + 0.30*wind + 0.2
            'mpc':  (0.15, 0.1),   # tracking_error <= ... + 0.15*wind + 0.1
            'hinf': (0.50, 1.0),   # tracking_error <= ... + 0.50*wind + 1.0
        }
        self.step_contract_cache: Dict[str, PolyhedralIoContract] = {}
        self.pipeline_cache: Dict[str, PolyhedralIoContract] = {}
        self._build_contracts()
        logger.info(f"PactiContractLibrary initialized with {len(self.contracts)} contracts, "
                    f"{len(self.step_contract_cache)} step contracts cached")

    def _build_contracts(self):
        """Build all component contracts."""
        self._build_sensor_contracts()
        self._build_estimator_contract()
        self._build_controller_contracts()
        self._build_actuator_contract()
        self._build_dynamics_contract()
        self._build_cached_contracts()

    def _build_sensor_contracts(self):
        """
        Sensor contracts: Environment -> Measurement quality
        """
        # GPS Contract
        self.contracts['gps'] = PolyhedralIoContract.from_strings(
            input_vars=['gps_satellites', 'gps_hdop'],
            output_vars=['position_meas_error', 'velocity_meas_error'],
            assumptions=[
                'gps_satellites >= 4',      # Minimum for 3D fix
                'gps_satellites <= 24',     # Max visible satellites
                'gps_hdop >= 0.5',          # Best case HDOP
                'gps_hdop <= 10',           # Worst acceptable HDOP
            ],
            guarantees=[
                'position_meas_error >= 0',
                'position_meas_error <= 0.5 * gps_hdop',  # Error scales with HDOP
                'velocity_meas_error >= 0',
                'velocity_meas_error <= 0.1 * gps_hdop',
            ]
        )

        # IMU Contract
        self.contracts['imu'] = PolyhedralIoContract.from_strings(
            input_vars=['imu_temp_deviation'],  # Deviation from calibration temp
            output_vars=['attitude_meas_error', 'rate_meas_error'],
            assumptions=[
                'imu_temp_deviation >= 0',
                'imu_temp_deviation <= 20',  # Max 20C from calibration
            ],
            guarantees=[
                'attitude_meas_error >= 0',
                'attitude_meas_error <= 0.02 + 0.005 * imu_temp_deviation',
                'rate_meas_error >= 0',
                'rate_meas_error <= 0.05 + 0.01 * imu_temp_deviation',
            ]
        )

    def _build_estimator_contract(self):
        """
        Estimator (EKF) contract: Measurement quality -> State estimate quality
        """
        self.contracts['ekf'] = PolyhedralIoContract.from_strings(
            input_vars=['position_meas_error', 'velocity_meas_error',
                       'attitude_meas_error', 'rate_meas_error'],
            output_vars=['position_est_error', 'velocity_est_error',
                        'attitude_est_error', 'rate_est_error'],
            assumptions=[
                'position_meas_error >= 0',
                'position_meas_error <= 5',
                'velocity_meas_error >= 0',
                'velocity_meas_error <= 1',
                'attitude_meas_error >= 0',
                'attitude_meas_error <= 0.2',
                'rate_meas_error >= 0',
                'rate_meas_error <= 0.3',
            ],
            guarantees=[
                # EKF reduces error through filtering
                'position_est_error >= 0',
                'position_est_error <= 0.7 * position_meas_error + 0.1',
                'velocity_est_error >= 0',
                'velocity_est_error <= 0.7 * velocity_meas_error + 0.05',
                'attitude_est_error >= 0',
                'attitude_est_error <= 0.5 * attitude_meas_error + 0.01',
                'rate_est_error >= 0',
                'rate_est_error <= 0.5 * rate_meas_error + 0.02',
            ]
        )

    def _build_controller_contracts(self):
        """
        Controller contracts: State error + disturbances -> Control quality
        """
        # PID Controller - efficient but limited operating envelope
        self.contracts['pid'] = PolyhedralIoContract.from_strings(
            input_vars=['position_est_error', 'velocity_est_error',
                       'wind_speed', 'initial_tracking_error'],
            output_vars=['tracking_error', 'control_effort', 'settling_time'],
            assumptions=[
                'position_est_error >= 0',
                'position_est_error <= 2',       # Good state estimate required
                'velocity_est_error >= 0',
                'velocity_est_error <= 0.5',
                'wind_speed >= 0',
                'wind_speed <= 3',               # Low wind only
                'initial_tracking_error >= 0',
                'initial_tracking_error <= 5',
            ],
            guarantees=[
                'tracking_error >= 0',
                # Tracking error bounded by estimation error + wind effect
                'tracking_error <= position_est_error + 0.3 * wind_speed + 0.2',
                'control_effort >= 0.2',
                'control_effort <= 0.8',
                'settling_time >= 1',
                'settling_time <= 5',
            ]
        )

        # MPC Controller - optimal, intermediate envelope
        self.contracts['mpc'] = PolyhedralIoContract.from_strings(
            input_vars=['position_est_error', 'velocity_est_error',
                       'wind_speed', 'initial_tracking_error'],
            output_vars=['tracking_error', 'control_effort', 'settling_time'],
            assumptions=[
                'position_est_error >= 0',
                'position_est_error <= 5',       # Wider than PID (model handles more)
                'velocity_est_error >= 0',
                'velocity_est_error <= 2',
                'wind_speed >= 0',
                'wind_speed <= 8',               # Intermediate envelope
                'initial_tracking_error >= 0',
                'initial_tracking_error <= 10',
            ],
            guarantees=[
                'tracking_error >= 0',
                # MPC optimizes: better than PID (α<1), narrower than H-inf
                'tracking_error <= 0.3 * position_est_error + 0.15 * wind_speed + 0.1',
                'control_effort >= 0.1',
                'control_effort <= 0.7',
                'settling_time >= 1.5',
                'settling_time <= 7',
            ]
        )

        # H-infinity Controller - robust but conservative
        self.contracts['hinf'] = PolyhedralIoContract.from_strings(
            input_vars=['position_est_error', 'velocity_est_error',
                       'wind_speed', 'initial_tracking_error'],
            output_vars=['tracking_error', 'control_effort', 'settling_time'],
            assumptions=[
                # H-inf has much weaker assumptions (almost always applicable)
                'position_est_error >= 0',
                'position_est_error <= 10',
                'velocity_est_error >= 0',
                'velocity_est_error <= 5',
                'wind_speed >= 0',
                'wind_speed <= 15',              # Handles high wind
                'initial_tracking_error >= 0',
                'initial_tracking_error <= 20',
            ],
            guarantees=[
                'tracking_error >= 0',
                # Worse tracking but guaranteed stable
                'tracking_error <= position_est_error + 0.5 * wind_speed + 1.0',
                'control_effort >= 0.3',
                'control_effort <= 0.9',
                'settling_time >= 3',
                'settling_time <= 10',
            ]
        )

    def _build_actuator_contract(self):
        """
        Actuator contract: Control commands -> Actual forces/torques
        """
        self.contracts['actuator'] = PolyhedralIoContract.from_strings(
            input_vars=['control_effort', 'battery_voltage', 'motor_temp'],
            output_vars=['thrust_accuracy', 'torque_accuracy'],
            assumptions=[
                'control_effort >= 0',
                'control_effort <= 1',
                'battery_voltage >= 10.5',       # Min safe voltage (3S LiPo)
                'battery_voltage <= 12.6',       # Max voltage
                'motor_temp >= 10',
                'motor_temp <= 80',              # Max safe temp
            ],
            guarantees=[
                # Accuracy degrades with voltage drop and temperature
                'thrust_accuracy >= 0.9',
                'thrust_accuracy <= 1.1',
                'torque_accuracy >= 0.9',
                'torque_accuracy <= 1.1',
            ]
        )

    def _build_dynamics_contract(self):
        """
        Dynamics contract: One timestep evolution
        Maps current state + control to next state bounds
        """
        self.contracts['dynamics'] = PolyhedralIoContract.from_strings(
            input_vars=['tracking_error', 'control_effort', 'wind_speed', 'dt'],
            output_vars=['next_tracking_error', 'position_change'],
            assumptions=[
                'tracking_error >= 0',
                'tracking_error <= 10',
                'control_effort >= 0',
                'control_effort <= 1',
                'wind_speed >= 0',
                'wind_speed <= 15',
                'dt >= 0.01',
                'dt <= 0.1',
            ],
            guarantees=[
                'next_tracking_error >= 0',
                # Error evolves based on control and disturbances
                # Good control reduces error, wind increases it
                'next_tracking_error <= 0.9 * tracking_error + 0.1 * wind_speed',
                'position_change >= 0',
                'position_change <= 2 * dt',  # Max 2 m/s movement
            ]
        )

    def _build_cached_contracts(self):
        """Pre-compute commonly used contract compositions for fast horizon planning."""
        dynamics = self.contracts.get('dynamics')
        gps = self.contracts.get('gps')
        ekf = self.contracts.get('ekf')

        for ctrl_name in ['pid', 'mpc', 'hinf']:
            ctrl = self.contracts.get(ctrl_name)

            # Step contract: ctrl ∘ dynamics (used every timestep in _try_plan)
            try:
                self.step_contract_cache[ctrl_name] = ctrl.compose(dynamics)
                logger.info(f"Cached step contract: {ctrl_name} ∘ dynamics")
            except Exception as e:
                logger.warning(f"Failed to cache step contract for {ctrl_name}: {e}")

            # Pipeline contract: gps ∘ ekf ∘ ctrl (full sensor-to-control chain)
            try:
                self.pipeline_cache[ctrl_name] = gps.compose(ekf).compose(ctrl)
                logger.info(f"Cached pipeline contract: gps ∘ ekf ∘ {ctrl_name}")
            except Exception as e:
                logger.warning(f"Failed to cache pipeline for {ctrl_name}: {e}")

    def get_step_contract(self, controller: str) -> Optional[PolyhedralIoContract]:
        """Return the pre-computed ctrl ∘ dynamics contract, or None if unavailable."""
        return self.step_contract_cache.get(controller)

    def get_controller_ss_error(self, controller: str, wind_speed: float) -> float:
        """
        Compute the steady-state tracking error bound for a given controller and wind speed.

        Derived from controller contract guarantees:
            e_ss = wind_coeff * wind_speed + constant
        """
        wind_coeff, constant = self.controller_ss_params.get(
            controller, self.controller_ss_params['hinf']  # fallback to most conservative
        )
        return wind_coeff * wind_speed + constant

    def get_contract(self, name: str) -> Optional[PolyhedralIoContract]:
        """Get a contract by name."""
        return self.contracts.get(name)

    def compose_pipeline(self, controller: str = 'pid') -> PolyhedralIoContract:
        """
        Compose the full sensing-to-actuation pipeline.

        GPS + IMU -> EKF -> Controller -> Actuator

        Returns the composed contract showing:
        - System-level assumptions (on environment)
        - System-level guarantees (on performance)
        """
        if controller not in ['pid', 'mpc', 'hinf']:
            raise ValueError(f"Unknown controller: {controller}")

        # Start with sensor contracts
        # We need to merge GPS and IMU first (they're parallel)
        gps = self.contracts['gps']
        imu = self.contracts['imu']

        # Compose sequentially through the pipeline
        ekf = self.contracts['ekf']
        ctrl = self.contracts[controller]
        actuator = self.contracts['actuator']

        # GPS -> EKF (position/velocity path)
        # IMU -> EKF (attitude/rate path)
        # For simplicity, compose GPS with EKF first
        pipeline = gps.compose(ekf)

        # Then compose with controller
        pipeline = pipeline.compose(ctrl)

        # Finally compose with actuator
        pipeline = pipeline.compose(actuator)

        logger.info(f"Composed pipeline with {controller} controller")
        logger.debug(f"Pipeline assumptions: {pipeline.a}")
        logger.debug(f"Pipeline guarantees: {pipeline.g}")

        return pipeline

    def get_safety_bounds(self, contract: PolyhedralIoContract,
                          variable: str) -> Tuple[float, float]:
        """
        Get the bounds on a variable from a contract's guarantees.
        Uses Pacti's optimization to find tight bounds.
        """
        try:
            bounds = contract.get_variable_bounds(variable)
            return bounds
        except Exception as e:
            logger.warning(f"Could not get bounds for {variable}: {e}")
            return (float('-inf'), float('inf'))


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Pacti Contract Library Test")
    print("=" * 60)

    library = PactiContractLibrary()

    print("\n=== Available Contracts ===")
    for name in library.contracts:
        print(f"  - {name}")

    print("\n=== GPS Contract ===")
    print(library.contracts['gps'])

    print("\n=== PID Controller Contract ===")
    print(library.contracts['pid'])

    print("\n=== Composing Pipeline with PID ===")
    try:
        pipeline = library.compose_pipeline('pid')
        print(f"Input vars: {pipeline.inputvars}")
        print(f"Output vars: {pipeline.outputvars}")

        # Get bounds on tracking error
        bounds = library.get_safety_bounds(pipeline, 'tracking_error')
        print(f"\nTracking error bounds: {bounds}")
    except Exception as e:
        print(f"Composition error: {e}")

    print("\n=== Composing Pipeline with MPC ===")
    try:
        pipeline_mpc = library.compose_pipeline('mpc')
        print(f"Input vars: {pipeline_mpc.inputvars}")
        print(f"Output vars: {pipeline_mpc.outputvars}")
        bounds = library.get_safety_bounds(pipeline_mpc, 'tracking_error')
        print(f"Tracking error bounds (MPC): {bounds}")
    except Exception as e:
        print(f"Composition error: {e}")

    print("\n=== Composing Pipeline with H-inf ===")
    try:
        pipeline_hinf = library.compose_pipeline('hinf')
        bounds = library.get_safety_bounds(pipeline_hinf, 'tracking_error')
        print(f"Tracking error bounds (H-inf): {bounds}")
    except Exception as e:
        print(f"Composition error: {e}")

    print("\n" + "=" * 60)
    print("Contract Library Test Complete")
    print("=" * 60)
