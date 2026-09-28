"""
Hierarchical Contract Framework for Adaptive Drone Control
Uses relational Assume-Guarantee contracts with linear constraints

Key Concepts:
- Every component has a contract (assumptions -> guarantees)
- Guarantees are RELATIONAL: output bounds depend on input values via equations
- Contracts compose algebraically through the pipeline
- Runtime monitoring evaluates predicted bounds

This module holds the contract definitions (define_contracts). The contract
types are in spec.py, composition and pre-flight checks in preflight.py, and
the runtime monitor, HierarchicalContractMonitor, in monitor.py.
"""

import logging

from .spec import LinearConstraint, SimpleContract

logger = logging.getLogger(__name__)


class ContractLibraryMixin:
    """Component contract definitions of HierarchicalContractMonitor (monitor.py).

    define_contracts() sets the monitor's sensor_contract, estimator_contract,
    controller_contracts and actuator_contract.
    """

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
