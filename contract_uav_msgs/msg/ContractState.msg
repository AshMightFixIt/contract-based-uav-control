# Full per-cycle telemetry from the contract-based adaptive controller.
# Designed for live monitoring (PlotJuggler / rqt_plot) and offline tuning (ros2 bag).
# All numeric fields are plottable; frame is NED (negative z = altitude above ground).

std_msgs/Header header
float64 controller_time        # internal controller clock (s)

# --- Flight management ---
string flight_mode             # track | hover | land | emergency
string active_controller       # PID | MPC | Hinf
bool cbf_intervened            # safety filter modified the command this cycle
int32 cbf_intervention_count   # cumulative interventions

# --- Estimated state (EKF), NED ---
float64[3] position            # x(N), y(E), z(D)  metres
float64[3] velocity            # vx, vy, vz        m/s
float64[3] attitude            # roll, pitch, yaw  rad

# --- Setpoint and tracking error ---
float64[3] setpoint_position   # commanded position (NED)
float64[3] position_error      # setpoint - actual
float64 position_error_norm    # |position_error|  (primary tuning signal)

# --- Commands sent to PX4 ---
float64 cmd_thrust             # normalized collective [0..1], 0.5 = hover
float64[3] cmd_desired_attitude # roll, pitch, yaw setpoint (rad)
float64[3] cmd_desired_rates   # body rate setpoint (rad/s)
float64[3] cmd_torques         # controller torque output (N*m, sim units)
float64[3] attitude_error      # desired_attitude - actual_attitude (rad)

# --- Horizon planner (Pacti) ---
bool planner_active
bool planner_emergency
float64 planner_safety_margin  # safety_limit - predicted_tracking_error
int32 planner_horizon

# --- Contract status (string enum per component) ---
# satisfied | assumptions_violated | guarantees_violated | both_violated | unknown
string contract_sensors
string contract_estimator
string contract_controller
string contract_actuators
