"""The horizon planner runs without Pacti (W4-09, T1; K10, K11).

The planner reads config/contracts.json (ContractBounds) and never imports
Pacti. Before W4-09 it imported Pacti at module level, so the core turned it
off when Pacti was missing (core.py, HORIZON_PLANNER_AVAILABLE).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")

# Run in a child process with Pacti blocked, as if it were not installed:
# sys.modules['pacti'] = None makes every "import pacti..." raise ImportError.
BLOCKED_PACTI_RUN = r"""
import json, sys
sys.modules['pacti'] = None
import numpy as np
from contract_uav_core.core import AdaptiveDroneController
from contract_uav_core.planning.integrated_planner import IntegratedPlanner

ctrl = AdaptiveDroneController(dt=0.02, use_horizon_planner=True)
conditions = {'gps_satellites': 12.0, 'gps_hdop': 1.0, 'imu_temperature': 25.0, 'imu_calibrated': 1.0,
              'battery_voltage': 12.4, 'motor_temperature': 25.0, 'wind_speed': 1.0, 'disturbance': 0.2}
target = np.array([10.0, 0.0, -5.0])
ok, msg = ctrl.pre_flight_check(conditions, {'target_position': target, 'waypoints': [target]})
ctrl.start_mission()
statuses, replans = [], 0
for k in range(60):
    raw = {'gps': {'position': np.array([0.02 * k, 0.0, -5.0]), 'velocity': np.array([1.0, 0.0, 0.0]),
                   'valid': True, 'satellites': 12, 'hdop': 1.0},
           'imu': {'attitude': np.zeros(3), 'rates': np.zeros(3), 'valid': True, 'calibrated': True,
                   'temperature': 25.0},
           'battery': {'voltage': 12.4}, 'motors': {'temperature': 30.0}}
    _, tel = ctrl.control_step(raw)
    statuses.append(tel['planner']['plan_status'])
    replans += bool(tel['planner']['replanned'])

planner = IntegratedPlanner()
windy = [planner.update(t, {'position': np.array([0.0, 0.0, -5.0]), 'velocity': np.zeros(3),
                            'attitude': np.zeros(3)}, {'wind_speed': w}, np.array([5.0, 0.0, -5.0]))[0]
         for t, w in ((0.0, 1.0), (0.5, 5.0), (1.0, 12.0))]
print(json.dumps({
    'preflight': ok, 'planner_on': ctrl.use_horizon_planner, 'planner': type(ctrl.horizon_planner).__name__,
    'statuses': sorted(set(statuses)), 'replans': replans, 'windy': windy,
    'pacti_modules': sorted(m for m in sys.modules if m == 'pacti' or m.startswith('pacti.')
                            if sys.modules[m] is not None),
    'pacti_contracts_imported': 'contract_uav_core.planning.pacti_contracts' in sys.modules,
}))
"""


def test_planner_runs_with_pacti_blocked(run_python, tmp_path):
    proc = run_python(["-c", BLOCKED_PACTI_RUN], cwd=tmp_path, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out['preflight'] is True
    assert out['planner_on'] is True and out['planner'] == 'IntegratedPlanner'
    assert out['statuses'] == ['feasible'] and out['replans'] >= 2   # 60 steps of 0.02 s, every 0.5 s
    assert out['windy'] == ['pid', 'mpc', 'hinf']                    # wind 1, 5, 12 m/s
    assert out['pacti_modules'] == [] and out['pacti_contracts_imported'] is False


def test_planner_modules_do_not_import_pacti(run_python, tmp_path):
    """Even where Pacti is installed, importing and running the planner loads none of it."""
    code = (
        "import json, sys, numpy as np\n"
        "from contract_uav_core.planning.integrated_planner import IntegratedPlanner\n"
        "p = IntegratedPlanner()\n"
        "p.update(0.0, {'position': np.array([0.0, 0.0, -5.0])}, {'wind_speed': 2.0}, np.zeros(3))\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] == 'pacti' "
        "or m.endswith('pacti_contracts') or m.endswith('export_contracts'))))\n"
    )
    proc = run_python(["-c", code], cwd=tmp_path, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []


def test_contract_bounds_come_from_the_file():
    from contract_uav_core.contracts.envelope import contract_bounds
    from contract_uav_core.planning.horizon_planner import ContractBounds, HorizonPlanner

    bounds = HorizonPlanner().contracts
    assert isinstance(bounds, ContractBounds)
    doc = contract_bounds()
    for ctrl in ('pid', 'mpc', 'hinf'):
        assert bounds.input_bounds[ctrl] == {v: tuple(b) for v, b in doc['contracts'][ctrl]['input_bounds'].items()}
    assert bounds.decay == 0.9
    assert bounds.steady_state_error('pid', 2.0) == 0.3 * 2.0 + 0.2

    # A changed file changes the planner's check: MPC's wind limit 8 -> 9 m/s.
    doc['contracts']['mpc']['input_bounds']['wind_speed'] = [0.0, 9.0]
    values = {'position_est_error': 0.5, 'velocity_est_error': 0.1, 'wind_speed': 8.5,
              'initial_tracking_error': 0.0}
    assert bounds.violated_assumption('mpc', values) == 'wind_speed'
    assert ContractBounds(doc).violated_assumption('mpc', values) is None


@pytest.mark.parametrize('ctrl', ['pid', 'mpc', 'hinf'])
def test_planner_safety_trigger_uses_the_wind_envelope(ctrl):
    """IntegratedPlanner._safety_critical replans when the wind exceeds the plan's controller limit."""
    from contract_uav_core.contracts.envelope import wind_envelope
    from contract_uav_core.planning.horizon_planner import Plan, PlanStatus
    from contract_uav_core.planning.integrated_planner import IntegratedPlanner

    limit = wind_envelope()[ctrl]
    planner = IntegratedPlanner()
    planner.state.current_plan = Plan(horizon_length=10, steps=[], status=PlanStatus.FEASIBLE,
                                      min_safety_margin=4.0, recommended_controller=ctrl)
    drone = {'position': np.array([0.0, 0.0, -5.0]), 'velocity': np.zeros(3), 'attitude': np.zeros(3)}
    assert not planner._safety_critical(drone, {'wind_speed': limit})
    assert planner._safety_critical(drone, {'wind_speed': limit + 0.01})


@pytest.mark.pacti
def test_bounds_equal_what_pacti_gives(pacti_available):
    """Needs Pacti: the planner's bounds are Pacti's get_variable_bounds, exactly (the old runtime call)."""
    if not pacti_available:
        pytest.skip("needs pacti: pip install -r requirements/pacti.txt")
    from contract_uav_core.planning.horizon_planner import ContractBounds
    from contract_uav_core.planning.pacti_contracts import PactiContractLibrary

    lib, bounds = PactiContractLibrary(), ContractBounds()
    for ctrl in ('pid', 'mpc', 'hinf'):
        contract = lib.get_contract(ctrl)
        pacti = {str(v): tuple(contract.get_variable_bounds(str(v))) for v in contract.inputvars}
        assert bounds.input_bounds[ctrl] == pacti, ctrl
