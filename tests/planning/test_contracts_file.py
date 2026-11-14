"""config/contracts.json: the planner's contracts, composed offline by Pacti (W4-09, K10, K11).

planning/export_contracts.py writes the file from PactiContractLibrary
(planning/pacti_contracts.py); the planner reads it at run time without Pacti.
These tests keep the file from drifting:

- without Pacti: the file records the sha256 of pacti_contracts.py, which must
  match the source as it is now; its 8 contract records must equal the
  pacti_library records of the contracts tool (tools/contracts_offline/out/);
  and the planner's numbers must be the ones it used before W4-09;
- with Pacti: a fresh export must equal the committed file, byte for byte.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("core_package")

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "contract_uav_core" / "contract_uav_core"
CONTRACTS_FILE = PACKAGE_DIR / "config" / "contracts.json"
SOURCE_FILE = PACKAGE_DIR / "planning" / "pacti_contracts.py"
TOOL_OUTPUT = REPO_ROOT / "tools" / "contracts_offline" / "out" / "contracts.json"
REGENERATE = 'python -c "from contract_uav_core.planning.export_contracts import main; main()"'

# The steady-state error parameters (wind coefficient, constant) that
# pacti_contracts.py hard-coded before W4-09 (controller_ss_params), and the
# decay that horizon_planner.py hard-coded (0.9 ** steps).
OLD_SS_PARAMS = {'pid': (0.30, 0.2), 'mpc': (0.15, 0.1), 'hinf': (0.50, 1.0)}
OLD_DECAY = 0.9


@pytest.fixture(scope="module")
def doc():
    return json.loads(CONTRACTS_FILE.read_text(encoding="utf-8"))


def test_file_matches_its_source(doc):
    """Any edit to pacti_contracts.py needs a new export (the sha256 is over the file with LF line ends)."""
    digest = hashlib.sha256(SOURCE_FILE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert doc["source"]["path"] == "contract_uav_core/planning/pacti_contracts.py"
    assert doc["source"]["sha256"] == digest, (
        f"config/contracts.json is stale: pacti_contracts.py changed since the last export. "
        f"In a venv with requirements/pacti.txt, run from the repository root:\n  {REGENERATE}")


def test_eight_records_equal_the_contracts_tool(doc):
    """The same 8 contracts, terms and input bounds as the tool's pacti_library records."""
    tool = json.loads(TOOL_OUTPUT.read_text(encoding="utf-8"))
    records = {r["contract"]: r for r in tool["contracts"] if r["source"] == "pacti_library"}
    assert sorted(doc["contracts"]) == sorted(records) == [
        'actuator', 'dynamics', 'ekf', 'gps', 'hinf', 'imu', 'mpc', 'pid']
    for name, rec in doc["contracts"].items():
        expected = {k: records[name][k] for k in rec}
        assert rec == expected, name


def test_planner_numbers_are_the_ones_used_before(doc):
    """Derived from the guarantees, the planner's numbers equal what the code hard-coded before W4-09."""
    planner = doc["planner"]
    assert planner["dynamics_decay"] == OLD_DECAY
    assert {c: (p["wind_speed"], p["constant"]) for c, p in planner["steady_state_error"].items()} == OLD_SS_PARAMS
    # Both terms the closed form leaves out are named (review R-09, N1).
    assert "estimation-error term is not used" in planner["derivation"]
    assert "its wind_speed term is not used either" in planner["derivation"]


def test_controller_input_bounds(doc):
    """What the planner checks before it plans with a controller (Pacti's get_variable_bounds)."""
    bounds = {c: doc["contracts"][c]["input_bounds"] for c in ('pid', 'mpc', 'hinf')}
    assert bounds == {
        'pid': {'initial_tracking_error': [0.0, 5.0], 'position_est_error': [0.0, 2.0],
                'velocity_est_error': [0.0, 0.5], 'wind_speed': [0.0, 3.0]},
        'mpc': {'initial_tracking_error': [0.0, 10.0], 'position_est_error': [0.0, 5.0],
                'velocity_est_error': [0.0, 2.0], 'wind_speed': [0.0, 8.0]},
        'hinf': {'initial_tracking_error': [0.0, 20.0], 'position_est_error': [0.0, 10.0],
                 'velocity_est_error': [0.0, 5.0], 'wind_speed': [0.0, 15.0]},
    }


@pytest.mark.pacti
def test_fresh_export_equals_committed_file(pacti_available):
    """Needs Pacti: compose again and compare with the committed file, byte for byte (LF line ends)."""
    if not pacti_available:
        pytest.skip("needs pacti: pip install -r requirements/pacti.txt")
    from contract_uav_core.planning import export_contracts

    fresh = export_contracts.render(export_contracts.build())
    committed = CONTRACTS_FILE.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert fresh == committed, f"config/contracts.json is stale; regenerate it:\n  {REGENERATE}"
