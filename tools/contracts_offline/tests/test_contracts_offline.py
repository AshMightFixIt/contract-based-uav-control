"""Tests for tools/contracts_offline.

Run from the repo root:  python -m pytest -p no:cacheprovider tools/contracts_offline/tests
The whole module is skipped when pacti is not installed.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pacti", reason="pacti is not installed; pip install -r tools/contracts_offline/requirements.txt")

TOOL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_DIR.parents[1]
sys.dont_write_bytecode = True
if str(TOOL_DIR.parent) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR.parent))

from contracts_offline import SCHEMA_VERSION, cli  # noqa: E402

# Pinned: one row per (component, variable, bound-or-coefficient) at commit 057743f.
# If the contracts change, regenerate out/ and update this number deliberately.
EXPECTED_ROWS = 153


@pytest.fixture(scope="module")
def built():
    return cli.build(REPO_ROOT)


def _composed(doc, source, ctrl, method="pacti.compose"):
    recs = [r for r in doc["composed"] if (r["source"], r["controller"], r["method"]) == (source, ctrl, method)]
    assert len(recs) == 1
    return recs[0]


def _rows(rows):
    return {r["row_id"]: r for r in rows}


# --------------------------------------------------------------------------
# Reproductions
# --------------------------------------------------------------------------
def test_reproduces_f_a1_01_framework_constants_composed_with_pacti(built):
    """F-A1-01: the framework's own constants, composed soundly, give PID hdop <= 2.7619, MPC <= 4.6667."""
    _, doc, _ = built
    pid = _composed(doc, "framework", "pid")["envelope"]["gps_hdop"]
    mpc = _composed(doc, "framework", "mpc")["envelope"]["gps_hdop"]
    assert pid[1] == pytest.approx(2.7619, abs=1e-4)
    assert mpc[1] == pytest.approx(4.6667, abs=1e-4)
    assert pid[0] == pytest.approx(0.5) and mpc[0] == pytest.approx(0.5)
    # SimpleContract.compose (what pre-flight uses) keeps the sensor's hdop <= 5: the K05 gap.
    assert _composed(doc, "framework", "pid", "SimpleContract.compose")["envelope"]["gps_hdop"][1] == 5.0


def test_reproduces_probe_m1_bounds_from_pacti_library(built):
    """probe_m1_bounds: Pacti-library MPC wind [0, 8], H-inf wind [0, 15] (and PID [0, 3])."""
    _, doc, _ = built
    contracts = {(r["source"], r["contract"]): r for r in doc["contracts"]}
    assert contracts[("pacti_library", "mpc")]["input_bounds"]["wind_speed"] == [0.0, 8.0]
    assert contracts[("pacti_library", "hinf")]["input_bounds"]["wind_speed"] == [0.0, 15.0]
    assert contracts[("pacti_library", "pid")]["input_bounds"]["wind_speed"] == [0.0, 3.0]
    for method in ("pacti.compose", "PactiContractLibrary.compose_pipeline"):
        assert _composed(doc, "pacti_library", "mpc", method)["envelope"]["wind_speed"] == [0.0, 8.0]
        assert _composed(doc, "pacti_library", "hinf", method)["envelope"]["wind_speed"] == [0.0, 15.0]


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------
def test_row_count_and_f_a1_09_rollup(built):
    files, doc, rows = built
    s = doc["reconciliation"]
    assert len(rows) == s["rows"] == EXPECTED_ROWS
    assert s["match"] + s["mismatch"] + s["framework_only"] + s["pacti_only"] == EXPECTED_ROWS
    assert files["reconciliation.csv"].decode("utf-8").count("\n") == EXPECTED_ROWS + 1
    roll = s["f_a1_09_rollup"]
    assert (roll["groups"], roll["match"], roll["not_match"]) == (55, 15, 40)
    assert roll["agrees_with_f_a1_09"]


def test_wind_rows_carry_all_three_copies(built):
    _, _, rows = built
    r = _rows(rows)
    mpc = r["MPC|wind_speed|A.upper"]
    assert (mpc["framework_value"], mpc["pacti_value"], mpc["status"]) == (15.0, 8.0, "mismatch")
    assert mpc["framework_loc"] == "src/contracts/contract_framework.py:695"
    assert mpc["pacti_loc"] == "src/planning/pacti_contracts.py:154"
    assert mpc["third_copy_value"] == "8" and mpc["third_copy_matches"] == "pacti_library"
    assert "src/planning/horizon_planner.py:368" in mpc["third_copy_loc"]
    assert "src/planning/integrated_planner.py:210" in mpc["third_copy_loc"]
    assert r["HINF|wind_speed|A.upper"]["third_copy_matches"] == "pacti_library"
    assert r["PID|wind_speed|A.upper"]["third_copy_matches"] == "both"


def test_effect_column_semantics(built):
    _, _, rows = built
    r = _rows(rows)
    assert r["PID|position_error|A.upper"]["effect_framework"] == "pid: cut (upstream max 5.35)"
    assert r["MPC|wind_speed|A.upper"]["effect_framework"] == "mpc: slack (composed hi 9.609375)"
    assert r["MPC|wind_speed|A.upper"]["effect_pacti"] == "mpc: binds (composed hi 8)"
    assert "mpc: cut" in r["ACTUATOR|control_effort|A.upper"]["effect_framework"]
    assert r["PID|tracking_error|G.upper.const"]["effect_framework"] == "pid: not assumed downstream"


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------
def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_contracts_json_schema_and_value_agnostic(built):
    files, doc, _ = built
    assert json.loads(files["contracts.json"]) == doc
    assert doc["schema_version"] == SCHEMA_VERSION
    prov = doc["provenance"]
    assert "source_commit" in prov and re.fullmatch(r"[0-9a-f]{64}", prov["inputs_sha256"])
    assert doc["environment"]["pacti"] == "0.3.1"
    assert {r["source"] for r in doc["composed"]} == {"framework", "pacti_library"}
    for rec in doc["composed"]:
        assert set(rec["envelope"]) == set(rec["input_vars"])
    forbidden = {"authoritative", "winner", "preferred", "selected", "chosen"}
    assert not forbidden & set(_keys(doc))


def test_outputs_have_no_timestamps(built):
    files, _, _ = built
    stamp = re.compile(rb"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
    for name, data in files.items():
        assert not stamp.search(data), name
        assert b"\r\n" not in data, name


def test_two_runs_are_byte_identical(tmp_path, built):
    files, _, _ = built
    outs = []
    for seed in ("1", "4242"):
        out = tmp_path / f"run{seed}"
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, str(TOOL_DIR / "run.py"), "--out", str(out)],
                       check=True, env=env, cwd=str(REPO_ROOT), capture_output=True)
        outs.append({n: (out / n).read_bytes() for n in cli.OUT_FILES})
    assert outs[0] == outs[1]
    assert outs[0] == files


def test_committed_outputs_are_current(built):
    files, doc, _ = built
    committed = TOOL_DIR / "out" / "contracts.json"
    if not committed.is_file():
        pytest.skip("no committed out/ yet")
    if doc["provenance"]["source_commit"] is None:
        pytest.skip("git history unavailable (shallow clone?)")
    if json.loads(committed.read_text(encoding="utf-8"))["environment"] != doc["environment"]:
        pytest.skip("installed pacti/numpy/scipy differ from the versions out/ was generated with")
    for name in cli.OUT_FILES:
        assert (TOOL_DIR / "out" / name).read_bytes() == files[name], (
            f"out/{name} is stale: run python tools/contracts_offline/run.py and commit out/")
