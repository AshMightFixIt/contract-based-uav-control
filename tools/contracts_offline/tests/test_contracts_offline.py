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
from contracts_offline import admissible as adm  # noqa: E402

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
# Envelope semantics (review M1)
# --------------------------------------------------------------------------
def _mid_point(rec):
    return {v: (lo + hi) / 2 for v, (lo, hi) in rec["envelope"].items()}


def test_envelope_semantics_are_explicit(built):
    _, doc, _ = built
    assert "per-input projection" in doc["envelope_semantics"]
    assert "sum(coefficients[v] * v) <= constant" in doc["assumption_term_form"]
    for rec in doc["composed"]:
        assert rec["assumptions"], rec["method"]
        if rec["sound"]:
            assert rec["envelope_is_admissible_set"] == (not rec["coupled_assumptions"])
            assert bool(rec["box_counterexamples"]) == (not rec["envelope_is_admissible_set"])
    flags = {(r["source"], r["controller"]): r["envelope_is_admissible_set"]
             for r in doc["composed"] if r["method"] == "pacti.compose"}
    assert flags == {("framework", "pid"): True, ("framework", "mpc"): False, ("framework", "hinf"): False,
                     ("pacti_library", "pid"): True, ("pacti_library", "mpc"): True,
                     ("pacti_library", "hinf"): True}


def test_no_unsound_reference_record_claims_an_admissible_set(built):
    """The repo's own SimpleContract.compose is unsound (K05); its records must not look usable."""
    files, doc, _ = built
    for rec in doc["composed"]:
        assert isinstance(rec["sound"], bool) and rec["composition"] and isinstance(rec["reference_only"], bool)
        if not rec["sound"]:
            assert rec["envelope_is_admissible_set"] is not True, (rec["source"], rec["controller"])
            assert rec["reference_only"] is True
    unsound = [r for r in doc["composed"] if r["method"] == "SimpleContract.compose"]
    assert len(unsound) == 3
    for rec in unsound:
        assert rec["sound"] is False and rec["envelope_is_admissible_set"] is False
        assert rec["composition"] == "SimpleContract.compose (unsound, reference only)"
    # only pacti.compose records are primary (sound, not reference-only)
    primary = {(r["source"], r["controller"]) for r in doc["composed"] if r["sound"] and not r["reference_only"]}
    assert primary == {(s, c) for s in ("framework", "pacti_library") for c in ("pid", "mpc", "hinf")}
    assert all(r["method"] == "pacti.compose" for r in doc["composed"] if r["sound"] and not r["reference_only"])
    assert "Only records with sound == true" in doc["envelope_semantics"]
    md = files["reconciliation.md"].decode("utf-8")
    assert "SimpleContract.compose (pre-flight today; unsound, reference only, NOT an admissible set)" in md
    assert "SimpleContract.compose (today; unsound, reference only)" in md


def test_box_corner_is_infeasible_against_full_composite_assumptions(built):
    """(hdop 4.667, wind 9.609) is inside the framework MPC envelope box but NOT admissible."""
    _, doc, _ = built
    rec = _composed(doc, "framework", "mpc")
    env = rec["envelope"]
    corner = dict(_mid_point(rec), gps_hdop=env["gps_hdop"][1], wind_speed=env["wind_speed"][1])
    assert corner["gps_hdop"] == pytest.approx(4.6667, abs=1e-4)
    assert corner["wind_speed"] == pytest.approx(9.609375)
    assert all(env[v][0] <= x <= env[v][1] for v, x in corner.items())  # a box check would accept it

    res = adm.check(rec, corner)
    assert not res["unchecked"] and not res["unknown_inputs"]
    assert [v["term"] for v in res["violated"]] == ["0.0525*gps_hdop + 0.08*wind_speed <= 0.795"]
    assert res["violated"][0]["lhs"] == pytest.approx(1.01375)
    assert adm.admissible(rec, corner) is False
    # admissible neighbours on the same constraint
    assert adm.admissible(rec, dict(corner, gps_hdop=0.5)) is True        # wind 9.609375 needs hdop 0.5
    assert adm.admissible(rec, dict(corner, wind_speed=6.875)) is True    # hdop 4.667 allows wind 6.875
    # H-inf: the same effect between wind and disturbance
    h = _composed(doc, "framework", "hinf")
    hc = dict(_mid_point(h), wind_speed=h["envelope"]["wind_speed"][1], disturbance=h["envelope"]["disturbance"][1])
    assert adm.admissible(h, hc) is False
    # the recorded counterexample is the same corner
    [cx] = rec["box_counterexamples"]
    assert cx["point"] == {"gps_hdop": env["gps_hdop"][1], "wind_speed": env["wind_speed"][1]}
    # fails closed when an input is missing
    with pytest.raises(ValueError):
        adm.admissible(rec, {"gps_hdop": 1.0, "wind_speed": 1.0})


def test_box_corner_is_infeasible_in_pacti_composite():
    """The same corner, checked on Pacti's own composed contract (unrounded, no JSON)."""
    from pacti.iocontract import Var

    from contracts_offline import compose, extract

    _, monitor = extract.import_framework(REPO_ROOT)
    _, lib = extract.import_pacti_library(REPO_ROOT)
    fw = {sc.name: compose.simple_contract_to_pacti(sc, []) for sc in extract.framework_contracts(monitor)}
    final = compose.build_pipelines(monitor, lib, fw)["framework"]["mpc"][-1][4]
    box = {v.name: compose.a_only_bounds(final, v.name) for v in final.inputvars}
    point = {v: (lo + hi) / 2 for v, (lo, hi) in box.items()}
    point.update(gps_hdop=box["gps_hdop"][1], wind_speed=box["wind_speed"][1])
    assert final.a.contains_behavior({Var(k): x for k, x in point.items()}) is False
    point.update(gps_hdop=0.5, wind_speed=9.6)
    assert final.a.contains_behavior({Var(k): x for k, x in point.items()}) is True


def test_admissible_cli(capsys):
    contracts = TOOL_DIR / "out" / "contracts.json"
    if not contracts.is_file():
        pytest.skip("no committed out/ yet")
    rc = adm.main(["framework", "mpc", "gps_hdop=4.6", "wind_speed=9.6", "--partial",
                   "--contracts", str(contracts)])
    out = capsys.readouterr().out
    violated = [line for line in out.splitlines() if line.startswith("VIOLATED")]
    assert rc == 1 and "NOT ADMISSIBLE" in out
    assert len(violated) == 1 and "0.0525*gps_hdop + 0.08*wind_speed <= 0.795" in violated[0]
    assert adm.main(["framework", "mpc", "gps_hdop=2", "wind_speed=8", "--partial",
                     "--contracts", str(contracts)]) == 0
    assert adm.main(["framework", "mpc", "gps_hdop=2", "--contracts", str(contracts)]) == 2


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


def _loc_of(relpath, needle):
    """Return "relpath:N" for the single line of relpath containing needle.

    Looking the line up by content keeps these checks exact without
    breaking whenever unrelated edits shift line numbers.
    """
    lines = (REPO_ROOT / relpath).read_text(encoding="utf-8").splitlines()
    hits = [i for i, line in enumerate(lines, 1) if needle in line]
    assert len(hits) == 1, f"{needle!r} found {len(hits)} times in {relpath}"
    return f"{relpath}:{hits[0]}"


def test_wind_rows_carry_all_three_copies(built):
    _, _, rows = built
    r = _rows(rows)
    mpc = r["MPC|wind_speed|A.upper"]
    assert (mpc["framework_value"], mpc["pacti_value"], mpc["status"]) == (15.0, 8.0, "mismatch")
    assert mpc["framework_loc"] == _loc_of(
        "src/contracts/contract_framework.py", '"wind_speed": (0.0, 15.0)')
    assert mpc["pacti_loc"] == _loc_of(
        "src/planning/pacti_contracts.py", "'wind_speed <= 8'")
    assert mpc["third_copy_value"] == "8" and mpc["third_copy_matches"] == "pacti_library"
    wind_limits = "wind_limits = {'pid': 3.0, 'mpc': 8.0, 'hinf': 15.0}"
    third = [loc.strip() for loc in mpc["third_copy_loc"].split(";")]
    assert _loc_of("src/planning/horizon_planner.py", wind_limits) in third
    assert _loc_of("src/planning/integrated_planner.py", wind_limits) in third
    assert r["HINF|wind_speed|A.upper"]["third_copy_matches"] == "pacti_library"
    assert r["PID|wind_speed|A.upper"]["third_copy_matches"] == "both"


def test_conversion_notes_cite_current_source_lines(built):
    _, doc, _ = built
    notes = {n["id"]: n["where"] for n in doc["conversion_notes"]}
    sensors = _loc_of("src/contracts/contract_framework.py", "self.sensor_contract = SimpleContract(")
    assert notes["C5-SENSORS"] == f"GPS_IMU_Sensors (cf:{sensors.rsplit(':', 1)[1]})"
    first = int(_loc_of("src/planning/pacti_contracts.py", "def compose_pipeline(").rsplit(":", 1)[1])
    cited_first, cited_last = (int(x) for x in notes["C6-PIPELINE"].removeprefix("pc:").split("-"))
    assert cited_first == first
    lines = (REPO_ROOT / "src/planning/pacti_contracts.py").read_text(encoding="utf-8").splitlines()
    body = lines[cited_first - 1:cited_last]
    # The cited range holds the whole method and nothing after it.
    assert sum(line.lstrip().startswith("def ") for line in body) == 1
    after = next((line for line in lines[cited_last:] if line.strip()), "")
    assert after.lstrip().startswith("def ") or not after.startswith(" " * 8)


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
    assert re.fullmatch(r"[0-9a-f]{64}", prov["inputs_sha256"])
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


def test_outputs_contain_no_git_commit_id(built):
    """Commit IDs change under cherry-pick/rebase/squash; out/ must not depend on them (review M2)."""
    files, doc, _ = built
    commit_id = re.compile(rb"\b[0-9a-f]{40}\b")  # a 64-hex sha256 does not match (no word boundary at 40)
    for name, data in files.items():
        assert not commit_id.search(data), name
    assert not any("commit" in k and k != "commit_policy" for k in _keys(doc["provenance"]))


def test_no_machine_specific_paths_in_tool():
    """Nothing committed under the tool (sources, README, out/) or its workflow names a local path."""
    # Built from parts so this file does not match itself.
    patterns = ["/" + "tmp" + "/", "scratch" + "pad", "/" + "root" + "/", "/" + "home" + "/", "$" + "S/"]
    files = [p for p in TOOL_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files.append(REPO_ROOT / ".github" / "workflows" / "contracts_offline.yml")
    hits = [(str(p.relative_to(REPO_ROOT)), pat) for p in files if p.is_file()
            for pat in patterns if pat in p.read_text(encoding="utf-8", errors="replace")]
    assert not hits


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
    if json.loads(committed.read_text(encoding="utf-8"))["environment"] != doc["environment"]:
        pytest.skip("installed pacti/numpy/scipy differ from the versions out/ was generated with")
    for name in cli.OUT_FILES:
        assert (TOOL_DIR / "out" / name).read_bytes() == files[name], (
            f"out/{name} is stale: run python tools/contracts_offline/run.py and commit out/")
