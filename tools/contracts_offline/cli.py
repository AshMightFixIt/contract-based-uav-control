"""Command line entry: build the outputs, then write them or check them against out/."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from . import SCHEMA_VERSION, TOOL_NAME, TOOL_VERSION

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_OUT = PACKAGE_DIR / "out"
OUT_FILES = ("contracts.json", "reconciliation.csv", "reconciliation.md")


def _static_notes(fw_bounds, flags: List[dict]) -> List[dict]:
    from .model import FRAMEWORK_PATH, fmt

    notes = [
        {"id": "C1-IO", "source": "framework", "where": "all contracts",
         "text": "SimpleContract declares no input/output lists. Converted with inputs = assumption "
                 "variables plus every variable used in a guarantee coefficient, and outputs = guarantee "
                 "output variables."},
        {"id": "C2-FORM", "source": "framework", "where": "all contracts",
         "text": "Assumption `var in (lo, hi)` became `-var <= -lo` and `var <= hi`. Guarantee "
                 "`y <= sum(c*x) + k` became `y - sum(c*x) <= k`; `y >= sum(c*x) + k` became "
                 "`-y + sum(c*x) <= -k`. Numbers are passed to Pacti as floats (no string round trip). "
                 "Pacti's constructor simplifies guarantees against assumptions (simplify=True, the "
                 "default the Pacti library also uses)."},
    ]
    multi = [b for b in fw_bounds if b.kind == "G" and len(b.coefficients) >= 2]
    if multi:
        where = ", ".join(sorted({b.loc.replace(FRAMEWORK_PATH, 'cf') for b in multi}))
        notes.append({
            "id": "C3-MULTI", "source": "framework", "where": where,
            "text": f"{len(multi)} guarantees have coefficients on several inputs. Each became ONE "
                    "half-space over all of its inputs, which is the relational meaning of "
                    "LinearConstraint when every input is present. Flag: SimpleContract.evaluate_bound "
                    "treats a missing input as 0 (F-A1-04) while Pacti quantifies over it, so the two "
                    "differ only when an input is missing at run time. No input was dropped, split or "
                    "bounded separately."})
    eq = sorted({(b.component, b.var, b.loc) for b in fw_bounds if b.kind == "A"
                 and any(o.kind == "A" and o.var == b.var and o.contract == b.contract
                         and o.direction != b.direction and o.constant == b.constant for o in fw_bounds)})
    for comp, var, loc in eq:
        val = next(b.constant for b in fw_bounds if b.var == var and b.kind == "A" and b.component == comp)
        notes.append({"id": "C4-EQUALITY", "source": "framework", "where": loc.replace(FRAMEWORK_PATH, "cf"),
                      "text": f"{comp} assumes `{var}` in [{fmt(val)}, {fmt(val)}], a degenerate interval. "
                              "Converted to two inequalities (an equality). It stays a box bound in the "
                              "envelope."})
    notes += [
        {"id": "C5-SENSORS", "source": "framework", "where": "GPS_IMU_Sensors (cf:461)",
         "text": "The framework has one sensor contract for GPS and IMU. It is composed as one stage, as "
                 "compose_pipeline does; its rows are split into GPS and IMU by variable for the table."},
        {"id": "C6-PIPELINE", "source": "pacti_library", "where": "pc:252-290",
         "text": "The Pacti-library pipeline is built here as (gps || imu) -> ekf -> controller -> "
                 "actuator, to match the framework's sensors -> estimator -> controller -> actuators. "
                 "PactiContractLibrary.compose_pipeline leaves imu out (F-A1-25); that variant is kept "
                 "in contracts.json as a cross-check. The dynamics contract is in neither pipeline; its "
                 "rows are Pacti-only with no effect on the composed assumptions."},
        {"id": "C7-ALIAS", "source": "both", "where": "comparison layer",
         "text": "Rows are lined up through a renaming layer that the repo does not have (model.ALIASES). "
                 "Judgement call: H-inf `stabilization_time` (framework) is compared with H-inf "
                 "`settling_time` (Pacti library). `imu_temperature` and `imu_temp_deviation` are NOT "
                 "aliased because they measure different things. Every renamed row says so in its note."},
        {"id": "C8-ENVELOPE", "source": "both", "where": "section 1",
         "text": "Envelope bounds are linear programs over the composed assumptions only. They are "
                 "cross-checked against Pacti's get_variable_bounds (assumptions and guarantees), which "
                 "the planner and the Wave-2 probes use; any difference is flagged as P-AGDIFF."},
    ]
    return notes + sorted(flags, key=lambda f: (f["id"], f["source"], f["where"], f["text"]))


def _tightenings(bounds_by_source, composed_primary) -> List[dict]:
    from .compose import STAGE_COMPONENTS
    from .model import CONTROLLER_COMPONENT, close

    out = []
    for rec in composed_primary:
        src, ctrl = rec["source"], rec["controller"]
        comps = set(STAGE_COMPONENTS["sensors"]) | set(STAGE_COMPONENTS["estimator"]) \
            | set(STAGE_COMPONENTS["actuator"]) | {CONTROLLER_COMPONENT[ctrl]}
        for var, (lo, hi) in sorted(rec["envelope"].items()):
            own = [b for b in bounds_by_source[src] if b.kind == "A" and b.var == var and b.component in comps]
            terms = [t["text"] for t in rec["assumptions"] if var in t["coefficients"]]
            ups = sorted((b.constant, b.loc) for b in own if b.direction == "upper")
            los = sorted(((b.constant, b.loc) for b in own if b.direction == "lower"), reverse=True)
            if ups and hi is not None and hi < ups[0][0] and not close(hi, ups[0][0]):
                out.append({"source": src, "controller": ctrl, "variable": var, "direction": "upper",
                            "component_bound": ups[0][0], "component_loc": ups[0][1],
                            "composed_bound": hi, "composed_terms": terms})
            if los and lo is not None and lo > los[0][0] and not close(lo, los[0][0]):
                out.append({"source": src, "controller": ctrl, "variable": var, "direction": "lower",
                            "component_bound": los[0][0], "component_loc": los[0][1],
                            "composed_bound": lo, "composed_terms": terms})
    return out


def _bound_record(b) -> dict:
    from .model import rnd

    return {
        "source": b.source, "component": b.component, "contract": b.contract, "kind": b.kind,
        "variable": b.var, "direction": b.direction,
        "coefficients": {n: rnd(c) for n, c in b.coefficients},
        "constant": rnd(b.constant), "loc": b.loc,
        "via": {k: list(v) for k, v in b.via if v},
    }


def build(repo_root: Path) -> Tuple[Dict[str, bytes], dict, List[dict]]:
    """Compute everything; return ({file name: bytes}, contracts.json document, rows)."""
    from . import compose, extract, provenance, reconcile, render
    from .model import (ALIASES, COMPONENT_ORDER, FRAMEWORK_CONTRACT_COMPONENT, FRAMEWORK_PATH,
                        FRAMEWORK_SENSOR_CONTRACT, NOT_ALIASED, PACTI_CONTRACT_COMPONENT, PACTI_LIB_PATH,
                        SIG_DIGITS, rnd)

    repo_root = repo_root.resolve()
    prev_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)  # the libraries log at INFO; keep the tool quiet
        _, monitor = extract.import_framework(repo_root)
        _, lib = extract.import_pacti_library(repo_root)

        flags: List[dict] = []
        fw_bounds = extract.framework_bounds(repo_root, monitor, flags)
        pc_bounds = extract.pacti_bounds(repo_root, lib, flags)
        third = extract.third_copy(repo_root)

        fw_pacti = {sc.name: compose.simple_contract_to_pacti(sc, flags)
                    for sc in extract.framework_contracts(monitor)}
        pipelines = compose.build_pipelines(monitor, lib, fw_pacti)

        primary = []
        notes = {
            "framework": "framework constants (define_contracts) converted to PolyhedralIoContract and "
                         "composed with Pacti: sensors -> estimator -> controller -> actuators",
            "pacti_library": "PactiContractLibrary contracts composed with Pacti: (gps || imu) -> ekf -> "
                             "controller -> actuator",
        }
        for src in ("framework", "pacti_library"):
            for ctrl, stages in pipelines[src].items():
                primary.append(compose.composed_record(src, ctrl, stages, "pacti.compose", notes[src], flags))
        reference = compose.reference_records(monitor, lib)

        envelopes = {src: {r["controller"]: r["envelope"] for r in primary if r["source"] == src}
                     for src in ("framework", "pacti_library")}
        bounds_by_source = {"framework": fw_bounds, "pacti_library": pc_bounds}
        effects = {}
        for src in ("framework", "pacti_library"):
            raw = compose.analyse_effects(bounds_by_source[src], pipelines[src], envelopes[src])
            effects[src] = {b: compose.summarise_effect(v, b.component) for b, v in raw.items()}

        rows = reconcile.build_rows(fw_bounds, pc_bounds, third, effects, flags)
        summary = reconcile.summarise(rows)

        contracts = compose.contract_records(
            "framework", fw_pacti,
            lambda n: "GPS+IMU" if n == FRAMEWORK_SENSOR_CONTRACT else FRAMEWORK_CONTRACT_COMPONENT.get(n, n))
        contracts += compose.contract_records(
            "pacti_library", lib.contracts, lambda n: PACTI_CONTRACT_COMPONENT.get(n, n.upper()))

        def bound_sort(b):
            comp = COMPONENT_ORDER.index(b.component) if b.component in COMPONENT_ORDER else 99
            return (b.source, comp, b.contract, b.kind, b.var, b.direction, b.loc)

        composed = sorted(primary + reference, key=lambda r: (r["source"], r["controller"], r["method"]))
        tightenings = _tightenings(bounds_by_source, primary)
        for t in tightenings:
            for k in ("component_bound", "composed_bound"):
                t[k] = rnd(t[k])

        doc = {
            "schema_version": SCHEMA_VERSION,
            "generator": {"name": TOOL_NAME, "version": TOOL_VERSION},
            "value_policy": render.VALUE_POLICY,
            "provenance": provenance.provenance(repo_root),
            "environment": provenance.environment(),
            "float_format": f"rounded to {SIG_DIGITS} significant digits; null = unbounded",
            "sources": {
                "framework": {"path": FRAMEWORK_PATH, "module": "contracts.contract_framework",
                              "objects": "HierarchicalContractMonitor().define_contracts()"},
                "pacti_library": {"path": PACTI_LIB_PATH, "module": "planning.pacti_contracts",
                                  "objects": "PactiContractLibrary().contracts"},
            },
            "variable_aliases": [{"component": c, "pacti_library": p, "framework": f, "reason": w}
                                 for c, p, f, w in ALIASES],
            "not_aliased": [{"component": c, "framework": f, "pacti_library": p, "reason": w}
                            for c, f, p, w in NOT_ALIASED],
            "bounds": [_bound_record(b) for b in sorted(fw_bounds + pc_bounds, key=bound_sort)],
            "contracts": contracts,
            "composed": composed,
            "tightenings": tightenings,
            "third_copy": third,
            "reconciliation": {**summary, "files": ["reconciliation.csv", "reconciliation.md"],
                               "columns": list(render.CSV_COLUMNS)},
            "conversion_notes": _static_notes(fw_bounds, flags),
        }
        files = {
            "contracts.json": render.contracts_json(doc),
            "reconciliation.csv": render.reconciliation_csv(rows),
            "reconciliation.md": render.reconciliation_md(doc, rows),
        }
        return files, doc, rows
    finally:
        logging.disable(prev_disable)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="contracts_offline", description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT,
                        help="repository root (default: two levels above this package)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default: %(default)s)")
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the files in --out differ from a fresh build")
    args = parser.parse_args(argv)

    if importlib.util.find_spec("pacti") is None:
        print("contracts_offline: pacti is not installed. Install the tool's requirements:\n"
              "  python -m pip install -r tools/contracts_offline/requirements.txt", file=sys.stderr)
        return 2

    files, doc, _ = build(args.repo_root)
    from .provenance import last_input_commit

    # Console only, never written to out/ (see provenance.py).
    print(f"contracts_offline: inputs sha256 {doc['provenance']['inputs_sha256']}; last commit touching "
          f"the inputs (log only): {last_input_commit(args.repo_root.resolve()) or 'unknown'}")
    if args.check:
        stale = [n for n in OUT_FILES if not (args.out / n).is_file() or (args.out / n).read_bytes() != files[n]]
        for n in stale:
            print(f"contracts_offline: {args.out / n} is out of date", file=sys.stderr)
        return 1 if stale else 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name in OUT_FILES:
        (args.out / name).write_bytes(files[name])
    s = doc["reconciliation"]
    print(f"contracts_offline: wrote {', '.join(OUT_FILES)} to {args.out}")
    print(f"  rows {s['rows']}: match {s['match']}, mismatch {s['mismatch']}, "
          f"framework_only {s['framework_only']}, pacti_only {s['pacti_only']}")
    return 0
