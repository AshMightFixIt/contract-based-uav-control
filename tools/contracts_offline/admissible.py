"""Exact point check against a composed record of ``out/contracts.json``.

Standard library only (no Pacti), so a consumer can copy or import it.

Why it exists: ``composed[].envelope`` is a per-input projection (each input's
range with the others free). When a record has ``coupled_assumptions``, that box
is larger than the admissible set; e.g. for framework MPC the box corner
``gps_hdop = 4.667, wind_speed = 9.609`` violates
``0.0525*gps_hdop + 0.08*wind_speed <= 0.795``. The admissible set is exactly
the conjunction of ``composed[].assumptions``, each term meaning
``sum(coefficients[v] * v) <= constant``. Check points against those terms, not
against the box.

Library use::

    rec = load_record("tools/contracts_offline/out/contracts.json", "framework", "mpc")
    admissible(rec, {"gps_hdop": 2.0, "wind_speed": 8.0, ...})   # all inputs required

CLI (exit 0 admissible, 1 violated, 2 incomplete point or bad usage)::

    python tools/contracts_offline/admissible.py framework mpc gps_hdop=4.6 wind_speed=9.6 --partial
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Terms in contracts.json are rounded to 10 significant digits, so allow a
# relative slack well above that rounding (about 5e-10) and far below any
# engineering tolerance.
REL_TOL = 1e-9
DEFAULT_CONTRACTS = Path(__file__).resolve().parent / "out" / "contracts.json"


def load_record(path, source: str, controller: str, method: str = "pacti.compose") -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    for rec in doc["composed"]:
        if (rec["source"], rec["controller"], rec["method"]) == (source, controller, method):
            return rec
    raise KeyError(f"no composed record for source={source!r} controller={controller!r} method={method!r}")


def check(record: dict, point: Dict[str, float], rel_tol: float = REL_TOL) -> dict:
    """Evaluate every composed assumption term at ``point``.

    Returns ``{"violated": [...], "unchecked": [...], "unknown_inputs": [...]}``. A term is
    unchecked when the point lacks one of its variables.
    """
    violated: List[dict] = []
    unchecked: List[dict] = []
    for term in record["assumptions"]:
        coefs = term["coefficients"]
        missing = sorted(v for v in coefs if v not in point)
        if missing:
            unchecked.append({"term": term["text"], "missing": missing})
            continue
        lhs = sum(float(a) * float(point[v]) for v, a in coefs.items())
        k = float(term["constant"])
        if lhs > k + rel_tol * max(1.0, abs(k)):
            # lhs/constant are in the raw term form sum(a*v) <= constant; "variable" is set for
            # single-variable terms, whose text is shown as a plain bound.
            violated.append({"term": term["text"], "lhs": lhs, "constant": k,
                             "variable": next(iter(coefs)) if len(coefs) == 1 else None})
    unknown = sorted(set(point) - set(record["input_vars"]))
    return {"violated": violated, "unchecked": unchecked, "unknown_inputs": unknown}


def admissible(record: dict, point: Dict[str, float], rel_tol: float = REL_TOL, partial: bool = False) -> bool:
    """True iff ``point`` satisfies every composed assumption term.

    Fails closed: raises ValueError when a term cannot be evaluated because the point
    lacks a variable, unless ``partial=True`` (then only the evaluable terms count).
    """
    res = check(record, point, rel_tol)
    if res["unchecked"] and not partial:
        missing = sorted({v for u in res["unchecked"] for v in u["missing"]})
        raise ValueError(f"point lacks inputs {missing}; pass them or use partial=True")
    return not res["violated"]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Check a point against a composed record's full assumptions.")
    p.add_argument("source", choices=("framework", "pacti_library"))
    p.add_argument("controller", choices=("pid", "mpc", "hinf"))
    p.add_argument("values", nargs="+", metavar="var=value")
    p.add_argument("--method", default="pacti.compose")
    p.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    p.add_argument("--partial", action="store_true", help="check only the terms whose variables are given")
    args = p.parse_args(argv)
    try:
        point = {k: float(v) for k, v in (item.split("=", 1) for item in args.values)}
    except ValueError:
        p.error("values must look like var=number")
    rec = load_record(args.contracts, args.source, args.controller, args.method)
    res = check(rec, point)
    for v in res["violated"]:
        if v.get("variable"):
            print(f"VIOLATED  {v['term']}   ({v['variable']} = {point[v['variable']]:.10g})")
        else:
            print(f"VIOLATED  {v['term']}   (left side {v['lhs']:.10g} > {v['constant']:.10g})")
    for u in res["unchecked"]:
        print(f"unchecked {u['term']}   (missing {', '.join(u['missing'])})")
    if res["unknown_inputs"]:
        print(f"unknown inputs for this record: {', '.join(res['unknown_inputs'])}")
    if res["unchecked"] and not args.partial:
        print("INCOMPLETE: give every input, or pass --partial")
        return 2
    print("ADMISSIBLE" if not res["violated"] else "NOT ADMISSIBLE")
    return 0 if not res["violated"] else 1


if __name__ == "__main__":
    sys.exit(main())
