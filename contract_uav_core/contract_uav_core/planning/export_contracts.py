"""
Offline export of the planner's contracts to config/contracts.json (W4-09, K10, K11).

Design time only. This module imports Pacti (requirements/pacti.txt); the flight
code never imports it. The horizon planner (horizon_planner.py) reads the file
this module writes, through contracts/envelope.py, and checks the same bounds
with plain Python, so Pacti is not needed at run time.

What it does:
- builds PactiContractLibrary() (pacti_contracts.py, the planner's contract
  source: gps, imu, ekf, pid, mpc, hinf, actuator, dynamics);
- writes one record per contract (8 records): its input and output variables,
  its assumption and guarantee terms as Pacti holds them, and the bounds of
  every input from Pacti's get_variable_bounds (the call the planner used to
  make on every replan). The records have the same form and the same numbers as
  the pacti_library records of tools/contracts_offline/out/contracts.json;
- derives the two numbers the planner's closed-form error bound needs: each
  controller's steady-state error (the wind coefficient and the constant of its
  tracking_error guarantee) and the dynamics decay (the tracking_error
  coefficient of the dynamics guarantee);
- records the sha256 of pacti_contracts.py (CRLF read as LF), so a test without
  Pacti can tell that the file is stale.

Every number is rounded to 10 significant digits (as the contracts tool does).
The output is deterministic: no time stamps, keys sorted.

Regenerate after any change to planning/pacti_contracts.py, in a venv with
requirements/pacti.txt (from the repository root):
    python -c "from contract_uav_core.planning.export_contracts import main; main()"
Check without writing (exit status 1 when the committed file is stale):
    python -c "import sys; from contract_uav_core.planning.export_contracts import main; sys.exit(main(['--check']))"
"""

import argparse
import hashlib
import json
import math
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .pacti_contracts import PactiContractLibrary

SCHEMA_VERSION = 1
CONTROLLERS = ('pid', 'mpc', 'hinf')
SIG_DIGITS = 10

PACKAGE_DIR = Path(__file__).resolve().parent.parent
SOURCE_FILE = PACKAGE_DIR / 'planning' / 'pacti_contracts.py'
OUTPUT_FILE = PACKAGE_DIR / 'config' / 'contracts.json'
SOURCE_NAME = 'contract_uav_core/planning/pacti_contracts.py'


# --------------------------------------------------------------------------
# Numbers and terms, rendered as tools/contracts_offline/model.py does
# --------------------------------------------------------------------------
def rnd(x: Optional[float]) -> Optional[float]:
    """Round to SIG_DIGITS significant digits; None for None, NaN and inf; no negative zero."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    v = float(f"{x:.{SIG_DIGITS}g}")
    return 0.0 if v == 0 else v


def _fmt(x: float) -> str:
    return f"{rnd(x):.{SIG_DIGITS}g}"


def _linear_text(coefficients: Sequence[Tuple[str, float]]) -> str:
    parts = []
    for name, c in sorted(coefficients):
        if rnd(c) == 0:
            continue
        mag = _fmt(abs(c))
        term = name if mag == "1" else f"{mag}*{name}"
        if not parts:
            parts.append(term if c > 0 else f"-{term}")
        else:
            parts.append(("+ " if c > 0 else "- ") + term)
    return " ".join(parts) if parts else "0"


def term_text(coefficients: Sequence[Tuple[str, float]], constant: float) -> str:
    """``sum(a*v) <= k``; a single-variable term renders as a bound on that variable."""
    coefs = [(n, c) for n, c in sorted(coefficients) if rnd(c) != 0]
    if len(coefs) == 1:
        name, a = coefs[0]
        return f"{name} <= {_fmt(constant / a)}" if a > 0 else f"{name} >= {_fmt(constant / a)}"
    return f"{_linear_text(coefs)} <= {_fmt(constant)}"


def term_record(term) -> dict:
    """One Pacti term as {coefficients, constant, text}, meaning sum(coefficients[v] * v) <= constant."""
    coefs = sorted((v.name, float(c)) for v, c in term.variables.items())
    return {
        "coefficients": {n: rnd(c) for n, c in coefs},
        "constant": rnd(term.constant),
        "text": term_text(coefs, float(term.constant)),
    }


def sorted_terms(termlist) -> List[dict]:
    """The terms of a Pacti term list, deduplicated and sorted (as the contracts tool sorts them)."""
    recs = [term_record(t) for t in termlist.terms]
    uniq = {(r["text"], tuple(sorted(r["coefficients"].items())), r["constant"]): r for r in recs}
    return [uniq[k] for k in sorted(uniq, key=lambda k: (len(k[1]), [n for n, _ in k[1]], k[0]))]


def _names(variables) -> List[str]:
    return sorted(v.name for v in variables)


# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------
def contract_record(contract) -> dict:
    """Input/output variables, terms, and Pacti's bounds of every input (get_variable_bounds)."""
    bounds = {}
    for var in _names(contract.inputvars):
        lo, hi = contract.get_variable_bounds(var)
        bounds[var] = [rnd(lo), rnd(hi)]
    return {
        "input_vars": _names(contract.inputvars),
        "output_vars": _names(contract.outputvars),
        "assumptions": sorted_terms(contract.a),
        "guarantees": sorted_terms(contract.g),
        "input_bounds": bounds,
    }


def _upper_bound_on(record: dict, output: str, contract: str) -> Dict[str, float]:
    """The one guarantee ``output <= sum(c*x) + k`` of a record, as {x: c, ..., 'constant': k}."""
    found = [t for t in record["guarantees"] if t["coefficients"].get(output, 0.0) > 0]
    if len(found) != 1:
        raise ValueError(f"{contract}: expected one upper bound on {output}, found {len(found)}")
    term = found[0]
    a = term["coefficients"][output]
    bound = {v: rnd(-c / a) for v, c in term["coefficients"].items() if v != output}
    bound["constant"] = rnd(term["constant"] / a)
    return bound


def planner_record(contracts: Dict[str, dict]) -> dict:
    """The numbers of the planner's closed-form bound e_t = d^t e_0 + (1 - d^t) e_ss(wind)."""
    steady = {}
    for ctrl in CONTROLLERS:
        bound = _upper_bound_on(contracts[ctrl], "tracking_error", ctrl)
        steady[ctrl] = {"wind_speed": bound.get("wind_speed", 0.0), "constant": bound["constant"]}
    dynamics = _upper_bound_on(contracts["dynamics"], "next_tracking_error", "dynamics")
    return {
        "steady_state_error": steady,
        "dynamics_decay": dynamics["tracking_error"],
        "derivation": (
            "steady_state_error[c]: the wind_speed coefficient and the constant of controller c's "
            "guarantee tracking_error <= ... (its estimation-error term is not used); "
            "dynamics_decay: the tracking_error coefficient of the dynamics guarantee "
            "next_tracking_error <= ... (its wind_speed term is not used either: the closed form "
            "takes the wind's effect from each controller's e_ss instead, so that term would count "
            "the wind twice); e_ss(wind) = wind_speed * wind + constant. "
            "The planner left out both terms before W4-09 too"),
    }


def source_sha256(path: Path = SOURCE_FILE) -> str:
    """sha256 of a source file with CRLF read as LF (so a Windows checkout gives the same hash)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build() -> dict:
    """Build the document (needs Pacti)."""
    lib = PactiContractLibrary()
    contracts = {name: contract_record(lib.contracts[name]) for name in sorted(lib.contracts)}
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "contract_uav_core.planning.export_contracts",
        "source": {"path": SOURCE_NAME, "objects": "PactiContractLibrary().contracts",
                   "sha256": source_sha256()},
        "environment": {"pacti": metadata.version("pacti")},
        "term_form": ("each term means sum(coefficients[v] * v) <= constant; input_bounds[v] is "
                      "Pacti's get_variable_bounds(v), [min, max]; numbers are rounded to "
                      f"{SIG_DIGITS} significant digits"),
        "contracts": contracts,
        "planner": planner_record(contracts),
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export the planner's contracts to config/contracts.json.")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if the committed file differs from a fresh export")
    args = parser.parse_args(argv)
    text = render(build())
    if args.check:
        committed = OUTPUT_FILE.read_text(encoding="utf-8").replace("\r\n", "\n") if OUTPUT_FILE.is_file() else None
        if committed != text:
            print(f"export_contracts: {OUTPUT_FILE.name} is out of date; regenerate it", file=sys.stderr)
            return 1
        print(f"export_contracts: {OUTPUT_FILE.name} is up to date")
        return 0
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"export_contracts: wrote {OUTPUT_FILE.name} ({len(text)} bytes)")
    return 0
