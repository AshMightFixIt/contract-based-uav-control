"""Deterministic writers: sorted keys, fixed float formatting, LF line endings, no timestamps."""

from __future__ import annotations

import csv
import io
import json
from typing import Dict, List

from . import SCHEMA_VERSION, TOOL_NAME, TOOL_VERSION
from .model import (
    ALIASES,
    CONTROLLER_COMPONENT,
    CONTROLLERS,
    NOT_ALIASED,
    SHORT_PATH,
    SOURCE_LABEL,
    canonical_var,
    close,
    fmt,
    fmt_interval,
)

CSV_COLUMNS = (
    "row", "row_id", "component", "variable", "variable_framework", "variable_pacti", "kind", "direction",
    "item", "framework_value", "framework_loc", "framework_via", "pacti_value", "pacti_loc",
    "third_copy_value", "third_copy_loc", "third_copy_matches", "status", "a1_row",
    "effect_framework", "effect_pacti", "note",
)

VALUE_POLICY = ("value-agnostic: both libraries are recorded side by side under their own 'source'; "
                "no value is selected as authoritative. Choosing the values is the user's decision (DC).")


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------
def contracts_json(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return fmt(value)
    return str(value)


def reconciliation_csv(rows: List[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for r in rows:
        w.writerow([_cell(r[c]) for c in CSV_COLUMNS])
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------
def _short(loc: str) -> str:
    if not loc:
        return ""
    parts = []
    for one in loc.split("; "):
        path, _, line = one.rpartition(":")
        parts.append(f"{SHORT_PATH.get(path, path)}:{line}")
    return "; ".join(parts)


def _val(value, loc: str) -> str:
    if value is None:
        return "—"
    return f"{fmt(value)} ({_short(loc)})" if loc else fmt(value)


def _interval(pair) -> str:
    if pair is None:
        return "—"
    return fmt_interval(pair[0], pair[1])


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def _canon_map(record: dict, key: str, component: str) -> Dict[str, list]:
    src = record["source"]
    return {canonical_var(component, src, v): b for v, b in record.get(key, {}).items()}


def _glance_section(doc: dict, lines: List[str]) -> None:
    """Upper bounds of wind_speed and gps_hdop per controller, every source in one table."""
    comp = {(r["source"], r["controller"], r["method"]): r for r in doc["composed"]}
    own = {}
    for b in doc["bounds"]:
        if b["kind"] == "A" and b["direction"] == "upper":
            own.setdefault((b["source"], b["component"], b["variable"]), []).append((b["constant"], b["loc"]))
    comps_of = {c: ("GPS", "IMU", "EKF", CONTROLLER_COMPONENT[c], "ACTUATOR") for c in CONTROLLERS}

    def own_hi(src, ctrl, var):
        vals = sorted(v for comp_ in comps_of[ctrl] for v in own.get((src, comp_, var), []))
        return f"{fmt(vals[0][0])} ({_short(vals[0][1])})" if vals else "—"

    def env_hi(src, ctrl, method, var):
        rec = comp[(src, ctrl, method)]
        pair = rec["envelope"].get(var)
        if pair is None:
            return "—"
        text = fmt(pair[1]) if pair[1] is not None else "+inf"
        if "assumptions" in rec:
            single = [t["constant"] / t["coefficients"][var] for t in rec["assumptions"]
                      if list(t["coefficients"]) == [var] and t["coefficients"][var] > 0]
            if pair[1] is not None and not (single and close(min(single), pair[1])):
                text += " (coupled)"
        return text

    lines.append("At a glance: upper bounds of the two inputs that decide tier switching. "
                 "\"component\" is the tightest bound any single component of the pipeline assumes; "
                 "\"(coupled)\" means the composed bound comes from a constraint over several inputs, "
                 "listed under the controller's table below.")
    lines.append("")
    lines.append("| controller | input | framework component | framework composed (sound) | framework "
                 "SimpleContract (today) | Pacti library component | Pacti library composed (sound) | "
                 "third copy |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ctrl in CONTROLLERS:
        for var in ("wind_speed", "gps_hdop"):
            tc = ""
            if var == "wind_speed":
                vals = sorted({t["values"][ctrl] for t in doc["third_copy"] if ctrl in t["values"]})
                tc = " / ".join(fmt(x) for x in vals)
            lines.append(
                f"| {CONTROLLER_COMPONENT[ctrl]} | `{var}` | {own_hi('framework', ctrl, var)} | "
                f"{env_hi('framework', ctrl, 'pacti.compose', var)} | "
                f"{env_hi('framework', ctrl, 'SimpleContract.compose', var)} | "
                f"{own_hi('pacti_library', ctrl, var)} | "
                f"{env_hi('pacti_library', ctrl, 'pacti.compose', var)} | {tc} |")
    lines.append("")


def _envelope_section(doc: dict, lines: List[str]) -> None:
    comp = {(r["source"], r["controller"], r["method"]): r for r in doc["composed"]}
    third = doc["third_copy"]
    for ctrl in CONTROLLERS:
        cc = CONTROLLER_COMPONENT[ctrl]
        fw = comp[("framework", ctrl, "pacti.compose")]
        sc = comp[("framework", ctrl, "SimpleContract.compose")]
        pl = comp[("pacti_library", ctrl, "pacti.compose")]
        maps = [_canon_map(r, "envelope", "") for r in (fw, sc, pl)]
        variables = sorted(set().union(*[set(m) for m in maps]))
        lines.append(f"### {cc}")
        lines.append("")
        lines.append("| top-level input | framework, Pacti-composed (sound) | framework, SimpleContract.compose "
                     "(pre-flight today, unsound) | Pacti library, Pacti-composed (sound) | third copy |")
        lines.append("|---|---|---|---|---|")
        for v in variables:
            tc = ""
            if v == "wind_speed":
                vals = sorted({t["values"][ctrl] for t in third if ctrl in t["values"]})
                if vals:
                    tc = "<= " + " / ".join(fmt(x) for x in vals) + " (" + "; ".join(
                        _short(f"{t['path']}:{t['line']}") for t in third) + ")"
            cells = [_interval(m.get(v)) for m in maps]
            lines.append(f"| `{v}` | " + " | ".join(cells) + f" | {tc} |")
        lines.append("")
        for rec in (fw, pl):
            coupled = rec["coupled_assumptions"]
            label = SOURCE_LABEL[rec["source"]]
            if coupled:
                terms = ", ".join(f"`{t['text']}`" for t in coupled)
                lines.append(f"- Coupled composed assumptions, {label}: {terms}")
            else:
                lines.append(f"- Coupled composed assumptions, {label}: none (the envelope is a box)")
        outs = [_canon_map(r, "output_bounds", cc) for r in (fw, pl)]
        ovars = sorted(set(outs[0]) | set(outs[1]))
        lines.append("")
        lines.append("| composed guarantee (output) | framework, Pacti-composed | Pacti library, Pacti-composed |")
        lines.append("|---|---|---|")
        for v in ovars:
            lines.append(f"| `{v}` | {_interval(outs[0].get(v))} | {_interval(outs[1].get(v))} |")
        lines.append("")


def _tightenings(doc: dict, lines: List[str]) -> None:
    lines.append("### Where sound composition is tighter than every component's own bound")
    lines.append("")
    lines.append("These are the G1=>A2 obligations at work: a bound on an internal variable (or on the "
                 "actuator's `control_effort`) becomes a constraint on the top-level inputs.")
    lines.append("")
    found = False
    for t in doc["tightenings"]:
        found = True
        terms = ", ".join(f"`{x}`" for x in t["composed_terms"])
        lines.append(f"- {SOURCE_LABEL[t['source']]} · {CONTROLLER_COMPONENT[t['controller']]} · `{t['variable']}` "
                     f"{t['direction']}: components allow {fmt(t['component_bound'])} "
                     f"({_short(t['component_loc'])}); composed {fmt(t['composed_bound'])}. "
                     f"Composed terms on this input: {terms}")
    if not found:
        lines.append("- none")
    lines.append("")


def _rows_section(rows: List[dict], lines: List[str]) -> None:
    current = None
    for r in rows:
        if r["component"] != current:
            current = r["component"]
            lines.append("")
            lines.append(f"### {current}")
            lines.append("")
            lines.append("| # | variable · item | framework | Pacti library | third copy | status | A1 row | "
                         "effect on the composed assumptions |")
            lines.append("|---|---|---|---|---|---|---|---|")
        var = r["variable"]
        if r["variable_framework"] and r["variable_pacti"] and r["variable_framework"] != r["variable_pacti"]:
            var = f"{r['variable_framework']} / {r['variable_pacti']}"
        elif not r["variable_framework"] and r["variable_pacti"]:
            var = r["variable_pacti"]
        fv = _val(r["framework_value"], r["framework_loc"])
        if r["framework_via"]:
            fv += f" via {r['framework_via']}"
        pv = _val(r["pacti_value"], r["pacti_loc"])
        tc = ""
        if r["third_copy_value"]:
            tc = f"{r['third_copy_value']} ({_short(r['third_copy_loc'])}); = {r['third_copy_matches']}"
        status = r["status"] if r["status"] == "match" else f"**{r['status']}**"
        effects = []
        if r["effect_framework"]:
            effects.append(f"fw: {r['effect_framework']}")
        if r["effect_pacti"]:
            effects.append(f"pc: {r['effect_pacti']}")
        if r["note"]:
            effects.append(f"note: {r['note']}")
        a1 = "" if r["a1_row"] is None else str(r["a1_row"])
        cells = [str(r["row"]), f"`{var}` · {r['item']}", fv, pv, tc, status, a1, "<br>".join(effects)]
        lines.append("| " + " | ".join(_md_escape(c) for c in cells) + " |")
    lines.append("")


def reconciliation_md(doc: dict, rows: List[dict]) -> bytes:
    prov, env, summ = doc["provenance"], doc["environment"], doc["reconciliation"]
    roll = summ["f_a1_09_rollup"]
    L: List[str] = []
    L.append("# Contract reconciliation: framework vs Pacti library")
    L.append("")
    L.append("> **Choosing the contract values is the user's decision (decision DC).** This report is "
             "value-agnostic. It lays the two libraries side by side and shows what each one composes to "
             "under sound assume-guarantee composition (Pacti). It does not say which values are authoritative.")
    L.append("")
    L.append(f"Generated by `{TOOL_NAME}` {TOOL_VERSION} (schema {SCHEMA_VERSION}); pacti {env['pacti']}, "
             f"numpy {env['numpy']}, scipy {env['scipy']}. Inputs sha256 `{prov['inputs_sha256']}` "
             "(content hash of the four input files; no git commit ID is recorded, so cherry-picks and "
             "squashes do not change this file). "
             "Regenerate with `python tools/contracts_offline/run.py` from the repo root. "
             "The same data is in `reconciliation.csv` (full paths) and `contracts.json`.")
    L.append("")
    L.append("Path keys: " + ", ".join(f"{v} = `{k}`" for k, v in sorted(SHORT_PATH.items(), key=lambda x: x[1]))
             + ". Numbers are rounded to 10 significant digits.")
    L.append("")
    L.append("## 1. Composed per-controller envelopes, side by side")
    L.append("")
    L.append("Each cell is the range of a top-level input allowed by the composed assumptions of "
             "sensors -> estimator -> controller -> actuators (the pipeline of the framework's "
             "`compose_pipeline`). `—` means the input does not exist in that source. Sound columns are "
             "Pacti compositions (A1 and (G1 => A2)); the SimpleContract column is what pre-flight "
             "computes today and drops G1 => A2 (F-A1-01, K05). The Pacti-library pipeline composes "
             "gps and imu in parallel before ekf; the library's own `compose_pipeline` leaves imu out "
             "(F-A1-25), and that variant is in `contracts.json` under method "
             "`PactiContractLibrary.compose_pipeline`.")
    L.append("")
    _glance_section(doc, L)
    _envelope_section(doc, L)
    _tightenings(doc, L)
    L.append("## 2. Headline")
    L.append("")
    L.append(f"- Rows: **{summ['rows']}** (one per component, variable and bound or coefficient). "
             f"Both libraries have the item in {summ['both_present']} rows: {summ['match']} match, "
             f"{summ['mismatch']} mismatch. One library only: {summ['framework_only']} framework-only, "
             f"{summ['pacti_only']} Pacti-only. Not matching in total: {summ['not_match']}.")
    agree = "yes" if roll["agrees_with_f_a1_09"] else "NO"
    L.append(f"- F-A1-09 cross-check: grouping the rows into A1's hand-built table gives {roll['groups']} "
             f"groups (A1: {roll['groups_expected']}), {roll['match']} match and {roll['not_match']} do not "
             f"(A1: 15 and 40). Same matched rows as A1: {agree}. {roll['rows_outside_a1_table']} rows fall "
             "outside A1's 55 (items A1 listed as 'not counted', or did not list).")
    L.append("- The wind envelopes per controller, all sources, are in section 1. The third copy "
             "(`wind_limits` in hp and ip) is compared on the controller `wind_speed` rows.")
    L.append("")
    L.append("## 3. Conversion notes and ambiguities")
    L.append("")
    for n in doc["conversion_notes"]:
        L.append(f"- **{n['id']}** ({n['source']}, {n['where']}): {n['text']}")
    L.append("")
    L.append("## 4. How to read the rows")
    L.append("")
    L.append("- `A.lower` / `A.upper`: an assumption bound. `G.<dir>.const` / `G.<dir>.coef[x]`: the constant "
             "or the coefficient of `x` in a guarantee `y <= sum(c*x) + k` (upper) or `y >= ...` (lower).")
    L.append("- Status: `match` / `mismatch` when both libraries have the item; `framework_only` / "
             "`pacti_only` otherwise. A coefficient is an implicit 0 when that library's guarantee "
             "exists but has no term in `x` (noted on the row).")
    L.append("- Effect (fw = framework pipeline, pc = Pacti-library pipeline, per controller, or `all:` "
             "when the same for all three): for an assumption on a top-level input, `binds` if the composed "
             "envelope bound equals it, else `slack` with the composed bound; for an assumption on an "
             "internal variable, `cut` if the upstream composite can exceed it (Pacti adds a constraint on "
             "the top-level inputs; that constraint can still be dominated by a tighter one, so section 1 "
             "is the net result) or `discharged` if it cannot; for a guarantee, the downstream "
             "obligations it feeds through the guarantees of the same pipeline (direction-aware), listing "
             "the cuts. `n/a`: the component is not in that pipeline.")
    L.append("- Effect is computed under each library's own values. What would change if a single value were "
             "swapped to the other library's is not computed.")
    L.append("- Variable aliases used to line rows up (the repo has no renaming layer):")
    for comp, p, f, why in ALIASES:
        where = "all components" if comp == "*" else comp
        L.append(f"  - `{p}` (Pacti library) = `{f}` (framework), {where}: {why}.")
    for comp, f, p, why in NOT_ALIASED:
        L.append(f"  - not aliased, {comp}: `{f}` (framework) vs `{p}` (Pacti library): {why}.")
    L.append("")
    L.append("## 5. Reconciliation rows")
    _rows_section(rows, L)
    text = "\n".join(L).rstrip("\n") + "\n"
    return text.encode("utf-8")
