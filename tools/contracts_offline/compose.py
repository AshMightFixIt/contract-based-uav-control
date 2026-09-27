"""Convert the framework's SimpleContracts to Pacti, compose every pipeline, and
work out how each bound affects the composed assumptions.

Pipeline (as ``HierarchicalContractMonitor.compose_pipeline`` defines it):
sensors -> estimator -> controller -> actuators.

* framework:     GPS_IMU_Sensors -> EKF_Estimator -> <controller> -> Actuators,
                 each converted to a ``PolyhedralIoContract``.
* pacti_library: (gps || imu) -> ekf -> <controller> -> actuator. GPS and IMU
                 share no variables, so Pacti composes them in parallel. The
                 library's own ``compose_pipeline`` leaves IMU out (F-A1-25); it
                 is recorded separately as a cross-check.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pacti.contracts import PolyhedralIoContract
from pacti.iocontract import Var
from pacti.terms.polyhedra.polyhedra import PolyhedralTerm, PolyhedralTermList

from .model import (
    CONTROLLER_COMPONENT,
    CONTROLLERS,
    FRAMEWORK_CONTROLLER_KEY,
    Bound,
    close,
    fmt,
    rnd,
    term_text,
)

STAGES = ("sensors", "estimator", "controller", "actuator")
STAGE_COMPONENTS = {"sensors": ("GPS", "IMU"), "estimator": ("EKF",), "actuator": ("ACTUATOR",)}


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------
def simple_contract_to_pacti(sc, flags: List[dict]) -> PolyhedralIoContract:
    """Convert one framework SimpleContract into a PolyhedralIoContract.

    * inputs  = assumption variables + every variable used in a guarantee coefficient
    * outputs = guarantee output variables
    * assumption ``var in (lo, hi)`` -> ``-var <= -lo`` and ``var <= hi``
    * guarantee ``y <= sum(c*x) + k``  -> ``y - sum(c*x) <= k``
      guarantee ``y >= sum(c*x) + k``  -> ``-y + sum(c*x) <= -k``
    """
    outputs = sorted({c.output_var for c in sc.guarantees})
    inputs = set(sc.assumptions)
    for c in sc.guarantees:
        inputs |= set(c.coefficients)
    both = sorted(inputs & set(outputs))
    if both:
        flags.append({"id": "C-IO", "source": "framework", "where": sc.name,
                      "text": f"variables {both} are both guarantee outputs and inputs; cannot convert"})
        raise ValueError(f"{sc.name}: variables are both inputs and outputs: {both}")
    unassumed = sorted(inputs - set(sc.assumptions))
    if unassumed:
        flags.append({"id": "C-UNASSUMED", "source": "framework", "where": sc.name,
                      "text": f"guarantee coefficients use inputs with no assumption: {unassumed}; "
                              "converted as unconstrained Pacti inputs"})
    a_terms = []
    for var, (lo, hi) in sorted(sc.assumptions.items()):
        a_terms.append(PolyhedralTerm({Var(var): -1.0}, -float(lo)))
        a_terms.append(PolyhedralTerm({Var(var): 1.0}, float(hi)))
    g_terms = []
    for c in sc.guarantees:
        s = 1.0 if c.is_upper_bound else -1.0
        coefs: Dict[str, float] = {c.output_var: s}
        for x, v in sorted(c.coefficients.items()):
            coefs[x] = coefs.get(x, 0.0) - s * float(v)
        g_terms.append(PolyhedralTerm({Var(k): v for k, v in sorted(coefs.items())}, s * float(c.constant)))
    contract = PolyhedralIoContract(
        PolyhedralTermList(a_terms), PolyhedralTermList(g_terms),
        [Var(x) for x in sorted(inputs)], [Var(x) for x in outputs],
    )
    if len(contract.g.terms) != len(g_terms):
        flags.append({"id": "C-SIMPLIFY", "source": "framework", "where": sc.name,
                      "text": f"Pacti's constructor simplified {len(g_terms)} guarantee terms to "
                              f"{len(contract.g.terms)} (default simplify=True, as the Pacti library uses)"})
    return contract


# --------------------------------------------------------------------------
# Serialisation helpers
# --------------------------------------------------------------------------
def _vars(term) -> List[Tuple[str, float]]:
    return sorted((v.name, float(c)) for v, c in term.variables.items())


def term_record(term) -> dict:
    coefs = _vars(term)
    return {
        "coefficients": {n: rnd(c) for n, c in coefs},
        "constant": rnd(term.constant),
        "text": term_text(coefs, float(term.constant)),
    }


def sorted_terms(termlist) -> List[dict]:
    recs = [term_record(t) for t in termlist.terms]
    uniq = {(r["text"], tuple(sorted(r["coefficients"].items())), r["constant"]): r for r in recs}
    return [uniq[k] for k in sorted(uniq, key=lambda k: (len(k[1]), [n for n, _ in k[1]], k[0]))]


def a_only_bounds(contract, var: str) -> Tuple[Optional[float], Optional[float]]:
    """Bounds of ``var`` over the contract's assumptions only."""
    obj = {Var(var): 1.0}
    return contract.a.optimize(obj, maximize=False), contract.a.optimize(obj, maximize=True)


def ag_bounds(contract, var: str) -> Tuple[Optional[float], Optional[float]]:
    """``get_variable_bounds``: bounds over assumptions and guarantees (what the planner and probes use)."""
    lo, hi = contract.get_variable_bounds(var)
    return lo, hi


def _pair(lo, hi) -> List[Optional[float]]:
    return [rnd(lo), rnd(hi)]


def names(vs) -> List[str]:
    return sorted(v.name for v in vs)


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
def build_pipelines(monitor, lib, fw_pacti: Dict[str, PolyhedralIoContract]):
    """Return {source: {controller: [(stage, components, contract_names, stage_contract, composite)]}}."""
    out: Dict[str, Dict[str, list]] = {"framework": {}, "pacti_library": {}}

    # framework: sensors and estimator composite shared by all controllers
    s_name = monitor.sensor_contract.name
    e_name = monitor.estimator_contract.name
    a_name = monitor.actuator_contract.name
    fs, fe, fa = fw_pacti[s_name], fw_pacti[e_name], fw_pacti[a_name]
    f_se = fs.compose(fe)
    for ctrl in CONTROLLERS:
        c_name = monitor.controller_contracts[FRAMEWORK_CONTROLLER_KEY[ctrl]].name
        fc = fw_pacti[c_name]
        p3 = f_se.compose(fc)
        p4 = p3.compose(fa)
        out["framework"][ctrl] = [
            ("sensors", STAGE_COMPONENTS["sensors"], [s_name], fs, fs),
            ("estimator", STAGE_COMPONENTS["estimator"], [e_name], fe, f_se),
            ("controller", (CONTROLLER_COMPONENT[ctrl],), [c_name], fc, p3),
            ("actuator", STAGE_COMPONENTS["actuator"], [a_name], fa, p4),
        ]

    # Pacti library: gps || imu, then ekf, controller, actuator
    lc = lib.contracts
    sens = lc["gps"].compose(lc["imu"])
    p_se = sens.compose(lc["ekf"])
    for ctrl in CONTROLLERS:
        p3 = p_se.compose(lc[ctrl])
        p4 = p3.compose(lc["actuator"])
        out["pacti_library"][ctrl] = [
            ("sensors", STAGE_COMPONENTS["sensors"], ["gps", "imu"], sens, sens),
            ("estimator", STAGE_COMPONENTS["estimator"], ["ekf"], lc["ekf"], p_se),
            ("controller", (CONTROLLER_COMPONENT[ctrl],), [ctrl], lc[ctrl], p3),
            ("actuator", STAGE_COMPONENTS["actuator"], ["actuator"], lc["actuator"], p4),
        ]
    return out


def composed_record(source: str, ctrl: str, stages, method: str, note: str, flags) -> dict:
    final = stages[-1][4]
    envelope = {}
    for v in names(final.inputvars):
        lo, hi = a_only_bounds(final, v)
        glo, ghi = ag_bounds(final, v)
        if not (close(lo, glo) and close(hi, ghi)):
            flags.append({"id": "P-AGDIFF", "source": source, "where": f"{ctrl} {v}",
                          "text": f"bounds over A ({fmt(lo)}, {fmt(hi)}) differ from get_variable_bounds "
                                  f"over A|G ({fmt(glo)}, {fmt(ghi)})"})
        envelope[v] = _pair(lo, hi)
    outputs = {v: _pair(*ag_bounds(final, v)) for v in names(final.outputvars)}
    terms = sorted_terms(final.a)
    coupled = [t for t in terms if len(t["coefficients"]) > 1]
    return {
        "source": source,
        "controller": ctrl,
        "method": method,
        "note": note,
        "pipeline": [n for st in stages for n in st[2]],
        "input_vars": names(final.inputvars),
        "output_vars": names(final.outputvars),
        "assumptions": terms,
        "coupled_assumptions": coupled,
        "envelope": envelope,
        # The box equals the admissible set only when every assumption term has one variable.
        "envelope_is_admissible_set": not coupled,
        "output_bounds": outputs,
    }


def box_counterexamples(rec: dict) -> List[dict]:
    """Corners of the envelope box that violate a coupled assumption term.

    Each one proves that the box is not the admissible set. For every coupled term,
    the corner is taken at the envelope end that maximises the term's left side.
    """
    out = []
    for t in rec.get("coupled_assumptions", []):
        point = {}
        for v, a in sorted(t["coefficients"].items()):
            lo, hi = rec["envelope"][v]
            point[v] = hi if a > 0 else lo
        if any(x is None for x in point.values()):
            continue
        lhs = sum(a * point[v] for v, a in t["coefficients"].items())
        if lhs > t["constant"] and not close(lhs, t["constant"]):
            out.append({"point": point, "term": t["text"], "lhs": rnd(lhs), "constant": t["constant"]})
    return out


def _box_terms(box: Dict[str, Tuple[float, float]]) -> List[dict]:
    """Assumption terms for a plain box, in the same form as Pacti terms (sum(a*v) <= c)."""
    terms = []
    for v, (lo, hi) in sorted(box.items()):
        for a, c in ((-1.0, -float(lo)), (1.0, float(hi))):
            terms.append({"coefficients": {v: rnd(a)}, "constant": rnd(c), "text": term_text([(v, a)], c)})
    return terms


def reference_records(monitor, lib) -> List[dict]:
    """The two compositions the repo itself performs, for comparison (not sound / not complete)."""
    recs = []
    for ctrl in CONTROLLERS:
        sc = monitor.compose_pipeline(FRAMEWORK_CONTROLLER_KEY[ctrl])
        recs.append({
            "source": "framework",
            "controller": ctrl,
            "method": "SimpleContract.compose",
            "note": "what pre-flight uses today; drops the G1=>A2 obligation on internal variables "
                    "(F-A1-01, K05), so this envelope is NOT sound",
            "pipeline": sc.name.split(">>"),
            "input_vars": sorted(sc.assumptions),
            "assumptions": _box_terms(sc.assumptions),
            "coupled_assumptions": [],
            "envelope": {v: _pair(lo, hi) for v, (lo, hi) in sorted(sc.assumptions.items())},
            "envelope_is_admissible_set": True,
        })
    for ctrl in CONTROLLERS:
        p = lib.compose_pipeline(ctrl)
        envelope = {v: _pair(*a_only_bounds(p, v)) for v in names(p.inputvars)}
        terms = sorted_terms(p.a)
        coupled = [t for t in terms if len(t["coefficients"]) > 1]
        recs.append({
            "source": "pacti_library",
            "controller": ctrl,
            "method": "PactiContractLibrary.compose_pipeline",
            "note": "the library's own method; composes gps -> ekf -> controller -> actuator and "
                    "leaves the IMU contract out (F-A1-25)",
            "pipeline": ["gps", "ekf", ctrl, "actuator"],
            "input_vars": names(p.inputvars),
            "assumptions": terms,
            "coupled_assumptions": coupled,
            "envelope": envelope,
            "envelope_is_admissible_set": not coupled,
        })
    return recs


def contract_records(source: str, contracts: Dict[str, PolyhedralIoContract], component_of) -> List[dict]:
    """Per-contract records: I/O, terms and the bounds of every input (probe_m1 style)."""
    recs = []
    for name in sorted(contracts):
        c = contracts[name]
        recs.append({
            "source": source,
            "contract": name,
            "component": component_of(name),
            "input_vars": names(c.inputvars),
            "output_vars": names(c.outputvars),
            "assumptions": sorted_terms(c.a),
            "guarantees": sorted_terms(c.g),
            "input_bounds": {v: _pair(*ag_bounds(c, v)) for v in names(c.inputvars)},
        })
    return recs


# --------------------------------------------------------------------------
# Effect of each bound on the composed assumptions
# --------------------------------------------------------------------------
def _flip(d: str) -> str:
    return "lower" if d == "upper" else "upper"


def analyse_effects(bounds: List[Bound], pipelines: Dict[str, list], envelopes: Dict[str, dict]):
    """Return {bound: {controller: effect_text}} for one library.

    * A bound on a top-level input: "binds" if the composed envelope bound equals
      it, else "slack" with the composed bound.
    * A bound on an internal variable (an output of the upstream stages): the
      G1=>A2 obligation. "discharged" if the upstream composite already keeps
      the variable inside the bound, else "cut" (Pacti adds a constraint on the
      top-level inputs).
    * A guarantee: which downstream obligations it feeds (direction-aware data
      flow through the guarantees of the same pipeline), and whether any of
      them is a cut.
    """
    cache: Dict[tuple, Tuple[Optional[float], Optional[float]]] = {}
    effects: Dict[Bound, Dict[str, str]] = {b: {} for b in bounds}
    by_component: Dict[str, List[Bound]] = {}
    for b in bounds:
        by_component.setdefault(b.component, []).append(b)

    for ctrl, stages in pipelines.items():
        stage_of = {}
        for i, st in enumerate(stages):
            for comp in st[1]:
                stage_of[comp] = i
        in_pipe = [b for b in bounds if b.component in stage_of]

        # obligations (A bounds) and their status in this pipeline
        status: Dict[Bound, Tuple[str, str]] = {}
        for b in in_pipe:
            if b.kind != "A":
                continue
            i = stage_of[b.component]
            upstream = stages[i - 1][4] if i > 0 else None
            internal = upstream is not None and b.var in names(upstream.outputvars)
            if not internal:
                lo, hi = envelopes[ctrl].get(b.var, (None, None))
                env = hi if b.direction == "upper" else lo
                if close(env, b.constant):
                    status[b] = ("binds", f"binds (composed {'hi' if b.direction == 'upper' else 'lo'} {fmt(env)})")
                else:
                    shown = fmt(env) if env is not None else "unbounded"
                    status[b] = ("slack", f"slack (composed {'hi' if b.direction == 'upper' else 'lo'} {shown})")
            else:
                key = (ctrl, i, b.var)
                if key not in cache:
                    cache[key] = ag_bounds(upstream, b.var)
                ulo, uhi = cache[key]
                if b.direction == "upper":
                    cut = uhi is None or (uhi > b.constant and not close(uhi, b.constant))
                    shown = f"upstream max {fmt(uhi) if uhi is not None else 'unbounded'}"
                else:
                    cut = ulo is None or (ulo < b.constant and not close(ulo, b.constant))
                    shown = f"upstream min {fmt(ulo) if ulo is not None else 'unbounded'}"
                status[b] = ("cut", f"cut ({shown})") if cut else ("discharged", f"discharged ({shown})")
            effects[b][ctrl] = status[b][1]

        # direction-aware data flow: (x, dx) -> (y, dy) for every guarantee y ~ c*x
        edges: Dict[Tuple[str, str], set] = {}
        for g in in_pipe:
            if g.kind != "G":
                continue
            for x, c in g.coefficients:
                dx = g.direction if c > 0 else _flip(g.direction)
                edges.setdefault((x, dx), set()).add((g.var, g.direction))

        def reach(start):
            seen, todo = {start}, [start]
            while todo:
                node = todo.pop()
                for nxt in sorted(edges.get(node, ())):
                    if nxt not in seen:
                        seen.add(nxt)
                        todo.append(nxt)
            return seen

        for g in in_pipe:
            if g.kind != "G":
                continue
            i = stage_of[g.component]
            r = reach((g.var, g.direction))
            fed = [b for b in status
                   if stage_of[b.component] > i and (b.var, b.direction) in r and status[b][0] in ("cut", "discharged")]
            cuts = sorted(b.label() for b in fed if status[b][0] == "cut")
            if cuts:
                effects[g][ctrl] = "feeds cut: " + ", ".join(cuts)
            elif fed:
                effects[g][ctrl] = "feeds only discharged obligations"
            else:
                effects[g][ctrl] = "not assumed downstream"
    return effects


def summarise_effect(per_ctrl: Dict[str, str], component: str) -> str:
    if not per_ctrl:
        return "not in any pipeline" if component == "DYNAMICS" else "n/a"
    vals = [per_ctrl[c] for c in CONTROLLERS if c in per_ctrl]
    if len(per_ctrl) == len(CONTROLLERS) and len(set(vals)) == 1:
        return f"all: {vals[0]}"
    return "; ".join(f"{c}: {per_ctrl[c]}" for c in CONTROLLERS if c in per_ctrl)
