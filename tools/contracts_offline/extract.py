"""Read both contract libraries read-only and normalise every bound.

* Values come from the imported runtime objects (``src`` is put on ``sys.path``
  inside this process only; bytecode writing is switched off so nothing is
  written under ``src/``).
* File:line locations come from parsing the same source files with ``ast``.
* The planner's hard-coded ``wind_limits`` dicts are local variables inside
  methods, so they can only be read from the source (``ast``), not imported.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .model import (
    FRAMEWORK_CONTRACT_COMPONENT,
    FRAMEWORK_PATH,
    FRAMEWORK_SENSOR_CONTRACT,
    PACTI_CONTRACT_COMPONENT,
    PACTI_LIB_PATH,
    SENSOR_VAR_COMPONENT,
    THIRD_COPY_PATHS,
    Bound,
)


class ExtractionError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Imports (read-only)
# --------------------------------------------------------------------------
def _ensure_src_on_path(repo_root: Path) -> Path:
    src = (repo_root / "src").resolve()
    sys.dont_write_bytecode = True  # never write __pycache__ under src/
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _import_from_src(repo_root: Path, module: str):
    src = _ensure_src_on_path(repo_root)
    mod = importlib.import_module(module)
    path = Path(mod.__file__).resolve()
    if src not in path.parents:
        raise ExtractionError(f"{module} was imported from {path}, not from {src}")
    return mod


def import_framework(repo_root: Path):
    mod = _import_from_src(repo_root, "contracts.contract_framework")
    monitor = mod.HierarchicalContractMonitor()
    monitor.define_contracts()
    return mod, monitor


def import_pacti_library(repo_root: Path):
    mod = _import_from_src(repo_root, "planning.pacti_contracts")
    return mod, mod.PactiContractLibrary()


def framework_contracts(monitor) -> List:
    """The framework's SimpleContracts in pipeline order (controllers sorted by key)."""
    out = [monitor.sensor_contract, monitor.estimator_contract]
    out += [monitor.controller_contracts[k] for k in sorted(monitor.controller_contracts)]
    out.append(monitor.actuator_contract)
    return out


# --------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------
def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _loc(path: str, first: int, last: int) -> str:
    return f"{path}:{first}" if first == last else f"{path}:{first}-{last}"


def _names_in(node, consts: Dict[str, int]) -> Tuple[str, ...]:
    if node is None:
        return ()
    names = sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in consts})
    return tuple(f"{n} (:{consts[n]})" for n in names)


def _framework_ast(repo_root: Path):
    tree = ast.parse((repo_root / FRAMEWORK_PATH).read_text(encoding="utf-8"))
    fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "define_contracts"),
        None,
    )
    if fn is None:
        raise ExtractionError(f"define_contracts not found in {FRAMEWORK_PATH}")
    consts: Dict[str, int] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            consts.setdefault(node.targets[0].id, node.lineno)
    contracts = {}
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and _call_name(node) == "SimpleContract"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        name = ast.literal_eval(kw["name"])
        assumptions = {}
        for key, val in zip(kw["assumptions"].keys, kw["assumptions"].values):
            elts = val.elts if isinstance(val, ast.Tuple) else [None, None]
            assumptions[ast.literal_eval(key)] = {"line": key.lineno, "lo": elts[0], "hi": elts[1]}
        guarantees = []
        for call in kw["guarantees"].elts:
            args = list(call.args)
            ckw = {k.arg: k.value for k in call.keywords}
            out = ast.literal_eval(args[0] if args else ckw["output_var"])
            coef_node = args[1] if len(args) > 1 else ckw.get("coefficients")
            const_node = args[2] if len(args) > 2 else ckw.get("constant")
            coefs = {}
            if isinstance(coef_node, ast.Dict):
                coefs = {ast.literal_eval(k): v for k, v in zip(coef_node.keys, coef_node.values)}
            guarantees.append(
                {"first": call.lineno, "last": call.end_lineno, "output": out,
                 "coef_nodes": coefs, "const_node": const_node}
            )
        contracts[name] = {"assumptions": assumptions, "guarantees": guarantees}
    return consts, contracts


def _pacti_ast(repo_root: Path):
    tree = ast.parse((repo_root / PACTI_LIB_PATH).read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Attribute)
                and tgt.value.attr == "contracts" and isinstance(node.value, ast.Call)
                and _call_name(node.value) == "from_strings"):
            continue
        name = ast.literal_eval(tgt.slice)
        kw = {k.arg: k.value for k in node.value.keywords}
        out[name] = {
            part: [(ast.literal_eval(e), e.lineno) for e in kw[part].elts]
            for part in ("assumptions", "guarantees")
        }
    return out


# --------------------------------------------------------------------------
# Framework bounds
# --------------------------------------------------------------------------
def _framework_component(contract_name: str, var: str, flags: List[dict]) -> str:
    if contract_name == FRAMEWORK_SENSOR_CONTRACT:
        comp = SENSOR_VAR_COMPONENT.get(var)
        if comp is None:
            flags.append({"id": "X-SENSOR-VAR", "source": "framework", "where": contract_name,
                          "text": f"sensor variable {var!r} has no GPS/IMU mapping; reported under SENSORS"})
            return "SENSORS"
        return comp
    comp = FRAMEWORK_CONTRACT_COMPONENT.get(contract_name)
    if comp is None:
        flags.append({"id": "X-CONTRACT", "source": "framework", "where": contract_name,
                      "text": f"unknown framework contract {contract_name!r}; reported under its own name"})
        return contract_name.upper()
    return comp


def framework_bounds(repo_root: Path, monitor, flags: List[dict]) -> List[Bound]:
    consts, index = _framework_ast(repo_root)
    bounds: List[Bound] = []
    for sc in framework_contracts(monitor):
        entry = index.get(sc.name)
        if entry is None:
            flags.append({"id": "X-NOAST", "source": "framework", "where": sc.name,
                          "text": "no SimpleContract(...) literal found for this contract; locations unknown"})
            entry = {"assumptions": {}, "guarantees": []}
        for var, (lo, hi) in sorted(sc.assumptions.items()):
            a = entry["assumptions"].get(var)
            loc = _loc(FRAMEWORK_PATH, a["line"], a["line"]) if a else f"{FRAMEWORK_PATH}:?"
            comp = _framework_component(sc.name, var, flags)
            for direction, value, node_key in (("lower", lo, "lo"), ("upper", hi, "hi")):
                via = (("bound", _names_in(a[node_key], consts)),) if a else ()
                bounds.append(Bound("framework", comp, sc.name, "A", var, direction, (), float(value), loc, via))
        for i, c in enumerate(sc.guarantees):
            g = entry["guarantees"][i] if i < len(entry["guarantees"]) else None
            if g is None or g["output"] != c.output_var:
                flags.append({"id": "X-ASTORDER", "source": "framework", "where": sc.name,
                              "text": f"guarantee #{i} ({c.output_var}) could not be matched to its source line"})
                g = None
            loc = _loc(FRAMEWORK_PATH, g["first"], g["last"]) if g else f"{FRAMEWORK_PATH}:?"
            via = []
            if g:
                via.append(("const", _names_in(g["const_node"], consts)))
                for var in sorted(c.coefficients):
                    via.append((f"coef[{var}]", _names_in(g["coef_nodes"].get(var), consts)))
            comp = _framework_component(sc.name, c.output_var, flags)
            coefs = tuple(sorted((k, float(v)) for k, v in c.coefficients.items()))
            direction = "upper" if c.is_upper_bound else "lower"
            bounds.append(Bound("framework", comp, sc.name, "G", c.output_var, direction, coefs,
                                float(c.constant), loc, tuple(via)))
    return bounds


# --------------------------------------------------------------------------
# Pacti-library bounds
# --------------------------------------------------------------------------
def _term_key(variables: Dict[str, float], constant: float):
    return (tuple(sorted((k, round(v, 12)) for k, v in variables.items())), round(constant, 12))


def _term_vars(term) -> Dict[str, float]:
    return {v.name: float(c) for v, c in term.variables.items()}


def pacti_term_to_bound(term, kind: str, outputs, component: str, contract: str, loc: str, flags) -> Bound:
    """Normalise a Pacti term ``sum(a*v) <= k`` to a Bound on one subject variable."""
    vars_ = _term_vars(term)
    k = float(term.constant)
    if kind == "A":
        subjects = sorted(vars_)
    else:
        subjects = sorted(v for v in vars_ if v in outputs)
    if len(subjects) != 1:
        flags.append({"id": "X-COUPLED", "source": "pacti_library", "where": f"{contract} ({loc})",
                      "text": f"{kind} term over {sorted(vars_)} has {len(subjects)} subject variables; "
                              "kept in contracts.json, left out of the row table"})
        return None
    subject = subjects[0]
    a = vars_[subject]
    direction = "upper" if a > 0 else "lower"
    coefs = tuple(sorted((v, -c / a) for v, c in vars_.items() if v != subject))
    const = k / a
    return Bound("pacti_library", component, contract, kind, subject, direction, coefs,
                 0.0 if const == 0 else const, loc, ())


def pacti_bounds(repo_root: Path, lib, flags: List[dict]) -> List[Bound]:
    from pacti.terms.polyhedra.serializer import polyhedral_termlist_from_string

    index = _pacti_ast(repo_root)
    bounds: List[Bound] = []
    for name in sorted(lib.contracts):
        c = lib.contracts[name]
        component = PACTI_CONTRACT_COMPONENT.get(name, name.upper())
        if name not in PACTI_CONTRACT_COMPONENT:
            flags.append({"id": "X-CONTRACT", "source": "pacti_library", "where": name,
                          "text": f"unknown Pacti-library contract {name!r}; reported under {component}"})
        outputs = {v.name for v in c.outputvars}
        entry = index.get(name, {"assumptions": [], "guarantees": []})
        for kind, part, terms in (("A", "assumptions", c.a.terms), ("G", "guarantees", c.g.terms)):
            lines: Dict[tuple, List[int]] = {}
            for text, line in entry[part]:
                for t in polyhedral_termlist_from_string(text):
                    lines.setdefault(_term_key(_term_vars(t), float(t.constant)), []).append(line)
            used = set()
            for term in terms:
                key = _term_key(_term_vars(term), float(term.constant))
                src_lines = lines.get(key)
                if src_lines:
                    loc = _loc(PACTI_LIB_PATH, src_lines[0], src_lines[0])
                    used.add(key)
                else:
                    loc = f"{PACTI_LIB_PATH}:?"
                    flags.append({"id": "X-NOLINE", "source": "pacti_library", "where": name,
                                  "text": f"runtime {kind} term {term} has no literal source string "
                                          "(Pacti may have simplified it)"})
                b = pacti_term_to_bound(term, kind, outputs, component, name, loc, flags)
                if b is not None:
                    bounds.append(b)
            for key in sorted(set(lines) - used):
                flags.append({"id": "X-DROPPED", "source": "pacti_library", "where": name,
                              "text": f"source {kind} string at line(s) {lines[key]} has no runtime term "
                                      "(simplified away by Pacti at construction)"})
    return bounds


# --------------------------------------------------------------------------
# Third copy: planner wind limits
# --------------------------------------------------------------------------
def third_copy(repo_root: Path) -> List[dict]:
    found = []
    for rel in THIRD_COPY_PATHS:
        tree = ast.parse((repo_root / rel).read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "wind_limits"
                        and isinstance(node.value, ast.Dict)):
                    values = {ast.literal_eval(k): float(ast.literal_eval(v))
                              for k, v in zip(node.value.keys, node.value.values)}
                    found.append({"path": rel, "line": node.lineno, "function": fn.name,
                                  "variable": "wind_speed", "direction": "upper", "values": values})
    unique = {(d["path"], d["line"]): d for d in found}  # nested functions are walked twice
    return [unique[k] for k in sorted(unique)]
