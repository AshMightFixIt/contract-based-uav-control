"""Build one reconciliation row per (component, variable, bound-or-coefficient).

Row items
* ``A.lower`` / ``A.upper``: an assumption bound on an input.
* ``G.<dir>.const``: the constant of a guarantee ``y <= sum(c*x) + k`` (or ``>=``).
* ``G.<dir>.coef[x]``: the coefficient of input ``x`` in that guarantee. When the
  guarantee exists in both libraries but one has no ``x`` term, that side's
  coefficient is an implicit 0 (exact, and noted on the row).

Status: ``match`` / ``mismatch`` when both libraries have the bound,
``framework_only`` / ``pacti_only`` otherwise. Variables are lined up through
``model.ALIASES``; nothing else is renamed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .model import (
    COMPONENT_ORDER,
    CONTROLLER_COMPONENT,
    Bound,
    canonical_var,
    close,
    fmt,
)

# --------------------------------------------------------------------------
# F-A1-09 cross-check: row numbers of the hand-built 55-row table in the A1
# contracts audit (finding F-A1-09; not part of this repo). Only the grouping is
# taken from A1; every value and status is recomputed from the code.
# None = not one of A1's 55 rows.
# --------------------------------------------------------------------------
_ANY = "*"
A1_GROUPS: Dict[Tuple[str, str, str, str], int] = {
    ("GPS", "gps_satellites", "A", _ANY): 1,
    ("GPS", "gps_hdop", "A", _ANY): 2,
    ("GPS", "position_meas_error", "G", "upper"): 3,
    ("GPS", "position_meas_error", "G", "lower"): 4,
    ("GPS", "velocity_meas_error", "G", "upper"): 5,
    ("GPS", "velocity_meas_error", "G", "lower"): 6,
    ("IMU", "imu_temperature", "A", _ANY): 7,
    ("IMU", "imu_temp_deviation", "A", _ANY): 7,
    ("IMU", "imu_calibrated", "A", _ANY): 8,
    ("IMU", "rate_meas_error", "G", "upper"): 9,
    ("IMU", "attitude_meas_error", "G", "upper"): 10,
    ("IMU", "rate_meas_error", "G", "lower"): 11,
    ("IMU", "attitude_meas_error", "G", "lower"): 11,
    ("EKF", "position_meas_error", "A", _ANY): 12,
    ("EKF", "velocity_meas_error", "A", _ANY): 13,
    ("EKF", "rate_meas_error", "A", _ANY): 14,
    ("EKF", "attitude_meas_error", "A", _ANY): 15,
    ("EKF", "position_error", "G", "upper"): 16,
    ("EKF", "velocity_error", "G", "upper"): 17,
    ("EKF", "attitude_error", "G", "upper"): 18,
    ("EKF", "position_error", "G", "lower"): 19,
    ("EKF", "velocity_error", "G", "lower"): 19,
    ("EKF", "attitude_error", "G", "lower"): 19,
    ("EKF", "rate_est_error", "G", _ANY): 20,
    ("EKF", "nees_bound", "G", _ANY): 21,
    ("EKF", "nis_bound", "G", _ANY): 21,
    ("PID", "position_error", "A", _ANY): 22,
    ("PID", "velocity_error", "A", _ANY): 23,
    ("PID", "wind_speed", "A", _ANY): 24,
    ("PID", "disturbance", "A", _ANY): 25,
    ("PID", "initial_tracking_error", "A", _ANY): 26,
    ("PID", "tracking_error", "G", "upper"): 27,
    ("PID", "tracking_error", "G", "lower"): 28,
    ("PID", "settling_time", "G", "upper"): 29,
    ("PID", "settling_time", "G", "lower"): 30,
    ("PID", "control_effort", "G", "upper"): 31,
    ("PID", "control_effort", "G", "lower"): 32,
    ("PID", "max_tilt", "G", _ANY): 33,
    ("MPC", "position_error", "A", _ANY): 34,
    ("MPC", "velocity_error", "A", _ANY): 35,
    ("MPC", "wind_speed", "A", _ANY): 36,
    ("MPC", "computation_time", "A", _ANY): 37,
    ("MPC", "initial_tracking_error", "A", _ANY): 38,
    ("MPC", "tracking_error", "G", "upper"): 39,
    ("MPC", "settling_time", "G", "upper"): 40,
    ("MPC", "settling_time", "G", "lower"): 41,
    ("MPC", "control_effort", "G", "upper"): 42,
    ("MPC", "control_effort", "G", "lower"): 43,
    ("HINF", "position_error", "A", _ANY): 44,
    ("HINF", "velocity_error", "A", _ANY): 45,
    ("HINF", "wind_speed", "A", _ANY): 46,
    ("HINF", "disturbance", "A", _ANY): 47,
    ("HINF", "initial_tracking_error", "A", _ANY): 48,
    ("HINF", "tracking_error", "G", "upper"): 49,
    ("HINF", "stabilization_time", "G", _ANY): 50,
    ("HINF", "control_effort", "G", _ANY): 51,
    ("HINF", "tilt_angle", "G", _ANY): 52,
    ("HINF", "rate_damping", "G", _ANY): 52,
    ("ACTUATOR", "control_effort", "A", _ANY): 53,
    ("ACTUATOR", "battery_voltage", "A", _ANY): 54,
    ("ACTUATOR", "motor_temperature", "A", _ANY): 55,
}
# The claim under test: F-A1-09 lists these 15 of its 55 rows as matches.
A1_CLAIMED_MATCHES = (4, 6, 11, 16, 19, 24, 28, 30, 34, 35, 39, 41, 43, 49, 53)
A1_ROW_COUNT = 55


def a1_group(component: str, var: str, kind: str, direction: str) -> Optional[int]:
    return A1_GROUPS.get((component, var, kind, direction), A1_GROUPS.get((component, var, kind, _ANY)))


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------
def _key(b: Bound) -> Tuple[str, str, str, str]:
    return (b.component, canonical_var(b.component, b.source, b.var), b.kind, b.direction)


def _index(bounds: List[Bound], flags: List[dict]) -> Dict[tuple, Bound]:
    out: Dict[tuple, Bound] = {}
    for b in bounds:
        k = _key(b)
        if k in out:
            flags.append({"id": "R-DUP", "source": b.source, "where": f"{b.loc} and {out[k].loc}",
                          "text": f"two {k[2]} {k[3]} bounds on {k[1]} in {k[0]}; the second is reported as #2"})
            k = k[:3] + (k[3] + "#2",)
        out[k] = b
    return out


def _third_copy_for(component: str, var: str, kind: str, direction: str, third: List[dict]):
    ctrl = {v: k for k, v in CONTROLLER_COMPONENT.items()}.get(component)
    if ctrl is None or kind != "A":
        return None
    hits = [t for t in third if t["variable"] == var and t["direction"] == direction and ctrl in t["values"]]
    if not hits:
        return None
    return [(t["values"][ctrl], f"{t['path']}:{t['line']}") for t in hits]


def _sort_key(row: dict):
    comp = COMPONENT_ORDER.index(row["component"]) if row["component"] in COMPONENT_ORDER else len(COMPONENT_ORDER)
    item = row["item"]
    item_rank = 0 if item.endswith(".const") or item.startswith("A.") else 1
    return (comp, row["component"], row["kind"], row["variable"], row["direction"], item_rank, item)


def build_rows(fw: List[Bound], pc: List[Bound], third: List[dict], effects: Dict[str, Dict[Bound, str]],
               flags: List[dict]) -> List[dict]:
    fi, pi = _index(fw, flags), _index(pc, flags)
    rows = []
    for key in sorted(set(fi) | set(pi)):
        component, var, kind, direction = key
        f, p = fi.get(key), pi.get(key)
        base = {
            "component": component,
            "variable": var,
            "variable_framework": f.var if f else "",
            "variable_pacti": p.var if p else "",
            "kind": kind,
            "direction": direction,
            "a1_row": a1_group(component, var, kind, direction.split("#")[0]),
            "effect_framework": effects["framework"].get(f, "") if f else "",
            "effect_pacti": effects["pacti_library"].get(p, "") if p else "",
        }
        if kind == "A":
            items = [("A." + direction, (f.constant if f else None), (p.constant if p else None), "bound", "")]
        else:
            items = [(f"G.{direction}.const", f.constant if f else None, p.constant if p else None, "const", "")]
            fc = {canonical_var(component, "framework", n): (n, c) for n, c in (f.coefficients if f else ())}
            pcf = {canonical_var(component, "pacti_library", n): (n, c) for n, c in (p.coefficients if p else ())}
            for x in sorted(set(fc) | set(pcf)):
                fv = fc[x][1] if x in fc else (0.0 if f else None)
                pv = pcf[x][1] if x in pcf else (0.0 if p else None)
                note = []
                if f and x not in fc:
                    note.append(f"implicit 0 in framework (its guarantee has no {x} term)")
                if p and x not in pcf:
                    note.append(f"implicit 0 in Pacti library (its guarantee has no {x} term)")
                via_key = f"coef[{fc[x][0]}]" if x in fc else ""
                items.append((f"G.{direction}.coef[{x}]", fv, pv, via_key, "; ".join(note)))
        for item, fv, pv, via_key, note in items:
            row = dict(base)
            notes = [note] if note else []
            if f and p and f.var != p.var:
                notes.append(f"renamed: framework {f.var} / Pacti library {p.var}")
            if fv is not None and pv is not None:
                status = "match" if close(fv, pv) else "mismatch"
            elif fv is not None:
                status = "framework_only"
            else:
                status = "pacti_only"
            tc = _third_copy_for(component, var, kind, direction, third)
            tc_value, tc_loc, tc_matches = "", "", ""
            if tc:
                vals = sorted({v for v, _ in tc})
                tc_value = " / ".join(fmt(v) for v in vals)
                tc_loc = "; ".join(loc for _, loc in tc)
                if len(vals) == 1:
                    m_f = fv is not None and close(vals[0], fv)
                    m_p = pv is not None and close(vals[0], pv)
                    tc_matches = {(True, True): "both", (True, False): "framework",
                                  (False, True): "pacti_library", (False, False): "neither"}[(m_f, m_p)]
                else:
                    tc_matches = "copies disagree"
            via = f.via_for(via_key) if (f and via_key) else ()
            row.update({
                "item": item,
                "framework_value": fv,
                "framework_loc": f.loc if f else "",
                "framework_via": ", ".join(via),
                "pacti_value": pv,
                "pacti_loc": p.loc if p else "",
                "third_copy_value": tc_value,
                "third_copy_loc": tc_loc,
                "third_copy_matches": tc_matches,
                "status": status,
                "note": "; ".join(notes),
            })
            row["row_id"] = f"{component}|{var}|{item}"
            rows.append(row)
    rows.sort(key=_sort_key)
    for i, r in enumerate(rows, 1):
        r["row"] = i
    return rows


def summarise(rows: List[dict]) -> dict:
    counts = {s: 0 for s in ("match", "mismatch", "framework_only", "pacti_only")}
    for r in rows:
        counts[r["status"]] += 1
    groups: Dict[int, List[str]] = {}
    for r in rows:
        if r["a1_row"] is not None:
            groups.setdefault(r["a1_row"], []).append(r["status"])
    matched = sorted(g for g, st in groups.items() if all(s == "match" for s in st))
    rollup = {
        "groups": len(groups),
        "groups_expected": A1_ROW_COUNT,
        "match": len(matched),
        "not_match": len(groups) - len(matched),
        "matched_rows": matched,
        "claimed_matched_rows": list(A1_CLAIMED_MATCHES),
        "agrees_with_f_a1_09": len(groups) == A1_ROW_COUNT and tuple(matched) == A1_CLAIMED_MATCHES,
        "rows_outside_a1_table": sum(1 for r in rows if r["a1_row"] is None),
    }
    return {
        "rows": len(rows),
        "both_present": counts["match"] + counts["mismatch"],
        **counts,
        "not_match": len(rows) - counts["match"],
        "f_a1_09_rollup": rollup,
    }
