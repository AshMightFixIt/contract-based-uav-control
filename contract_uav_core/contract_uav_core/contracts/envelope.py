"""
Read-only getters: the wind limits and the planner's contract bounds (W4-09).

No behaviour of its own and no limit values of its own: everything is read from
config/contracts.json, which planning/export_contracts.py writes offline (with
Pacti) from the planner's contract set, planning/pacti_contracts.py.

- contract_bounds(): the whole file, as a fresh copy (the horizon planner reads
  its assumption bounds and error-bound numbers from it).
- wind_envelope(): each controller's wind limit in the planner's contract set,
  {'pid': 3.0, 'mpc': 8.0, 'hinf': 15.0} m/s (the upper bound of the
  controller contract's wind_speed input).
- emergency_wind_threshold(): 15.0 m/s, the wind above which no controller
  contract of that set holds (H-infinity's limit), so the planner has no
  feasible plan and goes to its emergency. The flight-mode supervisor's
  EMERGENCY trigger (control/supervisor.py, wind > 15.0) and the framework's
  MPC limit (contracts/library.py) are the same 15.0; a test checks this.

The two contract sets disagree (K03): the framework set (contracts/library.py),
which pre-flight and the switching policy use, allows 3/15/25 m/s for
PID/MPC/H-infinity, the planner's set 3/8/15. Using wind_envelope() for
switching would lower the MPC's limit from 15 to 8 m/s. Which set is official
is the owner's decision DC (docs/ROADMAP.md); until then use
emergency_wind_threshold() only, and wind_envelope() for the planner's view.
"""

import copy
import functools
import json
from pathlib import Path
from typing import Any, Dict

CONTRACTS_FILE = Path(__file__).resolve().parent.parent / 'config' / 'contracts.json'
SCHEMA_VERSION = 1
CONTROLLERS = ('pid', 'mpc', 'hinf')


@functools.lru_cache(maxsize=1)
def _load() -> Dict[str, Any]:
    try:
        text = CONTRACTS_FILE.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{CONTRACTS_FILE} is missing: it is package data of contract_uav_core (setup.py "
            f"package_data); reinstall the package") from None
    doc = json.loads(text)
    if doc.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f"{CONTRACTS_FILE.name}: schema_version {doc.get('schema_version')!r}, "
                         f"expected {SCHEMA_VERSION}")
    return doc


def contract_bounds() -> Dict[str, Any]:
    """config/contracts.json as a fresh copy (callers may change it freely)."""
    return copy.deepcopy(_load())


def wind_envelope() -> Dict[str, float]:
    """Each controller's wind limit (m/s) in the planner's contract set: {'pid', 'mpc', 'hinf'} -> float."""
    contracts = _load()['contracts']
    return {c: float(contracts[c]['input_bounds']['wind_speed'][1]) for c in CONTROLLERS}


def emergency_wind_threshold() -> float:
    """The wind (m/s) above which no controller contract holds: H-infinity's limit, 15.0."""
    return wind_envelope()['hinf']
