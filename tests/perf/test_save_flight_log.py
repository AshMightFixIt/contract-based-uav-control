"""W4-07 T3: save_flight_log keeps the log in memory as it was, and never overwrites it.

Two bugs in docs/KNOWN_ISSUES.md ("Smaller issues found during the fixes"),
both reproduced at 291fa13 before the fix:
- saving turned the arrays of the in-memory log into lists: the entry was
  copied shallowly, and the conversion wrote into the shared inner dicts;
- the contract metrics went to filename.replace('.json', '_contracts.json'),
  which is the log's own name when it has no '.json', so they overwrote the log
  (and a '.json' inside a directory name was replaced too).

The fixture ``legacy_save`` is save_flight_log at 291fa13, verbatim: for a
name whose only '.json' is its extension, the new one writes the same two
files, byte for byte.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("core_package")


@pytest.fixture(scope="module")
def legacy_save():
    """save_flight_log at 291fa13, verbatim (indented into this fixture)."""
    import logging
    logger = logging.getLogger(__name__)

    def save_flight_log(self, filename: str):
        """Save flight log for analysis."""
        import json

        log_serializable = []
        for entry in self.flight_log:
            entry_copy = entry.copy()
            for key in ['state', 'control', 'system_conditions']:
                if key in entry_copy:
                    for subkey, val in entry_copy[key].items():
                        if isinstance(val, np.ndarray):
                            entry_copy[key][subkey] = val.tolist()
            log_serializable.append(entry_copy)

        with open(filename, 'w') as f:
            json.dump(log_serializable, f, indent=2)

        logger.info(f"Flight log saved to {filename}")

        contract_log = filename.replace('.json', '_contracts.json')
        self.contract_monitor.export_metrics(contract_log)

    return save_flight_log


@pytest.fixture
def flown(patrol):
    """A controller after 200 steps of the patrol."""
    def fly(steps=200, **kwargs):
        run = patrol.Patrol(**kwargs)
        for _ in range(steps):
            run.step()
        return run.controller
    return fly


def arrays_of(log):
    return [(k, sub, v) for entry in log for k in ('state', 'control', 'system_conditions')
            for sub, v in entry[k].items() if isinstance(v, np.ndarray)]


def test_saving_keeps_the_arrays_in_memory(flown, tmp_path):
    ctrl = flown()
    before = copy.deepcopy(list(ctrl.flight_log))
    assert len(arrays_of(before)) >= 200 * 4      # position, velocity, attitude, rates at least
    ctrl.save_flight_log(str(tmp_path / 'one.json'))
    after = arrays_of(ctrl.flight_log)
    assert len(after) == len(arrays_of(before))
    assert all(a.tobytes() == b.tobytes() for (_, _, a), (_, _, b) in zip(after, arrays_of(before)))
    ctrl.save_flight_log(str(tmp_path / 'two.json'))  # and a second save writes the same
    assert (tmp_path / 'one.json').read_bytes() == (tmp_path / 'two.json').read_bytes()


def test_same_files_as_before_for_a_json_name(flown, legacy_save, tmp_path):
    ctrl = flown()
    old = SimpleNamespace(flight_log=copy.deepcopy(ctrl.flight_log), contract_monitor=ctrl.contract_monitor)
    legacy_save(old, str(tmp_path / 'old.json'))
    ctrl.save_flight_log(str(tmp_path / 'new.json'))
    assert (tmp_path / 'new.json').read_bytes() == (tmp_path / 'old.json').read_bytes()
    assert (tmp_path / 'new_contracts.json').read_bytes() == (tmp_path / 'old_contracts.json').read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['new.json', 'new_contracts.json',
                                                           'old.json', 'old_contracts.json']


@pytest.mark.parametrize('name, contracts', [
    ('flight.json', 'flight_contracts.json'),
    ('flight', 'flight_contracts.json'),
    ('flight.JSON', 'flight_contracts.json'),
    ('flight.log', 'flight.log_contracts.json'),
    ('runs.json/flight.json', 'runs.json/flight_contracts.json'),
    ('runs.json/flight', 'runs.json/flight_contracts.json'),
])
@pytest.mark.parametrize('as_path', [False, True], ids=['str', 'Path'])
def test_contract_metrics_never_overwrite_the_log(flown, tmp_path, name, contracts, as_path):
    ctrl = flown(steps=50)
    (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / name
    ctrl.save_flight_log(target if as_path else str(target))
    log = json.loads(target.read_text())
    metrics = json.loads((tmp_path / contracts).read_text())
    assert isinstance(log, list) and len(log) == 50 and log[0]['state']['position'] == list(
        ctrl.flight_log[0]['state']['position'])
    assert list(metrics) == ['history'] and len(metrics['history']) == 5 * 50
    assert len([p for p in tmp_path.rglob('*') if p.is_file()]) == 2


def test_a_bounded_log_saves_its_newest_entries(flown, tmp_path):
    ctrl = flown(steps=120, flight_log_len=30)
    ctrl.save_flight_log(str(tmp_path / 'bounded.json'))
    log = json.loads((tmp_path / 'bounded.json').read_text())
    assert [entry['time'] for entry in log] == [entry['time'] for entry in ctrl.flight_log]
    assert len(log) == 30 and np.isclose(log[-1]['time'], 119 * 0.02)
