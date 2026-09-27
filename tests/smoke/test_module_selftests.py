"""Smoke tests: each module's ``if __name__ == "__main__":`` self-test exits 0.

This replaces test_all.sh. Each module runs as documented in DEVELOPMENT.md
(``PYTHONPATH=src python src/<module>.py``), but from a copy of the repository
in pytest's temp dir. Only the exit code is checked, not the printed output.
The three src/planning modules import pacti, so they are skipped without it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# module path -> needs pacti
SELF_TESTS = {
    "src/contracts/contract_framework.py": False,
    "src/estimation/contract_ekf.py": False,
    "src/control/controllers.py": False,
    "src/control/flight_mode_supervisor.py": False,
    "src/safety/cbf_filter.py": False,
    "src/adaptive_control_system.py": False,
    "src/planning/pacti_contracts.py": True,
    "src/planning/horizon_planner.py": True,
    "src/planning/integrated_planner.py": True,
}


def _params():
    for module, needs_pacti in SELF_TESTS.items():
        marks = [pytest.mark.pacti] if needs_pacti else []
        yield pytest.param(module, needs_pacti, id=module, marks=marks)


@pytest.mark.smoke
@pytest.mark.parametrize("module, needs_pacti", list(_params()))
def test_module_selftest_exits_zero(module, needs_pacti, repo_copy, run_python, pacti_available):
    if needs_pacti and not pacti_available:
        pytest.skip("needs pacti: pip install -r requirements-legacy.txt")
    proc = run_python([repo_copy / module], cwd=repo_copy, pythonpath=[repo_copy / "src"], timeout=600)
    assert proc.returncode == 0, (
        f"{module} exited with {proc.returncode}\n"
        f"--- stdout (tail) ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr (tail) ---\n{proc.stderr[-3000:]}"
    )


@pytest.mark.smoke
def test_selftest_list_covers_every_main_block():
    """Fails when a module gains or loses a __main__ block, so the list stays complete."""
    found = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "src").rglob("*.py")
        if "__name__ ==" in path.read_text(encoding="utf-8")
    )
    assert found == sorted(SELF_TESTS)
