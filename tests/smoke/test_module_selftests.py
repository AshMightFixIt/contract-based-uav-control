"""Smoke tests: each module's ``if __name__ == "__main__":`` self-test exits 0.

This replaces test_all.sh. Each module runs as
``python -m contract_uav_core.<module>``, from pytest's temp dir, against the
installed package (checked by the ``core_package`` fixture in conftest.py).
Only the exit code is checked, not the printed output. The three planning
modules import pacti, so they are skipped without it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "contract_uav_core" / "contract_uav_core"

# module -> needs pacti
SELF_TESTS = {
    "contract_uav_core.contracts.monitor": False,
    "contract_uav_core.estimation.ekf": False,
    "contract_uav_core.control.switcher": False,
    "contract_uav_core.control.supervisor": False,
    "contract_uav_core.safety.cbf": False,
    "contract_uav_core.core": False,
    "contract_uav_core.planning.pacti_contracts": True,
    "contract_uav_core.planning.horizon_planner": True,
    "contract_uav_core.planning.integrated_planner": True,
}


def _params():
    for module, needs_pacti in SELF_TESTS.items():
        marks = [pytest.mark.pacti] if needs_pacti else []
        yield pytest.param(module, needs_pacti, id=module, marks=marks)


@pytest.mark.smoke
@pytest.mark.parametrize("module, needs_pacti", list(_params()))
def test_module_selftest_exits_zero(module, needs_pacti, core_package, tmp_path, run_python, pacti_available):
    if needs_pacti and not pacti_available:
        pytest.skip("needs pacti: pip install -r requirements/pacti.txt")
    proc = run_python(["-m", module], cwd=tmp_path, timeout=600)
    assert proc.returncode == 0, (
        f"python -m {module} exited with {proc.returncode}\n"
        f"--- stdout (tail) ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr (tail) ---\n{proc.stderr[-3000:]}"
    )


@pytest.mark.smoke
def test_selftest_list_covers_every_main_block():
    """Fails when a module gains or loses a __main__ block, so the list stays complete."""
    found = sorted(
        ".".join((PACKAGE_DIR.name,) + path.relative_to(PACKAGE_DIR).with_suffix("").parts)
        for path in PACKAGE_DIR.rglob("*.py")
        if "__name__ ==" in path.read_text(encoding="utf-8")
    )
    assert found == sorted(SELF_TESTS)
