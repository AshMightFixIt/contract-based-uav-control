"""Golden characterisation tests for the two deterministic benchmark scripts.

These pin the *current* behaviour so refactors can be shown to preserve it.
Each test runs a benchmark script unchanged, as a child process from a copy of
the top-level scripts (the core comes from the installed contract_uav_core),
and compares its printed results with a pinned text file.

What is compared: the results block of stdout, from the first ``Running ...``
status line up to (not including) ``Generating comparison plots...``. That is
the per-controller status lines, the metrics table(s) and the qualitative
summary. Banners, the scenario description, the pacti warning and the
``Plot saved to: <path>`` line are left out. Line endings are normalised to LF
and trailing whitespace is stripped, so the comparison is identical on Windows.

Goldens:
  benchmark_controllers.nopacti.txt         requirements-only (pacti hidden)
  benchmark_sensor_degradation.nopacti.txt  requirements-only (pacti hidden)
  benchmark_controllers.pacti.txt           pacti 0.3.1, planner active; this is
                                            RESULTS.md section 1 (slow, opt-in)

To regenerate after a reviewed, intentional behaviour change:
  make golden-update         (the two requirements-only goldens)
  make golden-update-pacti   (the pacti golden; needs requirements-legacy.txt)

Known host dependence (review note N2): the Adaptive column depends on the
MPC's wall-clock solve time (time.perf_counter() in
contract_uav_core/contract_uav_core/control/mpc.py) staying under the 50 ms
contract bound it is checked against (check_conditions['computation_time'] in
contract_uav_core/contract_uav_core/switching_policy.py), so a heavily loaded
CI runner could in principle flip a switching decision. A golden diff confined
to the Adaptive column that does not reproduce on re-run points to this, not to
a behaviour change.
"""
from __future__ import annotations

import difflib
import importlib.metadata
import platform
import sys
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).resolve().parent

START_PREFIX = "Running "
END_LINE = "Generating comparison plots..."

# Column layout of print_comparison_table() in benchmark_controllers.py:
# metric name left-aligned in 28 characters, then one 18-character cell per mode.
METRIC_WIDTH = 28
CELL_WIDTH = 18


def normalise(text: str) -> str:
    """LF line endings, no trailing whitespace on any line."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n"))


def extract_results(stdout: str) -> str:
    lines = normalise(stdout).split("\n")
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(START_PREFIX))
        end = lines.index(END_LINE, start)
    except (StopIteration, ValueError):
        raise AssertionError(
            "Could not find the results block (from a line starting with "
            f"{START_PREFIX!r} to {END_LINE!r}) in the script output:\n{stdout[-3000:]}"
        ) from None
    block = lines[start:end]
    while block and not block[-1]:
        block.pop()
    return "\n".join(block) + "\n"


def environment_summary() -> str:
    parts = [f"python {platform.python_version()} ({sys.platform})"]
    for dist in ("numpy", "matplotlib", "pacti"):
        try:
            parts.append(f"{dist} {importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{dist} not installed")
    return ", ".join(parts)


def check_golden(actual: str, golden_name: str, update: bool) -> None:
    golden_path = GOLDEN_DIR / golden_name
    if update:
        with open(golden_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(actual)
        return
    if not golden_path.exists():
        pytest.fail(f"Missing golden file {golden_path}; create it with --update-goldens.")
    expected = normalise(golden_path.read_text(encoding="utf-8"))
    if actual == expected:
        return
    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f"golden/{golden_name}",
            tofile="actual",
        )
    )
    pytest.fail(
        f"Output differs from golden/{golden_name}\n"
        f"environment: {environment_summary()}\n\n{diff}\n"
        "If this change is intended, regenerate the golden (see this module's "
        "docstring) and justify the diff in review.",
        pytrace=False,
    )


def run_benchmark(run_python, repo_copy, script, pythonpath=(), timeout=1800):
    proc = run_python([repo_copy / script], cwd=repo_copy, pythonpath=pythonpath, timeout=timeout)
    assert proc.returncode == 0, (
        f"{script} exited with {proc.returncode}\n"
        f"--- stdout (tail) ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr (tail) ---\n{proc.stderr[-3000:]}"
    )
    return proc


def adaptive_cell(results: str, metric: str) -> str:
    row = next(line for line in results.split("\n") if line.startswith(metric))
    return row[METRIC_WIDTH:METRIC_WIDTH + CELL_WIDTH].strip()


@pytest.mark.golden
@pytest.mark.parametrize(
    "script, golden_name",
    [
        ("benchmark_controllers.py", "benchmark_controllers.nopacti.txt"),
        ("benchmark_sensor_degradation.py", "benchmark_sensor_degradation.nopacti.txt"),
    ],
)
def test_benchmark_matches_golden_without_pacti(
    script, golden_name, repo_copy, run_python, no_pacti_dir, update_goldens
):
    proc = run_benchmark(run_python, repo_copy, script, pythonpath=[no_pacti_dir], timeout=900)
    check_golden(extract_results(proc.stdout), golden_name, update_goldens)


@pytest.mark.golden
@pytest.mark.slow
@pytest.mark.pacti
def test_benchmark_controllers_with_pacti_reproduces_results_section_1(
    repo_copy, run_python, pacti_available, update_goldens
):
    """RESULTS.md section 1 needs the horizon planner, hence pacti (~8.5 min)."""
    if not pacti_available:
        pytest.skip("pacti is not installed: pip install -r requirements-legacy.txt")
    proc = run_benchmark(run_python, repo_copy, "benchmark_controllers.py", timeout=3600)
    results = extract_results(proc.stdout)
    if not update_goldens:
        # The headline number of RESULTS.md section 1; without pacti it is 63.240.
        assert adaptive_cell(results, "Mission Time (s)") == "53.780", environment_summary()
    check_golden(results, "benchmark_controllers.pacti.txt", update_goldens)
