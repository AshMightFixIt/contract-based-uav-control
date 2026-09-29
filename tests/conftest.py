"""Shared fixtures for the characterisation harness (tests/golden, tests/smoke, tests/ros).

The core is the installed package ``contract_uav_core``
(``pip install -e ./contract_uav_core``); ``core_package`` checks that it
resolves to this checkout. Every script under test runs in a child process,
from a throwaway copy of the ``benchmarks/`` and ``demos/`` scripts in pytest's
temporary directory. The scripts save their plots into ``outputs/`` next to
their own folder (found from ``__file__``), so running from the copy keeps the
working tree clean, and CHILD_ENV stops Python from writing ``__pycache__``
into the package.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "contract_uav_core" / "contract_uav_core"
# The script folders that repo_copy copies.
SCRIPT_DIRS = ("benchmarks", "demos")

# Applied to every child process: headless plotting/pygame, UTF-8 output (the
# code logs characters such as U+2192 that a cp1252 pipe on Windows cannot
# encode), and no bytecode files.
CHILD_ENV = {
    "MPLBACKEND": "Agg",
    "SDL_VIDEODRIVER": "dummy",
    "SDL_AUDIODRIVER": "dummy",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Rewrite tests/golden/*.txt from the current output instead of "
        "comparing. Only for a reviewed, intentional behaviour change.",
    )


@pytest.fixture(scope="session")
def update_goldens(request) -> bool:
    return bool(request.config.getoption("--update-goldens"))


@pytest.fixture(scope="session")
def pacti_available() -> bool:
    return importlib.util.find_spec("pacti") is not None


@pytest.fixture(scope="session")
def core_package(tmp_path_factory) -> Path:
    """Path of ``contract_uav_core/core.py``, checked to be this checkout's.

    A child process, run outside the repository, must resolve
    ``contract_uav_core.core`` to this checkout's file; otherwise the tests would
    exercise some other copy of the core. Needs an editable install.
    """
    expected = PACKAGE_DIR / "core.py"
    proc = run_child(
        ["-c", "import importlib.util as u; s = u.find_spec('contract_uav_core.core'); "
               "print(s.origin if s and s.origin else '')"],
        cwd=tmp_path_factory.mktemp("core_check"),
        timeout=120,
    )
    found = proc.stdout.strip()
    if proc.returncode != 0 or not found or not Path(found).is_file() or not Path(found).samefile(expected):
        pytest.fail(
            "contract_uav_core must be installed from this checkout, in editable mode:\n"
            f"  {sys.executable} -m pip install -e ./contract_uav_core\n"
            f"expected {expected}\nfound    {found or 'nothing'}\n{proc.stderr[-2000:]}",
            pytrace=False,
        )
    return expected


@pytest.fixture
def repo_copy(tmp_path, core_package) -> Path:
    """A fresh copy of the ``benchmarks/`` and ``demos/`` scripts, outside the repo."""
    dest = tmp_path / "repo"
    dest.mkdir()
    for folder in SCRIPT_DIRS:
        (dest / folder).mkdir()
        for script in (REPO_ROOT / folder).glob("*.py"):
            shutil.copy2(script, dest / folder / script.name)
    return dest


@pytest.fixture(scope="session")
def no_pacti_dir(tmp_path_factory) -> Path:
    """A directory holding a stub ``pacti`` package whose import fails.

    Placed first on PYTHONPATH, it makes a child process behave as if pacti
    were not installed. The stub raises ModuleNotFoundError with
    ``name="pacti"``, as a genuinely missing package does, so it matches both
    ``except ImportError`` (today's guards in contract_uav_core) and a narrower
    ``except ModuleNotFoundError``. The requirements-only goldens therefore
    hold even in an environment that has pacti.
    """
    root = tmp_path_factory.mktemp("no_pacti")
    (root / "pacti").mkdir()
    (root / "pacti" / "__init__.py").write_text(
        "raise ModuleNotFoundError(\n"
        '    "pacti hidden by the test harness (tests/conftest.py)", name="pacti"\n'
        ")\n",
        encoding="utf-8",
    )
    return root


def run_child(args, cwd, pythonpath=(), timeout=1800):
    """Run ``sys.executable *args`` with CHILD_ENV.

    PYTHONPATH is replaced, never inherited, so a PYTHONPATH exported for the
    main checkout cannot leak in. Output is decoded as UTF-8 in text mode, so
    CRLF from Windows becomes LF.
    """
    env = dict(os.environ)
    env.update(CHILD_ENV)
    env.pop("PYTHONPATH", None)
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in pythonpath)
    return subprocess.run(
        [sys.executable] + [str(a) for a in args],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


@pytest.fixture
def run_python():
    """Return ``run(args, cwd, pythonpath=(), timeout=...)``; see run_child."""
    return run_child
