"""Shared fixtures for the characterisation harness (tests/golden, tests/smoke).

Every script under test runs in a child process, from a throwaway copy of the
repository (``src/`` plus the top-level ``*.py`` scripts) in pytest's temporary
directory. The benchmark scripts save PNGs next to ``__file__`` and Python
writes ``__pycache__`` next to the modules it imports, so running from the copy
keeps the working tree clean.
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


@pytest.fixture
def repo_copy(tmp_path) -> Path:
    """A fresh copy of ``src/`` and the top-level scripts, outside the repo."""
    dest = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT / "src",
        dest / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for script in REPO_ROOT.glob("*.py"):
        shutil.copy2(script, dest / script.name)
    return dest


@pytest.fixture(scope="session")
def no_pacti_dir(tmp_path_factory) -> Path:
    """A directory holding a stub ``pacti`` package whose import fails.

    Placed first on PYTHONPATH, it makes a child process behave as if pacti
    were not installed. The stub raises ModuleNotFoundError with
    ``name="pacti"``, as a genuinely missing package does, so it matches both
    ``except ImportError`` (today's guards in src/) and a narrower
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


@pytest.fixture
def run_python():
    """Return ``run(args, cwd, pythonpath=(), timeout=...)``.

    Runs ``sys.executable *args`` with CHILD_ENV. PYTHONPATH is replaced, never
    inherited, so a PYTHONPATH exported for the main checkout cannot leak in.
    Output is decoded as UTF-8 in text mode, so CRLF from Windows becomes LF.
    """

    def _run(args, cwd, pythonpath=(), timeout=1800):
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

    return _run
