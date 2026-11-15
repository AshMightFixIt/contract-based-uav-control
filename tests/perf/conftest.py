"""Fixtures for the W4-07 tests (A-R4): the patrol scenario of patrol.py, loaded by its path."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATROL_FILE = Path(__file__).resolve().parent / "patrol.py"


@pytest.fixture(scope="session")
def patrol(core_package):
    """The module patrol.py (the closed-loop patrol of the core on the reference plant)."""
    spec = importlib.util.spec_from_file_location("w4_07_patrol", PATROL_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
