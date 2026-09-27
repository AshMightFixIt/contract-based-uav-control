"""Offline contracts tool.

Composes the repo's two contract libraries soundly with Pacti and reports where
they disagree. It is an offline design tool: nothing here runs on the robot, and
it never modifies ``src/``.

Run from the repo root with either of::

    python tools/contracts_offline/run.py
    python -m tools.contracts_offline
"""

TOOL_NAME = "tools/contracts_offline"
TOOL_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
