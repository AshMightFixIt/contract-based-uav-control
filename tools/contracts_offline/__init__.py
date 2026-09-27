"""Offline contracts tool.

Composes the repo's two contract libraries soundly with Pacti and reports where
they disagree. It is an offline design tool: nothing here runs on the robot, and
it never modifies ``src/``.

Run from the repo root with either of::

    python tools/contracts_offline/run.py
    python -m tools.contracts_offline
"""

TOOL_NAME = "tools/contracts_offline"
TOOL_VERSION = "1.1.0"
# Schema history:
#   1.0.0  initial
#   2.0.0  provenance.source_commit* removed: out/ no longer depends on git commit IDs
#   2.1.0  envelope_semantics, assumption_term_form, composed[].envelope_is_admissible_set and
#          composed[].box_counterexamples added; every composed record carries assumptions
SCHEMA_VERSION = "2.1.0"
