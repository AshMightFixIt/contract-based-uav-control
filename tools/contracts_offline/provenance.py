"""Provenance: a content hash of the inputs, and the tool's environment. No timestamps.

The committed outputs deliberately contain no git commit ID. Commit IDs change
under cherry-pick, rebase and squash, and a commit cannot contain its own ID, so
an ID in ``out/`` would make the "regenerated out/ == committed out/" check depend
on history. ``inputs_sha256`` identifies the inputs by content instead.
``last_input_commit`` is only printed to stdout for the log.
"""

from __future__ import annotations

import hashlib
import subprocess
from importlib import metadata
from pathlib import Path
from typing import List, Optional

from .model import INPUT_PATHS

HASH_RULE = ("sha256 over '<path>\\0<sha256 of the file with CRLF normalised to LF>\\n' "
             "for the input files sorted by path")


def _lf_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def input_hashes(repo_root: Path) -> dict:
    files: List[dict] = []
    h = hashlib.sha256()
    for rel in INPUT_PATHS:
        digest = hashlib.sha256(_lf_bytes((repo_root / rel).read_bytes())).hexdigest()
        files.append({"path": rel, "sha256": digest})
        h.update(f"{rel}\0{digest}\n".encode("utf-8"))
    return {"inputs": files, "inputs_sha256": h.hexdigest()}


def last_input_commit(repo_root: Path) -> Optional[str]:
    """For the console only: the last commit that touched an input (None without git history).

    In a shallow clone ``git log`` would report the graft commit even if it touched no
    input, so the history is reported as unknown instead.
    """
    try:
        shallow = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--is-shallow-repository"],
                                 capture_output=True, check=False)
        if shallow.returncode == 0 and shallow.stdout.decode().strip() == "true":
            return "unknown (shallow clone)"
        res = subprocess.run(["git", "-C", str(repo_root), "log", "-1", "--format=%H", "--", *INPUT_PATHS],
                             capture_output=True, check=False)
    except OSError:
        return None
    out = res.stdout.decode().strip() if res.returncode == 0 else ""
    return out or None


def environment() -> dict:
    env = {}
    for pkg in ("pacti", "numpy", "scipy"):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[pkg] = None
    return env


def provenance(repo_root: Path) -> dict:
    return {**input_hashes(repo_root), "hash_rule": HASH_RULE,
            "commit_policy": "no git commit ID is recorded in out/; the inputs are identified by "
                             "inputs_sha256, so cherry-pick, rebase and squash do not change out/"}
