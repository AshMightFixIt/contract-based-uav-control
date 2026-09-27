"""Provenance: the source commit and a content hash of the inputs. No timestamps.

``source_commit`` is the last commit that touched any input file, not HEAD.
Committing the regenerated ``out/`` therefore does not make ``out/`` stale.
It needs full git history (CI checks out with ``fetch-depth: 0``).
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
COMMIT_RULE = "last commit that touched any input file: git log -1 --format=%H -- <inputs>"


def _lf_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _git(repo_root: Path, *args: str) -> Optional[bytes]:
    try:
        res = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, check=False)
    except OSError:
        return None
    return res.stdout if res.returncode == 0 else None


def input_hashes(repo_root: Path) -> dict:
    files: List[dict] = []
    h = hashlib.sha256()
    for rel in INPUT_PATHS:
        digest = hashlib.sha256(_lf_bytes((repo_root / rel).read_bytes())).hexdigest()
        files.append({"path": rel, "sha256": digest})
        h.update(f"{rel}\0{digest}\n".encode("utf-8"))
    return {"inputs": files, "inputs_sha256": h.hexdigest()}


def source_commit(repo_root: Path) -> dict:
    shallow = _git(repo_root, "rev-parse", "--is-shallow-repository")
    out = _git(repo_root, "log", "-1", "--format=%H", "--", *INPUT_PATHS)
    commit = out.decode().strip() if out else ""
    if not commit:
        return {"source_commit": None, "source_commit_matches_inputs": None,
                "source_commit_note": "git history unavailable"}
    if shallow is not None and shallow.decode().strip() == "true":
        return {"source_commit": None, "source_commit_matches_inputs": None,
                "source_commit_note": "shallow clone: history unavailable (use fetch-depth: 0)"}
    matches = True
    for rel in INPUT_PATHS:
        blob = _git(repo_root, "show", f"{commit}:{rel}")
        if blob is None or _lf_bytes(blob) != _lf_bytes((repo_root / rel).read_bytes()):
            matches = False
    return {"source_commit": commit, "source_commit_matches_inputs": matches,
            "source_commit_note": "" if matches else "working tree differs from source_commit; "
                                                        "trust inputs_sha256"}


def environment() -> dict:
    env = {}
    for pkg in ("pacti", "numpy", "scipy"):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[pkg] = None
    return env


def provenance(repo_root: Path) -> dict:
    return {**source_commit(repo_root), "source_commit_rule": COMMIT_RULE,
            **input_hashes(repo_root), "hash_rule": HASH_RULE}
