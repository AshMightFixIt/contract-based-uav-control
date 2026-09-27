"""Script entry: ``python tools/contracts_offline/run.py [--check] [--out DIR]``.

Imports this folder as the package ``contracts_offline`` by putting ``tools/``
on ``sys.path``, so the repo needs no ``tools/__init__.py``.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts_offline.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
