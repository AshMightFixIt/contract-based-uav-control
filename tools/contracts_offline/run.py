"""Script entry: ``python tools/contracts_offline/run.py [--check] [--out DIR]``.

Loads this folder as the package ``contracts_offline`` from its location
(``importlib.util.spec_from_file_location``), so the repo needs no
``tools/__init__.py`` and ``sys.path`` is left as it is.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True

_PACKAGE_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "contracts_offline", _PACKAGE_DIR / "__init__.py", submodule_search_locations=[str(_PACKAGE_DIR)])
_package = importlib.util.module_from_spec(_spec)
sys.modules["contracts_offline"] = _package
_spec.loader.exec_module(_package)

main = importlib.import_module("contracts_offline.cli").main

if __name__ == "__main__":
    sys.exit(main())
