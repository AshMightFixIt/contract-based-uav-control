"""Guards for the two-level package layout ``contract_uav_core/contract_uav_core/``.

With the default editable install (``pip install -e ./contract_uav_core``,
setuptools' import-hook finder), a Python started in the repository root
imports the outer project folder ``contract_uav_core/`` as a namespace
package: the inner ``__init__.py`` never runs there, while the submodules still
resolve to the inner package. Code in that ``__init__.py`` would therefore run
or not depending on the working directory. These tests keep the file
docstring-only and check that the core still resolves from the repository root.
See docs/DEVELOPMENT.md, "Package layout".
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "contract_uav_core" / "contract_uav_core"


def test_package_init_is_docstring_only():
    init = PACKAGE_DIR / "__init__.py"
    body = ast.parse(init.read_text(encoding="utf-8"), filename=str(init)).body
    first = body[0] if body else None
    is_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    code = [f"line {node.lineno}: {type(node).__name__}" for node in (body[1:] if is_docstring else body)]
    assert is_docstring and not code, (
        f"{init.relative_to(REPO_ROOT).as_posix()} must hold only a docstring (no imports, "
        "assignments or other code): from the repository root it is not executed "
        f"(see docs/DEVELOPMENT.md, 'Package layout'). Found: {code or 'no docstring'}"
    )


def test_core_resolves_into_the_package_from_the_repo_root(core_package, run_python):
    proc = run_python(
        ["-c", "import contract_uav_core.core as c; print(c.__file__)"], cwd=REPO_ROOT, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    found = Path(proc.stdout.strip().splitlines()[-1])
    assert found.samefile(core_package), found
    assert PACKAGE_DIR.resolve() in found.resolve().parents
