"""
Run the code examples of the documentation, so the docs cannot go stale.

Every ```python block of a page runs in order, in one shared namespace, in a
temporary directory where the repository's ``data/`` folder is available.

Inline results are checked too: in ``print(expr)  # value`` the printed text
must equal ``value``, and lines of ``# ...`` comments right after a
``print`` in a loop are not checked (they are illustrative).

A block preceded by ``<!-- no-test -->`` is skipped (e.g. shell commands or
code needing external services).
"""

from __future__ import annotations

import contextlib
import io
import os
import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = sorted(
    [DOCS / "index.md", DOCS / "getting-started.md", *sorted((DOCS / "guide").glob("*.md"))],
    key=str,
)
BLOCK = re.compile(r"(<!-- no-test -->\s*)?```python\n(.*?)```", re.S)
INLINE_RESULT = re.compile(r"^(\s*)print\((.*)\)\s+#\s+(.+)$")


def python_blocks(page: Path) -> list[str]:
    return [code for skip, code in BLOCK.findall(page.read_text()) if not skip]


def checked(code: str) -> str:
    """Turn ``print(x)  # value`` lines into assertions on the printed text."""
    lines = []
    for line in code.splitlines():
        match = INLINE_RESULT.match(line)
        if match:
            indent, expr, expected = match.groups()
            lines.append(
                f"{indent}assert _printed({expr}) == {expected.strip()!r}, "
                f"(_printed({expr}), {expected.strip()!r})"
            )
        else:
            lines.append(line)
    return "\n".join(lines)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_page_examples_run(page, tmp_path, monkeypatch):
    blocks = python_blocks(page)
    if not blocks:
        pytest.skip("no Python examples")

    (tmp_path / "data").symlink_to(ROOT / "data")
    monkeypatch.chdir(tmp_path)
    # What print(*args) would output, for checking "# value" comments.
    namespace: dict = {"__name__": "__docs__", "_printed": lambda *a: " ".join(map(str, a))}

    for number, code in enumerate(blocks, start=1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(checked(code), f"{page.name}[block {number}]", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - reported to the user
            raise AssertionError(
                f"{page.relative_to(DOCS)}, block {number} failed: {exc!r}\n\n{code}"
            ) from exc

    os.chdir(ROOT)
