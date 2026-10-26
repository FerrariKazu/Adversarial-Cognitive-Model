#!/usr/bin/env python3
"""
Check the Step 7.5 recovery cell text exactly as it would appear if copied
into a Kaggle code cell.

This is not a substitute for the notebook itself, but it helps confirm that
the recovery block in cloud_setup/Kaggle_J1_FOUNDATION.py does not contain a
dangling/unterminated string literal when treated as a lone code cell.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "cloud_setup" / "Kaggle_J1_FOUNDATION.py"


def extract_recovery_cell_text(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    # The recovery cell begins at the line containing this marker.
    marker = "if DRY_RUN:\n    print(\"[DRY-RUN] RECOVERY PRE-FLIGHT exists"
    start = src.find(marker)
    if start == -1:
        raise SystemExit("STOP: recovery cell marker not found in notebook file")
    # The cell code effectively runs until the dispatch block below it. We
    # slice to a safe upper bound: the next markdown step header after the
    # recovery block.
    tail = src[start:]
    end_marker = "# %% [markdown]\n# ## Step 7:"
    end = tail.find(end_marker)
    if end == -1:
        # Fallback: take everything after the marker.
        cell = tail
    else:
        cell = tail[:end]
    return cell


def main() -> int:
    try:
        cell_text = extract_recovery_cell_text(NOTEBOOK)
    except SystemExit as e:
        print(e)
        return 2

    # Normalize the cell the way a notebook runner might: collapse leading
    # markdown-ish comment lines so the first executed line is the recovery
    # condition block.
    lines = cell_text.splitlines()
    body_lines = []
    started = False
    for line in lines:
        if not started:
            if line.startswith("# %%") or line.startswith("#"):
                continue
            started = True
        body_lines.append(line)
    body = "\n".join(body_lines)

    try:
        ast.parse(body)
    except SyntaxError as e:
        print("SYNTAX ERROR in extracted recovery cell:")
        print(f"  line {e.lineno}: {e.msg}")
        print(f"  text: {e.text!r}")
        return 1

    print("Recovery cell extraction parses OK.")
    print(f"  notebook: {NOTEBOOK}")
    print(f"  cell line count: {body.count(chr(10)) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
