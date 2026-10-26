#!/usr/bin/env python3
"""Resolve generated-number placeholders in the manuscript.

The handbook is Markdown, not LaTeX, so the registry is injected through
``{{N:metric.id}}`` tokens instead of ``\\newcommand`` macros.  The contract is
the same one the directive asks for: a protected value is typed once, in
``numbers.json``, and every manuscript occurrence is rendered from it.  The
substitution is applied by the PDF builder, so a stale PDF cannot be produced
from a changed registry without the build picking it up.

    {{N:sweep.ff112.clean}}            -> 73.79 %
    {{N:paired.ff_minus_rec_ppg}}      -> +5.29 pp
    {{N:cost.macs_ff}}                 -> 1.456 GMAC

Usage:
    .venv/bin/python scripts/handbook/render_numbers.py --coverage
    .venv/bin/python scripts/handbook/render_numbers.py --check
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs" / "rhan_nxa" / "data" / "numbers.json"
SRC = ROOT / "docs" / "rhan_nxa"
TOKEN = re.compile(r"\{\{N:([a-z0-9_.]+)\}\}")


class UnknownMetric(KeyError):
    pass


def load() -> dict:
    with open(REGISTRY, encoding="utf-8") as f:
        return json.load(f)["metrics"]


def resolve(text: str, metrics: dict | None = None, where: str = "<text>") -> str:
    """Replace every placeholder; raise rather than ship an unresolved token."""
    if metrics is None:
        metrics = load()

    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key not in metrics:
            raise UnknownMetric(f"{where}: unknown metric id {key!r}")
        return metrics[key]["display"]

    out = TOKEN.sub(repl, text)
    if "{{N:" in out:
        raise ValueError(f"{where}: malformed placeholder left in output")
    return out


def manuscript_files() -> list[pathlib.Path]:
    return sorted(p for p in SRC.rglob("*.md") if "data" not in p.parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail on unknown ids or unresolved tokens")
    ap.add_argument("--coverage", action="store_true",
                    help="list metrics with no manuscript consumer")
    args = ap.parse_args()

    metrics = load()
    used: dict[str, list[str]] = {}
    bad = 0
    for path in manuscript_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in TOKEN.finditer(text):
            key = m.group(1)
            used.setdefault(key, []).append(str(path.relative_to(SRC)))
            if key not in metrics:
                print(f"ERROR {path.relative_to(ROOT)}: unknown metric {key}")
                bad += 1
        if "{{N:" in TOKEN.sub("", text):
            print(f"ERROR {path.relative_to(ROOT)}: malformed placeholder")
            bad += 1

    if args.coverage:
        unused = sorted(set(metrics) - set(used))
        print(f"{len(used)} of {len(metrics)} metrics are consumed by the manuscript")
        for key in unused:
            print(f"  no consumer: {key}")
    else:
        print(f"{len(used)} distinct metrics used in {len(manuscript_files())} files")
    return 1 if (bad and args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
