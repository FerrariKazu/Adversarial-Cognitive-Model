#!/usr/bin/env python3
"""Validate the numeric registry, the claim graph, the lineage and the figures.

This is the stage the PDF validator calls.  It checks traceability and
consistency; it does **not** claim that any experiment is scientifically
correct.  A registry can be internally perfect and still hold a wrong number —
that is what ``verify_claims.py`` and the audit reports are for.

Checks
------
1. The registry on disk matches a fresh build (no stale value).
2. Schema constraints, provenance requirements, display formatting, derived
   values that must close.
3. Every ``{{N:...}}`` token in the manuscript resolves; no malformed tokens.
4. A retracted metric never appears as a literal in the executive overview, and
   any literal occurrence elsewhere states nearby that it is retracted.
5. The claim graph only cites metrics and files that exist, and never marks a
   claim VERIFIED while citing a RETRACTED metric.
6. Lineage edges reference existing nodes; cumulative epochs close.
7. Every figure has an existing file, a status, alt text and a known data id.

Exit status is non-zero on any ERROR.  Findings are printed with file and id.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "rhan_nxa" / "data"
SRC = ROOT / "docs" / "rhan_nxa"

sys.path.insert(0, str(ROOT / "scripts" / "handbook"))
import build_numbers  # noqa: E402
import render_numbers  # noqa: E402

ERRORS: list[str] = []
WARNINGS: list[str] = []


def err(msg: str) -> None:
    ERRORS.append(msg)


def warn(msg: str) -> None:
    WARNINGS.append(msg)


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def check_fresh() -> None:
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/handbook/build_numbers.py"),
                           "--check"], capture_output=True, text=True)
    if proc.returncode != 0:
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.startswith(("STALE", "MISSING", "REGISTRY", "  -")):
                err(line)
        if not proc.stdout.strip():
            err("build_numbers.py --check failed with no diagnostic")


def check_schema(doc: dict) -> None:
    for e in build_numbers.validate(doc):
        err(e)
    for mid, m in doc["metrics"].items():
        if (m["status"] == "VERIFIED" and not m["derivation"]
                and not m["source_artifacts"]):
            # code-only provenance is legitimate for constants read from source
            if not m["source_code"]:
                warn(f"{mid}: VERIFIED with no provenance at all")


def manuscript() -> list[pathlib.Path]:
    return sorted(p for p in SRC.rglob("*.md") if "data" not in p.parts)


def check_placeholders(metrics: dict) -> None:
    for path in manuscript():
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(ROOT))
        for m in render_numbers.TOKEN.finditer(text):
            if m.group(1) not in metrics:
                err(f"{rel}: unknown metric id {m.group(1)}")
        stripped = render_numbers.TOKEN.sub("", text)
        if "{{N:" in stripped:
            err(f"{rel}: malformed placeholder")


def check_retracted(metrics: dict) -> None:
    retracted = {mid: m for mid, m in metrics.items() if m["status"] == "RETRACTED"}
    for mid, m in retracted.items():
        literal = m["display"]
        for path in manuscript():
            text = path.read_text(encoding="utf-8")
            rel = str(path.relative_to(ROOT))
            start = 0
            while True:
                i = text.find(literal, start)
                if i < 0:
                    break
                start = i + len(literal)
                if "{{N:" in text[max(0, i - 40):i]:
                    continue
                window = text[max(0, i - 400):i + 400].lower()
                if "retract" not in window and "invalid" not in window:
                    if "01_executive_overview" in rel:
                        err(f"{rel}: retracted value {literal} ({mid}) appears in the "
                            f"executive overview without a retraction note")
                    else:
                        warn(f"{rel}: literal {literal} ({mid}) without a nearby "
                             f"retraction note")


def check_claims(metrics: dict) -> None:
    claims = load("claims.json")["claims"]
    ids = [c["id"] for c in claims]
    if len(ids) != len(set(ids)):
        err("claims.json: duplicate claim id")
    for c in claims:
        for mid in c["metric_ids"]:
            if mid not in metrics:
                err(f"claim {c['id']}: unknown metric {mid}")
            elif metrics[mid]["status"] == "RETRACTED" and c["status"] in {
                    "VERIFIED", "SUPPORTED_WITH_LIMITATIONS"}:
                err(f"claim {c['id']}: cites retracted metric {mid} as {c['status']}")
        for loc in c["locations"]:
            if not (SRC / loc).exists():
                err(f"claim {c['id']}: location does not exist: {loc}")
        for art in c["artifacts"]:
            if not (ROOT / art).exists():
                err(f"claim {c['id']}: artifact missing: {art}")


def check_lineage() -> None:
    man = load("training_lineage.json")
    nodes = {n["id"] for n in man["nodes"]}
    for e in man["edges"]:
        if e["from"] not in nodes or e["to"] not in nodes:
            err(f"lineage: edge {e['from']}->{e['to']} references an unknown node")
    for nid, cum in man["cumulative_epochs"].items():
        node = next((n for n in man["nodes"] if n["id"] == nid), None)
        if node is None:
            err(f"lineage: cumulative epoch entry for unknown node {nid}")
            continue
        if cum < node["epochs"]:
            err(f"lineage: {nid} cumulative {cum} < phase epochs {node['epochs']}")


def check_experiments() -> None:
    import hashlib

    man = load("experiment_manifest.json")
    for e in man["experiments"]:
        path = ROOT / e["path"]
        if not path.exists():
            err(f"experiment {e['id']}: artifact missing {e['path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if e["sha256"] and digest != e["sha256"]:
            err(f"experiment {e['id']}: artifact changed since the registry was built "
                f"({e['path']}) — rebuild the registry")


def check_figures(metrics: dict) -> None:
    man = load("figure_manifest.json")
    seen = set()
    for f in man["figures"]:
        if f["id"] in seen:
            err(f"figure {f['id']}: duplicate id")
        seen.add(f["id"])
        if not (ROOT / f["file"]).exists():
            err(f"figure {f['id']}: file missing {f['file']}")
        if f["status"] not in {"EVIDENCE", "CONCEPTUAL"}:
            err(f"figure {f['id']}: status must be EVIDENCE or CONCEPTUAL")
        if not f.get("alt_text"):
            err(f"figure {f['id']}: missing alt text")
        if f.get("embedded_number"):
            err(f"figure {f['id']}: image carries an embedded figure number")
        for mid in f.get("data_ids", []):
            if mid not in metrics:
                err(f"figure {f['id']}: unknown data id {mid}")
        # every figure must be referenced by some chapter
        name = f["file"].split("/")[-1]
        if not any(name in p.read_text(encoding="utf-8") for p in manuscript()):
            err(f"figure {f['id']}: not referenced by any chapter ({name})")


def main() -> int:
    metrics = load("numbers.json")["metrics"]
    check_fresh()
    check_schema(load("numbers.json"))
    check_placeholders(metrics)
    check_retracted(metrics)
    check_claims(metrics)
    check_lineage()
    check_experiments()
    check_figures(metrics)

    for w in WARNINGS:
        print(f"WARN  {w}")
    for e in ERRORS:
        print(f"ERROR {e}")
    print(f"registry: {len(metrics)} metrics; "
          f"{sum(1 for m in metrics.values() if m['status'] == 'RETRACTED')} retracted; "
          f"{len(ERRORS)} errors, {len(WARNINGS)} warnings")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
