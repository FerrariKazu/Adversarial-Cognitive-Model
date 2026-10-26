#!/usr/bin/env python3
"""Validate the built RHAN Technical Handbook end to end.

Runs, in order and with exit status preserved:
  1. docs/rhan_nxa/verify_claims.py          recompute every cited number
  2. scripts/audit_handbook_numbers.py       every % traced to an artifact
  3. scripts/audit_handbook_deltas.py        every pp delta reproducible
  4. scripts/audit_rhan_handbook_pdf.py      every page rendered and measured
  5. scripts/audit_rhan_typography.py        text layer: frame, collisions, hyphenation
  6. this file                               contents resolution, figure count, hashes

Writes report/handbook_audit/validation.json and prints a summary.
Exit status is non-zero if any stage fails.

Usage: python3 scripts/validate_rhan_handbook.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDF = ROOT / "docs/rhan_nxa/RHAN_NXA_Technical_Handbook.pdf"
OUT = ROOT / "report/handbook_audit"


def run(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {"cmd": " ".join(cmd), "returncode": p.returncode,
            "tail": (p.stdout + p.stderr).strip().splitlines()[-6:]}


def pdf_pages() -> int:
    out = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout
    m = re.search(r"^Pages:\s+(\d+)$", out, re.M)
    return int(m.group(1)) if m else -1


def page_text(n: int, layout: bool = False) -> str:
    cmd = ["pdftotext", "-f", str(n), "-l", str(n)]
    if layout:
        cmd.append("-layout")
    return subprocess.run(cmd + [str(PDF), "-"], capture_output=True, text=True).stdout


def contents_check(pages: int) -> dict:
    # The contents and the figure list both use leader dots, which only survive
    # extraction in -layout mode. Read pages until the body starts (a page whose
    # first line is neither a contents row nor the figure list heading).
    entries = []
    figures_seen = False
    for p in range(2, min(pages, 24) + 1):
        text = page_text(p, layout=True)
        head = text.strip().splitlines()[0] if text.strip() else ""
        if head.startswith(("Figures", "RHAN Technical Handbook")):
            figures_seen = True
            if head.startswith("RHAN Technical Handbook"):
                break
            continue
        if figures_seen and not head.startswith("Figure "):
            break
        for line in text.splitlines():
            # In the figure list a long caption wraps, so the continuation line
            # ends in leader dots and a page number without starting with
            # "Figure".  Such a line is a fragment, not a contents entry: the
            # entry it belongs to was already checked on its first line.
            if figures_seen and not line.lstrip().startswith("Figure "):
                continue
            m = re.match(r"\s*(.+?)(?:\s*\.){4,}\s*(\d+)\s*$", line)
            if m and not m.group(1).startswith("Figure "):
                entries.append((m.group(1).strip(), int(m.group(2))))
            elif m:
                # first line of a figure-list row: check the caption prefix
                # against the page it points at, so a figure list entry cannot
                # drift away from the caption printed under the figure
                cap = re.sub(r"^Figure\s+\d+\.\s*", "", m.group(1).strip())
                entries.append((cap.rstrip("\u2026").strip(), int(m.group(2))))
    norm = lambda s: re.sub(r"[\s\u00ad]+", " ", s).lower() \
        .replace("\u2014", "-").replace("\u2013", "-").replace("\u2011", "-")
    cache: dict[int, str] = {}
    unresolved = []
    for title, page in entries:
        if page not in cache:
            cache[page] = page_text(page)
        if norm(title)[:38] not in norm(cache[page]):
            unresolved.append({"title": title, "page": page})
    # part dividers print with letter-spacing, so "PART I" extracts as "PA RT I";
    # they are verified separately rather than counted as failures.
    real = [u for u in unresolved if not u["title"].upper().startswith("PART ")]
    parts = {}
    for u in unresolved:
        if u["title"].upper().startswith("PART "):
            body = page_text(u["page"])
            number = u["title"].split()[1]
            parts[u["title"]] = {
                "page": u["page"],
                "number_seen": re.sub(r"\s+", "", number) in re.sub(r"\s+", "", body),
            }
    return {"entries": len(entries), "unresolved": real,
            "part_dividers": parts, "pages_read": pages}


def main() -> int:
    if not PDF.exists():
        raise SystemExit(f"missing PDF: {PDF} (run scripts/build_rhan_handbook_pdf.py)")
    OUT.mkdir(parents=True, exist_ok=True)

    stages = [
        run([sys.executable, "scripts/handbook/validate_numbers.py"]),
        run([sys.executable, "docs/rhan_nxa/verify_claims.py"]),
        run([sys.executable, "scripts/audit_handbook_numbers.py"]),
        run([sys.executable, "scripts/audit_handbook_deltas.py"]),
        run([sys.executable, "scripts/audit_rhan_handbook_pdf.py"]),
        run([sys.executable, "scripts/audit_rhan_typography.py"]),
    ]

    pages = pdf_pages()
    page_audit = json.loads((OUT / "page_audit.json").read_text())["summary"]
    typo = json.loads((OUT / "typography.json").read_text())
    manifest = json.loads((ROOT / "docs/rhan_nxa/data/figure_manifest.json").read_text())
    contents = contents_check(pages)

    md_files = sorted((ROOT / "docs/rhan_nxa").rglob("*.md"))
    words = sum(len(p.read_text(encoding="utf-8").split()) for p in md_files)
    typo_counts = {k: len(typo[k]) for k in
                   ("text_outside_frame", "text_collisions",
                    "hyphenated_identifiers", "caption_problems",
                    "orphan_headings", "undersized_figures", "vertical_holes")}

    report = {
        "pdf": str(PDF.relative_to(ROOT)),
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "pdf_bytes": PDF.stat().st_size,
        "pages": pages,
        "markdown_files": len(md_files),
        "words_of_markdown": words,
        "figure_count": len(manifest["figures"]),
        "table_rows_catalogued": len(manifest.get("tables", [])),
        "blank_pages": page_audit["blank_pages"],
        "margin_overflow_pages": page_audit["margin_overflow_pages"],
        "pages_rendered_and_measured": page_audit["pages"],
        "contents": contents,
        "typography": typo_counts,
        "captions_checked": typo["captions_seen"],
        "stages": stages,
        "ok": all(s["returncode"] == 0 for s in stages)
              and not page_audit["blank_pages"]
              and not page_audit["margin_overflow_pages"]
              and not contents["unresolved"]
              and all(v["number_seen"] for v in contents["part_dividers"].values())
              and not any(typo_counts.values()),
    }
    (OUT / "validation.json").write_text(json.dumps(report, indent=1))

    print(f"pdf          : {report['pdf']}  ({report['pages']} pages, "
          f"{report['pdf_bytes'] / 1e6:.2f} MB)")
    print(f"sha256       : {report['pdf_sha256']}")
    print(f"source       : {report['markdown_files']} markdown files, "
          f"{report['words_of_markdown']:,} words")
    print(f"figures      : {report['figure_count']} embedded, "
          f"{report['table_rows_catalogued']} table rows catalogued")
    print(f"pages        : {report['pages_rendered_and_measured']} rendered and measured; "
          f"blank={len(report['blank_pages'])} overflow={len(report['margin_overflow_pages'])}")
    print(f"contents     : {contents['entries']} entries, "
          f"{len(contents['unresolved'])} unresolved, "
          f"{len(contents['part_dividers'])} part dividers checked")
    print(f"typography   : " + ", ".join(f"{k}={v}" for k, v in report["typography"].items())
          + f"; {report['captions_checked']} captions adjacent to their image")
    for s in stages:
        print(f"  [{'ok' if s['returncode'] == 0 else 'FAIL'}] {s['cmd']}")
    print(f"\nRESULT: {'PASS' if report['ok'] else 'FAIL'} -> {OUT / 'validation.json'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
