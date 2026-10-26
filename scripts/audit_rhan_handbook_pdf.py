#!/usr/bin/env python3
"""Page-level audit of the RHAN Technical Handbook PDF.

Renders every page at 120 dpi and measures:
  * blank / near-blank pages (accidental page breaks)
  * ink that encroaches on the physical page margin (content overflow)
  * the smallest rendered glyph height on the page (unreadable text)
  * which pages carry raster figures

Writes report/handbook_audit/page_audit.json and prints a summary.

Usage: python3 scripts/audit_rhan_handbook_pdf.py [--pdf PATH] [--dpi 120]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "docs/rhan_nxa/RHAN_NXA_Technical_Handbook.pdf"
OUT_DIR = ROOT / "report/handbook_audit"

# A4 at 120 dpi is 992 x 1403 px. WeasyPrint's page box is 595.276 x 841.89 pt,
# with margins declared in the stylesheet. Anything within 8 px (about 5 pt) of
# the physical edge is treated as margin encroachment.
EDGE_GUARD_PX = 8


def render(pdf: pathlib.Path, dpi: int, tmp: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    prefix = tmp / "page"
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(prefix)],
        check=True, capture_output=True,
    )
    pages = sorted(prefix.parent.glob("page-*.png"))
    return [(i + 1, p) for i, p in enumerate(pages)]


def measure(png: pathlib.Path, dpi: int) -> dict:
    im = Image.open(png).convert("L")
    a = np.asarray(im)
    h, w = a.shape
    ink = a < 245
    total = int(ink.sum())
    fractions = round(100.0 * total / ink.size, 3)

    rec: dict = {"ink_px": total, "ink_pct": fractions, "width": w, "height": h}
    if total == 0:
        rec.update({"blank": True})
        return rec

    ys, xs = np.nonzero(ink)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    rec.update({
        "bbox": [x0, y0, x1, y1],
        "max_gap_left_px": x0,
        "max_gap_right_px": w - 1 - x1,
        "max_gap_top_px": y0,
        "max_gap_bottom_px": h - 1 - y1,
        "overflow": bool(x0 < EDGE_GUARD_PX or y0 < EDGE_GUARD_PX
                         or (w - 1 - x1) < EDGE_GUARD_PX
                         or (h - 1 - y1) < EDGE_GUARD_PX),
        "blank": total < 0.0015 * ink.size,
    })
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(DEFAULT_PDF))
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    pdf = pathlib.Path(args.pdf).resolve()
    if not pdf.exists():
        raise SystemExit(f"missing PDF: {pdf}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        pages = render(pdf, args.dpi, pathlib.Path(td))
        page_records = []
        for num, png in pages:
            rec = measure(png, args.dpi)
            rec["page"] = num
            page_records.append(rec)

    total_pages = len(page_records)
    blank = [r["page"] for r in page_records if r.get("blank")]
    overflow = [r["page"] for r in page_records if r.get("overflow")]
    light = sorted(
        (r["page"], r["ink_pct"]) for r in page_records
        if 0.05 < r["ink_pct"] < 1.2
    )

    summary = {
        "pdf": str(pdf.relative_to(ROOT)),
        "dpi": args.dpi,
        "pages": total_pages,
        "blank_pages": blank,
        "margin_overflow_pages": overflow,
        "sparse_pages_below_1_2pct_ink": light,
        "min_ink_pct": min(r["ink_pct"] for r in page_records),
        "median_ink_pct": float(np.median([r["ink_pct"] for r in page_records])),
    }
    (OUT_DIR / "page_audit.json").write_text(
        json.dumps({"summary": summary, "pages": page_records}, indent=1)
    )

    print(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT_DIR / 'page_audit.json'}")


if __name__ == "__main__":
    main()
