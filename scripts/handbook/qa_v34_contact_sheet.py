#!/usr/bin/env python3
"""Visual QA for the v3.4 figures, at the size the reader sees them.

Three things this checks that the automated validator cannot:

1. **Where each v3.4 figure lands in the PDF.** The caption title is searched in
   the rendered text, so the page number is a fact about the shipped PDF rather
   than about the manifest.
2. **The size the figure is printed at.** Width in millimetres on the page and
   the effective font size after scaling: a 170 mm drawing printed at 120 mm has
   its 6 pt labels rendered at 4.2 pt, which is the failure mode that produces a
   figure that looks fine in the generator and is unreadable in print.
3. **A contact sheet.** Small thumbnails side by side make inconsistencies
   (two greys, two arrowhead weights, a missing badge) visible in one glance.
   It does not replace looking at a dense diagram on its own.

    python3 scripts/handbook/qa_v34_contact_sheet.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import fitz  # PyMuPDF

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "rhan_nxa"
PDF = DOCS / "RHAN_NXA_Technical_Handbook.pdf"
OUT = ROOT / "report" / "handbook_audit_v34"
MANIFEST = DOCS / "data" / "figure_manifest_v34.json"


def main() -> int:
    if not PDF.exists():
        print(f"missing {PDF}; build the PDF first", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]

    doc = fitz.open(PDF)

    def norm(s: str) -> str:
        return " ".join(s.replace("`", "").replace("\u00ad", "").split())

    # The caption printed in the PDF is the markdown alt text.  The list of
    # figures truncates it at ~100 characters, so the needle is the *tail* of
    # the alt text: only the body caption contains it.
    alt: dict[str, str] = {}
    for path in sorted(DOCS.rglob("*.md")):
        if path.relative_to(DOCS).parts[0] == "07_deliverables":
            continue
        for m in re.finditer(r"!\[(.*?)\]\(\.\./figures/([^)\s]+)\)",
                             path.read_text(encoding="utf-8"), re.S):
            alt[m.group(2)] = m.group(1)

    pages_text = [norm(doc[i].get_text()) for i in range(doc.page_count)]
    report = []

    for rec in rows:
        name = rec["file"].split("/")[-1]
        needle = norm(alt.get(name, ""))[-60:]
        page = next((i for i, t in enumerate(pages_text) if needle and needle in t),
                    None)
        rects = []
        if page is not None:
            for img in doc[page].get_images(full=True):
                xref = img[0]
                for r in doc[page].get_image_rects(xref):
                    rects.append((r.width * 25.4 / 72, r.height * 25.4 / 72))
        wide = max((w for w, _ in rects), default=0.0)
        # The schematic is drawn at 170 mm; a narrower print scales its type.
        scale = wide / 170.0 if wide else 0.0
        smallest_pt = 5.8 * scale if scale else 0.0
        report.append({
            "id": rec["id"],
            "file": rec["file"].split("/")[-1],
            "page": (page + 1) if page is not None else None,
            "print_width_mm": round(wide, 1),
            "scale": round(scale, 2),
            "smallest_label_pt": round(smallest_pt, 2),
            "ok": bool(page is not None and smallest_pt >= 5.0),
        })

    # contact sheet: thumbnails, 2 columns
    thumbs = []
    for r in report:
        p = ROOT / "docs" / "rhan_nxa" / "figures" / r["file"]
        if not p.exists():
            continue
        pix = fitz.Pixmap(str(p))
        scale = 900 / max(pix.width, 1)
        if scale < 1.0:
            pix = fitz.Pixmap(pix, int(pix.width * scale), int(pix.height * scale),
                              None)
        thumbs.append(fitz.Pixmap(pix))

    if thumbs:
        from PIL import Image  # noqa: PLC0415  (only needed for the sheet)

        cols = 2
        imgs = [Image.open(ROOT / "docs" / "rhan_nxa" / "figures" / r["file"])
                .convert("RGB") for r in report]
        cell_w = max(im.width for im in imgs) + 20
        cell_h = max(im.height for im in imgs) + 30
        rows_n = (len(imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (cell_w * cols, cell_h * rows_n), "white")
        for i, im in enumerate(imgs):
            sheet.paste(im, ((i % cols) * cell_w + 10,
                             (i // cols) * cell_h + 10))
        out_png = OUT / "contact_sheet_v34.png"
        sheet.save(out_png)
        print(f"contact sheet -> {out_png.relative_to(ROOT)}")

    (OUT / "figure_placement_v34.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'id':<24} {'page':>5} {'mm':>7} {'scale':>6} {'min pt':>7}")
    for r in report:
        print(f"{r['id']:<24} {str(r['page']):>5} {r['print_width_mm']:>7.1f} "
              f"{r['scale']:>6.2f} {r['smallest_label_pt']:>7.2f}"
              f"{'' if r['ok'] else '   <-- CHECK'}")
    doc.close()
    return 0 if all(r["ok"] for r in report) else 1


if __name__ == "__main__":
    sys.exit(main())
