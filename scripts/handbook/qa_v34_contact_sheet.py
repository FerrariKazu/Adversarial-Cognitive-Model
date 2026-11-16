#!/usr/bin/env python3
"""Print-size QA for the ten v3.4 schematics (release check, not a build step).

The schematics are drawn at 170 mm and printed inside the handbook's 170 mm text
block.  What matters is not the drawing size but the *printed* one: a label has a
point size only once it is placed on a page.  This script therefore reads the
shipped PDF, finds where each v3.4 image was actually placed, converts that box to
millimetres, and reports the effective size of the smallest label in the drawing.

    effective_pt = FS_BADGE * (printed_width_mm / PAGE_W)

`FS_BADGE` is imported from `figures/schematics.py`, the same module the drawings
use, so the floor cannot drift from the artwork it describes.  The script also
crops each figure as printed and tiles them into a contact sheet, which is what a
reviewer looks at instead of ten separate page renders.

Outputs (both under report/handbook_audit_v34/):
    figure_placement_v34.json   page, printed width (mm), scale, smallest label
    contact_sheet_v34.png       the ten figures as printed, labelled

Exit status: 0 when every figure was located and every label clears the floor
(MIN_LABEL_PT), 1 otherwise — a figure that was printed too small fails the check
instead of being reported and forgotten.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/rhan_nxa/figures"))

from schematics import FS_BADGE, PAGE_W  # noqa: E402

PDF = ROOT / "docs/rhan_nxa/RHAN_NXA_Technical_Handbook.pdf"
MANIFEST = ROOT / "docs/rhan_nxa/data/figure_manifest_v34.json"
OUT = ROOT / "report/handbook_audit_v34"
JSON_OUT = OUT / "figure_placement_v34.json"
SHEET_OUT = OUT / "contact_sheet_v34.png"

#: The floor this edition sets for a printed label (pt).  5 pt is the usual
#: journal minimum for figure text; 5.5 here leaves headroom for the RIP.
MIN_LABEL_PT = 5.5
MM_PER_PT = 25.4 / 72.0


def png_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def fingerprint(path: Path) -> tuple[tuple[int, int], bytes]:
    """A cheap content fingerprint: 32x32 greyscale, one byte per pixel.

    Pixel dimensions alone are not unique — two schematics can share a canvas
    size — so the placement is resolved by content, not by size.
    """
    with Image.open(path) as im:
        return im.size, im.convert("L").resize((32, 32), Image.BILINEAR).tobytes()


def locate(doc: "fitz.Document", target: tuple[tuple[int, int], bytes]
           ) -> list[tuple[int, "fitz.Rect"]]:
    """Every page on which *this* image is actually drawn.

    WeasyPrint pools image XObjects document-wide, so `get_images()` lists far
    more images than a page draws; `get_image_rects()` returns only the boxes the
    page's content stream paints, which is the placement we want.  A page can
    draw several images of the same pixel size, so the embedded bytes are decoded
    and compared with the PNG by fingerprint.
    """
    size, sig = target
    decode_cache: dict[int, bytes] = {}
    hits = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            if (img[2], img[3]) != size:
                continue
            # The page resource dict is pooled document-wide, so the same xref is
            # listed by many pages; decode each once and let get_image_rects()
            # decide whether this page actually paints it.
            if xref not in decode_cache:
                raw = doc.extract_image(xref)["image"]
                with Image.open(io.BytesIO(raw)) as emb:
                    decode_cache[xref] = emb.convert("L").resize(
                        (32, 32), Image.BILINEAR).tobytes()
            if decode_cache[xref] != sig:
                continue
            for rect in page.get_image_rects(xref):
                hits.append((page.number + 1, rect))
    return hits


def crop(page: "fitz.Page", rect: "fitz.Rect", dpi: int = 200) -> Image.Image:
    pix = page.get_pixmap(clip=rect, dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = json.loads(MANIFEST.read_text())["figures"]
    doc = fitz.open(PDF)

    rows, crops = [], []
    for fig in figures:
        path = ROOT / fig["file"]
        size, sig = fingerprint(path)
        hits = locate(doc, (size, sig))
        if len(hits) != 1:
            pages = [p for p, _ in hits]
            raise SystemExit(
                f"{fig['id']}: expected exactly one placement of {size[0]}x{size[1]} px, "
                f"found {len(hits)} (pages {pages})")
        page_no, rect = hits[0]
        width_mm = rect.width * MM_PER_PT
        scale = width_mm / PAGE_W
        min_pt = FS_BADGE * scale
        rows.append({
            "id": fig["id"],
            "file": Path(fig["file"]).name,
            "page": page_no,
            "print_width_mm": round(width_mm, 1),
            "scale": round(scale, 2),
            "smallest_label_pt": round(min_pt, 2),
            "ok": bool(min_pt >= MIN_LABEL_PT),
        })
        crops.append((fig["id"], page_no, crop(doc[page_no - 1], rect)))

    rows.sort(key=lambda r: r["page"])   # stable, page order, for diffing
    JSON_OUT.write_text(json.dumps(rows, indent=2) + "\n")

    # --- contact sheet: 2 columns x 5 rows, caption strip under each figure ----
    col_w = max(c.width for _, _, c in crops)
    thumbs = [(fig_id, page, im.resize(
        (int(im.width * col_w / im.width), int(im.height * col_w / im.width)),
        Image.LANCZOS)) for fig_id, page, im in crops]
    strip = 26
    row_h = max(t.height for _, _, t in thumbs) + strip
    pad, cols = 16, 2
    rows_n = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (pad + cols * (col_w + pad),
                              pad + rows_n * (row_h + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (fig_id, page, thumb) in enumerate(thumbs):
        x = pad + (i % cols) * (col_w + pad)
        y = pad + (i // cols) * (row_h + pad)
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y, x + col_w - 1, y + thumb.height - 1], outline="#d8d5cc")
        row = rows[i]
        draw.text((x, y + thumb.height + 8),
                  f"{fig_id}  ·  p. {row['page']}  ·  "
                  f"{row['print_width_mm']} mm  ·  min {row['smallest_label_pt']} pt",
                  fill="#333333")
    sheet.save(SHEET_OUT)

    print(f"contact sheet -> {SHEET_OUT.relative_to(ROOT)}")
    print(f"{'id':24} {'page':>5} {'mm':>8} {'scale':>6} {'min pt':>7}")
    for r in rows:
        flag = "" if r["ok"] else "  BELOW FLOOR"
        print(f"{r['id']:24} {r['page']:5d} {r['print_width_mm']:8.1f} "
              f"{r['scale']:6.2f} {r['smallest_label_pt']:7.2f}{flag}")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
