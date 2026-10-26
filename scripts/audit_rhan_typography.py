#!/usr/bin/env python3
"""Typographic audit of the rendered RHAN handbook.

`audit_rhan_handbook_pdf.py` measures ink at page scale; this script reads the
PDF text layer and checks the four defect classes that a page-image measurement
cannot see:

  * body text outside the declared text block (a table or identifier that
    overflows the 20 mm right margin);
  * two text spans drawn on top of each other (a collision, not a wrap);
  * an identifier broken across a line by automatic hyphenation
    (`pg-d50_no_ran-dom_start`), which silently corrupts a path or key;
  * a figure caption separated from its image by more than a page element.

The declared page box comes from the build stylesheet: A4, margins
22 mm top / 20 mm right / 20 mm bottom / 20 mm left.  Running heads and the
folio live in those margins by design and are excluded from the frame check.

Writes report/handbook_audit/typography.json; exits 1 if any check fails.

Usage: python3 scripts/audit_rhan_typography.py [--pdf PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import unicodedata

import fitz

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "docs/rhan_nxa/RHAN_NXA_Technical_Handbook.pdf"
OUT = ROOT / "report/handbook_audit"
SRC = ROOT / "docs/rhan_nxa"

MM = 72.0 / 25.4
PAGE_W, PAGE_H = 595.276, 841.89
FRAME = {
    "x0": 20 * MM,
    "x1": PAGE_W - 20 * MM,
    "y0": 22 * MM,
    "y1": PAGE_H - 20 * MM,
}
TOL = 1.5           # pt of slack: glyph bboxes include side bearings
CAPTION_GAP_MAX = 26.0   # pt between the image bottom and the caption baseline


def is_combining(text: str) -> bool:
    """True for a span that is only a combining accent (x̃, d̄) drawn over its base."""
    body = text.strip()
    return bool(body) and all(unicodedata.combining(ch) for ch in body)


def spans(page: fitz.Page) -> list[dict]:
    """Every text span with its bbox, size and page-space rect."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                txt = s["text"]
                if not txt.strip():
                    continue
                out.append({
                    "text": txt,
                    "bbox": tuple(round(v, 2) for v in s["bbox"]),
                    "size": round(s["size"], 2),
                })
    return out


def image_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Placed image boxes. `get_image_info` is one pass; `get_images` +
    `get_image_rects` is O(document) per page because WeasyPrint shares one
    resource dictionary across all 234 pages."""
    return [fitz.Rect(i["bbox"]) for i in page.get_image_info()]


def in_margin_band(bbox: tuple[float, float, float, float]) -> bool:
    """Running head (above the text block) or folio (below it)."""
    return bbox[3] <= FRAME["y0"] + 2 or bbox[1] >= FRAME["y1"] - 2


def overflows(bbox) -> bool:
    x0, y0, x1, y1 = bbox
    return (x0 < FRAME["x0"] - TOL or x1 > FRAME["x1"] + TOL
            or y0 < FRAME["y0"] - TOL or y1 > FRAME["y1"] + TOL)


def overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def area(r) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


IDENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-/]*[A-Za-z0-9_]$")


def identifiers() -> set[str]:
    """Every backticked identifier in the manuscript worth protecting."""
    ids: set[str] = set()
    for p in SRC.rglob("*.md"):
        for m in re.finditer(r"`([^`\n]{4,90})`", p.read_text(encoding="utf-8")):
            tok = m.group(1).strip()
            if IDENT_RE.match(tok) and ("_" in tok or "/" in tok or "-" in tok):
                ids.add(tok)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(DEFAULT_PDF))
    args = ap.parse_args()
    pdf = pathlib.Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"missing PDF: {pdf}")

    doc = fitz.open(pdf)
    known = identifiers()

    frame_hits, collisions, hyphens, caption_gaps = [], [], [], []
    orphans, small_figs, gaps = [], [], []
    seen_caption = 0

    dividers = []
    for pno, page in enumerate(doc, start=1):
        sp = spans(page)
        # a part divider is mostly white by design: it carries the part number,
        # its title and the chapter list, and nothing else.  The hole check
        # would flag every one of them, so they are named and skipped.
        # The divider title is letter-spaced on the page, so "PART I" extracts
        # as "PA RT  I" and "APPENDICES" as "A P P E N D I C E S".  Compare the
        # text with the spacing removed, which is what the reader sees.
        squeezed = re.sub(r"\s+", "", page.get_text() or "").upper()
        is_divider = squeezed.startswith("PART") or squeezed.startswith("APPENDICES")

        # a heading must not be the last thing on the page: it would arrive
        # separated from the paragraph it introduces
        body = [s for s in sp if not in_margin_band(s["bbox"])]
        if body:
            lowest = max(body, key=lambda s: s["bbox"][3])
            if lowest["size"] >= 11.5 and lowest["bbox"][3] > FRAME["y1"] - 80:
                orphans.append({"page": pno, "heading": lowest["text"][:60],
                                "bottom_pt": round(lowest["bbox"][3], 1)})

        # a figure much narrower than the column is either a mistake in the
        # generator or a deliberate inset; the handbook has no insets
        boxes = image_rects(page)
        for r in boxes:
            width = r.x1 - r.x0
            if width < 0.55 * (FRAME["x1"] - FRAME["x0"]):
                small_figs.append({"page": pno, "width_pt": round(width, 1),
                                   "height_pt": round(r.y1 - r.y0, 1)})

        # a hole in the middle of the text block: a bad page break that leaves
        # the reader with half a page of white.  Images occupy space with no
        # text, so the bands include them.
        if is_divider:
            dividers.append(pno)
        else:
            bands = [(s["bbox"][1], s["bbox"][3]) for s in body]
            bands += [(r.y0, r.y1) for r in boxes]
            bands.sort()
            prev_bottom = FRAME["y0"]
            for top, bottom in bands:
                if top > FRAME["y0"] + 1 and top - prev_bottom > 150:
                    gaps.append({"page": pno, "gap_pt": round(top - prev_bottom, 1),
                                 "starts_at_pt": round(top, 1)})
                prev_bottom = max(prev_bottom, bottom)

        for s in sp:
            if in_margin_band(s["bbox"]):
                continue
            if overflows(s["bbox"]):
                frame_hits.append({"page": pno, "text": s["text"][:70],
                                   "bbox": s["bbox"]})

        # collisions: two spans sharing more than a sliver of their own area.
        # Combining marks share their base glyph's box by construction.
        for i, a in enumerate(sp):
            if is_combining(a["text"]):
                continue
            for b in sp[i + 1:]:
                if is_combining(b["text"]):
                    continue
                ov = overlap(a["bbox"], b["bbox"])
                if ov <= 4.0:
                    continue
                small = min(area(a["bbox"]), area(b["bbox"])) or 1.0
                if ov / small > 0.30:
                    collisions.append({"page": pno, "a": a["text"][:50],
                                       "b": b["text"][:50], "overlap_pt2": round(ov, 1)})

        # hyphenated identifiers: a line ending in "-" whose join is a real key
        lines = page.get_text("text").split("\n")
        for j in range(len(lines) - 1):
            m = re.search(r"([A-Za-z0-9_./]{2,})[-\u2010\u00ad]$", lines[j].strip())
            if not m:
                continue
            nxt = re.match(r"\s*([A-Za-z0-9_./]{1,})", lines[j + 1])
            if not nxt:
                continue
            joined = m.group(1) + nxt.group(1)
            if joined in known or m.group(1) in known:
                hyphens.append({"page": pno, "broken": lines[j].strip()[-40:],
                                "rest": lines[j + 1].strip()[:40],
                                "reads_as": joined})

        # captions: the caption span is "Figure N." inside its own <figure>
        # The list of figures repeats "Figure N." as a link.  Those entries are
        # navigation, not captions, so a caption is a span outside any link.
        link_rects = [fitz.Rect(l["from"]) for l in page.get_links()]
        for s in sp:
            if not re.match(r"^Figure \d+\.$", s["text"].strip()):
                continue
            box = fitz.Rect(s["bbox"])
            if any(box.intersects(r) for r in link_rects):
                continue
            seen_caption += 1
            cap_top = s["bbox"][1]
            above = [b for b in boxes if b.y1 <= cap_top + 1]
            if not above:
                caption_gaps.append({"page": pno, "caption": s["text"].strip(),
                                     "gap_pt": None,
                                     "why": "no image above the caption on this page"})
                continue
            nearest = max(b.y1 for b in above)
            gap = round(cap_top - nearest, 1)
            if gap > CAPTION_GAP_MAX:
                caption_gaps.append({"page": pno, "caption": s["text"].strip(),
                                     "gap_pt": gap, "why": f"gap exceeds {CAPTION_GAP_MAX} pt"})

    report = {
        "pdf": str(pdf.relative_to(ROOT)),
        "pages": doc.page_count,
        "frame": {k: round(v, 1) for k, v in FRAME.items()},
        "text_outside_frame": frame_hits,
        "text_collisions": collisions,
        "hyphenated_identifiers": hyphens,
        "caption_problems": caption_gaps,
        "orphan_headings": orphans,
        "undersized_figures": small_figs,
        "vertical_holes": gaps,
        "part_divider_pages": dividers,
        "captions_seen": seen_caption,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "typography.json").write_text(json.dumps(report, indent=1))

    ok = True
    for key in ("text_outside_frame", "text_collisions",
                "hyphenated_identifiers", "caption_problems",
                "orphan_headings", "undersized_figures", "vertical_holes"):
        n = len(report[key])
        print(f"  [{'ok' if n == 0 else 'FAIL'}] {key}: {n}")
        if n:
            ok = False
            for rec in report[key][:6]:
                print("        ", json.dumps(rec)[:180])
    print(f"  captions checked: {seen_caption}")
    print(f"\nwrote {OUT / 'typography.json'}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
