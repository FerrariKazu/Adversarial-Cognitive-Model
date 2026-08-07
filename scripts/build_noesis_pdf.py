#!/usr/bin/env python3
"""Build docs/NOESIS_FOUNDATION.pdf from docs/NOESIS_FOUNDATION.md.

Pipeline
--------
    markdown -> preprocess (cover page + page-2 opening, drop handwritten TOC)
             -> pandoc -t html5 (fragments)
             -> assemble (cover + generated TOC + body) with CSS
             -> WeasyPrint (CSS paged media) -> PDF

Output layout
    page 1        full-bleed cover: the OfficialNOESIS artwork covers the
                  entire physical page edge-to-edge (zero margins, no text)
    page 2        title (NOESIS + subtitle) and the Preface
    pages 3-4     Table of Contents with REAL page numbers
    pages 5+      Chapters 1-9, then the Appendix

Why WeasyPrint?
    LuaLaTeX is unusable on this machine: texlive-luatex (luaotfload's Lua
    modules) is not installed and there is no passwordless sudo, so fontspec
    cannot load any OpenType font. wkhtmltopdf 0.12.6 hangs headless on this
    box. WeasyPrint is pure Python with full CSS paged-media support — @page
    margin boxes (running head, page-number footer), named pages (furniture-
    free cover and opening), and target-counter(), which lets the Table of
    Contents carry page numbers computed from the final layout.

The TOC entries are generated from the actual heading ids pandoc emits, so
every anchor matches. The two Mermaid blocks render as verbatim source
(readable, and each is paired with an ASCII diagram in the manuscript).

Usage
-----
    python3 scripts/build_noesis_pdf.py                  # 300 dpi cover (default)
    python3 scripts/build_noesis_pdf.py --dpi 200        # ~5.5 MB PDF, lighter sharing
    python3 scripts/build_noesis_pdf.py --export-cover /tmp/NOESIS_cover.png

Requires: pandoc, weasyprint (pip-installable, Pango system libs), Pillow.
"""
from __future__ import annotations

import html as html_mod
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "NOESIS_FOUNDATION.md"
OUT = ROOT / "docs" / "NOESIS_FOUNDATION.pdf"
BUILD = ROOT / "build" / "noesis"
COVER_SRC = ROOT / "rhan_core" / "OfficialNOESIS.png"

# Artbook-cover geometry: A4 canvas. Default render is 300 dpi (2480x3507 px,
# print quality); --dpi 200 halves the pixel budget (~5.5 MB PDF) for lighter
# sharing. The artwork is a dark-field design (landscape, ~1.2:1) on a
# portrait page; rather than cropping it, the full artwork (aspect preserved,
# never stretched) is centered on the page and the vertical bands are filled
# by extending the artwork's own dark paper texture outward (mirror-tiled
# field pixels, feathered seams) — one continuous sheet.
DPI = 300
COVER_W_PX, COVER_H_PX = 2480, 3507  # recomputed by set_cover_dpi()
COVER_DST = BUILD / f"OfficialNOESIS_artbook_{DPI}dpi.png"
FEATHER_PX = 24  # half-width of the artwork/extension seam blend, in px


def set_cover_dpi(dpi: int) -> None:
    """Recompute the cover canvas size (and its cache file) for a DPI.

    A4 is 210x297 mm; the default 300 dpi yields the 2480x3507 canvas used
    throughout the artbook composite. Lowering the DPI shrinks only the cover
    image (the PDF's body text stays vector) — enough to take the file from
    ~10 MB to ~5.5 MB at 200 dpi.
    """
    global DPI, COVER_W_PX, COVER_H_PX, COVER_DST
    DPI = dpi
    COVER_W_PX = round(2480 * dpi / 300)
    COVER_H_PX = round(3507 * dpi / 300)
    COVER_DST = BUILD / f"OfficialNOESIS_artbook_{DPI}dpi.png"

TITLE = "NOESIS — A Framework for Biologically Inspired Perceptual Intelligence"

CSS = """
/* ---- page furniture ---- */
@page {
  size: A4;
  margin: 20mm 20mm 18mm 20mm;
  @top-left {
    content: "NOESIS — A Framework for Biologically Inspired Perceptual Intelligence";
    font-family: 'DejaVu Serif', serif;
    font-size: 7.5pt; font-style: italic; color: #6b6b6b;
    vertical-align: bottom; margin-bottom: 5mm;
  }
  @bottom-right {
    content: counter(page);
    font-family: 'DejaVu Serif', serif;
    font-size: 9pt; color: #444;
    vertical-align: top; margin-top: 5mm;
  }
}
/* Cover and opening pages carry no furniture. */
@page front {
  @top-left { content: none; }
  @bottom-right { content: none; }
}
/* ---- base typography ---- */
html, body { margin: 0; padding: 0; }
html { font-family: 'DejaVu Serif', serif; font-size: 10.5pt; }
body { color: #141414; line-height: 1.5; }
p { text-align: justify; hyphens: auto; }
h1 {
  font-size: 20pt;
  line-height: 1.25;
  page-break-before: always;
  border-bottom: 0.6pt solid #9a9a9a;
  padding-bottom: 0.2em;
  margin-top: 0.1em;
}
h2 { font-size: 14pt; margin-top: 1.3em; }
h3 { font-size: 11.5pt; margin-top: 1.1em; }
pre {
  font-family: 'DejaVu Sans Mono', monospace;
  font-size: 8pt; line-height: 1.3;
  background: #f5f5f2; border: 0.5pt solid #dcdcd8;
  padding: 0.5em; white-space: pre-wrap; word-wrap: break-word;
}
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }
table { border-collapse: collapse; font-size: 9pt; margin: 0.7em 0; }
th, td { border: 0.5pt solid #8a8a8a; padding: 0.22em 0.45em; text-align: left; }
th { background: #efefec; }
blockquote { font-style: italic; color: #333; margin: 0.7em 1.3em; }
hr { border: none; border-top: 0.6pt solid #bbb; margin: 1.1em 0; }
/* ---- front matter ---- */
/* Page 1: full-bleed book cover. The artwork is composited onto the A4 page
   at build time (see make_artbook_cover) so its extended field fills the
   MediaBox edge-to-edge — no border, no furniture, no text of any kind.
   overflow:hidden clips any sub-pixel rounding. */
@page cover {
  margin: 0;
  @top-left { content: none; }
  @bottom-right { content: none; }
}
.cover {
  page: cover;
  page-break-after: always;
  overflow: hidden;
}
/* pandoc wraps the emblem in a <p>; kill its default margins so the image
   reaches the very top of the page instead of leaving a white band. */
.cover p { margin: 0; padding: 0; }
.cover img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}
/* Page 2: title + subtitle + the Preface, no page furniture. */
.opening {
  page: front;
  text-align: center;
  padding-top: 4em;
  page-break-after: always;
}
.opening h1 {
  page-break-before: auto;
  border: none;
  font-size: 28pt;
  letter-spacing: 0.08em;
  margin: 0 0 0.35em;
}
.opening h2 {
  font-weight: normal;
  font-size: 13.5pt;
  margin: 0 2em 2em;
}
.opening p {
  text-align: center;
  font-size: 12.5pt;
  line-height: 1.85;
  margin: 0.85em 1.4em;
}
/* ---- table of contents ---- */
.toc-page .toc-title { text-align: center; font-size: 18pt; margin: 0.1em 0 1em; }
ul.toc { list-style: none; margin: 0; padding: 0; }
ul.toc li { margin: 0.13em 0; }
ul.toc a { text-decoration: none; color: #161616; }
ul.toc a::after { content: leader(" . ") target-counter(attr(href), page); }
ul.toc li.ch { font-weight: bold; margin-top: 0.75em; }
ul.toc li.sec { padding-left: 1.7em; font-size: 9.5pt; }
"""


def _artwork_scaled_height() -> int:
    """Height the artwork occupies on the cover canvas (aspect preserved).

    Mirrors the scale arithmetic in make_artbook_cover — kept in one place so
    the canvas geometry can never drift between the composer and the verify.
    """
    from PIL import Image
    w, h = Image.open(COVER_SRC).size
    return round(COVER_W_PX / w * h)


def make_artbook_cover() -> pathlib.Path:
    """Composite the full NOESIS artwork onto an A4 canvas, artbook style.

    The artwork keeps its exact aspect ratio, is never cropped and never
    stretched, and is centered on the page. The portrait-page bands above and
    below are filled by extending the artwork's own dark textured background
    outward: the artwork's genuine field pixels are mirror-tiled (a mirror
    boundary is pixel-identical by construction, so no tile seam can show)
    and the two artwork/extension seams are feathered. The result reads as one
    uninterrupted piece of artwork covering the entire sheet.

    Returns the path of the full-bleed asset (cached between builds).
    """
    from PIL import Image, ImageOps

    if not COVER_SRC.exists():
        raise SystemExit(
            f"ERROR: cover image not found: {COVER_SRC} — the manuscript's "
            "cover references rhan_core/OfficialNOESIS.png; add it before building"
        )
    # Cache is invalidated when either the source artwork OR this script's
    # compositing logic changes (mtime), so a regenerated OfficialNOESIS.png
    # or an edited make_artbook_cover() can never ship a stale composite.
    script_mtime = pathlib.Path(__file__).stat().st_mtime
    if (
        COVER_DST.exists()
        and COVER_DST.stat().st_mtime >= COVER_SRC.stat().st_mtime
        and COVER_DST.stat().st_mtime >= script_mtime
    ):
        print(f"  [cover] using cached {COVER_DST.name}")
        return COVER_DST
    BUILD.mkdir(parents=True, exist_ok=True)

    img = Image.open(COVER_SRC).convert("RGB")
    w, h = img.size
    px = img.load()

    # Extent of the artwork (emblem + vines) on the dark field.
    corners = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1]]
    bg = tuple(sum(c[i] for c in corners) // 4 for i in range(3))
    tol = 24
    minx, miny, maxx, maxy = w, h, -1, -1
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            r, g, b = px[x, y]
            if (
                abs(r - bg[0]) > tol
                or abs(g - bg[1]) > tol
                or abs(b - bg[2]) > tol
            ):
                minx, maxx = min(minx, x), max(maxx, x)
                miny, maxy = min(miny, y), max(maxy, y)
    if maxx < 0:
        raise SystemExit("ERROR: could not locate the artwork on its field")

    # Scale the artwork to full page width (exact aspect ratio preserved).
    scale = COVER_W_PX / w
    ah = round(h * scale)
    assert ah == _artwork_scaled_height(), "geometry helper drifted"
    img = img.resize((COVER_W_PX, ah), Image.Resampling.LANCZOS)

    # Pure-field strips (no emblem pixels) to source the texture extension.
    top_field_h = max(8, round(miny * scale))
    bot_field_h = max(8, round((h - 1 - maxy) * scale))
    band_h = (COVER_H_PX - ah) // 2
    if band_h < 1:
        raise SystemExit("ERROR: artwork does not fit on the A4 page")

    def extend(strip, target_h, boundary_at_bottom):
        """Tile a field strip to target_h rows, mirror-flipped so the tile
        adjacent to the artwork matches the artwork's boundary row exactly
        (boundary side is the band's bottom when boundary_at_bottom)."""
        w0, sh = strip.size
        flipped = ImageOps.flip(strip)
        out = Image.new("RGB", (w0, target_h))
        if boundary_at_bottom:
            y, tile = target_h, flipped
            while y > 0:
                take = min(sh, y)
                out.paste(tile.crop((0, sh - take, w0, sh)), (0, y - take))
                y -= take
                tile = ImageOps.flip(tile)
        else:
            y, tile = 0, flipped
            while y < target_h:
                take = min(sh, target_h - y)
                out.paste(tile.crop((0, 0, w0, take)), (0, y))
                y += take
                tile = ImageOps.flip(tile)
        return out

    canvas = Image.new("RGB", (COVER_W_PX, COVER_H_PX))
    top = extend(img.crop((0, 0, COVER_W_PX, top_field_h)), band_h, True)
    bot = extend(img.crop((0, ah - bot_field_h, COVER_W_PX, ah)), band_h, False)
    canvas.paste(top, (0, 0))
    canvas.paste(img, (0, band_h))
    canvas.paste(bot, (0, band_h + ah))

    # Feather each artwork/extension seam so no join is visible.
    for seam in (band_h, band_h + ah):
        for dy in range(1, FEATHER_PX + 1):
            a = canvas.crop((0, seam - dy, COVER_W_PX, seam - dy + 1))
            b = canvas.crop((0, seam + dy, COVER_W_PX, seam + dy + 1))
            t = dy / float(FEATHER_PX)
            canvas.paste(Image.blend(a, b, t), (0, seam - dy))
            canvas.paste(Image.blend(b, a, t), (0, seam + dy))

    canvas.save(COVER_DST, optimize=True)
    print(f"  [cover] {COVER_SRC.name} -> {COVER_DST.name} "
          f"{COVER_W_PX}x{COVER_H_PX}px artbook cover "
          f"({COVER_DST.stat().st_size / 1e6:.1f} MB, artwork "
          f"{COVER_W_PX}x{ah} centered on {band_h}px extended field bands)")
    return COVER_DST


def split_front_matter(text: str) -> tuple[str, str]:
    """Return (cover_page_md, body_md).

    cover_page_md: cover block + page-2 opening. The handwritten TOC and the
    surrounding ``---`` rules are dropped; a real, page-numbered TOC is
    generated later from the pandoc-produced heading ids.
    body_md: everything from '# Chapter 1' onward, untouched.
    """
    try:
        idx_ch1 = text.index("# Chapter 1")
    except ValueError:
        raise SystemExit(
            "ERROR: '# Chapter 1' heading not found in manuscript"
        ) from None
    front = text[:idx_ch1]
    body = text[idx_ch1:]

    m = re.search(r'<div align="center">.*?</div>', front, re.S)
    if not m:
        raise SystemExit("ERROR: cover block not found in manuscript")
    cover_block = m.group(0).replace(
        '<div align="center">', '<div class="cover" align="center">'
    )
    # The cover is an image plate only — strip the title, subtitle and version
    # lines that follow the emblem, so the page carries nothing but the art.
    cover_block = re.sub(
        r"\n# NOESIS\n\n## A Framework for Biologically Inspired "
        r"Perceptual Intelligence\n\n\*\*Version 1\.0\*\*",
        "", cover_block, count=1,
    )
    if "# NOESIS" in cover_block:  # fail loudly rather than print a titled cover
        raise SystemExit(
            "ERROR: could not strip the cover title/subtitle — the manuscript's "
            "cover text changed; update the strip pattern in split_front_matter()"
        )

    p2 = re.search(
        r"---\n\n(We did not begin.*?Perception became the objective\.)\n\n---",
        front, re.S,
    )
    if not p2:
        raise SystemExit("ERROR: page-2 opening block not found in manuscript")
    sentences = "\n\n".join(
        ln.strip() for ln in p2.group(1).splitlines() if ln.strip()
    )
    # Page 2 opens with the title and subtitle, then the Preface.
    title_block = (
        "# NOESIS\n\n"
        "## A Framework for Biologically Inspired Perceptual Intelligence\n\n"
    )
    # No explicit page-break div between cover and opening: each carries
    # page-break-after in the CSS (an extra div would force a blank page).
    opening = f'<div class="opening">\n\n{title_block}{sentences}\n\n</div>'
    cover_page = cover_block + "\n\n" + opening
    return cover_page, body


def pandoc_fragment(md_text: str, out_html: pathlib.Path) -> None:
    md_path = BUILD / "part.md"
    md_path.write_text(md_text, encoding="utf-8")
    # implicit_figures would wrap the cover emblem in <figure><figcaption>
    # (duplicating the alt text on the page) — disable it.
    subprocess.run(
        ["pandoc", str(md_path), "-f", "markdown-implicit_figures",
         "-t", "html5", "-o", str(out_html)],
        cwd=ROOT, check=True,
    )


def build_toc(body_html: pathlib.Path) -> str:
    """Generate the Table of Contents from pandoc's heading ids."""
    frag = body_html.read_text(encoding="utf-8")
    entries = re.findall(
        r"<h([12]) id=\"([^\"]+)\">(.*?)</h\1>", frag, re.S
    )
    if not entries:
        raise SystemExit("ERROR: no h1/h2 headings found in body HTML")
    rows = []
    for level, ident, raw_text in entries:
        text = re.sub(r"<[^>]+>", "", raw_text)
        text = html_mod.unescape(text).strip()
        cls = "ch" if level == "1" else "sec"
        rows.append(
            f'<li class="{cls}"><a href="#{ident}">{html_mod.escape(text)}</a></li>'
        )
    return (
        '<div class="toc-page">\n'
        '<h2 class="toc-title">Table of Contents</h2>\n'
        '<ul class="toc">\n' + "\n".join(rows) + "\n</ul>\n</div>"
    )


def verify_full_bleed() -> None:
    """Verify page 1 is a full-bleed image plate (warning only, never fatal).

    Rasterizes page 1 and checks that the artwork reaches all four page edges
    and that no corner is white. Skipped silently if pdftoppm is unavailable.
    """
    if shutil.which("pdftoppm") is None:
        return
    from PIL import Image

    subprocess.run(
        ["pdftoppm", "-f", "1", "-l", "1", "-r", "40", "-png",
         str(OUT), "/tmp/noesis_cover"],
        check=True, capture_output=True,
    )
    png = sorted(pathlib.Path("/tmp").glob("noesis_cover-*.png"))[0]
    img = Image.open(png).convert("L")
    w, h = img.size
    px = img.load()
    minx, miny, maxx, maxy = w, h, -1, -1
    for y in range(h):
        for x in range(w):
            if px[x, y] < 235:
                minx, maxx = min(minx, x), max(maxx, x)
                miny, maxy = min(miny, y), max(maxy, y)
    if maxx < 0:
        print("  [verify] WARNING: cover page appears blank")
        return
    tol = 3
    edges = (
        minx <= tol and maxx >= w - 1 - tol
        and miny <= tol and maxy >= h - 1 - tol
    )
    corners = [px[2, 2], px[w - 3, 2], px[2, h - 3], px[w - 3, h - 3]]
    full_bleed = edges and all(c < 150 for c in corners)
    # The emblem must be fully visible: its bright core inset from every page
    # edge (never cropped) and spanning most of the page width.
    bminx, bminy, bmaxx, bmaxy = w, h, -1, -1
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            if px[x, y] > 150:
                bminx, bmaxx = min(bminx, x), max(bmaxx, x)
                bminy, bmaxy = min(bminy, y), max(bmaxy, y)
    emblem_ok = (
        bmaxx >= 0
        and bminx >= 0.05 * w and w - 1 - bmaxx >= 0.05 * w
        and bminy >= 0.05 * h and h - 1 - bmaxy >= 0.05 * h
        and (bmaxx - bminx) >= 0.5 * w
    )
    if full_bleed and emblem_ok:
        print("  [verify] cover OK: full-bleed, emblem fully visible & centered")
    else:
        if not full_bleed:
            print("  [verify] WARNING: cover is NOT full-bleed — inspect layout")
        if not emblem_ok:
            print("  [verify] WARNING: emblem missing, cropped, or off-center")

    # Artbook seam continuity: the artwork's field is extended above and below
    # by mirror-tiling (see make_artbook_cover). Find where the artwork region
    # ends on the page and measure the worst luminance step across the two
    # seams — a visible join (> ~6) means the extension logic regressed.
    if not COVER_SRC.exists():  # verification is never fatal
        return
    _art_h = _artwork_scaled_height()
    art_top = round((COVER_H_PX - _art_h) / 2 / COVER_H_PX * h)
    art_bot = art_top + round(_art_h / COVER_H_PX * h)
    worst = 0.0
    for seam in (art_top, art_bot):
        if 0 < seam < h:
            for x in range(0, w, 2):
                worst = max(worst, abs(px[x, seam - 1] - px[x, seam]))
    if worst > 6:
        print(
            f"  [verify] WARNING: cover field seam visible "
            f"(worst |step|={worst:.1f} luminance at rows {art_top}/{art_bot})"
        )
    elif worst > 0:
        print(
            f"  [verify] cover field seams invisible "
            f"(worst |step|={worst:.1f} luminance)"
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Build docs/NOESIS_FOUNDATION.pdf from the NOESIS "
        "founding manuscript (pandoc + WeasyPrint).",
    )
    ap.add_argument(
        "--dpi", type=int, default=300, choices=(200, 300),
        help="cover render resolution (default 300; 200 shrinks the PDF to "
        "~5.5 MB for lighter sharing)",
    )
    ap.add_argument(
        "--export-cover", metavar="PATH", default=None,
        help="copy the standalone artbook cover composite PNG to PATH "
        "(for previewing or sharing without opening the PDF)",
    )
    args = ap.parse_args(argv)
    set_cover_dpi(args.dpi)

    if not SRC.exists():
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        return 1
    if shutil.which("pandoc") is None:
        print("ERROR: required tool 'pandoc' not on PATH", file=sys.stderr)
        return 1
    try:
        import weasyprint  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: weasyprint not installed — run: "
            "python3 -m pip install --user weasyprint",
            file=sys.stderr,
        )
        return 1

    BUILD.mkdir(parents=True, exist_ok=True)
    cover_path = make_artbook_cover()
    if args.export_cover:
        dest = pathlib.Path(args.export_cover)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cover_path, dest)
        print(f"  [cover] exported standalone cover PNG -> {dest}")

    cover_md, body_md = split_front_matter(SRC.read_text(encoding="utf-8"))
    cover_html = BUILD / "cover_frag.html"
    body_html = BUILD / "body_frag.html"
    pandoc_fragment(cover_md, cover_html)
    pandoc_fragment(body_md, body_html)
    print("  [pandoc] markdown -> HTML fragments (cover + body)")

    cover_frag = cover_html.read_text(encoding="utf-8")
    body_frag = body_html.read_text(encoding="utf-8")
    # Absolute path for the cover emblem (WeasyPrint base_url is the repo root).
    cover_frag = cover_frag.replace("../rhan_core/OfficialNOESIS.png", str(COVER_DST))
    toc_html = build_toc(body_html)

    full = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        f"<title>{TITLE}</title>\n"
        f"<style>\n{CSS}\n</style>\n</head>\n<body>\n"
        f"{cover_frag}\n"
        f"{toc_html}\n"
        f"{body_frag}\n"
        "</body>\n</html>\n"
    )
    html_path = BUILD / "noesis.html"
    html_path.write_text(full, encoding="utf-8")
    print(f"  [assemble] cover + TOC ({toc_html.count('<li ')}) + body")

    print("  [weasyprint] rendering PDF (paged media)...")
    weasyprint.HTML(string=full, base_url=str(ROOT)).write_pdf(str(OUT))
    print(f"  [weasyprint] OK — {OUT.stat().st_size / 1e6:.1f} MB")

    if shutil.which("pdfinfo") is not None:  # verification-only; never fatal
        info = subprocess.run(
            ["pdfinfo", str(OUT)], capture_output=True, text=True, check=True
        )
        for line in info.stdout.splitlines():
            if line.startswith(("Pages", "Page size", "File size")):
                print(f"  [pdf]       {line}")
    verify_full_bleed()
    print(f"\nDone: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
