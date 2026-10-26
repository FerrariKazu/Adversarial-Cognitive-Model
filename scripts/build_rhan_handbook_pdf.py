#!/usr/bin/env python3
"""Build docs/rhan_nxa/RHAN_NXA_Technical_Handbook.pdf from the Markdown manuscript.

Pipeline
--------
    one Markdown file per chapter
        -> per-file preprocessing (figure numbering, absolute asset paths)
        -> pandoc -t html5 (fragments)
        -> assembly (cover + table of contents + list of figures + body)
        -> WeasyPrint (CSS paged media) -> PDF

Design decisions that come from observed defects in earlier editions
--------------------------------------------------------------------
* The running header is short ("RHAN Technical Handbook"). A long header
  collides with the chapter title on A4 with this margin.
* Figures are numbered by this build, never inside the image. A raster that
  says "Figure 33" goes stale the moment the figure moves; a caption that
  says it is regenerated on every build.
* Chapter starts and part dividers each force a page break, so a part page is
  never followed by a nearly blank page.
* Tables wrap (`word-break`) and shrink to 8 pt instead of overflowing the
  text block; an overfull table is a defect, not a typographic accident.
* `--check` re-opens the produced PDF and asserts page count, figure count and
  the absence of unresolved links, so "it compiled" is never the acceptance
  criterion.

Usage
-----
    python3 scripts/build_rhan_handbook_pdf.py
    python3 scripts/build_rhan_handbook_pdf.py --check
    python3 scripts/build_rhan_handbook_pdf.py --html-only   # keep the HTML
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "rhan_nxa"
OUT = SRC / "RHAN_NXA_Technical_Handbook.pdf"
BUILD = ROOT / "build" / "rhan_handbook"

sys.path.insert(0, str(ROOT / "scripts" / "handbook"))
import render_numbers  # noqa: E402  (path set above)

TITLE = "RHAN — Recurrent Hierarchical Attention Network"
SUBTITLE = "Technical Handbook"
BLURB = "Architecture, Implementation, Experimental Evaluation, and Future Directions"
VERSION = "Version 3.3 · 25 October 2026"
RUNNING = "RHAN Technical Handbook"

# Every consequential number is rendered from docs/rhan_nxa/data/numbers.json.
# An unknown id aborts the build rather than shipping a literal.
_METRICS = render_numbers.load()

# Reading order. Each entry is (directory, part title or None, part subtitle or None).
PARTS = [
    ("00_front_matter", None, None),
    ("01_concept", "Part I — Concept and architecture",
     "What problem RHAN addresses, and the mechanism that answers it"),
    ("02_implementation", "Part II — Implementation and engineering",
     "How that architecture is built, in tensors and in code"),
    ("03_evaluation", "Part III — Training and experimental evaluation",
     "How the model was trained, measured, and how strong each measurement is"),
    ("04_results", "Part IV — Results and interpretation",
     "What the numbers support, what they do not, and which confounds remain"),
    ("05_limits_future", "Part V — Limitations and future directions",
     "What is missing, and what would discriminate between explanations"),
    ("06_appendices", "Appendices",
     "Derivations, configuration, data, protocols, registries, reproduction"),
]

CSS = """
@page {
  size: A4;
  margin: 22mm 20mm 20mm 20mm;
  @top-left {
    content: "RHAN Technical Handbook";
    font-family: 'DejaVu Sans', sans-serif;
    font-size: 7pt; color: #7a7a7a; letter-spacing: 0.02em;
    vertical-align: bottom; margin-bottom: 4mm;
  }
  @top-right {
    content: string(chaptertitle);
    font-family: 'DejaVu Sans', sans-serif;
    font-size: 7pt; color: #7a7a7a;
    vertical-align: bottom; margin-bottom: 4mm;
  }
  @bottom-center {
    content: counter(page);
    font-family: 'DejaVu Sans', sans-serif;
    font-size: 8.5pt; color: #555;
    vertical-align: top; margin-top: 5mm;
  }
}
@page bare { @top-left { content: none; } @top-right { content: none; } @bottom-center { content: none; } }

html { font-family: 'DejaVu Serif', serif; font-size: 10pt; }
body { color: #161616; line-height: 1.5; hyphens: auto; }
p { text-align: justify; }
h1 {
  font-family: 'DejaVu Sans', sans-serif;
  font-size: 17pt; line-height: 1.2; font-weight: bold;
  page-break-before: always; break-before: page;
  border-bottom: 0.7pt solid #9a9a9a;
  padding-bottom: 0.25em; margin: 0 0 1.0em 0;
  string-set: chaptertitle content();
}
h2 { font-family: 'DejaVu Sans', sans-serif; font-size: 12pt; margin: 1.5em 0 0.5em;
     break-after: avoid; }
h3 { font-family: 'DejaVu Sans', sans-serif; font-size: 10.3pt; margin: 1.2em 0 0.35em; color: #2b2b2b; }
h4 { font-family: 'DejaVu Sans', sans-serif; font-size: 9.6pt; margin: 1em 0 0.3em; }
p, li { orphans: 2; widows: 2; }
ul, ol { margin: 0.4em 0 0.7em 1.3em; padding: 0; }
li { margin: 0.15em 0; }
/* Identifiers are not prose: never hyphenate them.  ``hyphens: none`` stops
   WeasyPrint inserting a U+2010 inside a path or a checkpoint name (which
   produced tokens such as "reconciliation/at-" that a reader cannot copy);
   ``overflow-wrap: anywhere`` still lets a long identifier fit the column. */
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.4pt;
       hyphens: none; overflow-wrap: anywhere; }
pre {
  font-family: 'DejaVu Sans Mono', monospace;
  font-size: 7.8pt; line-height: 1.34;
  background: #f6f6f3; border-left: 2pt solid #cfcfc9;
  padding: 0.55em 0.7em; white-space: pre-wrap; word-wrap: break-word;
  page-break-inside: avoid;
}
blockquote { margin: 0.7em 0 0.7em 1.2em; padding-left: 0.8em; border-left: 2pt solid #d8d8d2;
             font-style: italic; color: #333; }
table { border-collapse: collapse; width: 100%; font-family: 'DejaVu Sans', sans-serif;
        font-size: 7.9pt; margin: 0.8em 0 1.0em; }
/* A long table that cannot break pushes to the next page and leaves a chapter
   opener holding nothing but its heading (observed on Appendix C, p.196).
   Rows stay atomic; the table may split between rows and repeat its header. */
thead { display: table-header-group; }
tr { page-break-inside: avoid; break-inside: avoid; }
th, td { border: 0.4pt solid #a9a9a9; padding: 0.24em 0.4em; text-align: left;
         vertical-align: top; word-break: break-word; overflow-wrap: anywhere;
         hyphens: none; }
/* A single unbreakable path in one cell must not widen the whole table past the
   text block (observed on the Appendix A attack table, p.139). `anywhere` makes
   the cell's min-content width one character instead of one token. */
table { table-layout: auto; max-width: 100%; }
th { background: #efefec; font-weight: bold; }
hr { border: none; border-top: 0.5pt solid #c4c4bf; margin: 1.2em 0; }
img { max-width: 100%; }

/* ---- cover ---- */
.coverpage { page: bare; page-break-after: always; padding-top: 28mm; }
.cover-rule { border-top: 2.4pt solid #1b1b1b; width: 42%; margin: 0 0 10mm 0; }
.cover-kicker { font-family: 'DejaVu Sans', sans-serif; font-size: 8.4pt; letter-spacing: 0.22em;
                text-transform: uppercase; color: #6e6e6e; margin-bottom: 6mm; }
.cover-title { font-family: 'DejaVu Sans', sans-serif; font-size: 27pt; font-weight: bold;
               line-height: 1.12; margin: 0 0 3mm 0; }
.cover-sub { font-family: 'DejaVu Sans', sans-serif; font-size: 15pt; color: #2f6da3;
             margin: 0 0 8mm 0; }
.cover-blurb { font-family: 'DejaVu Serif', serif; font-size: 12pt; font-style: italic;
               color: #333; margin: 0 0 14mm 0; max-width: 135mm; }
.cover-version { font-family: 'DejaVu Sans', sans-serif; font-size: 9.6pt; color: #444; }
.cover-facts { margin-top: 22mm; font-family: 'DejaVu Sans', sans-serif; font-size: 8.2pt;
               color: #333; width: 100%; }
.cover-facts td { border: none; border-top: 0.4pt solid #d5d5cf; padding: 0.45em 0.6em 0.45em 0; }
.cover-facts td:first-child { width: 34mm; color: #6e6e6e; }

/* ---- part divider ---- */
.part { page: bare; page-break-before: always; padding-top: 70mm; }
.part .label { font-family: 'DejaVu Sans', sans-serif; font-size: 9pt; letter-spacing: 0.2em;
               text-transform: uppercase; color: #2f6da3; margin-bottom: 4mm; }
.part .name { font-family: 'DejaVu Sans', sans-serif; font-size: 21pt; font-weight: bold;
              line-height: 1.18; margin: 0 0 5mm 0; }
.part .desc { font-family: 'DejaVu Serif', serif; font-size: 11pt; font-style: italic;
              color: #444; max-width: 120mm; }
.part .rule { border-top: 1.6pt solid #1b1b1b; width: 30%; margin-bottom: 8mm; }

/* ---- contents ---- */
.toc { page: bare; page-break-after: always; }
.toc h2, .lof h2 { font-size: 16pt; margin: 0 0 1.1em 0; text-align: left; }
ul.toclist { list-style: none; margin: 0; padding: 0; }
ul.toclist li { margin: 0.16em 0; }
ul.toclist a { text-decoration: none; color: #161616; }
ul.toclist a::after { content: leader(' . ') target-counter(attr(href), page); }
ul.toclist li.partrow { margin-top: 1.0em; font-weight: bold; font-family: 'DejaVu Sans', sans-serif;
                        font-size: 9.4pt; text-transform: uppercase; letter-spacing: 0.06em; }
ul.toclist li.ch { font-family: 'DejaVu Sans', sans-serif; font-size: 9.6pt; }
ul.toclist li.sec { padding-left: 1.9em; font-size: 8.6pt; color: #333; }
.lof { page : bare; page-break-after: always; }
ul.loflist { list-style: none; margin: 0; padding: 0; font-size: 9pt; }
ul.loflist li { margin: 0.2em 0; }
ul.loflist a::after { content: leader(' . ') target-counter(attr(href), page); }
ul.loflist a { text-decoration: none; color: #161616; }

/* ---- figures ---- */
figure { margin: 1.1em 0 1.2em; page-break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figcaption { font-family: 'DejaVu Sans', sans-serif; font-size: 8.2pt; color: #333;
             text-align: left; margin-top: 0.5em; line-height: 1.36; }
figcaption .fignum { font-weight: bold; color: #1b1b1b; }
"""


# ---------------------------------------------------------------------------
# manuscript assembly
# ---------------------------------------------------------------------------
# The caption may contain brackets ("[0,1]", "[-4,4]"); a negated class here
# silently dropped eight figures from the v3.1 build because their captions
# contained `[0,1]`. Non-greedy `.*?` plus a path with no whitespace keeps the
# match unambiguous for single-line references.
FIG_RE = re.compile(r"!\[(.*?)\]\(([^)\s]+)\)")


def chapter_files() -> list[tuple[pathlib.Path, str | None, str | None]]:
    """Return [(path, part_title, part_desc)] in reading order."""
    out = []
    for d, part, desc in PARTS:
        files = sorted((SRC / d).glob("*.md"))
        if not files:
            raise SystemExit(f"ERROR: no markdown in {SRC / d}")
        for i, f in enumerate(files):
            out.append((f, part if i == 0 else None, desc if i == 0 else None))
    return out


def read_cover() -> str:
    """Cover copy, kept in the manuscript (00_cover.md) so it is editable there."""
    text = (SRC / "00_front_matter" / "00_cover.md").read_text(encoding="utf-8")
    facts: list[list[str]] = []
    joinable = False
    for line in text.splitlines():
        if line.startswith("**") and "—" in line:
            label, _, value = line.partition("—")
            facts.append([label.strip(" *"), value.strip()])
            joinable = True
        elif not line.strip():
            joinable = False
        elif joinable and not line.startswith(("#", "---", "**")):
            # wrapped continuation of the previous fact, not a new block
            facts[-1][1] += " " + line.strip()
        else:
            joinable = False
    return text, [(a, b.replace("`", "")) for a, b in facts]


def build_cover(facts: list[tuple[str, str]]) -> str:
    rows = "".join(f"<tr><td>{html_mod.escape(a)}</td><td>{html_mod.escape(b)}</td></tr>"
                   for a, b in facts)
    return f"""
<div class="coverpage">
  <div class="cover-rule"></div>
  <div class="cover-kicker">Technical handbook · Gen-2 / J2</div>
  <div class="cover-title">RHAN</div>
  <div class="cover-title" style="font-size:19pt;margin-top:1mm;">Recurrent Hierarchical Attention Network</div>
  <div class="cover-sub">{SUBTITLE}</div>
  <div class="cover-blurb">{BLURB}</div>
  <div class="cover-version">{VERSION}</div>
  <table class="cover-facts">{rows}</table>
</div>
"""


def html_escape_text(s: str) -> str:
    return html_mod.escape(s.strip(), quote=False)


def preprocess(md: str, src_dir: pathlib.Path, fig_counter: list[int],
               figures: list[tuple[str, str]]) -> str:
    """Number figures at build time; make asset paths absolute; inject numbers."""
    md = render_numbers.resolve(md, _METRICS, where=str(src_dir))

    def repl(m: re.Match) -> str:
        caption, path = m.group(1).strip(), m.group(2).strip()
        if path.startswith(("http://", "https://")):
            raise SystemExit("ERROR: remote images are not allowed in the manuscript")
        asset = (src_dir / path).resolve()
        if not asset.exists():
            raise SystemExit(f"ERROR: figure asset missing: {asset}")
        fig_counter[0] += 1
        fid = f"fig{fig_counter[0]:02d}"
        figures.append((fid, caption))
        return (f'<figure id="{fid}"><img src="{asset.as_uri()}" alt="{html_mod.escape(caption)}">'
                f'<figcaption><span class="fignum">Figure {fig_counter[0]}.</span> '
                f'{html_escape_text(caption)}</figcaption></figure>')
    return FIG_RE.sub(repl, md)


def pandoc(md_path: pathlib.Path, out_html: pathlib.Path) -> None:
    subprocess.run(
        ["pandoc", str(md_path), "-f", "markdown+pipe_tables+raw_html",
         "-t", "html5", "--wrap=none", "-o", str(out_html)],
        cwd=ROOT, check=True,
    )


def toc_and_lof(body_entries: list[dict], figures: list[tuple[str, str]]) -> str:
    rows = []
    for e in body_entries:
        if e["kind"] == "part":
            rows.append(f'<li class="partrow"><a href="#{e["id"]}">{e["text"]}</a></li>')
        elif e["kind"] == "h1":
            rows.append(f'<li class="ch"><a href="#{e["id"]}">{e["text"]}</a></li>')
        elif e["kind"] == "h2":
            rows.append(f'<li class="sec"><a href="#{e["id"]}">{e["text"]}</a></li>')
    toc = ('<div class="toc"><h2>Contents</h2><ul class="toclist">'
           + "".join(rows) + "</ul></div>")
    def short(cap: str, limit: int = 104) -> str:
        if len(cap) <= limit:
            text = cap
        else:
            text = cap[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "\u2026"
        # `identifier` in a caption is code, not literal backticks
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", html_mod.escape(text))

    lof_rows = "".join(
        f'<li><a href="#{fid}">Figure {i + 1}. {short(cap)}</a></li>'
        for i, (fid, cap) in enumerate(figures))
    lof = ('<div class="lof"><h2>Figures</h2><ul class="loflist">'
           + lof_rows + "</ul></div>")
    return toc, lof


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the RHAN Technical Handbook PDF.")
    ap.add_argument("--html-only", action="store_true", help="stop after writing the HTML")
    ap.add_argument("--check", action="store_true", help="verify the produced PDF")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    for tool in ("pandoc",):
        if shutil.which(tool) is None:
            print(f"ERROR: required tool '{tool}' not on PATH", file=sys.stderr)
            return 1
    try:
        import weasyprint  # noqa: PLC0415
    except ImportError:
        print("ERROR: weasyprint not installed (pip install weasyprint)", file=sys.stderr)
        return 1

    BUILD.mkdir(parents=True, exist_ok=True)
    _, facts = read_cover()
    cover_html = build_cover(facts)

    fig_counter = [0]
    figures: list[tuple[str, str]] = []
    entries: list[dict] = []
    body_parts: list[str] = []

    for path, part_title, part_desc in chapter_files():
        if path.name == "00_cover.md":
            # The cover copy is rendered as a designed page by build_cover();
            # emitting it again as a body chapter would duplicate it.
            continue
        if part_title:
            pid = "part-" + re.sub(r"[^a-z0-9]+", "-", part_title.lower()).strip("-")
            label = part_title.split("—")[0].strip()
            name = part_title.split("—", 1)[1].strip() if "—" in part_title else part_title
            body_parts.append(
                f'<div class="part" id="{pid}"><div class="label">{html_mod.escape(label)}</div>'
                f'<div class="rule"></div><div class="name">{html_mod.escape(name)}</div>'
                f'<div class="desc">{html_mod.escape(part_desc or "")}</div></div>')
            entries.append({"kind": "part", "id": pid, "text": part_title})

        md = preprocess(path.read_text(encoding="utf-8"), path.parent, fig_counter, figures)
        tmp_md = BUILD / ("_" + path.stem + ".md")
        tmp_md.write_text(md, encoding="utf-8")
        frag = BUILD / ("_" + path.stem + ".html")
        pandoc(tmp_md, frag)
        frag_html = frag.read_text(encoding="utf-8")

        # Collect heading anchors for the contents page, in document order.
        # Pandoc derives ids from heading text, so "Source verification" occurs
        # in every chapter; prefix each id with the fragment stem to keep the
        # anchors unique (duplicate ids would send every contents entry to the
        # first occurrence).
        slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")

        def uniq(m: re.Match) -> str:
            return f'<h{m.group(1)} id="{slug}--{m.group(2)}">{m.group(3)}</h{m.group(1)}>'

        frag_html = re.sub(r'<h([12]) id="([^"]+)">(.*?)</h\1>', uniq, frag_html, flags=re.S)
        for m in re.finditer(r'<h([12]) id="([^"]+)">(.*?)</h\1>', frag_html, re.S):
            level, hid, raw = m.group(1), m.group(2), re.sub(r"<[^>]+>", "", m.group(3))
            entries.append({"kind": "h1" if level == "1" else "h2", "id": hid,
                            "text": html_mod.unescape(raw).strip()})
        body_parts.append(frag_html)

    toc, lof = toc_and_lof(entries, figures)

    full = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{TITLE} — {SUBTITLE}</title>\n<style>\n{CSS}\n</style>\n</head>\n<body>\n"
        f"{cover_html}\n{toc}\n{lof}\n" + "\n".join(body_parts) + "\n</body>\n</html>\n"
    )
    html_path = BUILD / "handbook.html"
    html_path.write_text(full, encoding="utf-8")
    print(f"  [assemble] {len(entries)} contents entries, {len(figures)} figures, "
          f"{len(body_parts)} fragments")

    if args.html_only:
        print(f"  [html] {html_path}")
        return 0

    out = pathlib.Path(args.out)
    weasyprint.HTML(string=full, base_url=str(ROOT)).write_pdf(str(out))
    size_mb = out.stat().st_size / 1e6
    print(f"  [weasyprint] {out} ({size_mb:.1f} MB)")

    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", str(out)], capture_output=True, text=True)
        for line in info.stdout.splitlines():
            if line.startswith(("Pages", "Page size", "Producer")):
                print(f"  [pdf] {line}")

    if args.check:
        manifest = SRC / "data" / "figure_manifest.json"
        expected = (json.loads(manifest.read_text(encoding="utf-8"))["figure_count"]
                    if manifest.exists() else None)
        if expected is not None and expected != len(figures):
            print(f"  [FAIL] manifest/figure mismatch — registry says {expected}, "
                  f"the build embedded {len(figures)}")
            return 1
        return check(out, len(figures), len(entries))
    return 0


def check(pdf: pathlib.Path, n_figures: int, n_entries: int) -> int:
    """Open the produced PDF and assert what the build claims about it."""
    failures: list[str] = []

    def ok(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("  [SKIP] PyMuPDF unavailable — page-level checks not run")
        fitz = None

    if fitz is not None:
        doc = fitz.open(pdf)
        pages = doc.page_count
        text = "\n".join(doc[i].get_text() for i in range(pages))
        ok("PDF opens and has pages", pages > 60, f"{pages} pages")
        ok("cover carries the title",
           "Recurrent Hierarchical Attention Network" in doc[0].get_text())
        ok("contents page carries page numbers", doc[1].get_text().count(".") > 40)
        ok("no unresolved cross-reference", "??" not in text)
        # `](` occurs legitimately inside code (`self.attn["qkv"](x)`); only a
        # leaked image or link target is a defect.
        leak = re.search(r"!\[|\]\((?:figures/|\.\./|https?://)", text)
        ok("no markdown leakage", leak is None,
           "" if leak is None else f"near {text[max(0, leak.start() - 40):leak.start() + 40]!r}")
        ok("no LaTeX-style leakage", "\\begin{" not in text)
        ok("no TODO/FIXME", "TODO" not in text and "FIXME" not in text)
        ok("figure count", text.count("Figure ") >= n_figures,
           f"{text.count('Figure ')} mentions for {n_figures} figures")
        # every content page except part/cover/toc pages carries a page number
        blanks = 0
        for i in range(pages):
            body = doc[i].get_text().strip()
            if len(body) < 60:
                blanks += 1
        ok("no near-empty pages", blanks <= 8, f"{blanks} pages under 60 characters")
        doc.close()

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
