#!/usr/bin/env python3
"""
RHAN-NXA Technical Handbook Publication Build Pipeline (Version 2.0)
===================================================================
A publication-grade document engineering pipeline conforming to the RHAN-NXA
Design System (DESIGN_SYSTEM.md) and weasyprint-pdf-skill architectural standards.

Key Features:
1. Native MathML typesetting (via latex2mathml) for vector-quality math rendering.
2. 6 Structural Part Divider pages with dark full-bleed thematic compositions.
3. Editorial Chapter Openings with metadata pills, titles, subtitles, and pull quotes.
4. Stylized semantic callouts (Distinction, Core Idea, Why It Matters, Implementation).
5. Epistemic Decision Status Badges (LOCKED, REQUIRED, UNKNOWN, CANDIDATE, DEFERRED, REJECTED).
6. Automatic Figure Gallery generation and visual contact sheet.
7. Automated visual QA page rasterization via pdftoppm.
"""

import os
import sys
import glob
import re
import subprocess
from pathlib import Path

import markdown
import io
import base64

# Matplotlib for high-quality SVG math rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Repository Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "rhan_nxa"
BOOK_DIR = DOCS_DIR / "book"
APP_DIR = DOCS_DIR / "appendices"
FIG_DIR = DOCS_DIR / "figures"
DESIGN_DIR = DOCS_DIR / "design"
GALLERY_DIR = DOCS_DIR / "figure_gallery"
QA_DIR = DOCS_DIR / "pdf_qa_pages"

UNIFIED_MD_PATH = DOCS_DIR / "RHAN_NXA_Handbook_Unified.md"
PDF_PATH = DOCS_DIR / "RHAN_NXA_Technical_Handbook.pdf"

# Structural Macro-Parts Specification
PARTS_SPEC = {
    "00": {
        "part_num": "PART I",
        "part_title": "Foundations & The Robustness Gap",
        "part_subtitle": "Deconstructing the Fragility of Passive Feedforward Classification",
        "synopsis": (
            "Empirical evidence from Generation-0 established that naive module accretion "
            "fails to produce true adversarial robustness. Part I introduces the core thesis of "
            "RHAN-NXA: visual perception must be organized as an active, recurrent hypothesis-testing "
            "loop that explicitly decouples parametric memory, working activations, and belief states."
        )
    },
    "04": {
        "part_num": "PART II",
        "part_title": "The Perceptual Belief State",
        "part_subtitle": "Formalization of the Canonical 5-Tuple State Space",
        "synopsis": (
            "The heart of RHAN-NXA is the explicit belief container B_t = (z_t, S_t, U_t, E_t, A_t). "
            "Part II details the holistic continuous manifold z_t, the None-propagating structural "
            "slot contract S_t, closed-form Dirichlet evidential uncertainty U_t, latent prediction "
            "error E_t, and precision-weighted UpdateNet dynamics."
        )
    },
    "09": {
        "part_num": "PART III",
        "part_title": "Active Perception & Foveation",
        "part_subtitle": "Investigation Dynamics, Coordinate Saccades, and Temporal Loops",
        "synopsis": (
            "Perception becomes active investigation when observation is selective. Part III "
            "analyzes the orthogonal recurrence axes of Option C, differentiable spatial "
            "foveation grids, active information sampling (AIS-v2) for maximal uncertainty "
            "reduction, and the complete recurrent perceptual loop."
        )
    },
    "13": {
        "part_num": "PART IV",
        "part_title": "Architectural Substrate & Gradient Flow",
        "part_subtitle": "CompactViT Transformer Engine and Optimization Boundaries",
        "synopsis": (
            "Rigorous parameter accounting and optimization boundaries. Part IV details the "
            "23.3M parameter visual transformer substrate with tied refinement, the 8-group "
            "optimizer architecture, pre-flight |dW| assertion protocols, and tensor dimensional "
            "invariants."
        )
    },
    "16": {
        "part_num": "PART V",
        "part_title": "The Training System & Experimental Record",
        "part_subtitle": "Durability Lifecycle, Confound Forensics, and Empirical Verification",
        "synopsis": (
            "Science advances through reproducible falsification. Part V presents two-phase "
            "atomic serialization with key parity verification, the staged experimental DAG, "
            "the forensic audit unmasking the Generation-0 SBR confound, robust accuracy records "
            "across 16 seeds, and the L_stab belief trajectory stability objective."
        )
    },
    "21": {
        "part_num": "PART VI",
        "part_title": "Synthesis, Decisions & Future Frontiers",
        "part_subtitle": "Epistemic Commitments, Architectural Anti-Patterns, and RHAN-LLM",
        "synopsis": (
            "Formalization of architectural decisions DR-001 through DR-010; deconstruction of "
            "common misunderstandings and anti-patterns; authoritative technical glossary; "
            "scope boundaries and outright rejections; and the evolutionary roadmap toward "
            "stateful multi-modal reasoning models."
        )
    },
    "A_": {
        "part_num": "APPENDICES",
        "part_title": "Reference Data & Specifications",
        "part_subtitle": "Interface ABCs, Tensor Specifications, and Port Protocols",
        "synopsis": (
            "Complete reference tables: authoritative tensor shape catalogs, formal abstract "
            "base class interface definitions, and the legacy codebase porting matrix."
        )
    }
}

# Publication Design System Print CSS
BOOK_CSS = """
@font-face {
    font-family: "DejaVu Math TeX Gyre";
    src: url("/usr/share/fonts/truetype/dejavu/DejaVuMathTeXGyre.ttf") format("truetype");
}
@font-face {
    font-family: "Latin Modern Math";
    src: url("/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf") format("opentype");
}
@font-face {
    font-family: "TeX Gyre Termes Math";
    src: url("/usr/share/texmf/fonts/opentype/public/tex-gyre-math/texgyretermes-math.otf") format("opentype");
}
@page {
    size: a4;
    margin: 22mm 18mm 22mm 18mm;
    @top-left {
        content: "RHAN-NXA TECHNICAL HANDBOOK";
        font-family: "Liberation Sans", sans-serif;
        font-size: 7.5pt;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #94A3B8;
        border-bottom: 0.5pt solid #E2E8F0;
        padding-bottom: 4px;
    }
    @top-right {
        content: "GENERATION-1 COGNITIVE CORE";
        font-family: "Liberation Sans", sans-serif;
        font-size: 7.5pt;
        font-weight: 600;
        letter-spacing: 1px;
        color: #94A3B8;
        border-bottom: 0.5pt solid #E2E8F0;
        padding-bottom: 4px;
    }
    @bottom-left {
        content: "PRE-REGISTRATION SPECIFICATION";
        font-family: "Liberation Sans", sans-serif;
        font-size: 7.5pt;
        color: #94A3B8;
        border-top: 0.5pt solid #E2E8F0;
        padding-top: 4px;
    }
    @bottom-right {
        content: counter(page);
        font-family: "Liberation Sans", sans-serif;
        font-size: 9pt;
        font-weight: 700;
        color: #0F172A;
        border-top: 0.5pt solid #E2E8F0;
        padding-top: 4px;
    }
}

@page :first {
    margin: 0;
    @top-left { content: none; }
    @top-right { content: none; }
    @bottom-left { content: none; }
    @bottom-right { content: none; }
}

@page part-page {
    background-color: #0F172A;
    color: #FAF9F5;
    margin: 0;
    @top-left { content: none; }
    @top-right { content: none; }
    @bottom-left { content: none; }
    @bottom-right { content: none; }
}

@page toc-page {
    margin: 22mm 18mm 22mm 18mm;
    @top-left { content: "TABLE OF CONTENTS"; }
    @top-right { content: "RHAN-NXA SPECIFICATION"; }
}

body {
    font-family: "Bitstream Charter", "Liberation Serif", Georgia, serif;
    font-size: 9.8pt;
    line-height: 1.54;
    color: #0F172A;
    background-color: #FFFFFF;
    margin: 0;
}

/* Cover Page */
.cover-page {
    page-break-after: always;
    height: 100vh;
    padding: 70px 50px;
    background: linear-gradient(145deg, #090D16 0%, #0F172A 55%, #1E293B 100%);
    color: #F8FAFC;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.cover-badge {
    font-family: "Liberation Sans", sans-serif;
    font-size: 9pt;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: #38BDF8;
    text-transform: uppercase;
    margin-bottom: 24px;
}

.cover-title {
    font-family: "Liberation Sans", sans-serif;
    font-size: 36pt;
    font-weight: 800;
    line-height: 1.10;
    margin: 0 0 16px 0;
    color: #FFFFFF;
    letter-spacing: -0.5px;
}

.cover-subtitle {
    font-family: "Bitstream Charter", "Liberation Serif", serif;
    font-size: 13.5pt;
    font-style: italic;
    color: #94A3B8;
    line-height: 1.45;
    max-width: 600px;
    margin: 0 0 35px 0;
}

.cover-meta {
    border-top: 1px solid rgba(255, 255, 255, 0.15);
    padding-top: 25px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    font-family: "Liberation Sans", sans-serif;
    font-size: 9pt;
    color: #94A3B8;
    line-height: 1.7;
}

.cover-meta strong {
    color: #F8FAFC;
}

/* Part Breaks */
.part-break {
    page: part-page;
    page-break-before: always;
    page-break-after: always;
    height: 100vh;
    padding: 120px 65px;
    box-sizing: border-box;
    background: #0F172A;
    color: #F8FAFC;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.part-number {
    font-family: "Liberation Sans", sans-serif;
    font-size: 14pt;
    font-weight: 700;
    letter-spacing: 3px;
    color: #38BDF8;
    margin-bottom: 14px;
}

.part-title {
    font-family: "Liberation Sans", sans-serif;
    font-size: 32pt;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 14px 0;
    color: #FFFFFF;
}

.part-subtitle {
    font-family: "Liberation Sans", sans-serif;
    font-size: 13pt;
    font-weight: 500;
    color: #94A3B8;
    margin: 0 0 28px 0;
}

.part-divider-line {
    width: 80px;
    height: 3px;
    background: #38BDF8;
    margin-bottom: 28px;
}

.part-synopsis {
    font-family: "Bitstream Charter", "Liberation Serif", serif;
    font-size: 11.5pt;
    font-style: italic;
    color: #CBD5E1;
    max-width: 520px;
    line-height: 1.65;
}

/* Chapter Openings */
.chapter-opening {
    page-break-before: always;
    margin-top: 10px;
    margin-bottom: 24px;
    border-bottom: 2px solid #0F172A;
    padding-bottom: 16px;
}

.chapter-pill {
    display: inline-block;
    background: #0F172A;
    color: #FFFFFF;
    font-family: "Liberation Sans", sans-serif;
    font-size: 9.5pt;
    font-weight: 800;
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 10px;
    letter-spacing: 0.5px;
}

.chapter-title {
    font-family: "Liberation Sans", sans-serif;
    font-size: 21pt;
    font-weight: 800;
    color: #0F172A;
    margin: 0 0 8px 0;
    letter-spacing: -0.3px;
    line-height: 1.2;
}

.chapter-subtitle {
    font-family: "Bitstream Charter", "Liberation Serif", serif;
    font-size: 11pt;
    font-style: italic;
    color: #475569;
    margin: 0 0 10px 0;
    line-height: 1.4;
}

.chapter-metadata-banner {
    background: #FAF9F5;
    border-left: 3px solid #0284C7;
    padding: 6px 12px;
    font-family: "Liberation Sans", sans-serif;
    font-size: 8.5pt;
    color: #475569;
    margin-top: 10px;
    border-radius: 0 4px 4px 0;
}

/* Headings */
h1 {
    font-family: "Liberation Sans", sans-serif;
    font-size: 18pt;
    font-weight: 800;
    color: #0F172A;
    margin-top: 28pt;
    margin-bottom: 12pt;
}

h2 {
    font-family: "Liberation Sans", sans-serif;
    font-size: 13pt;
    font-weight: 700;
    color: #0F172A;
    margin-top: 20pt;
    margin-bottom: 8pt;
    border-bottom: 0.75pt solid #E2E8F0;
    padding-bottom: 3px;
}

h3 {
    font-family: "Liberation Sans", sans-serif;
    font-size: 11pt;
    font-weight: 700;
    color: #1E293B;
    margin-top: 14pt;
    margin-bottom: 6pt;
}

p {
    margin-top: 0;
    margin-bottom: 9pt;
    text-align: justify;
}

.lead-p {
    font-size: 10.5pt;
    font-weight: 500;
    line-height: 1.6;
    color: #0F172A;
    margin-bottom: 14pt;
}

/* Callouts System */
.callout {
    border-left: 4px solid #0284C7;
    background: #FAF9F5;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin: 16pt 0;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #0284C7;
    page-break-inside: avoid;
}

.callout-distinction {
    border-left: 4px solid #D97706;
    background: #FFFBEB;
    border-color: #FCD34D;
    border-left-width: 4px;
}

.callout-idea {
    border-left: 4px solid #0D9488;
    background: #F0FDFA;
    border-color: #99F6E4;
    border-left-width: 4px;
}

.callout-warning {
    border-left: 4px solid #DC2626;
    background: #FEF2F2;
    border-color: #FECACA;
    border-left-width: 4px;
}

.callout-title {
    font-family: "Liberation Sans", sans-serif;
    font-size: 9pt;
    font-weight: 800;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
    color: #0F172A;
}

.callout p {
    font-size: 9.3pt;
    margin: 0;
    text-align: left;
}

/* Mathematical Presentation — Matplotlib SVG Images */
/* Inline math: vertically aligned to text baseline */
img.math-inline-img {
    display: inline-block;
    vertical-align: -0.35em;
    height: 1.2em;
    width: auto;
    max-height: 1.6em;
    border: none;
    background: transparent;
    padding: 0;
    margin: 0 1px;
}

/* Display math: centered block with generous vertical breathing room */
.math-display {
    background: #FAF9F5;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 14px 20px;
    margin: 14pt 0;
    text-align: center;
    page-break-inside: avoid;
    overflow-x: auto;
    line-height: 1.0;
}

img.math-display-img {
    display: block;
    margin: 0 auto;
    max-width: 90%;
    height: auto;
    border: none;
    background: transparent;
    padding: 0;
}

/* Fallback for failed renders */
code.math-display-fallback {
    display: block;
    text-align: center;
    font-family: "Liberation Mono", monospace;
    font-size: 9pt;
    color: #475569;
    padding: 4px;
}

code.math-inline-fallback {
    font-family: "Liberation Mono", monospace;
    font-size: 8.8pt;
    color: #475569;
    background: #F1F5F9;
    padding: 1px 3px;
    border-radius: 2px;
}

.math-inline {
    display: inline;
}

/* Figures & Diagrams */
.figure-container {
    margin: 18pt 0;
    text-align: center;
    page-break-inside: avoid;
}

.figure-container img {
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
}

.figure-caption {
    font-family: "Liberation Sans", sans-serif;
    font-size: 8.5pt;
    color: #475569;
    margin-top: 8px;
    text-align: left;
    line-height: 1.45;
    padding: 0 4px;
}

.figure-caption strong {
    color: #0F172A;
}

/* Tabular Presentation */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 14pt 0;
    font-size: 8.8pt;
    font-family: "Liberation Sans", sans-serif;
    page-break-inside: avoid;
}

th {
    background-color: #0F172A;
    color: #FFFFFF;
    font-weight: 700;
    padding: 7px 10px;
    text-align: left;
    border: 0.5pt solid #0F172A;
    font-size: 8.5pt;
}

td {
    padding: 6.5pt 10px;
    border-bottom: 0.5pt solid #E2E8F0;
    border-left: 0.5pt solid #F1F5F9;
    border-right: 0.5pt solid #F1F5F9;
    vertical-align: top;
}

tr:nth-child(even) td {
    background-color: #F8FAFC;
}

/* Status Badges */
.badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: "Liberation Sans", sans-serif;
    font-size: 7.5pt;
    font-weight: 700;
    letter-spacing: 0.3px;
    white-space: nowrap;
}
.badge-locked { background: #E0F2FE; color: #0284C7; border: 1px solid #BAE6FD; }
.badge-req { background: #FEF3C7; color: #D97706; border: 1px solid #FDE68A; }
.badge-unk { background: #EDE9FE; color: #7C3AED; border: 1px dashed #C4B5FD; }
.badge-cand { background: #CCFBF1; color: #0D9488; border: 1px dotted #5EEAD4; }
.badge-def { background: #F1F5F9; color: #64748B; border: 1px solid #CBD5E1; }
.badge-rej { background: #FEE2E2; color: #DC2626; border: 1px solid #FCA5A5; }

/* Code Blocks & Pygments Styling */
pre {
    background-color: #0F172A;
    color: #F8FAFC;
    border-radius: 6px;
    padding: 12px 14px;
    font-family: "Liberation Mono", monospace;
    font-size: 8.3pt;
    line-height: 1.45;
    margin: 12pt 0;
    page-break-inside: avoid;
    white-space: pre-wrap;
    word-break: break-word;
}

code {
    font-family: "Liberation Mono", monospace;
    font-size: 8.6pt;
    background-color: #F1F5F9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0F172A;
}

pre code {
    background-color: transparent;
    padding: 0;
    color: inherit;
}

/* Lists */
ul, ol {
    margin-top: 0;
    margin-bottom: 9pt;
    padding-left: 22px;
}

li {
    margin-bottom: 3.5pt;
}

/* Table of Contents */
.toc-container {
    page: toc-page;
    page-break-before: always;
    page-break-after: always;
    padding-top: 10px;
}

.toc-title {
    font-family: "Liberation Sans", sans-serif;
    font-size: 24pt;
    font-weight: 800;
    margin-bottom: 20px;
    border-bottom: 2px solid #0F172A;
    padding-bottom: 8px;
}

.toc-part-header {
    font-family: "Liberation Sans", sans-serif;
    font-size: 11pt;
    font-weight: 800;
    color: #0284C7;
    margin-top: 16px;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 3px;
}

.toc-item {
    display: flex;
    justify-content: space-between;
    font-family: "Liberation Sans", sans-serif;
    font-size: 9pt;
    padding: 3px 0;
    border-bottom: 1px dotted #E2E8F0;
}

.toc-item-title {
    font-weight: 600;
    color: #0F172A;
}

.toc-item-sub {
    font-family: "Bitstream Charter", "Liberation Serif", serif;
    font-style: italic;
    color: #64748B;
    margin-left: 8px;
}

hr {
    border: 0;
    height: 1px;
    background-color: #E2E8F0;
    margin: 18pt 0;
}
"""

# Cache for rendered math SVGs (avoids re-rendering the same expression)
_math_svg_cache: dict = {}


def _normalize_latex_for_mpl(latex: str) -> str:
    """Normalize LaTeX for matplotlib mathtext compatibility.
    
    Matplotlib's mathtext is not full LaTeX — it handles most math but has gaps.
    Key substitutions:
      \\text{...}        -> \\mathrm{...}   (upright roman text in math)
      \\operatorname{...}-> \\mathrm{...}   (named operators)
      \\displaystyle     -> (remove)         (not supported; use larger fontsize instead)
      \\mathbb{R}        -> \\mathbb{R}      (supported natively)
      \\|                -> \\Vert           (double vertical bar)
    """
    # Remove \displaystyle (not supported by mathtext)
    latex = re.sub(r'\\displaystyle\s*', '', latex)
    # Replace \text{...} with \mathrm{...}
    latex = re.sub(r'\\text\{', r'\\mathrm{', latex)
    # Replace \operatorname{...} with \mathrm{...}
    latex = re.sub(r'\\operatorname\{', r'\\mathrm{', latex)
    # Replace \| with \Vert for double norm bars
    latex = latex.replace(r'\|', r'\Vert')
    return latex


def latex_to_svg_data_uri(latex: str, fontsize: float = 11.0, display: bool = False) -> str:
    """Renders a LaTeX expression to an inline SVG data URI using matplotlib mathtext.
    
    For display math, uses a larger fontsize instead of \\displaystyle.
    Returns an <img> tag with the SVG embedded as a data URI.
    Returns a styled fallback span if rendering fails.
    """
    cache_key = (latex, fontsize, display)
    if cache_key in _math_svg_cache:
        return _math_svg_cache[cache_key]

    normalized = _normalize_latex_for_mpl(latex)

    try:
        expr = r'$' + normalized + r'$'

        fig = plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_alpha(0.0)  # Transparent background

        t = fig.text(0.0, 0.0, expr,
                     fontsize=fontsize,
                     ha='left', va='bottom',
                     color='#0F172A')  # Ink color matching body text

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = t.get_window_extent(renderer=renderer)

        # Avoid zero-size figures
        w_in = max((bbox.width + 14) / fig.dpi, 0.1)
        h_in = max((bbox.height + 10) / fig.dpi, 0.1)
        fig.set_size_inches(w_in, h_in)

        buf = io.BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight',
                    transparent=True, pad_inches=0.04,
                    dpi=fig.dpi)
        plt.close(fig)

        svg_bytes = buf.getvalue()
        b64 = base64.b64encode(svg_bytes).decode('ascii')
        data_uri = f'data:image/svg+xml;base64,{b64}'

        if display:
            result = f'<img class="math-display-img" src="{data_uri}" alt="{latex}">'
        else:
            result = f'<img class="math-inline-img" src="{data_uri}" alt="{latex}">'
        
        _math_svg_cache[cache_key] = result
        return result
    except Exception as exc:
        plt.close('all')  # Safety cleanup
        # Graceful fallback: styled code span
        if display:
            return f'<code class="math-display-fallback">{latex}</code>'
        return f'<code class="math-inline-fallback">{latex}</code>'


def convert_latex_in_text(text: str) -> str:
    """Preprocesses Markdown text, converting LaTeX equations to matplotlib SVG images.
    
    Processing order:
    1. Protect fenced code blocks and inline code spans from conversion.
    2. Convert display math $$...$$  -> centred SVG image block.
    3. Convert inline math $...$ -> baseline-aligned SVG image.
    4. Restore protected code spans.
    """
    # Step 1 — Protect code blocks and inline code
    code_cache = []
    def save_code(match):
        code_cache.append(match.group(0))
        return f"%%VERBATIM_CODE_{len(code_cache)-1}%%"

    text = re.sub(r'```[\s\S]*?```', save_code, text)
    text = re.sub(r'`[^`\n]+`', save_code, text)

    # Step 2 — Convert display math $$ ... $$
    def replace_display(match):
        latex = match.group(1).strip()
        if not latex:
            return match.group(0)
        img_tag = latex_to_svg_data_uri(latex, fontsize=13.0, display=True)
        return f'<div class="math-display">{img_tag}</div>'

    text = re.sub(r'\$\$(.*?)\$\$', replace_display, text, flags=re.DOTALL)

    # Step 3 — Convert inline math $ ... $
    def replace_inline(match):
        latex = match.group(1).strip()
        if not latex or '\n' in latex:
            return match.group(0)
        return latex_to_svg_data_uri(latex, fontsize=10.5, display=False)

    text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', replace_inline, text)

    # Step 4 — Restore code blocks
    for i, code_item in enumerate(code_cache):
        text = text.replace(f"%%VERBATIM_CODE_{i}%%", code_item)

    return text

def transform_chapter_markdown(content: str, chapter_prefix: str) -> str:
    """Transforms raw markdown chapter into designed editorial layout."""
    lines = content.splitlines()
    if not lines:
        return content

    first_line = lines[0].strip()
    rest_lines = lines[1:]

    # Parse # Chapter XX — Title: Subtitle
    ch_match = re.match(r'^#\s*Chapter\s*([0-9A-Za-z]+)\s*[—:-]\s*(.*?)(?::\s*(.*))?$', first_line)
    if ch_match:
        ch_num = ch_match.group(1)
        ch_title = ch_match.group(2).strip()
        ch_sub = ch_match.group(3).strip() if ch_match.group(3) else ""

        # Extract leading blockquote if it's a reading level / overview
        meta_html = ""
        quote_html = ""
        remaining_idx = 0
        for i, line in enumerate(rest_lines):
            sline = line.strip()
            if sline.startswith(">"):
                clean_q = sline.lstrip("> *").rstrip("*").strip()
                if "Level" in clean_q or "reading" in clean_q:
                    meta_html = f'<div class="chapter-metadata-banner">{clean_q}</div>'
                else:
                    quote_html = f'<div class="chapter-quote">"{clean_q}"</div>'
                remaining_idx = i + 1
            elif sline.startswith("---") or not sline:
                remaining_idx = i + 1
            else:
                break

        rest_body = "\n".join(rest_lines[remaining_idx:])

        sub_tag = f'<div class="chapter-subtitle">{ch_sub}</div>' if ch_sub else ""
        header_block = f"""
<div class="chapter-opening">
    <div class="chapter-pill">CHAPTER {ch_num}</div>
    <div class="chapter-title">{ch_title}</div>
    {sub_tag}
    {meta_html}
    {quote_html}
</div>
"""
        return header_block + "\n" + rest_body

    return content

def generate_part_break_html(part_info: dict) -> str:
    """Generates a full-bleed dark page divider for macro-parts."""
    return f"""
<div class="part-break">
    <div class="part-number">{part_info["part_num"]}</div>
    <div class="part-title">{part_info["part_title"]}</div>
    <div class="part-subtitle">{part_info["part_subtitle"]}</div>
    <div class="part-divider-line"></div>
    <div class="part-synopsis">{part_info["synopsis"]}</div>
</div>
"""

def generate_toc_html(chapters: list, appendices: list) -> str:
    """Generates an editorial Table of Contents page."""
    toc_lines = [
        '<div class="toc-container">',
        '    <div class="toc-title">Table of Contents</div>'
    ]

    current_part = None
    all_files = [(ch, "chapter") for ch in chapters] + [(app, "appendix") for app in appendices]

    for fpath, ftype in all_files:
        fname = Path(fpath).name
        prefix = fname[:2] if ftype == "chapter" else "A_"
        
        # Check if new part starts
        if prefix in PARTS_SPEC and PARTS_SPEC[prefix]["part_num"] != current_part:
            current_part = PARTS_SPEC[prefix]["part_num"]
            toc_lines.append(f'    <div class="toc-part-header">{current_part} &bull; {PARTS_SPEC[prefix]["part_title"]}</div>')

        # Read first line for title
        with open(fpath, "r", encoding="utf-8") as f:
            first = f.readline().strip()
            m = re.match(r'^#\s*(?:Chapter\s*([0-9A-Za-z]+)\s*[—:-]\s*)?(.*?)(?::\s*(.*))?$', first)
            if m:
                num = m.group(1) or ("App " + fname[0])
                name = m.group(2).strip()
                sub = m.group(3).strip() if m.group(3) else ""
                sub_html = f'<span class="toc-item-sub">— {sub}</span>' if sub else ""
                toc_lines.append(
                    f'    <div class="toc-item">'
                    f'        <span class="toc-item-title">{num} &bull; {name}{sub_html}</span>'
                    f'    </div>'
                )

    toc_lines.append('</div>')
    return "\n".join(toc_lines)

def build_figure_gallery():
    """Generates PNG raster previews of all SVGs and creates an HTML gallery."""
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    svg_files = sorted(FIG_DIR.glob("*.svg"))
    print(f"[*] Building figure gallery for {len(svg_files)} figures...")

    gallery_cards = []
    for svg_p in svg_files:
        png_p = GALLERY_DIR / f"{svg_p.stem}.png"
        # Render SVG to PNG using resvg or weasyprint
        try:
            # Weasyprint simple render for raster preview
            cmd = ["weasyprint", str(svg_p), str(png_p)]
            subprocess.run(cmd, capture_output=True)
        except Exception:
            pass

        gallery_cards.append(f"""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 20px;">
            <h3 style="margin-top:0; font-family: sans-serif; font-size: 14px; color: #0F172A;">{svg_p.name}</h3>
            <img src="../figures/{svg_p.name}" style="max-width: 100%; border-radius: 4px; border: 1px solid #CBD5E1;">
        </div>
        """)

    index_html = f"""<!DOCTYPE html>
<html>
<head><title>RHAN-NXA Figure Gallery</title></head>
<body style="background: #F8FAFC; padding: 30px; font-family: sans-serif;">
    <h1 style="color: #0F172A;">RHAN-NXA Technical Figure Manifest & Gallery</h1>
    <p style="color: #64748B;">Automated visual catalog of all {len(svg_files)} verified publication diagrams.</p>
    <div style="display: grid; grid-template-columns: 1fr; gap: 20px;">
        {"".join(gallery_cards)}
    </div>
</body>
</html>
"""
    with open(GALLERY_DIR / "INDEX.html", "w") as f:
        f.write(index_html)
    print(f"    Figure gallery generated at: {GALLERY_DIR / 'INDEX.html'}")

def build_pdf():
    """Main publication assembly and WeasyPrint PDF compilation."""
    chapters = sorted(glob.glob(str(BOOK_DIR / "*.md")))
    appendices = sorted(glob.glob(str(APP_DIR / "*.md")))

    print(f"[1/5] Assembling {len(chapters)} chapters and {len(appendices)} appendices...")
    
    # Build Table of Contents HTML
    toc_html = generate_toc_html(chapters, appendices)

    # Assemble content with Part Breaks and MathML
    body_blocks = [toc_html]
    unified_md_blocks = []

    for ch_path in chapters:
        fname = Path(ch_path).name
        prefix = fname[:2]

        # Insert Part Break if applicable
        if prefix in PARTS_SPEC:
            body_blocks.append(generate_part_break_html(PARTS_SPEC[prefix]))

        with open(ch_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            unified_md_blocks.append(raw_content)
            unified_md_blocks.append("\n\n---\n\n")

            # Transform to designed chapter format
            transformed = transform_chapter_markdown(raw_content, prefix)
            # Convert LaTeX math to MathML
            math_converted = convert_latex_in_text(transformed)
            # Convert markdown to HTML
            ch_html = markdown.markdown(
                math_converted,
                extensions=["tables", "fenced_code", "sane_lists"]
            )
            body_blocks.append(ch_html)

    for app_path in appendices:
        fname = Path(app_path).name
        if fname.startswith("A_"):
            body_blocks.append(generate_part_break_html(PARTS_SPEC["A_"]))

        with open(app_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            unified_md_blocks.append(raw_content)
            unified_md_blocks.append("\n\n---\n\n")

            transformed = transform_chapter_markdown(raw_content, "App")
            math_converted = convert_latex_in_text(transformed)
            app_html = markdown.markdown(
                math_converted,
                extensions=["tables", "fenced_code", "sane_lists"]
            )
            body_blocks.append(app_html)

    # Write unified markdown for portability & source inspection
    unified_md_text = "".join(unified_md_blocks)
    with open(UNIFIED_MD_PATH, "w", encoding="utf-8") as f:
        f.write(unified_md_text)
    print(f"[2/5] Unified Markdown written to: {UNIFIED_MD_PATH} ({len(unified_md_text.splitlines())} lines).")

    # Cover Page HTML
    cover_html = f"""
    <div class="cover-page">
        <div>
            <div class="cover-badge">Research Monograph &bull; Technical Field Manual</div>
            <div class="cover-title">RHAN-NXA<br>Technical Handbook</div>
            <div class="cover-subtitle">Active Foveated Inference, Latent Prediction Errors, and Recurrent Belief Dynamics in Adversarial Machine Perception</div>
        </div>
        <div class="cover-meta">
            <div>
                <strong>Architecture:</strong> Generation-1 Visual Core<br>
                <strong>Belief Substrate:</strong> 5-Tuple B_t = (z_t, S_t, U_t, E_t, A_t)<br>
                <strong>Transformer Scale:</strong> ~23.3M Trainable Parameters
            </div>
            <div>
                <strong>Epistemic Framework:</strong> Dirichlet Evidential Inference<br>
                <strong>Document Status:</strong> Pre-Registration Specification<br>
                <strong>Publication Version:</strong> 2.0 (Monograph Redesign)
            </div>
        </div>
    </div>
    """

    # Post-process body HTML for custom badges and callouts
    assembled_body = "\n".join(body_blocks)

    # Style status badges
    badge_replacements = [
        (r'\[● LOCKED\]', '<span class="badge badge-locked">&#x25CF; LOCKED</span>'),
        (r'\[▲ REQUIRED\]', '<span class="badge badge-req">&#x25B2; REQUIRED</span>'),
        (r'\[\? UNKNOWN\]', '<span class="badge badge-unk">? UNKNOWN</span>'),
        (r'\[◆ CANDIDATE\]', '<span class="badge badge-cand">&#x25C6; CANDIDATE</span>'),
        (r'\[○ DEFERRED\]', '<span class="badge badge-def">&#x25CB; DEFERRED</span>'),
        (r'\[✕ REJECTED\]', '<span class="badge badge-rej">&#x2715; REJECTED</span>'),
        (r'\*\*LOCKED\*\*', '<span class="badge badge-locked">LOCKED</span>'),
        (r'\*\*REQUIRED\*\*', '<span class="badge badge-req">REQUIRED</span>'),
        (r'\*\*UNKNOWN\*\*', '<span class="badge badge-unk">UNKNOWN</span>'),
        (r'\*\*REJECTED\*\*', '<span class="badge badge-rej">REJECTED</span>'),
        (r'\*\*DEFERRED\*\*', '<span class="badge badge-def">DEFERRED</span>'),
    ]
    for pattern, rep in badge_replacements:
        assembled_body = re.sub(pattern, rep, assembled_body)

    # Wrap figures in container with styled captions
    def wrap_figure(match):
        alt = match.group(1)
        src = match.group(2)
        # normalize path relative to DOCS_DIR
        clean_src = src.replace("file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/", "")
        clean_src = clean_src.replace("file:///home/ferrarikazu/Adversarial Cognitive Model/docs/rhan_nxa/", "")
        return f"""
<div class="figure-container">
    <img src="{clean_src}" alt="{alt}">
    <div class="figure-caption">{alt}</div>
</div>
"""
    assembled_body = re.sub(r'<img\s+alt="([^"]*)"\s+src="([^"]*)"\s*/?>', wrap_figure, assembled_body)

    # Full HTML Document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>RHAN-NXA Technical Handbook</title>
    <style>
{BOOK_CSS}
    </style>
</head>
<body>
{cover_html}
{assembled_body}
</body>
</html>
"""

    print("[3/5] Compiling publication HTML to PDF via WeasyPrint...")
    from weasyprint import HTML

    html_doc = HTML(string=full_html, base_url=str(DOCS_DIR))
    html_doc.write_pdf(str(PDF_PATH))

    pdf_size_mb = os.path.getsize(PDF_PATH) / (1024 * 1024)
    print(f"\n=======================================================")
    print(f"SUCCESS: RHAN-NXA Technical Handbook PDF Built Successfully!")
    print(f"Output File: {PDF_PATH}")
    print(f"File Size:   {pdf_size_mb:.2f} MB")
    print(f"=======================================================\n")

    # Generate Figure Gallery
    print("[4/5] Updating Figure Gallery & Previews...")
    build_figure_gallery()

    # Visual QA via pdftoppm
    print("[5/5] Executing Visual QA: Rasterizing sample PDF pages...")
    QA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Rasterize first 10 pages and select mid-book pages at 150 DPI
        cmd_qa = ["pdftoppm", "-png", "-r", "150", "-f", "1", "-l", "12", str(PDF_PATH), str(QA_DIR / "page")]
        subprocess.run(cmd_qa, capture_output=True)
        print(f"    Rasterized sample pages saved to: {QA_DIR}")
    except Exception as e:
        print(f"    Visual QA pdftoppm warning: {e}")

if __name__ == "__main__":
    build_pdf()
