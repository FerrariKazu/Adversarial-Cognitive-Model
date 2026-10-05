# RHAN-NXA Technical Handbook: Publication Engine Decision Record

**Document Version:** 1.0.0  
**Status:** APPROVED & BINDING  
**Evaluator:** Technical & Architectural Documentation Team  
**Evaluation Target:** WeasyPrint vs. Typst for Monograph-Grade PDF Production  

---

## 1. Executive Summary

Following a formal prototyping comparison on identical representative content (cover page, chapter openings, mathematical derivations, architecture SVGs, epistemic decision tables, and code blocks), **WeasyPrint (v69.0)** with **native MathML conversion (via `latex2mathml`) and designed Print CSS** has been selected as the primary publication engine for the RHAN-NXA Technical Handbook.

Typst (v0.11.1 CLI / v0.15.0 engine) was thoroughly investigated and prototyped using the `typst-author` skill methodology. While Typst demonstrated exceptional text-shaping and concise syntax for standalone documents, its architectural friction when operating as a compilation target for a 27-chapter Git-native Markdown repository outweighed its micro-typographical advantages.

---

## 2. Comparative Evaluation Matrix

| Evaluation Dimension | Weight | Typst (v0.11.1 / v0.15.0) | WeasyPrint (v69.0 + MathML) | Decision / Rationale |
|---|---|---|---|---|
| **Typography & Fonts** | High | **9.5 / 10** (HarfBuzz, native tracking) | **9.0 / 10** (Charter / Liberation Serif) | Typst has superior justification controls; WeasyPrint is very close with proper CSS. |
| **Mathematical Typesetting** | Critical | **9.5 / 10** (Native TeX-grade layout) | **9.0 / 10** (Vector MathML + DejaVu Math) | `latex2mathml` translates all display & inline math to native vector MathML. |
| **SVG & Diagram Fidelity** | Critical | **8.0 / 10** (`resvg` / `usvg`, filter limits)| **9.5 / 10** (Full SVG specs, filters, CSS) | WeasyPrint handles all 19 SVG diagrams, SVG CSS stylesheets, and filters natively. |
| **Markdown Interoperability** | Critical | **5.5 / 10** (Requires custom transpiler) | **9.5 / 10** (Native GFM + Python-Markdown) | 27 chapters & 3 appendices remain 100% native Markdown viewable on GitHub/IDEs. |
| **Identifier Handling** | High | **6.0 / 10** (Underscores `_` break markup) | **9.5 / 10** (HTML escapes correctly) | Code terms like `_nx_trainer`, `z_t`, `S_t` in prose break Typst without heavy escaping. |
| **Page-Break & Layout Control** | High | **9.0 / 10** (Declarative blocks, `#place`)| **9.0 / 10** (CSS3 Paged Media `@page`, flex)| Both offer precise control over margins, headers, footers, and breaks. |
| **Table & Code Presentation** | Medium | **9.0 / 10** (Built-in syntax highlighting) | **9.0 / 10** (Pygments codehilite, styled CSS)| Both produce clean, unbordered/bordered tabular data and dark code containers. |
| **Build Reproducibility** | High | **7.5 / 10** (Requires root flag, binary) | **9.5 / 10** (Standard Python ecosystem) | WeasyPrint compiles cleanly in standard Linux/WSL/cloud CI environments. |
| **Overall Score** | - | **64.0 / 80** | **74.0 / 80** | **WeasyPrint Selected** |

---

## 3. Detailed Technical Analysis

### A. Why WeasyPrint + MathML Was Selected
1. **Zero-Drift Markdown Authority**: The 27 chapters in `docs/rhan_nxa/book/` and 3 appendices in `docs/rhan_nxa/appendices/` represent the canonical source of truth. With WeasyPrint, researchers can view, edit, and review chapters using standard Markdown viewers and GitHub web previews without intermediate conversion distortion.
2. **Vector MathML Integration**: By preprocessing all `$ ... $` and `$$ ... $$` blocks through `latex2mathml`, WeasyPrint 69.0 renders true vector MathML utilizing installed mathematical glyphs (`DejaVu Math TeX Gyre`), completely eliminating blurry equations or raw ASCII leaks.
3. **Flawless SVG Pipeline**: The handbook relies on 19 SVG technical figures containing CSS classes, gradient defs, and custom clipPaths. WeasyPrint renders these natively with 100% vector fidelity, zero clipping, and exact color match to CSS tokens.
4. **Adoption of `weasyprint-pdf-skill` Principles**: Rather than treating HTML as a dumb conversion intermediate, the build pipeline applies a formal Print CSS architecture (`DESIGN_SYSTEM.md`) with explicit page masters (`@page :first`, `@page part-page`, `@page chapter-first`), running headers, alternating page numbers, and curated typographic scales.

### B. Why Typst Was Not Selected for This Repository
1. **Syntax Delimiter Clashes in Technical Prose**: In Typst, the underscore character `_` is a native delimiter for emphasis (`_italic_`). In a systems neuroscience and machine learning codebase with variables like `_nx_trainer`, `z_t`, `S_t`, `L_stab`, and `a_{t+1}`, standard technical prose repeatedly triggered parser errors or unclosed delimiter exceptions unless pre-escaped.
2. **No Native GFM Parsing**: Pandoc 2.9 (the LTS package on Ubuntu 20.04/22.04) does not support Typst output. Authoring in Typst would require either maintaining duplicate `.typ` files (abandoning Markdown) or maintaining a fragile custom regex AST transpiler.
3. **Sandbox / Path Isolation**: Typst rejects file imports that reference assets outside its project root unless `--root .` is explicitly passed, creating fragility across relative subfolder paths.

---

## 4. Skills Utilized & Evaluated

### Utilized Skills
* **`weasyprint-pdf-skill`** (`slapash/weasyprint-pdf-skill`): Used for print CSS architecture, page-geometry rules, semantic HTML structuring, and visual PDF QA methodologies.
* **`typst-author`** (`apcamargo/typst-skills`): Used to author, configure, and compile the Typst prototype for comparative benchmarking.
* **Visual SVG & Diagram Skills** (`svg-diagram`, `diagram-design`): Used to audit, resize, and de-overflow all 19 SVG figures.

### Evaluated But Not Adopted
* **Pandoc Typst Writer**: Evaluated for automated Markdown-to-Typst conversion; rejected due to LTS version constraints (`pandoc 2.9.2` lacks Typst output).
* **Browserless Headless Chrome / Puppeteer**: Evaluated for HTML-to-PDF; rejected due to lack of standard CSS Paged Media support (poor margin boxes, no native `@top-right` / `@bottom-left` running headers).
