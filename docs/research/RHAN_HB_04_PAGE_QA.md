# 4 — Page and typography QA

The handbook is produced as Markdown → pandoc → HTML → WeasyPrint → PDF. That chain is powerful and
opaque: a bad table or a long identifier compiles without complaint and reaches the reader as a
cropped column or a word split across lines. This document describes the two independent
measurements applied to the *rendered* PDF — a typography audit of the text layer and a pixel audit
of every page — plus the contact sheets used for visual review, and the defects that were found this
way in earlier editions.

## 1. The text block

Every check is relative to the declared text frame, which is derived from the CSS:

| Edge | Position |
|---|---:|
| left | 20 mm = 56.7 pt |
| right | 538.6 pt (= page width − 20 mm) |
| top | 22 mm = 62.4 pt |
| bottom | 785.2 pt (= page height − 20 mm) |

Running heads and folios sit outside this frame by design, so any span in the top or bottom margin
band (within 2 pt of it) is exempt from the frame check. Everything else must be inside, with 1.5 pt
of slack for glyph side bearings.

## 2. The seven typography checks

`scripts/audit_rhan_typography.py` reads the PDF's text layer and runs seven checks. Each has an
explicit threshold, and the thresholds are the design decision — the check is only as good as its
definition.

| Check | Definition | Threshold | Why it exists |
|---|---|---|---|
| text outside frame | a span's bbox extends past the text frame | 1.5 pt slack | the failure a reader sees as a cropped line |
| text collisions | two spans overlap by more than a sliver of the smaller | >4 pt² and >30 % of the smaller box | two labels drawn on top of each other; combining accents are exempt by Unicode category |
| hyphenated identifiers | a line ends in `-` and the join matches a backticked identifier in the manuscript | identifier set from all `*.md` | `patch_embed` split as `patch_` / `embed` — a code name broken by auto-hyphenation |
| caption problems | the caption span ("Figure N.") must sit within 26 pt below an image, and not be a list-of-figures link | 26 pt | a caption separated from its figure, or a caption for a figure that is not on the page |
| orphan headings | the lowest body span is a heading and sits within 80 pt of the frame bottom | size ≥ 11.5 pt | a section title stranded at the foot of a page |
| undersized figures | an image is narrower than 55 % of the text width | 55 % | a figure accidentally inserted at a fraction of its designed width (the book has no deliberate insets) |
| vertical holes | a gap of more than 150 pt inside the text block, images counted as occupied | 150 pt | a bad page break leaving half a page blank |

Part dividers are the documented exception to the hole check: they are mostly white by design, so
they are detected (the page text, whitespace removed, starts with `PART` or `APPENDICES`) and skipped
by name rather than by loosening the threshold. On the v3.4 build there are six such pages: 26, 72,
113, 170, 194 and 204.

Two details matter for reproducibility:

- **Captions are identified structurally.** The string `Figure N.` appears both as a caption and as a
  link in the list of figures. A caption is the span that is *not* inside a link, which is why the
  check counts 107 captions on a book with 107 figures and does not double-count the list.
- **The identifier set is read from the manuscript**, not hard-coded: every backticked token of four
  characters or more containing `_`, `/` or `-` is protected. A new identifier is covered the moment
  it is written.

## 3. The pixel audit

`scripts/audit_rhan_handbook_pdf.py` renders every page at 120 dpi with `pdftoppm` and measures the
ink box:

| Measurement | Result on v3.4 |
|---|---|
| pages rendered and measured | 254 |
| blank pages (ink < 0.15 % of the page) | 0 |
| pages with ink inside the physical margin guard | 0 |
| minimum ink coverage | 0.565 % (a part divider) |
| median ink coverage | 8.69 % |
| maximum ink coverage | 38.1 % (a full-page figure plate) |

![Ink coverage across the 254 rendered pages](RHAN_HB_figures/hb07_page_ink.png)

The six sparse pages below 1.2 % are exactly the six part dividers, which is how the "no blank page"
statement stays honest: the tool lists near-empty pages rather than hiding them behind a threshold,
and the reason each one is sparse is a design decision on record.

## 4. Contact sheets

Two kinds of sheet are produced:

- `report/handbook_audit/contact_sheets/` — every page of the book, tiled, for a human pass;
- `report/handbook_audit_v34/contact_sheet_v34.png` — the ten v3.4 figures cropped from the rendered
  pages, each labelled with its page, printed width and smallest label.

The second is the one that was actually used during this release. It answers a question the metrics
cannot: whether the drawing *reads* at print size. A figure can pass every numeric check and still be
unreadable because a dashed edge disappears or two boxes read as one; that is a judgement made on
the crop, not on the JSON.

## 5. Defects this pipeline has caught

The value of a check is best shown by what it found. From the v3.2 and v3.3 rounds:

| Defect | Found by | Fix |
|---|---|---|
| Appendix A attack table overflowed the right margin by 51 pt | text-outside-frame check | long paths moved into the paragraph below; cells hold short references |
| Appendix B's nine-column module matrix overran the text block | text-outside-frame check | the matrix is printed as two panels |
| A long identifier still overflowed inside a table cell | text-outside-frame check | CSS `overflow-wrap: anywhere` inside cells instead of widening the table |
| A figure inserted between two table rows, splitting the table | page review of the rendered PDF | the figure moved below the table |
| 48 contents entries all pointing at page 16 | contents-resolution check | unique anchors per section |
| A duplicate cover page (the cover markdown rendered again as a body chapter) | page review | removed |
| Captions describing a different image than the one placed | caption-adjacency + review | captions rewritten against the placed file |
| Six part dividers flagged as "vertical holes" | vertical-hole check | the pages are detected and exempted *by name*, with the reason documented |

The pattern is consistent: the automated checks catch geometry and structure, and the visual pass
catches semantics. Neither replaces the other.

## 6. What these checks cannot see

- **Truth.** A perfectly typeset wrong number passes every check above. That is stage 2–4's job (§3
  of the numbers pipeline), and even those cannot catch a wrong interpretation.
- **Caption–image agreement beyond adjacency.** The caption check verifies that a caption is near an
  image; it does not verify that the caption describes *that* image. This exact defect occurred in an
  earlier edition and was found by reading, not by tooling.
- **Raster legibility.** Ink coverage says a page has content, not that a 5 pt label is readable. The
  v3.4 print-size check addresses that for the ten new figures specifically, because those drawings
  contain the smallest type in the book.
- **Colour and contrast.** No contrast measurement is performed on the tinted boxes; the palette is
  chosen to survive greyscale and deuteranopia, but that is a design argument, not a check.
- **Table semantics.** A table can be inside the frame, break cleanly and still mix units across
  columns; the per-column provenance rule is a writing convention, not a validator.
- **Reading order and rhythm.** Whether a page feels crowded, or whether a chapter opens well, is not
  measured. The contact sheets exist so a human can make that call.

## 7. The standing limitation

No human has read all 254 pages. Every page has been rendered and measured, the text layer is checked
for the seven defects above, and the v3.4 figures were inspected at print size — but a page-by-page
human read remains the appropriate last step before circulation. `07_deliverables/validation_report.md`
states this rather than implying otherwise, and this dossier repeats it because it is the single most
important qualification of everything in §2–§3.
