# 5 — Verification log

This is the record of what was actually run against the delivered handbook, with the output that
matters and the exact artifact each line refers to. It is a log, not a claim of completeness: §6
lists what was *not* verified.

Environment: Python 3.10, PyMuPDF, PyTorch (CPU for the hooked run), pandoc, WeasyPrint 69.0,
`pdftoppm` from Poppler, on Linux.

## 1. Rebuild

```bash
python3 docs/rhan_nxa/figures/model_facts.py --write      # hooked run -> data/model_facts.json
python3 docs/rhan_nxa/figures/make_v34_schematics.py --merge
python3 scripts/build_rhan_handbook_pdf.py
```

Build tail:

```
  [assemble] 361 contents entries, 107 figures, 48 fragments
  [weasyprint] docs/rhan_nxa/RHAN_NXA_Technical_Handbook.pdf (9.9 MB)
  [pdf] Producer:        WeasyPrint 69.0
  [pdf] Pages:           254
  [pdf] Page size:       595.276 x 841.89 pts (A4)
```

The build substitutes every `{{N:id}}` token from `data/numbers.json`; an unknown id would have
aborted it. It did not.

## 2. Validation

```bash
python3 scripts/validate_rhan_handbook.py
```

```
pdf          : docs/rhan_nxa/RHAN_NXA_Technical_Handbook.pdf  (254 pages, 9.91 MB)
sha256       : 93253add4d9465d2f0df167199c1d26d0d46b8ea772ea643ffa09007ff5997e6
source       : 50 markdown files, 77,458 words
figures      : 107 embedded, 15 table rows catalogued
pages        : 254 rendered and measured; blank=0 overflow=0
contents     : 361 entries, 0 unresolved, 5 part dividers checked
typography   : text_outside_frame=0, text_collisions=0, hyphenated_identifiers=0,
               caption_problems=0, orphan_headings=0, undersized_figures=0,
               vertical_holes=0; 107 captions adjacent to their image
  [ok] scripts/handbook/validate_numbers.py
  [ok] docs/rhan_nxa/verify_claims.py
  [ok] scripts/audit_handbook_numbers.py
  [ok] scripts/audit_handbook_deltas.py
  [ok] scripts/audit_rhan_handbook_pdf.py
  [ok] scripts/audit_rhan_typography.py

RESULT: PASS -> report/handbook_audit/validation.json
```

Individual stage tails, for the two that are quantitative:

```
registry: 276 metrics; 2 retracted; 0 errors, 0 warnings
unexplained : 0
distinct pp values: 55; occurrences: 129
matched: 129   unmatched: 0
```

## 3. Print-size check

```bash
python3 scripts/handbook/qa_v34_contact_sheet.py
```

```
id                        page       mm  scale  min pt
V34-ARCH-OVERVIEW           35    165.8   0.98    5.66
V34-BLOCK-ANATOMY           37    165.8   0.98    5.66
V34-FOVEA-MECHANISM         43    165.8   0.98    5.66
V34-GIST-PATH               45    165.8   0.98    5.66
V34-DIRICHLET-FANOUT        56    165.8   0.98    5.66
V34-BELIEF-STEP             58    165.8   0.98    5.66
V34-TRAIN-STEP              84    165.8   0.98    5.66
V34-EXEC-SCHEDULE           95    165.8   0.98    5.66
V34-PROTOCOL-MATRIX        139    165.8   0.98    5.66
V34-REUSE-ACCOUNT          161    165.8   0.98    5.66
```

Exit status 0. A figure that failed to clear 5.5 pt, or that could not be located exactly once in
the PDF, would have failed the command instead of printing a warning.

## 4. Registry tests

```bash
python3 -m pytest tests/test_handbook_registry.py -q
# 19 passed in 0.37s
```

The suite covers the registry schema, the derived arithmetic, the placeholder contract, the claim
graph, the lineage, the palette and the five-generator manifest merge. The merge test is the one
that would catch a v3.4 regression:

```python
v34 = json.loads((DATA / "figure_manifest_v34.json").read_text())
expected = (len(priority) + len(carried) + len(suite["figures"])
            + len(chapter["figures"]) + len(v34["figures"]))
assert {f["id"] for f in v34["figures"]} <= {f["id"] for f in man["figures"]}
assert man["figure_count"] == len(man["figures"]) == expected
```

Before the v3.4 merge was wired into the test it read `107 == 97` and failed — which is how the
missing generator was noticed rather than by counting figures by hand.

## 5. This dossier

```bash
python3 docs/research/RHAN_HB_figures/make_research_figures.py
# wrote 10 figures to docs/research/RHAN_HB_figures
# wrote docs/research/RHAN_HB_metrics.json
```

The generator exits non-zero if its own text-overlap check reports a collision. It reports none.
`RHAN_HB_metrics.json` is written by the same run, so every number in documents 1–6 traces to an
artifact and moves when the artifact moves.

## 6. What the log does not establish

| Not verified | Why it matters | Where it is stated |
|---|---|---|
| a human page-by-page read of all 254 pages | geometry checks cannot see a caption that describes the wrong figure | handbook §5 limitation; `07_deliverables/validation_report.md`; §7 of this dossier |
| that a schematic is a *correct* depiction | the check proves provenance and legibility, not reading | `RHAN_HB_02_FIGURE_SYSTEM.md` §7 |
| that the 147 transcribed metrics equal the raw run | the raw output is not in the repository | registry status field, printed in Appendix G |
| that the PDF build is byte-reproducible | the SHA identifies *this* build; a rebuild may differ in metadata | this log records the SHA rather than claiming determinism |
| robustness numbers beyond the recorded battery | single checkpoint per arm; no matched-ε APGD for `ff_112` | handbook Chapters 22, 25, 29 |

## 7. Build identity

Two builds of v3.4 exist in this repository's history:

| Build | SHA-256 | Difference |
|---|---|---|
| 28 October 2026 | `ae80fdff…d2454ce9` | the release as first assembled |
| 15 November 2026 | `93253add…ff5997e6` | appendix G.1 now lists five generators; the duplicated `G.7` heading is resolved as G.7 / G.8; the three wrong `handbook_chapter` labels in the v3.4 manifest are corrected. No claim or number changed; word count moved 77,449 → 77,458 |

The version stamp stays **3.4** because the corrections are internal consistency fixes within the
release, not a new edition. If a future reader needs to reproduce the exact October file, the
October sources differ from the current tree only in those three places.
