# 6 — What generalises, and what to do next

The v3.4 work is small in volume — ten figures, one library, one check — but it changed how the
handbook is produced. This document records the parts worth keeping, the parts that are still weak,
and the concrete next steps.

## 1. Five things that worked

**Facts from a hooked run, not from the source.** `model_facts.py` answers questions the code cannot
answer by reading: how many times a block is applied in one pass (56), how many times the tokeniser
is called (5), how many evidential passes the gaze policy adds (18). The first attempt to report a
parameter budget also showed why this matters: the deduplicated total and the naive child sum differ
by 3.1 M parameters because two modules hold the same weights. A document that quoted either number
without saying which it meant would be wrong in a way no reviewer could see.

**One primitive library.** Ten drawings that share a canvas, a palette, a badge and four edge types
read as one system. The alternative — ten figures each with its own conventions — is what the
previous edition had, and it is why its architecture material read as a collection of diagrams
rather than an account of a model.

**Print size measured on the delivered file.** The source drawing is 170 mm; the page prints 165.8 mm.
The check that matters recomputes the ratio from the PDF and multiplies the smallest declared font by
it. That is the difference between "the labels are 5.8 pt" (false at print time) and "the smallest
printed label is 5.66 pt" (true, and verifiable by anyone with the PDF).

**Generate, merge, then test the merge.** Five generators write into one manifest. The test that
counts generators and asserts every v3.4 id is present failed with `107 == 97` when the merge was
missing — a defect that would otherwise have shipped as "the new figures exist but are not in the
book". Counting in a test, from the files, is cheap insurance.

**Status before number.** Every printed value carries `VERIFIED`, `SUPPORTED_WITH_LIMITATIONS` or
`RETRACTED`. The reader does not have to trust the prose to know how much weight a number bears, and
the two retracted values stay visible with their withdrawal rather than disappearing.

## 2. Four things that are still weak

1. **No automated check that a drawing is correct.** Provenance and legibility are checked; meaning
   is not. The only mitigation is the four-way evidence class and the alt text, both of which are
   written by the same person who drew the figure.
2. **Caption–figure agreement is unchecked.** The typography audit verifies adjacency, not
   correspondence. An earlier edition shipped a caption describing a different image; it was found
   by reading.
3. **No human read of all 254 pages.** Every page is measured; none is read end-to-end by a person
   in the final pass. This remains the largest gap in the QA story and is stated in the handbook,
   the validation report and this dossier.
4. **The PDF is not byte-reproducible.** The SHA identifies the build; the sources reproduce it
   functionally. If exact reproduction matters, the build needs pinned fonts and stripped metadata.

## 3. Next steps, in order of value

| Step | Why | Effort |
|---|---|---|
| A semantic caption check: compare the figure's manifest title/alt text with the caption's first sentence | catches the defect class that only a reader finds today | medium |
| Visual regression: store a 32×32 fingerprint per figure and fail when a drawing changes without a manifest update | catches silent artwork drift, which the current gates cannot see | low |
| A page-by-page human read, recorded as a checklist with initials and date | closes the standing limitation honestly | one day |
| Deterministic PDF: fixed font subsetting, stripped creation date, `SOURCE_DATE_EPOCH` | makes the SHA a reproducible identity | medium |
| Extend the print-size check to all 107 figures, not just the ten with the smallest labels | the 97 carried figures were reviewed at 3.3 but not re-measured in 3.4 | low |
| A lint for `handbook_chapter` in the manifest against the section that references the figure | the three wrong chapter labels would have failed it | low |

## 4. Checklist for the next edition

1. `python3 docs/rhan_nxa/figures/model_facts.py --write` — refresh facts if any code changed.
2. Regenerate every figure family that reads the changed facts; do not hand-edit a PNG.
3. `make_v34_schematics.py --merge` (or the current equivalent) — and read the merge count it prints.
4. `python3 scripts/handbook/build_numbers.py --check` — a stale registry must fail here.
5. `python3 scripts/build_rhan_handbook_pdf.py` — note the page count it prints.
6. `python3 scripts/validate_rhan_handbook.py` — all six gates, not five.
7. `python3 scripts/handbook/qa_v34_contact_sheet.py` — print sizes on the delivered file.
8. Open the contact sheet and look at the smallest-type figures at 200 dpi.
9. Diff `docs/research/RHAN_HB_metrics.json` — anything that moved should be explainable in one
   sentence, or it is a regression.
10. Update `CHANGELOG.md`, the figure inventory and Appendix G together; the three are cross-cited.
11. Re-run `pytest tests/test_handbook_registry.py`.
12. Record what is *not* verified in the validation report before the edition is circulated.

## 5. Open questions

- **Should the v3.4 schematics live in the handbook or in a separate architecture note?** They are
  drawn from code and will age with it; every code change that alters a shape makes a drawing stale.
  A generated architecture note would make that staleness obvious; a PDF hides it.
- **How much drawing is too much?** The book is at 107 figures and 2.37 pages per figure. The
  coverage deliverable lists 14 chapters with fewer than two figures; that is deliberate, not a gap
  to fill.
- **Can the evidence class be derived rather than declared?** Class A could in principle be inferred
  from the drawing code (does it read `model_facts.json`? does it read a `report/` artifact?), which
  would remove one manual step and one opportunity to mislabel. Not attempted here.
