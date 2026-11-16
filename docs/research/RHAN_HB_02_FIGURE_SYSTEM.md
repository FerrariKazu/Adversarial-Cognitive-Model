# 2 — The v3.4 figure system

This document is about the drawing system, not about RHAN. It describes the primitives, the rules
that bind a drawing to the code, the facts module that supplies every number, and the print-size
check that measures the delivered PDF. It is written so the method can be reused for another
document without importing the RHAN-specific content.

![How a drawing reaches the page](RHAN_HB_figures/hb04_schematic_pipeline.png)

## 1. What a schematic may and may not claim

A schematic in this book has one job: make a *computational* claim legible — what runs, in what
order, how often, with what shapes. It may not imply a measurement. The four-way class is the
mechanism that enforces this:

| Class | Meaning | How it is drawn |
|---|---|---|
| **A** | implementation-derived: every element traces to a code path | solid boxes, `IMPLEMENTATION-DERIVED` badge |
| **B** | experimentally derived: values come from a recorded artifact | solid boxes, `EXPERIMENTALLY DERIVED` badge |
| **C** | conceptual: design intent with no executed path | hatched fill, `CONCEPTUAL` badge |
| **D** | mixed: an implementation drawing with one measured element | `MIXED: A + B` badge, the measured element is sourced in the caption |

Eight of the ten v3.4 figures are class A, one is B (`V34-PROTOCOL-MATRIX`, whose cells are protocol
facts read from the attack audit) and one is D (`V34-FOVEA-MECHANISM`, which adds one measured number
— the 2.84e-5 maximum resampling difference — to a code-path drawing). No class C figure was added:
the release draws computation that exists.

The badge is rendered from the same string as the manifest's `evidence_class`, so the drawing cannot
claim one class while the record says another.

## 2. The primitives

`docs/rhan_nxa/figures/schematics.py` is a millimetre-coordinate canvas, which is the whole reason
the drawings can be reasoned about in print terms:

| Constant | Value | Why |
|---|---:|---|
| `PAGE_W` | 170.0 mm | the A4 text block width every full-width figure is authored for |
| `FS_TITLE` | 8.6 pt | panel titles |
| `FS_BODY` | 7.3 pt | box titles and labels |
| `FS_SMALL` | 6.5 pt | sub-labels |
| `FS_SHAPE` | 6.1 pt | tensor-shape annotation (monospace) |
| `FS_BADGE` | 5.8 pt | evidence badge — the smallest type in any drawing |

The canvas is `figsize=(w/25.4, h/25.4)` with `xlim=(0, w)` and `ylim=(h, 0)`, so one unit is one
millimetre and a coordinate in the code is a distance on paper. Font sizes are point sizes at
authoring scale; §6 shows how they are checked after the 2.5 % printing reduction.

The visual grammar is four edge types and one badge:

| Element | Encoding | Used for |
|---|---|---|
| data edge | solid, ink | tensors flowing forward |
| gradient edge | dashed, brown | a backward path (drawn only where the gradient path is the point) |
| control edge | dotted, grey | a selection or schedule decision, not a tensor |
| shared link | dash-dot, blue | "these two boxes are the same parameters" |
| badge | outlined chip, top right | the evidence class, in words |

Fills are tints keyed by role (neutral, wash, blue, orange, green, violet, grey), and hatching marks
anything that is not implemented. Boxes take a title, an optional sub-label and an optional shape
tag; the shape tag is set in monospace so a reader can tell a tensor dimension from prose.

Every drawing runs a **fit check** before saving: each text artist registered with a maximum width is
measured against that width in millimetres, and an overflow is reported. This catches the failure
mode where a label is correct but runs out of its box — a class of defect that compiles cleanly and
looks wrong on paper.

## 3. Facts before drawing: `model_facts.py`

No shape, count or parameter total in a v3.4 drawing is typed into the drawing code. They come from
`docs/rhan_nxa/data/model_facts.json`, written by `figures/model_facts.py`, which:

1. builds the real `Gen2FoundationModel` for each phase;
2. runs **one real forward pass** with forward hooks on the module objects;
3. records, per module, the number of applications and the input/output shapes;
4. records parameter counts twice — a deduplicated total and the naive sum over `children()`.

The last point is the most useful part of the module, because the two numbers disagree and both are
defensible answers to different questions. For `gen1_core`:

| Quantity | Value |
|---|---:|
| deduplicated total | 27,740,495 |
| naive child sum | 30,859,315 |
| difference | 3,118,820 |

The gist encoder *holds* the backbone's patch embedder (the zero-new-tokeniser invariant) and the
AIS-v2 policy holds the predictor, so a child-sum counts both twice. Anything that reports a
parameter budget has to say which number it means; the drawings use the deduplicated one, and the
manifest records the module it belongs to.

Application counts can only come from a run. For `gen1_core`, one forward pass gives:

| Module | Applications | Reading |
|---|---:|---|
| `trunk.block` | 48 | 12 blocks × 4 glimpses |
| `tied_refinement.block` | 8 | 1 tied block × 2 passes × 4 glimpses |
| **block applications** | **56** | the figure's chip |
| `patch_embed` | 5 | gist + 4 crops — the tokeniser is called once per view |
| `evidential_head` | 23 | 4 loop steps + 18 AIS-v2 candidate scorings + 1 readout |
| `gist_encoder`, `cls_head`, `ev_readout` | 1 each | once per pass |

A reader cannot get the 56, the 5 or the 23 by reading the model source; `TiedRecurrence` calls one
block K times and the same trunk is called once per glimpse. That is exactly why the numbers are
hooked rather than transcribed.

## 4. Rules applied to every drawing

1. **Read numbers at generation time.** A drawing imports `model_facts.json`; it never copies a value
   from an earlier figure, a caption or a log.
2. **Import constants from the code being drawn.** `PATCH_SIZE`, `TRUNK_DEPTH` and the curriculum
   functions are imported, so a model change alters the drawing.
3. **No figure numbers in titles or in the artwork.** Numbering belongs to the document build; a
   raster with "Figure 12" baked in goes stale the moment a figure is inserted above it.
4. **Declare the evidence class and the sources** in the manifest record next to the drawing.
5. **Bound the update, not just the label.** Each drawing declares its limitations in the manifest
   (e.g. "the 2.84e-5 figure is one audit run on one input, not a bound").
6. **Alt text and a limitation per figure.** All 107 records carry both; the ten v3.4 records carry
   the four-way class as well.
7. **Self-check layout before saving** (text overlaps and box fit).
8. **One visual language.** Palette, arm colours and edge grammar come from `figures/style.py` and
   `figures/schematics.py`, so the ten can be read against each other and against the 97 older
   figures.

## 5. Honest duplication

Three v3.4 figures overlap an existing figure by design, and the text cross-references rather than
duplicating silently:

| New | Existing | Different question |
|---|---|---|
| `V34-ARCH-OVERVIEW` | `figA_two_arms` | the whole pass vs the two-arm contrast |
| `V34-EXEC-SCHEDULE` | `figJ_recurrence_unrolled` | *when* work happens vs *what* the loop computes |
| `V34-FOVEA-MECHANISM` | `translate_vs_zoom` | how the crop is cut vs what it can see |

Three figures were considered and **not** drawn, because the evidence does not exist:

- a component-level runtime profile — no profiler artifact;
- an attention-map figure — the block returns no attention weights;
- a per-arm calibration plot — one checkpoint, no per-sample artifact.

The rule is the same one that governs the registry: a picture that has no artifact behind it would
be a class C drawing pretending to be class A, and the book already has enough conceptual figures
without adding more.

## 6. Print size is measured, not asserted

A drawing at 170 mm is placed in a 170 mm text block, but the placed box is not exactly 170 mm: the
CSS lays it out at 165.8 mm, so the artwork is reduced to **97.5 %** and every label shrinks with it.
The smallest label in a drawing is the 5.8 pt evidence badge, which therefore prints at
**5.66 pt** — above the 5.5 pt floor this edition sets, with headroom for the rasteriser.

![Printed geometry of the ten schematics](RHAN_HB_figures/hb05_print_geometry.png)

`scripts/handbook/qa_v34_contact_sheet.py` is the check. It does not trust the source drawing or the
build log; it opens the delivered PDF and:

1. reads the ten records from `data/figure_manifest_v34.json`;
2. finds the image on the page by **content fingerprint** (32×32 greyscale of the embedded image vs
   the PNG), not by pixel size — two schematics can share a canvas size, and an early version of the
   check had to be fixed for exactly that ambiguity: `V34-GIST-PATH` and another 170 × 78 drawing are
   both 1635 × 765 px;
3. converts the placed box to millimetres;
4. recomputes `scale = printed_mm / PAGE_W` and `min_pt = FS_BADGE × scale`, importing `FS_BADGE`
   from the drawing library so the floor cannot drift from the artwork;
5. writes `report/handbook_audit_v34/figure_placement_v34.json` and a contact sheet of the ten
   figures **as printed**, each labelled with page, width and smallest label;
6. exits non-zero if any figure is below the floor or cannot be located exactly once.

All ten pass: 165.8 mm, scale 0.98, 5.66 pt. The check is deliberately a release gate rather than a
report: a figure that prints too small should fail the build, not be discovered by a reader.

## 7. What this system cannot do

- **It cannot tell whether a drawing is *correct*.** It guarantees that a number came from the code
  and that the label is legible; whether the box labelled "evidential" is in the right place is a
  reading question. The four-way class and the alt text are the mitigation, not a proof.
- **The overlap check is a screening tool.** It flags text artists overlapping by more than 15 % of
  the smaller box; it cannot see a label clipped by a panel edge, and it says nothing about a line
  that passes through a box.
- **Print size is a necessary condition, not sufficient.** 5.66 pt is legible at 300 dpi in the PDF;
  it is not comfortable. Anything longer than a badge at that size should be checked by eye on the
  contact sheet.
- **The content fingerprint assumes the embedded image is the PNG.** If the build ever re-encodes
  figures (e.g. to JPEG), the check must compare perceptually instead of exactly, and the current
  exact-signature match will (correctly) fail rather than silently accept.
