# 1 — What v3.4 changed, and why

The handbook moved from 243 pages / 97 figures (v3.3) to 254 pages / 107 figures (v3.4). No number
in the 276-metric registry moved, and no experiment was re-run. The release is a *representation*
release: it adds drawings of computation that already existed and the machinery to keep those
drawings tied to the code.

![107 records by family; orange marks the families that received a v3.4 drawing](RHAN_HB_figures/hb02_figure_families.png)

## 1. The problem the release addresses

v3.3 answered "what was measured" well: 97 figures, 59 of them `EVIDENCE`, each traceable to an
artifact. It answered "what does the model compute" less well. The architecture material was a
wallsheet drawing of the *design*, with the same boxes whether or not the corresponding code runs.
Three concrete confusions followed, and each is now fixed by a drawing that is read off a hooked run:

1. **"Thirteen blocks" vs "twelve blocks applied fifty-six times."** The design drawing showed the
   trunk as one box; the reader could not see that a forward pass applies the block stack 56 times
   (4 glimpses × 14 block applications for `gen1_core`) against a single 12 for the feedforward arm.
2. **Which parts run once and which run per step.** Nothing in the book separated the gist, the
   tokeniser, the loop body and the readout by *when* they execute, so a reader could not tell
   sequential work from repeated work.
3. **Which consumers of a tensor are real.** The evidential head produces a vector with four claimed
   downstream readers; only three are wired, and the design drawing showed all four.

A schematic cannot settle a performance question, but it can settle these three, and it can do so
from code rather than from intent.

## 2. The ten figures

Page numbers are those printed by the v3.4 build; a figure inserted earlier in the book moves them,
which is why no drawing carries a figure number.

| Manifest id | Page | Section | Shows | Class | Read from |
|---|---:|---|---|---|---|
| `V34-ARCH-OVERVIEW` | 35 | 3.1 | the whole pass: one tokeniser, 12 + 1 blocks, four-step loop, two-head readout, application chips | A | hooked `forward` |
| `V34-BLOCK-ANATOMY` | 37 | 3.3 | one block's operation order, its MACs and parameter share | A | `backbone.py` + analytic MAC formula |
| `V34-FOVEA-MECHANISM` | 43 | 4.2 | θ from the gaze, and why `s = 1` means a glimpse adds no detail | D | `foveation.py` + `step3_resolution_audit.json` |
| `V34-GIST-PATH` | 45 | 4.3 | `z_0` as a token mean, and `fuse()` as uncalled | A | `gist.py`, `model.py` |
| `V34-DIRICHLET-FANOUT` | 56 | 6.1 | one evidence vector, three wired consumers, one absent | A | `evidential_head.py`, `precision.py`, `ais_v2_policy.py` |
| `V34-BELIEF-STEP` | 58 | 6.4 | one update iteration and its four detach sites | A | `update_net_v2.py`, `model.py` |
| `V34-TRAIN-STEP` | 84 | 11.2 | one batch in three precision lanes: attack fp32, loss autocast, scaler/clip | A | `train_gen2_foundation.py`, `step6_latency_benchmark.json` |
| `V34-EXEC-SCHEDULE` | 95 | 13.3 | what runs at each step, with per-cell application counts | A | hooked `forward` |
| `V34-PROTOCOL-MATRIX` | 139 | 22.1 | four attack protocols and what each can support | B | `step1_attack_audit.json`, J2 reconciliation |
| `V34-REUSE-ACCOUNT` | 161 | 24.2 | parameter sets against applications per pass | A | `model_facts.json` |

Class: **A** implementation-derived, **B** experimentally derived, **C** conceptual, **D** mixed.
There is no class C figure in this release, because every drawing explains computation that exists
in the shipped code. Where a claim has no code or artifact behind it — e.g. the dotted "next gaze"
edge in the architecture figure — the element is drawn as a control/selection path and labelled, not
promoted into a finding.

![Evidence status of the 107 records, with the v3.4 class split](RHAN_HB_figures/hb03_evidence_composition.png)

## 3. What changed in the manuscript

Seven chapters received a figure and the paragraph that reads it. None was rewritten.

| File | Change |
|---|---|
| `01_concept/03_architecture_overview.md` | §3.1 gains the forward-pass overview + a paragraph on the three places it differs from the wallsheet; §3.3 gains the block anatomy |
| `01_concept/04_recurrence_information_flow.md` | §4.2 gains the fovea mechanism; §4.3 the gist path |
| `01_concept/06_belief_refinement.md` | §6.1 gains the evidence fan-out; §6.4 the belief step |
| `02_implementation/11_training_pipeline.md` | §11.2 gains the three-lane training step |
| `02_implementation/13_attention_and_recurrence_impl.md` | §13.3 gains the execution schedule |
| `03_evaluation/22_adversarial_evaluation.md` | §22.1 gains the protocol matrix |
| `03_evaluation/24_computational_cost.md` | §24.2 gains the reuse account |

Each inserted paragraph states what the reader should *see*, in the same voice as the surrounding
text. From §3.1:

> The figure is the answer to "what does the model actually compute", and it differs from a
> wallsheet architecture in three places: the tokenizer is a single node with two inbound paths
> (gist and crop), the refinement block is drawn inside the trunk box because it is a thirteenth
> block and not a second network, and the loop is drawn around the parts that are repeated rather
> than around the whole model. The count on each chip is an application count: `patch_embed` runs
> five times per pass, the block stack fifty-six.

## 4. New files

| File | Role |
|---|---|
| `figures/schematics.py` | the drawing library: millimetre canvas, module box, data/gradient/control/shared edges, evidence badge, fit check |
| `figures/make_v34_schematics.py` | the ten drawings, the v3.4 manifest, and the merge into the canonical manifest |
| `figures/model_facts.py` | runs the real model under forward hooks and writes shapes, call counts and parameter totals |
| `data/model_facts.json` | the facts file every drawing imports (weight-independent: shapes and counts) |
| `data/figure_manifest_v34.json` | the ten records before the merge: id, class, sources, alt text, limitations |
| `scripts/handbook/qa_v34_contact_sheet.py` | print-size check on the shipped PDF + contact sheet of the ten figures as printed |
| `report/handbook_audit_v34/figure_placement_v34.json` | page, printed width, scale, smallest label, pass/fail per figure |
| `report/handbook_audit_v34/contact_sheet_v34.png` | the ten figures cropped from the rendered pages, labelled with their metrics |
| `07_deliverables/figure_inventory_v34.md` | per-figure purpose, class, source, section, inspection status; the three figures considered and not drawn |

## 5. Bookkeeping that follows

1. **The manifest merges five generators**, not four (priority, carried legacy, suite, chapter,
   v3.4). `tests/test_handbook_registry.py` counts the fifth from the manifest file and asserts that
   every v3.4 id is present in the merged record, so a generator that stops merging fails a test
   rather than silently shrinking the book.
2. **`PROV-TRACE-001` was re-rendered.** Its bars are read from the manifest at generation time, so
   leaving it would have printed 97 records in a book whose Appendix G.8 says 107. The re-render also
   moved the "figures citing a metric" bar from 34 to 39. This is a count refresh, not a new claim.
3. **Appendix G gained G.8** (the ten figures, sources and print geometry). Its generator list now
   says five. The duplicate `G.7` heading that the insertion would otherwise have created was
   resolved: table provenance stays G.7, schematics are G.8.
4. **The front matter's version stamp** moved to `Version 3.4 · 28 October 2026`, and `CHANGELOG.md`
   gained the 3.4 entry that the Provenance page points at.

## 6. Defects found while making this record

Documenting the release is also how three latent inconsistencies were found. They are fixed in the
same tree; each is the kind of error that survives because nothing reads the field.

| Defect | Where | Fix |
|---|---|---|
| Three v3.4 records named the wrong chapter (`EXEC-SCHEDULE` = 4, `BLOCK-ANATOMY` = 13, `GIST-PATH` = 3) | `data/figure_manifest_v34.json`, produced by `make_v34_schematics.py` | chapter strings corrected to 13 / 3 / 4 in the generator; the three figures regenerated and re-merged |
| Appendix G.1 said "four generators exist" and listed four | `06_appendices/G_figure_and_table_provenance.md` | says five and names `make_v34_schematics.py` |
| Two sections numbered `G.7` | same appendix | table provenance stays G.7; the schematics section is G.8, and the G.8 cross-reference in the appendix intro was updated |

The chapter field is advisory — figure placement is computed from the chapter sources, and coverage
is computed from those references, so no page moved because of the wrong label. It still mattered:
the manifest is what a reader consults to find a figure, and a wrong chapter sends them to the wrong
part of the book.

## 7. Where the figures sit

![Figures per chapter, coloured by part; chapter 22 carries 13, chapter 4 nine](RHAN_HB_figures/hb08_chapter_density.png)

Density is not uniform and is not meant to be. The evaluation chapter that compares four attack
protocols (22) and the geometry chapter (4) are the densest, because those are the places where a
reader must see several things side by side. Two chapters (1 and 2) carry a single figure — the
orientation opener — and fourteen *files* in the book (those two, front matter and most appendices)
carry fewer than two. `figure_coverage.py` lists them by name in
`07_deliverables/figure_coverage.md` rather than padding them; a chapter with one good figure should
not gain a second one to satisfy a ratio.

## 8. What did not change

- The 97 carried figures are byte-identical except `provenance_layers.png` (count refresh above).
- No figure was deleted, replaced or renumbered; the v3.4 ten are additive.
- The registry is unchanged at 276 metrics (127 `VERIFIED`, 147 `SUPPORTED_WITH_LIMITATIONS`,
  2 `RETRACTED`).
- The v3.3 limitations stand, including the missing matched-ε adaptive attack for `ff_112` and the
  single checkpoint per arm. v3.4 adds drawings, not evidence.
