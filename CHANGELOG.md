# Changelog — RHAN Technical Handbook

The handbook's change record, moved out of the PDF so the document can stay a
technical account rather than an audit trail. One page in the PDF (front matter,
*Provenance*) points here.

Values quoted below are the ones the corrected build reports. Where a value was
withdrawn, the withdrawn value is kept so a reader who saw the earlier edition can
find out what happened to it.

---

## 3.3 — 25 October 2026

This edition adds one section and two appendix sections; no earlier number was edited and no experiment
was re-run. It carries 243 pages and 97 figures, and the numeric registry grows from 135 to 276
metrics.

**The closure suite (Chapter 26.10).** Six arms evaluated in one kernel (environment `2461bc2`), one
loader, one attack wrapper and one budget: 5,000 clean images and the canonical 1,000-slice set at
5.417/255. The section prints the headline table for all six arms, the worst-case waterfall, the
T-sweep, the oracle-halting curve and its exit histogram, the four-row paired McNemar matrix, and the
5 × 5 linear-CKA matrix. Two findings do not appear elsewhere in the edition: robust accuracy is
strictly monotone in T on one checkpoint (a test-time-averaging effect, not a trained mechanism), and
the loop's clean advantage over its epoch-matched control is significant on 5,000 images while its
robust advantage is not.

**Provenance, stated the hard way.** The run's printed output was supplied as a transcription, so the
141 new metrics carry `SUPPORTED_WITH_LIMITATIONS` rather than `VERIFIED`; the artifact is
`report/closure/closure_run_2461bc2.json`, whose header says what it is and what it is not. Ten cells
disagree with `canonical_headline_preflight.json` or `controls_eval_20261021_202939.json` in the
second decimal place, and Chapter 26.10 prints both values side by side instead of averaging them.
The C4 arm has no locally recorded evaluation at all, and its checkpoint is absent from the
repository, so that row rests entirely on the transcription.

**Appendix E.7 — the six audit corrections.** R1–R6 as a table: APGD normalisation, step size, the
ε-unit ambiguity, the `ff_56` transform, slice RNG synchronisation and checkpoint identity, each with
the state it replaced and the size of the correction. Three are now enforced by tests or harness
assertions rather than described.

**Appendix H.10 — hashes.** The two dataset hashes were recomputed in this repository and match the
recorded values byte for byte, as do five of the six checkpoint hashes. The C4 checkpoint and the
preservation archive are reported, not verified, and the table says so.

**What did not change.** The retracted `ff_112` APGD values stay retracted; the headline robustness
evidence is still the pixel-space J2 battery; the recurrence-versus-control result is still quoted
with its test-time-averaging confound. The closure suite refines that confound rather than removing
it: it shows the loop's robust margin over C1 is not statistically established at 1,000 slices.

## 3.2 — 22 October 2026

This edition is a rendering and coverage release: no number changed, and no experiment was re-run.
It closes the Part II figure gap, repairs three layout defects found in the rendered PDF, and adds a
text-layer audit that the previous editions did not have.

**Figures — 84 to 97.** Six figures close the thinnest chapters: `MATH-UPDATES-001` (Ch. 7, the
blend and the belief update with the guarantee each carries), `IMPL-SHAPES-001` (Ch. 10, the three
forward paths in tensor shapes, with the sequence length read from `pos_embed` in the shipped
checkpoint), `IMPL-SEQ-001` (Ch. 13, the 18-token sequence and the fused QKV reshape),
`IMPL-MIXPREC-001` (Ch. 14, the dtype path through a training step and its attack),
`EVID-COVERAGE-001` (Ch. 25, which arm holds which kind of evidence) and `LIMIT-HYPOTHESES-001`
(Ch. 29, the 16 registered hypotheses by status). `CAL-CURVE-001` (Ch. 23, temperature scaling) was
generated in the 3.1 suite but never merged into the manifest or placed in a chapter; it is now in
both. Two more give the appendices the coverage they lacked: `LEDGER-STATUS-001` (App. F, the 34
ledger records by classification, counted at build time — the replacement for the earlier edition's
figure whose six hard-coded bars summed to 19) and `PROV-TRACE-001` (App. G, the record count in
each provenance layer). Density is 97 figures in 238 pages, one per 2.45.

Six further figures carry the mechanism and the design question: `ARCH-EXEC-001` (Ch. 3, the
architecture as designed against what the three evaluated arms execute), `GEOM-RECURRENCE-001`
(Ch. 4, what a glimpse samples and what the loop computes), `TRAIN-GRAD-001` (Ch. 11, forward and
backward flow per arm), `DIAG-BELIEF-001` (Ch. 23, whether the belief machinery responds to
evidence), `LINEAGE-DAG-001` and `EXP-FACTORIAL-001` (Ch. 27, the checkpoint lineage and the 2×2
design with only three of its four cells built). The belief diagnostic is the one that reports
against the design: `U_t` is near-constant and ranks errors worse than softmax entropy, and the
surprise probe shows a ratio of 1.04 where a mechanism that responded to evidence would show a
separation.

**Two evidence asymmetries are now stated in the handbook, not just in the ledger.** The only
recorded evaluation of the belief arms is an 8-sample, one-seed, one-step ε-sweep that reports 0.0 %
at every ε; Chapter 23.5 and Chapter 25.4 say so, and `EVID-COVERAGE-001` shows the empty cells. No
arm — including the four in the J2 battery — has per-sample outcome rows, which is why the n = 1,000
paired interval remains an open item rather than a computed one.

**Layout defects found in the rendered PDF and fixed.**

1. The Appendix A attack table overflowed the right margin by 51 pt: one cell carried five long
   paths, and the widest unbreakable token set the table's minimum width. The paths moved into the
   paragraph below it and the cells now hold short references.
2. The Appendix B module-presence matrix, nine columns of unbreakable arm names, overran the text
   block by the same margin. It is now printed as two panels of the same matrix.
3. A long identifier in an Appendix A cell still overflowed after the first fix; the CSS now allows
   a break inside a table cell (`overflow-wrap: anywhere`) instead of widening the table.

**Validation.** `scripts/audit_rhan_typography.py` is new and runs as a sixth stage of
`scripts/validate_rhan_handbook.py`. It reads the PDF text layer and checks: body text inside the
declared text block, no two text spans drawn on top of each other (combining accents excluded by
Unicode category), no identifier broken across a line by automatic hyphenation, every caption
within 26 pt of its image, no heading as the last item on a page, no figure narrower than 55 % of
the column, and no vertical hole of more than 150 pt inside the text block (part dividers excluded
by name). All seven are clean on this build; the first run found the Appendix A overflow and the
six part dividers, which is how the exceptions came to be documented rather than silently allowed.

**The ineffective epsilon assertion is gone.** `assert abs(eps_norm * MEAN_STD * 255 -
round(eps_norm * MEAN_STD * 255)) > 1e-6 or eps_norm in {0.0, 0.031, 0.062, 0.094}` passed for
almost any input, because the first disjunct is true whenever the product is not an integer, so it
could not establish which unit a radius was in. It has been deleted and replaced by
`tests/test_epsilon_units.py`, which converts the registered radii through
`curriculum_gen2.norm_to_pixel_eps` in both directions and checks the failure mode directly: a pixel
fraction (2/255, 4/255, 8/255) handed to a normalised interface is off by 1/mean(σ) ≈ 4.4, and
0.094 is 5.42/255 while 8/255 is 0.1388 normalised — a different radius, not a spelling of the same
one.

Contact sheets for all 238 pages are written to `report/handbook_audit/contact_sheets/`. No human
has reviewed them page by page; `07_deliverables/validation_report.md` states that limitation
rather than implying otherwise.

**Front matter.** Version 3.2, and the provenance page now lists the text-layer audit and points at
the contact sheets instead of claiming visual review.

---

## 3.1 — 21 October 2026

**Scientific corrections**

1. **The stored `ff_112` APGD figures are invalid, for a different reason than 3.0
   gave.** 3.0 recorded that `15.6 %` and `16.8 %` were measured at 8/255
   (ε_norm ≈ 0.1388) and withdrew the comparison on that basis. That explanation is
   withdrawn. `git show 938ada3^:training/eval_gen2_foundation.py` line 262 passes
   `eps=0.031` to `torchattacks` on tensors that `tf_val` has already normalised, so
   the radius was 0.031 in normalised units (≈1.9/255), and the library attack then
   clamped to `[0, 1]`, which is not the valid range of a normalised ImageNet tensor
   (≈−2.12 to 2.64). Both values are `RETRACTED` in `docs/rhan_nxa/data/numbers.json`.
   No valid `ff_112` APGD measurement exists at any radius.
2. **J2 would have inherited the defect.** `cloud_setup/colab_j2_eval_train.py`
   builds a normalised loader and calls `torchattacks.APGD` on it with no wrapper, so
   Job A's APGD rows would be invalid in the same way. Job B trains `ff_56` through
   the `g2_gist_only` recipe (30 epochs: 10 clean, then ε 0.031→0.094), not the
   `ff_112` recipe. Job A's checkpoint list omits `ff_56`. All three are recorded in
   Chapter 22.9 as code changes needed before the run is worth its compute.
3. **Robust accuracy is an upper bound.** Chapter 22.8 described finite-attack
   accuracy as a lower bound on robustness. A finite search can miss examples, so the
   measured value is an upper bound on true robust accuracy; no certified bound exists
   in this project.
4. **The arm ranking does not reverse.** Chapter 22.7 said it did. Both D8 (n = 300)
   and the audit (n = 500) put `recurrence_only` ahead of `gen1_core` under APGD-CE;
   what changes is the estimated size of the advantage (4.4 vs 2.4 pp). Rewritten.
5. **The inherited-exposure confound is in the comparison table.** Chapter 27.1 listed
   "15 versus 15 adversarial epochs", which is true only of the final phase.
   `recurrence_only` inherits 30 epochs from `g2_gist_only` (10 clean + 20 adversarial)
   before its own 15; `ff_112` warm-starts from DINOv2 and trains 15. The table now
   shows phase, inherited, cumulative and effective-trainable counts.
6. **The implemented clamp is documented as implemented.** `run_pgd_ce_eval` projects
   with `min(max(x_adv, x−ε), x+ε)` and then `clamp(-4, 4)`. The valid normalised range
   is inside that box, so the realised search set is a superset of the intended one.
   Because it only adds candidates, measured accuracy remains an upper bound. The
   threat-set figure caption said the clip returned candidates to the valid range; it
   now states the box the code applies, with the channel bounds tabulated (22.1.1).
7. **The false project-duration phrase is gone.** "For two years of project history"
   was not supported by any record in the repository.
8. **Naming resolved.** The acronym expands to *Recurrent Hierarchical Attention
   Network*, matching the prior LaTeX monograph's title page and `pdfsubject`. *Hybrid*
   is retained for the four attention mechanisms of Chapter 5 and is no longer offered
   as the expansion. Appendix I records the legacy "Robust Hierarchical" form used by
   the generated figure suites.

**New infrastructure**

* `docs/rhan_nxa/data/numbers.json` — 68 registered metrics with unit, display
  precision, status, artifacts, code, derivation and limitations; two are `RETRACTED`.
* `docs/rhan_nxa/data/numbers.schema.json` — the schema the registry is checked against.
* `docs/rhan_nxa/data/claims.json` — claim → metric → artifact → section graph.
* `docs/rhan_nxa/data/training_lineage.json` — checkpoint lineage with cumulative epochs.
* `docs/rhan_nxa/data/experiment_manifest.json` — the ten recorded artifacts with SHA-256 digests;
  a changed artifact fails validation until the registry is rebuilt.
* `docs/rhan_nxa/data/figure_manifest.json` — 32 figures with stable ids, evidence
  class, alt text, data ids, arm colours, chapter and limitations; 21 `EVIDENCE`,
  11 `CONCEPTUAL`.
* `docs/rhan_nxa/data/figure_suite_manifest.json` — the 19 suite figures, merged into
  the canonical manifest above by `make_priority_figures.py`.
* `scripts/handbook/build_numbers.py` — deterministic registry builder; `--check` fails
  on a stale registry.
* `scripts/handbook/render_numbers.py` — `{{N:id}}` substitution used by the PDF build;
  an unknown id aborts the build.
* `scripts/handbook/validate_numbers.py` — registry, claim, lineage and figure checks,
  now stage 1 of `scripts/validate_rhan_handbook.py`.
* `tests/test_handbook_registry.py` — 17 tests, including the ε and input-range
  arithmetic quoted in Chapter 22.1.1.
* `docs/rhan_nxa/figures/style.py` — shared palette, markers, rcParams and overlap
  checker; the legacy generator's constants are asserted equal to it.
* Five new figures: component status overlay (`ARCH-STATUS-001`), unrolled recurrence
  (`ARCH-RECURRENCE-001`), lineage DAG (`LINEAGE-DAG-001`), factorial grid
  (`EXP-FACTORIAL-001`), multi-radius robustness curve (`ROBUSTNESS-CURVE-001`).
* Nineteen further figures from the requested families, 12 `EVIDENCE` and 7
  `CONCEPTUAL`, generated by `docs/rhan_nxa/figures/make_figure_suite.py` and placed by
  `docs/rhan_nxa/figures/place_figures.py`: block diagram with tensor shapes, parameter
  waterfall, TRADES step, glimpse footprints and contact sheet, gaze order, configured
  schedule, decision gates, dependency roadmap (replacing the ASCII block in 30.5),
  evidence ladder, confound matrix, forest plot, dumbbell, attack-strength bars, UpSet
  overlap, per-subset raincloud, Pareto, order-sensitivity heatmap, graphical abstract.
  Nine requested families are **not** drawn because no artifact could support them; the
  list with reasons is Appendix G.6 and §D of `VALIDATION_REPORT.md`.
* The validator now checks figure-list entries against the page each figure points at,
  and ignores caption continuation lines in the list of figures (previously counted as
  unresolved contents entries once captions grew long enough to wrap).

---

## 3.0 — J2 edition

Corrections carried into 3.1 and still standing:

1. **ε conversion** is 1.477× (0.1388 normalised for 8/255 against 0.094), not 3×.
2. **Paired PGD-10 interval** recomputed: +5.29 pp, 95 % CI [3.95, 6.64], paired
   t(7) = 9.31. The older t ≈ 3.9 corresponds to an unpaired calculation.
3. **MAC accounting**: the feedforward arms were charged a tied-refinement block they
   never execute. `ff_112` is 1.456 G, not 1.696 G; the ratio to the four-pass loop is
   0.802, not 0.934. Chapter 7, 10, 17, 24, 26, 28, Appendix A and figure H corrected.
4. **Effective capacity**: `ff_112` trains 1,756,800 *fewer* parameters than the loop,
   measured from optimizer state (177 vs 191 entries); the nominal +18,432 is
   positional embeddings and reverses once dead weights are removed.
5. **LayerScale reference**: trunk γ norms have median 11.9 (4.3–141), not 1.31.
6. **T = 1 → T = 4 robust gain** is 5.6 pp, not 4.6; the random-gaze drop is 4.2 pp,
   not 4.4; cross-attack agreement is 1.7 pp, not 2.3.
7. **Predictor cost** derived from layer shapes: 2.76 M per call, 49.7 M total.
8. Presentation: duplicate cover removed, truncated cover facts restored, duplicate
   contents anchors fixed, figure removed from a table, list-of-figures captions
   trimmed at word boundaries, eight figure defects reworked at source.
