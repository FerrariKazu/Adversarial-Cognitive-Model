# Scientific audit — RHAN Technical Handbook v3.1

Issue register for the v3.1 remediation. Each entry records the affected
locations, the claim that was there, the authoritative source, the decision, and
the evidence that validates it. Statuses are deliberately narrow:

| Status | Meaning |
|---|---|
| **Resolved** | The claim now matches a verified source, and a check enforces it. |
| **Resolved — retracted** | The old claim was wrong; it is preserved as withdrawn, not deleted. |
| **Unresolved** | The repository cannot settle it. The claim is marked unresolved in the PDF. |
| **Deferred** | Requires an experiment or compute that was not run. Never presented as done. |

Machine-readable counterparts: `docs/rhan_nxa/data/numbers.json` (values and
statuses), `claims.json` (claim → metric → artifact), `training_lineage.json`,
`data/figure_manifest.json`. Checks: `scripts/handbook/validate_numbers.py`,
`tests/test_handbook_registry.py`, and the stages run by
`scripts/validate_rhan_handbook.py`.

---

## N01 — MAC accounting contradiction

* **Severity:** critical.
* **Locations:** Ch 29.2 (L7/L8), Ch 24.2, Ch 7, 10, 17, 26, 28, exec overview, App A, figure H.
* **Was:** 1.82 vs 1.70 GMAC, ratio 1.07×.
* **Source:** `Gen2FoundationModel.forward` returns after `_trunk_forward()` when `is_gist_only`,
  so the tied refinement block never executes in the feedforward arms. Analytic count includes
  attention.
* **Decision:** `ff_112` = 1.456 GMAC, `recurrence_only` = 1.816 GMAC, ratio 0.802. Ch 29 rewritten;
  every other occurrence already used the corrected pair.
* **Validation:** `cost.macs_ff`, `cost.macs_rec`, `cost.macs_ratio` in the registry;
  `verify_claims.py` §6 asserts 1.456 G; `grep` for `1.70 GMAC` / `1.07×` returns nothing.
* **Status:** resolved.

## N02 — Robustness headline and the corrected attack

* **Severity:** critical.
* **Locations:** exec overview, Ch 22, Ch 26, Ch 29, App D.
* **Was:** the PGD-10 paired difference (+5.29 pp) presented without qualification; retention
  computed from PGD-10; an invalid APGD value used in the same discussion.
* **Source:** audit (n = 500, α = 2.5ε/50) gives `recurrence_only` 24.8 % against 32.6 % under the
  old step rule; the D3 final rows give `ff_112` 38.0 % and `recurrence_only` 32.4 % under the old
  rule, 24.8 % corrected.
* **Decision:** the PGD-10 headline stays, labelled as PGD-10 and paired over 8 subsets. Retention
  is reported under PGD-50 on identical 500-image protocols: `ff_112` 0.526, `recurrence_only` 0.394.
  The APGD headline for `ff_112` is marked unavailable.
* **Validation:** `paired.*`, `final.ff.retention_pgd50`, `final.rec.retention_pgd50`;
  `audit.rec.pgd50_corrected` = 24.8.
* **Status:** resolved, with the strongest-attack comparison **unresolved** (no valid `ff_112`
  APGD row).

## N03 — Robust accuracy described as a lower bound

* **Severity:** critical.
* **Locations:** Ch 22.8 (block quote), Ch 17.
* **Was:** "These are lower bounds on robustness under stronger attacks."
* **Source:** a finite attack search can miss adversarial examples, so measured robust accuracy is
  an upper bound on true robust accuracy under the threat model.
* **Decision:** rewritten with the three-way distinction (accuracy under a named attack; empirical
  robustness; certified robustness, which is unavailable).
* **Validation:** the phrase "upper bound" and the table of three statement kinds are in Ch 22.8;
  `tests` do not enforce prose, so this is a reviewed edit, not a machine check.
* **Status:** resolved.

## N04 — False project-duration claim

* **Severity:** high.
* **Locations:** Ch 22.4.
* **Was:** "For two years of project history…".
* **Source:** no record establishes the project's age; the artifact timestamps in the tree span
  October 2026.
* **Decision:** removed; the section now cites the producing revision and its date.
* **Validation:** `grep -rn "two years" docs/rhan_nxa` returns nothing.
* **Status:** resolved.

## N05 — "The arm ranking reverses"

* **Severity:** high.
* **Locations:** Ch 22.7.
* **Was:** the ranking reverses between D8 and the audit.
* **Source:** D8 APGD-CE: 22.7 vs 18.3 (Δ 4.4); audit: 23.6 vs 21.2 (Δ 2.4). `recurrence_only` is
  ahead in both.
* **Decision:** the advantage's *size* changes; its sign does not. McNemar p = 0.090 at n = 500, so
  neither sample ranks the arms. Rewritten.
* **Validation:** `audit.paired_rec_minus_gen1_apgd`, `d8.rec.apgd_ce`, `d8.gen1.apgd_ce`.
* **Status:** resolved.

## N06 — Training-exposure confound

* **Severity:** critical.
* **Locations:** Ch 27.1, Ch 29.
* **Was:** "Adversarial epochs at ε = 0.094: 15 vs 15."
* **Source:** `PHASE_LADDER` = g2_gist_only → recurrence_only; `run_gen2_phase` inherits the previous
  phase's FINAL weights; `PHASE_TOTAL_EPOCHS` gives 30 + 15. `ff_112` warm-starts from DINOv2 with 15.
  The inherited 30 epochs are 10 clean + 20 adversarial (`curriculum_for_phase_epoch`).
* **Decision:** the table shows phase, inherited, cumulative (45 vs 15) and effective-trainable
  counts, and states that architecture-only attribution is limited.
* **Validation:** registry `lineage.*`; `training_lineage.json`; the lineage figure draws from the
  manifest, not from hand-typed numbers.
* **Status:** resolved as a disclosure; the causal question remains **unresolved** because the
  epoch-matched control (`g2_gist_only_cont`) has not been run.

## N07 — E1 cell D tensor shapes and budget

* **Severity:** critical.
* **Locations:** Ch 30.1 (E1).
* **Was:** cell D needs "relax the `(B,4,4)` reshape" and "`n_tokens=64`"; cost ≈ 6.5 GPU-h per seed
  from 4 × 0.45 s/batch.
* **Source:** `PATCH_SIZE = 14`, so a 56-px fovea gives 4×4 = 16 patch tokens; `model.py:215` reshapes
  to `(B,4,4)`, which is correct; `PositionSensitiveErrorPool` defaults to `n_tokens=16`. The real
  blocker is that `Gen2FoundationModel(fovea_size=…)` passes `img_size=fovea_size`, so a 112-px input
  with a 56-px fovea has no constructor path.
* **Decision:** prerequisites corrected; the GPU-hour budget **withdrawn** pending a measured dry run.
  No budget is quoted.
* **Validation:** the shape arithmetic is asserted in `tests/test_handbook_registry.py`
  (`test_epsilon_and_input_range_arithmetic` covers ε only; token counts follow from
  `PATCH_SIZE` in `backbone.py` and are stated in the chapter). No dry run was executed.
* **Status:** partially resolved (shapes); **deferred** (throughput and budget).

## N08 — Attack clamp versus threat-set figure

* **Severity:** critical.
* **Locations:** Ch 22.1 figure caption, Ch 22.1.1 (new).
* **Was:** the figure caption said the clip returns candidates to the valid input range.
* **Source:** `run_pgd_ce_eval` clamps to `[-4, 4]`, which strictly contains the valid normalised
  range (−2.118 to 2.640).
* **Decision:** the implemented set is documented as the ε-ball ∩ `[-4,4]`; the caption and a new
  subsection tabulate the bounds. Because the realised set is a superset of the intended one, the
  measured accuracy remains an upper bound. The attack code was **not** changed: changing it would
  invalidate every number in the chapter, and the current clamp cannot inflate robustness.
* **Validation:** channel bounds asserted in `test_epsilon_and_input_range_arithmetic`; the clamp
  constants are cited by line number.
* **Status:** resolved as documentation; a projection onto the valid range is **deferred** as an
  attack code change requiring re-measurement.

## N09 — "No J2 artifacts"

* **Severity:** critical.
* **Locations:** exec overview, Ch 22.4, 29.5, App C.
* **Was:** blanket statements that no J2 artifacts exist.
* **Source:** Jobs A/B/C have no artifacts in this checkout (App C.3). The *script* exists and is
  inspectable, which is a different fact from "no artifacts".
* **Decision:** the text distinguishes instrument from artifact and adds Ch 22.9.
* **Validation:** `grep -rn "No J2" docs/rhan_nxa` returns nothing; App C.3 lists the expected output
  paths.
* **Status:** resolved.

## N10 — J2 as a run

* **Severity:** critical.
* **Locations:** Ch 22.9 (new), App C.3.
* **Was:** J2 described only as an intention.
* **Source:** reading `cloud_setup/colab_j2_eval_train.py` at HEAD shows (i) `build_loader` normalises
  and `collect_outcomes` then calls `torchattacks.APGD` on normalised tensors with no wrapper — the
  §22.4 defect; (ii) Job B trains `ff_56` as `g2_gist_only` with a renamed checkpoint, i.e. 30 epochs
  and a different curriculum from `ff_112`; (iii) Job A's checkpoint list omits `ff_56`.
* **Decision:** documented as three code defects to fix before running; no result claimed. The
  "+18.6 pp" claim has no record in the repository, so there is nothing to retract and nothing to
  support; the handbook does not use it.
* **Validation:** source-verification table in Ch 22.9 cites the functions and constants.
* **Status:** resolved as an audit; the experiments are **deferred**.

## N11 — The locked decision rule

* **Severity:** high.
* **Location:** Ch 30, Ch 17.
* **Was:** a 2σ rule referred to without definition.
* **Source:** the only 2σ mentions in the tree are Job C's "2σ training-variance bar" and the
  `ff_112_seed42/99` row in `flags.py`. No document defines whether σ is a standard deviation of
  seed accuracies, a standard error, or the σ of a paired difference.
* **Decision:** the rule is not applied. The handbook states the ambiguity and what would settle it
  (≥ 3 training seeds per arm, a paired difference, and a pre-registered σ definition).
* **Validation:** `training_lineage.json` records `g2_gist_only_cont` as a planned control.
* **Status:** **unresolved** — flagged, not reinterpreted.

## N12 — Stale interpretations and claims

* **Severity:** high.
* **Locations:** whole manuscript.
* **Was:** several sentences retained the old interpretation after the number changed (Ch 22.7,
  22.8, 26.6, 29.2, 30.1, App F HY-009).
* **Source:** each is listed above.
* **Decision:** each was rewritten in place; retracted values stay in the registry with their
  reason, and the validator fails if a retracted value appears in the executive overview without a
  retraction note.
* **Validation:** `scripts/handbook/validate_numbers.py` (retracted-value scan, claim graph);
  `CHANGELOG.md` records the superseded values.
* **Status:** resolved for the issues listed; a full prose audit of all 47 files was not performed
  by a human reader.

---

## Issues found in this pass beyond the brief

### A1 — The v3.0 explanation of the `ff_112` APGD figures was itself wrong

* **Severity:** critical.
* **Was:** 15.6 % and 16.8 % were "measured at 8/255 (ε_norm ≈ 0.1388)".
* **Source:** `git show 938ada3^:training/eval_gen2_foundation.py` line 262 passes `eps=0.031` to
  `torchattacks` on tensors normalised by `tf_val`; `torchattacks` clamps to `[0,1]`, outside the
  valid normalised range. The artifact records no ε for the block, so the revision, not the
  artifact, is the authority.
* **Decision:** both values `RETRACTED`; the 8/255 reading withdrawn; the 1.48× conversion retained
  only as arithmetic.
* **Validation:** registry `final.ff.apgd_ce_invalid` / `apgd_t_invalid` with four limitations
  each; Ch 22.4; App F `EV-APGD-BUDGET-AMBIGUITY`; `CHANGELOG.md` 3.1 §1.
* **Status:** resolved — retracted.

### A2 — Naming

* **Severity:** medium.
* **Was:** "Recurrent Hybrid Attention Network" in the Markdown handbook; "Recurrent Hierarchical
  Attention Networks" in the LaTeX monograph title page and `pdfsubject`; "Robust Hierarchical
  Attention Network" in generated figure suites.
* **Decision:** canonical expansion is *Recurrent Hierarchical Attention Network*; *hybrid* names
  the four mechanisms of Ch 5; the legacy "Robust Hierarchical" form is recorded as legacy. Cover,
  running title, README and glossary changed together.
* **Validation:** `grep -rn "Hybrid Attention"` returns only the mechanism chapter and explanatory
  notes.
* **Status:** resolved.

### A3 — A caption with square brackets bypassed the figure pipeline

* **Severity:** medium.
* **Was:** the threat-set figure was written as a raw `<figure>` by pandoc because the caption's
  bracketed interval broke the build's image regex; it received no build-time number and appeared
  in no list of figures.
* **Decision:** caption rewritten without brackets; the build now reports 32 figures, all consumed
  by its own preprocessor.
* **Validation:** the build's figure count equals the manifest count, and no `src="../figures`
  survives into the HTML.
* **Status:** resolved.

---

## Notebooks

The brief refers to "Notebook 1 and Notebook 2". The tree contains **one** notebook, `T4x2.ipynb`:
four cells (clone the repository, change directory, execute a long inline script, one empty cell),
three of which carry stored outputs. It is a Colab dispatch wrapper, not an analysis source: it
computes no metric on its own, reads no artifact into a figure, and its stored outputs are execution
logs with no recorded configuration, checkpoint or seed that would let them be tied to a result. It
was therefore **not executed** and none of its stored output is used as evidence. The scripted
instrument that does the equivalent work is `cloud_setup/colab_j2_eval_train.py`, audited in
Chapter 22.9, and its artifacts are absent (Appendix C.3).

## Not done, and why

| Item | Reason |
|---|---|
| Matched-ε `ff_112` APGD, AutoAttack, EOT at registered strength, epoch-matched control, two extra seeds | require GPU runs; the scripts exist but are not executed here |
| E1 cell D dry run and GPU-hour budget | requires the model change that separates input size from fovea size; no budget is quoted without it |
| Training-seed confidence intervals | one checkpoint per arm; every interval in the handbook is a subset interval and says so |
| Human page-by-page visual read of the PDF | pages were rasterised and measured, and figures pass an overlap check; no page was inspected by eye in this pass |
| Prose audit of every paragraph | the numeric and claim audits are mechanical; the prose review covered the chapters named in the register |
| Executing any notebook | the only notebook is a dispatch wrapper with stale logs and no provenance; see *Notebooks* above |
