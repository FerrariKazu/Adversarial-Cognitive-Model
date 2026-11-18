# 07 — Training programme

## 1. Why staged

The system is not trained end-to-end. Three reasons, each grounded in this repository's history:

1. **Shared-gradient competition.** Earlier RHAN generations showed that individually promising
   components can train worse when optimized jointly, because their gradients compete. Adding the
   planner's objective to the encoder's would confound "the encoder is weak" with "the joint
   objective is badly conditioned".
2. **Measurement.** Each stage has an acceptance criterion that can fail on its own. A joint system
   that reaches 70 % tells you nothing about which component is responsible.
3. **Compute.** One 8 GB GPU. A joint model with a recurrent planner is not trainable at a scale
   where its results would mean anything; staged components are.

The stages are cumulative in *capability*, not in *parameters*: Stage B does not fine-tune Stage A;
Stage C fits a reliability model on Stage A's frozen features; Stage D fits the pair likelihood on
Stage A/B features; Stage E only changes the selection rule.

## 2. Stage A — visual competence

**Question.** Can the encoder plus a task head reach a competent single-observation cause classifier
on this simulator?

**Data.** 400 training scenes, 100 calibration scenes, 100 development scenes, 400 test scenes
(Gate E); scene-level split, no scene appears twice (`docs/11`). Each scene contributes one
observation (base action).

**Model.** Frozen `HandcraftedBackbone` features (8-d) → multinomial logistic regression (or a
2-layer MLP with 32 hidden units for the DINOv2 path). Optional: `DinoV2Backbone` 384-d CLS → linear
head, features cached once.

**Loss.** Cross-entropy with class weights (the four causes are balanced by construction; weights are
1.0 unless the generator drifts).

**Optimizer.** LBFGS for logistic regression (deterministic, no learning-rate sweep); AdamW
(3e-4, 200 steps, batch 64) for the MLP. Weight decay 1e-4. Three seeds (17, 42, 99) for the MLP.

**Acceptance.**

- overall accuracy ≥ 0.80 on development;
- balanced accuracy ≥ 0.75;
- **defect-vs-reflection balanced accuracy between 0.55 and 0.80** — this band is the *expected
  ambiguity*. A value near 1.0 means the simulator leaked a cue that makes the task trivial and the
  scene generator must be reviewed before Gate D. This is an unusual acceptance criterion and it is
  deliberate: on this task, a single observation *should not* fully resolve defect from reflection.

**Artifacts.** `results/stage_a.json` (metrics per seed), `results/stage_a_model.json` (coefficients),
`results/confusion_single_observation.json` (4×4).

**Failure handling.** If accuracy < 0.80, inspect the confusion matrix before changing the model:
- defect→reflection confusion is expected and does not count as failure of Stage A;
- shadow→none or none→shadow confusion indicates the shadow renderer is too weak;
- all-classes confusion indicates a feature bug, not a model bug.

## 3. Stage B — spatial evidence

**Question.** Can the system say *where* the evidence is, not only what it is?

**Data.** The same scenes, with ground-truth defect masks (line rasterisation) for defect scenes and
empty masks otherwise.

**Model.** A matched-filter localizer: the observation's oriented dark-ridge response, thresholded
at the 99th percentile of the response on training *null* scenes. The learned part is the threshold
(the response a clean plate produces), not the response itself. A U-Net is specified for Gate B but
not implemented in v1 because 400 scenes cannot train one honestly.

**Metrics.** IoU at threshold 0.3; pointing-game accuracy (max response inside the true mask);
localization-conditioned accuracy: among correct classifications, the fraction whose peak is inside
the true region (the "right answer for the right reason" check).

**Acceptance.** IoU ≥ 0.25, pointing ≥ 0.85, localization-conditioned accuracy ≥ 0.90. A high
classifier accuracy with low pointing accuracy is a red flag: it means the classifier is using a
scene-level cue (gloss, background) rather than the anomaly.

**Artifacts.** `results/stage_b.json`, `results/localization_examples/` (PNG per failure case, capped
at 12).

## 4. Stage C — reliability

**Question.** Can the system tell when its evidence is unreliable, and does that improve selective
prediction?

**Data.** Calibration split (100 scenes × 4 actions) for fitting; development split for evaluation.

**Models.**

1. **Temperature scaling** on the Stage A posterior: one parameter `T`, fitted by minimizing NLL on
   calibration. Report ECE (15 bins), Brier, NLL before and after.
2. **Quality → error model**: logistic regression on the six quality components, target = the record's
   leading-hypothesis LLR sign is correct. Report its own ECE (it is a probability model and can be
   miscalibrated too).
3. **Conformal risk control** for abstention: choose the smallest posterior threshold `λ` such that
   the upper confidence bound on selective risk at coverage ≥ 0.5 is ≤ 0.10, using the
   Learn-Then-Test bound on the calibration split. Report the achieved coverage on development.

**Acceptance.**

- ECE after temperature ≤ 0.08;
- selective risk at coverage 0.5 ≤ 0.25 on development;
- coverage at the 0.20 risk target ≥ 0.40. This is the Gate 2 lesson: a certificate at 11 % coverage
  is not a perception system. The acceptance fails if the only way to meet the risk target is to
  abstain on most scenes. The 0.20 target (not 0.10) is set because the task deliberately keeps a
  20-point defect/reflection ambiguity in a single look; a 10 % selective risk at 40 % coverage is
  not attainable from an 80 %-accurate base classifier, and pretending otherwise would force the
  system to abstain on almost everything — exactly the failure this gate exists to catch.

**Artifacts.** `results/stage_c.json`, `results/risk_coverage.json`, `results/calibration_reliability.json`.

## 5. Stage D — sequential evidence and belief revision

**Question.** Does the ledger-based belief revise correctly when a disambiguating observation arrives,
and does the dependence discount prevent overconfidence?

**Data.** Episodes of up to 3 observations, generated by a fixed **policy-independent** schedule for
fitting (base → light → zoom), so that the fitted pair likelihoods are not selected by the policy
being evaluated. 600 calibration episodes, 200 development episodes.

**Fits.** Pair-feature Gaussians per cause (persistence, bright_shift); conflict tolerance; debt
thresholds on calibration only.

**Comparisons.**

| Arm | Evidence pooling |
|---|---|
| `ledger` | per-record LLRs, dependence weights, pair evidence, quality weights |
| `pooled` | one averaged feature vector per episode, no dependence, no pair evidence |
| `no_pair` | ledger without pair evidence (ablation for H4) |
| `no_dep` | ledger with all weights = 1 (ablation for H3) |

**Metrics.** Final accuracy; revision accuracy (fraction of initially-wrong episodes corrected);
ECE; NLL; mean n_eff; contradiction rate; abstention rate.

**Acceptance.**

- ledger revision accuracy ≥ pooled + 0.05 (95 % CI excluding 0);
- ledger ECE ≤ pooled ECE (point estimate; CI reported);
- `no_dep` shows higher confidence on duplicate-heavy episodes (mean max-posterior at equal accuracy),
  demonstrating that the discount is doing something.

**Artifacts.** `results/stage_d.json`, `results/revision_traces.jsonl` (20 examples).

## 6. Stage E — adaptive observation

**Question.** At a fixed budget, does falsification-directed selection reduce decision error?

**Data.** 400 held-out test scenes, never used for any fit. Policy weights frozen before the run.

**Protocol.** For each scene and policy: reset the episode, run the policy for exactly B = 3
observations, commit the final MAP (no abstention in the primary analysis; abstention is a secondary
outcome). Scene-level pairing: every policy sees the same scene with the same seed.

**Primary endpoint.** Mean per-scene decision error; paired difference falsification − EIG, with a
95 % percentile bootstrap CI (10,000 resamples) and McNemar's exact test.

**Secondary.** Error vs confidence and vs random; cost-weighted error; abstention rate at the
calibrated threshold; coverage at risk target; calibration; mean n_eff; predicted-vs-realised score
correlation.

**Acceptance.** H1 confirmed if the primary CI's upper bound < 0 and Holm-corrected McNemar p < 0.05
across the three comparisons, with the point estimate ≤ −0.05.

**Artifacts.** `results/stage_e.json`, `results/episodes.jsonl`, `results/stage_e_report.md`.

## 7. Compute budget

| Stage | Hardware | Time (v1 scale) |
|---|---|---|
| A (handcrafted) | CPU | < 5 s |
| A (DINOv2, cached features) | RTX 4060, 8 GB | 3–6 min for 1,000 images at 224² |
| B | CPU | < 5 s |
| C | CPU | < 5 s |
| D | CPU | 30–60 s for 1,200 episodes × 3 observations |
| E | CPU | 60–120 s for 400 scenes × 4 policies |

The demo runs A–E at reduced N in about a minute on CPU. The full Gate E run is a few minutes. No
stage requires more than 4 GB of GPU memory; the DINOv2 path at batch 32 and 224² uses about 1.5 GB.

## 8. What is not trained

- the planner weights (frozen constants, sensitivity-checked);
- the dependence constants (designed, empirically checked);
- the temperature for the demo (fitted on 100 calibration scenes, stored in the model JSON);
- any component on test data. The test split is created by `seed >= 10000` and is never passed to a
  `fit` function; `verify` asserts that no fitted artifact's provenance names a test scene.

## 9. Reproducibility rules

- every fit writes its inputs' hashes and seeds to `results/*.json`;
- `python -m rhan_bi.cli verify` re-runs the split check and the hash checks;
- changing a feature definition bumps `feature_version`, which invalidates old models rather than
  silently mixing them;
- `results/` is append-only within a run; a rerun writes a new timestamped directory, never overwrites.
