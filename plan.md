## Yes this is different from proposed_plan.md.

# RHAN-NXA — PLAN.md

Version 3.0 · 2026-10-14 · Replaces Proposed_Plan.md (the Gen-2/Gen-3 master architecture plan).
Last verified against: the coding agent's training summaries through 2026-10-14 (all 104 epochs), its S7 evaluation report, and the earlier HF-log summaries. No log or repo file was re-read directly. Anything tagged [reported] must be re-checked before it carries a decision.

---

## 0. Core principles, and how to use this file

RHAN is an active recurrent perceptual system: it decides what to observe next, keeps a belief, predicts what it will see, and revises the belief from prediction error. Five testable commitments:

| # | Principle | The measurement that tests it | Claim |
|---|---|---|---|
| P1 | More observation gives a better belief | accuracy and robustness vs. number of glimpses at test time | CLAIM-2 |
| P2 | Uncertainty tracks error | ECE, NLL, error-detection AUROC of U_t vs. a softmax control | CLAIM-4 |
| P3 | Adaptive gaze beats non-adaptive gaze | AIS-v2 vs. fixed, random gaze at equal compute | CLAIM-5 |
| P4 | Prediction error improves belief | belief_with_f vs. belief_no_f; update-net v2 vs. v1 | CLAIM-3 |
| P5 | Robustness without a clean-accuracy collapse | clean and robust of the same checkpoint, labeled | CLAIM-1 |

**If you are lost:** §11 (symptoms) → §1 (state) → §5 (next action).

**Status words (only these):** REQUIRED, PENDING DECISION, EXPERIMENTAL CANDIDATE, DEFERRED, REJECTED.
**Evidence tags:** [measured], [reported] (an agent said so, not re-checked), [inferred], [hypothesis].

**Rules**
1. Every "implemented" or "working" claim cites a test or a log line.
2. One ID, one meaning. §10 is the only registry, and a script checks it for duplicates.
3. Every metric is labeled [CLEAN] or [ROBUST @ε].
4. A threshold not anchored to a measurement is PENDING DECISION.
5. A failed gate stops the run. Only the owner advances it.
6. No recipe or architecture edits to an in-flight run. Changes go in a new arm.
7. Report final-epoch checkpoints, never best-on-validation or "peak" columns.
8. Epoch-to-epoch robust moves under about 4 pp (n=512) are noise. Only finals and paired tests carry claims.
9. If a log disagrees with a document, the log wins until the code is read. Record the mismatch in §1.
10. Every edit is dated and updates the "Last verified" line.

---

## 1. STATE (edit every session)

**Headline [measured, agent-summarized].** The ladder and the S7 evaluation are finished. The best model so far is `recurrence_only` (63.44 / 34.77 at PGD-10). No later phase improved either axis, and phases 5–6 are worse on both (E10). S7 measured only the final model: 58.33 clean, 30.04 PGD-10, 28.60 PGD-50, and **21.00 under EOT-PGD, which is the honest robust number for a stochastic model** (E15). The loop's claimed benefits (P1–P4) have still not been measured. The eval-only checks in §5 measure them without training.

Run: Gen-2 foundation ladder on Kaggle and Colab T4. ImageNet-100 (126,689 train / 5,000 val). DINOv2-S warm start. TRADES at ε_norm 0.094 (≈5.4/255 pixel). Finished on 2026-10-14: 104 epochs logged (phase 6 stopped at epoch 14 of 15 [reported; cause not stated]) and the S7 evaluation of `foundation_gen1_core_best_clean.pth`.

| Phase | Epochs | Final [CLEAN] | Final [ROBUST @0.094, PGD-10, n=512] | Source quality |
|---|---|---|---|---|
| 1 g2_gist_only | 30/30 | 62.18 | 30.47 | JSONL |
| 2 recurrence_only | 15/15 | 63.44 | 34.77 | epochs 11–15 JSONL; epochs 2–8 approximate |
| 3 belief_no_f | 15/15 | 61.26 | 34.18 | JSONL |
| 4 belief_with_f | 15/15 | 60.86 | not in the summary (max 33.98, epoch 1) | summary |
| 5 ais_v2_swap | 15/15 | 58.90 | 28.71 | summary |
| 6 gen1_core | 14/15 | 58.46 | 29.69 | summary |

Gate W (frozen-trunk linear probe, 5 epochs): 58.52% [CLEAN], 174/174 keys matched [reported]. Gate C passed (epoch 10 clean 63.16 ≥ 58.52). Gate R passed (epoch 17 clean 64.82 ≥ 31.3). The agent labeled Gate R "ε=0.094", but epoch 17 trains at ε≈0.047.

**S7 evaluation of the phase-6 model [reported].** 8 eval seeds × 300 images, then 500 images for the stronger attacks.

| Condition | Accuracy |
|---|---|
| [CLEAN], 8×300 | 58.33 (σ across subsets 3.35) |
| [ROBUST] PGD-10 at ε_norm 0.031 / 0.062 / 0.094 | 48.33 / 39.13 / 30.04 |
| [ROBUST] PGD-10 at 2/255 / 4/255 pixel | 47.33 / 37.13 |
| [CLEAN], 500-image slice | 56.40 |
| [ROBUST] PGD-50, ε_norm 0.094, 500 images | 28.60 |
| [ROBUST] EOT-PGD (5 gaze samples, 20 steps), ε_norm 0.094, 500 images | **21.00** |

**Open blockers, ranked by what they stop**
- B1. The ladder regresses after phase 2 (E10), and the cause is unattributed: restart schedule, AIS-v2 swap, or the extra phase. Controls: G2-CTL4-fixed-continued, G2-REC1-inherit-lr.
- B2. The recurrence gain is not attributable. No matched-epoch or matched-view control exists (CLAIM-2). Controls: G2-CTL1-gist-continued, G2-CTL2-multiview-mean.
- B3. The refinement block may be inert (E6). Check: G2-D1-refinement-ablation.
- B4. Phase 6 (`gen1_core`) has no defined delta from phase 5, and it ended level with it (clean −0.44, robust +0.98). In Gen-1 the two were bit-identical. Compare the trainable-parameter sets and hash the weights. If nothing differs, it is a restart of phase 5 and should be dropped from future ladders.
- B5. Selection bias. "Best" checkpoints are chosen on the same val set used for reporting. Use finals, and carve a select split (G2-INF1-select-split).
- B6. Harness bugs: STALL fires during the ε ramp, Gate R is mislabeled and too loose, and best-robust may compare epochs at different ε.
- B7. One training seed.
- B8. The logged loss is 0.55·CE + 1.375·KL (28 of 28 rows, residual < 1e-4), not the documented 0.45·CE + 0.55·2.5·KL (E12). Check: G2-D6-loss-weight-audit.
- B9. Evidence quality. Phase 2 epochs 2–8 are approximate, phase 4's final robust is missing, and an earlier agent said training "has not started" while the HF logs show 104 epochs. G2-D4-log-audit settles it.
- B10. S7 covers one checkpoint. There is no PGD-50 or EOT number for phases 2–5, so "stochastic gaze is a defense" has no deterministic control. Check: G2-D8-adaptive-attack-finals.
- B11. The agent's S7 verdicts misread two things: EOT (a stronger attack, not a defense) and the seed σ (binomial subset noise, not model stability). See E15–E16.

---

## 2. Evidence log: what the run has told us

Noise floors: [CLEAN] n=5,000 → ±0.7 pp (1σ). [ROBUST] n=512 → ±2.0–2.1 pp (1σ). Unpaired differences carry σ ≈ 1.0 pp (clean) and ≈ 2.9 pp (robust). A single seed adds unmeasured variance on top.

- **E1 [measured] The warm start works.** Gate W 58.52% frozen. One epoch of fine-tuning reaches 64.00%. The 174-key count equals 148 + 24 LayerScale gammas + cls/pos (my arithmetic), consistent with LayerScale being loaded.
- **E2 [measured] The recipe survives the ramp.** At the top ε (epochs 25–30) clean is 62.2% and robust is 30.5%. Gen-1 from scratch gave 5.82 / 0.00. [inferred] Robust values during epochs 1–24 are not at a fixed ε (they rise as training ε approaches the evaluated ε and drop at stage changes). Do not compare them across epochs.
- **E3 [measured] Clean saturates immediately.** Best clean is 64.00% at epoch 1 and 64.82% at epoch 17. [inferred] The ceiling is the input: a 96-px source (verify) reduced to a 56-px, 16-token gist. Optimization is not the limit.
- **E4 [measured] Every phase with data dips when the learning rate peaks.** Trough vs. the first epoch: phase 1 −8.3 pp (64.00 → 55.68), phase 4 −2.3 (58.72 → 56.46), phase 5 −6.4 (58.32 → 51.88), phase 6 −4.0 (56.60 → 52.56). [inferred] A 5e-5 trunk peak is too high for an already-adapted trunk. That default was a choice, not a decision.
- **E5 [measured] Overfitting signature in phases 1–3.** Train CE falls 0.70 → 0.46 → 0.30 across phases 1–3, while val clean stays 61–64% and robust stays 34–35%. Validation CE is not logged, so the gap is not directly measured. Phases 5–6 do not show it (E11).
- **E6 [measured/qualified] Recurrence gain.** Robust 30.47 → 34.77 (+4.3 pp, ≈2σ unpaired), clean +1.26 pp. Not attributable: phase 2 had 15 extra epochs at full ε, and it sees 4 views plus the gist instead of 1. The reported 255% drift came from a LayerScale tensor with initial norm 1.96e-4 = √384 × 1e-5, which is exactly the from-scratch init. A new block starts there. [hypothesis] The refinement block's gammas are still near 1e-5, so the block is functionally inert.
- **E7 [measured] The evidential head costs clean accuracy.** Phase 3 vs. phase 2: clean −2.18 pp (about 2σ unpaired), robust −0.6 pp. Much smaller than the −9.9 pp belief_no_f drop in the Gen-1 ImageNet-100 ladder. Confounded with the belief loop itself.
- **E8 [measured] Phase 4 recovered.** CE spiked to 5.16 at the start (new modules not identity-initialized) and clean fell to 56.46 at epoch 3, yet it ended at 60.86 with the lowest train loss so far (0.5084). Clean is −0.40 pp vs. its parent, inside the noise. Robust at epoch 15 is not in the summary.
- **E9 [measured] Anchors from Gen-0/Gen-1.** Gen-1 results before the autocast fix were effectively clean-trained. AIS-v2 r = 0.706 came from an STL-10-era smoke checkpoint.
- **E10 [measured] After phase 2 every phase ended below its parent on clean, and phases 5–6 also lost robustness.**

  | Phase | Final [CLEAN] | Δ vs. parent | Final [ROBUST] | Final train loss |
  |---|---|---|---|---|
  | 1 g2_gist_only | 62.18 | — | 30.47 | 0.9020 |
  | 2 recurrence_only | 63.44 | +1.26 | 34.77 | 0.6661 |
  | 3 belief_no_f | 61.26 | −2.18 | 34.18 | 0.5508 |
  | 4 belief_with_f | 60.86 | −0.40 | not reported | 0.5084 |
  | 5 ais_v2_swap | 58.90 | −1.96 | 28.71 | 0.6612 |
  | 6 gen1_core (ep 14) | 58.46 | −0.44 | 29.69 | 0.6428 |

  Cumulative clean change from phase 2 to phase 6 is −4.98 pp. Robust never exceeded 29.69 in any of the 29 logged epochs of phases 5–6, against 34.2–34.8 at the end of phases 2–3. Phase 6 ended level with phase 5, so its 14 epochs bought nothing measurable. The 5.5 pp robust gap is only 1.9σ unpaired, but it persists across epochs and the clean gap points the same way. Treat it as probably real until a paired eval says otherwise.
  [inferred] Phases 3 and 4 hit the same LR-peak trough and recovered, so the restart schedule alone does not explain phases 5–6. Phase 5's total loss was 0.6667 at epoch 1 (0.16 above phase 4's final, while the LR was still at most a third of its peak), rose to 1.1467 at epoch 6, and ended at 0.6612: 15 epochs bought nothing. Phase 6 repeats the curve (0.7467 → 1.0935 → 0.6428 at epoch 14).
  [hypothesis] AIS-v2's candidate selection hands the belief loop a glimpse distribution that differs from the fixed grid it was trained on, and 15 epochs at this LR do not re-fit it. Test: G2-CTL4-fixed-continued.
- **E11 [measured] Phases 5–6 are under-fit, not over-fit.** Final train loss is 0.64–0.66 against 0.51–0.55 in phases 3–4, and val is lower as well. Across the LR-peak excursion, CE moves 0.37 → 1.12 → 0.37 while KL stays between 0.33 and 0.39. [inferred] The excursion damages the classification path, not the smoothness term.
- **E12 [measured arithmetic] The logged loss does not match the documented recipe.** On all 28 rows available (phase 5, and phase 6 through epoch 13), total = 0.5500·CE + 1.3750·KL (least-squares fit, max residual 9e-5). The documented recipe is 0.45·CE + 0.55·2.5·KL, whose residual reaches 0.11. Either the code weights CE by 0.55 (for example `w·(CE + β·KL)`), or the logged "CE" is not the term the 0.45 multiplies. The KL:CE weight ratio is 2.5 in the log and 3.06 in the document. Resolve by reading the loss line (G2-D6-loss-weight-audit).
- **E13 [measured] The agent's narrative over-reads noise.** Its "Dynamics / Evolution" column reports single-epoch robust moves ("jumps +2.73%", "breaks 28%"). At n=512 the 1σ noise is about 2 pp. The "peak" columns are maxima over noisy epochs and are biased upward. Use finals.
- **E14 [reported] S7 on the phase-6 model.** Clean 58.33 (8×300) and 58.46 (full val, epoch 14) agree. PGD-10 accuracy falls smoothly with ε: 58.33 → 48.33 → 39.13 → 30.04 at ε_norm 0, 0.031, 0.062, 0.094, and the pixel-space points (47.33 at 2/255, 37.13 at 4/255) sit where the norm-space curve predicts. PGD-50 (28.60, n=500) is level with PGD-10 (30.04) inside the noise (σ_diff ≈ 2.2 pp), and the retention ratios match (50.7% vs. 51.5% of clean). [inferred] This rules out gross masking of the deterministic gradient. It is a weak test: it does not replace AutoAttack, or a large-ε check where accuracy must fall to about 0.
- **E15 [measured arithmetic] EOT is the real robust number, and it exposes a 7.6 pp gap.** The EOT-PGD attack (5 gaze samples, 20 steps) takes accuracy from 28.60 to 21.00 on the same 500 images (37.2% of the 56.40 clean). The non-adaptive PGD-50 over-estimated robustness by 7.6 pp (≈2.8σ unpaired), so part of the PGD-10/50 number came from the gaze randomness, not from the weights. The agent called 21.00 a "resilient active defense". By the standard rule for stochastic defenses it is the opposite: the headline robust accuracy of this model is at most 21.00, because 5 samples and 20 steps is a modest attack and a stronger one can only go lower. A stochastic gaze counts as a defense only if its EOT accuracy beats a deterministic control's PGD-50 accuracy. The deterministic phase 2 and phase 3 finals scored 34.77 and 34.18 at PGD-10 (n=512), with no randomness to average out. Their PGD-50 and EOT numbers have not been measured. [hypothesis] They retain more of their clean accuracy than phase 6 does under EOT (54.8% for phase 2 at PGD-10 vs. 37.2% for phase 6 under EOT). "EOT" here means Expectation Over Transformation, not "over time".
- **E16 [measured arithmetic] The "seed" spread is subset noise.** Across 8 eval seeds the robust σ is 2.89 pp, and the binomial σ of a 300-image subset at 30% is 2.65 pp. The clean σ is 3.35 against 2.85, and the other conditions are 2.8–3.7 against 2.7–2.9. The spread measures how noisy a 300-image subset is. It says nothing about model stability. All 8 are eval seeds on one trained model, so there is still one training seed. The standard error of an 8-seed mean is about 1.0 pp, and a single 300-image result has a ±6 pp (2σ) band.
- **E17 [reported] Phase 6 stopped at epoch 14 of 15.** The cause is not stated. S7 evaluated `best_clean`. The clean peak (58.46) is at epoch 14, so best equals final and B5 does not bite here.

---

## 3. Claims and decision rules

- **CLAIM-1 (recipe).** Warm start, clean warmup, and TRADES ramp give a model that is neither collapsed nor non-robust. Status: SUPPORTED at phase 1 [measured]. Future gate thresholds are PENDING DECISION; anchors are warmup-end clean 63.16 and Gate W 58.52.
- **CLAIM-2 (the loop beats its null).** Recurrence/belief/gaze beats the gist-only null at matched epochs and matched views. Status: NOT ESTABLISHED. The only gain (phase 2) is confounded (E6), and no later phase improved on it (E10). Requires G2-CTL1-gist-continued and G2-CTL2-multiview-mean. Test: paired, 8 eval seeds, |Δ| > 2σ_combined.
- **CLAIM-3 (belief dynamics help).** belief_with_f beats belief_no_f; update-net v2 beats v1 from the same parent and seed. Status: NOT SUPPORTED YET. Clean is 60.86 vs. 61.26 (−0.40, inside the noise), and phase 4's robust is missing.
- **CLAIM-4 (uncertainty is useful).** The Dirichlet head beats a softmax control on ECE, NLL, and error-detection AUROC. Status: UNTESTED on these checkpoints. G2-CAL2-evidential-gate is eval-only.
- **CLAIM-5 (active gaze helps).** AIS-v2 beats fixed and random gaze at equal T. Status: NOT SUPPORTED so far. ais_v2_swap ended below its parent on both axes (E10), uncontrolled. Decisive evidence: G2-D7-gaze-swap-eval (eval-only) and G2-CTL4-fixed-continued.
- **CLAIM-6 (robustness holds under the strongest adaptive attack).** The headline robust accuracy is the minimum over PGD-50, AutoAttack, and EOT-PGD with at least 20 samples. Status: PARTIAL. PGD-50 (28.60) and EOT-5 (21.00) are measured for phase 6 only (E15). Gate A (§5) applies.
- **CLAIM-7 (the loop earns its cost).** At an equal observation budget (tokens or FLOPs), the loop beats a feed-forward model on a task where the budget binds. Status: UNTESTED. Whether Gen-2 glimpses carry detail that the 56-px gist lacks is unverified, so the 56-px setup may be unable to test it. Tests: G2-D5-glimpse-scaling now, G3-BUD1-budget-curve later.

---

## 4. Reference configuration (as reported; correct it if the code differs)

- Data: ImageNet-100, light augmentation (RandomResizedCrop 0.35–1, hflip), no Mixup/CutMix.
- Trunk: DINOv2-S/14 with LayerScale, antialiased bicubic pos-embed interpolation to the 4×4 grid. Gist = full image resized to 56×56 (16 tokens).
- Optimizer: AdamW, weight decay 0.05, trunk peak LR 5e-5, per-group clip 1.0, fp16, batch 48. Schedule: 3-epoch warmup, then cosine to 0. **Every phase restarts this schedule from zero (no LR carry-over).**
- Loss, documented: 0.45·CE + 0.55·β·KL, β = 2.5. Loss, as logged: 0.55·CE + 1.375·KL (E12). Until G2-D6-loss-weight-audit settles it, quote the logged form. PGD-4 train, PGD-10 validation, ε_norm 0.094.
- Curriculum: 10 clean epochs; ramp over epochs 11–24 in two linear stages (0.0157 → 0.0835); hold 0.094 from epoch 25. Later phases: 15 epochs at full ε from epoch 1, inheriting the previous phase's final weights.
- Deviation from the earlier three-rung prompt: the ramp is smooth, not three rungs. Gate R at epoch 17 therefore evaluates ε ≈ 0.047, not 0.094.
- Seed 41. Sessions save and exit at 11 h. HF repos: rhan-nxa-g2-checkpoints (best) and rhan-nxa-g2-checkpoints-rolling (dataset repos). "Best" is chosen on val, so do not report from it (B5).

---

## 5. Roadmap

**W1: the ladder and S7 are done (2026-10-14).**
- Freeze all six finals: copy, sha256, never train on them again. They are data.
- S7 evaluated phase 6 only. The comparison that matters (all six finals, paired, with PGD-50 and EOT) is G2-D3-final-eval plus G2-D8-adaptive-attack-finals.
- Do not promote phase 6 to "the Gen-2 model" by default. By E10 the best checkpoint so far is `recurrence_only`, and by E15 phase 6's honest robust number is at most 21.

**W1.5: eval-only checks (no training, one agent prompt, the first thing to run).** They test P1–P4 on checkpoints that already exist.
Order by cost: G2-D6-loss-weight-audit (code read), G2-D4-log-audit, G2-D1-refinement-ablation, G2-D3-final-eval, G2-D8-adaptive-attack-finals, G2-D5-glimpse-scaling, G2-D7-gaze-swap-eval, G2-CAL2-evidential-gate, G2-AIS5-gate-rerun, G2-D2-feature-drift, G2-RES1-probe-112. Infrastructure for later runs: G2-INF1-select-split.

**W1.6: decisive controls (training, 15 epochs each, about 10 GPU-hours each [estimate]; run two at a time on Kaggle and Colab).**
1. G2-CTL4-fixed-continued: belief_with_f final, same protocol, fixed grid, no new modules. If it holds at ≥ 60.0 clean with robust within 2σ of its parent, AIS-v2 is what regressed (CLAIM-5 fails as built). If it also falls to about 59 / 29, the restart protocol is the cause and G2-REC1-inherit-lr becomes the priority. Thresholds are PENDING DECISION, anchored to the parent's 60.86 and the 0.7 pp clean noise.
2. G2-CTL1-gist-continued and G2-CTL2-multiview-mean: the two CLAIM-2 nulls. Add G2-CTL3-recurrence-continued only if CLAIM-3 is later disputed by epoch effects.
3. G2-REC1-inherit-lr: the same as CTL4 with a lower trunk peak LR and layer-wise decay. Compare paired against CTL4.

After W1.5 and W1.6 the decision is one of three:
- (a) The loop shows value on a non-accuracy axis (D5, D7, CAL2): keep building on `recurrence_only` or `belief_no_f` with the REC1 protocol.
- (b) It shows none: report Gen-2 as `recurrence_only` (plus the RES1 result) and treat the loop claims as Gen-3 hypotheses.
- (c) RES1 shows a large resolution gain: W3 comes first.

**Gates before any Gen-3 build.** Thresholds are PENDING DECISION, anchored to the noise floors in §2.
- Gate L (loop value, eval-only): accuracy at T=4 beats T=1 by more than 2σ (paired) on a checkpoint with the belief loop, and does not fall as T rises to 8. Source: G2-D5-glimpse-scaling.
- Gate A (adaptive robustness): the headline robust number comes from the strongest adaptive attack (§9). Stochastic gaze counts as a defense only if its EOT accuracy is at least a deterministic control's PGD-50 accuracy. Source: G2-D8-adaptive-attack-finals.
- Gate B (budget) and Gate S (search): defined in §8. They decide whether the Gen-3 build is worth starting.

**W2: one-flag arms, at most 15 epochs each, one at a time.** Parent = the latest phase final that is not worse than the best earlier final by 2σ (paired). Provisionally that is `recurrence_only`.
Recipe: G2-REC2-weight-ema, G2-REC3-identity-init.
Belief: G2-BEL1-updatenet-v1 (control), G2-BEL2-updatenet-v2, G2-BEL3-target-ema.
Precision: G2-PRE1-constant, G2-PRE2-one-minus-u, G2-PRE3-learned-scalar, G2-PRE4-spatial-map.
Gaze: G2-AIS1-fixed, G2-AIS2-random, G2-AIS3-logit-only, G2-AIS4-mlp.
Calibration: G2-CAL1-softmax-control.
Promote an arm only if it wins a paired 8-seed comparison and trips no health flag.

**W3: resolution (conditional).** G2-RES2-rebuild-112 only if G2-RES1-probe-112 adopts. This is a new substrate, so the ladder re-runs; call it Gen-2.1. Design it with the lessons of E10: identity-init at each boundary, a lower inherited LR, and no phase without a defined delta.

**W4: final evaluation** on one finished model: G2-EVAL1-pgd50-aa, G2-EVAL2-eot, G2-EVAL3-shape-texture, G2-EVAL4-occlusion-clutter, G2-EVAL5-human-mini.

**W5: backlog (after W2).** Memory (G2-MEM1-glimpse-memory, G2-MEM2-spatial-bank-ior), factorized predictive belief (G2-JEP1-monolithic … G2-JEP6-factor-disagreement-gaze), G2-HALT1-adaptive-stop, G2-BEL4-bound-adaptive, G2-STR1-minimal-slots.

**W6: vision.** Gen-3 and Gen-4 (§8).

---

## 6. Mechanism ledger

One status word per row. "Next ID" is the experiment that would change the status.

| Mechanism | Status | Evidence | Next ID |
|---|---|---|---|
| DINOv2-S warm start with LayerScale and antialiased pos interpolation | REQUIRED | E1 | G2-D1-refinement-ablation (γ stats) |
| Clean warmup, then TRADES ramp | REQUIRED | E2 | none |
| Full-image gist input (56 px, 16 tokens) | REQUIRED | the null for resolution; E3 | G2-RES1-probe-112 |
| Higher-resolution gist (112 px, 64 tokens) | EXPERIMENTAL CANDIDATE | highest-leverage hypothesis; E3 [inferred] | G2-RES1-probe-112 |
| Per-phase LR restart (warmup to 5e-5, cosine to 0) | PENDING DECISION | E4, E10; likely to be replaced | G2-REC1-inherit-lr |
| Full-model weight EMA | EXPERIMENTAL CANDIDATE | E5; a standard remedy for robust overfitting | G2-REC2-weight-ema |
| Identity-init of new modules at phase boundaries | PENDING DECISION | E8 recovered without it; E10 | G2-REC3-identity-init |
| Select split carved from train | REQUIRED | B5; for new runs | G2-INF1-select-split |
| Tied refinement block ("recurrence") | PENDING DECISION | E6; phase 2 is still the best model | G2-D1-refinement-ablation, G2-CTL1-gist-continued, G2-CTL2-multiview-mean |
| Fixed 4-crop gaze pattern | REQUIRED | the control, and phase 5's parent (E10) | G2-CTL4-fixed-continued, G2-AIS1-fixed |
| Test-time glimpse scaling | PENDING DECISION | the P1 test; not yet measured | G2-D5-glimpse-scaling |
| Dirichlet head (U_t) | PENDING DECISION | E7 | G2-CAL1-softmax-control, G2-CAL2-evidential-gate |
| UpdateNet v1 (mean-pooled error) | REQUIRED | the control; phase 4 is the v1 run (E8) | G2-BEL1-updatenet-v1 |
| UpdateNet v2 (observed-feature fusion + spatial error) | EXPERIMENTAL CANDIDATE | not yet run | G2-BEL2-updatenet-v2 |
| Predictor target EMA | EXPERIMENTAL CANDIDATE | not yet run | G2-BEL3-target-ema |
| Precision: constant, 1−U, learned scalar, spatial map | PENDING DECISION | not yet run | G2-PRE1-constant … G2-PRE4-spatial-map |
| AIS-v2 mechanism (K=4 candidates) | PENDING DECISION | built and tested for mechanism; benefit not supported (E10); r = 0.706 came from an STL-10-era smoke checkpoint | G2-CTL4-fixed-continued, G2-D7-gaze-swap-eval, G2-AIS5-gate-rerun |
| Stochastic gaze as a defense | PENDING DECISION | E15: EOT 21.00 vs. PGD-50 28.60 for phase 6; no deterministic control yet | G2-D8-adaptive-attack-finals |
| MLP, then recurrent/transformer policies | EXPERIMENTAL CANDIDATE | the current policy has one learnable scalar | G2-AIS4-mlp |
| Inhibition of return (fixation memory) | EXPERIMENTAL CANDIDATE | human-like; cheap | G2-MEM2-spatial-bank-ior |
| Glimpse memory | EXPERIMENTAL CANDIDATE | none | G2-MEM1-glimpse-memory |
| Adaptive stopping | DEFERRED | until D5 shows monotone gains | G2-HALT1-adaptive-stop |
| Belief-bound (Δz ≤ 0.1) variants | DEFERRED | none | G2-BEL4-bound-adaptive |
| Phase 6 `gen1_core` as a separate phase | DEFERRED | E10, B4; define its delta or drop it | none |
| L_stab | DEFERRED | diagnostic-only today; a training objective needs a responsiveness guard | none |
| Factorized predictive belief (JEPA-Anything-inspired) | DEFERRED | until W2; OPF verified at abstract level only | G2-JEP1-monolithic |
| Slots (S_t) | DEFERRED | the legacy 16-slot version is REJECTED (Gen-0) | G2-STR1-minimal-slots |
| Pixel reconstruction as a primary objective | REJECTED | mechanism: precision dilution; stats: −0.90 pp, not significant | none |
| Multi-branch input pyramid | REJECTED | for Gen-2; tokenizer confound | none |
| Search over a 9-term λ family | REJECTED | loss competition | none |
| V1 front-end | DEFERRED | needs a standalone ablation | none |
| External retrieval / RAG memory | REJECTED | no motivating evidence | none |

---

## 7. Memory

"Memory" means six different things here. Only the first two exist.

| Level | Meaning | Status | ID |
|---|---|---|---|
| M0 | Weights θ | exists | none |
| M1 | Within-image working state (the B_t trajectory) | exists | none |
| M2 | Glimpse memory: a bounded buffer of the last T glimpses (tokens, location, prediction error), read by the update net through small cross-attention | EXPERIMENTAL CANDIDATE | G2-MEM1-glimpse-memory |
| M3 | Spatial bank: a coarse grid (≈6×6 cells) of the latest evidence per region plus a visit timestamp. The gaze policy reads it for inhibition of return; the readout reads it for coverage | EXPERIMENTAL CANDIDATE | G2-MEM2-spatial-bank-ior |
| M4 | Cross-image episodic and semantic memory (persistent scene/object state, retrieval). Needs sequential data (the UCF-101/TDV pipeline), not shuffled ImageNet-100 | DEFERRED | G3-MEM3-episodic |
| — | External retrieval / RAG | REJECTED | none |

Build rules for every memory module:
- Tensor-in, tensor-out, attached to the belief state.
- A disable flag that reproduces the reference forward exactly.
- Its own optimizer group, a hard capacity cap, and decay.
- Reset per image, with a test that fails if state leaks across samples.
- Tests: gradient reaches its parameters; a synthetic overwrite test; a parameter-matched control (same parameters, memory read zeroed).
- Metrics: re-fixation rate, image-area coverage, accuracy and robustness vs. T, occupancy.

Why inhibition of return first: it is documented in human visual search, it is cheap, and it gives CLAIM-5 a concrete test (does the policy stop re-visiting the same place?). It also sidesteps the E10 problem, because it changes which glimpse is chosen without changing the glimpse distribution the belief loop sees at the first step.

---

## 8. Beyond Gen-2

### 8.1 What would make this matter

RHAN's claim is about evidence acquisition: a perceiver that keeps a belief, knows what it does not know, chooses what to look at next, and stops when it has enough. ImageNet-100 top-1 cannot show that, and it cannot be won on this budget, because robust-accuracy leaderboards are 224-px models trained at a scale a T4 cannot reach. Three outcomes would count, in rising order of strength:

1. **A sharp, well-controlled result, positive or negative.** E15 is already one: a stochastic gaze looks robust to PGD and loses 7.6 pp to EOT. This is a graduation-level contribution, and the harness is built for it.
2. **An evidence-seeking advantage at an equal budget, in a regime where one feed-forward pass cannot see the answer.** That means high-resolution scenes with small targets. Gaze can be scored against known target locations, so no eye-tracking is needed.
3. **A small interactive demo (Gen-4)** in which the model asks to look again, and answers more accurately and with calibrated uncertainty than a raw VLM of the same size.

The area is active and crowded. Guided visual search for multimodal LLMs on high-resolution images started with [V* (CVPR 2024)](https://arxiv.org/abs/2312.14135), and zoom/crop agents have multiplied since, for example [multi-turn grounding RL for high-resolution reasoning](https://arxiv.org/abs/2507.05920) and [CVSearch](https://arxiv.org/abs/2605.23655). The zoom idea is not new. What could be new here is the belief state, with prediction-error and uncertainty-driven gaze, measured under controlled budgets and adaptive attack. Do not build Gen-3 or Gen-4 on a loop that has not shown value (Gate L), or any gain will be unattributable.

### 8.2 Substrate and gates for Gen-3

- **Substrate (G3-HRV1-high-res-substrate).** Glimpses are native-resolution crops of a high-resolution source, and the gist is a low-resolution global view. Without this, glimpses cannot add information and the loop cannot beat a feed-forward pass at equal budget. First verify whether Gen-2 glimpses add detail beyond the 56-px gist (G2-RES1-probe-112, G2-D5-glimpse-scaling).
- **Gate B (budget), G3-BUD1-budget-curve.** Accuracy vs. tokens (or FLOPs) for (a) adaptive gaze, (b) fixed-grid gaze, (c) random gaze, and (d) one feed-forward pass at matched tokens. Pass: (a) beats (d) by more than 2σ at the budget where (d) is weakest, and beats (b) and (c) in paired tests.
- **Gate S (search), G3-VS1-synthetic-search.** Targets sit at known locations in cluttered 384–512 px scenes (ImageNet-100 objects pasted on backgrounds, with the target defined by a cue). Gaze-to-evidence alignment is measured against ground truth. Pass: alignment and accuracy beat fixed and random gaze at equal observations, and accuracy rises with observations. G3-VS2-counting adds counting small objects without double-counting, where inhibition of return should matter.

### 8.3 Reasoning ladder: what "reasoning perception" means here

Each row is an ability with an operational test. A row without a test is not a claim.

| Ability | Operational test | Gen | ID |
|---|---|---|---|
| Evidence seeking | gaze-to-evidence alignment; accuracy vs. observations | 3 | G3-VS1-synthetic-search |
| Counting without double-counting | count error vs. glimpses; effect of inhibition of return | 3 | G3-VS2-counting |
| Metacognition ("not sure yet") | calibration; abstention and stop/continue curves | 2–3 | G2-CAL2-evidential-gate, G2-HALT1-adaptive-stop |
| Counterfactual ("what would I see there?") | predictor error at unvisited locations, scored after actually looking | 3 | G3-CF1-counterfactual |
| Object persistence | identity through occlusion in sequences | 3 | G3-MEM3-episodic |
| Compositional questions | relation queries over W_t | 4 | G4-SLM1-belief-readout |

### 8.4 Gen-3: persistent world state, memory, TDV

- **W_t** is one object that perception updates, memory persists, prediction projects, and reasoning reads. Contents: entities, locations, attributes, relations, unresolved hypotheses, next-best-observation.
- **Build order**, each its own arm against the previous: G3-WS1-world-state, G3-OBJ1-minimal-objects (only if the slot rule below passes), G3-MEM3-episodic, G3-CF1-counterfactual, G3-RSN1-reasoning. Memory comes in the §7 order: M2 and M3 first (G2-MEM1-glimpse-memory, G2-MEM2-spatial-bank-ior), then M4.
- **TDV (G3-TDV1-temporal-pipeline)** is your sequential-video pipeline (the UCF-101/TDV work). M4 and object persistence need it. Start it only after Gate S, so that sequences are used by a loop that already shows value on single scenes.
- **Slot re-entry rule (from Gen-0).** Try 2–4 slots only. Per-slot decodability must be significantly above chance and distinct across slots. Zeroing the largest slot must reduce accuracy. If "improves when zeroed" reappears, that is the third confirmation of rejection.
- **Factorized predictive belief (JEPA-Anything, [arXiv 2609.20800](https://arxiv.org/abs/2609.20800)).** A reference mechanism, not a blueprint. Verified from the abstract only: orthogonal predictive factorization splits latent targets into complementary factors, learns them through dedicated pathways, and recombines them. The EMA target and the representation-vs-operational-mode claims are second-hand. Arms are G2-JEP1-monolithic (control) through G2-JEP6-factor-disagreement-gaze, all parameter- and FLOP-matched. Orthogonality is a tested mechanism, not an assumption, and factors get no semantic labels (probe them afterwards).

### 8.5 Gen-4: a small language model, and action

An SLM adds an interface and a place to learn when to ask. It does not add perception. Whether it adds reasoning is a question the experiments answer, with these controls:

- **G4-API1-perceptual-state-api.** The SLM consumes W_t as tokens, never pixels: top-k classes with probabilities, U_t, the coverage map, candidate regions, the latest prediction errors.
- **G4-SLM1-belief-readout.** RHAN frozen. The SLM answers questions from API tokens only. Controls: a plain classifier head on the same belief, and the SLM with the API tokens shuffled.
- **G4-SLM2-perceptual-actions.** The SLM emits LOOK, ZOOM, REOBSERVE, STOP, first by imitating the AIS policy or oracle gaze, with RL later.
- **G4-VLM1-raw-vs-rhan.** The same SLM with a frozen vision encoder and no loop (Raw-VLM) against RHAN-VLM, on questions that need evidence seeking. Measure accuracy, observations used, gaze-to-evidence alignment, and calibration.
- **G4-VLA1-physical-action** comes last.

Use a ≤ 0.5B open SLM (check current options before choosing; fine-tune with LoRA, which fits a T4). A language model that only restates the classifier's top-1 is not a result.

---

## 9. Evaluation protocol

- **Splits:** train / select (carved from train, class-stratified, 5,000 images) / val (reporting only).
- **Monitoring:** clean on the full select split. Robust PGD-10 on a fixed 512-image, seed-pinned select subset at a printed, fixed ε. Log validation CE.
- **Reporting:** final-epoch checkpoints. Full 5,000-image clean pass. For robust numbers, use at least 1,000 images where the attack allows. Paired tests, |Δ| > 2σ_combined. Print the binomial σ next to any "seed σ": when they match, the spread is subset noise, not stability. One training seed for exploration, at least 3 for any headline claim (cost PENDING).
- **Attacks:** PGD-10 is for monitoring only. A headline robust number is the minimum over PGD-50, AutoAttack (APGD-CE, APGD-T, FAB-T, Square) for deterministic models, and EOT-PGD with at least 20 samples and 50 steps for stochastic ones. Also run a large-ε sanity check, because accuracy must fall to near 0 at large ε (for example 16/255 and above) or the attack is suspect. EOT means Expectation Over Transformation. Every robust claim cites attack, samples, steps, restarts, and n. Report norm-space ε and pixel-space ε (ε_norm × 0.225 × 255: 0.094 ≈ 5.4/255; also 2/255 and 4/255).
- **Calibration (Gate U):** ECE, NLL, error-detection AUROC for U_t on held-out data. Thresholds PENDING, anchored to the softmax control.
- **Human-like:** shape/texture on Geirhos cue-conflict (human 95.9% shape decisions, source checked earlier), occlusion, clutter and peripheral degradation, and a 2AFC mini-study with d′. Gaze alignment against humans needs eye-tracking data (Tier 2). Gaze alignment against ground-truth target locations (Gate S) does not.
- **Label rule:** no unlabeled "val_acc" anywhere.

---

## 10. Experiment registry (the only source of IDs)

Waves: 1.5 eval-only and infrastructure; 1.6 decisive controls; 2 one-flag arms; 3 resolution; 4 final evaluation; 5 backlog; 6 Gen-3 and Gen-4.

| ID | Wave | One change vs. reference | Decision rule |
|---|---|---|---|
| G2-D1-refinement-ablation | 1.5 | Zero the refinement block at eval; print mean abs(γ) of the refinement and trunk LayerScale per phase final | Accuracy unchanged and mean abs(γ) < 1e-3 → block inert; fix before any recurrence claim |
| G2-D2-feature-drift | 1.5 | Cosine similarity of trunk features vs. DINOv2 on a fixed 256-image batch, per phase final | Informs the trunk-LR arm |
| G2-D3-final-eval | 1.5 | All six finals on identical images: full-val clean; 8×300 PGD-10 at ε_norm {0.031, 0.062, 0.094} and pixel {2/255, 4/255} | The table for CLAIM-1/2/3; paired, 2σ |
| G2-D4-log-audit | 1.5 | JSONL session ids and timestamps vs. agent reports; list epochs missing JSONL lines; recover phase 4's final robust; find why phase 6 stopped at 14 | Resolves B9 |
| G2-D5-glimpse-scaling | 1.5 | Test-time T ∈ {1, 2, 4, 6, 8} on the phase 2–6 finals | Gate L; gates G2-HALT1-adaptive-stop |
| G2-D6-loss-weight-audit | 1.5 | Read the total-loss line in the recipe code; print the effective CE and KL weights from one logged batch | Resolves B8; §4 then quotes one form |
| G2-D7-gaze-swap-eval | 1.5 | At test time only, run the phase 5 and 6 finals with AIS-v2, fixed-grid, and random gaze on identical images | If fixed ≥ AIS-v2 within 2σ, the policy adds nothing at test time (CLAIM-5) |
| G2-D8-adaptive-attack-finals | 1.5 | PGD-50 (500 images) on all six finals; EOT-PGD (N=20, 50 steps) on the stochastic phase 5–6 finals; a large-ε check | Gate A; obfuscation gap = PGD-50 − EOT |
| G2-RES1-probe-112 | 1.5 | Frozen-trunk linear probe at a 112-px gist (Gate W protocol) | Adopt if ≥ 58.52 + 5 pp (threshold PENDING) |
| G2-INF1-select-split | 1.5 | Carve the 5,000-image select split | All new gates use it |
| G2-CAL2-evidential-gate | 1.5 | Gate U on the trained Dirichlet head of the finals, eval-only | Until G2-CAL1-softmax-control exists, compare U_t with the softmax max-prob of the same logits |
| G2-AIS5-gate-rerun | 1.5 | Candidate-preference r and center-bias histogram on the phase 5–6 finals | Replaces the STL-10-era r = 0.706 |
| G2-CTL4-fixed-continued | 1.6 | belief_with_f final continued 15 epochs, same protocol, fixed grid, no new modules | Paired vs. phase 5: separates AIS-v2 from the restart protocol |
| G2-CTL1-gist-continued | 1.6 | Gist-only continued 15 epochs at full ε | CLAIM-2 null (matched epochs) |
| G2-CTL2-multiview-mean | 1.6 | 4 crops + gist, mean-pooled, no refinement or belief, 15 epochs | CLAIM-2 null (matched views) |
| G2-REC1-inherit-lr | 1.6 | G2-CTL4-fixed-continued with a lower trunk peak LR and layer-wise decay | Dip shrinks, and final clean and robust are not worse than CTL4 by 2σ |
| G2-CTL3-recurrence-continued | 2 | recurrence_only continued 15 epochs | Only if CLAIM-3 is disputed by epoch effects |
| G2-REC2-weight-ema | 2 | EMA (0.999) of all weights, used for eval | Robust-vs-final gap |
| G2-REC3-identity-init | 2 | Zero-init last layer of new update/readout modules | Phase-start clean within 0.7 pp of the parent |
| G2-BEL1-updatenet-v1 | 2 | Control: mean-pooled error, no EMA | Baseline for CLAIM-3; phase 4 is a v1 run, so re-run only if the protocol differs |
| G2-BEL2-updatenet-v2 | 2 | Observed-feature fusion plus spatial error | Paired vs. BEL1, same parent and seed |
| G2-BEL3-target-ema | 2 | EMA target encoder for the predictor | Paired vs. reference |
| G2-BEL4-bound-adaptive | 5 | Learned or adaptive update bound | Only after BEL1–3 |
| G2-PRE1-constant | 2 | Precision = constant | Paired vs. PRE3 |
| G2-PRE2-one-minus-u | 2 | Precision = 1 − U (legacy) | Paired vs. PRE3 |
| G2-PRE3-learned-scalar | 2 | Precision = learned scalar (current) | Reference |
| G2-PRE4-spatial-map | 2 | Precision = learned spatial map | Paired vs. PRE3 |
| G2-AIS1-fixed | 2 | Fixed gaze pattern | Control for CLAIM-5; reuse CTL4's gaze |
| G2-AIS2-random | 2 | Random gaze locations | Control for CLAIM-5 |
| G2-AIS3-logit-only | 2 | AIS-v2 with the single learnable scalar | Reference policy |
| G2-AIS4-mlp | 2 | MLP policy | Paired vs. AIS3 |
| G2-CAL1-softmax-control | 2 | Plain softmax head at belief_no_f | Baseline for CLAIM-4 |
| G2-RES2-rebuild-112 | 3 | Rebuild the ladder at a 112-px gist | Only if G2-RES1-probe-112 adopts; apply the E10 lessons |
| G2-EVAL1-pgd50-aa | 4 | AutoAttack (APGD-CE, APGD-T, FAB-T, Square) on 500+ images, final model | Final model only; PGD-50 is already measured for phase 6 (28.60) |
| G2-EVAL2-eot | 4 | EOT-PGD with N ≥ 20 and ≥ 50 steps on the final stochastic model | Final model only; EOT-5 with 20 steps is already measured (21.00) |
| G2-EVAL3-shape-texture | 4 | Geirhos cue-conflict | Report vs. human baseline |
| G2-EVAL4-occlusion-clutter | 4 | Occlusion, clutter, peripheral degradation | Report |
| G2-EVAL5-human-mini | 4 | 2AFC mini-study with d′ | Report |
| G2-MEM1-glimpse-memory | 5 | M2 buffer | Paired vs. reference, parameter-matched |
| G2-MEM2-spatial-bank-ior | 5 | M3 bank plus inhibition of return | Re-fixation rate falls, accuracy not worse |
| G2-HALT1-adaptive-stop | 5 | Learned stopping | Only if G2-D5-glimpse-scaling shows monotone gains |
| G2-JEP1-monolithic | 5 | Control predictor | Baseline for the JEP arms |
| G2-JEP2-multihead | 5 | Multi-head predictor | Matched params and FLOPs |
| G2-JEP3-factorized | 5 | Unconstrained factorized predictor | Matched params and FLOPs |
| G2-JEP4-orthogonal | 5 | Orthogonal factorized predictor | Matched params and FLOPs |
| G2-JEP5-factor-error | 5 | Factor-specific error | Paired vs. JEP3 |
| G2-JEP6-factor-disagreement-gaze | 5 | Factor disagreement as a gaze signal | Paired vs. AIS3 |
| G2-STR1-minimal-slots | 5 | 2–4 slots under the re-entry rule | §8.4 rule |
| G3-HRV1-high-res-substrate | 6 | Native-resolution glimpse crops plus a low-res gist, on a high-resolution source | Prerequisite for Gates B and S |
| G3-BUD1-budget-curve | 6 | Accuracy vs. tokens or FLOPs: adaptive, fixed, random, and feed-forward | Gate B |
| G3-VS1-synthetic-search | 6 | Targets at known locations in cluttered 384–512 px scenes | Gate S |
| G3-VS2-counting | 6 | Count small objects across glimpses | Count error falls with inhibition of return |
| G3-WS1-world-state | 6 | World-state container | Starts only after Gate B or Gate S |
| G3-OBJ1-minimal-objects | 6 | Minimal objects | Only if the slot rule passes |
| G3-TDV1-temporal-pipeline | 6 | The sequential-video pipeline (UCF-101/TDV) | After Gate S; prerequisite for M4 |
| G3-MEM3-episodic | 6 | Cross-image episodic memory (M4) | Needs G3-TDV1-temporal-pipeline |
| G3-CF1-counterfactual | 6 | Counterfactual glimpse prediction | Error at unvisited locations, scored after looking |
| G3-RSN1-reasoning | 6 | Reasoning over W_t | After G3-WS1-world-state and G3-MEM3-episodic |
| G4-API1-perceptual-state-api | 6 | Perceptual-state tokens | Prerequisite for the SLM arms |
| G4-SLM1-belief-readout | 6 | Frozen RHAN plus an SLM reading API tokens | Beats the plain classifier head and the shuffled-token control |
| G4-SLM2-perceptual-actions | 6 | The SLM emits LOOK, ZOOM, REOBSERVE, STOP | Beats the AIS policy at equal observations |
| G4-VLM1-raw-vs-rhan | 6 | Raw-VLM vs. RHAN-VLM | Accuracy, observations used, gaze-to-evidence alignment, calibration |
| G4-VLA1-physical-action | 6 | Physical action | Last |

---

## 11. When you are lost

Each row is an incident this project already had.

| Symptom | First check | It happened when |
|---|---|---|
| A metric looks wrong | Is it labeled CLEAN or ROBUST? Which split and mode? | Unlabeled val_acc hid a clean collapse |
| Robustness is flat everywhere | Did adversarial gradient reach every group, and did params move (pre/post-clip norms)? | Autocast bug: 0/163 backbone tensors; HPC starved at ~1e-5 |
| Two phases give identical numbers | Hash the state dicts | gen1_core ≡ ais_v2_swap (Gen-1) |
| "Same config" gives different results | Dataset pins, config hash, commit, parent checkpoint | Unpinned datasets: +12.13 vs. +7.75 pp |
| Two unrelated ablations match | Diff the configs embedded in the evaluated checkpoints | D2 and D3 secretly carried SBR |
| Collapse at the adversarial ramp | Warm start and clean phase first? | From-scratch TRADES: 5.82% |
| A table disagrees with its CSV | Run the consistency assertion | E1 summary vs. per-seed data |
| The doc says X is implemented | Find the test or log line | DINOv2 claimed, never loaded |
| ε seems too strong or weak | Print the pixel equivalent; compare PGD-10 vs. PGD-50 | The "24/255" claim vs. ≈5.4/255 |
| Two agent reports disagree | Check JSONL session ids, HF file list, timestamps | Oct 12: "not started" vs. 63 epochs logged |
| A huge percentage drift | Print which tensor and its absolute value | 255% on a ~1e-4 tensor |
| The same number for several epochs | Check n and the LR at those epochs | 34.77 (178/512) ×3 |
| Train loss falls, val is flat | Overfitting: weight EMA, select-split early stopping | E5 |
| A new phase starts below its parent | The init is not identity | CE spike 5.16 |
| A dip at the LR peak | Peak LR and warmup | E4 |
| STALL fires during the ramp | ε is moving; compare within a stage only | B6 |
| "Best" looks great | It was selected on val; use the final checkpoint | B5 |
| Each phase ends below its parent | Run the null continuation (same protocol, no change) before blaming the new module | Phases 5–6 (E10) |
| The logged loss does not match the documented formula | Fit total ≈ a·CE + b·KL on the logged rows, then read the loss line | 0.55·CE logged vs. 0.45 documented (E12) |
| Robust accuracy looks great for a stochastic model | Re-run with EOT (N ≥ 20) and compare with a deterministic control | PGD-50 28.60 vs. EOT 21.00 (E15) |
| "Stable across seeds" | Compare the seed σ with the binomial σ. Eval seeds or training seeds? | σ 2.89 vs. binomial 2.65 at n=300 (E16) |
| An agent narrates single-epoch moves | Compare with the ±2σ noise; use finals | "Robustness jumps +2.73%" (E13) |
| A verdict reads "confirms no X" from one test | Name the test that would falsify it, and run that | PGD-10 ≈ PGD-50 "confirms no masking" (E14) |

---

## 12. Launch and session checklist

1. Which commit and entry point? Which hashed parent checkpoint? Which config hash?
2. One-epoch smoke on a 2k subset: print s/epoch and projected total hours. Stop if the projection exceeds the quota.
3. Which gates stop the run, and who advances it?
4. On resume: the banner shows epoch, best, and checkpoint sha256, and the JSONL's last epoch matches the checkpoint.
5. Save and exit at 11 h. Verify HF sync with a one-line status.
6. After a phase: evaluate the final checkpoint on identical images, update §1, and commit the JSONL summary.
7. Before quoting any robust number: attack name, N, steps, restarts, n, and the deterministic control (§9).
8. Before starting an arm: which registry ID, which parent, which single change, and what result would make you drop it.

---

## 13. Compute (measure first)

Fill from the logs; do not trust estimates.

| Phase | s/epoch | Epochs | GPU-hours | Platform |
|---|---|---|---|---|
| g2_gist_only | TBD | 30 | TBD | Kaggle |
| recurrence_only | TBD | 15 | TBD | Kaggle |
| belief_no_f | TBD | 15 | TBD | Kaggle/Colab |
| belief_with_f | TBD | 15 | TBD | Colab |
| ais_v2_swap | TBD | 15 | TBD | Colab |
| gen1_core | TBD | 14 | TBD | Colab |

The agent estimated about 68 GPU-hours for the 105-epoch ladder [reported, unmeasured], which implies about 39 minutes per epoch. The four W1.6 runs (15 epochs each) would then cost about 39 GPU-hours. Kaggle gives about 30 GPU-hours a week, plus Colab. Nothing runs locally. The W1.5 eval-only work should take minutes to a few hours of T4 time in total [estimate].

---

## 14. Changelog

v3.0 (2026-10-14):
- Rebuilt around the actual run and its evidence, and folded in the finished ladder and the S7 evaluation.
- Fixed the boundary table (several Gen-3 items were listed under Gen-2) and the λ-search contradiction.
- Replaced the colliding K-series IDs with a single registry (seven waves).
- Defined the reference configuration, an ordered set of waves, and the gates (L, A, and B/S for Gen-3).
- Replaced the compute section with a measure-first rule.
- Added memory, factorized predictive belief, the impact thesis, a Gen-3/Gen-4 path with a small language model, and the symptom table.
- Corrected two agent readings of S7 (EOT and seed σ) and flagged the logged-vs-documented loss weights.
- Dropped the construction-time calibration gate in favor of Gate U on the trained model.