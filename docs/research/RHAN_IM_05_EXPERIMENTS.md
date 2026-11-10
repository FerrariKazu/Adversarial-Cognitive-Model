# 05 — Experiments, ablations and evaluation protocol

Everything here is a **plan**; nothing below has been run on the real checkpoint. Thresholds are drafts to be frozen in a pre-registration *before* the corresponding run (the repository already works this way: `prereg/stage0_decision_v1.json`). Simulated results that motivate the plan are in `RHAN_IM_RESULTS.md`.

## 1. Scene set

Composited scenes: a 224×224 canvas; background = a natural image from a class that is **never** a target; with probability `1 − p_null` (default `p_null = 0.3`) one target image of a target class is pasted at a unit-aligned position (size 2, 4 or 6 fine units = 28/56/84 px; opacity 1.0 or 0.6). Target classes = half of the ImageNet-100 class list (50), background classes = the other half. Training scenes come from the `train` split; calibration and test scenes come from `val`, built from **disjoint halves** of each class's sorted image list (every second file), so that no image is used for both the standardiser and the measurement; test scenes are generated once from a fixed seed and hash-pinned. With the usual 50 val images per class each half has 25, so scenes reuse source images with different sizes, positions and opacities.

*Limits stated up front:* pasted squares have sharp edges and context mismatch; this is a controlled stimulus with a **known footprint**, not natural object-in-scene imagery. A natural-scene check (real objects with boxes) is a later experiment (E14) and nothing is claimed beyond the stimulus.

## 2. Arms (all evaluated on the same scenes)

| Arm | Description | Role |
|---|---|---|
| A0 | `ff_112` (seeds 17, 42, 99): whole canvas resized to 112 px, one pass | feedforward gist baseline (the validated checkpoints, untouched) |
| A1 | `ff_112` trunk, fixed-lattice coverage of native windows, logits averaged | strongest non-adaptive baseline |
| A2 | `recurrence_112` (seeds 17, 42): the validated flagship | recompute comparator at its own FLOPs |
| A3 | RHAN-IM, random-without-replacement looks | policy ablation |
| A4 | RHAN-IM, MAP-cell looks | policy ablation |
| **A5** | **RHAN-IM, Koopman policy** (the proposal) | |
| A6 | RHAN-IM, exact one-step lookahead (EVOI) on a subset | closed form vs exact |
| A7 | RHAN-IM, top-2-contrast ("severe test", C5) | policy ablation |
| A8 | learned glimpse policy on the same sensor (RL/gradient policy over posterior features; AdaptiveNN-style) | learned competitor; implemented only if time allows, otherwise reported as not run |
| A9 | C4/C7 "recompute" controls | negative control for the thesis (cheap versions only) |

State ablations (on A5): sum vs best combination; constant `d′`; no per-position standardisation; no within-look `VIF_null` correction (and a per-target-size `VIF`); no scale mixture when shared difficulty is flagged; no clip; no cap; no gist look; no minimum-looks floor; head-only vs tuned trunk (S3); stopping by nominal λ vs plug-in vs LTT.

## 3. Statistical protocol (applies to every experiment)

- **Matched budgets.** Same scenes, same ε, same attack steps and restarts, same evaluation harness as `stage0_decision_v1` for the worst-of-3 attacks (PGD-50, APGD-CE, APGD-DLR; bitwise-AND worst case). FLOPs per arm counted with `FlopCounterMode`; every comparison reported **at matched looks and at matched FLOPs**.
- **Seeds.** Three training seeds (17, 42, 99, as in the existing runs) for any trained component (evidence head, S3/S4 tuning); ≥ 5 scene-sampling seeds for evaluation splits. Final checkpoint only; best-checkpoint selection is banned.
- **Tests.** Paired comparisons on identical scenes: exact McNemar for accuracy differences; paired bootstrap (10,000 resamples) for continuous metrics (ECE, looks, risk); Holm correction across the pre-registered family; a practical effect-size floor (+2 pp unless stated) so that "significant but trivial" is not a pass.
- **Superiority across seeds.** Re-use the existing rule: `min over seeds(candidate) > max over seeds(comparator)` **and** pooled McNemar p < 0.01.
- **Independence and splits (for the LTT claims).** Learn-then-Test needs independent calibration items drawn from the deployment distribution, on a split that is disjoint from everything fitted upstream (evidence head, standardiser, `d′`, VIFs, κ, policy settings), with the threshold grid and `k_min` fixed before the calibration losses are seen (`rhan_im/stopping.py` cannot check any of this). Exchangeability alone does not give the binomial tail. Scenes that reuse a source image are not independent, and the scene design above does reuse them (25 val images per class per half), so for E5 the calibration and test scenes use **one scene per source image** and come from an image-level split of ImageNet-100 into four disjoint sets (head training, measurement of `d′`/standardiser/VIFs, LTT calibration, final test), fixed before any E5 number is seen. Shifted conditions are reported separately and *expected* to break the guarantee.
- **No assumption of correlates.** Internal stability or smoothness is **measured** against robustness, not assumed to predict it (E10).

## 4. Experiments

| ID | Question | Metrics | Pass (draft) | Kill / revise |
|---|---|---|---|---|
| E1 | **Gate 1.** Does the frozen `ff_112` trunk + head behave like the equal-variance SDT sensor, and does looking closer pay? | `d′` map (8×8) per size/opacity; z-ROC slope; null/positive sd; wrong-class leakage; native-over-gist ratio; cross-look ρ; within-look `VIF_null` and `VIF_pos`, the corrected block-level `d_block`, spillover onto uncovered tokens | slope [0.8, 1.25]; null sd [0.85, 1.2]; centre `d′ ≥ 1.5`; `d_block ≥ 2.0`; native/gist ≥ 1.5; `VIF_null` reported and, if > 1.25, corrected; shared difficulty flagged if `VIF_pos / VIF_null` > 1.25 | see `04` §11 P1–P2; `RHAN_IM_07_GATE1_PREREG_DRAFT.md` |
| E2 | Is the posterior calibrated? | ECE (15 bins), NLL, reliability, per look count, **per target size** and, if shared difficulty was flagged, per difficulty tercile; ablations constant `d′`, wrong mode, no `VIF_null` correction, no scale mixture | ECE ≤ 0.03 (measured `d′`); ≥ 0.08 for constant `d′` | ECE > 0.05 with measured `d′` ⇒ likelihood misspecified: stop and diagnose |
| E3 | Does the computed policy beat coverage? | accuracy at K = 1…8; looks to risk α; paired differences vs A1 | A5 − A1 ≥ +2 pp at K = 4 (CI excludes 0), at matched FLOPs | not met ⇒ report; policy not the contribution |
| E4 | Accuracy-vs-compute frontier | accuracy vs FLOPs for A0, A1, A2, A5, (A8) | A5 on or above the A1/A2 frontier | below A1 everywhere ⇒ adaptive looking does not pay on this sensor |
| E5 | Is risk-controlled stopping valid and useful? | violation rate over ≥ 500 calibration/test resamples (α = δ = 0.10), mean looks, coverage (selective variant), vs plug-in and nominal | LTT violation ≤ δ; plug-in ≥ 2δ at n_cal = 300; looks saved vs fixed budget ≥ 25 % at equal error | LTT violation > δ ⇒ an LTT assumption is broken (independence by source image, split disjointness, grid and `k_min` fixed in advance): find which |
| E6 | Clean and adversarial accuracy under budgets and threat models | clean acc; worst-of-3 robust acc at ε = 5.417/255 for (a) whole-canvas ℓ∞ with fixed look sequence, (b) adaptive: EOT over look sequences (≥ 4), straight-through/soft-policy variants, (c) patch attacks (14/28/56 px, adversarial placement over the lattice), (d) look-targeted: perturb only the windows the policy will pick, (e) compute-exhaustion (maximise looks / delay stop) | A5 robust ≥ A0 at ≤ equal FLOPs (CI excludes a loss); (e) raises mean looks < 2× | loss to A0 ⇒ no robustness claim; (e) ≥ 2× ⇒ add leave-one-look-out margin / hard cap |
| E7 | Are the certificates meaningful on the real trunk? | **look-granular** certificate (implement first): certified fraction vs `m`, `κ`, `u_cap`; empirical attack success among certified (must be 0); exhaustive soundness at small size | certified fraction (m = 1 look) ≥ 50 % at some `κ` with clean accuracy loss ≤ 2 pp | any flip among certified ⇒ bug; fraction < 10 % ⇒ no certificate claim (state as negative) |
| E8 | Corruptions and shift | accuracy, ECE, looks, LTT violation under corruptions (severity 1–5), new backgrounds, new resolutions | degradation reported with CIs; guarantee *expected* to break under shift — measure by how much | silent failure (confidence stays high while accuracy falls) ⇒ report as a limitation |
| E9 | OOD / open set | AUROC of `p(H_0)` and of max posterior on non-target-class pastes and target-absent scenes; selective risk-coverage vs max-softmax of A0 | AUROC ≥ A0's | worse than max-softmax ⇒ report |
| E10 | Representation stability, intermediate-state sensitivity, trajectory consistency — **and whether they relate to robustness** | decision consistency across looks; rank stability of the ignorance map; state sensitivity `‖Δ log-weights‖ / ‖Δx‖` per look; Spearman correlation of each with per-scene adversarial survival, with bootstrap CIs, per seed | pre-registered as exploratory: **no claim** that internal stability predicts robustness unless the CI excludes 0 in ≥ 2/3 seeds | correlation ≈ 0 ⇒ say so |
| E11 | Frequency robustness | Fourier-sensitivity maps (single-basis perturbations); band-limited PGD (low/high); A0 vs A5 | report; no pre-set threshold (exploratory) | — |
| E12 | Convergence and variance | S1 loss and held-out `d′` vs step; look-count distribution; seed-to-seed spread of every headline metric | spread reported; headline gaps must exceed spread | gaps inside seed spread ⇒ not a result |
| E13 | Close the Gen-2 question (optional) | Stage A replication `recurrence_112_seed17_b64` vs `ff_b64`; control `c4_four_views_no_refine` | pre-registered rule as in `stage0_decision_v1` | off the critical path |
| E14 | Natural-scene validation | real objects with boxes on natural backgrounds; same metrics as E3, E5 | — | claims limited to the stimulus until this is run |

## 5. Adaptive-attack protocol (E6, spelled out because it is the usual place for overclaiming)

The posterior is differentiable in the scores, hence in the pixels, **given a fixed look sequence**; the policy's argmax is not. Three white-box strategies, reported separately and combined by taking the worst case per scene:
1. *Fixed-sequence:* run the policy on `x`, freeze the sequence, attack the final posterior (PGD-50/APGD) — weakest.
2. *EOT over sequences:* average gradients over sequences obtained by perturbing the posterior before the argmax (≥ 4 samples) — handles the discrete choice statistically.
3. *Soft-policy relaxation:* replace the argmax by a softmax over gains with temperature annealing; straight-through gradient — a distinct attack surface.
The harness's worst-case rule is unchanged (a scene counts as robust only if every attack fails). Compute-exhaustion attacks maximise `E[looks]` under the same ε. An attack is reported with its own sanity check: it must reduce accuracy to the floor on an undefended control (the `κ = ∞`, no-clip posterior).

## 6. What would change the plan

Any kill condition above is a result, reported as such, and triggers the corresponding branch in `RHAN_IM_03_CANDIDATES.md` §9 (switch conditions). Experiments E13 and A8–A9 are optional; everything else is on the critical path in the order E1 → E2/E3/E5 → E6/E7 → E8–E12.
