# Gate-2 Erratum 02 and Rules v1.2: Train-Holdout d′ and Implementation Clarifications

Status: **RULES AMENDMENT, 17 Nov 2026.** This erratum supersedes `prereg/gate2_rules_v1_1.json` only where it says so. It does not rewrite v1.0 or v1.1. The v1.1 markdown draft is unchanged and remains pinned by the v1.1 file.
Scope: **INTERNAL QUALIFICATION ONLY.** No spatial-resolution claim is added.

## 1. The defect v1.2 fixes

v1.1 assigned the Gate-2 fit pool a null-only role: 850 non-reused null scenes for the standardisers, $VIF_{\text{null}}$, and $\kappa$. It did not say where the positive scenes for $d_{\text{tok}}$ and $d_{\text{gist}}$ come from. Reusing Gate 1's validation d′ is invalid because Gate 1's calibration and test pools overlap the Gate-2 pools (Gate 1 calibration vs Gate 2 fit/cal/test: 900/1700, 800/1700, 800/1600; Gate 1 test vs Gate 2 fit/cal/test: 800/1700, 900/1700, 800/1600), and `rhan_im/stopping.py` requires the LTT calibration split to be disjoint from every split used to fit the sensor, including d′.

## 2. Train-holdout d′

The head and the d′ measurement now use disjoint train images, pinned in `prereg/gate2_train_holdout.json`:

- **Rule:** per class, sort train file names by code point; hold out the first 16 images of every target class and the first 32 images of every background class.
- **Holdout size:** 800 target images + 1,600 background images = 2,400 images.
- **Head training:** the remaining 124,289 train images. The head is refitted from scratch exactly as in Gate 1 (600 steps, batch 48, AdamW, lr 2e-3, cosine decay, seed matching the evaluation seed).
- **d′ measurement:** target scenes use the 800 held-out target images and 800 of the held-out background images; null scenes for the cross-look correlation use the other 800 background images. No holdout image is used in head training, and no image is reused within the d′ fit.
- **Fit pool:** unchanged null-only role. The 850 non-reused null scenes from `fit_pool_part_0_of_3` still fit the standardisers, the within-look VIFs, and $\kappa$.
- **No image overlap:** head-training images and d′-fit images are disjoint by construction.

The holdout file is pinned by SHA-256 in `prereg/gate2_rules_v1_2.json`. The generator is `scripts/rhan_im/gate2_split.py`; `--check` recomputes the rule against the data tree.

## 3. Sensor fit at the operating point

- **Operating cell:** target size 4 units (56 px), opacity 1.0.
- **d′:** `d_tok` is the measured (8×8) token map at the operating cell; `d_gist` is the positive-count-weighted mean of the measured gist map at the same cell. Both are measured on the train holdout. If either is not finite, the run refuses; there is no fallback to Gate 1.
- **Standardisers:** per (look type, token position, class) null mean / sd on the 850 fit-pool null scenes.
- **VIF:** `vif_tok` and `vif_gist` are measured on the fit-pool null scenes with the operating-cell d′ as weights. Each is used only if it is finite and > 1.25; otherwise its correction factor is 1.0. This is the v1.1 equicorrelated correction, applied separately to the native and gist looks.
- **κ:** the 99.9th percentile of the absolute VIF-corrected null scores (native and gist pooled). If not finite, the run refuses.
- **Combination mode:** measured, not copied. Two independent native windows per d′-fit scene give `rho_null`; `sum` if `rho_null <= 0.2`, otherwise `best`. This is the Gate 1 rule and the conservative choice between 0.2 and 0.6.
- **Robustness knobs:** `unit_cap = None` (exact Bayes). Gate 2 does not claim the bounded-influence certificates.

## 4. Scene plans and the size-conditioned posterior

The v1.1 scene counts and no-reuse contract are unchanged. v1.2 fixes how the sizes are assigned:

- **Fit pool:** 850 null scenes; no target image is consumed.
- **Calibration pool:** 425 target + 425 null scenes. Target scenes use sizes 2, 4, and 6 in a balanced round-robin (about 142/142/141). Null scenes are assigned an evaluation size in the same balanced way. Target scenes consume 425 target and 425 background images; null scenes consume the remaining 425 background images.
- **Test pool:** 400 target + 400 null scenes, with target sizes 2, 4, and 6 balanced (about 134/133/133). Null scenes are assigned evaluation sizes in the same balanced way. Target scenes consume 400 target and 400 background images; null scenes consume the remaining 400 background images.
- **Target opacity:** 1.0 for all E2/E5 scenes. v1.1 did not specify an opacity; the 0.6 cell from Gate 1 is not part of Gate 2.
- **Size-conditioned evaluation:** each scene is evaluated with the footprint of its assigned evaluation size. This includes null scenes, which are assigned sizes in the same balanced way. The posterior is therefore conditioned on the generating size; Gate 2 makes no claim about inferring size from pixels. E2 reports per-size results on this basis, and E5 pools the independent scenes.
- **No reuse:** the plans consume each image at most once within a pool; fit, calibration, test, and train-holdout images are disjoint.

## 5. Look policy

- **K = 8.** Look 1 is the gist view. Looks 2–8 are seeded independent draws (with replacement) from the 25 native-window candidates. The sequence is a deterministic function of the scene seed and look step; it is not adaptive.
- **No spatial claim:** the sequence is not selected from the image or from the posterior.

## 6. E2: calibration

- **Primary metric:** ECE with 15 equal-frequency bins at K = 8 on the full 800-scene test pool, using the Bayes-decision confidence and correctness.
- **Per-size diagnostics:** the same ECE at K = 8 for the size-2, size-4, and size-6 subsets.
- **Secondary metrics:** NLL and Brier score at K = 8; ECE at every K = 1..8 as a diagnostic (not a pass criterion).
- **Decision:** PASS if primary ECE ≤ 0.03; INCONCLUSIVE if 0.03 < ECE ≤ 0.05; FAIL/DIAGNOSE if ECE > 0.05. v1.2 fixes the primary at K = 8 instead of leaving "max over K" implicit.

## 7. E5: Learn-Then-Test

- **Primary risk:** selective committed error, `P(decided AND wrong)`, as implemented by `stopped_stats_selective`. Coverage is reported next to it. Forced-answer error is a secondary report. This matches the Gate-2 selective-classification scope; a policy that abstains is visible through coverage and the look-saving criterion.
- **Calibration:** 850 scenes; each bootstrap resample draws `n_cal = 300` with replacement; `B = 500` resamples.
- **Threshold grid:** fixed before the losses are seen, ordered conservative to aggressive:
  `0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50`.
- **Minimum look floor:** `k_min = 1`, fixed for every threshold. The v1.1 wording that suggested selecting `k_min,b` is superseded: `rhan_im/stopping.py` requires `k_min` to be fixed before the calibration losses are seen, and selecting it from the same losses would break the LTT guarantee.
- **Validity:** the fraction of resamples whose test committed error is > 0.10 must be ≤ 0.10.
- **Efficiency:** mean looks saved versus the K = 8 horizon must be ≥ 25%, with mean test committed error no higher than the fixed-horizon committed error.
- **Conservative control:** the fixed-threshold plug-in rule is reported. Its violation rate is a diagnostic of the setting, not a pass/fail criterion: a low rate means the setting is easy, not that LTT is invalid.
- **Kill condition:** an aggregate violation rate > 0.10 invalidates the risk-control claim and triggers diagnosis.

## 8. Aggregation

- Seeds 17, 42, 99.
- E2: PASS if at least two seeds PASS; INCONCLUSIVE if not PASS but at least two seeds are PASS or INCONCLUSIVE; otherwise FAIL.
- E5: validity and efficiency are each aggregated 2-of-3. E5 PASS requires both aggregated criteria to pass. The conservative-control diagnostic is reported but does not decide the gate.

## 9. What is still deferred

The measurement code is not yet locked. v1.2 pins the train holdout, this erratum, and the unchanged v1.1 markdown; the code-lock step must pin the Gate-2 library, measurement script, aggregator, and the upstream files before any Gate-2 data run.
