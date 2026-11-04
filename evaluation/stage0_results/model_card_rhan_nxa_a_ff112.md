# Model Card: RHAN-NXA, feedforward-112

> [!IMPORTANT]
> **Status:** DRAFT — AutoAttack results filled from T5 battery (`evaluation/stage0_results/t5_feedforward_battery.json`). Awaiting human review before upload.
> 
> **Family Context:** This checkpoint is the feedforward member of the RHAN-NXA family (the recurrent variant did not earn its cost under prereg consolidated-flagship-112/1). Under the pre-registered protocol [`prereg/stage0_decision_v1.json`](prereg/stage0_decision_v1.json) (`consolidated-flagship-112/1`), the recurrent variant did not earn its compute cost at 112 px, resulting in an official pre-registered verdict of **NULL** (see Stage 0 verdict). See the [Stage 0 Final Report](evaluation/stage0_results/stage0_final_report.md) and the [Superseded Report Note](evaluation/stage0_results/stage0_final_report_superseded.md). 
> 
> **Official Designation:** Seed 17 is designated as the official candidate per prereg field `evaluation/stage0_results/stage0_harness_results.json` / protocol designation (see `prereg/stage0_decision_v1.json` and harness results). Seed 17 checkpoint: `foundation_ff_112_seed17_final_d00fb750.pth`.

---

## Model Details
* **Official Model Name:** `RHAN-NXA, feedforward-112`
* **Architecture:** Compact Vision Transformer (`CompactViTGen2`) feedforward trunk with DINOv2-small warm-start, LayerScale, and patch projection ($14 \times 14$ patches).
* **Official Checkpoint:** `foundation_ff_112_seed17_final_d00fb750.pth` (Designated in advance prior to test scores; SHA8 = `d00fb750`).
* **Framework:** PyTorch / torchvision / timm.
* **Input Resolution:** $112 \times 112 \times 3$ RGB.
* **Normalization:** ImageNet mean $(0.485, 0.456, 0.406)$, std $(0.229, 0.224, 0.225)$.
* **Classes:** 100 classes (ImageNet-100 benchmark) — 100-class scope only.
* **Parameter Count:** 23,595,749 parameters (exact count from checkpoint state dict).
* **Release Date:** November 2026.
* **Repository / Checkpoint Archive:** `FerrariKazu/rhan-nxa-g2-checkpoints` (`closure/stage0/`) — *closure/stage0/ path flagged as unverified until upload*.

---

## Intended Use & Threat Model
* **Intended Use:** High-throughput, robust visual representation baseline for adversarial cognitive modeling research. Serves as the canonical feedforward control against recurrent and foveated vision architectures.
* **Threat Model:**
  * **Single threat model:** $L_\infty$ white-box attacks.
  * **Perturbation Norm:** $L_\infty$ bounded.
  * **Perturbation Budget:** $\varepsilon = 5.417 / 255 \approx 0.0212431$.
  * **Adversarial Objective:** Untargeted misclassification.
  * **Evaluation Surface:** Canonical ImageNet-100 validation set (5,000 images), evaluated at native $112 \times 112$ resolution.
  * **Attacks:** No attacks beyond AutoAttack standard were evaluated in this card.
* **Out-of-Scope:** Not intended for safety-critical deployment or arbitrary out-of-distribution inputs without domain adaptation.

---

## Training Procedure & Recipe
Sourced directly from checkpoint configurations and training logs (`gen2_epoch_log.jsonl` and `prereg/stage0_decision_v1.json`):
* **Training Pipeline:** `train_gen2_foundation.py` (`ff_112` curriculum).
* **Evaluation Protocol:** `consolidated-flagship-112/1`
* **Dataset:** ImageNet-100 (126,689 training images across 100 classes).
* **Training Objective:** TRADES adversarial training with $\beta = 2.5$.
* **Inner Adversarial Loop:** 4-step PGD (`trades_steps = 4`), step size $\alpha = 0.05875$ ($\varepsilon_{\text{norm}} = 0.0940 \pm 0.001$).
* **Optimizer:** AdamW multi-group optimizer with decoupled learning rates:
  * Trunk / Backbone parameters: $\text{LR} = 5.0 \times 10^{-5}$
  * Heads and projection layers: $\text{LR} = 5.0 \times 10^{-4}$
  * Weight decay: $0.05$, gradient clipping norm: $1.0$.
* **Schedule:** Cosine annealing with 3-epoch linear warmup over 15 total epochs.
* **Batch Size:** Resolved from checkpoint/prereg configuration: effective batch size **64** = micro-batch size **16** × gradient accumulation steps **4** (sourced from `prereg/stage0_decision_v1.json`: `effective_batch_size: 64`, `micro_batch_size: 16`, `grad_accum_steps: 4`). 
* **Seeds Trained:** Seeds 17, 42, 99.

---

## Quantitative Performance Summary (3-Seed Distribution)

### 5k Full Validation Set ($N = 5{,}000$, $\varepsilon = 5.417/255$)
Sourced from `evaluation/stage0_results/stage0_harness_results.json`:

| Metric | Designated Official (Seed 17) | 3-Seed Mean | 3-Seed Range (Min – Max) | JSON Path |
|---|---|---|---|---|
| **Clean Accuracy (5k)** | **74.30%** | **74.49%** | $[74.30\%,\; 74.60\%]$ | `stage0_harness_results.json -> arms.ff_112_seed*.clean_acc` |
| **Worst-Case Robust (5k)** | **36.04%** | **35.79%** | $[35.44\%,\; 36.04\%]$ (seed range 0.6 pp on 5k worst-case) | `stage0_harness_results.json -> arms.ff_112_seed*.worst_case_robust` |
| **PGD-50 (5k)** | 38.18% | 37.55% | $[37.00\%,\; 38.18\%]$ | `stage0_harness_results.json -> arms.ff_112_seed*.attacks.pgd50.robust_acc` |
| **APGD-CE (5k)** | 37.66% | 37.01% | $[36.52\%,\; 37.66\%]$ | `stage0_harness_results.json -> arms.ff_112_seed*.attacks.apgd_ce.robust_acc` |
| **APGD-DLR (5k)** | 38.10% | 37.95% | $[37.40\%,\; 38.34\%]$ | `stage0_harness_results.json -> arms.ff_112_seed*.attacks.apgd_dlr.robust_acc` |

### 1k Canonical Slice ($N = 1{,}000$) with Wilson 95% Confidence Intervals
Sourced from `evaluation/stage0_results/t5_feedforward_battery.json`. Note: 1k Wilson 95% CIs are roughly ±3 pp; cross-seed ordering unresolved.

| Metric | Seed 17 (Official) [Wilson 95% CI] | Seed 42 [Wilson 95% CI] | Seed 99 [Wilson 95% CI] | 3-Seed Mean |
|---|---|---|---|---|
| **Clean Accuracy (1k)** | 73.90% [71.09%, 76.53%] | 74.20% [71.40%, 76.82%] | 73.90% [71.09%, 76.53%] | 74.00% |
| **ECE (1k)** | 6.52% | 6.15% | 6.50% (verified from JSON; earlier note ~6.49 vs 6.50 - see source) | 6.39% |
| **PGD-50 (1k)** | 37.90% [34.94%, 40.95%] | 37.30% [34.36%, 40.34%] | 36.50% [33.57%, 39.53%] | 37.23% |
| **PGD-100 (1k)** | 37.60% [34.65%, 40.64%] | 37.10% [34.16%, 40.14%] | 36.30% [33.38%, 39.33%] | 37.00% |
| **PGD Gap (1k)** | 0.30 pp | 0.20 pp | 0.20 pp | 0.23 pp |

### AutoAttack Standard (1k canonical slice, $N = 1{,}000$) with Wilson 95% CIs
Sourced from `evaluation/stage0_results/t5_feedforward_battery.json -> arms.ff_112_seed*.autoattack` (AutoAttack standard only):

| Attack | Seed 17 (Official) [Wilson 95% CI] | Seed 42 [Wilson 95% CI] | Seed 99 [Wilson 95% CI] | 3-Seed Mean |
|---|---|---|---|---|
| **APGD-CE** | 37.40% [34.45%, 40.44%] | 36.90% [33.96%, 39.94%] | 36.30% [33.38%, 39.33%] | 36.87% |
| **APGD-T** | 35.00% [32.11%, 38.01%] | 35.80% [32.89%, 38.82%] | 35.00% [32.11%, 38.01%] | 35.27% |
| **FAB-T** | 35.40% [32.50%, 38.41%] | 36.00% [33.08%, 39.02%] | 35.40% [32.50%, 38.41%] | 35.60% |
| **Square** | 52.30% [49.20%, 55.38%] | 51.20% [48.10%, 54.29%] | 50.60% [47.50%, 53.69%] | 51.37% |
| **AA Worst-Case** | **35.00% [32.11%, 38.01%]** | **35.80% [32.89%, 38.82%]** | **34.80% [31.91%, 37.81%]** | **35.20%** |

### Epsilon Sweep (1k canonical slice, $N = 1{,}000$) with Wilson 95% CIs
Sourced from `evaluation/stage0_results/t5_feedforward_battery.json -> arms.ff_112_seed*.epsilon_sweep`:

| ε Factor | ε (/255) | Seed 17 [Wilson 95% CI] | Seed 42 [Wilson 95% CI] | Seed 99 [Wilson 95% CI] | 3-Seed Mean |
|---|---|---|---|---|---|
| 0.5× | 2.71 | 57.30% [54.21%, 60.33%] | 56.80% [53.71%, 59.84%] | 56.80% [53.71%, 59.84%] | 56.97% |
| 1.0× | 5.42 | 38.00% [35.04%, 41.05%] | 37.50% [34.55%, 40.54%] | 36.50% [33.57%, 39.53%] | 37.33% |
| 2.0× | 10.83 | 10.40% [8.66%, 12.45%] | 10.70% [8.93%, 12.77%] | 11.30% [9.48%, 13.41%] | 10.80% |

---

## Artifact Checksums
* `foundation_ff_112_seed17_final_d00fb750.pth`:
  `d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc`
* `foundation_ff_112_seed42_final_2920df54.pth`:
  `2920df54fe919a8e57dcca8788b5429d0d5f9622fb867723a4d248b063276f71`
* `foundation_ff_112_seed99_final_abde98a0.pth`:
  `abde98a0129f5637039eb98307b30862a04a3cffd0c4b749c8e5cf40c5d4ed4d`

---

## Limitations
1. **Benchmark Scope:** Trained and evaluated strictly on ImageNet-100 only (100 classes). Generalization to full ImageNet-1k or other vision domains is unverified.
2. **Fixed Low Resolution:** Operates strictly at $112 \times 112$ pixels.
3. **Threat Model:** Single $L_\infty$ white-box threat model. No attacks beyond AutoAttack standard evaluated.
4. **Perturbation Regime:** Bounded under $L_\infty$ at a single epsilon ($\varepsilon = 5.417/255$). Robustness degrades sharply under larger budgets (accuracy drops to $\approx 10.8\%$ at $2\times \varepsilon$).
5. **Diagnostic Subsample:** Detailed AutoAttack components (AutoAttack Standard on the 1k-slice) and epsilon sweeps were benchmarked on a canonical 1k-image subset ($N = 1{,}000$). 1k Wilson 95% CIs are roughly ±3 pp; cross-seed ordering unresolved.
6. **No Corruption/OOD Evaluation:** No corruption (e.g., ImageNet-C) or natural OOD evaluation has been performed.
7. **Calibration Error:** Expected Calibration Error (ECE) is $\approx 6.5\%$ on clean test data (1k slice; seed 99 ~6.50 as verified).
8. **Absolute Robustness:** Worst-case adversarial accuracy is bounded around $\approx 35\text{--}36\%$ on 5k (seed range 0.6 pp on 5k worst-case); AA worst-case ~35% on 1k.
9. **Extreme Budget Degradation:** At $2\times \varepsilon$ robust accuracy drops to $\approx 10\%$.
10. **Deployment Caution:** Strictly for scientific benchmarking. Not for safety-critical use or security-sensitive applications.
11. **Archive Path:** `closure/stage0/` path flagged as unverified until upload.
