# Model Card: RHAN-NXA, feedforward-112

> [!IMPORTANT]
> **Status:** DRAFT. Not for upload until every item under "Open items before release" is closed and this banner is removed.
>
> **Family context:** This checkpoint is the feedforward member of the RHAN-NXA family. Under the pre-registered protocol `consolidated-flagship-112/1` ([`prereg/stage0_decision_v1.json`](prereg/stage0_decision_v1.json)), the recurrent variant did not earn its compute cost at 112 px; the pre-registered verdict is **NULL**. See the [Stage 0 Final Report](evaluation/stage0_results/stage0_final_report.md) and the [superseded report note](evaluation/stage0_results/stage0_final_report_superseded.md).
>
> **Official designation:** Seed 17 (`foundation_ff_112_seed17_final_d00fb750.pth`) is the designated checkpoint. *Open item 1: cite the exact prereg JSON key and commit/date that fixed this choice before test scores were seen.*

---

## Model Details
* **Official Model Name:** `RHAN-NXA, feedforward-112`
* **Architecture:** Compact Vision Transformer (`CompactViTGen2`) feedforward trunk with DINOv2-small warm-start, LayerScale, and patch projection ($14 \times 14$ patches).
* **Official Checkpoint:** `foundation_ff_112_seed17_final_d00fb750.pth` (SHA8 = `d00fb750`).
* **Framework:** PyTorch / torchvision / timm.
* **Input Resolution:** $112 \times 112 \times 3$ RGB.
* **Normalization:** ImageNet mean $(0.485, 0.456, 0.406)$, std $(0.229, 0.224, 0.225)$.
* **Classes:** 100 classes (ImageNet-100 benchmark) — 100-class scope only.
* **Parameter Count:** 23,595,749 parameters (exact count from checkpoint state dict).
* **Release Date:** November 2026 (planned).
* **Repository / Checkpoint Archive:** `FerrariKazu/rhan-nxa-g2-checkpoints` (path to be confirmed at upload; Open item 6).

---

## Intended Use & Threat Model
* **Intended Use:** High-throughput, robust visual representation baseline for adversarial cognitive modeling research. Serves as the canonical feedforward control against recurrent and foveated vision architectures.
* **Threat Model:**
  * **Single threat model:** $L_\infty$ white-box attacks, untargeted misclassification.
  * **Perturbation budget (training):** $\varepsilon = 0.094$ in ImageNet-normalized space, one scalar for all channels. In pixel units this is $5.49 / 5.37 / 5.39$ per 255 for R / G / B (mean $5.417/255 \approx 0.02124$).
  * **Clamp (training):** adversarial inputs are clamped to $[-4, 4]$ in normalized space. This is not the same as clipping to the valid $[0,1]$ pixel range. *Open item 3: confirm the harness and AutoAttack conventions (normalized-space epsilon, clamp) and state them here.*
  * **Evaluation surface:** Canonical ImageNet-100 validation set (5,000 images) at native $112 \times 112$ resolution; AutoAttack standard and the epsilon sweep on a canonical 1k slice.
  * **Attacks:** No attacks beyond AutoAttack standard, PGD-50/100 and APGD-CE/DLR were evaluated.
* **Out-of-Scope:** Not intended for safety-critical deployment or arbitrary out-of-distribution inputs without domain adaptation.

---

## Training Procedure & Recipe
Sourced from `training/train_gen2_foundation.py` and the epoch logs (`gen2_epoch_log.jsonl`). Open item 4 (provenance check that training commit `512a562` has the same math as the code described here) must be closed before release.
* **Pipeline:** `train_gen2_foundation.py`, phase `ff_112`.
* **Dataset:** ImageNet-100 (126,689 training images, 100 classes). Training augmentation: RandomResizedCrop(112, scale 0.35–1.0), horizontal flip, ImageNet normalization.
* **Objective:** TRADES. Total loss $= w_{\text{trades}} \cdot \big(\mathrm{CE}(f(x), y) + \beta \cdot \mathrm{KL}(f(x_{\text{adv}}) \,\|\, f(x))\big)$ with $\beta = 2.5$ and $w_{\text{trades}} = 0.55$, i.e. effective weights 0.55 on CE and 1.375 on KL.
* **Inner loop:** 4-step PGD on the KL objective, uniform random start in $[-\varepsilon, \varepsilon]$, step size $\alpha = 2.5\,\varepsilon / \text{steps} = 0.05875$, $\varepsilon_{\text{norm}} = 0.094$. The attack runs in fp32 outside autocast.
* **Curriculum:** adversarial training with $\varepsilon_{\text{norm}} = 0.094$, $\beta = 2.5$, $w_{\text{trades}} = 0.55$, 4 PGD steps, constant across epochs (to be attached as a raw table; Open item 5).
* **Optimizer:** AdamW, trunk LR $5.0 \times 10^{-5}$, heads LR $5.0 \times 10^{-4}$, weight decay 0.05 on tensors with ndim $\ge 2$ only, gradient clipping 1.0 per parameter group.
* **Schedule:** 3-epoch linear warmup then cosine decay, 15 epochs total, stepped once per epoch.
* **Batch size:** *Open item 2.* The training code has no gradient accumulation; every loader batch is one optimizer step. The wording "effective batch 64 = micro-batch 16 × accumulation 4" in `prereg/stage0_decision_v1.json` does not match the code and is withdrawn. A 1,979-steps-per-epoch log (126,689 // 64) exists for the c4 control run; the batch-size argument used for the ff_112 runs must be confirmed from the launcher or logs (the CLI default is 48).
* **Seeds trained:** 17, 42, 99.

---

## Quantitative Performance Summary (3-Seed Distribution)

### 5k Full Validation Set ($N = 5{,}000$)
Sourced from `evaluation/stage0_results/stage0_harness_results.json`. The 3-seed mean is the less optimistic summary: seed 17 is the highest of the three seeds on worst-case, PGD-50 and APGD-CE (not on APGD-DLR).

| Metric | 3-Seed Mean | 3-Seed Range (Min – Max) | Seed 17 (designated) | JSON Path |
|---|---|---|---|---|
| **Clean Accuracy (5k)** | **74.49%** | $[74.30\%,\; 74.60\%]$ | 74.30% | `stage0_harness_results.json -> arms.ff_112_seed*.clean_acc` |
| **Worst-Case Robust (5k)** | **35.79%** | $[35.44\%,\; 36.04\%]$ | 36.04% | `stage0_harness_results.json -> arms.ff_112_seed*.worst_case_robust` |
| **PGD-50 (5k)** | 37.55% | $[37.00\%,\; 38.18\%]$ | 38.18% | `stage0_harness_results.json -> arms.ff_112_seed*.attacks.pgd50.robust_acc` |
| **APGD-CE (5k)** | 37.01% | $[36.52\%,\; 37.66\%]$ | 37.66% | `stage0_harness_results.json -> arms.ff_112_seed*.attacks.apgd_ce.robust_acc` |
| **APGD-DLR (5k)** | 37.95% | $[37.40\%,\; 38.34\%]$ | 38.10% | `stage0_harness_results.json -> arms.ff_112_seed*.attacks.apgd_dlr.robust_acc` |

### 1k Canonical Slice ($N = 1{,}000$) with Wilson 95% Confidence Intervals
Sourced from `evaluation/stage0_results/t5_feedforward_battery.json`. 1k Wilson 95% CIs are roughly ±3 pp; cross-seed ordering is unresolved.

| Metric | Seed 17 (designated) [Wilson 95% CI] | Seed 42 [Wilson 95% CI] | Seed 99 [Wilson 95% CI] | 3-Seed Mean |
|---|---|---|---|---|
| **Clean Accuracy (1k)** | 73.90% [71.09%, 76.53%] | 74.20% [71.40%, 76.82%] | 73.90% [71.09%, 76.53%] | 74.00% |
| **ECE (1k)** | 6.52% | 6.15% | 6.50% | 6.39% |
| **PGD-50 (1k)** | 37.90% [34.94%, 40.95%] | 37.30% [34.36%, 40.34%] | 36.50% [33.57%, 39.53%] | 37.23% |
| **PGD-100 (1k)** | 37.60% [34.65%, 40.64%] | 37.10% [34.16%, 40.14%] | 36.30% [33.38%, 39.33%] | 37.00% |
| **PGD Gap (1k)** | 0.30 pp | 0.20 pp | 0.20 pp | 0.23 pp |

### AutoAttack Standard (1k canonical slice, $N = 1{,}000$) with Wilson 95% CIs
Sourced from `evaluation/stage0_results/t5_feedforward_battery.json -> arms.ff_112_seed*.autoattack`.

| Attack | Seed 17 (designated) [Wilson 95% CI] | Seed 42 [Wilson 95% CI] | Seed 99 [Wilson 95% CI] | 3-Seed Mean |
|---|---|---|---|---|
| **APGD-CE** | 37.40% [34.45%, 40.44%] | 36.90% [33.96%, 39.94%] | 36.30% [33.38%, 39.33%] | 36.87% |
| **APGD-T** | 35.00% [32.11%, 38.01%] | 35.80% [32.89%, 38.82%] | 35.00% [32.11%, 38.01%] | 35.27% |
| **FAB-T** | 35.40% [32.50%, 38.41%] | 36.00% [33.08%, 39.02%] | 35.40% [32.50%, 38.41%] | 35.60% |
| **Square** | 52.30% [49.20%, 55.38%] | 51.20% [48.10%, 54.29%] | 50.60% [47.50%, 53.69%] | 51.37% |
| **AA Worst-Case** | **35.00% [32.11%, 38.01%]** | **35.80% [32.89%, 38.82%]** | **34.80% [31.91%, 37.81%]** | **35.20%** |

### Epsilon Sweep (1k canonical slice, $N = 1{,}000$) with Wilson 95% CIs
Sourced from `evaluation/stage0_results/t5_feedforward_battery.json -> arms.ff_112_seed*.epsilon_sweep`.

| ε Factor | ε (/255, mean over channels) | Seed 17 [Wilson 95% CI] | Seed 42 [Wilson 95% CI] | Seed 99 [Wilson 95% CI] | 3-Seed Mean |
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
1. **Benchmark scope:** Trained and evaluated on ImageNet-100 only (100 classes). Generalization to full ImageNet-1k or other domains is unverified.
2. **Fixed low resolution:** Operates at $112 \times 112$ pixels.
3. **Threat model:** Single $L_\infty$ white-box threat model; no attacks beyond those listed above.
4. **Single epsilon:** Trained and evaluated at one budget ($\varepsilon \approx 5.4/255$). Accuracy drops to about 10.8% at $2\times\varepsilon$.
5. **Diagnostic subsample:** AutoAttack components and the epsilon sweep were run on a canonical 1k-image slice; 1k Wilson 95% CIs are roughly ±3 pp, so cross-seed ordering is unresolved.
6. **No corruption/OOD evaluation:** No ImageNet-C or natural-OOD evaluation has been performed.
7. **Calibration:** ECE is 6.15–6.52% across the three seeds on the 1k slice (mean 6.39%).
8. **Absolute robustness:** Worst-case adversarial accuracy is about 35–36% on 5k (seed range 0.6 pp); AutoAttack worst-case is about 35% on 1k.
9. **Deployment caution:** For scientific benchmarking only; not for safety-critical or security-sensitive use.

---

## Open items before release (delete this section when closed)
1. Cite the exact prereg JSON key and commit/date that designated seed 17 before test scores.
2. Confirm the ff_112 batch size per optimizer step from the launcher argument or training log.
3. State the harness and AutoAttack epsilon and clamp conventions.
4. Provenance: show that training commit `512a562` has the same attack step size, loss weights, clamp, LR groups and augmentation as the code described above.
5. Attach the raw `curriculum_for_phase_epoch('ff_112', e)` table.
6. Confirm the Hugging Face path, upload, and state the licence terms (DINOv2 weights licence; ImageNet terms of use).