# Model Card: RHAN-NXA-A, Variant Feedforward-112

> [!IMPORTANT]
> **Status:** DRAFT — AutoAttack results filled from T5 battery (2026-11-02). Awaiting final human review before upload.

---

## Model Details
* **Official Model Name:** `RHAN-NXA-A, variant feedforward-112`
* **Architecture:** Compact Vision Transformer (`CompactViTGen2`) feedforward trunk with DINOv2-small warm-start, LayerScale, and patch projection ($14 \times 14$ patches).
* **Official Checkpoint:** `foundation_ff_112_seed17_final_d00fb750.pth` (Designated in advance prior to test scores; SHA8 = `d00fb750`).
* **Framework:** PyTorch / torchvision / timm.
* **Input Resolution:** $112 \times 112 \times 3$ RGB.
* **Normalization:** ImageNet mean $(0.485, 0.456, 0.406)$, std $(0.229, 0.224, 0.225)$.
* **Classes:** 100 classes (ImageNet-100 benchmark).
* **Release Date:** November 2026.
* **Repository / Checkpoint Archive:** `FerrariKazu/rhan-nxa-g2-checkpoints` (`closure/stage0/`).

---

## Intended Use & Threat Model
* **Intended Use:** High-throughput, robust visual representation baseline for adversarial cognitive modeling research. Serves as the canonical feedforward control against recurrent and foveated vision architectures.
* **Threat Model:**
  * **Perturbation Norm:** $L_\infty$ bounded.
  * **Perturbation Budget:** $\varepsilon = 5.417 / 255 \approx 0.0212431$.
  * **Adversarial Objective:** Untargeted misclassification.
  * **Evaluation Surface:** Canonical ImageNet-100 validation set (5,000 images), evaluated at native $112 \times 112$ resolution.
* **Out-of-Scope:** Not intended for safety-critical deployment or arbitrary out-of-distribution inputs without domain adaptation.

---

## Training Procedure & Recipe
Sourced directly from checkpoint configurations and training logs:
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
* **Batch Size:** Effective batch size 64 (micro-batch size 16 with 4 gradient accumulation steps).
* **Seeds Trained:** Seeds 17, 42, 99.

---

## Quantitative Performance Summary (3-Seed Distribution)

Evaluations performed on the canonical 5,000-image ImageNet-100 validation set ($\varepsilon = 5.417/255$):

| Metric | Designated Official (Seed 17) | 3-Seed Mean | 3-Seed Range (Min – Max) | Data Source |
|---|---|---|---|---|
| **Clean Accuracy (5k full val)** | **74.30%** | **74.49%** | $[74.30\%,\; 74.60\%]$ | `stage0_harness_results.json` |
| **Worst-Case Robust (5k)** | **36.04%** | **35.79%** | $[35.44\%,\; 36.04\%]$ | `stage0_harness_results.json` |
| **PGD-50 (5k)** | 38.18% | 37.55% | $[37.00\%,\; 38.18\%]$ | `stage0_harness_results.json` |
| **APGD-CE (5k)** | 37.66% | 37.01% | $[36.52\%,\; 37.66\%]$ | `stage0_harness_results.json` |
| **APGD-DLR (5k)** | 38.10% | 37.95% | $[37.40\%,\; 38.34\%]$ | `stage0_harness_results.json` |
| **Clean Accuracy (1k slice)** | **73.90%** | **74.00%** | $[73.90\%,\; 74.20\%]$ | `t5_feedforward_battery.json` |
| **ECE (1k)** | **6.52%** | **6.39%** | $[6.15\%,\; 6.52\%]$ | `t5_feedforward_battery.json` |
| **PGD-50 (1k)** | 37.90% | 37.30% | $[36.50\%,\; 37.90\%]$ | `t5_feedforward_battery.json` |
| **PGD-100 (1k)** | 37.60% | 37.10% | $[36.30\%,\; 37.60\%]$ | `t5_feedforward_battery.json` |
| **PGD Gap (1k)** | 0.30 pp | 0.20 pp | $[0.10\text{ pp},\; 0.30\text{ pp}]$ | `t5_feedforward_battery.json` |
| **Training Monitor Parity Gap** | **0.00 pp** | **0.01 pp** | $[0.00\text{ pp},\; 0.02\text{ pp}]$ | `stage0_harness_results.json` |

### AutoAttack Standard (1k canonical slice, T5 battery)

| Attack | Seed 17 (Official) | Seed 42 | Seed 99 | 3-Seed Mean | Source |
|---|---|---|---|---|---|
| **APGD-CE** | 37.40% | 36.90% | 36.30% | 36.87% | `t5_feedforward_battery.json` |
| **APGD-T** | 35.00% | 35.80% | 35.00% | 35.27% | `t5_feedforward_battery.json` |
| **FAB-T** | 35.40% | 36.00% | 35.40% | 35.60% | `t5_feedforward_battery.json` |
| **Square** | 52.30% | 51.20% | 50.60% | 51.37% | `t5_feedforward_battery.json` |
| **AA Worst-Case** | **35.00%** | **35.80%** | **34.80%** | **35.20%** | `t5_feedforward_battery.json` |

### Epsilon Sweep (1k canonical slice, T5 battery)

| ε Factor | ε (in /255) | Seed 17 | Seed 42 | Seed 99 | 3-Seed Mean | Source |
|---|---|---|---|---|---|---|
| 0.5× | 2.71 | 57.30% | 56.80% | 56.80% | 56.97% | `t5_feedforward_battery.json` |
| 1.0× | 5.42 | 38.00% | 37.50% | 36.50% | 37.33% | `t5_feedforward_battery.json` |
| 2.0× | 10.83 | 10.40% | 10.70% | 11.30% | 10.80% | `t5_feedforward_battery.json` |

---

## Computational Cost & Throughput
Measured on NVIDIA GeForce RTX 4060 (8GB VRAM) in FP32:
* **Batch Processing (Batch Size 48):** $25.15\text{ ms}$ per batch ($\approx 1,908\text{ images/second}$).
* **Single-Image Latency:** $3.0\text{ ms}$ forward pass ($\approx 333\text{ images/second}$ sequential throughput).
* **Parameter Count:** $\approx 22.0\text{M}$ parameters (DINOv2-small backbone + patch projection).

---

## Artifact Checksums
* `foundation_ff_112_seed17_final_d00fb750.pth`:
  `d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc`
* `foundation_ff_112_seed42_final_2920df54.pth`:
  `2920df54fe919a8e57dcca8788b5429d0d5f9622fb867723a4d248b063276f71`
* `foundation_ff_112_seed99_final_abde98a0.pth`:
  `abde98a0129f5637039eb98307b30862a04a3cffd0c4b749c8e5cf40c5d4ed4d`
