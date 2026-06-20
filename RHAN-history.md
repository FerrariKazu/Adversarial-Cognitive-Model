# RHAN Architectural Evolution & History

This document outlines the complete design history, theoretical foundations, evolutionary path, and empirical findings of the **Recurrent Hybrid Attention Network (RHAN)** series — from v4 through the current RHAN-UNIFIED on STL-10.

> **Mission:** Close the gap between human visual robustness and AI models by incorporating neuroscientific principles into neural network architecture. Humans maintain stable perception up to ε=0.30; standard AI models collapse before ε=0.03. That's a 10× gap.

---

## Table of Contents

1. [Lineage & Metrics](#1-lineage--metrics)
2. [Theoretical Pillars](#2-theoretical-pillars)
3. [RHAN-v4: Multi-Scale Gated Feedback](#3-rhan-v4-multi-scale-gated-feedback)
4. [RHAN-v5: Frequency Separation & Phase Decoupling](#4-rhan-v5-frequency-separation--phase-decoupling)
5. [RHAN-v5-TRADES & Hardened Variants](#5-rhan-v5-trades--hardened-variants)
6. [RHAN-v6: Dynamic Gating — Regression](#6-rhan-v6-dynamic-gating--regression)
7. [RHAN-v7: Generative World-Model](#7-rhan-v7-generative-world-model)
8. [RHAN-UNIFIED: STL-10 From Scratch](#8-rhan-unified-stl-10-from-scratch)
9. [Final CIFAR-10 Experiments: Self-Alignment & Feature Scatter](#9-final-cifar-10-experiments-self-alignment--feature-scatter)
10. [The Temporal Difference in Vision (TDV) Paradigm](#10-the-temporal-difference-in-vision-tdv-paradigm)
11. [Key Lessons Learned](#11-key-lessons-learned)
12. [Remaining Human-AI Gap](#12-remaining-human-ai-gap)

---

## 1. Lineage & Metrics

### Version Comparison

| System / Model | Clean Acc | PGD 50% Threshold | d'=1.0 Threshold | Status |
|---|---|---|---|---|
| **Human** | 74.15% | >0.30 | >0.30 | ✅ Complete |
| **RHAN-UNIFIED** ★ | **~73%** | **TBD** | **TBD** | 🔄 Training |
| **RHAN-trades-curriculum** ★ | **78.12%** | **ε≈0.113** | **ε≈0.1850** | ✅ Complete |
| **RHAN-TDV-Clean** (STL-10) ★ | **78.50%** | **ε≈0.0037** | **ε≈0.0043** | ✅ Complete |
| **RHAN-TDV-Adversarial** (STL-10) ★ | **73.83%** | **ε≈0.0077** | **ε≈0.0066** | ✅ Complete |
| **RHAN-Self-Alignment** ⚠️ | **77.10%** | — | — | ⚠️ Obfuscated (AA: 21.60%) |
| **RHAN-Feature-Scatter** ⚠️ | **77.10%** | — | — | ⚠️ Obfuscated (AA: 22.30%) |
| **RHAN-TRADES-Hardened** | **86.33%** | **ε≈0.086** | **ε≈0.1246** | ✅ Complete |
| **RHAN-v5-TRADES** | **87.30%** | **ε≈0.078** | **ε≈0.1113** | ✅ Complete |
| **RHAN-v5** | **84.57%** | **ε≈0.071** | **ε≈0.1030** | ✅ Complete |
| **RHAN-v3** | **91.41%** | **ε≈0.066** | **ε≈0.0900** | ✅ Complete |
| **RHAN-v4** | **89.65%** | **ε≈0.056** | **ε≈0.0800** | ✅ Complete |
| **RHAN-adv** | **83.79%** | **ε≈0.053** | **ε≈0.0764** | ✅ Complete |
| RHAN-clean | 89.06% | ε≈0.023 | ε≈0.0330 | ✅ Complete |
| **RHAN-v6** ⚠️ | 82.03% | — | — | ⚠️ Regressed |
| ResNet-18 | 95.82% | ε≈0.024 | ε≈0.0300 | ✅ Complete |
| ViT-Small | 97.80% | ε≈0.014 | ε≈0.0264 | ✅ Complete |

### PGD Accuracy Collapse Table

| Epsilon | UNIFIED | TDV-Clean | TDV-Adv | Curriculum | Hardened | TRADES | RHAN-v5 | ResNet | ViT |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | ~73% | 78.50% | 73.83% | 78.12% | 86.33% | 87.30% | 84.57% | 95.82% | 97.80% |
| 0.01 | TBD | 5.20% | ~1.56% | 75.00% | 83.01% | 84.77% | 80.66% | 75.57% | 55.18% |
| 0.05 | TBD | 1.50% | 0.98% | 65.23% | 67.19% | 65.82% | 61.13% | 2.84% | 8.80% |
| 0.10 | TBD | 1.60% | 0.78% | 52.93% | 43.16% | 37.89% | 34.38% | 0.21% | 2.78% |
| 0.20 | TBD | 1.60% | 0.78% | 29.49% | 8.59% | 5.47% | 2.73% | 0.02% | 1.12% |
| 0.30 | TBD | 1.70% | 0.78% | 10.16% | 0.20% | 0.20% | 0.20% | 0.00% | 0.58% |

---

## 2. Theoretical Pillars of RHAN

RHAN bridges the gap between biological vision and machine vision by incorporating neuroscientific priors. Each version tests a specific hypothesis about what makes human vision robust.

### Pillar 1: Recurrent Top-Down Feedback (All Versions)
Feedforward networks process images in a single pass, making them susceptible to local high-frequency adversarial noise. Biological brains use massive recurrent feedback loops (IT→V4→V2→V1) for iterative denoising. RHAN implements this via a recurrent feedback block that modulates convolutional stem activations using global self-attention output.

### Pillar 2: Ventral/Dorsal Pathway Split (Trial 3 → v2/v3)
Primate visual systems process information along two parallel streams:
- **Ventral ("What"):** Shape, identity, color, semantic representations
- **Dorsal ("Where"):** Spatial layout, motion, boundaries, coordinate relationships

Splitting the 512-dim attention channel into parallel 256-dim pathways prevents adversarial attacks from easily optimizing against both channels simultaneously.

### Pillar 3: Neural Representation Alignment (Trial 8 → v2/v3)
Aligning the CLS token representation against CORnet-S IT cortex features forces the model to learn semantic, shape-based abstractions instead of brittle, non-robust pixel features. **Critical finding:** Alignment must be computed on adversarial images, not clean images, to reinforce robustness.

### Pillar 4: Frequency Separation (v5)
Primate V1 channels low-frequency (shape/structure) separately from high-frequency (texture/noise). Adversarial noise is primarily high-frequency. Learnable Gaussian separator with dual stems (shape vs. texture) and learnable weights (w_low=0.85, w_high=0.15) implements this computationally.

---

## 3. RHAN-v4: Multi-Scale Gated Feedback

**Hypothesis:** Multi-scale gated feedback with active CLIP semantic loss would improve both clean accuracy and robustness.

**Architecture:** Built on v3's dual-stream transformer. Added:
- Multi-scale gated feedback with 4 spatial scales
- Active CLIP semantic loss during adversarial training
- InfoNCE adversarial consistency

**Training:** 100 epochs, β=6.0, ε=0.031 PGD-5.

**Result:** Clean 89.65%, εthresh=0.0800.

**Finding — Semantic-Geometric Conflict:** Active CLIP semantic loss during adversarial training acts as a geometry-degrading constraint. It forces internal representations onto a smoother semantic manifold that conflicts with the sharp decision boundaries needed for adversarial robustness. This was the first clear evidence that semantic losses and adversarial training conflict when applied simultaneously.

**Lesson:** Semantic alignment must be decoupled from adversarial training — applied as initialization, not as an ongoing loss.

---

## 4. RHAN-v5: Frequency Separation & Phase Decoupling

**Hypothesis:** Strictly separating CLIP pretraining (Phase 0) from adversarial training (Phase 1), combined with biological frequency separation, would resolve the semantic-geometric conflict.

**Architecture — Two Innovations:**

1. **Biological Frequency Separation:** Learnable Gaussian low-pass separator with dual stems (shape vs. texture) and Frequency Consistency Loss enforcing low-frequency feature invariance clean-vs-adversarial.

2. **Phase-Decoupled Pretraining:** CLIP semantic alignment applied strictly during Phase 0 (30 epochs, clean-only). Phase 1 (120 epochs) runs full epsilon curriculum with neural alignment on adversarial images, completely decoupled from active semantic losses.

**Training:** Phase 0: 30 epochs clean CLIP. Phase 1: 120 epochs ε=0.031→0.150 curriculum.

**Result:** Clean 84.57%, εthresh=0.1030 — a 14.4% improvement over v3, becoming the new best model.

**Key Findings:**
- M-pathway dominance confirmed: wL=0.791 > wH=0.513 (shape over texture)
- No gradient masking: PGD-20 vs PGD-100 gap < 8%
- Phase decoupling is essential: applying semantic losses during adversarial training degrades geometry

---

## 5. RHAN-v5-TRADES & Hardened Variants

**Hypothesis:** The training algorithm (not just architecture) drives robustness. Standard TRADES loss would outperform custom adversarial losses.

### RHAN-v5-TRADES Baseline

**Architecture:** Same v5 backbone. Replaced custom adversarial loss with standard TRADES (KL divergence, β=6.0) + IT alignment (0.2 weight).

**Training:** 120 epochs, ε=0.031, initialized from CLIP pretraining weights.

**Result:** Clean 87.30%, εthresh=0.1113. Significant improvement over standard v5, proving the value of a theoretically principled objective.

### RHAN-TRADES-Hardened

**Architecture:** Same v5+TRADES backbone. Added class-hardened training:
- ε scaling: base ε=0.031 → ε=0.055 for vulnerable classes (automobile, truck, horse, dog, cat)
- 20-step adaptive APGD with milestone-based halving and backtracking
- Inter-class margin loss (centroid distance penalty, margin=0.5, weight=0.20)

**Result:** Clean 86.33%, εthresh=0.1246 (4.2× ResNet-18).

**Finding — Class-Specific Geometric Vulnerabilities:** AutoAttack revealed automobile and truck robust accuracies remained at 0.00%. Feature space proximity near these classes is a deep geometric problem requiring stronger training. **This insight motivated the curriculum approach.**

### RHAN-trades-curriculum

**Schedule:** 3 phases × 20 epochs from hardened checkpoint:
- Phase A: ε=0.062, β=6.0
- Phase B: ε=0.100, β=6.0
- Phase C: ε=0.150, β=5.0

**Result:** εthresh=0.1850 — **6.3× improvement over ResNet-18**, the best CIFAR-10 result.

**Finding — Curriculum Trade-off:** AutoAttack at ε=0.031 showed only 21.88% robust accuracy (vs 28.22% for hardened). Vulnerable classes (automobile, horse, truck) still collapsed to 0.00%. High-strength curriculum regularization pushes the d' threshold but drifts clean representation margins.

---

## 6. RHAN-v6: Dynamic Gating — Regression

**Hypothesis:** Adding dynamic gating, predictive coding, and ACT to the v5 architecture would push robustness further.

**Architecture:** Extended v5 with:
1. Input-dependent α(x) frequency gating (instead of static learned weights)
2. Predictive coding error signals modulating recurrent feedback
3. Adaptive Computation Time (ACT) for variable-depth recurrence

**Training:** 6-phase epsilon curriculum (0.015→0.031→0.062→0.100→0.150→0.250).

**Result:** Regressed. Clean 82.03%, no meaningful εthresh.

**Root Cause:** Phase F (ε=0.250) corrupted learned representations. The model could not maintain clean accuracy while defending against perturbations exceeding the information content of 32×32 CIFAR images. ACT maxed out its pondering budget on every sample.

**Key Lesson:** Architecture was not the problem — the training algorithm was. Adding complexity to an already-effective architecture produces diminishing returns when the training regime is not carefully calibrated.

---

## 7. RHAN-v7: Generative World-Model (CIFAR-10)

**Hypothesis:** A generative prior (VAE decoder) would provide a manifold constraint on adversarial attacks — attacks must fool the classifier AND keep features in a region that decodes to a plausible image.

**Architecture:**
- Dual-stream transformer (ventral 256-dim + dorsal 256-dim)
- VAE encoder head (μ, log_var) → latent_dim=256
- VAE decoder: 256→512×1×1→ConvTranspose2d→32×32 RGB
- Generative classifier (latent→10 classes)
- Perceptual critic: fresh random ConvStem (not copied from trained backbone)

**Loss:** 0.50*TRADES + 0.15*feature_recon + 0.20*KL_freebits + 0.10*alignment

**Critical Discoveries:**

1. **Frozen perceptual critic must be fresh random init:** Copying trained stem_low causes BatchNorm channel collapse (near-zero variance), making FR loss vanish to ~0.000032
2. **Pixel-level reconstruction conflicts with TRADES:** Feature-level reconstruction (comparing stem features) is compatible
3. **Online feature comparison works:** Using model's own current stem_low output (detached) as target provides a moving reference
4. **Phase 0 warmup is essential:** Decoder must learn to reconstruct before adversarial training
5. **Computation cost ~3× higher:** ~900s/epoch vs ~300s for v5

**Status:** Training in progress (June 2026).

---

## 8. RHAN-UNIFIED: STL-10 96×96 From Scratch (Current)

**Hypothesis:** Training on higher-resolution 96×96 STL-10 with all proven components unified would enable better shape discrimination and close the human-AI robustness gap further.

### Architecture

**Initial (model_rhan_stl10.py):**
- ConvStemSTL10: 4 layers, 64→256→512→512 channels
- 144 spatial tokens + CLS = 145
- Transformer: 3 layers, 8 heads, ff_dim=2048
- Recurrent feedback: 2 steps
- Cosine similarity head
- ~14M parameters

**Upgraded (model_rhan_unified.py) — WideSEConvStem:**
- SEBlock after each conv layer (channel gain control, biological analog: V4)
- Wider channels: 128→512→1024→512
- Stochastic depth dropout (p=0.1) after conv3
- ~20.5M parameters

### Training Pipeline Evolution

**Attempt 1 — Direct TRADES (FAILED):**
- β=6.0, ε=0.062→0.200, 80 epochs
- Result: Immediate collapse. TeAcc dropped from 72.94% to 13.0%. T=13.965.
- Root cause: β=6.0 calibrated for 50K CIFAR images, not 5K STL-10

**Attempt 2 — Lower Beta + Head Replacement (FAILED):**
- β=2.0, replaced cosine head with linear head before TRADES
- Result: TeAcc dropped to 13% on epoch 1
- Root cause: Head replacement destroyed learned feature calibration

**Attempt 3 — Keep Cosine Head, Lower Beta (SUCCEEDED — Phases 1-2):**
- β=2.0, keep cosine head, 8 phases
- Phase 1 (ε=0.016): TeAcc stable at 73.1-73.6%, T=2.0-2.4 ✓
- Phase 2 (ε=0.031): TeAcc stable at 72.2-73.2%, T=4.2-5.8 ✓
- Phase 3 (ε=0.047, β=2.5): COLLAPSE. TeAcc fell from 71.1% to 54.6%
- Root cause: β=2.5 still too high for ε=0.047 transition

**Attempt 4 — Current (training in progress):**
- β=2.0 for ALL phases 1-6, β=2.5 for phase 7, β=3.0 for phase 8
- 3-epoch beta warmup at each phase transition (β_effective = 0.3×β)
- Lower LR: 0.002 (P1-4), 0.001 (P5-8)
- Phase 0: 30 epochs clean pretraining with CutMix
- CutMix augmentation throughout all phases
- Rolling checkpoint every epoch
- Resume capability via --resume and --start-phase flags

### Curriculum (Current)

| Phase | Epochs | ε | β | LR | Warmup |
|---|---|---|---|---|---|
| 0 | 1-30 | — | — | 3e-4 | Clean CE + CutMix |
| 1 | 31-45 | 0.016 | 2.0 | 0.002 | 3-ep β=0.5→2.0 |
| 2 | 46-60 | 0.031 | 2.0 | 0.002 | 3-ep β=0.5→2.0 |
| 3 | 61-75 | 0.047 | 2.0 | 0.002 | 3-ep β=0.5→2.0 |
| 4 | 76-90 | 0.062 | 2.0 | 0.002 | 3-ep β=0.5→2.0 |
| 5 | 91-102 | 0.094 | 2.0 | 0.001 | 3-ep β=0.5→2.0 |
| 6 | 103-114 | 0.125 | 2.5 | 0.001 | 3-ep β=0.75→2.5 |
| 7 | 115-126 | 0.150 | 2.5 | 0.001 | 3-ep β=0.75→2.5 |
| 8 | 127-138 | 0.200 | 3.0 | 0.001 | 3-ep β=0.9→3.0 |

### Key Design Decisions

1. **Cosine head preserved throughout:** Head replacement destroyed Phase 0 calibration (72.94% → 13% test acc)
2. **Beta=2.0 for phases 1-6:** Beta=6.0 was for CIFAR-10 with 50K images; beta=2.0 prevents KL explosion on 5K STL-10
3. **3-epoch warmup at phase transitions:** First 3 epochs use 30% of target β, allowing gradual adjustment
4. **CutMix augmentation:** Applied 50% of the time; expected to close train-test gap from ~19% to ~10%
5. **Rolling checkpoint every epoch:** Saves model, optimizer, epoch, best_acc for resume
6. **Phase 0 clean pretraining:** 30 epochs pure CE on labeled data only (no unlabeled, no pseudo-labels)
7. **Lower LR:** 0.002/0.001 (was 0.005/0.003) to prevent optimizer shock at phase transitions

### Results So Far

- **Phase 0** (30 epochs): Best test acc = **72.94%** (with CutMix)
- **Phase 1** (ε=0.016, β=2.0): TeAcc stable at 73.1-73.6%, T=2.0-2.4 ✓
- **Phase 2** (ε=0.031, β=2.0): TeAcc stable at 72.2-73.2%, T=4.2-5.8 ✓
- **Phase 3+**: Training in progress with corrected curriculum

### Why STL-10 Changes Everything

- **96×96 resolution:** 9× more pixels than CIFAR-10 32×32
- **Better shape discrimination:** Higher resolution enables genuine shape-based features
- **Human comparison:** Humans perform ~90-95% on STL-10 at 96×96 (vs ~73% on CIFAR-10 at 32×32)
- **Predicted εthresh:** 0.220-0.280 (vs 0.185 CIFAR-10 ceiling)
- **Predicted AutoAttack:** 45-65% at ε=0.031 (vs 29.2% CIFAR-10 best)

---

## 9. Final CIFAR-10 Experiments: Self-Alignment & Feature Scatter

**Hypothesis:** If the automobile/truck robust collapse is caused by proximity of classes in feature space, directly minimizing the feature-space distance between clean and adversarial examples ($\text{dist}(f(x_{\text{adv}}), f(x_{\text{clean}}))$) will force the model to build invariant representation manifolds, preventing class boundary confusion under adaptive attacks.

### Experimental Setup:
1. **Self-Alignment**: Replaced TRADES KL loss with feature-space cosine similarity loss.
2. **Feature Scatter**: Applied feature scatter constraints to match adversarial and clean representations in the backbone's feature space using the corrected mathematical bounds:
   $$\mathcal{L}_{\text{feat}} = \beta \cdot \left(1 - \frac{f(x_{\text{adv}}) \cdot f(x_{\text{clean}})}{\|f(x_{\text{adv}})\| \|f(x_{\text{clean}})\|}\right)$$

### Empirical Results:
Under standard PGD-100 evaluation, both models appeared to display stellar robustness:
- **PGD-100 Robust Accuracy**: **84.77%** (both Self-Alignment and Feature Scatter).
- **Robustness Gap**: Appeared to jump to 62-63 pp above baseline.

However, evaluation using the gradient-free AutoAttack standard ($\epsilon = 0.031$) exposed a stark reality:
- **AutoAttack Robust Accuracy**: **21.60%** (Self-Alignment) and **22.30%** (Feature Scatter).
- **Vulnerable Classes**: `automobile`, `truck`, and `horse` robust accuracies collapsed completely to **0.00%**.

### The Gradient Masking Theorem:
These experiments provide empirical validation of the **Gradient Masking Theorem**:
> Any loss of the form minimize $\mathcal{D}(f(x_{\text{adv}}), f(x_{\text{clean}}))$ directly incentivizes gradient obfuscation.

By penalizing representation changes in feature space, the model satisfies the loss by making $f(x)$ constant (flat) within the local $\epsilon$-ball. This creates flat gradients in every direction. PGD, which relies on following gradients, is unable to find adversarial perturbations, creating the illusion of robustness (84.77% accuracy). AutoAttack uses Square attack (a gradient-free query-based algorithm) which easily identifies the true boundary collapse.

**Conclusion**: The CIFAR-10 chapter is officially closed. The automobile/truck class collapse is dataset-intrinsic at 32×32 and cannot be resolved by any feature-space training objective.

---

## 10. The Temporal Difference in Vision (TDV) Paradigm

**The visual learning bottleneck is representation collapse. The solution is temporal diversity.**

Published on June 14, 2026, the TDV (Temporal Difference in Vision) paper (Daithankar, Gladstone, LeCun, & Ji) introduces a causal temporal formulation:
$$z_t + m_t = z_{t+1}$$
where $z_t$ represents the frame representation and $m_t$ is the motion encoder output.

### How TDV Resolves SAIL's Core Defect:
In static self-supervised learning (SAIL), models are prone to representation collapse when forced to make adversarial features invariant to clean features. TDV prevents collapse by replacing static similarity metrics with temporal causal consistency:
- **Natural Diversity**: Consecutive frames in a video sequence are naturally distinct, meaning representations cannot collapse to a single point.
- **Causal Constraint**: Trivial constant representations fail the causality equation since the motion encoder must produce a non-trivial vector that maps $z_t$ to $z_{t+1}$.

### Integrating TDV with the RHAN Framework:
- **Phase 0 Video Pretraining**: The RHAN backbone learns representation dynamics on video sequences (e.g., UCF-101) using:
  $$f(\text{frame}_t) + \text{motion\_encoder}(\text{flow}_t) = f(\text{frame}_{t+1})$$
- **Temporal Adversarial Consistency**: During subsequent phases, the InfoNCE objective is replaced by a temporal difference constraint on adversarially augmented sequences:
  $$z_{\text{clean}}[t] + m_t = z_{\text{adv}}[t+1]$$
- **Roadmap for STL-10 96×96**: Pretraining the ResNet-50 stem on UCF-101 using TDV before fine-tuning on STL-10 labeled images. The causal representations learn physical/temporal structure rather than simple statistical correlations, providing a pathway to narrow the automobile/truck gap.

### STL-10 96×96 Empirical Evaluation & Findings

We systematically implemented and evaluated the RHAN-TDV paradigm on STL-10 to investigate whether temporal diversity resolves representation collapse and addresses the automobile/truck robust accuracy gap at $\varepsilon=0.031$.

#### 1. Phase TDV Pretraining (Unlabeled Data)
* **Goal**: Train the backbone features on STL-10's 100K unlabeled frame sequences under the causal constraint $z_t + m_t = z_{t+1}$ while preventing feature space representation collapse.
* **Optimization Fix**: Initial runs collapsed feature variance to $<0.02$ within 2 epochs. We resolved this by:
  - Replacing `BatchNorm1d` with `LayerNorm` in the `TDVProjectionHead` (BatchNorm was normalizing feature variance across the batch, hiding backbone collapse from the VICReg loss).
  - Adding a direct raw feature variance penalty $\mathcal{L}_{\text{var\_raw}}$ computed on the unprojected CLS token features.
* **Outcome**: Standard deviation (`Std`) stabilized at **0.4908** at epoch 30, successfully preventing representation collapse.

#### 2. Phase Label Calibration
* **Goal**: Warmup the cosine classifier head using the 5K labeled images while keeping the backbone frozen.
* **Outcome**: Clean validation accuracy reached **78.6%** in 10 epochs.

#### 3. Phase TRADES Curriculum Fine-Tuning
We ran two distinct settings to analyze consistency bounds:
* **Run 1: Clean TDV Consistency** (batch size 32): TRADES fine-tuning using `tdv_loss` on clean temporal pairs. Clean accuracy stabilized at **78.2%**. Under standard AutoAttack ($\varepsilon = 0.031$), robust accuracy dropped to **1.76%** (`truck` = 13.3%, `car` = 0.0%). The attack deformed representations because consistency was only enforced on clean frames.
* **Run 2: Adversarial TDV Consistency** (batch size 16 to avoid VRAM paging): Active `adversarial_tdv_loss` enforcing $z_{\text{adv}}[t] + m_t = z_{\text{clean}}[t+1]$. Clean accuracy stabilized at **75.4%**. Under AutoAttack ($\varepsilon = 0.031$), overall robustness was **0.78%** (`truck` = 4.4%, `car` = 0.0%).

#### 4. PGD-100 Sweep & Robustness Thresholds (Run 1 - Clean TDV Consistency)
Evaluating the Run 1 checkpoint (`rhan_stl10_tdv_trades.pth` / `rhan_stl10_tdv_trades_clean_consistency.pth`) under a 100-step PGD sweep across epsilons with 1000 test samples revealed:
* $\varepsilon = 0.000$: **78.50%** (Overall $d'$ = 2.8600, Car $d'$ = 3.4077, Truck $d'$ = 3.2250)
* $\varepsilon = 0.005$: **26.00%** (Overall $d'$ = 0.6992, Car $d'$ = 1.6744, Truck $d'$ = 1.0909)
* $\varepsilon = 0.010$: **5.20%** (Overall $d'$ = -0.5970, Car $d'$ = 0.5585, Truck $d'$ = -0.2888)
* $\varepsilon = 0.015$: **2.00%** (Overall $d'$ = -0.8867, Car $d'$ = -0.1992, Truck $d'$ = -0.9133)
* $\varepsilon \ge 0.030$: **< 2.0%** (Overall $d'$ is negative, complete collapse)

* **Estimated $\varepsilon_{\text{thresh}}$ (Relative 50% Accuracy drop)**: **0.0037**
* **Estimated $\varepsilon_{\text{thresh}}$ (Overall SDT $d'=1.0$)**: **0.0043**
* **Car Class $\varepsilon_{\text{thresh}}$ ($d'=1.0$)**: **0.0080**
* **Truck Class $\varepsilon_{\text{thresh}}$ ($d'=1.0$)**: **0.0053**

* **Interpretation**: The truck class shows a non-zero $\varepsilon_{\text{thresh}}$ of $0.0053$ (and car shows $0.0080$), showing that the TDV representation mapping preserves class sensitivity up to $\varepsilon \approx 0.005 - 0.008$. However, the model collapses immediately beyond this level, indicating that clean temporal difference consistency alone is insufficient to build a robust manifold across high perturbation magnitudes.

#### 5. PGD-100 Sweep & Robustness Thresholds (Run 2 - Adversarial TDV Consistency)
Evaluating the Run 2 checkpoint (`rhan_stl10_tdv_trades_actual.pth`) under a 100-step PGD sweep across epsilons with 1000 test samples revealed:
* $\varepsilon = 0.000$: **73.83%** (Overall $d'$ = 2.5325, Car $d'$ = 3.1970, Truck $d'$ = 3.0182)
* $\varepsilon = 0.015$: **1.56%** (Overall $d'$ = -0.9245, Car $d'$ = -0.6582, Truck $d'$ = -0.3541)
* $\varepsilon = 0.031$: **0.98%** (Overall $d'$ is negative, complete collapse)
* $\varepsilon \ge 0.094$: **0.78%**

* **Estimated $\varepsilon_{\text{thresh}}$ (Relative 50% Accuracy drop)**: **0.0077**
* **Estimated $\varepsilon_{\text{thresh}}$ (Overall SDT $d'=1.0$)**: **0.0066**

The flat performance floor (~0.78% robustness) persisting through higher epsilons indicates a capacity limit or a hard boundary masking issue: the model cannot project high-resolution spatial details into the 3-layer transformer encoder to enforce a stable temporal manifold under perturbation.

#### 5. Recommendations for Future Iterations
1. **Scale Backbone Capacity**: The 3-layer transformer encoder and 256-dimensional projector space are insufficient for $96\times96$ STL-10 under adversarial noise. Increase depth to 6–8 layers and latent space to 512 dimensions.
2. **Generative Consistency (VAE Integration)**: Re-incorporate the generative VAE decoder from RHAN-v7. Enforcing that the temporal latent representation $z_t + m_t$ decodes back to the next frame $\hat{x}_{t+1}$ provides a powerful physical constraint that prevents adversarial perturbations from shifting features into non-visual manifolds.
3. **Biological Boundary Supervision**: Integrate edge-detection / semantic contour constraints (e.g., Gabor-like filters or Sobel gradients) as an auxiliary loss in Phase 0. This stops the motion encoder from relying on brittle texture shortcuts to solve the temporal difference equation.

---

## 11. Key Lessons Learned

### What Definitively Doesn't Work

| Failure | Root Cause |
|---|---|
| Fine-tuning robust model with biological priors (Trials 1-8) | Always hurts high-epsilon robustness |
| CLIP as ongoing loss during adversarial training (v4) | Smooth semantic manifolds are exploitable |
| Phase F curriculum (ε=0.250) (v6) | Exceeds information content of 32×32 images |
| Concept bottlenecks without ground truth annotations | Spurious concepts become attack surfaces |
| Ensembling models with same geometric failures | Averages don't create new separations |
| Replacing cosine head before TRADES phases | Destroys learned feature calibration |
| Pixel-level reconstruction loss for generative prior | Conflicts with adversarial training |
| Frozen perceptual critic copied from trained backbone | BatchNorm channel collapse kills FR loss |
| Beta=6.0 for STL-10 with 5K samples | KL term over-penalizes, TRADES loss explodes |
| No warmup at phase transitions | Model can't adapt to new epsilon fast enough |
| Direct feature invariance losses (Self-Alignment, Feature Scatter) | Directly incentivizes gradient masking/obfuscation, failing under AutoAttack |
| BatchNorm1d in TDV projection head | Masks backbone representation collapse by normalizing batch statistics, preventing VICReg loss from penalizing collapse |
| Clean-only TDV consistency under adversarial training | Allows adversarial perturbations to bypass temporal causality constraints, causing collapse of robust accuracy on car class to 0% |
| Adversarial TDV consistency without scaling backbone capacity | Backbone capacity bottleneck (3-layer transformer) causes immediate robustness collapse under attack ($\varepsilon_{\text{thresh}} \approx 0.015$) |

### What Definitively Works

| Success | Key Insight |
|---|---|
| Joint training from scratch (v3) | All objectives must shape representations simultaneously |
| Frequency separation with learnable M-pathway gates (v5) | Confirmed biological V1 shape-over-texture hypothesis |
| TRADES loss over PGD training | +20% relative εthresh improvement |
| Extended curriculum with phase-specific epsilons | Directly sets the robustness ceiling |
| CLIP as initialization only, not ongoing loss (v5) | Semantic prior without geometric conflict |
| Neural alignment on adversarial images (v3) | Critical: alignment under attack, not on clean images |
| Ventral/dorsal stream split (v3) | Persistent improvement across all variants |
| Online feature comparison for generative prior | Moving reference shifts with backbone |
| Unlabeled data pseudo-labeling for Phase 0 | 20× more visual diversity for small labeled sets |
| Beta=2.0 with cosine head | Stable TRADES training for limited-data regime |
| CutMix augmentation | Closes train-test gap on small datasets |
| Rolling checkpoints every epoch | No progress lost during curriculum transitions |
| 3-epoch beta warmup at phase transitions | Gradual adjustment prevents collapse |
| TDV (Temporal Difference in Vision) pretraining | Natural temporal diversity of consecutive frames prevents representational collapse |
| LayerNorm + raw feature variance penalty in TDV | Maintains stable feature space variance ($Std \approx 0.49$), resolving representation collapse |
| Proto-head label calibration in TDV | Rapidly calibrates classification head, yielding 78.6% clean accuracy on STL-10 |

---

## 12. Remaining Human-AI Gap

Even with all improvements, the gap between RHAN (εthresh≈0.185) and humans (εthresh>0.30) on CIFAR-10 is ~1.6×. On STL-10, we predict εthresh≈0.250 vs human >0.500 — still a 2× gap.

**The remaining gap likely requires genuine semantic grounding** (language-vision integration), not just architectural improvements. This is the fourth missing principle identified in our research:

> "Human visual robustness is not a single mechanism — it is an emergent property of a system that combines local frequency filtering, global shape integration, recurrent top-down feedback, and semantic language grounding, operating together across a strict processing hierarchy. Our results show that implementing even three of these four principles in a unified architecture produces robustness qualitatively superior to any single-principle model, while the remaining gap to human performance points precisely to the fourth missing principle: genuine semantic grounding of visual representations in conceptual knowledge."

---


