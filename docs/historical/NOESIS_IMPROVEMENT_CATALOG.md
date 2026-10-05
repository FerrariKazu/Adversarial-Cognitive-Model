# NOESIS Improvement Catalog
## All Vectors for Advancing Toward the Perception Goal

*Last updated: 2026-08-31*
*Status: Living document — update after each experiment cycle*
*Foundation alignment: All recommendations checked against `docs/NOESIS_FOUNDATION.md` (2,863 lines)*

---

## Executive Summary

The project has validated two architectural pillars (AIS + HPC) on STL-10, reaching **D = AIS+HPC at 34.02±3.24% PGD-100** (+9.79pp over TRADES baseline, 16 seeds). E1 (adding recon-mod) shows regression: −0.90pp vs D. Two new experiments are now implemented and ready to train:
- **E3 (D + T=6)**: More foraging steps = more Banach contraction. Config change only.
- **E2 (D + SBR)**: Slot attention replaces flat belief vector. Structured representation.

This document catalogs every identified improvement vector, organized by:
- **Confidence**: validated, in-progress, unexplored, or failed
- **Effort**: quick (days), medium (weeks), or long (months)
- **Expected impact**: high (>5pp PGD-100), medium (2-5pp), or low (<2pp)

---

## Current Baseline Performance (D = AIS+HPC)

| Metric | TRADES Large (A) | D (AIS+HPC) | Δ |
|--------|-----------------|-------------|---|
| Clean acc | 54.84% | 55.13% | +0.29 pp |
| PGD-100 ε=0.094 | 24.03% | **33.17%** | **+9.14 pp** |
| 16-seed PGD-100 | 24.03% | **34.38±1.94%** | **+11.54 pp** |
| p-value | — | ~2×10⁻⁵ | significant |

---

## Category 1: Training Regime Changes

### 1.1 Stronger Adversarial Training (PGD-20 at train time)

| Field | Detail |
|-------|--------|
| **What** | Currently trains with PGD-10 steps; evaluate PGD-20 at training time |
| **Why** | TRADES with PGD-10 can be fooled by attacks with >10 steps; PGD-20 closes this gap |
| **Evidence** | Finding 14 shows PGD-20 at train time improves robustness without masking |
| **Effort** | Medium (2-3 days) — change one training flag, retrain D config |
| **Risk** | ~2× slower per epoch due to doubled PGD steps |
| **Expected** | +2-4pp PGD-100 (eliminates weak-attack overfitting) |
| **Status** | **UNEXPLORED** |
| **Priority** | HIGH — low risk, meaningful improvement |

### 1.2 Higher β in TRADES

| Field | Detail |
|-------|--------|
| **What** | Currently β=2.0→2.5 across curriculum; try β=3.0 or β=4.0 |
| **Why** | Higher β forces stronger KL regularization, pushing the model to maintain output distribution under attack |
| **Evidence** | Finding 12: β=6.0 exploded on STL-10 with 5K samples; β=2.0-2.5 worked. The ceiling hasn't been tested between 2.5 and 6.0 |
| **Effort** | Low (1 day) — change β schedule, retrain |
| **Risk** | β too high causes TRADES loss explosion (Finding 10: β=6.0 failed) |
| **Expected** | +1-3pp PGD-100 if β=3.5-4.0 is feasible |
| **Status** | **UNEXPLORED** (only β=2.0 and β=6.0 tested) |
| **Priority** | HIGH — cheap to test, high potential |

### 1.3 Extended Training with Early Stopping

| Field | Detail |
|-------|--------|
| **What** | Train 150-200 epochs with patience-based early stopping (monitor PGD-10, not clean acc) |
| **Why** | Current 60 epochs may not fully converge the HPC head or AIS gating |
| **Evidence** | D training showed test acc plateauing at ~55%, but PGD-10 robustness was still improving at epoch 60 |
| **Effort** | Low (1 day) — change max_epochs, add early stopping |
| **Risk** | Overfitting on 5K labeled samples (mitigated by 41.6K pseudo-labels) |
| **Expected** | +0-2pp PGD-100 (diminishing returns likely) |
| **Status** | **UNEXPLORED** (only 60 epochs tested) |
| **Priority** | MEDIUM — cheap but likely small gains |

### 1.4 Cosine Annealing with Warm Restarts

| Field | Detail |
|-------|--------|
| **What** | Add warm restarts to the cosine LR schedule (e.g., every 30 epochs) |
| **Why** | Warm restarts help escape sharp minima, which are less robust than flat minima |
| **Evidence** | No restart experiments run; standard cosine schedule used throughout |
| **Effort** | Low (1 day) |
| **Risk** | May destabilize training if restarts are too aggressive |
| **Expected** | +1-2pp PGD-100 (flat minima tend to be more robust) |
| **Status** | **UNEXPLORED** |
| **Priority** | MEDIUM |

### 1.5 Label Smoothing

| Field | Detail |
|-------|--------|
| **What** | Apply label smoothing (ε=0.1) to the classification targets |
| **Why** | Prevents the model from becoming overconfident on clean examples, which reduces gradient masking and improves generalization |
| **Evidence** | Common in robust training literature; not tested in this pipeline |
| **Effort** | Trivial (1 line change) |
| **Risk** | Negligible |
| **Expected** | +0.5-1pp PGD-100 |
| **Status** | **UNEXPLORED** |
| **Priority** | LOW-MEDIUM — easy but small gains |

### 1.6 Data Augmentation During Adversarial Training

| Field | Detail |
|-------|--------|
| **What** | Add RandAugment, CutOut, or AutoAugment during TRADES training |
| **Why** | Standard augmentation increases the diversity of training examples, making the model more robust to distribution shift |
| **Evidence** | Standard practice in adversarial training; not currently used |
| **Effort** | Low (1 day) |
| **Risk** | May interact with PGD attacks (augmented images are harder to attack) |
| **Expected** | +1-3pp PGD-100 |
| **Status** | **UNEXPLORED** |
| **Priority** | HIGH — standard practice, easy to implement |

---

## Category 2: Architecture Changes

### 2.1 More Foraging Steps (T=6, T=8, or T=10)

| Field | Detail |
|-------|--------|
| **What** | Increase max_foraging_steps from 4 to 6 (first), then 8 if T=6 shows improvement |
| **Why** | More steps = more evidence accumulation = more Banach contraction of adversarial noise (γ^T shrinks with T) |
| **Evidence** | Finding 16: AIS halting triggers for only 7.7% of samples; at T=4, the model barely uses the adaptive mechanism. Foundation §3.1: "each step multiplies the perturbation by γ < 1" |
| **T=6 vs T=8 vs T=10** | See analysis below |
| **Effort** | Low (change one config parameter) |
| **Risk** | More memory (O(T) per sample), slower training |
| **Expected** | +2-4pp PGD-100 at T=6; diminishing returns at T=8, T=10 |
| **Status** | **IMPLEMENTED** as Stage 4-E3 (ablation matrix entry E3_ais_hpc_t6, PENDING). Ready to train on Colab/Kaggle. |
| **Priority** | 🔴 CRITICAL — directly tests the framework's core robustness argument |
| **Foundation alignment** | Principle II (iterative inference), §3.1 (Banach contraction), §5.2 (AIS) |

### 2.2 More HPC Levels (hpc_num_levels=2)

| Field | Detail |
|-------|--------|
| **What** | Add a second HPC prediction level (e.g., orientation map) on top of the edge-map level |
| **Why** | Hierarchical predictive coding with multiple levels should capture both low-level (edges) and mid-level (orientations, textures) prediction errors |
| **Evidence** | Level 1 (edge_map) is validated; the roadmap explicitly calls for "one level per validation cycle" |
| **Effort** | Medium (2-3 days) — implement the second predictor, wire into the HPC stack |
| **Risk** | More parameters, harder to train, may overfit |
| **Expected** | +2-5pp PGD-100 if mid-level prediction adds meaningful error signals |
| **Status** | **SCAFFOLDED** (HierarchicalPredictiveStack exists, only level 1 implemented) |
| **Priority** | HIGH — next planned step in the HPC roadmap |

### 2.3 Higher Capacity Belief State (1024-dim)

| Field | Detail |
|-------|--------|
| **What** | Increase proj_dim from 512 to 1024 |
| **Why** | The 512-dim belief state may be a bottleneck for complex visual representations |
| **Evidence** | No ablation on belief state size has been run |
| **Effort** | Low (change one config parameter) |
| **Risk** | ~2× more parameters in belief pathway, potential overfitting |
| **Expected** | +0-2pp PGD-100 (diminishing returns with more parameters) |
| **Status** | **UNEXPLORED** |
| **Priority** | MEDIUM |

### 2.4 Larger Fovea Size (64×64 instead of 48×48)

| Field | Detail |
|-------|--------|
| **What** | Increase fovea_size from 48 to 64 |
| **Why** | Larger foveal crop provides more high-resolution information per fixation |
| **Evidence** | Human fovea covers ~2° of visual angle; at 96×96, 48×48 = 50% of the image, which is already very large |
| **Effort** | Low (change one config parameter) |
| **Risk** | Less peripheral context available (fovea + periphery must sum to image) |
| **Expected** | +0-1pp PGD-100 (already at 50% coverage) |
| **Status** | **UNEXPLORED** |
| **Priority** | LOW |

### 2.5 Multi-Resolution Gaze Policy

| Field | Detail |
|-------|--------|
| **What** | Replace the current single-scale gaze policy with a multi-resolution approach (coarse-to-fine) |
| **Why** | Biological vision uses coarse saccades followed by fine fixations; this could improve gaze efficiency |
| **Evidence** | No multi-resolution gaze experiments run |
| **Effort** | High (1-2 weeks) — new architecture module |
| **Risk** | Complex, may not converge |
| **Expected** | +2-4pp PGD-100 if gaze efficiency improves under attack |
| **Status** | **UNEXPLORED** |
| **Priority** | MEDIUM — interesting but high effort |

### 2.6 Attention-Based Gaze Policy (AIS-v2)

| Field | Detail |
|-------|--------|
| **What** | Replace the current info-gain gaze policy with a learned attention mechanism that predicts where to look next |
| **Why** | The current AIS-v1 is a "relocated Eq. II" — not a genuine forward-looking information-gain policy. AIS-v2 would be |
| **Evidence** | Finding 16: the current gaze policy creates a stationary target for attackers (foraging consistency loss forces adversarial gaze = clean gaze) |
| **Effort** | High (2-3 weeks) — new attention module, new loss function |
| **Risk** | May be too complex to train from scratch |
| **Expected** | +3-6pp PGD-100 if gaze becomes adversarially adaptive |
| **Status** | **IDENTIFIED IN ROADMAP** as "Cluster 2 genuine info-gain gaze" |
| **Priority** | HIGH (long-term) — fundamental architectural improvement |

---

## Category 3: Dataset and Data Pipeline

### 3.1 Larger Evaluation Set (n=1000 instead of 300)

| Field | Detail |
|-------|--------|
| **What** | Increase n_samples from 300 to 1000 per seed |
| **Why** | 300 samples gives ±3-4% confidence intervals; 1000 samples would reduce this to ±1.5-2% |
| **Evidence** | Current 16-seed results have ±1.94% std, which is partly sample noise |
| **Effort** | Low (change one flag) |
| **Risk** | ~3× slower evaluation |
| **Expected** | No improvement in model performance, but tighter confidence intervals |
| **Status** | **UNEXPLORED** |
| **Priority** | MEDIUM — for final publication, not for development |

### 3.2 Better Pseudo-Labeling (Higher Confidence Threshold)

| Field | Detail |
|-------|--------|
| **What** | Increase pseudo-label confidence threshold from 0.65 to 0.80 or 0.85 |
| **Why** | Lower-confidence pseudo-labels may introduce noisy labels that hurt robustness |
| **Evidence** | Finding 17: synthetic data eroded high-ε robustness. Cleaner pseudo-labels may help |
| **Effort** | Low (change one threshold, regenerate pseudo-labels) |
| **Risk** | Fewer pseudo-labeled samples (may drop from 41.6K to ~20K) |
| **Expected** | +1-2pp PGD-100 if noisy labels are hurting |
| **Status** | **UNEXPLORED** |
| **Priority** | MEDIUM |

### 3.3 CIFAR-100 or Tiny-ImageNet as Additional Training Data

| Field | Detail |
|-------|--------|
| **What** | Pre-train the backbone on CIFAR-100 or Tiny-ImageNet before fine-tuning on STL-10 |
| **Why** | More diverse visual features from 100 classes could improve generalization |
| **Evidence** | Finding 14: scaling from 5K to 46.6K images improved clean accuracy by +11.5pp |
| **Effort** | Medium (1-2 days) |
| **Risk** | Distribution shift between CIFAR-100 (32×32) and STL-10 (96×96) |
| **Expected** | +1-3pp PGD-100 |
| **Status** | **UNEXPLORED** |
| **Priority** | LOW-MEDIUM |

### 3.4 Adversarial Training with Diverse Epsilon Schedule

| Field | Detail |
|-------|--------|
| **What** | Randomize epsilon during training (e.g., sample ε ~ U(0, 0.094) each batch) |
| **Why** | Fixed epsilon schedules create decision boundaries optimized for specific ε values |
| **Evidence** | Standard practice in TRADES literature |
| **Effort** | Low (1 day) |
| **Risk** | May slow convergence if ε varies too widely |
| **Expected** | +1-2pp PGD-100 |
| **Status** | **UNEXPLORED** |
| **Priority** | MEDIUM |

---

## Category 4: Mechanism Improvements (AIS / HPC Specific)

### 4.1 Fix Underutilized Halting (7.7% trigger rate) — **CORRECTED**

| Field | Detail |
|-------|--------|
| **What** | ~~Lower~~ **Raise** ais_halt_threshold from 0.35 to 0.50, OR increase T instead |
| **Why** | Only 7.7% of samples trigger halting. But this is CORRECT behavior — most samples NEED all 4 steps. The Banach contraction proof (§3.1) says each step multiplies adversarial noise by γ < 1. More steps = more contraction = better robustness. Halting is a compute-saving optimization, not a robustness mechanism. |
| **Evidence** | Foundation §5.2: "Entropy-gated halting has no step-count penalty. It stops when there is genuinely nothing left to learn — when uncertainty is below threshold — not when a budget is exhausted." The model correctly doesn't halt because it needs all steps. |
| **Correct intervention** | Increase T (more steps), not lower threshold (fewer steps). See §2.1 (More Foraging Steps). |
| **Status** | **CORRECTLY UNDERUSED** — not a bug, a feature |
| **Priority** | LOW — the real fix is T=6, not threshold tuning |

### 4.2 Information-Gain Based Halting Criterion

| Field | Detail |
|-------|--------|
| **What** | Replace entropy-based halting with a prediction-error-based criterion (halt when the HPC prediction error drops below a threshold) |
| **Why** | Entropy measures model uncertainty, not input difficulty. HPC prediction error measures how well the model understands the input |
| **Evidence** | Finding 16: halt loss penalizes foraging depth, conflicting with Banach contraction |
| **Effort** | Medium (2-3 days) — new halting criterion, new loss term |
| **Risk** | May require careful tuning of the HPC error threshold |
| **Expected** | +2-4pp PGD-100 if halting becomes input-adaptive |
| **Status** | **IDENTIFIED IN ROADMAP** as "Cluster 2 genuine info-gain halting" |
| **Priority** | HIGH — directly fixes a known failure mode |

### 4.3 Stronger HPC Supervision (w_hpc=0.30)

| Field | Detail |
|-------|--------|
| **What** | Increase hpc_error_weight from 0.10 to 0.30 |
| **Why** | HPC error is at ~0.14 and flat; stronger supervision may force the model to learn better predictive representations |
| **Evidence** | HPC error converged to ~0.14 early and never improved further across 60 epochs |
| **Effort** | Low (change one config parameter) |
| **Risk** | May dominate the total loss and hurt classification |
| **Expected** | +1-3pp PGD-100 if HPC becomes a stronger regularizer |
| **Status** | **UNEXPLORED** |
| **Priority** | HIGH — cheap to test |

### 4.4 Multi-Scale HPC Targets (Edge + Orientation + Texture)

| Field | Detail |
|-------|--------|
| **What** | Predict multiple visual features at different scales (edges, orientations, textures) rather than just edge maps |
| **Why** | Different adversarial perturbations affect different feature scales; multi-scale prediction catches more |
| **Evidence** | Rao & Ballard (1999): predictive coding operates at multiple cortical areas simultaneously |
| **Effort** | High (1-2 weeks) — multiple predictor heads, multiple loss terms |
| **Risk** | Complex, may overfit |
| **Expected** | +3-5pp PGD-100 if multi-scale prediction provides diverse error signals |
| **Status** | **IDENTIFIED IN ROADMAP** as "Cluster 1 multi-scale world model" |
| **Priority** | MEDIUM (long-term) |

### 4.5 Precision Modulator Redesign

| Field | Detail |
|-------|--------|
| **What** | Replace the scalar precision_modulator.gain with a per-channel or per-head precision weighting |
| **Why** | A single scalar gain may not capture the different precision needs of different feature channels |
| **Evidence** | Recon-mod (E1) showed that precision modulation changes Π_D ordering (airplane replaces truck) |
| **Effort** | Medium (2-3 days) |
| **Risk** | More parameters, may destabilize training |
| **Expected** | +1-2pp PGD-100 |
| **Status** | **UNEXPLORED** |
| **Priority** | MEDIUM |

---

## Category 5: Planned Major Features (SBR + IWM)

### 5.1 Structured Belief Representation (SBR) — Pillar 3

| Field | Detail |
|-------|--------|
| **What** | Replace the flat 512-dim belief state with object-slot-based structured representations (Slot Attention, MAC networks) |
| **Why** | Object-centric representations should be more robust to adversarial perturbation because they capture scene structure, not pixel statistics |
| **Evidence** | Finding 9: concept bottlenecks failed because they were applied to flat features; structured beliefs may succeed |
| **Effort** | High (2-3 months) — entire new module, training pipeline |
| **Risk** | Very high — may not converge, may be too complex for STL-10 |
| **Expected** | +5-10pp PGD-100 if object slots become adversarially invariant |
| **Status** | **IMPLEMENTED** as Stage 4-E2 (ablation matrix entry E2_ais_hpc_sbr, PENDING). 16-slot Slot Attention with GRU + MLP refinement, temporal gating, attention entropy uncertainty. Unit tests pass (gradient flow, backward compat, slot diversity). Ready to train on Colab/Kaggle. |
| **Priority** | 🔴 CRITICAL — directly tests structured belief robustness hypothesis |

### 5.2 Internal World Model (IWM) — Pillar 4

| Field | Detail |
|-------|--------|
| **What** | Add a generative world model (Dreamer/MuZero-style) that can simulate future states and predict what the scene should look like |
| **Why** | A world model can detect adversarial perturbations by comparing predicted vs actual visual input |
| **Evidence** | Finding 10: generative priors (VAE decoder) provide manifold constraints; IWM extends this to full scene understanding |
| **Effort** | Very high (3-6 months) — generative model, rollout mechanism, reward prediction |
| **Risk** | Very high — Dreamer-style models are complex and data-hungry |
| **Expected** | +5-15pp PGD-100 if the world model can denoise adversarial inputs |
| **Status** | **SCAFFOLDED** (NullWorldModel exists, zero-param passthrough) |
| **Priority** | HIGH (long-term) — most ambitious feature |

### 5.3 SBR + IWM Integration

| Field | Detail |
|-------|--------|
| **What** | Combine SBR (structured beliefs) with IWM (world model) for jointly structured + predictive representations |
| **Why** | Object slots provide the "what" and the world model provides the "what should be" — together they enable genuine scene understanding |
| **Evidence** | The ROADMAP.md identifies this as the path to human-level robustness |
| **Effort** | Very high (6+ months) |
| **Risk** | Unknown — no prior work combines these in an adversarial setting |
| **Expected** | Potentially transformative |
| **Status** | **FUTURE** |
| **Priority** | HIGH (long-term) |

---

## Category 6: Evaluation Improvements

### 6.1 AutoAttack Evaluation

| Field | Detail |
|-------|--------|
| **What** | Evaluate against AutoAttack (APGD-CE + APGD-DLR + FAB + Square) instead of just PGD-100 |
| **Why** | AutoAttack is the gold standard; PGD-100 may miss gradient masking (Finding 12) |
| **Evidence** | Finding 12: Self-Alignment scored 84.77% PGD-100 but only 21.60% AutoAttack |
| **Effort** | Low (add AutoAttack to eval pipeline) |
| **Risk** | May reveal that current robustness is partially masked |
| **Expected** | No improvement in model, but honest assessment |
| **Status** | **UNEXPLORED** for RHANNext |
| **Priority** | HIGH — essential for publication |

### 6.2 Certified Robustness Bounds

| Field | Detail |
|-------|--------|
| **What** | Compute randomized smoothing certificates or interval bound propagation (IBP) certificates |
| **Why** | Provides provable guarantees on robustness, not just empirical |
| **Evidence** | Standard in robust ML literature |
| **Effort** | High (1-2 weeks) — new evaluation pipeline |
| **Risk** | Certificates are often loose; may not reflect empirical performance |
| **Expected** | No improvement, but stronger claims |
| **Status** | **UNEXPLORED** |
| **Priority** | MEDIUM (for publication) |

### 6.3 Per-Class Robustness Analysis

| Field | Detail |
|-------|--------|
| **What** | Report per-class PGD-100 accuracy for all 10 STL-10 classes |
| **Why** | Aggregate numbers hide class-specific failures (Finding 7: auto/truck collapsed to 0% on CIFAR-10) |
| **Evidence** | The D eval currently only reports macro accuracy |
| **Effort** | Low (modify eval script to log per-class) |
| **Risk** | None |
| **Expected** | Reveals where improvements are needed |
| **Status** | **UNEXPLORED** for RHANNext |
| **Priority** | HIGH — essential for understanding |

### 6.4 Lens Mechanistic Evaluation Under Attack

| Field | Detail |
|-------|--------|
| **What** | Run the Lens analysis (belief drift, Π_D trajectory, gaze trajectory, HPC error maps) specifically on adversarial examples |
| **Why** | Understanding HOW the model processes adversarial inputs is more valuable than knowing IF it classifies correctly |
| **Evidence** | The E1 evaluation protocol specifies Lens analysis, but it hasn't been run on D yet |
| **Effort** | Medium (2-3 days) — extend eval pipeline |
| **Risk** | None |
| **Expected** | Reveals mechanism-level insights |
| **Status** | **PLANNED** (in E1 protocol) |
| **Priority** | HIGH |

---

## Category 7: Training Data Improvements

### 7.1 Higher Quality Pseudo-Labels

| Field | Detail |
|-------|--------|
| **What** | Use an ensemble of models to generate pseudo-labels, or use consistency regularization |
| **Why** | Single-model pseudo-labels are noisy; ensembles reduce noise |
| **Evidence** | Finding 17: synthetic data eroded robustness; cleaner labels may help |
| **Effort** | Medium (1-2 days) |
| **Risk** | More compute for label generation |
| **Expected** | +1-3pp PGD-100 |
| **Status** | **UNEXPLORED** |
| **Priority** | MEDIUM |

### 7.2 Adversarial Training with Hard Examples

| Field | Detail |
|-------|--------|
| **What** | Mine hard adversarial examples during training and oversample them |
| **Why** | The model sees mostly "easy" examples; hard examples improve the decision boundary |
| **Evidence** | TRADES with hard mining is standard practice |
| **Effort** | Medium (1-2 days) |
| **Risk** | May cause overfitting to specific attack patterns |
| **Expected** | +1-2pp PGD-100 |
| **Status** | **UNEXPLORED** |
| **Priority** | MEDIUM |

### 7.3 Curriculum on Dataset Complexity

| Field | Detail |
|-------|--------|
| **What** | Start training with easy classes (airplane, ship) and gradually add hard classes (car, truck) |
| **Why** | Easy classes build stable representations; hard classes benefit from warm-started features |
| **Evidence** | The car/truck pair is the most vulnerable on CIFAR-10 (Finding 7) |
| **Effort** | Medium (1-2 days) |
| **Risk** | May bias the model toward easy classes |
| **Expected** | +1-2pp PGD-100 on hard classes |
| **Status** | **UNEXPLORED** |
| **Priority** | LOW-MEDIUM |

---

## Category 8: Integration with Future Pillars

### 8.1 Self-Monitoring / Abstention (Cluster 7)

| Field | Detail |
|-------|--------|
| **What** | Add a confidence calibration module that can say "I don't know" when the input is too adversarial |
| **Why** | Humans don't always classify correctly under noise — they recognize uncertainty. A model that abstains on hard examples has higher effective accuracy |
| **Evidence** | Finding 5: AI models are overconfident (0.89-1.00) while humans show calibrated uncertainty |
| **Effort** | Low-Medium (3-5 days) |
| **Risk** | May reduce effective coverage |
| **Expected** | +5-10pp effective accuracy if the model can correctly identify which examples to abstain on |
| **Status** | **IDENTIFIED IN ROADMAP** as "Cluster 7 cheapest next new idea" |
| **Priority** | HIGH — cheapest high-impact addition |

### 8.2 Temporal Consistency (Cluster 3)

| Field | Detail |
|-------|--------|
| **What** | Add temporal consistency loss using video sequences (UCF-101) |
| **Why** | Consecutive video frames are naturally robust to adversarial perturbation; temporal consistency provides a self-supervised anti-collapse signal |
| **Evidence** | Finding 13: TDV (Temporal Difference in Vision) provides causal temporal constraints |
| **Effort** | Medium-High (1-2 weeks) — video data pipeline, temporal loss |
| **Risk** | Domain shift between video (UCF-101) and images (STL-10) |
| **Expected** | +3-5pp PGD-100 if temporal consistency regularizes representations |
| **Status** | **IDENTIFIED IN ROADMAP** as "Cluster 3" |
| **Priority** | MEDIUM (long-term) |

### 8.3 Distributional / Multi-Hypothesis Belief (Cluster 6)

| Field | Detail |
|-------|--------|
| **What** | Extend Π_D from a scalar to a full (μ, Σ) distribution over hypotheses |
| **Why** | A distributional belief can represent competing hypotheses (e.g., "this might be a car OR a truck") rather than collapsing to a single point estimate |
| **Evidence** | Current Π_D is a single precision scalar; the ROADMAP identifies this as Cluster 6 |
| **Effort** | High (2-3 weeks) |
| **Risk** | Complex, may not converge |
| **Expected** | +2-4pp PGD-100 if multi-hypothesis belief prevents premature commitment |
| **Status** | **IDENTIFIED IN ROADMAP** |
| **Priority** | MEDIUM (long-term) |

---

## Priority Ranking (Sorted by Impact/Effort) — Foundation-Aligned

*Updated after reading `docs/NOESIS_FOUNDATION.md`. The framework's discipline (one mechanism at a time, gradient-reachability tests, status tags) constrains what "quick wins" can mean.*

### Quick Wins (< 1 week, likely >1pp improvement)

| Rank | Improvement | Expected | Effort | Foundation Alignment |
|------|------------|----------|--------|---------------------|
| 1 | **More foraging steps** (T=6) | +2-4pp | 1 day | Principle II (iterative inference), Banach contraction (§3.1) |
| 2 | **Stronger HPC supervision** (w_hpc=0.15) | +1-3pp | 1 day | Principle V (hierarchical prediction) — start gradual, not 0.30 |
| 3 | **Data augmentation** (RandAugment) | +1-3pp | 1 day | Data pipeline improvement, not mechanism change |
| 4 | **Higher β in TRADES** (β=3.5) | +1-3pp | 1 day | Standard robust training practice |
| 5 | **PGD-20 at train time** | +2-4pp | 2-3 days | Harder adversarial training |
| 6 | **Self-monitoring / abstention** | +5-10pp effective | 3-5 days | Cluster 7, cheapest high-impact addition |

### Medium-Term (1-4 weeks)

| Rank | Improvement | Expected | Effort |
|------|------------|----------|--------|
| 8 | **AIS-v2 (info-gain gaze)** | +3-6pp | 2-3 weeks |
| 9 | **Multi-scale HPC targets** | +3-5pp | 1-2 weeks |
| 10 | **Temporal consistency (TDV)** | +3-5pp | 1-2 weeks |
| 11 | **AutoAttack evaluation** | honest assessment | 1 week |
| 12 | **Per-class robustness analysis** | reveals failures | 2-3 days |

### Long-Term (months)

| Rank | Improvement | Expected | Effort |
|------|------------|----------|--------|
| 13 | **SBR (object-slot beliefs)** | +5-10pp | 2-3 months |
| 14 | **IWM (world model)** | +5-15pp | 3-6 months |
| 15 | **SBR + IWM integration** | transformative | 6+ months |

---

## T=6 vs T=8 vs T=10: Detailed Analysis

The Banach contraction bound says each step multiplies adversarial noise by γ < 1. If γ ≈ 0.8 (typical for well-trained recurrent models):

| T | γ^T | Attenuation | Additional compute vs T=4 | Marginal gain |
|---|-----|-------------|---------------------------|---------------|
| 4 | 0.410 | 59% | baseline | — |
| 6 | 0.262 | 74% | +50% | +15 pp |
| 8 | 0.168 | 83% | +100% | +9 pp |
| 10 | 0.107 | 89% | +150% | +6 pp |

**Diminishing returns:** T=4→6 gains 15pp attenuation, T=6→8 gains only 9pp, T=8→10 gains only 6pp.

**But the deeper issue:** γ < 1 is a THEORETICAL bound, not an empirical measurement. The actual γ depends on the model's architecture and training. If the model doesn't actually contract (γ ≥ 1), then T=10 won't help either.

**Practical constraints on STL-10:**
- Each step requires foveal crop + foveal stream + belief update = ~15M FLOPs
- At T=10, that's 2.5× more memory and compute per sample
- With 46.6K training images and 60 epochs, T=10 would take ~20 hours on a T4 vs ~8 hours for T=6
- Overfitting risk increases with trajectory length on small datasets

**Recommended approach:**
1. **T=6 first** (50% more compute). If it shows >2pp PGD-100 improvement, proceed.
2. **T=8 second** (100% more compute than T=4). Only if T=6 shows clear improvement.
3. **T=10 only if T=8 also shows improvement** and we have compute budget.

**Why not T=10 immediately?** If T=6 shows no improvement, T=10 won't either (the model isn't contracting). If T=6 shows improvement, the marginal gain from T=8 is smaller, and T=10 is smaller still. Start with the minimum intervention that tests the hypothesis.

---

## What Has Already Failed (Do Not Repeat)

| Approach | Why it failed | Reference |
|----------|--------------|-----------|
| Recon-mod for robustness | Pixel reconstruction conflicts with adversarial training | E1 results, Finding 10 |
| Self-Alignment / Feature Scatter | Directly induces gradient masking | Finding 12 |
| Concept Bottlenecks | Binary concepts can't resolve continuous feature overlap | Finding 9 |
| Phase-clamped adversarial training | Fine-tuning a robust model with new objectives always hurts | FINDINGS.md §What failed |
| CLIP as ongoing loss | Smooth semantic manifolds are exploitable | Finding 7 |
| β=6.0 TRADES | KL term over-penalizes, loss explodes | Finding 10 |
| Synthetic data scaling | Eroded high-ε robustness | Finding 17 |
| Three auxiliary losses (foraging, precision, halt) | Directly oppose the robustness objective | Finding 16 |

---

## Decision Framework: What to Try Next

Given the current state (D validated, E1 preliminary negative), the recommended sequence is:

1. **Immediate (this week)**: Fix halting threshold + test w_hpc=0.30 + add RandAugment
2. **Next cycle**: PGD-20 training + higher β + more foraging steps
3. **Before publication**: AutoAttack evaluation + per-class analysis + Lens mechanistic evaluation
4. **After E1 verdict**: Decide whether to pursue AIS-v2, multi-scale HPC, or SBR based on which mechanism shows the most promise

---

*This document should be updated after each experiment cycle with actual results.*
