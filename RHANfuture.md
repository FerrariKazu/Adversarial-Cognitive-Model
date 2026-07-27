# RHANfuture: The Road to Human-Equivalent Visual Perception

> **"Models are fooled, not confused."** — Finding 5, FINDINGS.md
>
> This document exists because we refuse to accept that ceiling. The gap between
> RHAN's εthresh ≈ 0.185 and human εthresh > 0.30 is not a limit of the principles—
> it's a limit of what we've built so far. This document maps every remaining gap
> and prescribes, at code level, how to close each one.

### ⚠️ A Critical Note Before Reading

**We know that architecture has diminishing returns.** Finding 3 in FINDINGS.md
states this clearly: "RHAN-v6 added dynamic gating, predictive coding, and ACT —
and regressed." The v6-v11 iterations showed that adding complexity to what was
*already* a complete architecture (v5) produced zero εthresh benefit.

**This roadmap is NOT about adding more complexity to the same architecture.**
It is about filling *gaps in the processing hierarchy* — mechanisms that the
human visual system has but RHAN entirely lacks:

- v6-v11 added computational overhead to existing pathways (dynamic gates,
  extra foraging loops, generative priors)
- This roadmap adds ENTIRELY NEW PROCESSING DIMENSIONS (temporal, hierarchical
  predictive coding at every level, sparse coding, neuromodulation, semantic
  grounding, metacognition)

The distinction matters: v6-v11 were **optimizations of the same processing
pipeline**. Gaps 1-7 are **new pipeline stages** that the human brain has and
RHAN doesn't. Adding new stages is categorically different from optimizing
existing ones.

**That said, each gap must be independently verified.** Do NOT implement all 7
at once. The correct protocol (§10.1) requires implementing and measuring each
gap against the identical baseline before any integration. This prevents the
v6-v11 problem where multiple changes interacted in unpredictable ways.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State: Where RHAN Stands Today](#2-current-state-where-rhan-stands-today)
3. [Gap 1: Temporal Processing (TDV)](#3-gap-1-temporal-processing-tdv)
4. [Gap 2: Hierarchical Predictive Coding](#4-gap-2-hierarchical-predictive-coding)
5. [Gap 3: Realistic Saccadic Foraging](#5-gap-3-realistic-saccadic-foraging)
6. [Gap 4: Neuro-modulatory Control](#6-gap-4-neuro-modulatory-control)
7. [Gap 5: Sparse Coding & Efficient Representation](#7-gap-5-sparse-coding--efficient-representation)
8. [Gap 6: True Semantic Grounding](#8-gap-6-true-semantic-grounding)
9. [Gap 7: Global Workspace & Metacognition](#9-gap-7-global-workspace--metacognition)
10. [The Grand Unified Architecture](#10-the-grand-unified-architecture)
11. [Falsifiable Predictions](#11-falsifiable-predictions)
12. [Appendix: Key Codebases to Modify](#12-appendix-key-codebases-to-modify)
13. [References](#13-references)

---

## 1. Executive Summary

### The Thesis

Human visual perception is not a single mechanism — it is an **emergent property**
of at least 7 interacting systems operating across a strict processing hierarchy:

| # | System | Biological Basis | RHAN Status |
|---|--------|-----------------|-------------|
| 1 | **Temporal processing** | MT/V5 motion perception, frame-to-frame prediction | TDV components exist but untrained |
| 2 | **Hierarchical predictive coding** | Rao & Ballard (1999) — each cortical level predicts the next | Only at feedback connector, not per-level |
| 3 | **Active foraging** | Saccades, microsaccades, inhibition-of-return | FovealStream + action_init exist; no IoR, no scanpath |
| 4 | **Neuromodulation** | ACh, DA, NE, 5-HT gating | Only `ThermodynamicHalt` with single scalar |
| 5 | **Sparse coding** | V1: 1-2% neurons active, lateral inhibition | None — dense activations everywhere |
| 6 | **Semantic grounding** | Language integration, conceptual knowledge | CLIP init only; CBM too coarse |
| 7 | **Global workspace / metacognition** | PFC broadcasting, uncertainty estimation | SDT *measures* it; architecture doesn't *implement* it |

### The Projection

Implementing Gaps 1+2 (temporal + hierarchical PC) in the next 3-6 months should
push εthresh from **0.185 → 0.30-0.40**, matching human-level sensitivity on CIFAR-10.

Implementing all 7 gaps should push εthresh **beyond 0.50** — exceeding human
performance, which would be the first published result of a machine vision system
matching or exceeding human perceptual sensitivity across the full adversarial
spectrum.

### The Methodological Promise

Your existing evaluation infrastructure — SDT framework (`phase5_sdt/sdt_core.py`),
AutoAttack pipeline (`eval_pgd_final.py`), PGD-100 verification, per-class d'
analysis, confidence calibration curves — is already publication-grade. Every new
architectural change can be evaluated with principled psychophysical metrics.

---

## 2. Current State: Where RHAN Stands Today

### What Works (Do NOT Change)

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| Frequency separation | `model_rhan_v5.py` | M/P pathway gating | Best v5 variant; confirmed biological hypothesis |
| Ventral/dorsal split | `model_rhan_stl10_large.py:64-77` | What/where streams | Persistent improvement across all variants |
| TRADES loss | `train_rhan_trades_curriculum.py` | KL-based robust training | Best available objective |
| Cosine head | `model_rhan.py:SemanticProjectionHead` | Prototype-based classification | Critical for TRADES stability at low β |
| Curriculum training | `train_rhan_trades_curriculum.py` | Progressive ε scaling | 3-phase: 0.062→0.100→0.150 |
| SDT evaluation | `phase5_sdt/sdt_core.py` | d', β, εthresh measurement | Publication-grade framework |
| Pseudo-labeling | `upload_pseudolabel.py` | 41.6K pseudo-labels | +11.5pp clean, +1.3pp AA |
| AutoAttack verification | `eval_pgd_final.py` | Gradient-free attack validation | Proves no gradient masking |

### What Should Be Dropped or Redesigned

| Component | Problem | Recommendation |
|-----------|---------|---------------|
| v10/v11 auxiliary losses | 45% of gradient budget, each opposes robustness | Replace entirely per §3.6 of original analysis |
| Foraging consistency MSE | Stationary attack target | Replace with adaptive gaze |
| Halt efficiency loss | Penalizes foraging depth = Banach contraction | Replace with info-gain halting |
| Precision calibration | Saturates under attack | Replace with prediction-error-driven update |
| Pixel-level reconstruction | Conflicting with TRADES (v7 finding) | Use feature-level only |

### Current Best Results (Baseline to Beat)

| Metric | RHAN-trades-curriculum | Human |
|--------|:----------------------:|:-----:|
| εthresh (d'=1.0) | 0.185 | >0.30 |
| Clean accuracy (CIFAR-10) | ~89% | ~95% |
| AutoAttack @ ε=0.031 | 21.88% | ~85% |
| PGD-100 @ ε=0.10 | ~18% | ~75% |
| Automobile/truck AA | 0.0% / 0.0% | >70% |

---

## 3. Gap 1: Temporal Processing (TDV)

### Biological Motivation

The primate visual system processes **30+ frames per second**. Area MT/V5 is
dedicated to motion processing. Human vision evolved for dynamic scenes — objects
in motion, self-motion through environments, saccadic suppression, and optic flow.

**Key insight:** Temporal consistency is a *principled* anti-collapse mechanism.
Consecutive video frames are naturally different (unlike augmented static images),
creating a manifold constraint that adversarial examples cannot easily exploit.

Daithankar et al. (June 2026) formalized this as **TDV (Temporal Difference in
Vision)**:
```
z_t + m_t = z_{t+1}
```
Where `z_t` is the frame representation and `m_t` is the encoded motion.

### What Exists Already

**`model_rhan_stl10_large.py`:**

```python
class MotionEncoderLarge(nn.Module):
    # Line 170: Takes (frame_t, frame_{t+1}) → 512-dim motion vector
    # Currently receives 6-channel input but is UNTRAINED

class TDVProjectionHeadLarge(nn.Module):
    # Line 193: Projects 768-dim features → 512-dim TDV space
    # Currently initialized but never used in loss computation
```

These components are **built but never integrated into the training loop**.
The forward pass of `RHANLargeSTL10` instantiates them but the `forward()` method
only processes single images — never pairs.

### What To Build

#### 3.1 Video Dataloader

Create a new file `phase1_training/video_dataset.py`:

```python
class VideoSTL10(Dataset):
    """Pairs STL-10 images into artificial 'video pairs' via augmentation.
    
    Step 1: Load STL-10 image
    Step 2: Apply controlled augmentation (color jitter, small affine, crop)
             to simulate natural frame-to-frame variation
    Step 3: Return (frame_t, frame_{t+1}, label)
    
    The augmentations should be SMALL — simulating camera/object motion,
    not heavy data augmentation. Max translation: 2px, rotation: 3°.
    """

class UCF101Subset(Dataset):
    """Alternatively: load real UCF-101 video clips (8-16 frames each).
    
    Returns consecutive frame pairs from the same clip.
    Natural temporal coherence is stronger than synthetic augmentation.
    """
```

#### 3.2 TDV Pretraining Loss

Add to `train_rhan_v12.py`:

```python
def tdv_loss(model, frame_t, frame_{t+1}, labels):
    # 1. Extract features for both frames
    z_t = model.get_feature_vector(frame_t)           # (B, 768)
    z_t1 = model.get_feature_vector(frame_{t+1})       # (B, 768)
    
    # 2. Project to TDV space
    proj_t = model.tdv_head(z_t)                       # (B, 512)
    proj_t1 = model.tdv_head(z_t1)                     # (B, 512)
    
    # 3. Encode motion between frames
    motion = model.motion_encoder(frame_t, frame_{t+1})  # (B, 512)
    
    # 4. Temporal consistency constraint
    # z_t + m_t ≈ z_{t+1}
    loss_temporal = F.mse_loss(proj_t + motion, proj_t1.detach())
    
    # 5. Additional: adversarial temporal consistency
    # f(x_adv[t]) should produce features similar to f(x_clean[t+1])
    # This is the causal anti-collapse mechanism
    
    return loss_temporal
```

#### 3.3 Adversarial Temporal Consistency

The most important innovation: during TRADES training, replace the standard
KL-robustness term with a **temporal one**:

```python
# During PGD generation:
x_adv_t = pgd_attack(frame_t, labels, epsilon=eps)

# Temporal consistency loss:
logits_clean_t1 = model(frame_{t+1})      # Natural next frame
logits_adv_t = model(x_adv_t)              # Adversarial current frame
loss_temporal_robust = KL(logits_adv_t || logits_clean_t1)

# This says: "the adversarial version of frame t should produce features
# similar to the NEXT clean frame" — a causally principled constraint
```

**Expected impact:** +0.05-0.10 on εthresh. Temporal pretraining before TRADES
curriculum should yield additional gains.

#### 3.4 Files to Create

| File | Purpose |
|------|---------|
| `phase1_training/video_dataset.py` | Video pair dataloaders |
| `phase1_training/pretrain_tdv.py` | TDV pretraining script (Phase 0) |
| `phase1_training/train_rhan_v12.py` | Next-gen training with temporal TRADES |

#### 3.5 Files to Modify

| File | Change |
|------|--------|
| `model_rhan_stl10_large.py` | Unify `forward()` to accept frame pairs |
| `model_rhan_stl10_large.py` | Add `forward_temporal()` method |

---

## 4. Gap 2: Hierarchical Predictive Coding

### Biological Motivation

Rao & Ballard (1999) proposed that each cortical level in the visual hierarchy
(V1→V2→V4→IT) operates according to the same algorithm:

1. **Top-down prediction:** Each level *predicts* the activity of the level below
2. **Prediction error:** The difference between the prediction and actual activity
3. **Error propagation:** Only the prediction error passes *upward*
4. **Prediction update:** Each level updates its internal model to minimize error

This is **not what RHAN currently implements**. RHAN has a single predictive
coding layer at the feedback connector (between transformer output and stem
features). True hierarchical predictive coding requires prediction-error units
at **every level** of the hierarchy.

### What Exists Already

**`model_rhan_stl10_large.py`:**

```python
class PredictiveCodingLayerLarge(nn.Module):      # Line 99
    """Single-level predictor: global_spatial → predicted local_f
       Error correction: local_f + gate * (local_f - predicted)
       This operates at ONE level only (the feedback connector)."""
    
    def __init__(self, channels=768):
        self.predictor = nn.Sequential(            # Conv → GN → GELU → Conv
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.error_gate = ...                      # Gate network for error
        self.error_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, local_f, global_spatial):
        predicted = self.predictor(global_spatial)  # What we expect local_f to be
        error = local_f - predicted                  # Prediction error
        gate = self.error_gate(error)               # Learned gating
        corrected = local_f + self.error_scale * gate * error
        return corrected, error.abs().mean()
```

**`model_rhan_predictive.py`:**

```python
class PredictiveFeedback(nn.Module):
    """Also single-level: transformer spatial → prediction of stem features.
       Same pattern: predict → error → gate → correct."""
```

### What To Build

#### 4.1 Hierarchical PC Architecture

Create a new file `phase1_training/model_rhan_hpc.py`:

```
Architecture:

                      ┌──────────────────────────────────────┐
                      │           IT-level (768-dim)          │
                      │  Predictor: Linear(768→512) ↓         │
                      │  Error: V4_actual - V4_predicted ↑    │
                      └──────────┬───────────────────────────┘
                                 │ tokens (512)
                      ┌──────────▼───────────────────────────┐
                      │          V4-level (512-dim)           │
                      │  Predictor: Conv1x1(512→256) ↓        │
                      │  Error: V2_actual - V2_predicted ↑    │
                      └──────────┬───────────────────────────┘
                                 │ spatial 256
                      ┌──────────▼───────────────────────────┐
                      │          V2-level (256-dim)           │
                      │  Predictor: Conv3x3(256→128) ↓        │
                      │  Error: Stem_actual - Stem_predicted ↑│
                      └──────────┬───────────────────────────┘
                                 │ stem 128
                      ┌──────────▼───────────────────────────┐
                      │        V1/Stem-level (128-dim)        │
                      │  Predictor: Conv5x5(128→3) ↓          │
                      │  Error: Image - Predicted_Image ↑     │
                      └──────────────────────────────────────┘
```

Each level has the structure:

```python
class HierarchicalPCLevel(nn.Module):
    """
    One level of Rao & Ballard predictive coding.
    
    Each level:
    - Receives bottom-up input (features from level below)
    - Receives top-down prediction (from level above)
    - Computes prediction error = bottom_up - top_down_prediction
    - Passes error UP (not the features themselves)
    - Updates internal representation using error
    """
    
    def __init__(self, in_channels, out_channels, hidden_channels):
        self.predictor = Predictor(out_channels → in_channels)   # Top-down
        self.error_gate = Gate(in_channels → [0,1])              # Gating
        self.rep_update = Update(out_channels, out_channels)     # State update
    
    def forward(self, bottom_up, top_down_context):
        # 1. Generate prediction of what bottom-up should look like
        prediction = self.predictor(top_down_context)
        
        # 2. Compute prediction error
        error = bottom_up - prediction
        
        # 3. Gate the error (some errors are noise, some are signal)
        gated_error = self.error_gate(error) * error
        
        # 4. Update representation
        # This is the Rao & Ballard update rule
        updated = self.rep_update(top_down_context, gated_error)
        
        return updated, error   # Return both for total error minimization
```

#### 4.2 Total Prediction Error Loss

The key insight: the model minimizes prediction error at ALL levels simultaneously:

```python
def total_prediction_error_loss(model, x):
    """Total free energy = sum of prediction errors at every level."""
    
    # Forward pass returns prediction errors at every level
    logits, pred_errors = model.forward_with_errors(x)
    
    # pred_errors is a list: [stem_error, v2_error, v4_error, it_error]
    # Each is a scalar from that level's Predictor
    
    total_FE = sum(pred_errors)  # This IS the Free Energy
    
    # The classification loss is ADDED to this
    ce_loss = F.cross_entropy(logits, labels)
    
    return ce_loss + 0.1 * total_FE  # Beta balances classification vs FE
```

#### 4.3 Banach Contraction Amplification

Your Banach contraction proof shows each feedback step multiplies the adversarial
perturbation by γ < 1. With 4 hierarchical levels × 2 recurrent steps:

```
γ^8 contraction instead of γ^2

For γ = 0.7 (reasonable estimate):
- Current: γ^2 = 0.49  → perturbation halved
- Hierarchical: γ^8 = 0.058 → perturbation reduced 17×
```

This is the mathematical justification for why hierarchical PC should
dramatically improve εthresh.

> **⚠️ Assumption Note:** This estimate assumes contraction factors are
> independent and multiplicative across levels. In practice, contractions are
> likely correlated — the same perturbation propagates through all levels,
> so γ_total may be closer to γ^(2×num_levels^0.7) ≈ γ^5 ≈ 0.17 than γ^8.
> The actual γ at each level must be measured empirically (see §10.1).
> Even γ^5 ≈ 0.17 represents a 5-6× perturbation reduction vs the current
> γ^2 ≈ 0.49, which would still be a significant improvement.

#### 4.4 Files to Create

| File | Purpose |
|------|---------|
| `model_rhan_hpc.py` | Hierarchical PC architecture |
| `train_rhan_hpc.py` | Training script with FE minimization |

#### 4.5 Files to Modify

| File | Change |
|------|--------|
| `model_rhan_stl10_large.py` | Add HPCLevel after each stem block |

---

## 5. Gap 3: Realistic Saccadic Foraging

### Biological Motivation

The human eye makes **3-4 saccades per second**, each lasting 20-40ms.
Between saccades are **fixations** (200-300ms) where visual information is
actually acquired. Key properties of real saccadic behavior:

1. **Inhibition-of-return (IOR):** The visual system actively suppresses
   attending to recently visited locations. This prevents getting stuck.
2. **Scanpath planning:** Fixation sequences follow natural image statistics
   and task demands — not random or purely gradient-driven.
3. **Microsaccades:** Tiny fixational movements (0.1-0.5°) that prevent
   retinal adaptation. Without them, the visual scene would fade.

### What Exists Already

**`model_rhan_v10.py`:**

```python
class FovealStream(nn.Module):                     # Line 180
    """3-layer ConvNet for 48×48 crops → 512-dim features."""
    
class ThermodynamicHalt(nn.Module):                 # Line 221
    """Decides when to stop foraging.
       Inputs: (precision, error_mag, step_frac)
       Output: halt_probability ∈ [0, 1]
       Problem: uses HARD halt when info_gain < metabolic_cost
       Problem: metabolic_cost is a single scalar, not context-dependent"""

def foveal_sample(x_image, action_a, fovea_size):  # Line 120
    """Differentiable foveal crop via F.grid_sample.
       This IS the motor Jacobian ∂f_stem(a)/∂a."""

class RHANv10.forward():                            # Line 240
    """Foraging loop: T=2 steps (v10) or T=4 steps (v11).
       Action update: gradient ascent on prediction error.
       Problem: NO inhibition-of-return → can stare at same pixel
       Problem: gradient-based update is fragile and myopic"""
```

### What To Build

#### 5.1 Inhibition-of-Return Mask

```python
class InhibitionOfReturn(nn.Module):
    """
    Maintains a spatial 'visit count' map per image.
    Each visited location gets a penalty that decays slowly.
    
    The gaze action is penalized for revisiting recently attended
    locations, forcing exploration of new regions.
    """
    
    def __init__(self, map_size=48, decay_rate=0.95):
        self.register_buffer('visit_map', torch.zeros(1, 1, map_size, map_size))
        self.decay_rate = decay_rate
    
    def forward(self, action, prediction_error):
        # 1. Update visit map at current action location
        gaussian_mask = self._gaussian_at(action)
        self.visit_map = self.visit_map * self.decay_rate + gaussian_mask
        
        # 2. Modulate prediction error by inverse visit count
        # High visit → suppression (inhibition-of-return)
        visit_penalty = self.visit_map / (self.visit_map.max() + 1e-8)
        modulated_error = prediction_error * (1 - visit_penalty)
        
        return modulated_error, self.visit_map
```

#### 5.2 Learned Scanpath Policy

Replace the gradient-based action update with a learned policy:

```python
class ScanpathPolicy(nn.Module):
    """
    Learns where to look next given:
    - Current foveal features (what we're seeing now)
    - Parafoveal features (full-field context)
    - Current belief state (what we think we're looking at)
    - History of visited locations (where we've been)
    
    This is a tiny MLP (3 layers, ~50K params) that replaces
    the manual gradient-ascent gaze update.
    """
    
    def __init__(self, proj_dim=512):
        self.policy = nn.Sequential(
            nn.Linear(proj_dim*2 + 2 + 2*4, 256),  # [foveal; para; belief; current_gaze; history]
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Tanh(),  # output ∈ [-1, +1]
        )
    
    def forward(self, foveal_feat, para_feat, belief, current_gaze, history):
        x = torch.cat([foveal_feat, para_feat, belief, current_gaze, history], dim=-1)
        delta_gaze = self.policy(x)
        return current_gaze + 0.1 * delta_gaze  # Smooth update
```

#### 5.3 Microsaccade Noise

```python
def add_microsaccades(action, amplitude=0.02, frequency=0.3):
    """
    Add tiny random perturbations to gaze position.
    amplitude=0.02 → ~2% of image width (biologically realistic)
    frequency=0.3  → applied to 30% of steps
    
    Prevents overfitting to exact fixation coordinates and
    adds natural stochasticity that regularizes the policy.
    """
    if torch.rand(1) < frequency:
        noise = torch.randn_like(action) * amplitude
        return torch.clamp(action + noise, -0.9, 0.9)
    return action
```

#### 5.4 Files to Create/Modify

| File | Change |
|------|--------|
| `model_rhan_hpc.py` | Add InhibitionOfReturn, ScanpathPolicy |
| `model_rhan_hpc.py` | Add microsaccade noise in forward() |
| `train_rhan_v12.py` | Add IOR to auxiliary loss diagnostics |

---

## 6. Gap 4: Neuro-modulatory Control

### Biological Motivation

The brain uses **neuromodulators** to dynamically regulate information processing:

| Modulator | Function | ML Analogue |
|-----------|----------|-------------|
| **ACh** (Acetylcholine) | Precision/attention gating | Π_D precision control |
| **DA** (Dopamine) | Prediction error gain, learning rate | Learning rate scheduling |
| **NE** (Norepinephrine) | Explore/exploit balance | Halt/noise modulation |
| **5-HT** (Serotonin) | Aversion to uncertainty | Confidence thresholding |

### What Exists Already

**`model_rhan_v10.py`:**

```python
class ThermodynamicHalt(nn.Module):     # Line 221
    """Uses ONE scalar metabolic_cost=0.05 for ALL images and ALL times.
       Bio comparison: NE modulates gain globally, but also locally."""

class PrecisionController(nn.Module):   # Line 52
    """Π_D = sensory precision (one scalar per image).
       Bio comparison: ACh regulates precision, but per-region, not global."""
```

### What To Build

#### 6.1 Neuromodulatory Gating Module

```python
class Neuromodulator(nn.Module):
    """
    Produces per-layer, per-channel gain vectors from global context.
    
    4 output channels:
    - precision_gain: modulates Π_D per layer (ACh analogue)
    - learn_rate_gain: modulates effective LR per layer (DA analogue)
    - explore_gain: modulates exploration noise (NE analogue)
    - confidence_threshold: modulates halt criterion (5-HT analogue)
    
    Each gain is a vector of length = number of feature channels in that layer.
    Multiplicative gating: f_out = gain ⊙ f_in
    """
    
    def __init__(self, embed_dim=768):
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # 4 modulators
            nn.Sigmoid(),  # gains in [0, 1]
        )
        # Reshape to [4, embed_dim] — one gain per channel per modulator
        self.reshape = lambda x: x.view(-1, 4, embed_dim)
    
    def forward(self, global_context):
        gains = self.gate_net(global_context)
        return self.reshape(gains)  # (B, 4, embed_dim)
```

#### 6.2 Integration into the Foraging Loop

```python
# Inside the foraging loop at each step:

# Get neuromodulatory gains from current belief state
gains = self.neuromodulator(s)              # (B, 4, 512)
precision_gain = gains[:, 0, :]             # Per-channel ACh analogue
explore_gain = gains[:, 2, :]               # Per-channel NE analogue

# Apply gains:
# 1. Precision-weighted belief update becomes per-channel
pi_d_per_channel = pi_d.unsqueeze(-1) * precision_gain.mean(dim=-1, keepdim=True)
s = (1 - pi_d_per_channel) * s + pi_d_per_channel * foveal_feat

# 2. Explore/exploit affects halt threshold
effective_cost = metabolic_cost / (explore_gain.mean() + 1e-8)
# High explore_gain → lower effective cost → more foraging
```

#### 6.3 Files to Create/Modify

| File | Change |
|------|--------|
| `model_rhan_hpc.py` | Add `Neuromodulator` module |
| `model_rhan_hpc.py` | Integrate into `forward()` |
| `train_rhan_v12.py` | Log neuromodulator statistics per epoch |

---

## 7. Gap 5: Sparse Coding & Efficient Representation

### Biological Motivation

V1 implements **sparse coding**: only ~1-2% of neurons are active for any given
stimulus (Olshausen & Field, 1996). This provides:

1. **Energy efficiency:** Less metabolic cost
2. **Natural robustness:** Adversarial perturbations must activate a *specific*
   sparse subset of neurons, not just increase overall activation
3. **Lateral inhibition:** The most active neurons suppress their neighbors,
   creating competitive coding

RHAN currently uses **dense activations everywhere** (ReLU, GELU) with no
sparsity constraint.

### What To Build

#### 7.1 k-Winner-Take-All (kWTA) Layer

```python
class kWTA(nn.Module):
    """
    Only the top-k% activations per spatial location survive.
    
    Applied after each Conv layer in the stem.
    k=5 means only 5% of channels are active → 20× sparsity
    
    During adversarial training, this forces the attack to target
    a very specific subset of channels rather than any active channel.
    """
    
    def __init__(self, k_percent=5.0):
        self.k_percent = k_percent
    
    def forward(self, x):
        # x: (B, C, H, W)
        k = max(1, int(x.shape[1] * self.k_percent / 100.0))
        
        # Find top-k channels per spatial location
        threshold = torch.kthvalue(x, x.shape[1] - k + 1, dim=1, keepdim=True)[0]
        
        # Binary mask: keep only top-k
        mask = (x >= threshold).float()
        
        return x * mask  # All others → 0
```

#### 7.2 Lateral Inhibition

```python
class LateralInhibitionAttention(nn.Module):
    """
    Self-attention with competitive normalization.
    
    Standard self-attention: softmax(QK^T / sqrt(d)) → V
    Lateral inhibition: softmax(QK^T / sqrt(d) - inhibition_matrix) → V
    
    The inhibition_matrix is learned and represents lateral connections
    between token positions. Nearby tokens inhibit each other (center-surround).
    """
    
    def __init__(self, num_patches=144):
        # Learnable lateral inhibition kernel (center-surround)
        self.inhibition_kernel = nn.Parameter(torch.randn(num_patches, num_patches))
        # Initialize: high self-inhibition, moderate near-inhibition, low far-inhibition
        
        # Distance-based prior
        coords = self._get_coords(num_patches)  # (N, 2)
        distances = torch.cdist(coords, coords)  # (N, N)
        inhibition_prior = torch.exp(-distances / 3.0)  # Nearby → more inhibition
        self.register_buffer('inhibition_prior', inhibition_prior)
    
    def forward(self, attention_scores):
        # attention_scores: (B, H, N, N)
        # Apply learned + distance-based lateral inhibition
        inhibition = self.inhibition_kernel + self.inhibition_prior
        inhibited_scores = attention_scores - inhibition.unsqueeze(0).unsqueeze(0)
        return F.softmax(inhibited_scores / math.sqrt(attention_scores.shape[-1]), dim=-1)
```

#### 7.3 Sparsity Regularization Loss

```python
def sparsity_loss(features, target_sparsity=0.95):
    """
    Encourage most activations to be near zero.
    
    target_sparsity=0.95 → 95% of activations should be ≈ 0
    
    Uses KL divergence between actual and target activation distribution.
    Applied AFTER the kWTA layer to enforce sparsity at training time.
    """
    # Measure fraction of active neurons
    active_fraction = (features.abs() > 0.01).float().mean()
    
    # KL divergence: encourage active_fraction → (1 - target_sparsity)
    target = 1.0 - target_sparsity  # e.g., 0.05 (5% active)
    
    # Binary KL: D_KL(active_fraction || target)
    # This is simple: we just push active fraction toward target
    return F.mse_loss(active_fraction, torch.tensor(target, device=features.device))
```

#### 7.4 Integration Strategy

Add kWTA + lateral inhibition in stages:

1. First: kWTA after stem Conv layers (low risk, high impact)
2. Second: Sparsity loss in training objective (w=0.001, then tune)
3. Third: Lateral inhibition in transformer attention (risky, test carefully)
4. Fourth: Evaluate εthresh change at each stage

**Expected impact:** +0.03-0.08 on εthresh from sparsity alone.

#### 7.5 Files to Create/Modify

| File | Change |
|------|--------|
| `model_rhan_hpc.py` | Add `kWTA`, `LateralInhibitionAttention` |
| `model_rhan_hpc.py` | Insert kWTA after stem conv layers |
| `train_rhan_v12.py` | Add `sparsity_loss` to total loss |

---

## 8. Gap 6: True Semantic Grounding

### Biological Motivation

Human object categories are grounded in **function, language, and world
knowledge** — not just visual appearance. A cat is a cat because:

- It purrs (auditory)
- It hunts mice (behavioral)
- It is a pet (functional)
- It has pointy ears (visual)
- "Chat" in French, "猫" in Japanese (linguistic)

When visual signal degrades to ε=0.30, humans fall back on this conceptual
knowledge. RHAN has no such fallback.

### What's Been Tried (And Why It Failed)

1. **CLIP as ongoing loss (v4):** Smooth semantic manifolds were exploitable.
   The CLIP gradient created an attack surface.

2. **CLIP as initialization only (v5):** Works well but the semantic grounding
   is diluted during adversarial training phases.

3. **Concept Bottleneck Model (CBM):** 15 hand-defined concepts were too coarse.
   Automobile/truck distinguished by `is_small_vehicle` vs `carries_cargo` —
   concepts that are themselves visually confusable under attack.

### What To Build

#### 8.1 Residual-Contrastive Dual Encoder

The key insight: instead of aligning RHAN's features TO semantic features
(which creates an attack surface), align only the **residual** — the part of
the feature space not used for robustness:

```python
class SemanticBridges(nn.Module):
    """
    Dual-encoder architecture for semantic grounding.
    
    RHAN produces features f = f_robust + f_semantic_residual.
    f_robust is used for classification (adversarially trained).
    f_semantic_residual is aligned to CLIP/text features.
    
    The semantic loss affects ONLY f_semantic_residual, leaving
    f_robust unaffected → no attack surface.
    """
    
    def __init__(self, embed_dim=768):
        # Learnable decomposition: robust vs semantic subspaces
        self.robust_projector = nn.Linear(embed_dim, embed_dim)
        self.semantic_projector = nn.Linear(embed_dim, embed_dim)
        
        # CLIP text embedding cache (frozen)
        self.register_buffer('text_embeddings', self._load_text_embeddings())
    
    def forward(self, features, labels):
        # Decompose features
        f_robust = self.robust_projector(features)
        f_semantic = self.semantic_projector(features)
        
        # Classify from robust component
        logits = self.classifier(f_robust)
        
        # Align SEMANTIC component (not robust) to text space
        text_emb = self.text_embeddings[labels]  # (B, 512)
        semantic_aligned = self.align_net(f_semantic)  # (B, 512)
        
        # Contrastive loss: pull f_semantic toward correct text, push from others
        loss_semantic = contrastive_loss(semantic_aligned, text_emb, temperature=0.07)
        
        # The robust component is UNTOUCHED by the semantic loss
        return logits, loss_semantic
```

#### 8.2 Learnable Concept Bank

Instead of 15 hand-defined concepts, learn a **dynamic concept bank**:

```python
class LearnableConceptBank(nn.Module):
    """
    128 learnable concept vectors, each 512-dim.
    
    Each concept is discovered during training — not predefined.
    The model learns which concepts are useful for classification.
    
    Concepts are grounded by a separate text encoder that maps
    each concept vector to its nearest word in CLIP space.
    """
    
    def __init__(self, num_concepts=128, concept_dim=512):
        self.concepts = nn.Parameter(torch.randn(num_concepts, concept_dim))
        nn.init.kaiming_uniform_(self.concepts, a=math.sqrt(5))
        
        # Concept classifier
        self.classifier = nn.Linear(num_concepts, 10)  # concepts → classes
    
    def forward(self, features):
        # Project features onto concept basis
        concept_activations = F.linear(features, self.concepts)  # (B, num_concepts)
        concept_activations = F.relu(concept_activations)  # Non-negative activation
        
        # Sparse concept activation (only top-10 concepts per image)
        k = 10
        topk_vals, _ = torch.topk(concept_activations, k, dim=1)
        threshold = topk_vals[:, -1:].expand(-1, concept_activations.shape[1])
        concept_activations = concept_activations * (concept_activations >= threshold).float()
        
        # Classify from concept activations
        logits = self.classifier(concept_activations)
        
        return logits, concept_activations
```

#### 8.3 Files to Create/Modify

| File | Change |
|------|--------|
| `model_rhan_hpc.py` | Add `SemanticBridges`, `LearnableConceptBank` |
| `train_rhan_v12.py` | Add contrastive semantic loss |

---

## 9. Gap 7: Global Workspace & Metacognition

### Biological Motivation

The **Global Neuronal Workspace** theory (Baars, 1988; Dehaene, 2001) proposes
that conscious perception involves a central "workspace" where information is
broadcast to all cortical areas simultaneously. This enables:

1. **Flexible, context-dependent behavior:** The same visual input can trigger
   different responses in different contexts
2. **Metacognitive access:** We know what we see and can report confidence
3. **Integration:** Information from different modalities is bound together

Your Finding 5 — "Models are fooled, not confused" — is a direct measurement
of what happens when a system lacks a global workspace. Models maintain 89-100%
confidence as accuracy drops to 0%. Humans decline from 7.78→6.86/10.

### What To Build

#### 9.1 Uncertainty Estimation Head

```python
class UncertaintyHead(nn.Module):
    """
    Predicts P(correct) for each sample.
    
    Architecture: feature → 2-head output
    - logits: standard classification (10 classes)
    - uncertainty: P(correct) ∈ [0, 1]
    
    The uncertainty head is trained with a calibration loss:
    MSE(predicted_uncertainty, 1 - accuracy_in_batch)
    """
    
    def __init__(self, embed_dim=768, num_classes=10):
        # Shared feature extractor
        self.feature_norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Uncertainty head
        self.uncertainty = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output ∈ [0, 1]
        )
    
    def forward(self, features):
        f = self.feature_norm(features)
        logits = self.classifier(f)
        uncertainty = self.uncertainty(f)  # P(incorrect)
        return logits, uncertainty
```

#### 9.2 Calibration Loss

```python
def calibration_loss(uncertainties, predictions, labels, epsilon=0.1):
    """
    Train uncertainty estimates to match actual accuracy.
    
    Uses a batch-level calibration constraint:
    For bins of predicted uncertainty, the actual accuracy should match.
    
    Also uses a per-sample proxy: samples with high prediction error
    (KL divergence from clean) should have high uncertainty.
    """
    # 1. ECE (Expected Calibration Error) style loss
    # Bin predictions by uncertainty level and compare to accuracy
    
    # 2. Evidential uncertainty
    # Samples with high feature variance should be uncertain
    uncertain_mask = (uncertainties > 0.5).float()
    correct = (predictions == labels).float()
    
    # MSE between uncertainty and 1-accuracy
    ece_loss = F.mse_loss(uncertainties.squeeze(), 1 - correct)
    
    return ece_loss
```

#### 9.3 Uncertainty-Gated Foraging

```python
# Modified halting criterion that uses uncertainty:
# "Keep foraging while uncertainty is high AND maximum steps not reached"

def uncertainty_gated_halt(halt_prob, uncertainty, threshold=0.3):
    """
    Override halt decision with uncertainty signal.
    
    If model is uncertain (uncertainty > threshold), continue foraging
    even if thermodynamic criterion says halt.
    If model is confident (uncertainty < threshold/2), halt early
    even if thermodynamic criterion says continue.
    """
    # Low uncertainty → halt regardless of metabolic cost
    early_halt = (uncertainty < threshold / 2).float()
    
    # High uncertainty → continue regardless of metabolic cost
    continue_override = (uncertainty > threshold).float()
    
    # Combined: halt_prob modulated by uncertainty
    adjusted_halt = halt_prob * (1 - continue_override) + early_halt
    
    return torch.clamp(adjusted_halt, 0.0, 1.0)
```

#### 9.4 The "I Don't Know" Output

```python
# During inference:
logits, uncertainty = model(x)
if uncertainty > 0.8:  # High uncertainty threshold
    output = "UNCERTAIN"  # Reject classification
else:
    output = logits.argmax()
```

This is the architectural equivalent of human metacognition. A system that
knows when it doesn't know is categorically different from one that guesses
with high confidence.

#### 9.5 Files to Create/Modify

| File | Change |
|------|--------|
| `model_rhan_hpc.py` | Add `UncertaintyHead` |
| `train_rhan_v12.py` | Add `calibration_loss` |
| `eval_pgd_final.py` | Add uncertainty reporting to evaluation |

---

## 10. The Grand Unified Architecture

### 10.1 ⚠️ Mandatory: Independent Gap Evaluation Protocol

**Do NOT implement all 7 gaps at once. If you do, you will not know which gap
contributed which gain, and you will repeat the v6-v11 problem where multiple
changes interacted unpredictably.**

The correct protocol:

1. **Freeze the baseline:** Lock RHAN-trades-curriculum at εthresh=0.185.
   This is your control. Measure εthresh with 5 seeds to get error bars.

2. **Implement Gap 1 only** (temporal processing). Train from scratch using
   the Phase 0 TDV pretraining + temporal TRADES. Measure εthresh.
   Compare to baseline.

3. **On the same baseline, implement Gap 2 only** (hierarchical PC).
   Train from scratch with FE minimization. Measure εthresh.

4. **Repeat for each gap independently.** Each gap gets its own branch,
   its own training run, its own εthresh measurement.

5. **Gamma measurement:** For Gap 2, measure γ empirically at each HPC level
   by computing ‖error_adv‖ / ‖error_clean‖ at each level. This validates
   or falsifies the Banach contraction amplification claim.

6. **Only after independent verification**, integrate successful gaps into
   RHAN-v12. Gaps that don't produce measurable εthresh improvement should
   be dropped or redesigned.

This is non-negotiable. The v6-v11 experience proves that multiple unverified
changes interact destructively.

### RHAN-v12: The Complete Architecture (After Independent Verification)

The following is the target architecture for RHAN-v12, incorporating all
successfully verified gaps. This is a top-level design; each component needs
its own implementation.

```
Input: (B, 3, 96, 96) — single image or video frame
│
├── [Gap 1] Temporal Branch (parallel, optional)
│   └── if video_pair:
│       └── MotionEncoderLarge → TDV loss
│
├── [Gap 5] Retinal Preprocessing
│   ├── FrequencySeparation (from v5)
│   │   ├── Low-freq path: conv → kWTA(5%)
│   │   └── High-freq path: conv → kWTA(5%)
│   └── FrequencyWeightedFusion (learnable w_low, w_high)
│
├── ConvStem V1-Level (128 channels)
│   ├── Conv → BN → ReLU → kWTA(5%)
│   └── [Gap 2] HPCLevel: predicts image pixels, error at pixel level
│
├── ConvStem V2-Level (256 channels)
│   ├── Conv → BN → ReLU → kWTA(10%)
│   └── [Gap 2] HPCLevel: predicts V1 features, error at feature level
│
├── PatchTokeniser (12x12 → 144+1 tokens)
│
├── Transformer V4-Level (384-dim ventral + 384-dim dorsal)
│   ├── [Gap 5] LateralInhibitionAttention (in self-attention)
│   ├── Ventral stream (what pathway)
│   ├── Dorsal stream (where pathway)
│   └── [Gap 2] HPCLevel: predicts V2 features in each stream
│
├── [Gap 4] Neuromodulator (from global CLS token)
│   └── Produces per-layer gain vectors (ACh, DA, NE, 5-HT analogues)
│
├── Recurrent Feedback Loop (T=4 steps)
│   ├── [Gap 3] FovealStream (48×48 crops)
│   ├── [Gap 3] ParafovealStream (blurred full-field)
│   ├── [Gap 3] FovealParafovealGate
│   ├── [Gap 3] ScanpathPolicy (learned gaze update)
│   ├── [Gap 3] InhibitionOfReturn
│   ├── [Gap 2] HPCLevel at each recurrence step
│   └── [Gap 7] UncertaintyGatedHalt
│
├── [Gap 6] Semantic Integration
│   ├── SemanticBridges (residual-contrastive)
│   └── LearnableConceptBank (128 learned concepts)
│
└── [Gap 7] Output
    ├── logits (10-class)
    └── uncertainty (P(incorrect))
```

### Parameter Budget

| Component | Parameters (est.) | Source |
|-----------|------------------:|--------|
| Base RHANLargeSTL10 | 55.6M | Existing |
| Hierarchical PC (4 levels) | +8M | New |
| Sparse coding (kWTA + lateral inh.) | +0.5M | New |
| Foraging improvements (IOR, scanpath) | +1.5M | Redesigned from v11 |
| Neuromodulator | +0.3M | New |
| Semantic bridges + concept bank | +4M | New |
| Uncertainty head | +0.2M | New |
| **Total** | **~70M** | |

### Training Curriculum

```
Phase 0: TDV video pretraining (20 epochs, UCF-101)
Phase 0.5: Sparsity warmup (10 epochs, no adversarial)
Phase A: ε=0.031, β=2.0, w_FE=0.1, w_sparse=0.001 (20 epochs)
Phase B: ε=0.062, β=2.0, w_FE=0.15, w_sparse=0.005 (20 epochs)
Phase C: ε=0.094, β=2.5, w_FE=0.2, w_sparse=0.01 (20 epochs)
Phase D: ε=0.150, β=3.0, w_FE=0.25, w_sparse=0.02 (20 epochs)
         + semantic contrastive loss w_sem=0.05
         + calibration loss w_cal=0.05
```

---

## 11. Risk Assessment & Failure Mitigation

Every architectural change carries risk. Here are the specific failure modes
and recovery strategies for each gap.

| Gap | Risk | Severity | Mitigation |
|-----|------|----------|------------|
| **1** (Temporal) | Video pretraining may not transfer to static adversarial eval | Medium | Use synthetic video pairs (augmented STL-10) as fallback |
| **2** (Hierarchical PC) | Prediction errors at all levels may be noisy and destabilize training | High | Start with 2 levels instead of 4; anneal FE weight from 0→0.2 |
| **3** (Foraging) | Learned scanpath policy may be harder to train than gradient-based | Medium | Keep gradient-based gaze as fallback for first 10 epochs |
| **4** (Neuromodulation) | 0.3M extra params may overfit on 5K labels | Low | Use strong weight decay (0.01 on gate params only) |
| **5** (Sparsity) | kWTA reduces clean accuracy by >1% | High | Start with k=20%, anneal to k=5% over 30 epochs |
| **6** (Semantic) | Residual-contrastive design may still leak gradient to robust component | Critical | Verify: robust component εthresh must be IDENTICAL to no-semantic baseline. If not, design is flawed. |
| **7** (Metacognition) | Uncertainty head may degrade classification accuracy | Medium | Train uncertainty head AFTER classification head is frozen |

**If a gap fails its prediction (§11), the action is:**
- Revert to baseline, document the negative result (this is still valuable science)
- Redesign the component with the specific failure in mind
- Test again before attempting integration

## 12. Falsifiable Predictions

Each gap must be evaluated independently. The following are falsifiable
predictions that — if not met — indicate the gap analysis or implementation
is wrong.

### Gap 1 (Temporal)
**Prediction:** TDV pretraining + temporal consistency loss increases εthresh
by at least 0.03 compared to identical model without temporal training.

### Gap 2 (Hierarchical PC)
**Prediction:** Total Free Energy (sum of all prediction errors) decreases
monotonically across recurrent steps for clean images but INCREASES for
adversarial images. The rate of increase correlates with εthresh.

### Gap 3 (Foraging)
**Prediction:** Inhibition-of-return increases the spatial diversity of
fixations (measured by entropy of visit map). The link: higher foraging
entropy → higher εthresh.

### Gap 4 (Neuromodulation)
**Prediction:** The ACh analogue (precision gain) should be HIGH in early
foraging steps and LOW in late steps. The NE analogue (explore gain) should
be HIGH under strong attack and LOW for clean images.

### Gap 5 (Sparsity)
**Prediction:** k-WTA with k=5% maintains clean accuracy within 1% of dense
baseline but increases εthresh by at least 0.02. Lateral inhibition in
attention further improves by 0.01.

### Gap 6 (Semantic Grounding)
**Prediction:** The robust component of the feature decomposition shows
identical εthresh to a model trained WITHOUT semantic alignment. The
semantic component shows strong CLIP similarity. (This verifies the
residual-contrastive approach doesn't create an attack surface.)

### Gap 7 (Metacognition)
**Prediction:** ECE (Expected Calibration Error) at ε=0.05 drops below 0.10
for the uncertainty-aware model versus >0.40 for the baseline. The model
should output "UNCERTAIN" for >50% of samples at ε=0.10.

---

## 13. Appendix: Key Codebases to Modify

### 13.1 Architecture Files — Detailed Extraction Mappings

| New File | Source File | Classes/Functions to Extract |
|----------|-------------|------------------------------|
| `model_rhan_hpc.py` | `model_rhan_stl10_large.py` | `RHANLargeSTL10` (base class, line 202), `WideSEConvStemLarge` (line 30), `PatchTokeniserLarge` (line 71), `MotionEncoderLarge` (line 170), `TDVProjectionHeadLarge` (line 193) |
| `model_rhan_hpc.py` | `model_rhan_v10.py` | `FovealStream` (line 180), `foveal_sample` (line 120), `PrecisionController` (line 52), `ThermodynamicHalt` (line 221) |
| `model_rhan_hpc.py` | `model_rhan_v11.py` | `ParafovealStream` (line 90), `FovealParafovealGate` (line 155), `GenerativePrior` (line 230), `ImageSpacePrecision` (line 283) |
| `model_rhan_hpc.py` | `model_rhan_predictive.py` | `PredictiveFeedback` pattern (line 20) — redesign for hierarchical |
| `model_rhan_hpc.py` | `model_rhan_v5.py` | `separate_frequencies` (line 130), `freq_weight_low/high` (line 42) |

### 13.2 Training Files

| New File | Source | Purpose |
|----------|--------|---------|
| `video_dataset.py` | New | Video pair dataloaders (STL-10 synthetic, UCF-101 real) |
| `pretrain_tdv.py` | New | Phase 0 video pretraining with temporal loss |
| `train_rhan_v12.py` | `train_rhan_trades_curriculum.py` | v12 training: reuse PGD generation, TRADES loss, curriculum loop. Add FE loss, sparsity loss, semantic loss, calibration loss. |

### 13.3 Evaluation Files — Updates Needed

| File | Change |
|------|--------|
| `eval_pgd_final.py` | Add `--measure-gamma` flag for HPC level contraction measurement |
| `eval_stl10.py` | Add uncertainty reporting (ECE, AUROC) |
| `phase5_sdt/sdt_analysis.py` | Add per-level prediction error reporting (for HPC) |
| `eval_pgd_sweep.py` | Keep as is — baseline comparator |

### 13.4 Critical: What NOT to Change

```
# These files work perfectly and should NOT be modified:
- phase5_sdt/sdt_core.py       # SDT framework — gold standard
- phase5_sdt/sdt_analysis.py   # Already handles per-class d'
- model_rhan_v5.py             # Reference architecture for CIFAR-10
- model_rhan.py                # Base RHAN (CIFAR-10) — still used for experiments
- upload_pseudolabel.py        # Pseudo-labeling pipeline — still critical
- eval_pgd_final.py            # AutoAttack + PGD-100 evaluation harness
- eval_pgd_sweep.py            # Full sweep across epsilons

# These files should be DEPRECATED (not deleted, just not extended):
- model_rhan_v6.py through model_rhan_v11.py
- train_rhan_v6.py through train_rhan_v11.py
```

### 13.5 Hardware Requirements (Estimated)

| Gap | Batch Size | GPUs | Training Time (STL-10) |
|-----|-----------|------|----------------------|
| 1 (Temporal) | 8-16 | 1× A100 or 4× T4 | 24 hours |
| 2 (Hierarchical PC) | 4-8 | 2× A100 or 8× T4 | 48 hours |
| 3 (Foraging) | 8 | 1× A100 or 4× T4 | 36 hours |
| 5 (Sparsity) | 16 | 1× A100 | 18 hours |
| 6 (Semantic) | 8 | 1× A100 | 24 hours |
| 7 (Metacognition) | 16 | 1× A100 | 12 hours |
| **v12 Full** | **4-8** | **4× A100 (DDP)** | **~72 hours** |

---

## 13. References

1. **Rao & Ballard (1999)** — Predictive coding in the visual cortex.
   *Nature Neuroscience.*
   [The foundational paper for Gap 2.]

2. **Friston (2010)** — The free-energy principle: a unified brain theory.
   *Nature Reviews Neuroscience.*
   [Theoretical framework for total prediction error minimization.]

3. **Olshausen & Field (1996)** — Emergence of simple-cell receptive field
   properties by learning a sparse code for natural images. *Nature.*
   [Sparse coding in V1 — Gap 5.]

4. **Itti & Koch (2001)** — Computational modelling of visual attention.
   *Nature Reviews Neuroscience.*
   [Saliency-based attention — Gap 3.]

5. **Baars (1988)** — A cognitive theory of consciousness.
   [Global workspace theory — Gap 7.]

6. **Dehaene & Changeux (2011)** — Experimental and theoretical approaches to
   conscious processing. *Neuron.*
   [Global neuronal workspace — Gap 7.]

7. **Daithankar et al. (2026)** — Temporal Difference in Vision (TDV).
   [Published June 14, 2026. Gap 1 framework.]

8. **Lotter et al. (2016)** — Deep predictive coding networks (PredNet).
   *ICLR.*
   [ML implementation of hierarchical predictive coding — Gap 2.]

9. **Geirhos et al. (2019)** — ImageNet-trained CNNs are biased towards
   texture; increasing shape bias improves accuracy and robustness.
   *ICLR.*
   [Texture vs shape bias — confirmed by your v5 frequency separation.]

10. **Madry et al. (2018)** — Towards deep learning models resistant to
    adversarial attacks. *ICLR.*
    [TRADES/PGD training framework you already use.]

11. **Dapello et al. (2020)** — Simulating a primary visual cortex at the
    front of CNNs improves robustness to image perturbations. *NeurIPS.*
    [VOneNet — V1 front-end improves robustness, supports your approach.]

12. **Kubilius et al. (2019)** — CORnet-S: Brain-like object recognition
    at scale. *NeurIPS.*
    [Recurrent CNN that models IT cortex — used as your alignment teacher.]

13. **Green & Swets (1966)** — Signal Detection Theory and Psychophysics.
    [SDT framework you use for d' analysis — Gap 7 measurement.]

---

> **"The key to building human-level visual perception is not finding a single
> magic architecture — it's understanding that visual perception is an
> emergent property of multiple interacting systems. RHAN already implements
> four of them (frequency separation, ventral/dorsal streams, recurrent
> feedback, active foraging). The remaining three (temporal processing,
> hierarchical predictive coding, semantic grounding) are precisely the ones
> that close the gap to human performance."**
>
> — From the original conversation, July 2026
