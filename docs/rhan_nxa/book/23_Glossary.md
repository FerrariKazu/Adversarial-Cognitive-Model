# Chapter 23 — Technical Glossary

> *Level 1–2 reading. Precise definitions, mathematical notations, code mappings, and authoritative status classifications for all core concepts in RHAN-NXA.*

---

## 1. Mathematical Notation and Conventions

| Notation | Dimension / Type | Description | Reference |
|---|---|---|---|
| $x$ | `(B, 3, 224, 224)` | Input image batch | [Ch. 10](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/10_Foveation.md) |
| $t$ | Scalar $\in \{0, 1, \dots, T\}$ | Glimpse index ($T = 4$ fixed) | [Ch. 09](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/09_Recurrence.md) |
| $B_t$ | 5-Tuple $(z_t, S_t, U_t, E_t, A_t)$ | Canonical perceptual belief state at step $t$ | [Ch. 04](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/04_Belief_State.md) |
| $z_t$ | `(B, 384)` | Global perceptual latent state ($D_z = 384$) | [Ch. 05](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/05_z_State.md) |
| $S_t$ | `Optional[...]` | Structural state ($S_t = \text{None}$ in Gen-1 core) | [Ch. 06](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/06_Structure_State.md) |
| $U_t$ | `(B,)` | Class-conditioned epistemic uncertainty via Dirichlet evidence | [Ch. 07](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/07_Uncertainty.md) |
| $E_t$ | `(B, 16, 384)` or `(B, 384)` | Realized prediction error at glimpse $t$ ($E_0 := 0$) | [Ch. 08](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md) |
| $\Pi_t$ | `(B, 1)` or `(B, 384)` | Precision weighting derived from inverse uncertainty | [Ch. 08](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md) |
| $a_t$ | `(B, 2)` $\in [-1, 1]^2$ | Fixation coordinates (normalized center coordinates) | [Ch. 11](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md) |
| $g_t$ | `(B, 3, 56, 56)` | Observed high-resolution foveal glimpse patch | [Ch. 10](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/10_Foveation.md) |
| $\hat{g}_t$ | `(B, 16, 384)` | Predicted patch-token features for glimpse $t$ | [Ch. 08](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md) |
| $A_t$ | Immutable Record | History of past gaze fixations $(a_1, \dots, a_t)$ | [Ch. 11](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md) |
| $\mathcal{L}_{\text{stab}}$ | Scalar | Trajectory stability loss over belief drift | [Ch. 20](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/20_L_stab.md) |

---

## 2. Alphabetical Terms

### Active Information Sampling Version 2 (AIS-v2)
**Status: REQUIRED as design; Isolated robustness contribution UNKNOWN.**  
The active sensing gaze policy that selects next fixation coordinates $a_{t+1}$ by maximizing predicted uncertainty reduction. It generates $K \in \{4, \dots, 8\}$ candidate coordinates, projects anticipated features using the shared predictor, and scores candidates using Dirichlet evidential entropy. Supersedes the rejected AIS-v1 policy.  
*Source code:* [noesis_vision/gaze/ais_v2_policy.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/gaze/ais_v2_policy.py). Cross-ref: [Ch. 11](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md).

### Action / Fixation Coordinate ($a_t$)
**Status: LOCKED.**  
The continuous coordinate pair $(y, x) \in [-1, 1]^2$ specifying the spatial center of the foveal bounding box relative to the input image, where $(-1, -1)$ represents the top-left corner and $(1, 1)$ represents the bottom-right corner.  
*Source code:* [noesis_vision/gaze/gaze_state.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/gaze/gaze_state.py). Cross-ref: [Ch. 10](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/10_Foveation.md).

### Belief State ($B_t$)
**Status: LOCKED.**  
The primary internal state of RHAN-NXA, structured as the canonical 5-tuple $(z_t, S_t, U_t, E_t, A_t)$. It satisfies the None-propagating contract: operations defined on $B_t$ must execute validly when $S_t = \text{None}$ (Gen-1 Core default). It enforces locked first-glimpse discrepancy ($E_0 := 0$), tracks gaze history ($A_{t+1} = A_t \cup \{a_{t+1}\}$), and exposes `.drift_to()` for evaluating trajectory divergence and `.detached_copy()` for history logging without graph retention.  
*Source code:* [noesis_vision/beliefs/vector_belief.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/vector_belief.py). Cross-ref: [Ch. 04](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/04_Belief_State.md).

### CompactViT Substrate
**Status: LOCKED.**  
The visual Transformer backbone featuring patch size $14 \times 14$, embedding dimension $D_z = 384$, 6 attention heads, and MLP dimension 1536. Configured to operate within a strict parameter budget of 20–25M (actual: ~23.3M excluding readout heads).  
*Source code:* [noesis_vision/models/backbone.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/backbone.py). Cross-ref: [Ch. 13](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/13_Architecture.md).

### DELTA_BOUND
**Status: LOCKED (Value: 0.1).**  
The hard architectural bound enforced by UpdateNet on latent state updates. The raw MLP update vector is passed through a hyperbolic tangent and scaled: $\Delta z_t = 0.1 \cdot \tanh(\cdot)$. Prevents single-glimpse adversarial corruption from destabilizing the global belief state.  
*Source code:* [noesis_vision/predictive_coding/update_net.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/update_net.py). Cross-ref: [Ch. 08](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md).

### Dirichlet Uncertainty ($U_t$)
**Status: REQUIRED for Gen-1.**  
Epistemic uncertainty quantified analytically from Dirichlet distribution parameters $\alpha_k$ produced by the EvidentialHead: $U_t = K / \sum \alpha_k \in (0, 1]$. Clamped to $[10^{-6}, 10^4]$ during evidence accumulation to ensure numerical stability. Measures class-conditioned certainty, not visual scene entropy.  
*Source code:* [noesis_vision/uncertainty/evidential_head.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/uncertainty/evidential_head.py). Cross-ref: [Ch. 07](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/07_Uncertainty.md).

### Drift Metric (`drift_to`)
**Status: LOCKED contract; distance choice PENDING DECISION.**  
The function computing sample-wise latent distance between two belief states, returning a differentiable `torch.Tensor` of shape `(B,)`. Exposes both $L_2$ Euclidean distance and Cosine distance.  
*Source code:* [noesis_vision/beliefs/drift.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/drift.py). Cross-ref: [Ch. 20](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/20_L_stab.md).

### Evidential Head
**Status: NEW implementation in Gen-1 (Published Sensoy et al. formulation).**  
A classification readout head that outputs non-negative Dirichlet evidence vectors $e_k \ge 0$ via a softplus activation, inducing a Dirichlet prior over class multinomials. Replaces standard Softmax to yield decoupled aleatoric and epistemic uncertainty.  
*Source code:* [noesis_vision/uncertainty/evidential_head.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/uncertainty/evidential_head.py). Cross-ref: [Ch. 07](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/07_Uncertainty.md).

### Foveation Module
**Status: LOCKED.**  
A differentiable spatial crop-and-rescale operator utilizing PyTorch `affine_grid` and `grid_sample` with bilinear interpolation. Extracts a high-resolution $56 \times 56$ patch $g_t$ centered at fixation coordinates $a_t$ from the full $224 \times 224$ input image.  
*Source code:* [noesis_vision/models/foveation.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/foveation.py). Cross-ref: [Ch. 10](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/10_Foveation.md).

### Gate 9 (Responsiveness Guard)
**Status: REQUIRED as design; threshold PENDING DECISION.**  
A pre-registered, two-sided stopping rule governing $\mathcal{L}_{\text{stab}}$. Evaluates whether a reduction in adversarial trajectory drift is accompanied by an acceptable level of responsiveness to novel semantic evidence on an out-of-distribution probe set. Zero drift on both inputs is classified as a terminal failure (representational collapse).  
*Source code:* [noesis_vision/beliefs/drift.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/drift.py). Cross-ref: [Ch. 20](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/20_L_stab.md).

### Gaze State ($A_t$)
**Status: LOCKED.**  
A non-learned, immutable historical trace recording the sequence of realized fixation coordinates $(a_1, \dots, a_t)$. Consumed by AIS-v2 to enforce spatial coverage and prevent repetitive fixation loops.  
*Source code:* [noesis_vision/gaze/gaze_state.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/gaze/gaze_state.py). Cross-ref: [Ch. 11](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md).

### GlimpseFeaturePredictor
**Status: REQUIRED (Shared architecture).**  
A neural predictor that projects prior belief state $z_{t-1}$ and prospective fixation coordinates $a_t$ into anticipated $16 \times 384$ patch-token features $\hat{g}_t$. Consumed jointly by realized prediction error calculation and prospective AIS-v2 gaze scoring. Total parameter budget: ~2.8M.  
*Source code:* [noesis_vision/predictive_coding/glimpse_predictor.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/glimpse_predictor.py). Cross-ref: [Ch. 08](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md).

### Multi-Group Optimizer
**Status: PORT VERBATIM from Generation-0.**  
An optimizer wrapper that partitions model parameters into 5 strictly isolated optimization groups (Backbone, Recurrent Block, Predictor, UpdateNet, Evidential Head), enabling group-specific learning rates, weight decays, and gradient isolation checks.  
*Source code:* [noesis_vision/core/multi_group_optimizer.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/multi_group_optimizer.py). Cross-ref: [Ch. 14](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/14_Gradient_Flow.md).

### Option C (Hybrid Recurrence)
**Status: LOCKED.**  
The architectural design choice coupling iterative, weight-tied token refinement within each glimpse (2–3 iterations of a shared Transformer block) with stateful belief evolution across discrete glimpses ($T = 4$). Balances compute scaling against parameter compact constraints.  
*Source code:* [noesis_vision/models/recurrent_block.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/recurrent_block.py). Cross-ref: [Ch. 09](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/09_Recurrence.md).

### Precision Weighting ($\Pi_t$)
**Status: LOCKED.**  
A dynamic scaling factor applied to prediction errors, inversely proportional to uncertainty: $\Pi_t = 1 - U_t$, clamped above a minimum precision floor $\Pi_{\text{floor}} = 10^{-4}$. High uncertainty dampens state updates, protecting the belief trajectory against noisy or adversarial evidence.  
*Source code:* [noesis_vision/predictive_coding/precision.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/precision.py). Cross-ref: [Ch. 08](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md).

### Prediction Error ($E_t$)
**Status: EXPERIMENTAL CANDIDATE.**  
The mismatch between observed foveal patch features $g_t^{\text{obs}}$ and predicted features $\hat{g}_t^{\text{pred}}$ at glimpse $t$. Enforces the boundary condition $E_0 := 0$ at the initial step. Fully differentiable during backpropagation.  
*Source code:* [noesis_vision/predictive_coding/interfaces.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/interfaces.py). Cross-ref: [Ch. 08](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md).

### Spatial Bottleneck Recurrence (SBR)
**Status: REJECTED (Gen-0 16-slot implementation); Concept is DEFERRED.**  
The Generation-0 object-centric slot-attention mechanism. Evaluated as dysfunctional in Gen-0 due to chance-level slot probes, slot-zeroing performance gains, and an uncorrected ~10 pp clean accuracy collapse. Replaced by $S_t = \text{None}$ in the Gen-1 core.  
*Source code:* Historical reference only. Cross-ref: [Ch. 06](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/06_Structure_State.md), [Ch. 19](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/19_Gen0_Evidence.md).

### UpdateNet
**Status: EXPERIMENTAL CANDIDATE.**  
A lightweight MLP that fuses prior belief $z_{t-1}$ with precision-weighted prediction error $\Pi_t \odot E_t$ to compute bounded state updates $\Delta z_t$.  
*Source code:* [noesis_vision/predictive_coding/update_net.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/update_net.py). Cross-ref: [Ch. 08](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md).

### V1 Frontend
**Status: EXPERIMENTAL CANDIDATE (Sequenced LAST).**  
A biologically inspired, fixed (non-learnable) bank of Gabor convolution filters positioned at the initial input layer. Staged for evaluation in Step 9 as an isolated ablation arm to prevent entangling low-level frequency filtering with recurrent belief dynamics.  
*Source code:* Staged module. Cross-ref: [Ch. 25](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/25_Future_Directions.md).

### Global Perceptual State ($z_t$)
**Status: LOCKED.**  
The dense vector of dimension $D_z = 384$ providing a unified, holistic summary of the visual scene at glimpse $t$. Serves as the substrate for classification, belief drift computation, and predictive coding.  
*Source code:* [noesis_vision/beliefs/vector_belief.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/vector_belief.py). Cross-ref: [Ch. 05](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/05_z_State.md).

---

## 3. Status Classification Summary

| Classification | Meaning in RHAN-NXA Specification | Components / Decisions |
|---|---|---|
| **LOCKED** | Fixed architectural choice; no modification permitted without formal RFC | $D_z = 384$, $T = 4$, Fovea $56 \times 56$, $E_0 := 0$, Tied Recurrence, Multi-Group Opt |
| **REQUIRED** | Mandatory core component; must be implemented and active in Gen-1 | $B_t$ interface, EvidentialHead, Shared Predictor, UpdateNet, AIS-v2 |
| **EXPERIMENTAL CANDIDATE** | Formally specified hypothesis; under active evaluation in the DAG | Predictive coding loop, AIS-v2 robustness, V1 Frontend, $\mathcal{L}_{\text{stab}}$ |
| **DEFERRED** | Conceptually accepted but postponed to maintain isolation | $S_t$ re-entry (2–4 slots), Adaptive Halting, Episodic Video Memory |
| **PENDING DECISION** | Value or threshold to be calibrated from baseline empirical distributions | Gate 9 responsiveness floor, Gate 6 correlation cutoff, Representation uncertainty |
| **REJECTED** | Explicitly dismissed; prohibited from entering codebase even as fallback | 16-slot SBR code, Pixel reconstruction, AIS-v1 gaze code, RAG/External retrieval |
| **UNKNOWN** | Empirical claim lacking unconfounded evidence | Isolated robustness contribution of AIS-v2 |
