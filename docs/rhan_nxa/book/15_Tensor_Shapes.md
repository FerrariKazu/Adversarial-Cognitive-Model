# Chapter 15 — Tensor Shapes and Interface Contracts: Dimensional Accounting

## 1. In one sentence
Every tensor flowing through RHAN-NXA has a precisely defined shape, dtype, gradient status, and coordinate convention; this chapter is the definitive reference table for those shapes, derived verbatim from verified source code.

## 2. Belief state tensors

| Symbol | Shape | Dtype | Gradient | Source |
|:---:|:---|:---:|:---:|:---|
| $z_t$ | $(B, 384)$ | float32 | **Yes** | `backbone.D_Z = 384` |
| $e_t$ (evidence) | $(B, C)$ | float32 | **Yes** | `EvidentialHead.forward` |
| $\alpha_t = e_t + 1$ | $(B, C)$ | float32 | **Yes** | `DirichletParams.alpha` |
| $U_t$ (scalar) | $(B,)$ | float32 | **Yes** | `DirichletParams.uncertainty` |
| $H(U_t)$ (entropy) | $(B,)$ | float32 | **Yes** | `DirichletParams.entropy()` |
| $E_t$ (prediction error) | $(B, 16, 384)$ | float32 | **Yes** | $N=(56/14)^2=16$ |
| $\Pi_t$ | $(B,)$ | float32 | **Yes** | `PrecisionFunction.forward` |
| $A_t$ gaze history entry | $(B, 2)$ | float32 | **No** | Detached at `GazeState.record` |
| $t$ (glimpse index) | int | — | — | `GazeState.current_glimpse_idx` |
| $S_t$ | **None** | — | — | Locked to None (Gen-1 core) |

## 3. Visual input and foveation tensors

| Symbol | Shape | Dtype | Gradient | Source |
|:---:|:---|:---:|:---:|:---|
| $x$ (full image) | $(B, 3, H, H)$ | float32 | Optional | Adversarial training may require |
| $a_t$ (gaze coords) | $(B, 2) \in [-1,+1]^2$ | float32 | **Yes** (training) | `foveation.py` convention |
| $x_\text{fov}$ (crop) | $(B, 3, 56, 56)$ | float32 | **Yes** | `foveal_sample` output |
| $\theta$ (affine) | $(B, 2, 3)$ | float32 | **Yes** | Constructed from $a_t$ |

## 4. Backbone tokens

| Symbol | Shape | Notes |
|:---:|:---|:---|
| Patch tokens | $(B, 16, 384)$ | $N = (56/14)^2 = 16$, $D = 384$ |
| CLS token | $(B, 1, 384)$ | Prefix token 0 |
| Register token | $(B, 1, 384)$ | Prefix token 1 |
| Full token seq. | $(B, 18, 384)$ | 2 prefix + 16 patch |
| Pooled $z_t$ | $(B, 384)$ | CLS position after refinement |

## 5. Predictor tensors

| Symbol | Shape | Gradient | Notes |
|:---:|:---|:---:|:---|
| Fourier features of $a_t$ | $(B, 32)$ | Yes | $n_\text{freq}=8$, 4 terms each |
| Conditioning input | $(B, D_z + 1 + 32)$ | Yes | Concatenation $[z_t ,\, U_t^\text{scalar} ,\, \text{Fourier}(a_t)]$ |
| Conditioning output | $(B, 384)$ | Yes | After cond MLP |
| Residual output | $(B, 384)$ | Yes | After res block |
| Predicted tokens | $(B, 16, 384)$ | **Yes** | NEVER detached |

## 6. AIS-v2 tensors

| Symbol | Shape | Gradient | Notes |
|:---:|:---|:---:|:---|
| Candidates | $(B, K, 2)$ | **No** | $K \in [4,8]$; detached coordinate records |
| Error map | $(B, 4, 4)$ | No | Token prediction-error surface (4×4 grid) |
| Saliency (t=0) | $(B, K)$ | No | Computed under `no_grad` |
| Reduction scores (t≥1) | $(B, K)$ | **Yes** | $H(U_t) - H(\hat{U}_k)$ |
| Gumbel weights | $(B, K)$ | Yes (via straight-through) | Training-time selection |
| Selected gaze | $(B, 2)$ | Yes (training) / No (inference) | |

## 7. Coordinate conventions (LOCKED)
- **Gaze coordinates**: $[-1, +1]^2$, origin at image center. Enforced in `foveal_sample`, `VectorBeliefState`, and `_validate_history_entry`.
- **Gaze maximum**: $|a_t| \le 0.9$ (clamped by `HeuristicCandidateSampler`).
- **Patch grid**: row-major, matching `backbone.py`'s `x[:, self.num_prefix_tokens:]` token extraction.

## 8. UpdateNet shapes

| Symbol | Shape | Gradient | Notes |
|:---:|:---|:---:|:---|
| $E_t^\text{pooled}$ | $(B, 384)$ | **Yes** | Mean of $(B, 16, 384)$ |
| UpdateNet input | $(B, 768)$ | Yes | Concatenation $[z_t ,\, E_t^\text{pooled}]$ |
| UpdateNet output | $(B, 384)$ | Yes | Bounded $[-0.1, +0.1]$ per component |
| $z_{t+1}$ | $(B, 384)$ | Yes | $z_t + \Pi_t \cdot \delta$ |

## 9. Key invariants (enforced in code)
1. **Batch consistency**: all tensors in $B_t$ share the same $B$ dimension. `VectorBeliefState.__init__` enforces this.
2. **Evidence non-negativity**: `U.evidence >= 0`. Enforced by `EvidentialHead` (softplus output, clamped).
3. **E_0 all-zero**: enforced by `VectorBeliefState.__init__` at `current_glimpse_idx == 0`.
4. **Gaze in bounds**: enforced by `foveal_sample`, `_validate_history_entry`, and `ConcreteGlimpseFeaturePredictor._validate`.
5. **UpdateNet output bound**: enforced by `tanh` scaling in `ConcreteUpdateNet.forward`.

## 10. Source references
- [`schema.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/schema.py): `RHANNXAConfig`
- [`interfaces.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/interfaces.py): `BeliefState` (shapes in docstrings)
- [`vector_belief.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/vector_belief.py): `VectorBeliefState`
- [`predictive_coding/interfaces.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/interfaces.py): `GlimpseFeaturePredictor`, `UpdateNet`
