# Chapter 05 — Global Perceptual State z_t: Holistic Manifold Representation

## 1. In one sentence
$z_t$ is the dense $384$-dimensional vector that serves as the system's running hypothesis about what it is looking at: updated by each glimpse's observed features and used to condition gaze selection, prediction, and uncertainty estimation.

## 2. Intuition
Think of $z_t$ as an internal "working answer" that evolves across the glimpse sequence. At $t=0$ it is the system's best guess from the first central observation. At $t=1$, after the fovea saccades to an informative region and the prediction error has been computed, the update network shifts $z_1$ toward a revised hypothesis. By $t=3$, if the active investigation was effective, $z_3$ should encode a confident, specific perceptual representation substantially different from $z_0$.

## 3. Mathematical formulation
$z_t$ is updated recurrently:

$$z_{t+1} = z_t + \Pi_t \cdot \text{UpdateNet}(z_t, E_t)$$

where:
- $\Pi_t = g(U_t)$ is a scalar sensory precision derived from the current uncertainty state.
- $E_t$ is the latent prediction error (see Chapter 08).
- $\text{UpdateNet}$ is a learned gating/mixing network that determines both the *direction* and *magnitude* of the belief correction.

At $t=0$, since $E_0 := \mathbf{0}$, this simplifies to:

$$z_1 = z_0 + \Pi_0 \cdot \text{UpdateNet}(z_0, \mathbf{0})$$

This is not a no-op: `UpdateNet` can respond to the zeroed error and use $z_0$'s state to produce a meaningful initial update direction.

## 4. Why RHAN-NXA needs it
A single pooled embedding from a single feedforward pass encodes everything the model ever knows about an image. If that single pass is corrupted by adversarial noise, there is no recovery mechanism. $z_t$ instead accumulates evidence over a sequence: each subsequent glimpse either confirms, challenges, or refines the current hypothesis. The temporal accumulation provides redundancy: a single perturbed observation cannot definitively collapse the running belief.

## 5. Implementation
- **Origin**: $z_0$ is produced by `CompactViT.encode_glimpse()` as the CLS-token-equivalent pooled output after the trunk and tied refinement block.
- **Dimensionality**: $D_z = 384$ — set in `noesis_vision/models/backbone.py` as `D_Z = 384`. This is **locked** by the DINOv2-small-shaped architecture; it is recorded in `RHANNXAConfig.d_z` once a run constructs the backbone.
- **Update**: The recurrent update formula is realized in `noesis_vision/predictive_coding/update_net.py` (`ConcreteUpdateNet`). See Chapter 12 for the complete loop.
- **Readout**: `VectorBeliefState.as_tensor()` returns $z_t$ directly when $S_t = \text{None}$ (the core build).

## 6. Tensor specification

| Tensor | Shape | Dtype | Gradient |
|:---|:---|:---|:---|
| $z_t$ | $(B, 384)$ | float32 | **Always** |
| Foveal crop (input) | $(B, 3, 56, 56)$ | float32 | Yes (via STN) |
| Token features | $(B, N, 384)$, $N=16$ | float32 | Yes |
| Pooled output | $(B, 384)$ | float32 | Yes |

where $N = (56 / 14)^2 = 16$ patch tokens from the $56 \times 56$ foveal crop with patch size $14$.

## 7. Warm start
The `CompactViT` trunk (`patch_embed`, `blocks`, `norm`) is initialized from a DINOv2-small checkpoint. The warm-start match fraction must reach $\ge 0.95$ over trunk keys; otherwise the load is rejected entirely (no silent partial loading). The new-by-design `refinement` block and position/CLS/register tokens are excluded from the threshold denominator.

## 8. Scientific status
- **$D_z = 384$**: **LOCKED** (Agent C set this; it is the only width compatible with a clean DINOv2-small warm-start mapping).
- **Recurrent update of $z_t$**: **LOCKED** design for Generation-1.
- **UpdateNet architecture**: **EXPERIMENTAL CANDIDATE** (the implementation is the best-reasoned default, resolved by the core validation experiments).

## 9. Limitations
- $z_t$ is a global vector; it cannot explicitly represent multiple spatial slots or object-level bindings without $S_t$. This is the motivation for the deferred slot re-entry.
- The fixed $D_z = 384$ does not scale with image complexity; changing it would break the DINOv2 warm-start mapping.

## 10. Related components
- Produced by: [`backbone.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/backbone.py) (`CompactViT.encode_glimpse`)
- Stored in: [`vector_belief.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/vector_belief.py)
- Updated by: `update_net.py` (Chapter 12)
- Illustrated in Figure 1 (`system_overview.svg`) and Figure 5 (`recurrence.svg`).
- Connected to Chapters 04, 09 (recurrence), 12 (complete loop), and 13 (architecture).
