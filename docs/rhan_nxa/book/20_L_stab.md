# Chapter 20 — L_stab: Belief Stability Diagnostic and Staged Objective

> *Level 2–3 reading. A mechanism whose most defining architectural property is when and under what preconditions it is allowed to act.*

---

## 1. In One Sentence

$\mathcal{L}_{\text{stab}}$ is a trajectory-stability penalty over belief states—**locked to a staged protocol** in Generation-1 as a diagnostic-only metric first, promotable to an active training objective only after the clean core architecture achieves a validated ImageNet-100 baseline.

---

## 2. The Intuition

Adversarial perturbations do not merely manipulate the final classification output of a neural network; they fundamentally derail *how* the model perceives. In a recurrent cognitive architecture like RHAN-NXA, an adversarial attack can violently destabilize the trajectory of internal belief states: a tiny norm-bounded input perturbation causes the internal sequence of representations $(B_1, B_2, B_3, B_4)$ to diverge wildly from the trajectory induced by the corresponding clean input.

$\mathcal{L}_{\text{stab}}$ measures this divergence: *how similar is the sequence of belief states under an adversarially perturbed image to the sequence under the clean image?*

While this formulation presents an immediately attractive training loss, introducing it prematurely creates an insurmountable scientific confound. If stability is optimized during core training, researchers cannot separate the causal contributions of the recurrent predictive coding loop from the regularization effect of the trajectory penalty. RHAN-NXA enforces strict staging to ensure single-mechanism attribution.

```
Clean Image   x   ───► [Recurrent Loop] ───► Trajectory B_t(x)   ──┐
                                                                    ├─► drift_to() ──► L_stab
Perturbed x + δ  ───► [Recurrent Loop] ───► Trajectory B_t(x+δ) ──┘
```

---

## 3. Mathematical Formulation

Let $B_t(x) = (z_t(x), S_t(x), U_t(x))$ denote the belief state at glimpse step $t \in \{1, \dots, T\}$ for clean input $x$, and let $B_t(x + \delta)$ denote the belief state for perturbed input $x + \delta$.

The drift between the two belief states is evaluated by the metric function $\text{drift\_to}(B_t(x), B_t(x + \delta))$ defined over the latent representations:

$$\mathcal{L}_{\text{stab}} = \frac{1}{T} \sum_{t=1}^T \text{drift}(B_t(x + \delta), B_t(x))$$

### Latent Component Distances

In the Gen-1 core ($S_t = \text{None}$), the drift is computed purely over the global perceptual state $z_t \in \mathbb{R}^{B \times D_z}$:

1. **Euclidean ($L_2$) Distance**:
   $$d_{z, L_2}(z, z') = \|z - z'\|_2$$
2. **Cosine Distance**:
   $$d_{z, \cos}(z, z') = 1 - \frac{z \cdot z'}{\|z\|_2 \|z'\|_2}$$

Both distance formulations are supported in code, with the definitive selection staged for empirical evaluation.

When weighted across components:
$$\text{drift}(B, B') = w_z \cdot d_z(z, z') + w_s \cdot d_s(S, S')$$
where default neutral weights are $w_z = 1.0, w_s = 0.0$ while $S_t = \text{None}$.

---

## 4. The Staged Protocol (LOCKED)

The staging rule for $\mathcal{L}_{\text{stab}}$ is strictly registered in the project master plan:

```
┌────────────────────────────────────────────────────────────────────────┐
│ (A) PHASE A: DIAGNOSTIC ONLY                                           │
│     Active through the entire core build and first ablation matrix.    │
│     Computed and logged per step/epoch; NEVER optimized against.       │
│     Gradients do not flow; loss coefficient λ_stab = 0.               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Core ImageNet-100 baseline
                                    │ validated and frozen (Step 6)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ (B) PHASE B: TRAINING OBJECTIVE                                        │
│     Promoted ONLY as an isolated ablation arm (Step 8 / +L_stab arm).   │
│     Trained with λ_stab > 0 against the Step 6 frozen checkpoint.      │
│     Must pass Gate 9 Responsiveness Guard before acceptance.           │
└────────────────────────────────────────────────────────────────────────┘
```

### Why Staging Is Non-Negotiable

If $\mathcal{L}_{\text{stab}}$ were enabled during initial training, the resulting robustness could stem either from the recurrent predictive coding loop or from the trajectory contrastive penalty. Prematurely combining them reproduces the exact attribution failures identified in Generation-0 (where the unvalidated 16-slot SBR masked the true performance of surrounding modules).

---

## 5. The Responsiveness Guard (Gate 9 Hard Rule)

The most critical principle governing $\mathcal{L}_{\text{stab}}$ is its two-sided failure guard:

> **"A model minimizing adversarial belief drift by becoming insensitive to ALL new evidence is a FAILURE, not a success."**

A trivial and degenerate way for a model to minimize belief drift between $x$ and $x + \delta$ is to collapse its dynamics—producing constant, invariant representations regardless of input changes. Under such collapse:
- Adversarial drift approaches zero: $\text{drift}(B_t(x+\delta), B_t(x)) \to 0$.
- However, sensitivity to genuine semantic updates also collapses to zero.

### Gate 9 Formulation

Under **Gate 9**, any evaluation of $\mathcal{L}_{\text{stab}}$'s adversarial drift reduction must be reported **alongside an Out-of-Distribution (OOD) / Novel-Evidence Responsiveness Score**, evaluated on a strictly disjoint probe set:

$$\text{Resp}(x_{\text{base}}, x_{\text{novel}}) = \frac{1}{T} \sum_{t=1}^T \text{drift}(B_t(x_{\text{novel}}), B_t(x_{\text{base}}))$$

```
                                  Gate 9 Outcome Matrix
                       ┌────────────────────────┬────────────────────────┐
                       │ High OOD Responsive    │ Low OOD Responsive     │
┌──────────────────────┼────────────────────────┼────────────────────────┤
│ Low Adversarial Drift│        PASSED          │    FAILED (TERMINAL)   │
│                      │ Genuine Robust State   │ Pathological Invariance│
├──────────────────────┼────────────────────────┼────────────────────────┤
│ High Adversar. Drift │        FAILED          │        FAILED          │
│                      │ Vulnerable Trajectory  │ Complete Dysfunction   │
└──────────────────────┴────────────────────────┴────────────────────────┘
```

A model showing low drift on *both* adversarial inputs and novel semantic inputs is graded **FAILED, full stop**. This is a pre-registered, terminal stopping condition that cannot be rationalized post hoc.

---

## 6. Implementation Reference

The stability metric is implemented in [noesis_vision/beliefs/drift.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/drift.py) as an interface method consumable by Agent H's objective pipeline:

```python
# noesis_vision/beliefs/drift.py

DISTANCES = ("l2", "cosine")
DEFAULT_WEIGHTS = {"z": 1.0, "s": 0.0}

def drift_to(belief, other, weights: Optional[Mapping[str, float]] = None,
             distance: str = "l2") -> torch.Tensor:
    """Weighted per-sample drift between two beliefs, shape (B,).
    
    Returns a differentiable (B,) tensor w.r.t. belief.z (and other.z).
    """
    if distance not in DISTANCES:
        raise ValueError(f"distance must be one of {DISTANCES}; got {distance!r}")

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    if belief.z.shape != other.z.shape:
        raise ValueError(f"z shape mismatch: {tuple(belief.z.shape)} vs {tuple(other.z.shape)}")

    if distance == "l2":
        d_z = torch.norm(belief.z - other.z, p=2, dim=-1)
    else:  # cosine
        d_z = 1.0 - F.cosine_similarity(belief.z, other.z, dim=-1)

    drift = w["z"] * d_z

    if belief.s is None and other.s is None:
        pass  # Zero-contribution placeholder for S=None core build
    else:
        raise NotImplementedError("Structured S-term drift requires S_t re-entry (Part 2 step 7)")

    return drift
```

### Resolved Contract Discrepancy

The preliminary contract draft specified `drift_to(other, weights) -> float`. However:
1. `L_stab` must eventually serve as a differentiable training loss in Phase B.
2. A Python `float` severs the autograd computation graph.
3. Returning shape `(B,)` allows sample-wise weighting and prevents premature batch reduction.

Therefore, `drift_to()` strictly returns a `torch.Tensor` of shape `(B,)`. Callers requiring a scalar logging value must explicitly call `.detach().mean().item()` on a detached copy.

---

## 7. Tensor Specification

| Tensor / Variable | Shape | Dtype | Differentiable? | Description |
|---|---|---|---|---|
| `belief.z` | `(B, 384)` | `float32` | Yes | Clean/target latent perceptual state |
| `other.z` | `(B, 384)` | `float32` | Optional | Perturbed/reference state (graph retained if optimizing pairwise) |
| `drift` | `(B,)` | `float32` | Yes | Sample-wise trajectory drift metric |
| `L_stab` (Scalar) | `()` | `float32` | Yes | Batch-averaged loss $\frac{1}{B} \sum_{i=1}^B \text{drift}_i$ |

---

## 8. Gradient Behavior and Differentiability

- In **Phase A (Diagnostic)**:
  $$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mathcal{L}_{\text{stab}}} = 0$$
  The trajectory drift is evaluated inside `torch.no_grad()` or explicitly detached before logging.
- In **Phase B (Objective)**:
  $$\nabla_z \mathcal{L}_{\text{stab}} = \nabla_z \left( \frac{1}{B} \sum_{i=1}^B \|z_i(x + \delta) - z_i(x)\|_2 \right)$$
  Gradients propagate through `belief.z` back through the recurrent ViT backbone and patch embedder. When generating adversarial perturbations $\delta$, gradients flow through the inner maximization loop:
  $$\delta^* = \arg\max_{\|\delta\|_p \le \epsilon} \left( \mathcal{L}_{\text{task}}(x + \delta) + \gamma \mathcal{L}_{\text{stab}}(x + \delta, x) \right)$$

---

## 9. Scientific Status and Decision Table

| Dimension | Specification | Scientific Status |
|---|---|---|
| Staged Protocol | Diagnostic first $\to$ Objective after Step 6 validation | **LOCKED** |
| Mechanism Class | Trajectory contrastive regularization | **EXPERIMENTAL CANDIDATE** |
| Gate 9 Two-Sided Rule | Joint evaluation with OOD novel-evidence probe | **REQUIRED as designed** |
| Gate 9 Thresholds | Cutoff for adversarial reduction & OOD sensitivity | **PENDING DECISION** (Calibrated from Phase A baseline) |
| Distance Metric | Selection between $L_2$ and Cosine distance | **PENDING DECISION** (Configurable in `drift.py`) |

---

## 10. What This Does NOT Mean

1. **"Diagnostic-only" does not mean trivial or unbuilt**: `drift_to()` is fully implemented, verified, and logged from Epoch 1 of the core build. It simply exerts zero gradient force on the model parameters.
2. **Gate 9 is not a post-hoc filter**: It is an active stopping condition. A model with zero drift that fails the responsiveness floor is discarded immediately; results cannot be rescued by claiming "robust invariance."
3. **$\mathcal{L}_{\text{stab}}$ promotion is not inevitable**: If the Step 6 clean core baseline fails validation, or if Phase B training induces parameter collapse, $\mathcal{L}_{\text{stab}}$ remains permanently a diagnostic metric.

---

## 11. Limitations and Open Questions

1. **Computational Overhead of Trajectory Computation**: Evaluating $\mathcal{L}_{\text{stab}}$ during training requires running forward passes on both $x$ and $x + \delta$ across all $T$ timesteps, doubling recurrent forward FLOPs during Phase B.
2. **Calibration of the Responsiveness Floor**: The minimum acceptable responsiveness threshold $\tau_{\text{resp}}$ cannot be chosen arbitrarily; it must be empirically calibrated from the clean core's variance across validation classes.
3. **Structured S-Term Integration**: When the structure state $S_t$ is reintroduced in future generations, an appropriate metric for slot-permutation-invariant drift must be defined and validated.

---

## 12. Related Components and System Context

- [Chapter 04 — The Belief State $B_t$](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/04_Belief_State.md): Defines the state interface and the `drift_to` contract.
- [Chapter 09 — Recurrence](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/09_Recurrence.md): Specifies the $T=4$ temporal dynamics evaluated by stability metrics.
- [Chapter 14 — Multi-Group Gradient Flow](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/14_Gradient_Flow.md): Tracks parameter update isolation and Phase B gradient paths.
- [Chapter 17 — Experimental DAG](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/17_Experimental_DAG.md): Formally defines the Step 6 $\to$ Step 8 gating dependency.
- Diagram Reference: [system_overview.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/system_overview.svg), [training_dag.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/training_dag.svg).
