# Chapter 09 — Computational and Perceptual Recurrence: Option C Hybrid Architecture

> *Level 2 reading. Disentangling fast within-glimpse token iterations from slow across-glimpse belief trajectories.*

---

## 1. In One Sentence

RHAN-NXA employs two nested recurrence loops: **within-glimpse refinement** (a Universal-Transformer-style tied-weight block applied 2–3 times per glimpse) and **across-glimpse investigation** ($T = 4$ sequential foveated observations per image), each locked in configuration.

---

## 2. Intuition

Most neural networks process a single input in a single forward pass. RHAN-NXA introduces two distinct levels of temporal repetition:

- **Within-glimpse (computational)**: After the $56 \times 56$ crop is patch-embedded, a single transformer block is applied 2–3 times in a loop to the same token sequence, using the **same parameter tensors** each iteration. This Universal-Transformer-style design allows the model to iteratively refine its internal token representations before pooling to $z_t$. The iteration count scales compute, not parameter count.

- **Across-glimpse (perceptual)**: After the within-glimpse refinement produces a $z_t$, the system selects a new fixation point via AIS-v2, foveates to it, and begins the next glimpse. This repeats for $T = 4$ glimpses. The across-glimpse loop is what allows the system to "investigate" rather than "classify in one shot."

---

## 3. Recurrence Temporal Architecture

![Figure 4. Computational vs. Perceptual Recurrence: Option C Hybrid Architecture. Across glimpses (t=0..3), the belief state evolves through foveated observation, prediction error, and precision updating. Within each glimpse, tokens circulate iteratively through a single shared Transformer encoder block (2–3 iterations), adding computational depth without parameter inflation.](figures/recurrence.svg)

---

## 4. Mathematical Structure

### Within-Glimpse Recurrence

Let $H^{(0)} = \text{PatchEmbed}(\text{Foveate}(x, a_t))$.

$$H^{(k)} = \text{RefinementBlock}_\theta(H^{(k-1)}), \quad k = 1, \dots, n_\text{iters}$$

$$z_t^{\text{raw}} = \text{Pool}(H^{(n_\text{iters})}), \quad \text{tokens}_t = H^{(n_\text{iters})}_{2:}$$

where `RefinementBlock` is the **same** `TransformerBlock` instance every iteration (tied weights). $n_\text{iters} \in \{2, 3\}$ (LOCKED).

### Across-Glimpse Recurrence

$$B_{t+1} = f(B_t, x, a_{t+1}), \quad t = 0, 1, 2, 3$$

where $f$ is the complete glimpse step: foveate → encode → predict error → compute precision → update belief → select next gaze. After $t = 3$, the final belief $B_3$ is used for classification.

---

## 5. Tied-Weight Enforcement

**"Same parameter tensors reused, not copies"** is enforced structurally:

`TiedRecurrence` holds **one** `TransformerBlock` instance and calls it $n_\text{iters}$ times. A forward+backward accumulates all iterations' gradients onto that single parameter set. This is tested by `tests/test_recurrent_vision_core.py::test_recurrent_tied_weights`: if iterations ever held copies, each copy would receive its own separate gradient and the test would catch the divergence.

---

## 6. Implementation Mapping

- **Within-glimpse**:
  - `TransformerBlock` (`recurrent_block.py`): Pre-LayerNorm transformer block with DINOv2-compatible naming (`norm1`, `attn.qkv`, `attn.proj`, `norm2`, `mlp.fc1`, `mlp.fc2`).
  - `TiedRecurrence` (`recurrent_block.py`): Holds one `TransformerBlock`, applies it $n_\text{iters}$ times; raises `ValueError` if called with $n_\text{iters} \notin \{2, 3\}$.
  - Contained in `CompactViT` as `self.refinement`; called via `self.refinement(x, self.num_refine_iters)`.

- **Across-glimpse**:
  - Orchestrated by the integration layer.
  - `num_glimpses = 4` is locked in `RHANNXAConfig`.
  - Adaptive halting is **DEFERRED** (gated by `step6_validated_result`).

---

## 7. Configuration Locking

Two separate enforcement points exist for the iteration count — both must stay synchronized:

| Enforcement Point | Code Location | Validation Rule |
|:---|:---|:---|
| **Schema-level** | `RHANNXAConfig.__post_init__` | `within_glimpse_iters ∈ {2, 3}` |
| **Model-level** | `TiedRecurrence.forward` | `num_iters ∈ [2, 3]` |

Similarly, `num_glimpses = 4` is locked in the schema:

```python
if self.enable_recurrence and self.num_glimpses != 4:
    raise ValueError("num_glimpses=... T=4 is LOCKED for the Gen-1 core ...")
```

---

## 8. Why Tied Weights?

Tied weights provide a critical property: **iteration count is a compute budget choice, not a parameter count choice**. The number of parameters is identical whether the block runs 2 or 3 times. This enables fair comparison between recurrence depths without confounding parameter count—an essential scientific hygiene property.

---

## 9. Why $T = 4$ Glimpses?

$T = 4$ is reused from the validated STL-10 project convention (Generation-0). It is not derived from first principles—it is a pragmatic starting point for the core validation. Adaptive halting (where $T$ varies per image) is technically possible but is deferred until the fixed-depth loop is validated, because adaptive halting has *"a documented history of fighting other objectives"* (schema.py).

---

## 10. Scientific Status

- **Tied within-glimpse weights**: **LOCKED** (Hybrid Option C, Part 1.C).
- **Within-glimpse iteration count 2–3**: **LOCKED** (Part 1.C).
- **$T = 4$ across-glimpse**: **LOCKED** (Part 1.C, Gen-1 core).
- **Adaptive halting**: **DEFERRED** (gated by `step6_validated_result`).

---

## 11. Related Components and System Cross-References

- Recurrent module: [noesis_vision/models/recurrent_block.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/recurrent_block.py)
- Substrate integration: [noesis_vision/models/backbone.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/backbone.py)
- Figure Reference: [recurrence.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/recurrence.svg).
