# Chapter 04 — The Belief State B_t: Explicit 5-Tuple Perceptual Architecture

> *Level 2 reading. The central typed data structure governing RHAN-NXA's running perceptual hypothesis.*

---

## 1. In One Sentence

The belief state $B_t = (z_t, S_t, U_t, E_t, A_t)$ is RHAN-NXA's central typed data structure: an explicit, immutable internal representation tracking what the visual system perceives ($z_t$), its structural composition ($S_t$), its evidential uncertainty ($U_t$), its latent prediction discrepancy ($E_t$), and its spatial gaze history trajectory ($A_t$).

---

## 2. Intuition

A classical neural network has no explicit memory of its own internal state between layer activations. All context is latent, entangled, and opaque. RHAN-NXA replaces this with a named, typed record that is passed from one glimpse to the next. Every component of the belief state has a defined mathematical role, a defined shape, a defined gradient rule, and a defined initialization condition. Nothing about the system's running hypothesis is implicit.

---

## 3. Belief State Visual Architecture

![Figure 2. The Canonical Perceptual Belief State B_t = (z_t, S_t, U_t, E_t, A_t). The formal dataclass encapsulates global perceptual state z_t, structural state S_t (None in Gen-1 Core), Dirichlet epistemic uncertainty U_t, prediction error E_t (with E_0 := 0 locked), and gaze trajectory A_t, enforcing None-propagation and sample-wise differentiability for trajectory drift.](figures/belief_state.svg)

---

## 4. Field-by-Field Description

| Field | Symbol | Shape | Gradient? | Role |
|:---|:---:|:---|:---:|:---|
| **Global Perceptual State** | $z_t$ | `(B, 384)` | **Always** | Dense continuous summary of perceived visual content at step $t$ |
| **Structure State** | $S_t$ | `None` (Core) | Only when not None | Slot-based object representation; **locked to None** in Gen-1 Core |
| **Epistemic Uncertainty** | $U_t$ | `(B, 1)` $\in (0, 1]$ | **Always** | Dirichlet uncertainty from `EvidentialHead`; feeds precision $\Pi_t$ |
| **Prediction Error** | $E_t$ | `(B, 16, 384)` | **Always** | Token-space discrepancy between expected and observed glimpse features ($E_0 := 0$) |
| **Gaze History Record** | $A_t$ | `(a_1, ..., a_t)` | **None** | Immutable coordinate history of past fixations; detached from graph ($A_{t+1} = A_t \cup \{a_{t+1}\}$) |

---

## 5. Why RHAN-NXA Needs It

Generation-0 treated internal state as implicit activations passed through a fixed computational graph. When the graph changed, the meaning of the activations changed with no formal interface boundary. RHAN-NXA installs an explicit, typed interface contract. Every agent in the codebase that reads or writes the belief state must do so through the `BeliefState` abstract base class (`noesis_vision/beliefs/interfaces.py`), making it impossible to silently change what the belief state means for one consumer without breaking all others.

---

## 6. Mathematical Formulation

The belief state evolves per-glimpse:

$$B_0 = (z_0, S_0 = \text{None}, U_0 = 1.0, E_0 := 0, A_0 = \emptyset)$$

$$B_{t+1} = (z_{t+1}, S_{t+1} = \text{None}, U_{t+1}, E_{t+1}, A_{t+1})$$

where the update from $B_t$ to $B_{t+1}$ is governed by:

$$\Pi_t = \max(1 - U_t, 10^{-4})$$

$$\Delta z_t = 0.1 \cdot \tanh(\text{UpdateNet}([z_t, \Pi_t \odot E_t^{\text{pooled}}]))$$

$$z_{t+1} = z_t + \Delta z_t$$

$$U_{t+1} = \frac{K}{\sum_{k=1}^K (\alpha_{t+1, k} + 1)}$$

$$E_{t+1} = \|g_{t+1}^{\text{obs}} - \hat{g}_{t+1}^{\text{pred}}\|_2$$

$$A_{t+1} = A_t \cup \{a_{t+1}\}$$

---

## 7. Implementation Substrate

- **Interface**: `noesis_vision.beliefs.interfaces.BeliefState` — abstract base class defining every member with shapes in docstrings.
- **Concrete class**: `noesis_vision.beliefs.vector_belief.VectorBeliefState` — the $S_t = \text{None}$ Gen-1 core implementation.
- **Diagnostic copy**: `VectorBeliefState.detached_copy()` — graph-free clone for diagnostics that can never corrupt the live training path.

---

## 8. Gradient Rules (LOCKED)

These rules are strictly enforced by `VectorBeliefState.__init__`:

1. **$z_t$**: carries gradients **always** — it is the primary update target.
2. **$S_t$**: carries gradients only when not None.
3. **$U_t$**: carries gradients always — precision $\Pi_t$ must backpropagate into the `EvidentialHead`.
4. **$E_t$**: carries gradients always — the predicted path (never the observed target) is left attached. Detaching $E_t$ before the update is the **single most repeated failure mode** in Gen-0's history.
5. **$A_t$**: carries **no gradient** — it is an immutable detached coordinate record.

---

## 9. The $E_0 := 0$ Boundary Condition (LOCKED)

At $t = 0$, there is no predecessor belief to compute a prediction error from. The `VectorBeliefState` constructor **structurally enforces** this: if `current_glimpse_idx == 0` and $E$ is not all-zero, construction raises `ValueError`. This ensures no component can inherit a different first-glimpse convention accidentally.

---

## 10. Scientific Status

- **Belief State as a typed interface**: **LOCKED** (Generation-1 architectural decision).
- **$S_t = \text{None}$ in the core build**: **LOCKED** per Part 1.D.
- **Slot-based $S_t$**: **REJECTED** as currently implemented (Gen-0 16-slot); **DEFERRED** (2–4 slots, post Step-6 validation).
- **Representation-level uncertainty**: **PENDING DECISION** (class-readout Dirichlet kept for Gen-1).

---

## 11. The None-Safety Invariant

Every consumer of a `BeliefState` must have an **explicit branch** for `S_t is None`. This includes:

- `as_tensor()`: returns $z_t$ when $S_t = \text{None}$; raises `NotImplementedError` for the $S_t \ne \text{None}$ path.
- `drift_to()`: uses $z_t$ only for the core build.
- The classifier head.
- The integration layer.

This is documented in the module header as: *"not a hypothetical: it is the actual default configuration."*

---

## 12. Related Components and System Cross-References

- Belief interface: [noesis_vision/beliefs/interfaces.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/interfaces.py)
- Concrete implementation: [noesis_vision/beliefs/vector_belief.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/vector_belief.py)
- Connected to: [Chapter 05 ($z_t$)](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/05_z_State.md), [Chapter 07 ($U_t$)](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/07_Uncertainty.md), [Chapter 08 ($E_t$)](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md), and [Chapter 11 ($A_t$)](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md).
