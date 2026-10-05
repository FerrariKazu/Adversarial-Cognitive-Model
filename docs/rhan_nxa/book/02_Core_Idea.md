# Chapter 02 — The Core Idea: Memory Types and Explicit Belief States

> *Level 1–2 reading. The foundational separation between model weights, working activations, and perceptual belief states.*

---

## 1. In One Sentence

The core idea of RHAN-NXA is the rigorous conceptual and computational distinction between long-term parametric weights ($\theta$), transient computational working activations ($h_t$), and an explicit, interpretable perceptual belief state $B_t = (z_t, S_t, U_t, E_t, A_t)$.

---

## 2. Intuition

In conventional deep learning literature, the term "memory" is used loosely to describe completely different computational structures:

- A static ResNet is said to "remember" image classes because its weights encode training patterns.
- A recurrent neural network is said to have "memory" because its hidden layer activations persist from one time step to the next.
- A Transformer is said to possess "memory" because its self-attention keys and values store token history across context windows.

This conceptual blurring leads to flawed architecture design. A recurrent hidden state $h_t$ is merely an unconstrained, high-dimensional vector of intermediate numbers; it has no internal semantics, no explicit uncertainty calibration, and no separate error tracking. If a network hallucinates or gets attacked, its hidden state drifts silently without the system "knowing" that its predictions were disproven.

RHAN-NXA resolves this by enforcing a strict separation:

1. **$\theta$ (Learned Parameters)**: The frozen or trained weights of the neural network. They store long-term semantic knowledge across training episodes.
2. **$h_t$ (Computational Working State)**: The transient activations within the visual transformer during the processing of a single foveal glimpse. This state exists solely to compute local token interactions and is discarded between glimpses.
3. **$B_t$ (Explicit Perceptual Belief State)**: The structured 5-tuple state $B_t = (z_t, S_t, U_t, E_t, A_t)$. This is the model's explicit hypothesis about the external world, tracking what it sees ($z_t$), its structural decomposition ($S_t$, with $S_t = \text{None}$ in Gen-1 Core), its evidential uncertainty ($U_t$), its latent prediction error ($E_t$, with $E_0 := 0$ locked), and its spatial gaze trajectory ($A_t$, with $A_{t+1} = A_t \cup \{a_{t+1}\}$).

---

## 3. Concrete Example

Imagine a security guard monitoring a dimly lit warehouse:

- The guard's lifetime knowledge of what doors, crates, and intruders look like is **$\theta$**.
- The rapid neural firing in the guard's visual cortex as their eyes fixate on a specific dark corner for 200 milliseconds is **$h_t$**.
- The guard's explicit mental assessment—*"There is an unverified moving shadow near Door 4; my certainty is low (30%); my previous expectation of an empty corridor was violated by motion; I have already checked Doors 1 and 2"*—is **$B_t$**.
- Because the guard possesses $B_t$, they decide to point their flashlight ($a_{t+1}$) directly at Door 4. A feedforward network, lacking $B_t$, would simply output a static label for the entire room in a single flash and ignore the discrepancy.

---

## 4. The Belief State Architecture

![Figure 2. The Canonical Perceptual Belief State B_t = (z_t, S_t, U_t, E_t, A_t). In Generation-1, z_t provides the global holistic representation, S_t = None obeys a strict None-propagating contract, U_t provides closed-form epistemic uncertainty derived from Dirichlet evidence parameters, E_t carries prediction discrepancy (with E_0 := 0 locked), and A_t preserves the gaze history trajectory.](figures/belief_state.svg)

---

## 5. Why RHAN-NXA Needs It

Without $B_t$, active vision is impossible. A gaze policy cannot decide where to look next unless it knows what is currently uncertain ($U_t$). A predictive coding engine cannot compute an error unless it has an explicit prior expectation derived from $z_t$. Collapsing $\theta$, $h_t$, and $B_t$ into an amorphous recurrent hidden state destroys the ability to audit, diagnose, or calibrate perceptual inquiry.

---

## 6. Mathematical Formulation

The functional relationship separating the three states is defined by:

$$h_t = \text{RefinementBlock}_\theta\big(\text{PatchEmbed}_\theta(g_t)\big) \quad \text{[Computational Activation]}$$

$$z_t^{\text{raw}} = \text{Pool}(h_t) \quad \text{[Global Perceptual Feature]}$$

$$U_t = \text{EvidentialHead}_\theta(z_t) = \frac{K}{\sum_{k=1}^K (\alpha_{t, k} + 1)} \quad \text{[Evidential Uncertainty]}$$

$$E_t = \|g_t^{\text{obs}} - \text{Predictor}_\theta(z_{t-1}, a_t)\|_2 \quad \text{[Prediction Error]}$$

$$A_{t+1} = A_t \cup \{a_{t+1}\} \quad \text{[Gaze Trajectory Record]}$$

$$B_t = (z_t, S_t, U_t, E_t, A_t) \quad \text{[Canonical 5-Tuple Belief State]}$$

Notice that parameters $\theta$ parameterize the operations, working activations $h_t$ compute the forward pass, and the resulting belief state $B_t$ persists across the temporal sequence of observations $t \in \{0, \dots, T-1\}$.

---

## 7. Implementation Mapping

- $\theta$ resides in `torch.nn.Parameter` dictionaries across `CompactViT`, `ConcreteUpdateNet`, `EvidentialHead`, and `ConcreteGlimpseFeaturePredictor`.
- $h_t$ exists as local intermediate tensors inside `CompactViT.forward` and `TiedRecurrence.forward`.
- $B_t$ is instantiated as an explicit dataclass instance of `noesis_vision.beliefs.vector_belief.VectorBeliefState`.

---

## 8. Authoritative Tensor Representation

- $\theta$: Collection of weight tensors $\in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, total $\sim 23.3\text{M}$ base parameters.
- $h_t$: Intermediate activation tensor $\in \mathbb{R}^{B \times 16 \times 384}$.
- $B_t$: Named tuple containing $z_t \in \mathbb{R}^{B \times 384}$, $S_t = \text{None}$, and $U_t \in \mathbb{R}^{B \times 1}$.
- Realized Error $E_t \in \mathbb{R}^{B \times 16 \times 384}$ and History $A_t = (a_1, \dots, a_t)$.

---

## 9. Gradient Behavior and Isolation

- $\theta$ receives gradients accumulated across all time steps via backpropagation through time (BPTT).
- $h_t$ gradients are computed during the within-glimpse backwards pass.
- $B_t$ components have distinct gradient contracts: $z_t$ and $U_t$ are gradient-bearing, $E_t$ carries gradient through its predicted term only (observed term is detached), and $A_t$ is a detached coordinate history.

---

## 10. Scientific Status

- Concept Separation ($\theta \ne h_t \ne B_t$): **REQUIRED** architectural principle.
- Implementation: **LOCKED**.

---

## 11. Empirical Evidence

Directly verified by the pre-flight multi-group gradient isolation checks (`tests/test_preflight_dw.py`) and belief state integrity tests (`tests/test_drift_to_gradient_flow.py`).

---

## 12. Limitations and Scope Boundaries

The belief state $B_t$ is instantiated per image and discarded after $T = 4$ glimpses. It does not currently persist across distinct video frames or episodes (episodic memory is **DEFERRED**).

---

## 13. Related Components and System Cross-References

- Belief state implementation: [Chapter 04 — The Belief State $B_t$](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/04_Belief_State.md)
- Recurrence mechanisms: [Chapter 09 — Computational and Perceptual Recurrence](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/09_Recurrence.md)
- Figure Reference: [belief_state.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/belief_state.svg) and [recurrence.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/recurrence.svg).
