# Chapter 00 — Executive Overview: The Recurrent Cognitive Vision Architecture

> *Level 1–2 reading. A comprehensive high-level blueprint of RHAN-NXA Generation-1.*

---

## 1. In One Sentence

RHAN-NXA (Generation-1) is a neurocomputationally inspired visual architecture organized around an explicit, recurrent perceptual belief state $B_t = (z_t, S_t, U_t, E_t, A_t)$ that actively investigates scenes through foveated observations, latent prediction-error updates, and epistemic uncertainty reduction rather than performing a single passive feedforward classification pass.

---

## 2. Intuition

Standard deep neural networks process an image in a single feedforward flash: pixels enter at one end, pass through stacked convolutional or attention layers, and output category logits at the other. If an adversarial attacker perturbs a small set of pixels, or if an object is partially occluded, the network has no opportunity to second-guess its initial impression. It cannot move its eyes to a clearer patch, it cannot notice that its internal expectations were violated, and it cannot update its hypothesis.

RHAN-NXA models visual perception as an active temporal loop of inquiry. Like a biological observer inspecting a cluttered or degraded scene, RHAN-NXA maintains an explicit representation of what it currently believes ($z_t$), measures its own evidential uncertainty ($U_t$), predicts what features it expects to encounter at potential fixation points, directs a differentiable foveal sensor to gather high-acuity information ($a_{t+1}$ via AIS-v2), computes the discrepancy between expectation and reality ($E_t$), and refines its internal belief state over a sequence of $T = 4$ discrete glimpses.

---

## 3. System Architecture Diagram

![Figure 1. RHAN-NXA System Overview: The Active Investigation Architecture. The Generation-1 core combines differentiable STN foveation, a compact ViT substrate with tied within-glimpse token refinement, a shared glimpse predictor evaluating latent feature discrepancies, and an active information sampling gaze policy guided by Dirichlet uncertainty reduction.](figures/system_overview.svg)

---

## 4. Concrete Example

Consider viewing a camouflaged animal partially concealed behind foliage. 

- **Under a standard feedforward Vision Transformer (ViT)**: The entire image is processed once. Texture artifacts from the leaves confuse the patch embeddings, and the model pathologically outputs "tree foliage" with 99.8% confidence.

- **Under RHAN-NXA**:
  1. **Glimpse 0 ($t = 0$)**: The model takes an initial observation at the center. Its belief vector $z_0$ is uncommitted, its evidential uncertainty $U_0$ is high ($U_0 = 1.0$), and prediction error is structurally zero ($E_0 := 0$).
  2. **Glimpse 1 ($t = 1$)**: Guided by high entropy in ambiguous regions, the active information seeking policy (AIS-v2) saccades to an ambiguous boundary region. The predictor forecasts the visual tokens expected at that location. The foveal sensor crops a $56 \times 56$ patch at full resolution. An error $E_1$ between predicted and observed features is calculated.
  3. **Belief Update ($t = 1 \to 2$)**: Weighted by sensory precision $\Pi_1 = 1 - U_0$, an update network shifts $z_1$ toward "feline flank contour." The evidential uncertainty $U_1$ sharpens.
  4. **Glimpses 2–3 ($t = 2, 3$)**: Additional targeted saccades verify diagnostic features (an eye, an ear contour). By $t = 3$, the final belief state $B_3$ exhibits low evidential uncertainty and correctly identifies the camouflaged predator.

---

## 5. Why RHAN-NXA Needs It

Generation-0 demonstrated that simply stacking modules—such as slot-attention bottlenecks, pixel reconstruction autoencoders, or auxiliary loss penalties—does not produce robust visual intelligence. Without an explicit, recurrent state that tracks hypotheses, errors, and gaze history, stacked components either fight each other's gradients or induce pathological gradient masking. RHAN-NXA introduces an explicit typed belief state as the central computational object of the system.

---

## 6. Mathematical Formulation

The central conceptual loop of Generation-1 executes over discrete glimpse steps $t \in \{0, 1, \dots, T-1\}$:

$$\hat{g}_t = \text{Predictor}(z_{t-1}, a_t)$$

$$g_t^{\text{obs}} = \text{Encode}(\text{Foveate}(x, a_t))$$

$$E_t = \|g_t^{\text{obs}} - \hat{g}_t\|_2 \quad (\text{with } E_0 := 0)$$

$$\Pi_t = \max(1 - U_{t-1}, 10^{-4})$$

$$\Delta z_t = 0.1 \cdot \tanh(\text{UpdateNet}([z_{t-1}, \Pi_t \odot E_t^{\text{pooled}}]))$$

$$z_t = z_{t-1} + \Delta z_t$$

where the canonical 5-tuple belief state is $B_t = (z_t, S_t, U_t, E_t, A_t)$, with $S_t = \text{None}$ strictly enforced in Generation-1 Core via the None-propagating contract, $E_0 := 0$ as a locked boundary condition, and gaze history trajectory $A_{t+1} = A_t \cup \{a_{t+1}\}$.

---

## 7. Implementation Substrate

The Generation-1 core substrate consists of:

- `noesis_vision.core.schema.RHANNXAConfig`: The centralized configuration schema enforcing decision statuses and boundary invariants.
- `noesis_vision.beliefs.vector_belief.VectorBeliefState`: The concrete container implementing $B_t$ with $S_t = \text{None}$.
- `noesis_vision.models.backbone.CompactViT`: A $\sim 23.3\text{M}$ parameter visual transformer with tied within-glimpse refinement.
- `noesis_vision.models.foveation.foveal_sample`: Differentiable Spatial Transformer Network (STN) grid sampling.
- `noesis_vision.uncertainty.evidential_head.EvidentialHead`: Dirichlet evidential neural network producing $U_t$.
- `noesis_vision.predictive_coding.glimpse_predictor.ConcreteGlimpseFeaturePredictor`: Shared predictor evaluating latent token features and candidate gaze locations.
- `noesis_vision.gaze.ais_v2_policy.AISv2GazePolicy`: Active information-seeking policy scoring candidates via Dirichlet entropy reduction.

---

## 8. Authoritative Tensor Representation

- Input visual scene: $x \in \mathbb{R}^{B \times 3 \times 224 \times 224}$
- Gaze coordinates: $a_t \in [-1, 1]^{B \times 2}$
- Foveal crop: $g_t \in \mathbb{R}^{B \times 3 \times 56 \times 56}$
- Global belief vector: $z_t \in \mathbb{R}^{B \times 384}$
- Dirichlet evidence: $e_t \in \mathbb{R}_{\ge 0}^{B \times 100}$, parameters $\alpha_t = e_t + 1$
- Latent prediction error: $E_t \in \mathbb{R}^{B \times 16 \times 384}$, pooled to $E_t^{\text{pooled}} \in \mathbb{R}^{B \times 384}$
- Gaze history record: $A_t = (a_1, \dots, a_t)$

---

## 9. Gradient Flow and Parameter Isolation

- **Gradient-bearing tensors**: $z_t$ carries gradients throughout the recurrent loop; $U_t$ backpropagates into `EvidentialHead` and through precision modulation; $E_t$ propagates gradients into the predictor and `UpdateNet` (while the observed target features are detached); gaze selection propagates gradients during training via Gumbel-Softmax straight-through estimation.
- **Detached records**: Gaze coordinates stored in $A_t$ are explicitly detached coordinate records.
- **Boundary isolation ($t = 0$)**: $E_0 := 0$ is a zero tensor that injects zero gradient into the predictive stack on the first glimpse.

---

## 10. Scientific Status

- Architecture Paradigm: **LOCKED** (Generation-1 core).
- Research Question: **EXPERIMENTAL CANDIDATE** (Motivating hypothesis, not established fact).

---

## 11. Empirical Evidence and Confounds

- Supported by the 16-seed Model D crossover experiment (`report/lens_e1_analysis/E1_FULL_AUDIT_REPORT.md`), demonstrating $+9.79\text{ pp}$ robust accuracy over a scaled TRADES baseline ($34.02\%$ vs $24.23\%$ at PGD-100, $\epsilon = 0.094$).
- Supported by the AIS-v2 targeted smoke-gate correlation ($r \approx 0.706$, $n=512$), confirming that candidate uncertainty reduction correlates with policy choice.
- **Preserved Confound**: The isolated 16-seed accuracy contribution of AIS-v2 and belief-HPC remains **UNKNOWN** due to an infrastructure confound in Generation-0 (`_nx_trainer` hardcoded SBR).
- Structural state $S_t$ is locked to `None` in the core build; slot-based representations have not yet been validated.

---

## 12. Related Components and System Cross-References

- Substrate specifications: [Chapter 13 — System Architecture](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/13_Architecture.md)
- Perceptual Loop details: [Chapter 12 — Complete Perceptual Loop](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/12_Complete_Perceptual_Loop.md)
- Formal Decision Records: [Chapter 21 — DR-001 through DR-010](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/21_Decision_Records.md)
