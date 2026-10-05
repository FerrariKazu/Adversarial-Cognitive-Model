# Chapter 12 — The Complete Perceptual Loop: Synthesis of Components

> *Level 2 reading. The central orchestration of foveation, predictive coding, precision modulation, and active sensing.*

---

## 1. In One Sentence

The complete perceptual loop is the orchestrated sequence of operations executed by the integration layer across $T = 4$ glimpses: foveate → encode → predict → compute error → compute precision → update belief → select next gaze, resulting in a final belief state $B_3$ used for classification.

---

## 2. The Perceptual Loop Architecture (HERO)

![Figure 3. The Complete Perceptual Investigation Loop (HERO). Over discrete steps t in {0..3}, RHAN-NXA executes an 8-stage cycle: maintaining belief state B_t, predicting expected features at prospective saccade targets, observing high-acuity foveal patches, evaluating prediction error E_t, precision weighting Π_t, bounded state updating via UpdateNet, Dirichlet evidential readout, and active gaze selection via AIS-v2.](figures/perceptual_loop.svg)

---

## 3. The Full Loop, Step by Step

### Initialization ($t = 0$)

1. **Initial Gaze Fixation**: $a_1 = (0, 0)$ (image center, canonical starting coordinate).
2. **Encode Initial Glimpse**: $(z_0^{\text{raw}}, \text{tokens}_0) = \text{CompactViT.encode\_glimpse}(x, a_1)$.
3. **Initial Uncertainty**: $U_0 = \text{EvidentialHead}(z_0^{\text{raw}})$.
4. **Initial Prediction Error**: $E_0 := \mathbf{0}$ (LOCKED; boundary condition enforced).
5. **Initial Gaze History**: $A_0 = \text{GazeState}().\text{record}(a_1.\text{detach}())$.
6. **Construct Initial Belief**: $B_0 = \text{VectorBeliefState}(z_0^{\text{raw}}, S_0 = \text{None}, U_0)$.

### Glimpse Step ($t = 1 \to 2$ and $t = 2 \to 3$)

7. **Propose Candidate Fixations**: AIS-v2 samples $K \in \{4, \dots, 8\}$ candidate locations $c_k \in [-1, 1]^2$.
8. **Forecast Expected Features**: $\hat{g}^{(k)} = \text{Predictor}(z_{t-1}, c_k)$ for each candidate.
9. **Score Information Gain**: Evaluate analytical Dirichlet entropy reduction $r_k = \Delta H(U_{t-1}, \hat{g}^{(k)})$.
10. **Select Next Gaze Location**: $a_{t+1} = \text{AISv2Policy}(r_1, \dots, r_K)$.
11. **Observe Foveal Patch**: $(z_t^{\text{raw}}, \text{tokens}_t) = \text{CompactViT.encode\_glimpse}(x, a_{t+1})$.
12. **Compute Discrepancy Error**: $E_t = \|\text{tokens}_t.\text{detach}() - \hat{g}_t\|_2$.
13. **Compute Sensory Precision**: $\Pi_t = \max(1 - U_{t-1}, 10^{-4})$.
14. **Bounded State Update**: $\Delta z_t = 0.1 \cdot \tanh(\text{UpdateNet}([z_{t-1}, \Pi_t \odot E_t^{\text{pooled}}]))$; $z_t = z_{t-1} + \Delta z_t$.
15. **Update Evidential Readout**: $U_t = \text{EvidentialHead}(z_t)$.
16. **Advance Gaze Record**: $A_t = A_{t-1}.\text{record}(a_{t+1}.\text{detach}()).\text{advance}()$.
17. **Construct Updated Belief**: $B_t = \text{VectorBeliefState}(z_t, S_t = \text{None}, U_t)$.
18. Loop until $t = T-1 = 3$.

### Final Classification Readout

19. **Final Readout**: $\text{logits} = \text{ClassifierHead}(B_3.z)$.
20. **Loss Computation**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(y, \hat{y}) + \lambda_{\text{pred}} \mathcal{L}_{\text{pred}} + \lambda_{\text{stab}} \mathcal{L}_{\text{stab}}$ (where $\lambda_{\text{stab}} = 0$ in Phase A).
21. **Isolated Optimizer Step**: `MultiGroupOptimizer.step()` with group-specific gradient clipping.

---

## 4. Gradient Flow Invariants

- **Detached paths**: Observed foveal tokens ($g_t^{\text{obs}}$ target), coordinates in $A_t$, candidate proposals.
- **Gradient-bearing paths**: $z_t$, evidence parameters $\alpha_t$, $E_t$ (via predicted features $\hat{g}_t$), precision $\Pi_t$, and AIS-v2 soft selection weights during training.

---

## 5. Implementation Mapping

The individual modules are implemented and contract-tested across the codebase:

- `CompactViT.encode_glimpse`: Foveal extraction and ViT processing.
- `VectorBeliefState`: Belief container enforcing the None-contract.
- `EvidentialHead`: Dirichlet evidence and epistemic uncertainty.
- `ConcreteGlimpseFeaturePredictor`: Shared patch-level predictor.
- `ConcreteUpdateNet`: Non-linear state update bounded by 0.1.
- `AISv2GazePolicy`: Active candidate generation and entropy reduction scoring.
- `GazeState`: Immutable coordinate history.

---

## 6. Scientific Status

- **Perceptual loop architecture**: **LOCKED** (Generation-1 design).
- **UpdateNet-based update**: **EXPERIMENTAL CANDIDATE** (Under DAG validation).
- **Precision-gated update**: **EXPERIMENTAL CANDIDATE**.
- **AIS-v2 isolated contribution**: **UNKNOWN** (Gen-0 SBR confound).

---

## 7. Related Components and System Cross-References

- System Overview: [Chapter 00 — Executive Overview](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/00_Executive_Overview.md)
- Gradient flow specifics: [Chapter 14 — Multi-Group Gradient Flow](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/14_Gradient_Flow.md)
- Figure Reference: [perceptual_loop.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/perceptual_loop.svg).
