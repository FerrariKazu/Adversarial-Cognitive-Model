# Chapter 03 — Perception as Investigation: Moving Beyond Feedforward Classification

> *Level 1–2 reading. The conceptual shift from instantaneous classification to iterative, hypothesis-driven exploration.*

---

## 1. In One Sentence

Perception as investigation is the foundational research hypothesis motivating RHAN-NXA: visual intelligence achieves adversarial robustness by actively seeking informative sensory evidence to resolve internal uncertainty rather than passively mapping static retinal patterns to class labels in a single feedforward pass.

---

## 2. Intuition

In classical computer vision benchmarks (ImageNet, CIFAR-10), an image is presented as an inert, static grid of pixels. The algorithm is evaluated as a passive classifier: it receives the entire image simultaneously and must make an instantaneous categorical declaration.

Biological vision evolved under completely different physical constraints. Human eyes possess a high-acuity central fovea spanning only about 2 degrees of the visual field, surrounded by a low-resolution periphery. Humans do not see an entire scene at 4K resolution simultaneously; instead, the brain makes 3 to 4 rapid, subconscious saccades per second. Each eye movement is an active perceptual probe: the brain forms a working hypothesis, identifies regions where its predictive model is uncertain or contradicted, and moves the fovea to sample high-frequency information that confirms or falsifies the hypothesis.

RHAN-NXA operationalizes this biological strategy. It treats an image not as an array to be ingested, but as an environment to be actively explored.

---

## 3. The Active Investigation Loop

![Figure 3. The Complete Perceptual Investigation Loop (HERO). Over discrete steps t in {0..3}, RHAN-NXA executes an 8-stage cycle: maintaining belief state B_t, predicting expected features at prospective saccade targets, observing high-acuity foveal patches, evaluating prediction error E_t, precision weighting Π_t, bounded state updating via UpdateNet, Dirichlet evidential readout, and active gaze selection via AIS-v2.](figures/perceptual_loop.svg)

---

## 4. Concrete Example

Consider classifying an ambiguous image that could be an *automobile* or a *truck*:

- **Feedforward Classifier**: The model computes global texture statistics. If an adversarial perturbation adds high-frequency noise resembling truck grille texture to a car, the classifier immediately collapses to "truck."

- **Perceptual Investigator (RHAN-NXA)**:
  - At $t = 0$, the model's central glimpse sees the vehicle cabin. Its evidential state $U_0$ indicates high ambiguity between automobile and truck.
  - Rather than guessing, the active policy recognizes that the visual evidence is insufficient. It computes the expected information gain of looking toward the rear bed versus the front wheels.
  - At $t = 1$, the fovea saccades to the rear cargo bed. High-acuity observation reveals an open pickup bed.
  - The prediction error between expected trunk contour and observed open bed updates the belief state $z_2$, sharply reducing evidential entropy and confirming "truck" based on structural geometry rather than brittle surface texture.

---

## 5. Why RHAN-NXA Needs It

The central finding of our psychophysics study (`FINDINGS.md: Finding 1`) was that feedforward vision models undergo catastrophic perceptual collapse ($d' < 1.0$) at $\epsilon < 0.03$, whereas humans maintain sensitivity beyond $\epsilon = 0.30$. Passive scaling of parameters (e.g., from ResNet to ViT to EfficientNet) does not bridge this $10\times$ gap; in fact, higher clean accuracy often correlates with increased fragility (`Finding 2`). Transforming perception from passive classification into active temporal investigation provides the dynamic degrees of freedom required to actively reject adversarial noise.

---

## 6. Mathematical Formulation

Under the active inference formulation, the selection of the next observation $a_{t+1}$ is governed by the maximization of expected information gain (reduction in entropy of the perceptual belief):

$$a_{t+1} = \arg\max_{a \in \mathcal{A}} \; \mathbb{E}_{p(\tilde{x} \mid B_t, a)}\Big[ H(U_t) - H\big(U(B_t \mid \tilde{x}, a)\big) \Big]$$

where $H(U_t)$ is the differential entropy of the Dirichlet distribution parameterizing the evidential state at time $t$, and $U(B_t \mid \tilde{x}, a)$ is the updated uncertainty resulting from hypothetical observation $\tilde{x}$ at fixation $a$.

---

## 7. Implementation Mapping

- The candidate observation generator is implemented in `noesis_vision.gaze.candidate_sampler.HeuristicCandidateSampler`.
- The expected information gain scoring is computed in `noesis_vision.gaze.ais_v2_policy.AISv2GazePolicy` via the method `_score_candidates_t_ge_1`.
- Differentiable observation sampling is performed by `noesis_vision.models.foveation.foveal_sample`.

---

## 8. Authoritative Tensor Representation

- Action space: $a_t \in [-1, 1]^{B \times 2}$ (Continuous 2D coordinates).
- Candidate set: $\mathcal{C}_t \in \mathbb{R}^{B \times K \times 2}$ ($K \in [4, 8]$ candidate fixations).
- Candidate reduction score: $r_k \in \mathbb{R}^{B \times K}$ representing expected entropy reduction.

---

## 9. Gradient Behavior and Policy Backprop

During training, the discrete selection $\arg\max_k r_k$ is relaxed using Gumbel-Softmax with temperature scaling or a Straight-Through Estimator (STE). This enables gradients from downstream classification and prediction error to backpropagate directly into the gaze policy parameters (`logit_scale`) and the shared glimpse predictor.

---

## 10. Scientific Status

- Research Question / Framing: **EXPERIMENTAL CANDIDATE** (Motivating hypothesis).
- Critical Instruction: Do NOT present "perception as investigation" as an established scientific conclusion. It is the active research hypothesis under validation in Generation-1.

---

## 11. Empirical Evidence and Baselines

- Supported by the 16-seed Model D crossover (+9.79 pp over TRADES baseline on PGD-100 at $\epsilon = 0.094$).
- Supported by the AIS-v2 smoke-gate correlation ($r \approx 0.706$).

---

## 12. Limitations and Scope Boundaries

- The current observation space is restricted to 2D image crops. 3D spatial rotation, depth foraging, and temporal video streaming are **DEFERRED**.
- Active foraging adds inference latency proportional to the number of glimpses $T$ ($T = 4$ is the fixed budget).

---

## 13. Related Components and System Cross-References

- Complete loop specification: [Chapter 12 — The Complete Perceptual Loop](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/12_Complete_Perceptual_Loop.md)
- Active Information Sampling: [Chapter 11 — AIS-v2 Gaze Policy](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md)
- Figure Reference: [perceptual_loop.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/perceptual_loop.svg).
