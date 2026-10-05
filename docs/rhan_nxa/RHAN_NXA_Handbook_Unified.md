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


---

# Chapter 01 — Why RHAN-NXA Exists: Lessons from Generation-0

## 1. In one sentence
RHAN-NXA exists because empirical evidence from Generation-0 demonstrated that naively stacking cognitively inspired modules onto feedforward vision models fails to produce a genuine robustness frontier without an explicit, unified perceptual belief organization.

## 2. Intuition
When artificial intelligence researchers attempt to make vision systems more robust or "brain-like," the standard approach is module accretion: take a standard convolutional or transformer network, bolt on a slot-attention module to detect objects, add a variational autoencoder to reconstruct pixels, insert an entropy penalty to encourage early stopping, and attach a secondary loss to align features.

Generation-0 systematically evaluated this approach. The outcome was a hard scientific lesson: simply stacking structured mechanisms does not automatically create robustness. Instead, stacked modules often create hidden gradient competitions, induce artificial gradient masking, or suffer from catastrophic representation collapse. RHAN-NXA Generation-1 represents a fundamental methodological shift: rather than continuing to accumulate uncoordinated modules, it reorganizes the computational process of vision around an explicit, recurrent perceptual belief state.

## 3. Concrete example
In Generation-0, we introduced a generative reconstruction prior (the "E1" experiment) designed to regularize features by forcing the model to reconstruct pixel patches. 
* On clean images, the reconstruction prior was beneficial: clean accuracy increased by $+3.10\text{ pp}$.
* Under adversarial attack, however, the mechanism backfired: PGD-100 robust accuracy decreased by $-0.90\text{ pp}$.
* The Lens audit revealed why: the pixel reconstruction loss competed with the classification objective for gradient capacity, systematically diluting sensory precision $\Pi_D$ across every time step and increasing internal representation drift under attack. The low-level pixel objective actively distracted the network from semantic categorization margins.

## 4. Why RHAN-NXA needs it
RHAN-NXA avoids the module stacking trap by enforcing three structural disciplines:
1. **Representational Parsimony:** Only mechanisms that survive rigorous ablation are retained in the core build. Mechanisms with negative evidence (such as 16-slot SBR) or ambiguous attribution are cleanly excised ($S_t = \text{None}$).
2. **Unified Predictive Function:** A single glimpse predictor serves both belief-updating and active gaze selection, preventing disparate heads from drifting out of alignment.
3. **Explicit Isolation:** Every architectural addition is evaluated in a strict, pre-registered Directed Acyclic Graph (DAG) against a frozen baseline.

## 5. Mathematical formulation
In Generation-0, the total loss function was often an ad-hoc linear combination of competing objectives:
$$\mathcal{L}_{\text{Gen0}} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{recon}} + \lambda_2 \mathcal{L}_{\text{halt}} + \lambda_3 \mathcal{L}_{\text{forage\_consist}} + \lambda_4 \mathcal{L}_{\text{prec\_cal}}$$

In RHAN-NXA Generation-1, the core architecture is governed by a streamlined task loss with evidential uncertainty, while auxiliary objectives are either eliminated or strictly staged:
$$\mathcal{L}_{\text{Gen1}} = \mathcal{L}_{\text{EDL}}(p_\alpha(y), y_{\text{true}}) + \beta \mathcal{L}_{\text{pred}}(E_t)$$
where $\mathcal{L}_{\text{EDL}}$ is the Dirichlet Evidential Deep Learning loss (cross-entropy with an expected categorical variance regularizer) and $\mathcal{L}_{\text{pred}}$ is the latent prediction error loss in encoder feature space.

## 6. Implementation
The transition from Generation-0 to Generation-1 is manifested in `noesis_vision`:
* Elimination of the legacy `_nx_trainer` script, which had hardcoded SBR across mechanism swap tests.
* Implementation of `noesis_vision.core.schema.REJECTED_OUTRIGHT`, which strictly prevents deprecated mechanisms (pixel reconstruction, 16-slot SBR, external RAG) from being toggled on.
* Enforcement of `noesis_vision.core.schema.GATED_FLAGS`, which raises runtime errors if experimental flags (such as $S_t$ or $L_{\text{stab}}$ objective promotion) are set without pre-registered validation strings.

## 7. Tensor representation
* Superseded Gen-0 reconstruction target: $x_{\text{recon}} \in \mathbb{R}^{B \times 3 \times 48 \times 48}$ (Pixel space, REJECTED).
* Generation-1 prediction error target: $F_{\text{obs}} \in \mathbb{R}^{B \times 16 \times 384}$ (Latent encoder token space, LOCKED).

## 8. Gradient behavior
In Generation-0, multi-loss competition led to gradient starvation. In Generation-1, gradient ownership is decoupled into 8 distinct optimizer parameter groups registered in `noesis_vision.core.multi_group_optimizer.py`. Every component's gradient magnitude is audited pre-flight via $|dW|$ assertions before any training run begins.

## 9. Scientific status
* Paradigm Shift: **LOCKED**.
* Framing: Generation-0 established critical baseline evidence and unmasked major confounders; Generation-1 reorganizes those discoveries into an explicit, auditable architecture.

## 10. Evidence
Documented in `FINDINGS.md` (Findings 10, 12, 16, 17) and `MASTER_PLAN.md` (Part 0). 

## 11. Limitations
The shift to Generation-1 does not guarantee that recurrence or active inference will fully bridge the human-AI robustness gap; it provides the rigorous, unconfounded substrate required to test that hypothesis cleanly.

## 12. Related components
Connected to Chapter 00 (Overview), Chapter 16 (Gen-0 Evidence), Chapter 17 (Experimental DAG), and Chapter 21 (Decision Records).


---

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


---

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


---

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


---

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


---

# Chapter 06 — Structure State S_t: The None-Contract and Slot Dynamics

## 1. In one sentence
$S_t$ is the slot-based structured/relational component of the belief state, currently **locked to None** in the Generation-1 core build, with re-entry deferred to post-Step-6 validation at 2–4 slots only.

## 2. Intuition
While $z_t$ is a dense global vector that summarizes the entire scene as one continuous representation, $S_t$ was designed to hold *structured*, object-level or relational representations—a small set of "slots," each encoding one meaningful scene element (e.g., "object at left carrying a pattern," "background region").

The idea draws on the Slot Attention literature: rather than pooling all visual information into one vector, the system maintains a small set of discrete slot vectors that compete for explaining different parts of the scene. In principle, this enables more compositional reasoning and more targeted attention.

## 3. Why it is locked to None in Gen-1

Generation-0 implemented a 16-slot `SlotAttention` bottleneck. **This specific implementation is rejected outright** for two reasons documented in `schema.py`:

1. The 16-slot architecture introduced severe gradient conflicts with the recurrent update path.
2. The SBR confound (`_nx_trainer` bug) means the isolated contribution of the slot mechanism can never be cleanly attributed from Gen-0 data.

The research plan mandates that the core Generation-1 loop be validated at $S_t = \text{None}$ **first**—establishing a clean, unconfounded baseline. Only after Step-6 produces a validated ImageNet-100 result will $S_t$ re-enter the system, at 2–4 slots and with an explicit structural isolation experiment.

## 4. Implementation status

| Status | Meaning |
|:---|:---|
| **REJECTED** | 16-slot Gen-0 implementation. Never re-introduced as a fallback without an explicit plan ruling. |
| **LOCKED (None)** | Current Gen-1 core build default. `enable_sbr = False` in `RHANNXAConfig`. |
| **DEFERRED** | 2–4 slot concept pending Step-6 validation. Gated by `step6_validated_result`. |

The gate is enforced in `RHANNXAConfig.__post_init__`:

```python
if getattr(self, "enable_sbr") and not getattr(self, "step6_validated_result"):
    raise ValueError(
        "enable_sbr=True requires step6_validated_result ..."
    )
```

Setting `enable_sbr = True` without a recorded `step6_validated_result` is a hard runtime error—not a warning, not a soft override.

## 5. Mathematical formulation (for documentation completeness)
When and if $S_t$ re-enters the system, the belief state becomes:

$$B_t = (z_t, S_t, U_t, E_t, A_t), \quad S_t \in \mathbb{R}^{B \times K \times D_s}$$

where $K \in \{2, 3, 4\}$ is the number of slots and $D_s$ is the slot dimension. The `as_tensor()` readout, which currently returns $z_t$ directly, would then require an explicit fusion mechanism—a `NotImplementedError` is deliberately raised in `VectorBeliefState.as_tensor()` for this path to prevent silent improvisation.

## 6. None-safety invariant (mandatory)
**Every consumer of a `BeliefState` must have an explicit branch for `S_t is None`.** The interfaces.py header states: *"this is not a hypothetical: it is the actual default configuration."*

This means:
- Do not write code that silently assumes `S_t` is a tensor.
- Do not write code that silently assumes `S_t` is None (which would break the future re-entry).
- Always test both cases independently.

## 7. Scientific status
- **16-slot Slot Attention (Gen-0)**: **REJECTED OUTRIGHT**.
- **$S_t = \text{None}$ in Gen-1 core**: **LOCKED**.
- **2–4 slot concept**: **EXPERIMENTAL CANDIDATE**, **DEFERRED** pending Step-6 result.

## 8. Source references
- [`schema.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/schema.py): `enable_sbr`, `sbr_num_slots`, `REJECTED_OUTRIGHT`, `GATED_FLAGS`.
- [`interfaces.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/interfaces.py): `BeliefState.s` property; None-safety documentation.
- [`vector_belief.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/vector_belief.py): `VectorBeliefState.as_tensor()` with explicit `S_t is None` branch.

## 9. Related components
- Chapter 04 (Belief State): The container that holds $S_t$.
- Chapter 21 (Decision Records): DR-001 through DR-004 document the original slot rejection rationale.
- Chapter 17 (Experimental DAG): Step-6 is the gating condition for $S_t$ re-entry.


---

# Chapter 07 — Dirichlet Uncertainty State U_t: Evidential Epistemic Doubt

> *Level 2 reading. Formally parameterizing second-order epistemic uncertainty via Dirichlet evidential deep learning.*

---

## 1. In One Sentence

$U_t$ is the system's calibrated evidential uncertainty over class hypotheses, represented as a Dirichlet distribution parameterized by non-negative evidence $e_t \in \mathbb{R}_{\ge 0}^{B \times C}$, which simultaneously gates belief updates (via precision $\Pi_t$), drives gaze selection (via entropy reduction), and tracks belief stability (via $\mathcal{L}_\text{stab}$).

---

## 2. Intuition

A softmax output gives a probability distribution over classes. But a high-probability softmax output and a low-probability softmax output both look like "confident" probabilities—the softmax cannot express *second-order uncertainty* (uncertainty about its own predictions). 

Dirichlet evidential learning provides exactly this. The Dirichlet concentration parameters $\alpha_t = e_t + 1$ represent accumulated evidence per class. When evidence is low for all classes ($e_t \approx \mathbf{0}$), the Dirichlet is nearly flat—the model has little to go on. When evidence is high for one class, the distribution is sharply peaked. This second-order uncertainty is the signal that drives active perception: the system should keep looking when $U_t$ remains diffuse.

---

## 3. Evidential Uncertainty Architecture

![Figure 7. Dirichlet Evidential Epistemic Uncertainty U_t in RHAN-NXA. The EvidentialHead outputs non-negative evidence e_k via a softplus layer, parameterizing a Dirichlet distribution Dir(α). Epistemic uncertainty U_t = K / S is bounded in (0, 1], decoupling aleatoric conflict from epistemic ignorance.](figures/uncertainty_dirichlet.svg)

---

## 4. Mathematical Formulation

**Dirichlet parameterization**:

$$e_t \in \mathbb{R}_{\ge 0}^{B \times C}, \quad \alpha_t = e_t + 1, \quad S_\alpha = \sum_{c=1}^C \alpha_{t,c}$$

**Uncertainty scalar** (for precision computation):

$$U_t = \frac{C}{S_\alpha} = \frac{C}{\sum_{c=1}^C \alpha_{t,c}} \in (0, 1]$$

**Differential entropy** (for AIS-v2 candidate scoring):

$$H(U_t) = \log B(\boldsymbol\alpha) + (S_\alpha - C)\psi(S_\alpha) - \sum_{c=1}^C (\alpha_{t,c} - 1)\psi(\alpha_{t,c})$$

where $\psi$ is the digamma function and $B(\boldsymbol\alpha)$ is the multivariate Beta function.

**Evidence production**:

$$e_t = \text{clamp}(\text{softplus}(\text{Linear}_2(\text{GELU}(\text{Linear}_1(z_t)))), e_\text{min}, e_\text{max})$$

with $e_\text{min} = 10^{-6}$ and $e_\text{max} = 10^4$.

---

## 5. Three Consumers of U_t

All three consumers read from the **same** `EvidentialHead` output—no parallel uncertainty mechanisms:

| Consumer | What it reads | Architectural Function |
|:---|:---|:---|
| **Precision $\Pi_t$** | $U_t$ scalar | Gates belief update magnitude: $\Pi_t = \max(1 - U_t, 10^{-4})$ |
| **AIS-v2 scoring** | $H(U_t)$ entropy | Measures expected entropy reduction $\Delta H$ per candidate gaze |
| **$\mathcal{L}_\text{stab}$ diagnostic** | Trajectory drift | Evaluates belief stability and Gate 9 responsiveness |

---

## 6. Implementation Mapping

- **Canonical class**: `noesis_vision.uncertainty.evidential_head.DirichletParams`
  - Fields: `evidence: torch.Tensor` ($(B, C)$, non-negative)
  - Properties: `alpha`, `uncertainty`, `entropy()`
- **Head module**: `noesis_vision.uncertainty.evidential_head.EvidentialHead`
  - Architecture: `Linear(d_z → 256) → GELU → Linear(256 → C) → softplus → clamp`
  - Accepts both `(B, D)` and `(B, N, D)` features (pools by mean for token inputs)
  - Returns `DirichletParams` whose evidence tensor carries the full autograd graph

---

## 7. Numerical Stability Contracts

The evidence clamp is contractually required and tested, not optional:

- **Low end** ($e_\text{min} = 10^{-6}$): Ensures $\alpha > 1$ strictly per class; prevents log/entropy consumers from encountering the $\alpha = 1$ edge case.
- **High end** ($e_\text{max} = 10^4$): Prevents float32 overflow in digamma downstream.
- **Entry clamp on features**: Input features are clamped to $[-10^4, +10^4]$ at head entry, so extreme features do not produce NaN inside the MLP.
- **NaN propagation is intentional**: The clamp does not sanitize NaN inputs—garbage-in is not silently repaired.

---

## 8. Lineage Resolution: EvidentialHead Status

In preliminary working notes, informal text occasionally labelled the `EvidentialHead` as "PORT VERBATIM". However, an audit confirmed no such class existed in Gen-0. The authoritative Part 5 Port Table officially locks its disposition as **NEW**:
- First implementation in this repository of Sensoy et al. (NeurIPS 2018).
- Implemented from scratch by Agent D in [noesis_vision/uncertainty/evidential_head.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/uncertainty/evidential_head.py).
- Supported by full unit test suites for non-negative softplus and entropy derivatives.

---

## 9. Scientific Status

- **Dirichlet evidential formulation**: **LOCKED** (one uncertainty representation, Part 1.F).
- **Class-readout vs. representation-level uncertainty**: **PENDING DECISION** (class-readout kept for Gen-1; representation-level deferred).
- **Implementation Status**: **NEW** (Authoritative, resolved in Appendix C).

---

## 10. Related Components and System Cross-References

- Evidential implementation: [noesis_vision/uncertainty/evidential_head.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/uncertainty/evidential_head.py)
- Precision modulation: [noesis_vision/predictive_coding/precision.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/precision.py)
- Active Information Sampling: [Chapter 11 — AIS-v2 Gaze Policy](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md)
- Figure Reference: [uncertainty_dirichlet.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/uncertainty_dirichlet.svg).


---

# Chapter 08 — Prediction Error E_t, Sensory Precision Π_t, and UpdateNet

> *Level 2–3 reading. Closed-loop hypothesis testing: latent token discrepancies, dynamic precision weighting, and bounded state updates.*

---

## 1. In One Sentence

$E_t$ is the discrepancy in latent token-feature space between what the system predicted it would see at its chosen fixation and what it actually observed—the raw material for belief updates; $\Pi_t$ is the learned precision scalar that weights how strongly $E_t$ shifts the belief through the bounded `UpdateNet` module.

---

## 2. Intuition

- **Prediction Error ($E_t$)**: Before each saccade, the system uses its current belief state to predict what visual features it expects to find at its next fixation location. When the fovea actually samples that location, the features it observes are compared to the prediction. The difference—large for surprising content, small for expected content—is $E_t$. This is the signal that triggers belief revision: *"What I saw was not what I expected, therefore my hypothesis must be updated."*

- **Sensory Precision ($\Pi_t$)**: Not all prediction errors are equally informative. If the system is already very uncertain (diffuse $U_t$), it may be appropriate to update more conservatively to avoid incorporating adversarial noise. $\Pi_t$ modulates this tradeoff: observations made under high certainty receive high precision, while high uncertainty dampens the update.

---

## 3. Prediction Error and UpdateNet Architecture

![Figure 6. Latent Prediction Error E_t, Precision Weighting Π_t, and Bounded UpdateNet. Predicted patch features are compared against observed patch features in CompactViT embedding space. The pooled error E_t is precision-weighted by Π_t and integrated via UpdateNet with DELTA_BOUND = 0.1 tanh clamping.](figures/prediction_error.svg)

---

## 4. Mathematical Formulation

### Prediction Error

$$E_t = \|g_t^{\text{obs}} - \hat{g}_t^{\text{pred}}\|_2, \quad t \ge 1$$

$$E_0 := \mathbf{0} \quad \text{(LOCKED — boundary condition at } t = 0\text{)}$$

where:
- $g_t^{\text{obs}} = \text{Encode}(\text{Foveate}(x, a_t))$ — **detached** observed token features (training target).
- $\hat{g}_t^{\text{pred}} = \text{Predictor}(z_{t-1}, a_t)$ — **non-detached** predicted features (must carry gradient).

### Sensory Precision

$$\Pi_t = \max(1 - U_{t-1}, 10^{-4})$$

with $\Pi_{\text{floor}} = 10^{-4}$ strictly enforced.

### Belief Update (UpdateNet)

$$\Delta z_t = 0.1 \cdot \tanh(\text{MLP}([z_{t-1}, \Pi_t \odot E_t^{\text{pooled}}]))$$

$$z_t = z_{t-1} + \Delta z_t$$

with hard architectural bound $\|\Delta z_t\|_\infty \le 0.1$.

---

## 5. The $E_0 := 0$ Boundary Condition (LOCKED)

At $t = 0$, the system has made exactly one observation (the central glimpse) and has no prior prediction to compare it against. The error is structurally defined as zero:

- Enforced by `VectorBeliefState.__init__` (raises `ValueError` if $E \ne \mathbf{0}$ at $t = 0$).
- Means the initial state is established without predictor discrepancy feedback.
- Tested by `tests/test_t0_produces_exact_zero_error`.

---

## 6. The Gradient Non-Detach Rule (Critical)

$E_t$ **must never be detached** before use in the belief update. This was the **single most repeated failure mode** in Generation-0's history: detaching the predicted features broke the gradient path into the predictor, causing it to receive zero training signal from downstream losses.

The rule is enforced across all core modules:
- `interfaces.py`: *"E_t is computed FROM a detached observed target and a non-detached predicted value; E_t itself must NEVER be detached before use in the update."*
- `glimpse_predictor.py`: *"NOT detached: the observed target is detached at call site; E_t is never detached before update."*
- `update_net.py`: *"Predicted-error path must carry gradient."*

---

## 7. Implementation Substrate

- **Predictor**: `noesis_vision.predictive_coding.glimpse_predictor.ConcreteGlimpseFeaturePredictor`
  - Input: $[z_t, a_{t+1}]$
  - Output: $(B, 16, 384)$ predicted token features
  - Parameter footprint: $\sim 2.8\text{M}$ parameters
- **UpdateNet**: `noesis_vision.predictive_coding.update_net.ConcreteUpdateNet`
  - Architecture: $\text{Linear}(768 \to 768) \to \text{GELU} \to \text{Linear}(768 \to 384) \to 0.1 \cdot \tanh(\cdot)$
  - Parameter footprint: $\sim 0.3\text{M}$ parameters
  - Enforces `DELTA_BOUND = 0.1`

---

## 8. Authoritative Tensor Specification

| Tensor | Shape | Gradient? | Description |
|:---|:---|:---:|:---|
| $\hat{g}_t^{\text{pred}}$ | `(B, 16, 384)` | **Yes** | Predicted patch tokens from predictor |
| $g_t^{\text{obs}}$ | `(B, 16, 384)` | **No** | Observed patch tokens (explicitly detached) |
| $E_t$ | `(B, 16, 384)` | **Yes** | Realized prediction discrepancy tensor |
| $E_t^{\text{pooled}}$ | `(B, 384)` | **Yes** | Mean-pooled token discrepancy for UpdateNet |
| $\Pi_t$ | `(B, 1)` | **Yes** | Precision scaling scalar $\ge 10^{-4}$ |
| $\Delta z_t$ | `(B, 384)` | **Yes** | Bounded state delta, $\Vert \Delta z_t \Vert_\infty \le 0.1$ |

---

## 9. Rejected Designs (LOCKED OUT)

The following targets for $E_t$ are **REJECTED OUTRIGHT** in `schema.py` (DR-002):

- **Pixel-space reconstruction**: Penalizes high-frequency details, dilutes precision, increases drift, and requires a heavy decoder.
- **Edge-map / hand-designed features**: Removes learned prediction from gradient paths; foreign representation space.
- **Raw addition $z_t + \lambda \Pi E_t$**: Dimension and manifold mismatch; UpdateNet non-linear mapping is mandatory.

---

## 10. Scientific Status

- **Latent token prediction as $E_t$ target**: **EXPERIMENTAL CANDIDATE** (Design LOCKED, empirical value under unconfounded evaluation).
- **UpdateNet architecture**: **REQUIRED**.
- **$E_0 := 0$ boundary condition**: **LOCKED**.
- **Non-detach rule**: **LOCKED**.

---

## 11. Related Components and System Cross-References

- Predictor implementation: [noesis_vision/predictive_coding/glimpse_predictor.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/glimpse_predictor.py)
- UpdateNet implementation: [noesis_vision/predictive_coding/update_net.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/update_net.py)
- Figure Reference: [prediction_error.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/prediction_error.svg).


---

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


---

# Chapter 10 — Differentiable Foveation: Spatial Bounding and Affine Sampling

> *Level 2 reading. Spatial Transformer Network mechanics: extracting high-acuity 56×56 crops while preserving end-to-end motor gradients.*

---

## 1. In One Sentence

Differentiable foveation is the mechanism by which RHAN-NXA extracts a $56 \times 56$ high-acuity crop from any location in the full input image using a Spatial Transformer Network (STN) affine grid, such that gradients from downstream objectives flow back through the fixation coordinates, enabling the gaze policy to be trained end-to-end.

---

## 2. Intuition

A biological fovea physically saccades to different locations in the visual field; RHAN-NXA simulates this computationally. Given a 2D gaze coordinate $a_t \in [-1, +1]^2$ (normalized, origin at image center), the foveation module constructs an affine transformation matrix $\Theta_t$ that, when applied via `torch.nn.functional.grid_sample`, produces a $56 \times 56$ bilinear-interpolated crop centered at $a_t$.

The critical property is **differentiability**: because $\Theta_t$ is constructed from the gaze coordinates via tensor operations (not Python floats), the autograd graph preserves the connection between the output crop and the input gaze coordinates. The motor Jacobian $\frac{\partial g_t}{\partial a_t}$ flows back through the crop into the gaze policy parameters.

---

## 3. Differentiable Foveation Architecture

![Figure 14. Differentiable Spatial Transformer Foveation. Given continuous fixation coordinates a_t in [-1, 1]^2, the module constructs an affine transformation matrix with fixed scale s = 56/224 = 0.25, generating a sampling grid via affine_grid and extracting a high-acuity 56×56 patch via bilinear interpolation. Gradients flow back through the motor Jacobian to train the gaze policy.](figures/foveation.svg)

---

## 4. Mathematical Formulation

The affine transformation for gaze $(x_g, y_g) \in [-1, +1]^2$ with scale $s = \text{fovea\_size} / H$:

$$\Theta_t = \begin{pmatrix} s & 0 & x_g \\ 0 & s & y_g \end{pmatrix} \in \mathbb{R}^{2 \times 3}$$

The output crop at pixel $(i, j)$ samples from the source image at the bilinear-interpolated location:

$$\text{crop}_{i,j} = \text{BilinearInterp}\left(x,\ s \cdot i_\text{norm} + x_g,\ s \cdot j_\text{norm} + y_g\right)$$

where $i_\text{norm}, j_\text{norm}$ are the normalized output pixel coordinates. Border padding is used for boundary coordinates.

---

## 5. Implementation Mapping

- **Function**: `noesis_vision.models.foveation.foveal_sample(x_image, gaze_coords, fovea_size=56)`
- **Framework**: PyTorch's `torch.nn.functional.affine_grid` + `torch.nn.functional.grid_sample`.
- **Sampling mode**: bilinear, border padding, `align_corners=False`.

### Differentiable Theta Construction (Source Extract)

```python
scale_col = torch.full((B, 1), scale, dtype=gaze_coords.dtype, device=gaze_coords.device)
zero_col = torch.zeros((B, 1), dtype=gaze_coords.dtype, device=gaze_coords.device)
row0 = torch.cat([scale_col, zero_col, gaze_coords[:, 0:1]], dim=1)
row1 = torch.cat([zero_col, scale_col, gaze_coords[:, 1:2]], dim=1)
theta = torch.stack([row0, row1], dim=1)  # (B, 2, 3)
```

The autograd graph is strictly preserved through `torch.cat` and `torch.stack`.

---

## 6. Key Adaptations from Generation-0

| Property | Gen-0 (STL-10) | RHAN-NXA Gen-1 | Scientific Rationale |
|:---|:---|:---|:---|
| Input resolution | Hardcoded 96 px | Parameterized $H = 224$ px | Scales to ImageNet-100 standard |
| Fovea size | 48 px | **56 px** | Clean patch grid ($56 = 4 \times 14$) |
| Patch divisibility | $48 / 14 = 3.43$ (fractional) | $56 / 14 = \mathbf{4.0}$ **(integer)** | Required by ViT patch embedder |
| Sampling conventions | bilinear, border, `align_corners=False` | **Unchanged** | Validated STN conventions |

---

## 7. Authoritative Tensor Specification

| Tensor | Shape | Gradient? | Description |
|:---|:---|:---:|:---|
| `x_image` | `(B, 3, 224, 224)` | Optional | Full visual scene batch |
| `gaze_coords` | `(B, 2)` | **Yes** | Continuous fixation coordinates in $[-1, 1]^2$ |
| `theta` | `(B, 2, 3)` | **Yes** | Affine transformation matrix parameter tensor |
| Output crop $g_t$ | `(B, 3, 56, 56)` | **Yes** | Extracted foveal glimpse patch |

---

## 8. Validation Rules and Boundary Asserts

`foveal_sample` enforces:

- Non-square input (`H ≠ W`) raises `ValueError`: foveation assumes a square frame.
- Gaze coordinates outside $[-1, +1]$ (tolerance $10^{-6}$) raise `ValueError`: *"a convention mismatch here corrupts every downstream shape."*
- `fovea_size > H` raises `ValueError`: crop cannot exceed image frame.

---

## 9. Scientific Status

- **Differentiable foveation mechanism**: **LOCKED** (ADAPT disposition from validated Gen-0 source).
- **Fovea size $56 \times 56$**: **LOCKED** (required for DINOv2-small patch divisibility).
- **Gaze coordinate convention $[-1, +1]$**: **LOCKED**.

---

## 10. Related Components and System Cross-References

- Foveation implementation: [noesis_vision/models/foveation.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/foveation.py)
- Substrate caller: [noesis_vision/models/backbone.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/backbone.py)
- Figure Reference: [foveation.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/foveation.svg).


---

# Chapter 11 — Active Information Sampling (AIS-v2) Gaze Policy and Gaze State A_t

> *Level 2–3 reading. Active visual inquiry: discrete candidate generation, analytical entropy reduction scoring, and the center-bias failure gate.*

---

## 1. In One Sentence

AIS-v2 (Active Information Seeking, version 2) is the trained gaze policy that selects each subsequent fixation location by scoring $K$ candidate locations against the expected reduction in Dirichlet uncertainty, using the shared predictor and evidential head as its exclusive scoring mechanism—no auxiliary scoring head is permitted.

---

## 2. Intuition

After each glimpse, the system must decide where to look next. The AIS-v2 policy answers this question by asking: *"For each of $K$ candidate locations, how much would my uncertainty decrease if I looked there?"* The candidate with the highest expected entropy reduction is selected.

This is implemented without introducing duplicate uncertainty or prediction modules: the **same** `ConcreteGlimpseFeaturePredictor` predicts what tokens would be observed at each candidate, and the **same** `EvidentialHead` maps those predicted tokens to a predicted Dirichlet distribution. The entropy of the predicted Dirichlet is the expected uncertainty after looking there; the score is the difference from the current entropy.

---

## 3. AIS-v2 Active Gaze Architecture

![Figure 5. AIS-v2 Active Information Sampling: Perception as Active Investigation. Candidates are sampled across the visual field avoiding past fixation history A_t. The shared predictor projects anticipated patch tokens for each candidate, and Dirichlet entropy reduction ΔH is evaluated. Candidate 3 (diagnostic boundary saccade) achieves the highest information gain and is selected.](figures/ais_v2.svg)

---

## 4. Mathematical Formulation

### Candidate Scoring ($t \ge 1$)

$$r_k = H(U_t) - H\!\left(\text{Dirichlet}\!\left(\hat{e}(B_t, c_k)\right)\right)$$

where $c_k$ is candidate $k$'s location, $\hat{e}(B_t, c_k)$ is the predicted Dirichlet evidence at $c_k$ via the shared predictor, and $H(\cdot)$ is the Dirichlet differential entropy (digamma formulation, `DirichletParams.entropy()`).

### Candidate Scoring ($t = 0$, Fallback)

$$r_k^{(0)} = \text{SaliencyMap}(|\text{observed tokens}|^2)_{@c_k}$$

No predictor call or Dirichlet construction exists in this initial branch. The `reduction` field of `CandidateScores` is `None`—structurally absent, not zero.

### Action Selection

- **Soft Selection (Training)**:
  $$w_k = \text{Gumbel-Softmax}\!\left(r_k \cdot \lambda, \tau\right), \quad a_{t+1} = \sum_{k=1}^K w_k \cdot c_k$$
- **Hard Selection (Evaluation)**:
  $$a_{t+1} = c_{\arg\max_k r_k}$$

---

## 5. The Parameter Reuse Boundary (Inviolable)

`AISv2GazePolicy` has one learned parameter: `logit_scale`, a scalar temperature. No per-candidate scoring head, no second predictor, and no second uncertainty module are permitted. This is enforced by:
- `__init__` type-checking: `predictor` must be `ConcreteGlimpseFeaturePredictor`; `evidential_head` must be `EvidentialHead`.
- Test: `tests/test_ais_v2_gaze_policy.py::test_shares_agent_e_predictor`.

---

## 6. Candidate Generation Protocol

`HeuristicCandidateSampler` generates $K \in \{4, \dots, 8\}$ candidates per glimpse:

| Property | Gen-0 (STL-10) | RHAN-NXA Gen-1 | Scientific Rationale |
|:---|:---|:---|:---|
| Anchor | argmax of pixel reconstruction error | argmax of **per-token prediction error** $\|g_t^{\text{obs}} - \hat{g}_t\|^2$ | Gen-1 operates in latent token space |
| Candidate 0 | Anchor (stay put) | **Unchanged** | Allows temporal persistence |
| Candidates $1 \dots K-1$ | Anchor + Gaussian noise ($\sigma = 0.12$) | **Unchanged** | Local exploration around surprise |
| Coordinate Clamp | $[-0.9, +0.9]$ | **Unchanged** | Prevents boundary clipping |
| Candidate Count $K$ | 4 to 8 (default 4) | **Unchanged** | Bound on evaluation compute |

---

## 7. Gaze State $A_t$ (Coordinate Record)

`noesis_vision.gaze.gaze_state.GazeState` is the canonical $A_t$ carrier:

- **Detached by construction**: Every `record()` call stores `location.detach().clone()`. The autograd graph cannot leak into the history record.
- **Capacity enforcement**: History raises loudly when `len(history) >= T`. Silent truncation was a known failure mode in Gen-0.
- **Guard logic**: `current` property raises `IndexError` at $t = 0$. The initial fixation is an explicit system input; no default is fabricated.

---

## 8. The Center-Bias Failure Condition

A gaze policy that consistently returns coordinates near the image center is a **FAILED** result, not a passing result with an asterisk. This is explicitly enforced by pre-registered verification gates: if AIS-v2 produces center-biased fixations across test sets, the run is rejected as dysfunctional.

---

## 9. Scientific Status and Epistemic Distinctions

- **AIS-v2 mechanism validity**: **REQUIRED** (Targeted empirical evidence: $r \approx 0.706$, $n = 512$).
- **Isolated system-level robustness contribution**: **UNKNOWN** (Gen-0 runs D2 and D3 were confounded by accidental `--enable-sbr` flag; currently under unconfounded DAG re-evaluation).
- **Center-bias result**: **TERMINAL FAILURE CONDITION**.
- **Candidate sampler heuristic**: **LOCKED**.

---

## 10. Related Components and System Cross-References

- Gaze policy implementation: [noesis_vision/gaze/ais_v2_policy.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/gaze/ais_v2_policy.py)
- Candidate generator: [noesis_vision/gaze/candidate_sampler.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/gaze/candidate_sampler.py)
- Gaze state container: [noesis_vision/gaze/gaze_state.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/gaze/gaze_state.py)
- Figure Reference: [ais_v2.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/ais_v2.svg).


---

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


---

# Chapter 13 — System Architecture: CompactViT and Parameter Budget

## 1. In one sentence
RHAN-NXA's visual substrate is `CompactViT`: a DINOv2-small-shaped trunk ($d_z = 384$, 12 blocks, patch size 14) plus a Universal-Transformer-style tied refinement block, totaling $\sim 23.3\text{M}$ parameters, within the locked 20–25M band, with a clean DINOv2 warm-start mapping.

## 2. Architecture overview

```
Input: (B, 3, H, H) full image
          │
          ▼
  foveal_sample(gaze_coords)    ← Differentiable STN crop (56×56)
          │
          ▼
  PatchEmbed Conv2d(3, 384, 14, 14)   → (B, N, 384)   [N = 16 patches]
          │
  [CLS token] [Register token] [Pos embed]
          │
  ┌─────────────────────────┐
  │   12 × TransformerBlock │  ← DINOv2-small-shaped trunk
  │   (width=384, heads=6)  │  ← Warm-started from DINOv2-small
  └─────────────────────────┘
          │
  LayerNorm (final)
          │
  ┌─────────────────────────┐
  │  TiedRecurrence:        │  ← 1 TransformerBlock, run 2–3 times
  │  2–3 × same block       │  ← New-by-design (no warm-start)
  └─────────────────────────┘
          │
     Pool (CLS position)         → pooled_z: (B, 384)
     Take patch tokens           → tokens:   (B, 16, 384)
```

## 3. Parameter budget

| Component | Parameters | Notes |
|:---|:---:|:---|
| 12 trunk blocks (attn + MLP, width 384) | ~21.29M | DINOv2-small-shaped |
| Patch embed (`Conv2d`) | ~0.23M | 14×14 stride |
| Pos/CLS/register tokens | ~0.01M | Small |
| Final LayerNorm | ~0.001M | |
| Tied refinement block (1 block × mlp_ratio 4) | ~1.77M | New-by-design |
| **Total** | **~23.3M** | In [20M, 25M] band |

The parameter count **does not change with the within-glimpse iteration count** because the refinement block is tied (one set of parameters, applied 2–3 times).

Test: `tests/test_backbone_param_count` asserts `total_params ∈ [20_000_000, 25_000_000]`.

## 4. Warm-start mapping
`CompactViT.load_dino_warm_start(path)` maps DINOv2-small checkpoint keys 1:1 to trunk module keys:

| DINOv2 key pattern | CompactViT key |
|:---|:---|
| `patch_embed.proj.*` | `patch_embed.proj.*` |
| `blocks.{n}.norm1.*` | `blocks.{n}.norm1.*` |
| `blocks.{n}.attn.qkv.*` | `blocks.{n}.attn["qkv"].*` |
| `blocks.{n}.attn.proj.*` | `blocks.{n}.attn["proj"].*` |
| `blocks.{n}.norm2.*` | `blocks.{n}.norm2.*` |
| `blocks.{n}.mlp.fc1.*` | `blocks.{n}.mlp["fc1"].*` |
| `blocks.{n}.mlp.fc2.*` | `blocks.{n}.mlp["fc2"].*` |
| `norm.*` | `norm.*` |

**New-by-design keys** (excluded from match fraction): `refinement.*`, `pos_embed`, `cls_token`, `register_token`.

**Threshold**: matched trunk fraction $\ge 0.95$. If below, `accepted=False` is returned and **no weights are applied**. The caller decides (STOP condition).

## 5. The `encode_glimpse` interface (load-bearing)
`CompactViT.encode_glimpse(image, gaze_coords, fovea_size=56) → (pooled_z, tokens)`

- `pooled_z`: $(B, D_z)$ — the CLS-position output after trunk + refinement. This is $z_t$ in the belief state.
- `tokens`: $(B, N, D_\text{feat})$ — patch-token features at the fixation. This is the predictor's target/score space.

This signature is **the load-bearing contract** consumed by Agent E (prediction target) and Agent F (candidate scoring). It must not be changed without a plan ruling.

## 6. Pillar heads (not counted in the budget)
The 20–25M budget is for the trunk and refinement block **excluding pillar heads**:
- `EvidentialHead`: classification readout Dirichlet ($\sim 0.3\text{M}$)
- `ConcreteGlimpseFeaturePredictor`: predictor ($\sim 2.8\text{M}$)
- `ConcreteUpdateNet`: belief update ($\sim 1.8\text{M}$)
- `PrecisionFunction`: precision mapping ($\sim 0.001\text{M}$)

Total system parameters (trunk + pillar heads): $\sim 28\text{M}$.

## 7. Scientific status
- **DINOv2-small-shaped architecture**: **LOCKED** ($D_z = 384$; Part 1.A).
- **20–25M parameter budget**: **LOCKED** (Agent C contract).
- **Warm-start from DINOv2-small**: **LOCKED** (clean mapping required; below-threshold loads are refused).
- **Tied refinement block**: **LOCKED** (Hybrid Option C; Part 1.C).

## 8. Source references
- [`backbone.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/backbone.py): `CompactViT`, `D_Z`, `PATCH_SIZE`, `TRUNK_DEPTH`, `WARM_START_MATCH_FRACTION_MIN`, `PARAM_COUNT_BAND`
- [`recurrent_block.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/recurrent_block.py): `TransformerBlock`, `TiedRecurrence`
- [`foveation.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/models/foveation.py): `foveal_sample`, `DEFAULT_FOVEA_SIZE`
- Illustrated in Figure 1 (`system_overview.svg`).
- Connected to Chapters 05, 09, and 10.


---

# Chapter 14 — Multi-Group Gradient Flow: Optimization Boundaries

## 1. In one sentence
RHAN-NXA uses a `MultiGroupOptimizerRegistry` that assigns each independently gradient-isolated component to its own named parameter group with its own per-group learning rate and per-group gradient clipping, preventing any single component's gradient from dominating the shared optimizer budget.

## 2. The Stage 2 incident (motivation)
Before the multi-group optimizer was introduced, all parameters shared a single global `clip_grad_norm_` call. The backbone's TRADES adversarial gradient is typically orders of magnitude larger than the auxiliary head gradients (e.g., HPC, predictive coding). With a global clip, the backbone owned the norm budget and the auxiliary heads received per-step parameter updates of magnitude $\sim 10^{-5}$—the **ratio-1.00 freeze**: the heads learned at a rate indistinguishable from frozen.

The fix (2026-08-13, now a standing architectural rule): each component registers its own named parameter group with:
- Its own `lr_multiplier` (group lr = `base_lr × multiplier`)
- Its own `clip_norm` (group gradient budget)

This is **not discretionary**. Every new trainable component must register its own group.

## 3. Optimizer group map

| Group name | Component | Notes |
|:---|:---|:---|
| `backbone` | `CompactViT` trunk + refinement | Always registered **first** (group 0) |
| `recurrence_refinement` | Tied refinement block | Subset of backbone—may be split |
| `predictor` | `ConcreteGlimpseFeaturePredictor` | Own pre-flight $\lvert dW \rvert$ check |
| `update_net` | `ConcreteUpdateNet` | Own pre-flight $\lvert dW \rvert$ check |
| `precision` | `PrecisionFunction` | Small; own group |
| `evidential_head` | `EvidentialHead` | Own group |
| `gaze_policy` | `AISv2GazePolicy` (logit_scale) | Single scalar parameter |
| `l_stab_diagnostic` | Diagnostic-only (Phase A) | Not an objective; own group |

Source: `schema.py::OPTIMIZER_GROUP_NAMES`.

## 4. Implementation

**`OptimizerGroupRegistry`** (`noesis_vision/core/multi_group_optimizer.py`):
- **`register(name, params, lr_multiplier, clip_norm)`**: A parameter may only belong to one group—double registration raises loudly (two momentum buffers = double updates).
- **`register_backbone(params)`**: Must be called first; backbone is always group 0.
- **`build_optimizer(base_lr)`**: Returns `torch.optim.SGD` with one param group per registered name.
- **`clip_grad_per_group()`**: Applies `clip_grad_norm_` to **each group's parameters separately**, using that group's own `clip_norm`. Never a global budget.

Source: **PORTED VERBATIM** from `rhan_core/optim/multi_group_optimizer.py` (Gen-0, validated across 3+ components).

Two Gen-0-only pieces were **not** ported (documented, not silently dropped):
- HPC/SBR/AIS LR-multiplier constants (Gen-0 component names)
- `default_group_spec(model, ...)` (Gen-0 model structure; Gen-1 group derivation belongs to Agent C/A)

## 5. The resume guard

`registry.resume_guard(optimizer_state_dict, saved_scheduler)` returns `True` only when:
1. Group **count** matches.
2. Group **names** match (catches parameter reordering with the same count).
3. LR-**ratio pattern** matches (uses scheduler `base_lrs` if available; rejects stale decayed lrs).

A `False` result means the caller must construct a fresh optimizer and warn loudly. Silent restoration of a mismatched checkpoint is the failure class that produced the Gen-0 stale-resume bug.

## 6. Pre-flight $|dW|$ check
Before any gradient-bearing smoke test, the pre-flight check measures per-group parameter step magnitude $|dW|$ after one synthetic forward+backward. Specifically:

- Capture $\theta$ before step.
- Execute one forward+backward.
- Step the optimizer.
- Measure $|\theta_\text{after} - \theta_\text{before}|$ per group.

A group with $|dW| \approx 0$ (relative to the backbone group) is in the Stage 2 freeze condition. **Test**: `tests/test_preflight_dw.py` asserts that the UpdateNet and predictor groups both receive non-trivial updates relative to the backbone.

## 7. Scientific status
- **Multi-group optimizer with per-group clipping**: **LOCKED** (Stage 2 fix; now a standing architectural rule).
- **Per-group LR multipliers**: **EXPERIMENTAL CANDIDATE** (specific multiplier values chosen per run, not locked).
- **Port status**: **PORTED VERBATIM** (validated across Gen-0).

## 8. Source references
- [`multi_group_optimizer.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/multi_group_optimizer.py): `OptimizerGroupRegistry`, `DEFAULT_CLIP_NORM`
- [`schema.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/schema.py): `OPTIMIZER_GROUP_NAMES`
- Illustrated in Figure 4 (`gradient_flow.svg`).
- Connected to Chapters 12 (complete loop) and 16 (training system).


---

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


---

# Chapter 16 — Training System: Checkpoint Durability and Resumption

## 1. In one sentence
RHAN-NXA's training infrastructure enforces atomic checkpoint saves, code-identity-gated resume, best/rolling checkpoint parity, and mandatory HF sync, eliminating the silent restart, stale-resume, and parity-gap incidents that disrupted Generation-0 training.

## 2. Five durability rules (each motivated by a real Gen-0 incident)

### Rule 1: Never a silent restart
When a rolling checkpoint exists—locally or on Hugging Face—resume is **mandatory**. A trainer that cannot restore the checkpoint must raise `CheckpointResumeError` and abort. It must never fall through to a fresh run. This closes the 2026-08-12 stale-resume bug where a Colab session restart silently began training from epoch 0, discarding 40+ epochs of state.

### Rule 2: Code-identity guard
Every checkpoint records `code_commit` (the git HEAD SHA at save time). Before restoring any checkpoint, `resume_commit_ok(ckpt, current_sha)` verifies the saved SHA matches the current code. Legacy checkpoints without a recorded SHA are **always refused**. This prevents silently resuming training with changed semantics.

### Rule 3: Atomic saves
Every write is `tmp_file → fsync → rename`. A runtime crash mid-save cannot produce a partial checkpoint that becomes the only artifact on disk.

### Rule 4: Best/rolling parity from commit one
Both the rolling (resume) artifact and the best-eval artifact are written by the same `save_state` call. Before any evaluation cites either artifact, `verify_best_rolling_parity()` checks that they agree on code identity and that the rolling checkpoint is at least as recent as the best. A parity gap—best artifact from a different code commit than the rolling—is flagged before it corrupts a reported metric.

### Rule 5: Mandatory HF sync
`save_rolling` and `save_best` accept an optional `uploader` callable. When provided, the artifact is pushed to Hugging Face immediately after the local write. A local-only checkpoint is one runtime reset away from loss.

## 3. Implementation

### `atomic_torch_save(target_path, obj)`
`tempfile.mkstemp → torch.save → fsync → os.rename`

### `save_rolling(path, *, epoch, model, optimizer, scheduler, extra, uploader)`
Records: `epoch`, `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`, `code_commit`, `saved_at_utc`, `kind="rolling"`.

### `save_best(path, *, model, config, metric_value, uploader)`
Records: `model.state_dict()`, `config.to_dict()`, `metric_value`, `code_commit`, `saved_at_utc`, `kind="best"`. The embedded config prevents eval/train config drift.

### `resume_or_abort(rolling_path, hf_repo_id, hf_filename, force_restart)`
Decision table:
| Condition | Action |
|:---|:---|
| `force_restart=True` | Return None (cold start, loud warning) |
| Local rolling exists | Load, check code identity, return state |
| No local, HF verified | Download, load, check code identity, return state |
| No local, HF unverifiable | **Raise `CheckpointResumeError`** |
| No local, HF not configured | Return None (genuinely fresh run) |

### `verify_best_rolling_parity(best_path, rolling_path)`
Returns `(bool, message)`. Eval scripts must call this before citing either artifact.

## 4. The optimizer resume guard
`OptimizerGroupRegistry.resume_guard(state_dict, saved_scheduler)` protects against restoring an optimizer state from a different group layout. It verifies group count, group names, and the LR-ratio pattern. See Chapter 14.

## 5. Port status
The checkpoint module is **ADAPTED** from Gen-0's `phase1_training/train_rhan_next.py` and `checkpoint_utils.py`:
- Core atomic save and all five rules: carried unchanged.
- Equivalence-edge registry (notebook-only commit equivalences): stays in Gen-0 file; Gen-1 trainers call it explicitly when needed.

## 6. Scientific status
- **Atomic saves**: **LOCKED** (infrastructure requirement; three incidents).
- **Code-identity guard**: **LOCKED** (infrastructure requirement).
- **Best/rolling parity**: **LOCKED** (infrastructure requirement).

## 7. Source references
- [`checkpoint.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/checkpoint.py): `atomic_torch_save`, `save_rolling`, `save_best`, `resume_or_abort`, `verify_best_rolling_parity`, `CheckpointResumeError`
- Illustrated in Figure 8 (`checkpoint_resume.svg`).
- Connected to Chapter 14 (gradient flow / optimizer resume guard) and Chapter 17 (experimental DAG).


---

# Chapter 17 — Experimental DAG: Staged Isolation Protocol

## 1. In one sentence
The RHAN-NXA experiment plan is structured as a two-graph DAG: a **build dependency graph** (what must exist before what can be built) and a **scientific experiment dependency graph** (Steps 1–11, the validation ladder), where every arm's primary comparison is against a **frozen Step-6 reference**, never against another arm.

## 2. The isolation principle
The defining failure of Generation-0 was that D2 (AIS-v2 test) and D3 (belief-HPC test) both ran with unintentionally-active SBR, making their results **uninterpretable as isolated tests**. The Gen-1 protocol was designed specifically to prevent this:

> **Every ablation arm flips exactly one flag versus the Step-6 frozen reference. Every arm's primary comparison is to Step-6, never to another arm.**

Comparing two arms against each other reintroduces multi-mechanism attribution collapse—exactly the D2/D3 confound reproduced.

## 3. The Step-6 frozen reference
Step 6 is the **integrated Gen-1 core**:
- $S_t = \text{None}$ (no SBR)
- $L_\text{stab}$ diagnostic-only (not an objective)
- AIS-v2 active
- Full belief loop: $z_t$, $U_t$, $E_t$, $\Pi_t$, $A_t$

Step-6 must produce a validated ImageNet-100 result before any ablation arm begins evaluation. This result is frozen; ablation arms compare against it.

## 4. The eleven-step validation ladder

| Step | What it tests | Primary comparison |
|:---:|:---|:---|
| 1 | Backbone only (no belief loop) | TRADES baseline |
| 2 | Backbone + within-glimpse recurrence | Step 1 |
| 3 | + Belief state (no $E_t$) | Step 2 |
| 4 | + Prediction error $E_t$ (no AIS-v2) | Step 3 |
| 5 | + AIS-v2 (no $L_\text{stab}$) | Step 4 |
| **6** | **Full core** ($S_t=\text{None}$, $L_\text{stab}$ diagnostic) | Step 5 |
| 7 | + $S_t$ (2–4 slots, gated) | Step 6 |
| 8 | + $L_\text{stab}$ as objective (gated) | Step 6 |
| 9 | + V1 Gabor frontend (standalone ablation first) | Step 6 |
| 10 | Param-matched backbone-only control | Step 6 |
| 11 | Compute-matched backbone-only control | Step 6 |

Steps 7 and 8 can be **built in parallel** (they share Step-6's substrate), but must be **evaluated independently against Step-6**, not against each other.

## 5. Gate structure
Each step has pre-registered pass/fail criteria:

| Gate | Criterion | Status |
|:---:|:---|:---:|
| G6 | Uncertainty–error correlation (Pearson $r$ threshold, PENDING) | PENDING |
| G7 | Center-bias: gaze distribution not center-concentrated (threshold PENDING) | REQUIRED |
| G8 | Structure gate (per-slot decodability, N/A while $S_t=\text{None}$) | N/A |
| G9 | Drift without collapse: $L_\text{stab}$ reduction alongside OOD responsiveness | REQUIRED as designed |
| G10 | No silent alteration: all arms have identical ablation columns | REQUIRED |

**Gate 9 critical rule**: low adversarial drift achieved by becoming insensitive to all new evidence is a **FAILURE**, not a success. Drift reduction must be reported alongside an OOD/novel-evidence responsiveness score on a disjoint probe set. Low drift on both simultaneously = FAILED.

## 6. Matched controls (mandatory for mechanism claims)
Before claiming any mechanism improves robustness, two controls must pass:

- **Parameter-matched**: backbone-only run with total parameters equal to the full model. Excludes "more parameters helped."
- **Compute-matched**: backbone-only run with FLOPs equal to the full model (via width expansion). Excludes "more compute helped."

The Gen-0 SBR rungs proved this failure mode is live: their clean accuracy gains tracked added capacity, not structural mechanism.

## 7. The confound documentation requirement
Every experiment that touches a mechanism that was active in Gen-0's D2/D3 (AIS-v2, belief-HPC) must document the confound clearly and run a clean re-test. The `_nx_trainer` SBR bug must not be silently forgotten.

## 8. Scientific status
- **Step-6 as frozen reference**: **REQUIRED** (isolation protocol).
- **Matched controls**: **REQUIRED** (standing rule from Gen-0 Pareto lesson).
- **Per-arm flag isolation**: **REQUIRED** (no arm may swap two flags simultaneously).

## 9. Source references
- `noesis_vision/RHAN_NXA/docs/24_Training_Phase_DAG.md`
- `noesis_vision/RHAN_NXA/docs/25_Ablation_Matrix.md`
- `noesis_vision/RHAN_NXA/docs/28_Gates_and_Compute_Accounting.md`
- [`schema.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/schema.py): `GATED_FLAGS`, `step6_validated_result`
- Illustrated in Figure 9 (`experiment_isolation.svg`).
- Connected to Chapters 16 (training system) and 19 (Gen-0 evidence).


---

# Chapter 18 — Evaluation Protocols: PGD Robustness and Verification Metrics

## 1. In one sentence
RHAN-NXA uses the Generation-0 validated 16-seed adversarial evaluation protocol: PGD-100 and PGD-50 at $\epsilon = 0.094$ in norm space, 300 samples/seed, 16 seeds (41–56), with adversarial accuracy as the primary metric and all comparators carried via a frozen registry.

## 2. The protocol specification

| Parameter | Value | Source |
|:---|:---|:---|
| Dataset | STL-10 test (Gen-0 / Gen-1 STL Phase) | Gen-0 validated |
| Samples per seed | 300 | Gen-0 protocol |
| Seeds | 16 (seeds 41–56) | Gen-0 protocol |
| Attack | PGD-100 (primary) and PGD-50 (secondary) | Gen-0 protocol |
| $\epsilon$ | 0.094 (normalized) | Gen-0 protocol |
| Primary metric | Adversarial accuracy (percentage) | |
| Secondary metric | Clean accuracy | |
| Significance test | $|\Delta| > 2\sqrt{\sigma_a^2 + \sigma_b^2}$ | Conservative crossover criterion |
| Comparators | Registered frozen (never re-evaluated) | Prevents seed-selection bias |

**Note**: AutoAttack evaluation is expected but details are pending.

## 3. Why 16 seeds?
Single-seed evaluation has high variance—a model can appear to "improve" by 3–5 percentage points on a single evaluation run just due to noise in the attack initialization. The 16-seed protocol provides:
- Standard deviation estimates across seeds ($\pm$ values in all reported numbers)
- A conservative crossover test that requires $|\Delta| > 2\sqrt{\sigma_a^2 + \sigma_b^2}$

This is what makes the Gen-0 headline numbers reliable: D vs TRADES (+9.79 pp adversarial, 2σ = 7.72) is real; ais_v2 vs D (+1.63 pp, 2σ = 8.26) is not significant.

## 4. The frozen comparator registry
All baseline comparators (D, TRADES, sbr2–sbr4, ais_v2, hpc_belief) are evaluated **once** and their numbers are stored in a registry. Subsequent arms compare against registry numbers; they never re-evaluate baselines. This prevents:
- Baseline numbers being updated after new arms reveal weakness
- Seed-selection bias in comparator evaluation

## 5. Evaluation scripts
The following evaluation scripts exist in the workspace:

| Script | Purpose |
|:---|:---|
| `eval_pgd_final.py` | Final PGD evaluation |
| `eval_aa_v2.py` | AutoAttack evaluation |
| `eval_pgd_sweep.py` | Sweep across configurations |
| `eval_quick_perclass.py` | Per-class accuracy breakdown |
| `evaluation/clean_and_robust.py` | Matched norm-space clean+robust evaluation |
| `evaluation/compactness_report.py` | Model compactness measurement |

## 6. Metric interpretation rules

### On adversarial accuracy
- Numbers are reported as mean ± std over 16 seeds.
- Only differences exceeding 2σ are described as "real."
- Smaller differences are reported as "not significant" and not interpreted causally.

### On clean accuracy
- A model should not purchase adversarial robustness improvements by sacrificing clean accuracy.
- The Gen-0 Pareto lesson: every SBR rung bought +4.5–8.4 pp clean and paid −6.3–10.4 pp adversarial.

### On gate results
- A gate pass means the pre-registered criterion passed.
- A gate pass does not mean the mechanism was validated as effective (see Chapter 22: sbr0 gate-pass example).

## 7. What the evaluation does NOT currently measure
- Robustness to adaptive attacks (AA second-order)
- Cross-dataset transfer
- Out-of-distribution generalization
- Computational cost at inference (though compactness is measured)

## 8. Scientific status
- **16-seed PGD protocol**: **REQUIRED** (matched to all Gen-0 numbers).
- **Frozen comparator registry**: **REQUIRED** (isolation rule).
- **Significance threshold $2\sqrt{\sigma_a^2 + \sigma_b^2}$**: **REQUIRED** (pre-registered conservative criterion).

## 9. Source references
- `evaluation/clean_and_robust.py`
- `evaluation/compactness_report.py`
- `noesis_vision/RHAN_NXA/docs/16_Gen0_Evidence_And_Confounds.md`
- Connected to Chapter 17 (experimental DAG) and Chapter 19 (Gen-0 evidence).


---

# Chapter 19 — Generation-0 Evidence: Empirical Record and Infrastructure Confounds

## 1. In one sentence
Generation-0 produced a 16-seed verified table establishing D's +9.79 pp robustness advantage over TRADES and the AIS-v2 candidate-preference correlation ($r = 0.706$), but the D2 (ais_v2) and D3 (hpc_belief) rows are confounded by an unintentionally-active SBR runner flag, making their isolated mechanism contributions UNKNOWN.

## 2. The Gen-0 ladder
Gen-0 ran a sequence of experiments on STL-10:

> **gen0 → sbr0 → sbr1 → sbr2 → sbr3 → sbr4 → ais_v2 (D2) → hpc_belief (D3)**

Each rung added or swapped one mechanism, gated by pre-registered criteria. Results were evaluated under the 16-seed protocol (Chapter 18).

## 3. Verified headline numbers

| Checkpoint | Clean acc | Adv acc PGD-100 @0.094 |
|:---|:---:|:---:|
| TRADES Large baseline | 54.81 ± 2.35 | 24.23 ± 1.94 |
| **D (donor: ais_hpc)** | **54.96 ± 2.37** | **34.02 ± 3.24** |
| E1 recon | 58.06 ± 2.38 | 33.12 ± 2.64 |
| E2 SBR (intentional) | 45.06 ± 3.30 | 33.42 ± 2.93 |
| E3 T6 | 57.25 ± 2.54 | 30.77 ± 2.98 |
| sbr2 | 63.35 ± 2.78 | 23.58 ± 2.45 |
| sbr3 | 59.44 ± 2.78 | 27.69 ± 3.11 |
| sbr4 | 61.79 ± 2.71 | 25.06 ± 2.95 |
| ais_v2 (D2) | 48.25 ± 2.20 | 35.65 ± 2.56 |
| hpc_belief (D3) | 45.06 ± 2.64 | 33.96 ± 2.26 |

*Source: `report/Gen0.md`, machine-verified against HF per-seed CSVs.*

## 4. What is VERIFIED (16-seed, matched protocol)

| Finding | Status |
|:---|:---:|
| D beats TRADES by +9.79 adv at equal clean (2σ = 7.72) | **VERIFIED** |
| ais_v2 vs TRADES: +11.42 adv (2σ = 6.42) — largest margin | **REAL, but confounded** |
| ais_v2 vs D: +1.63 adv (2σ = 8.26) — not significant | **NOT SIGNIFICANT** |
| hpc_belief vs D: −0.06 adv — nothing | **NOT SIGNIFICANT** |
| Every SBR rung vs D: clean +4.5–8.4pp (real), adv −6.3–10.4pp (real) | **VERIFIED** |
| E1 recon vs D: adv −0.90pp — not significant; clean +3.10pp — not significant | **NOT SIGNIFICANT** |

## 5. The confound: `_nx_trainer` SBR flag
**Affected**: ais_v2 (D2), hpc_belief (D3). Both evaluated via `sweep_rhan_nx_*` / the `_nx_trainer` runner, which **hardcodes `--enable-sbr`** for every stage. These were supposed to be isolated swap tests (AIS-v2 alone; belief-HPC alone). They are not. Both carry legacy SBR (16 slots, 512-dim) unintentionally.

**Not affected**: D, E1, E2, E3 — evaluated via the older `sweep_stage3_d_*` / `sweep_stage4_*` provenance path. E2's SBR is intentional. sbr2–sbr4 run through `_nx_trainer`, but SBR being active is correct by design.

**The tell**: D3's clean accuracy (45.06 ± 2.64) and E2's clean accuracy (45.06 ± 3.30) are **identical to two decimal places**, from experiments testing supposedly different mechanisms. This is the SBR-legacy signature dominating both results.

**Consequence**: AIS-v2's and belief-HPC's TRUE isolated effect on 16-seed clean/robust accuracy is **UNKNOWN**. Not "weak"—unknown. Everything else in the table stands as reported.

## 6. The salvageable result: AIS-v2 correlation ($r = 0.706$)
The AIS-v2 smoke-gate result was measured via a **narrower, more targeted diagnostic** (`ais_v2_smoke_gate_v1`, 512 samples) that measured whether the candidate-scoring head's predictions correlate with policy choice—largely orthogonal to whether SBR is active. This gives:

- g1 (gaze-shift): 0.109 ≥ 0.02 threshold ✓
- g2 (candidate-preference): **r = 0.706** (1536 pairs) ≥ 0.05 threshold ✓

This is graded **REQUIRED** evidence for the AIS-v2 mechanism. It is "the single most valuable new fact Gen-0 produced."

## 7. The Pareto lesson
Sorted by clean accuracy, adversarial accuracy comes out inverted across all Gen-0 rungs. Adding structured representational capacity (SBR) slides models along the clean/robust frontier but **never moves the frontier itself**. This is why Gen-1's matched-control requirement exists: any future "mechanism helps" claim must exclude "more parameters/compute helped."

## 8. Gate verdicts

| Gate | Result | Notes |
|:---|:---:|:---|
| sbr0 (`sbr0_gate_v1`) | Passed criteria | BUT: per-slot accuracies ≈ chance; slot vacuity not excluded |
| sbr1 (one-sided collapse) | Passed | |
| ais_v2 (`ais_v2_smoke_gate_v1`) | Passed | Strongest mechanism result of Gen-0 |
| hpc_belief gate | **NOT RECORDED** | Verdict not on roadmap; UNVERIFIED |
| sbr2 masking gate | Computed but not official | UNVERIFIED as artifact |

## 9. Provenance
- Per-seed CSVs: HF `FerrariKazu/rhan-eval-sweep`
- Aggregation script: `scratch/gen0_pull_hf_evals.py`
- Raw cache: `report/_gen0_pull.json`
- Full tables and contradiction log: `report/Gen0.md`
- Roadmap: `FerrariKazu/rhan-checkpoints-rolling/rhan_next_roadmap.json`

## 10. Source references
- `noesis_vision/RHAN_NXA/docs/16_Gen0_Evidence_And_Confounds.md`
- [`report/Gen0.md`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/report/Gen0.md)
- `report/lens_e1_analysis/E1_FULL_AUDIT_REPORT.md`
- Connected to Chapters 17 (experimental DAG) and 18 (evaluation protocols).


---

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


---

# Chapter 21 — Formal Decision Records: DR-001 through DR-010

> *Level 2–3 reading. Institutional memory: why each non-obvious architectural choice was made, what alternatives were evaluated and rejected, and what pre-registered empirical conditions could overturn it.*

---

## 1. Overview and Record Schema

The architecture of RHAN-NXA is governed by formal Decision Records. Each record preserves the historical rationale, mathematical reasoning, empirical evidence, rejected alternatives, remaining uncertainties, and the falsifying experiment for key architectural commitments.

Every record conforms to a strict 8-point schema:

```text
Decision:                         The core architectural or procedural choice.
Current choice:                   The active implementation setting in Generation-1.
Status:                           LOCKED | REQUIRED | EXPERIMENTAL CANDIDATE | 
                                  DEFERRED | PENDING DECISION | REJECTED.
Why:                              The foundational theoretical and causal rationale.
Evidence:                         Empirical numbers, seed counts, statistical tests.
What was rejected:                Specific alternatives dismissed and why.
What remains uncertain:           Unresolved trade-offs and theoretical boundaries.
Experiment that could change it:  Pre-registered empirical trigger for revision.
```

---

## 2. DR-001 — Why $S_t = \text{None}$ Initially

```text
Decision:                         Exclude explicit structural state from the Gen-1 core.
Current choice:                   S_t = None through the first integrated system and its
                                  first ablation matrix.
Status:                           REQUIRED (as the default configuration).
Why:                              The burden of proof is on added structural machinery. 
                                  The clean recurrent predictive coding core must validate
                                  its baseline capability before carrying complex, high-risk
                                  object-centric modules.
Evidence:                         Doubled empirical evidence:
                                  1. sbr0 ablation: zeroing out the top slot improved clean
                                     accuracy (retained ratio = 1.0157 vs a 0.7 floor designed
                                     to catch slot dependency).
                                  2. Clean collapse: a -9 to -10 pp clean accuracy degradation
                                     reproducibly appeared whenever the Gen-0 slot implementation
                                     was active—both intentionally (E2, sbr2–4) and 
                                     accidentally (D2, D3).
What was rejected:                Carrying forward the 16-slot SBR "just in case" or assuming
                                  that passing a low-threshold gate implied genuine object grouping.
What remains uncertain:           Whether any slot-based mechanism can pass a probe that detects
                                  true slot distinguishability.
Experiment that could change it:  Post-ImageNet-100 validation: a compact 2–4 slot variant evaluated
                                  with per-slot decodability distinguishable from each other and
                                  from chance under tightened Gate 1 conditions.
```

---

## 3. DR-002 — Why Pixel Reconstruction Was Rejected

```text
Decision:                         Do not use pixel-space reconstruction as E_t's prediction target.
Current choice:                   Rejected for Generation-1.
Status:                           REJECTED (for Gen-1).
Why:                              Two independent arguments:
                                  1. Mechanistic: pixel reconstruction penalizes high-frequency 
                                     textures, diluting precision Π_t and increasing belief drift
                                     (observed across Gen-0 Lens analysis).
                                  2. Engineering: pixel generation requires a heavy spatial decoder
                                     absent from the compact ViT substrate, inflating parameter 
                                     budgets and complicating gradient isolation.
Evidence:                         Gen-0 Lens analysis demonstrated increased drift. Experiment E1 
                                  measured a -0.90 pp clean drop vs Configuration D (statistically 
                                  inconclusive, but mechanistically disfavored and computationally 
                                  burdensome).
What was rejected:                Claiming "pixel reconstruction was proven catastrophically harmful" 
                                  (the effect was statistically marginal, but engineering cost was 
                                  unjustified).
What remains uncertain:           Whether a decoder-free dense spatial target could provide value in a 
                                  future multi-task setting.
Experiment that could change it:  A visual substrate with native dense spatial representations evaluated
                                  with significant, Lens-verified stability gains under matched controls.
```

---

## 4. DR-003 — Why Latent Glimpse Prediction Was Selected

```text
Decision:                         E_t is computed as the mismatch between predicted and observed 
                                  patch-level latent features at the next glimpse location.
Current choice:                   Locked as the Gen-1 design default.
Status:                           EXPERIMENTAL CANDIDATE (Design LOCKED, empirical value under 
                                  evaluation due to Gen-0 D3 confounding).
Why:                              1. Representationally symmetric by construction: predicted features 
                                     and observed features share the identical ViT patch-embedder space.
                                  2. Zero auxiliary decoder: the patch embedding output serves as both
                                     target format and ground truth.
                                  3. Unification with active sensing: a single predictor serves both 
                                     belief updating (E_t) and AIS-v2 gaze candidate scoring.
What was rejected:                Global state prediction (predicting z_{t+1} directly from z_t, which
                                  is trivial and prone to identity shortcuts); naive additive fusion 
                                  (z_t + λ Π E_t), which failed due to dimensional and scale mismatch.
What remains uncertain:           Whether latent prediction beats the null baseline (F = identity, no E_t) 
                                  under strictly unconfounded training.
Experiment that could change it:  The pre-registered unconfounded rerun of Configuration D3 (SBR disabled)
                                  benchmarked against Configuration D (null prediction).
```

---

## 5. DR-004 — Why $T = 4$ Glimpses

```text
Decision:                         Execute exactly T = 4 discrete glimpses per image forward pass.
Current choice:                   T = 4 for the initial build and validation matrix.
Status:                           REQUIRED for the first build (revisable via ablation).
Why:                              Reuses the validated setting from the project's empirical lineage
                                  rather than arbitrarily introducing another uncalibrated hyperparameter.
Evidence:                         Empirically verified in Gen-0 as balancing classification accuracy, 
                                  gaze trajectory convergence, and compute budgets.
What was rejected:                Deriving T dynamically at build time or using variable unrolled depths
                                  without empirical baselines.
What remains uncertain:           Whether T = 4 remains optimal when scaling from 96x96 inputs (STL-10) 
                                  to 224x224 inputs (ImageNet-100).
Experiment that could change it:  A rigorous T-sweep ablation (T ∈ {2, 4, 6, 8}) under parameter-matched
                                  and FLOP-matched controls.
```

---

## 6. DR-005 — Why Tied Within-Glimpse Recurrence

```text
Decision:                         Run a single shared Transformer encoder block iteratively 2–3 times 
                                  per glimpse before spatial pooling (Universal Transformer style).
Current choice:                   Weight-tied iterative token refinement.
Status:                           LOCKED as design (Option C — hybrid — selected narrowly).
Why:                              Option A alone (unrolled across glimpses with no internal refinement) 
                                  abandons the iterative token hypothesis. Option B alone (deep untied 
                                  stack per glimpse) balloons parameters and obscures whether gains stem 
                                  from depth or recurrence. Tied weights scale compute without adding 
                                  parameters, preserving the strict 20–25M budget.
Evidence:                         Theoretical capacity analysis and the SBR capacity-vs-mechanism risks.
What was rejected:                Untied stacked layers per glimpse; single-pass superficial token pooling.
What remains uncertain:           The exact performance delta between 2 versus 3 within-glimpse iterations.
Experiment that could change it:  Within-glimpse iteration count ablation (1 vs 2 vs 3 iterations) 
                                  evaluated under FLOP-matched baselines.
```

---

## 7. DR-006 — Why Adaptive Halting Is Deferred

```text
Decision:                         Enforce fixed glimpse depth T = 4; no learned or thresholded halting 
                                  in the initial Generation-1 build.
Current choice:                   Deferred.
Status:                           DEFERRED.
Why:                              Adaptive halting introduces an auxiliary halting policy and loss term 
                                  that has a documented history in this project of conflicting with primary 
                                  classification and stability objectives. The core loop must validate at 
                                  constant depth first.
Evidence:                         Gen-0 empirical record: in Model v10, a halting objective triggered 
                                  severe loss conflicts and numerical instabilities; AIS-v1 halting showed 
                                  marginal gains while complicating optimization.
What was rejected:                Simultaneous co-training of adaptive halting with the unvalidated core loop.
What remains uncertain:           Whether halting yields meaningful computational savings on natural images 
                                  without sacrificing worst-case adversarial robustness.
Experiment that could change it:  Reintroducing a halting head on a frozen, validated Step 6 core checkpoint 
                                  with strict halting-on vs halting-off controls.
```

---

## 8. DR-007 — Why AIS-v2 Uses the Shared Predictor

```text
Decision:                         Deploy a single GlimpseFeaturePredictor consumed jointly by the realized 
                                  glimpse prediction error (E_t) and the candidate gaze scoring policy.
Current choice:                   Shared predictor; gaze scoring via predicted uncertainty reduction from 
                                  Dirichlet evidential entropy.
Status:                           REQUIRED as a foundational design rule.
Why:                              Maintaining two separate predictors (one for state updates and one for gaze) 
                                  creates representation drift between where the model looks and how it updates. 
                                  A shared predictor enforces representational parsimony.
Evidence:                         Strong mechanism validation: Gen-0 Pearson correlation r = 0.706 between 
                                  predicted uncertainty reduction and actual error reduction.
What was rejected:                A dedicated, independent gaze-scoring network; predicting pooled global 
                                  vectors instead of localized patch tokens.
What remains uncertain:           Whether long-run training maintains alignment between realization updates 
                                  and planning scoring without loss interference.
Experiment that could change it:  Empirical divergence between update gradients and scoring gradients 
                                  under the shared predictor parameter group.
```

---

## 9. DR-008 — Why Uncertainty Derives from Dirichlet Evidence

```text
Decision:                         U_t is parameterized as class-conditioned Dirichlet evidence via the 
                                  EvidentialHead.
Current choice:                   Retained for Generation-1.
Status:                           REQUIRED for Gen-1 (with explicit acknowledgment of the scoping compromise: 
                                  readout-level, not belief-content-level).
Why:                              The EvidentialHead provides an established, closed-form epistemic uncertainty 
                                  metric ($K / \sum \alpha_k$) with analytical properties. Representation-level 
                                  uncertainty lacks a validated ground-truth target.
Evidence:                         Component validation across the Gen-0 lineage and literature (Sensoy et al.).
What was rejected:                Deferring all uncertainty representations (which would disable precision 
                                  weighting and AIS-v2 scoring); inventing untested self-supervised uncertainty 
                                  schemes during core integration.
What remains uncertain:           The theoretical gap between uncertainty about task labels and uncertainty 
                                  about visual latent features.
Experiment that could change it:  Validation of an unconfounded, self-supervised epistemic uncertainty head 
                                  trained directly on latent representation manifolds.
```

---

## 10. DR-009 — Why Representation-Level Uncertainty Is Deferred

```text
Decision:                         Do not attempt to construct direct uncertainty metrics over z_t or S_t 
                                  tensor contents in Generation-1.
Current choice:                   Pending decision; deferred past initial core validation.
Status:                           PENDING DECISION.
Why:                              Inventing complex representation-level density estimators or variance 
                                  heads mid-build repeats the Generation-0 failure pattern of stacking 
                                  unvalidated mechanisms onto an unverified base.
Evidence:                         The Master Implementation Plan's formal Tension Specification (Part 1.A).
What was rejected:                Equating class evidential uncertainty with visual feature entropy, or 
                                  claiming that U_t represents complete perceptual doubt.
What remains uncertain:           What supervision signal or generative target could calibrate latent-level 
                                  uncertainty without destabilizing ViT embeddings.
Experiment that could change it:  Same as DR-008.
```

---

## 11. DR-010 — Why the 16-Slot SBR Was Not Carried Forward

```text
Decision:                         Do not port the Generation-0 16-slot Spatial Bottleneck Recurrence (SBR) 
                                  into the RHAN-NXA architecture.
Current choice:                   Rejected as an implementation.
Status:                           REJECTED (Implementation only; the abstract concept of structural state 
                                  remains an EXPERIMENTAL CANDIDATE, DEFERRED).
Why:                              **Rejected Implementation $\neq$ Rejected Idea.** The 16-slot implementation 
                                  lacked evidence of functional specialization and imposed a severe, reproducible 
                                  performance degradation.
Evidence:                         1. sbr0 gate probes: individual slot classification accuracies ranged from 
                                     0.44 to 0.51 (statistically indistinguishable from random chance).
                                  2. Ablation paradox: zeroing out the dominant slot yielded a retained accuracy 
                                     of 1.0157, proving the network was not functionally dependent on the slots.
                                  3. Clean collapse: a persistent 9–10 percentage point penalty across E2, sbr2–4, 
                                     D2, and D3.
What was rejected:                Treating "gate passed" as evidence of genuine capability when the verification 
                                  gate was structurally incapable of detecting vacuous solutions.
What remains uncertain:           Whether any slot-based architectural mechanism can achieve functional 
                                  differentiation in compact visual backbones.
Experiment that could change it:  The DR-001 re-entry protocol: a clean 2–4 slot variant evaluated under 
                                  strictly hardened decodability gates.
```

---

## 12. Summary Matrix of Decision Records

| Record | Architectural Choice | Disposition | Status | Primary Rationale |
|---|---|---|---|---|
| **DR-001** | Initial Structural State $S_t = \text{None}$ | Locked Core | **REQUIRED** | Burden of proof on structure; avoid 16-slot collapse |
| **DR-002** | Rejection of Pixel Reconstruction | Dismissed Target | **REJECTED** | Mechanistically dilutes precision; requires heavy decoder |
| **DR-003** | Latent Glimpse Prediction Target | Core Target | **EXPERIMENTAL CANDIDATE** | Representationally symmetric; unifies prediction & gaze |
| **DR-004** | Fixed Glimpse Depth $T = 4$ | Recurrence Depth | **REQUIRED** | Validated project lineage; compute/accuracy balance |
| **DR-005** | Tied Within-Glimpse Recurrence | Block Iteration | **LOCKED** | Adds compute, not parameters; preserves 20–25M budget |
| **DR-006** | Adaptive Halting Deferral | Termination Policy | **DEFERRED** | Prevents objective conflict; isolates core loop validation |
| **DR-007** | Shared Predictor for Error & AIS-v2 | Gaze / Error Sync | **REQUIRED** | Eliminates dual-predictor drift; r = 0.706 mechanism link |
| **DR-008** | Dirichlet Evidential Uncertainty $U_t$ | Uncertainty Type | **REQUIRED** | Validated closed-form prior; avoids mid-build invention |
| **DR-009** | Deferral of Representation Uncertainty | Latent Entropy | **PENDING DECISION** | Lack of validated self-supervised training targets |
| **DR-010** | Rejection of 16-Slot SBR Code | Slot Attention | **REJECTED** | Chance-level slot decoding; reproducible -10 pp collapse |

---

## 13. Related Components and System Context

- [Chapter 06 — Structure State $S_t$](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/06_Structure_State.md): Context for DR-001 and DR-010.
- [Chapter 08 — Prediction Error $E_t$](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md): Context for DR-002 and DR-003.
- [Chapter 09 — Recurrence](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/09_Recurrence.md): Context for DR-004, DR-005, and DR-006.
- [Chapter 11 — AIS-v2 Gaze Policy](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md): Context for DR-007.
- [Chapter 19 — Generation-0 Evidence](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/19_Gen0_Evidence.md): Deep-dive into the empirical numbers underlying these records.


---

# Chapter 22 — Common Misunderstandings and Anti-Patterns

> *Level 1–2 reading. A definitive guide to misconceptions, architectural pitfalls, and common conceptual errors regarding the RHAN-NXA research codebase.*

---

## 1. Overview

Because RHAN-NXA synthesizes ideas from predictive coding, active sensing, evidential uncertainty, and recurrent vision, it is easy for researchers and engineers to map its mechanisms onto familiar standard deep learning patterns where they do not belong.

This chapter explicitly documents the **ten most prevalent misunderstandings** about RHAN-NXA, explains *why* each is wrong, details the architectural failure that results from holding it, and specifies the correct conceptual model.

---

## 2. Misunderstanding 1: "RHAN-NXA Just Looks at an Image 4 Times Instead of Once"

### The Misconception
Viewing RHAN-NXA as a glorified multi-crop test-time augmentation (TTA) pipeline or an ensemble of 4 glimpses averaged together.

### Why It Is Wrong
In standard multi-crop or attention mechanisms, multiple spatial patches are processed independently or in parallel and aggregated via simple pooling:
$$\hat{y} = \frac{1}{T} \sum_{t=1}^T f(x_{\text{crop}_t})$$

In RHAN-NXA, perception is **active, stateful, and sequential**:
1. Glimpse $t$ is selected conditionally based on where uncertainty reduction is predicted to be highest given current belief $B_{t-1}$.
2. The internal state $z_t$ is not replaced or averaged; it is refined recurrently through within-glimpse Transformer iterations and across-glimpse UpdateNet integration.
3. What is observed at glimpse $t$ is compared against what was *predicted* by $B_{t-1}$. The update is driven by the discrepancy $E_t$, precision-weighted by $\Pi_t$.

```
Static Multi-Crop:      Crop 1 ──► [Enc] ──┐
                         Crop 2 ──► [Enc] ──┼──► Average ──► Output
                         Crop 3 ──► [Enc] ──┘

RHAN-NXA:               B_0 ──► Plan a_1 ──► Observe g_1 ──► E_1 ──► UpdateNet ──► B_1
                                                                                   │
                                ┌──────────────────────────────────────────────────┘
                                ▼
                               Plan a_2 ──► Observe g_2 ──► E_2 ──► UpdateNet ──► B_2 ...
```

---

## 3. Misunderstanding 2: "$U_t$ Measures Uncertainty About Everything in the Scene"

### The Misconception
Assuming that $U_t$ represents full scene-level epistemic uncertainty or visual entropy over the entire image.

### Why It Is Wrong
$U_t$ in Generation-1 is computed strictly by the **EvidentialHead** operating on the global latent vector $z_t$:
$$U_t = \frac{K}{\sum_{k=1}^K (\alpha_k + 1)}$$
where $\alpha_k$ is the Dirichlet evidence parameter for class $k \in \{1, \dots, K\}$.

- $U_t$ is strictly **class-conditioned epistemic uncertainty** over the classification readout manifold.
- It does **not** quantify uncertainty about background pixels, unobserved spatial regions, 3D geometry, or object boundaries.
- Calling $U_t$ "scene uncertainty" conflates semantic task uncertainty with spatial or representational entropy. A model can be 100% confident that an image contains a dog while being completely ignorant of whether the dog has three legs or four.

---

## 4. Misunderstanding 3: "$S_t = \text{None}$ Means RHAN-NXA Has Abandoned Object-Centric Perception"

### The Misconception
Believing that setting $S_t = \text{None}$ in the Generation-1 core indicates that the project rejected the hypothesis of structural, object-centric representations.

### Why It Is Wrong
As documented in **DR-001** and **DR-010**:
- **A Rejected Implementation is Not a Rejected Idea.**
- The Generation-0 16-slot Spatial Bottleneck Recurrence (SBR) was rejected because its specific implementation collapsed: per-slot classification probes hovered at chance (0.44–0.51), slot zeroing improved accuracy (retained ratio 1.0157), and it reproducibly dragged clean accuracy down by ~10 percentage points.
- $S_t = \text{None}$ is an experimental discipline rule: the clean recurrent predictive coding core must validate its baseline before carrying structural modules.
- Re-entry of $S_t$ is pre-registered in Part 2 Step 7 as a compact 2–4 slot module evaluated under rigorous slot-decodability gates.

---

## 5. Misunderstanding 4: "Prediction Error $E_t$ Can Be Safely Detached During Training"

### The Misconception
Calling `.detach()` on $E_t = \|g_t^{\text{obs}} - \hat{g}_t^{\text{pred}}\|_2$ or on the input to UpdateNet to prevent gradient instability or save memory.

### Why It Is Wrong
Detaching $E_t$ completely destroys predictive coding:
1. If $E_t$ is detached, no gradient flows from the downstream classification loss $\mathcal{L}_{\text{task}}$ or stability loss $\mathcal{L}_{\text{stab}}$ back through the prediction error into the `GlimpseFeaturePredictor`.
2. The predictor would then receive gradients *only* from the auxiliary prediction loss $\mathcal{L}_{\text{pred}}$ (Group 3). It would never learn to predict features that are *useful for belief updating*.
3. More critically, the backbone would lose the gradient incentive to produce predictable representations. The closed-loop synergy between prediction, observation, and representation would be permanently severed.

---

## 6. Misunderstanding 5: "AIS-v2 Has Been Proven to Improve Adversarial Robustness"

### The Misconception
Citing Generation-0 benchmark tables as proof that AIS-v2 active sensing delivers superior adversarial robustness.

### Why It Is Wrong
This is the central scientific confound identified in Generation-0:
- While Configuration D achieved +9.79 pp over TRADES on STL-10, Configuration D **did not use AIS-v2**; it used fixed/null gaze.
- In Configurations D2 and D3 where AIS-v2 was tested, the training script accidentally enabled `--enable-sbr`, entangling AIS-v2 with the dysfunctional 16-slot SBR module.
- As a consequence, the **isolated system-level robustness contribution of AIS-v2 is scientifically UNKNOWN**.
- What *is* verified is the **mechanism validity** of AIS-v2: its gaze scoring metric correlates with actual error reduction at $r = 0.706$. But whether this mechanism yields net system-level robustness on ImageNet-100 is an open hypothesis currently undergoing unconfounded re-evaluation.

---

## 7. Misunderstanding 6: "$\mathcal{L}_{\text{stab}}$ Should Be Added to the Loss Function Right Away"

### The Misconception
Activating $\mathcal{L}_{\text{stab}}$ as an auxiliary loss term during initial training of the core model.

### Why It Is Wrong
Enabling $\mathcal{L}_{\text{stab}}$ prematurely violates the fundamental principle of single-mechanism attribution:
- If a model trained with $\mathcal{L}_{\text{stab}}$ exhibits high robustness, you cannot tell whether the robustness arose from the recurrent predictive coding loop or from the brute-force trajectory penalty.
- Furthermore, optimizing $\mathcal{L}_{\text{stab}}$ without baseline calibration risks triggering **Gate 9 failure**: the model can achieve zero drift simply by collapsing into representation invariance, ignoring all new evidence.
- The staged protocol is **LOCKED**: Phase A is strictly diagnostic-only. Phase B promotion occurs only after Step 6 baseline validation.

---

## 8. Misunderstanding 7: "UpdateNet Is Just an Overcomplicated Residual Connection"

### The Misconception
Replacing UpdateNet with a simple linear projection or raw element-wise addition:
$$z_t = z_{t-1} + \lambda \cdot \Pi_t \odot E_t$$

### Why It Is Wrong
1. **Representational Mismatch**: $z_t$ is a global summary vector of dimension $D_z = 384$. $E_t$ is a localized spatial patch error tensor of shape $(B, 16, 384)$ or a spatially pooled patch error. They do not share identical manifolds or semantic geometries.
2. **Dynamic Gating**: UpdateNet is a learned MLP that models non-linear interactions between prior belief, prediction discrepancy, and precision weighting.
3. **Bounded Updates**: UpdateNet enforces a strict `DELTA_BOUND = 0.1` via $\tanh$ scaling:
   $$\Delta z_t = \text{DELTA\_BOUND} \cdot \tanh(\text{MLP}([z_{t-1}, \Pi_t \odot E_t]))$$
   This prevents any single corrupted glimpse from catastrophically destabilizing the global belief state. Raw addition allows unbounded adversarial perturbation injection.

---

## 9. Misunderstanding 8: "Prediction Targets Should Be Pixel-Level for Maximum Detail"

### The Misconception
Arguing that the predictor should reconstruct raw pixel values $(56 \times 56 \times 3)$ rather than latent ViT patch embeddings.

### Why It Is Wrong
1. **High-Frequency Vulnerability**: Adversarial attacks inject imperceptible high-frequency pixel noise. Forcing the model to predict pixels forces it to spend predictive capacity modeling noise rather than semantic structure.
2. **Precision Dilution**: Gen-0 Lens analysis showed that pixel-space error heads lead to diffuse, low-precision error maps that destabilize belief trajectories.
3. **Architectural Asymmetry**: Predicting pixels requires a heavy deconvolutional or transposed-attention spatial decoder. Predicting latent patch tokens preserves complete symmetry: the encoder's patch embedder produces the ground truth, and the compact ViT produces the target.

---

## 10. Misunderstanding 9: "AIS-v2 Samples From the Continuous Entire Image Plane"

### The Misconception
Assuming AIS-v2 evaluates a continuous policy $\pi(a | z)$ over all continuous $(x, y) \in [-1, 1]^2$ coordinates via reinforcement learning or policy gradients.

### Why It Is Wrong
AIS-v2 uses a **differentiable discrete candidate ranking mechanism**:
1. A candidate generator proposes a discrete set of $K \in \{4, \dots, 8\}$ candidate fixation coordinates (combining prior salience, grid coverage, and exploration).
2. The shared predictor projects anticipated features $\hat{g}^{(k)}$ for each candidate.
3. Uncertainty reduction is scored analytically for each candidate using Dirichlet entropy.
4. Selection is executed via a tempered Softmax distribution during training (preserving gradient flow) or Argmax during deterministic evaluation.
5. This avoids the high variance and sample inefficiency of REINFORCE policy gradients.

---

## 11. Misunderstanding 10: "The Entire RHAN-NXA Perceptual Loop Has Been Validated End-to-End"

### The Misconception
Believing that the full recurrent loop (Backbone + Recurrence + Predictor + UpdateNet + AIS-v2 + Evidential Head) has already been proven to work together seamlessly in published benchmarks.

### Why It Is Wrong
The complete perceptual loop is an **actively developing architecture undergoing initial end-to-end integration**:
- Gen-0 validated components in isolation or in partially confounded pairings on STL-10.
- Generation-1 is the first codebase where all interfaces (`BeliefState`, `GlimpseFeaturePredictor`, `UpdateNet`, `AISv2GazePolicy`) are unified under strict type contracts and multi-group gradient isolation on ImageNet-100.
- Stating that the full system is "proven" ignores the critical experimental DAG:
  - Step 1: Pre-flight gradient isolation
  - Step 2: Single-pass ViT baseline
  - Step 3: Recurrent ViT baseline
  - Step 4: Predictor & UpdateNet integration
  - Step 5: Full loop integration
  - Step 6: Core ImageNet-100 validation
- Until Step 6 is completed and frozen, the full system performance remains an **empirical hypothesis**.

---

## 12. Quick Reference: Anti-Pattern vs RHAN-NXA Pattern

| Area | Anti-Pattern | RHAN-NXA Pattern |
|---|---|---|
| **Multi-Glimpse** | Independent crops averaged together | Stateful recurrent belief trajectory $B_0 \to B_1 \to \dots \to B_T$ |
| **Uncertainty** | Treated as general visual entropy | Dirichlet class-conditioned epistemic uncertainty ($U_t$) |
| **Structure** | Reintroducing 16-slot SBR code | $S_t = \text{None}$ core; clean 2–4 slot re-entry under tightened gates |
| **Prediction Error** | Detaching $E_t$ before state update | Fully differentiable backprop through $E_t$ to predictor |
| **AIS-v2 Status** | Claimed as proven robustness driver | Mechanism valid ($r = 0.706$); robustness contribution UNKNOWN |
| **Trajectory Loss** | Co-training $\mathcal{L}_{\text{stab}}$ from day one | Staged protocol: Phase A diagnostic; Phase B post-Step 6 |
| **State Fusion** | Raw linear addition $z + \lambda \Pi E$ | Bounded non-linear MLP mapping via `UpdateNet` |
| **Prediction Space** | Raw pixel generation ($56 \times 56 \times 3$) | Latent patch token embeddings ($16 \times 384$) |
| **Gaze Selection** | High-variance continuous RL policy | Differentiable scoring over $K \in \{4..8\}$ candidates |
| **System Maturity** | Claimed as proven end-to-end | Hypothesized architecture undergoing structured DAG validation |

---

## 13. Related Chapters and Cross-References

- [Chapter 08 — Prediction Error $E_t$](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md): Details on UpdateNet and differentiability contracts.
- [Chapter 11 — AIS-v2 Gaze Policy](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md): Mechanism validity versus system-level robustness.
- [Chapter 17 — Experimental DAG](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/17_Experimental_DAG.md): Structured execution and gating dependencies.
- [Chapter 19 — Generation-0 Evidence](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/19_Gen0_Evidence.md): Full forensic analysis of the D2/D3 SBR confounds.
- [Chapter 21 — Formal Decision Records](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/21_Decision_Records.md): Foundational rationales behind these constraints.


---

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


---

# Chapter 24 — Scope Boundaries and Outright Rejections

> *Level 2–3 reading. What is explicitly OUTSIDE the Generation-1 core, establishing rigid boundaries so engineering capacity is never squandered on mechanisms merely because they sound plausible.*

---

## 1. Why Scope Boundaries Exist

The most expensive code in machine learning research is code that should never have been written. In complex cognitive architectures, projects frequently suffer from "mechanism creep"—the tendency to incorporate attention mechanisms, memory stores, graph structures, or auxiliary losses simply because they appear in adjacent literature or sound intellectually appealing.

Generation-0 suffered directly from this phenomenon: the 16-slot Spatial Bottleneck Recurrence (SBR) was introduced without isolated baseline validation, ultimately causing a 10 percentage point clean accuracy collapse that contaminated subsequent active sensing experiments.

RHAN-NXA establishes rigid, non-negotiable **scope boundaries**. Every item documented in this chapter is either explicitly deferred pending empirical preconditions or rejected outright.

---

## 2. The `REJECTED_OUTRIGHT` Registry

In [noesis_vision/core/schema.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/schema.py), the codebase formalizes these exclusions programmatically. Any configuration attempting to instantiate a rejected mechanism triggers an immediate validation exception at startup:

```python
# noesis_vision/core/schema.py

REJECTED_OUTRIGHT: frozenset[str] = frozenset({
    "pixel_reconstruction",        # DR-002: heavy decoder, high-frequency noise, increases drift
    "edge_map_targets",            # Foreign representation space, hand-crafted extractor overhead
    "sbr_16_slot",                 # DR-010: chance slot decoding, -10 pp collapse signature
    "rag_external_retrieval",      # Chapter 22: zero empirical standing in visual robustness
    "ais_v1_relocated_gaze",       # Chapter 11: superseded by AIS-v2 design; no fallback path
    "raw_z_lambda_pi_e_addition",  # Chapter 08: dimension/manifold mismatch; UpdateNet required
})
```

---

## 3. Comprehensive Disposition Matrix

| Candidate Item | Status | Theoretical / Empirical Reason | Precondition for Re-Entry |
|---|---|---|---|
| **Representation-Level Uncertainty** | **PENDING DECISION** | No validated self-supervised training target; would require unverified machinery mid-build. | Validated self-supervised uncertainty formulation on representation manifolds. |
| **Episodic Memory (Across Images)** | **DEFERRED** | Unnecessary for static visual robustness; no evidence that cross-image memory is a limiting factor. | Empirical demonstration that cross-image persistence solves a specific robustness failure. |
| **External Retrieval / RAG** | **REJECTED OUTRIGHT** | Zero project evidence; exists only because "memory" appears in generic AI discourse. | **None.** Prohibited from entry into the visual perception engine. |
| **Temporal / Video Persistence** | **DEFERRED** | Conceptual formulation $B_t \to \text{predict} \to \text{frame}_{t+1}$ is preserved, but out of scope for static benchmarks. | Temporal video occlusion and tracking benchmark requiring sequential memory. |
| **Structural State ($S_t$ / Slots)** | **EXPERIMENTAL CANDIDATE, DEFERRED** | Gen-0 16-slot SBR collapsed; burden of proof is on structure after core validation. | Compact 2–4 slot module passing tightened Gate 1 distinguishability probes. |
| **Adaptive Halting** | **DEFERRED** | Documented history of objective conflict (Model v10 collapse); core loop must validate at fixed $T=4$. | Step 6 core baseline frozen; halting evaluated with strict on/off controls. |
| **Pixel Reconstruction Target** | **REJECTED FOR GEN-1** | Mechanistically dilutes precision, increases drift, requires heavy spatial decoder. | Native dense prediction visual substrate showing Lens-verified stability gains. |
| **Edge-Map / HPC Targets** | **REJECTED FOR GEN-1** | Hand-engineered feature extraction; foreign embedding space disrupts ViT symmetry. | Demonstrated failure of self-supervised latent targets on edge-dominant datasets. |
| **Gen-0 16-Slot SBR Code** | **REJECTED** | Retained ablation ratio 1.0157; chance slot probes (0.44–0.51); -10 pp collapse. | **None.** Implementation permanently deprecated; historical reference only. |
| **AIS-v1 Relocated-Gaze Code** | **REJECTED** | Mathematically superseded by AIS-v2 uncertainty reduction scoring. | **None.** Prohibited from porting, even as an emergency fallback path. |
| **3D / Depth Modules** | **DEFERRED** | 2D benchmark evidence does not indicate that depth representation is a primary bottleneck. | 3D adversarial attack benchmark isolating out-of-plane rotational vulnerabilities. |
| **Relational Scene Graphs** | **DEFERRED** | High parameter overhead; contingent on demonstrating appearance + structure failure. | Negative proof showing that $z_t$ and compact $S_t$ fail to capture relational pairs. |
| **$\mathcal{L}_{\text{stab}}$ Training Loss** | **EXPERIMENTAL CANDIDATE, STAGED** | Premature optimization prevents single-mechanism attribution; risks Gate 9 collapse. | Validated Step 6 ImageNet-100 baseline + calibrated Gate 9 responsiveness floor. |
| **V1 Gabor Frontend** | **EXPERIMENTAL CANDIDATE, SEQUENCED LAST** | Low-level orthogonal frequency filter; entangles robustness attribution if introduced early. | Core loop validated in Step 6; evaluated in isolated Step 9 standalone ablation arm. |

---

## 4. The "Do-Not-Implement" Rule

To maintain scientific integrity and code hygiene, the following guidelines are mandatory for all contributors, subagents, and automated pipelines:

1. **No Speculative Porting**: Never import or copy code from `rhan_core/` unless explicitly designated as `PORT VERBATIM` or `ADAPT` in the Infrastructure Port Table ([Chapter 27](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/27_Infrastructure_Port_Table.md)).
2. **No Fallback Paths for Rejected Code**: Do not retain `if use_legacy_ais: ...` or `if enable_sbr: ...` branches. Dead code and legacy fallbacks are the precise mechanisms through which confounds silently re-enter research systems.
3. **No Uncalibrated Thresholds**: Never invent numeric cutoffs for Gate 6 (correlation cutoff), Gate 7 (calibration threshold), or Gate 9 (responsiveness floor). These must be calibrated from baseline empirical distributions during Phase A diagnostic logging.
4. **No Raw Linear Addition**: Never bypass UpdateNet by computing $z_t = z_{t-1} + \lambda \Pi_t E_t$. State integration requires non-linear bounded mapping.

---

## 5. What IS in Scope (The Complete Gen-1 Core)

For complete clarity, the authorized Generation-1 core comprises exclusively:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AUTHORIZED GEN-1 CORE                           │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Substrate: CompactViT (14x14 patches, D_z = 384, 6 heads, 23.3M)   │
│ 2. Belief State: B_t = (z_t, S_t = None, U_t, E_t, A_t) None-propagation│
│ 3. Recurrence: Option C (2–3 tied within-glimpse iterations, T=4)      │
│ 4. Foveation: Differentiable 56x56 affine grid sampling from 224x224   │
│ 5. Glimpse Predictor: Shared 2.8M MLP predicting 16x384 patch tokens   │
│ 6. Predictive Coding: Latent error E_t, Precision Π_t, UpdateNet       │
│ 7. Active Sensing: AIS-v2 (K=4–8 candidates, Dirichlet entropy scoring)│
│ 8. Uncertainty: EvidentialHead (Dirichlet evidence via softplus)       │
│ 9. Stability: Diagnostic-only L_stab trajectory drift logging          │
│ 10. Optimization: 5-group optimizer with gradient isolation checks     │
└────────────────────────────────────────────────────────────────────────┘
```

If a proposed feature or module is not on this list, it is out of scope until the Master Experiment Plan authorizes its entry.

---

## 6. Related Chapters and Cross-References

- [Chapter 06 — Structure State $S_t$](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/06_Structure_State.md): In-depth examination of the SBR rejection and $S_t = \text{None}$ contract.
- [Chapter 08 — Prediction Error $E_t$](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/08_Prediction_Error.md): Why pixel reconstruction was rejected.
- [Chapter 11 — AIS-v2 Gaze Policy](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/11_AIS_v2.md): Why AIS-v1 relocated gaze code was rejected.
- [Chapter 20 — L_stab](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/20_L_stab.md): The staging rules governing belief stability.
- [Chapter 21 — Formal Decision Records](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/21_Decision_Records.md): Individual DR records for each rejected alternative.


---

# Chapter 25 — Future System Directions and Evolutionary Roadmap

> *Level 2–3 reading. Post-Generation-1 architectural extensions, disciplined re-entry protocols, and the evolutionary path toward multi-modal cognitive systems.*

---

## 1. Overview and Guiding Principles

The Generation-1 RHAN-NXA core is deliberately constrained to isolate the fundamental dynamics of recurrent predictive coding and active sensing on static images. However, the architectural design of $B_t = (z_t, S_t, U_t, E_t, A_t)$ was constructed from the outset with clean extension points.

This chapter details the **pre-registered future directions** for the architecture, the specific empirical gates required for each mechanism to enter the system, and the long-term conceptual bridge connecting RHAN-NXA to multi-modal language models (RHAN-LLM).

```
               ┌────────────────────────────────────────────────────────┐
               │              GEN-1 FROZEN VALIDATED CORE               │
               │         CompactViT + Predictive Coding + AIS-v2         │
               └───────────────────────────┬────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼────────────────────────────────┐
         ▼                                 ▼                                ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│ Structural Re-entry│             │  V1 Gabor Bank   │             │ Temporal / Video │
│ 2–4 Slot $S_t$   │             │ (Step 9 Standalone│             │ Persistence      │
│ (Gate 1 Probed)  │             │     Ablation)    │             │ (Frame-to-Frame) │
└────────┬─────────┘             └────────┬─────────┘             └────────┬─────────┘
         │                                │                                │
         └────────────────────────────────┼────────────────────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │              RHAN-LLM               │
                       │ Multi-Modal Cognitive Reasoning     │
                       │ Dynamic Belief State as Prompt Prefix│
                       └─────────────────────────────────────┘
```

---

## 2. Structural State Re-Entry Protocol ($S_t$)

The exclusion of the structure state in Gen-1 ($S_t = \text{None}$) is a temporary scoping discipline triggered by the empirical failure of the Gen-0 16-slot SBR. The protocol for re-evaluating object-centric representations is pre-registered in Part 2 Step 7:

### Re-Entry Requirements
1. **Low Slot Cardinality**: Restrict slot count to $M \in \{2, 3, 4\}$ (down from 16). Visual scenes in ImageNet-100 rarely contain more than 2–3 dominant foreground objects.
2. **Tightened Gate 1 Distinguishability**:
   - Each slot $s^{(m)}$ must train a linear readout probe on downstream semantic attributes.
   - The mutual information between slot representations must satisfy an orthogonality threshold:
     $$\frac{1}{M(M-1)} \sum_{i \neq j} \frac{|s^{(i)} \cdot s^{(j)}|}{\|s^{(i)}\| \|s^{(j)}\|} < \tau_{\text{ortho}}$$
   - Performance under slot zeroing must exhibit functional specialization: zeroing an object's slot must impair detection of that object without impairing other objects.
3. **No Clean Accuracy Penalty**: The integrated model must match or exceed the frozen Step 6 baseline ($\Delta \text{Acc}_{\text{clean}} \ge 0.0$ pp).

---

## 3. The Biologically Inspired V1 Gabor Frontend

Detailed in [Chapter 23 (V1 Frontend)](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/23_V1_Frontend.md), a fixed bank of Gabor filters represents a biologically grounded inductive bias.

### Properties and Build Order
- **Fixed and Non-Learnable**: Weights are initialized with multi-scale, multi-orientation Gabor functions and frozen. Zero added trainable parameters.
- **Sequenced Last (Step 9)**: Built only after the core loop is validated on ImageNet-100.
- **Standalone Ablation**: Evaluated in an 8-seed controlled arm. If the Gabor bank yields significant robustness gains without distorting belief drift, it will be integrated into the default vision pipeline.

```
Input Image ──► [Fixed Multi-Scale Gabor Bank] ──► CompactViT Patch Embedder ──► ...
```

---

## 4. Temporal and Video Persistence

While Gen-1 processes static images via a sequential 4-glimpse trajectory, the mathematical formulation inherently mirrors a temporal filtering process.

### The Temporal Hypothesis
In video streams, temporal continuity provides strong physical priors:
$$\text{frame}_t \longrightarrow B_t \longrightarrow \text{Predictor} \longrightarrow \widehat{\text{frame}}_{t+1}$$

Where static perception investigates spatial regions of an image, video perception tracks dynamic object transformations over time:
1. $E_t$ computes temporal prediction error between expected motion and realized optic flow.
2. $B_t$ persists across video frames, maintaining object identity through occlusions.
3. **Precondition for Initiation**: An empirical benchmark demonstrating that static frame-by-frame processing fails on temporal occlusion or trajectory continuity probes.

---

## 5. Adaptive Halting Re-Evaluation

In Generation-1, glimpse depth is locked to $T = 4$ (DR-004 and DR-006). Future generations will revisit adaptive halting under strict isolation:

- **Baseline Requirement**: The core loop must be frozen at Step 6.
- **Halting Policy**: Halting must be driven by Dirichlet epistemic uncertainty:
  $$\text{Halt if } U_t < \tau_{\text{halt}} \quad \text{or} \quad t = T_{\max}$$
- **Evaluation Criteria**: Compute savings (FLOPs reduction on simple images) must not be achieved at the expense of worst-case adversarial robustness under PGD-20 and AutoAttack evaluations.

---

## 6. Multi-Modal Integration: RHAN-LLM

The ultimate trajectory of the RHAN architecture is its synthesis with Large Language Models. Current Vision-Language Models (VLMs) suffer from severe visual hallucinations and adversarial susceptibility because they treat vision as a passive, single-pass projection of dense spatial grid tokens into text embedding space.

RHAN-NXA provides an active cognitive visual front-end for language reasoning:

```
┌────────────────────────────────────────────────────────────────────────┐
│                              RHAN-LLM                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Visual Input ──► [RHAN-NXA Active Engine]                             │
│                          │                                             │
│                          ▼                                             │
│               Belief Trajectory (B_1, ..., B_T)                        │
│                          │                                             │
│                          ▼                                             │
│  Cognitive Prefix:  [z_T, U_T, Fixation History A_T]                   │
│                          │                                             │
│                          ▼                                             │
│  User Query ────► [Cross-Attention Adapter] ──► Large Language Model   │
│                          │                                             │
│                          ▼                                             │
│               Grounded Multi-Modal Reasoning                           │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Conceptual Innovations
1. **Uncertainty-Grounded Generation**: The LLM conditions directly on $U_t$. When visual uncertainty is high, the language model can abstain, request clarification, or prompt the vision engine to take additional targeted glimpses.
2. **Active Visual Querying**: Instead of processing static tokens, the language model can generate linguistic queries that guide AIS-v2 gaze fixations toward regions relevant to complex questions.
3. **Adversarial Resilience**: The intrinsic robustness of the recurrent predictive coding loop shields the downstream language model from visual jailbreaks and adversarial image injections.

---

## 7. Evolutionary Timeline and Maturity Gates

| Stage | Milestones | Primary Deliverable | Status |
|---|---|---|---|
| **Phase 1 (Gen-1)** | Core loop integration, unconfounded ImageNet-100 baseline, DAG validation | RHAN-NXA Core Handbook & Checkpoints | **ACTIVE** |
| **Phase 2 (Gen-1+)** | Step 8 $\mathcal{L}_{\text{stab}}$ promotion, Step 9 V1 Frontend ablation | Trajectory Stability & Frequency Robustness | **STAGED** |
| **Phase 3 (Gen-2)** | Compact 2–4 slot $S_t$ re-entry, Video occlusion benchmark | Object-Centric Temporal Perception | **DEFERRED** |
| **Phase 4 (Gen-3)** | RHAN-LLM adapter, active question-answering gaze policy | Multi-Modal Cognitive Architecture | **CONCEPTUAL** |

---

## 8. Related Chapters and Cross-References

- [Chapter 06 — Structure State $S_t$](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/06_Structure_State.md): Context for structural re-entry.
- [Chapter 17 — Experimental DAG](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/17_Experimental_DAG.md): Step 7, Step 8, and Step 9 dependency order.
- [Chapter 20 — L_stab](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/20_L_stab.md): The staged promotion protocol.
- [Chapter 24 — Scope Boundaries](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/24_Scope_Boundaries.md): Rigid definitions of what is currently out of scope.
- Diagram Reference: [future_rhan_llm.svg](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/figures/future_rhan_llm.svg).


---

# Chapter 26 — Reproducibility Pipeline and Handbook Build System

> *Level 1–2 reading. Complete operational guide for executing the RHAN-NXA reproducibility suite, verifying experimental claims, and compiling this technical handbook into a publication-grade PDF.*

---

## 1. Overview and Core Philosophy

In scientific machine learning, an architecture is only as credible as its reproducibility. The RHAN-NXA research program enforces strict protocols to ensure that every table, metric, parameter count, and documentation asset can be regenerated deterministically from source code and pre-registered configurations.

This chapter provides the complete operational handbook for:
1. Environment configuration and dependency verification.
2. Running the pre-flight gradient isolation and smoke test suite.
3. Executing experimental DAG steps with seed parity.
4. Compiling the complete 27-chapter Technical Handbook into a unified, publication-grade document and PDF.

---

## 2. Environment and Dependency Specification

RHAN-NXA executes on standard Linux environments with PyTorch $\ge 2.0$ and CUDA acceleration.

### Core Dependencies
```bash
# Core Machine Learning Frameworks
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio

# Robustness & Adversarial Evaluation
pip install autoattack robustbench

# Documentation & PDF Compilation Toolchain
sudo apt-get update && sudo apt-get install -y pandoc texlive-xetex librsvg2-bin
# Alternatively: weasyprint or python-markdown
pip install weasyprint markdown PyYAML
```

---

## 3. Pre-Flight Verification Suite

Before launching any training or evaluation run, the system requires running the pre-flight verification script to confirm gradient isolation and interface integrity:

```bash
# Execute pre-flight gradient isolation checks
python3 -m unittest discover -s tests -p "test_*.py"
```

The pre-flight test suite verifies:
1. **$B_t$ Contract**: Confirms that all belief operations execute without exception when $S_t = \text{None}$.
2. **Gradient Flow**: Asserts that during predictor training, gradients do not leak into the backbone, and that during evidential head updates, the predictor remains unperturbed ($|\nabla_W| = 0$).
3. **Numerical Clamps**: Verifies that evidential evidence parameters obey $[\alpha_{\min}, \alpha_{\max}] = [10^{-6}, 10^4]$ and that $\Pi_t \ge 10^{-4}$.
4. **UpdateNet Bounding**: Asserts that $|\Delta z_t| \le 0.1$ across random batch tensors.

---

## 4. Checkpoint Durability and Resumption Parity

All training experiments must support atomic resumption without state loss:

```bash
# Checkpoint structure verification
python3 -c "
from noesis_vision.core.checkpoint import CheckpointManager
print('Checkpoint Manager Loaded Successfully')
"
```

### Resume Protocol
- The checkpoint manager persists model weights, 5-group optimizer states, learning rate schedulers, scalar history metrics, and RNG seeds (PyTorch, NumPy, Python standard library).
- When training resumes from epoch $k$, the rolling validation score at epoch $k$ must match the recorded best score to within $10^{-6}$ numerical tolerance.

---

## 5. Handbook Build Pipeline (`build_rhan_nxa_handbook.py`)

This entire Technical Handbook is generated deterministically from the Markdown source files in `docs/rhan_nxa/book/` and `docs/rhan_nxa/appendices/`.

The build automation script is located at:
`scripts/build_rhan_nxa_handbook.py`

### Build Command
To compile the handbook into a unified Markdown document and publication PDF:

```bash
python3 scripts/build_rhan_nxa_handbook.py
```

### Build Architecture
```
docs/rhan_nxa/book/
  00_Executive_Overview.md
  01_Why_RHAN_NXA.md
  ...
  26_Reproducibility.md
        │
        ├──► scripts/build_rhan_nxa_handbook.py ──► Combined Unified Markdown
        │                                           (RHAN_NXA_Handbook_Unified.md)
docs/rhan_nxa/appendices/                                   │
  A_Tensor_Reference.md                                     │ Pandoc / Weasyprint
  B_Interface_ABCs.md                                       ▼
  C_Port_Table.md                           RHAN_NXA_Technical_Handbook.pdf
```

The script performs the following tasks:
1. **Sequential Assembly**: Reads all 27 chapters in strict numerical order (`00_` through `26_`).
2. **Appendix Integration**: Appends Appendices A, B, and C with standardized hierarchical headers.
3. **Link Normalization**: Resolves local cross-file markdown links into internal document anchors for seamless reading.
4. **Diagram Embedding**: Resolves SVG and raster figure paths to absolute references in `docs/rhan_nxa/figures/`.
5. **PDF Compilation**: Invokes `pandoc` or `weasyprint` with academic typography, syntax highlighting, and an automated table of contents.

---

## 6. Reproducibility Checklist for New Experiments

Every new experimental finding added to the RHAN-NXA corpus must supply:

- [ ] **Exact Configuration File**: A YAML specification stored in `configs/`.
- [ ] **16-Seed Array**: Results evaluated across seeds 0 through 15 with mean and sample standard deviation reported.
- [ ] **Pre-Flight Log**: Zero-gradient leakage confirmed for all frozen parameter groups.
- [ ] **Confound Audit**: Verification that `--enable-sbr` or other rejected flags were not passed accidentally.
- [ ] **Matched Controls**: A parameter-matched and compute-matched control baseline evaluated on the identical hardware and batch size.
- [ ] **Checkpoint Artifact**: Publicly accessible model weights and optimizer states saved with SHA-256 checksums.

---

## 7. Related Chapters and Cross-References

- [Chapter 14 — Multi-Group Gradient Flow](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/14_Gradient_Flow.md): Pre-flight isolation details.
- [Chapter 16 — Training System](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/16_Training_System.md): Checkpoint durability and resume protocols.
- [Chapter 17 — Experimental DAG](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/17_Experimental_DAG.md): Step-by-step DAG execution sequence.
- [Chapter 18 — Evaluation Protocols](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/docs/rhan_nxa/book/18_Evaluation.md): Standardized AutoAttack and PGD evaluation scripts.


---

# Appendix A — Comprehensive Tensor Shape and Interface Reference

> *Authoritative tabular reference of all tensor shapes, dimensions, dtypes, and value domains across the RHAN-NXA architecture.*

---

## 1. Global Dimensions and Constants

| Symbol | Name | Canonical Value | Status | Defined In |
|---|---|---|---|---|
| $B$ | Batch size | Dynamic (e.g. 64, 128) | Runtime | DataLoader |
| $H, W$ | Full image spatial resolution | $224, 224$ | Locked | ImageNet-100 standard |
| $C$ | Color channels | $3$ | Locked | RGB format |
| $D_z$ | Latent embedding dimension | $384$ | Locked | `backbone.py` |
| $P$ | ViT patch spatial size | $14 \times 14$ | Locked | `backbone.py` |
| $H_g, W_g$ | Foveal glimpse resolution | $56, 56$ | Locked | `foveation.py` |
| $N_g$ | Patches per glimpse | $16$ ($4 \times 4$) | Locked | $(56/14) \times (56/14)$ |
| $T$ | Total glimpse steps per image | $4$ | Locked | `MASTER_PLAN.md` |
| $K_{\text{ais}}$ | Candidate gaze fixations | $4$ to $8$ (default 4) | Locked | `ais_v2_policy.py` |
| $C_{\text{cls}}$ | Classification categories | $100$ | Locked | ImageNet-100 |
| $\text{iter}_{\text{in}}$ | Within-glimpse token iterations | $2$ or $3$ (default 2) | Locked | `recurrent_block.py` |

---

## 2. Component Tensor Shape Table

| Component | Tensor Name | Shape | Dtype | Value Range / Constraints |
|---|---|---|---|---|
| **Input Image** | $x$ | `(B, 3, 224, 224)` | `float32` | Normalized ImageNet range |
| **Gaze Action** | $a_t$ | `(B, 2)` | `float32` | $(y, x) \in [-1, 1]^2$ |
| **Foveal Glimpse** | $g_t$ | `(B, 3, 56, 56)` | `float32` | Normalized ImageNet range |
| **Patch Embeddings** | $E_{\text{patches}}$ | `(B, 16, 384)` | `float32` | Unbounded real |
| **Refined Tokens** | $H_{\text{refined}}$ | `(B, 16, 384)` | `float32` | Post LayerNorm |
| **Pooled State** | $z_t^{\text{raw}}$ | `(B, 384)` | `float32` | Mean pooled over patch dimension |
| **Belief State** | $z_t$ | `(B, 384)` | `float32` | Global perceptual vector |
| **Belief State** | $S_t$ | `None` | N/A | Gen-1 Core default ($S_t = \text{None}$) |
| **Belief State** | $U_t$ | `(B,)` | `float32` | $U_t \in (0, 1]$, Clamped $[10^{-6}, 10^4]$ |
| **Dirichlet Evidence** | $e$ | `(B, 100)` | `float32` | $e_k \ge 0$ via Softplus |
| **Dirichlet Alpha** | $\alpha$ | `(B, 100)` | `float32` | $\alpha_k = e_k + 1 \ge 1.0$ |
| **Predicted Glimpse** | $\hat{g}_t$ | `(B, 16, 384)` | `float32` | Output of shared predictor |
| **Prediction Error** | $E_t$ | `(B, 16, 384)` | `float32` | $E_t = \Vert g_t^{\text{obs}} - \hat{g}_t^{\text{pred}} \Vert_2$; $E_0 := 0$ |
| **Precision** | $\Pi_t$ | `(B, 1)` or `(B, 384)` | `float32` | $\Pi_t \ge 10^{-4}$ (clamped floor) |
| **UpdateNet Delta** | $\Delta z_t$ | `(B, 384)` | `float32` | $\lvert \Delta z_t \rvert \le 0.1$ via $\tanh$ scaling |
| **AIS Candidates** | $a_{\text{cand}}$ | `(B, K, 2)` | `float32` | Candidate coordinates $\in [-1, 1]^2$ |
| **Candidate Scores** | $s_{\text{cand}}$ | `(B, K)` | `float32` | Information gain / entropy reduction |
| **Gaze Probabilities** | $p_{\text{gaze}}$ | `(B, K)` | `float32` | $\sum_{k=1}^K p_k = 1.0$ (Softmax) |
| **Trajectory Drift** | `drift` | `(B,)` | `float32` | Non-negative real |

---

## 3. Boundary Conditions and Interface Invariants

1. **$t = 0$ Initialization**:
   - $z_0 \leftarrow \mathbf{0} \in \mathbb{R}^{B \times 384}$ (or learned zero prior).
   - $S_0 \leftarrow \text{None}$.
   - $U_0 \leftarrow 1.0$ (maximal uncertainty prior).
   - $E_0 \leftarrow \mathbf{0}$ (identity prediction error).
   - $a_1 \leftarrow (0.0, 0.0)$ (center foveal glimpse).
2. **UpdateNet Invariant**:
   $$\|\Delta z_t\|_{\infty} \le 0.1$$
3. **Precision Weighting Invariant**:
   $$\Pi_t \ge 10^{-4}$$
4. **Dirichlet Evidence Invariant**:
   $$10^{-6} \le e_k \le 10^4 \quad \implies \quad 1 + 10^{-6} \le \alpha_k \le 10^4 + 1$$


---

# Appendix B — Architectural Interface Contracts and ABCs

> *Authoritative Python Abstract Base Class (ABC) definitions and typed interface signatures defining the RHAN-NXA contract boundaries.*

---

## 1. Overview

To guarantee modularity, prevent monolithic coupling, and enable independent subagent ownership, RHAN-NXA relies on strict Python abstract base classes. Any implementation component must satisfy these contracts.

All interfaces are formally defined in [noesis_vision/beliefs/interfaces.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/interfaces.py) and [noesis_vision/predictive_coding/interfaces.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/predictive_coding/interfaces.py).

---

## 2. The `BeliefState` Interface

```python
# noesis_vision/beliefs/interfaces.py

from abc import ABC, abstractmethod
from typing import Mapping, Optional
import torch

class BeliefState(ABC):
    """Abstract interface defining the tripartite perceptual belief state B_t."""

    @property
    @abstractmethod
    def z(self) -> torch.Tensor:
        """Global perceptual state vector, shape (B, D_z)."""
        pass

    @property
    @abstractmethod
    def s(self) -> Optional[torch.Tensor]:
        """Optional structural state tensor. Returns None in Gen-1 core."""
        pass

    @property
    @abstractmethod
    def u(self) -> torch.Tensor:
        """Dirichlet epistemic uncertainty scalar per sample, shape (B,)."""
        pass

    @abstractmethod
    def drift_to(self, other: "BeliefState",
                 weights: Optional[Mapping[str, float]] = None,
                 distance: str = "l2") -> torch.Tensor:
        """Computes sample-wise trajectory drift, returning a (B,) tensor.
        
        Must remain differentiable w.r.t. self.z (and other.z if attached).
        """
        pass

    @abstractmethod
    def detached_copy(self) -> "BeliefState":
        """Returns a detached copy of B_t with autograd severed, for logging."""
        pass

    @abstractmethod
    def updated(self, delta_z: torch.Tensor,
                delta_s: Optional[torch.Tensor] = None,
                new_u: Optional[torch.Tensor] = None) -> "BeliefState":
        """Constructs an updated BeliefState immutably."""
        pass
```

---

## 3. The `GlimpseFeaturePredictor` Interface

```python
# noesis_vision/predictive_coding/interfaces.py

from abc import ABC, abstractmethod
import torch

class GlimpseFeaturePredictor(ABC):
    """Shared predictor consumed jointly by prediction error and gaze planning."""

    @abstractmethod
    def forward(self, z_prev: torch.Tensor,
                a_next: torch.Tensor) -> torch.Tensor:
        """Predicts latent patch features at prospective gaze location.
        
        Args:
            z_prev: Prior global belief state, shape (B, D_z).
            a_next: Prospective fixation coordinate, shape (B, 2) in [-1, 1]^2.
            
        Returns:
            Predicted patch token features, shape (B, 16, D_z).
        """
        pass
```

---

## 4. The `UpdateNet` Interface

```python
# noesis_vision/predictive_coding/interfaces.py

from abc import ABC, abstractmethod
import torch

class UpdateNet(ABC):
    """Fuses prior belief with precision-weighted error into a bounded update."""

    DELTA_BOUND: float = 0.1

    @abstractmethod
    def forward(self, z_prev: torch.Tensor,
                error_term: torch.Tensor,
                precision: torch.Tensor) -> torch.Tensor:
        """Computes bounded state delta.
        
        Args:
            z_prev: Prior belief state, shape (B, D_z).
            error_term: Realized prediction error tensor, shape (B, D_z).
            precision: Dynamic precision weighting scalar, shape (B, 1).
            
        Returns:
            delta_z: Bounded update vector, shape (B, D_z), obeying
                     ||delta_z||_inf <= DELTA_BOUND.
        """
        pass
```

---

## 5. The `AISv2GazePolicy` Interface

```python
# noesis_vision/gaze/ais_v2_policy.py

from abc import ABC, abstractmethod
from typing import Tuple
import torch

class AISv2GazePolicy(ABC):
    """Active Information Sampling gaze policy."""

    @abstractmethod
    def select_action(self, belief: BeliefState,
                      history: torch.Tensor,
                      temperature: float = 1.0,
                      deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Proposes and selects the next fixation coordinates.
        
        Args:
            belief: Current BeliefState B_t.
            history: Past fixation history A_t, shape (B, t, 2).
            temperature: Softmax sampling temperature.
            deterministic: If True, uses argmax selection.
            
        Returns:
            Tuple of:
              - a_next: Selected fixation coordinate, shape (B, 2).
              - candidate_scores: Evaluated scores for all K candidates, shape (B, K).
        """
        pass
```


---

# Appendix C — Infrastructure Port Table and Lineage Resolution

> *Authoritative disposition table for all inherited and newly developed infrastructure components in RHAN-NXA Generation-1.*

---

## 1. Overview and Porting Principles

RHAN-NXA Generation-1 builds upon algorithmic insights from the prior STL-10 research exploration (Generation-0). However, to prevent technical debt and legacy confounds from migrating into the new system, every inherited module was subjected to a formal disposition review.

No module is ported by momentum; no component is rewritten by subjective taste.

### Porting Dispositions Defined
1. **PORT VERBATIM**: Copied as-is without semantic or structural alteration. The module is empirically validated and zero justification exists to modify it.
2. **ADAPT**: Semantics and mathematical logic preserved, but interface bindings are updated to match ImageNet-100 dimensions, new tensor namespaces, or normalization conventions.
3. **REJECT**: Prohibited from being ported. Must not exist even as a fallback or dead branch.
4. **NEW**: First implementation in this repository of an established scientific formulation. Carries zero inherited validation and requires full unit test coverage.

---

## 2. Master Infrastructure Port Table

| Component / Module | Source in Gen-0 | Disposition | Rationale and Technical Scope |
|---|---|---|---|
| **Multi-Group Optimizer + Registry** | `core/optimizer.py` | **PORT VERBATIM** | Validated across 3+ Gen-0 components; enables strictly isolated parameter groups and learning rates. |
| **Gradient Isolation Pre-Flight (`\|\nabla W\|`)** | `tests/preflight.py` | **PORT VERBATIM** | Standing rule: zero-leakage must be verified analytically before any smoke run is trusted. |
| **Checkpoint Resume + Parity Assertion** | `core/checkpoint.py` | **PORT VERBATIM** | Learned the hard way: bit-for-bit parity check between best and rolling models is non-negotiable. |
| **Summary Table vs CSV Parity Assertion** | `core/metrics.py` | **PORT VERBATIM** | Enforces that markdown tables in reports match raw CSV metric logs identically. |
| **Comparator Model Registry** | `benchmarks/registry.py` | **ADAPT** | Rebinds checkpoint paths and evaluation hooks for ImageNet-100; preserves donor-row discipline. |
| **Adversarial Evaluation (PGD / AutoAttack)** | `eval/attacks.py` | **ADAPT** | Preserves exact norm-bounded attack routines; updates image normalization and resolution ($224 \times 224$). |
| **Evidential Head** | None (Literature) | **NEW** | First implementation in this codebase of Sensoy et al. (2018); not ported from Gen-0. |
| **AIS-v1 Relocated-Gaze Module** | `gaze/ais_v1.py` | **REJECT** | Mathematically and empirically superseded by AIS-v2; banned from entering the codebase. |
| **16-Slot Spatial Bottleneck Recurrence** | `models/sbr.py` | **REJECT** | Chance-level slot decoding ($0.44–0.51$); caused reproducible $-10$ pp clean accuracy collapse. |

---

## 3. Discrepancy Resolution: The `EvidentialHead` Status

In early development notes, informal text occasionally referred to the `EvidentialHead` as "ported verbatim." However, a rigorous code audit confirmed that **no EvidentialHead class existed in the Generation-0 codebase**.

In the authoritative Master Implementation Plan (Part 5), the disposition was formally corrected and locked to **NEW**:
- It is the first implementation in this project of the Dirichlet evidential deep learning formulation (Sensoy et al., NeurIPS 2018).
- It inherits no empirical validation from Gen-0.
- It was constructed from scratch by Agent D in [noesis_vision/uncertainty/evidential_head.py](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/uncertainty/evidential_head.py) and is covered by dedicated unit test suites verifying non-negative evidence softplus mapping and Dirichlet entropy calculations.

---

## 4. The No-Parallel-Port Rule

The Generation-0 directory `rhan_core/` is preserved strictly as a **read-only historical archive**:
1. No Gen-1 module may import from `rhan_core`.
2. All components must import strictly from `noesis_vision.*`.
3. If an unported utility is required, it must pass an explicit RFC review, be assigned an ADAPT or PORT VERBATIM disposition, and be migrated into `noesis_vision/` under contract tests.

This rule completely eliminates silent dependency contamination and ensures complete reproducibility of the Generation-1 architecture.


---

