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
