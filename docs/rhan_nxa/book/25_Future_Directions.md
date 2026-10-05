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
