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
