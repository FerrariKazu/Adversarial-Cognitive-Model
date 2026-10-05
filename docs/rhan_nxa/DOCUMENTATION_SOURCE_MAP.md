# RHAN-NXA Documentation Source Map

*Internal architectural and provenance audit mapping concepts to verified repository implementations.*

| Concept | Source File | Symbol / Class / Function | Status | Handbook Chapter |
| :--- | :--- | :--- | :--- | :--- |
| **BeliefState Abstraction** | `noesis_vision/beliefs/interfaces.py`<br>`noesis_vision/beliefs/vector_belief.py` | `BeliefState`, `VectorBeliefState` | **LOCKED** (Core $S_t=\text{None}$) | Ch. 04 (`04_Belief_State.md`) |
| **Global Perceptual State $z_t$** | `noesis_vision/models/backbone.py` | `CompactViT`, `D_Z = 384`, `encode_glimpse` | **LOCKED** ($D_z=384$, ~23.3M params) | Ch. 05 (`05_z_State.md`) |
| **Structure State $S_t$ (Slots)** | `noesis_vision/core/schema.py`<br>`scripts/sbr0_gate.py` | `enable_sbr=False`, `sbr_num_slots=None` | **REJECTED** (Gen-0 16-slot);<br>**DEFERRED** (2–4 slots post-Step 6) | Ch. 06 (`06_Structure_State.md`) |
| **Dirichlet Uncertainty $U_t$** | `noesis_vision/uncertainty/evidential_head.py` | `DirichletParams`, `EvidentialHead` | **LOCKED** (Dirichlet formulation;<br>Provenance: unverified Gen-0 port) | Ch. 07 (`07_Uncertainty.md`) |
| **Prediction Error $E_t$** | `noesis_vision/predictive_coding/glimpse_predictor.py`<br>`noesis_vision/predictive_coding/update_net.py` | `ConcreteGlimpseFeaturePredictor`, `ConcreteUpdateNet` | **EXPERIMENTAL CANDIDATE** (Latent token error;<br>$E_0 := 0$ at $t=0$) | Ch. 08 (`08_Prediction_Error.md`) |
| **Sensory Precision $\Pi_t$** | `noesis_vision/predictive_coding/precision.py` | `PrecisionFunction` | **EXPERIMENTAL CANDIDATE** (Learned positive mapping $g(U_t)$) | Ch. 08 / Ch. 12 |
| **Computational Recurrence** | `noesis_vision/models/recurrent_block.py` | `TiedRecurrence`, `TransformerBlock` | **LOCKED** (Tied weights, 2–3 iters) | Ch. 09 (`09_Recurrence.md`) |
| **Perceptual Recurrence** | `noesis_vision/core/schema.py`<br>`noesis_vision/models/backbone.py` | `num_glimpses = 4` | **LOCKED** (Fixed $T=4$; Adaptive halting DEFERRED) | Ch. 09 (`09_Recurrence.md`) |
| **Differentiable Foveation** | `noesis_vision/models/foveation.py` | `foveal_sample`, `DEFAULT_FOVEA_SIZE = 56` | **LOCKED** (Normalized $[-1, +1]$ coords) | Ch. 10 (`10_Foveation.md`) |
| **AIS-v2 Gaze Policy** | `noesis_vision/gaze/ais_v2_policy.py`<br>`noesis_vision/gaze/candidate_sampler.py` | `AISv2GazePolicy`, `CandidateScores` | **REQUIRED** design ($r \approx 0.706$);<br>**PENDING DECISION** (16-seed effect) | Ch. 11 (`11_AIS_v2.md`) |
| **Gaze History State $A_t$** | `noesis_vision/gaze/gaze_state.py` | `GazeState` | **LOCKED** (Detached coordinate log) | Ch. 04 / Ch. 11 |
| **Perceptual Loop Dynamics** | `noesis_vision/predictive_coding/update_net.py`<br>`noesis_vision/gaze/ais_v2_policy.py` | `belief_update`, `select_gaze` | **EXPERIMENTAL CANDIDATE** | Ch. 12 (`12_Complete_Perceptual_Loop.md`) |
| **Substrate & Model Architecture** | `noesis_vision/models/backbone.py` | `CompactViT` | **LOCKED** (DINOv2-small-shaped trunk) | Ch. 13 (`13_Architecture.md`) |
| **Multi-Group Gradient Flow** | `noesis_vision/core/multi_group_optimizer.py`<br>`tests/test_preflight_dw.py` | `MultiGroupOptimizerRegistry`, `measure_group_dw` | **LOCKED** (Isolated parameter groups) | Ch. 14 (`14_Gradient_Flow.md`) |
| **Tensor Shapes & Interfaces** | `noesis_vision/core/schema.py`<br>`noesis_vision/predictive_coding/interfaces.py` | `RHANNXAConfig`, `GlimpseFeaturePredictor` | **LOCKED** (Agent 0 contract) | Ch. 15 / App. A |
| **Checkpoint & State Durability** | `noesis_vision/core/checkpoint.py` | `save_checkpoint_atomic`, `verify_parity` | **LOCKED** (Atomic sync, best/rolling parity) | Ch. 16 (`16_Training_System.md`) |
| **Experiment DAG & Isolation** | `noesis_vision/core/dependency_graph.md`<br>`noesis_vision/RHAN_NXA/MASTER_PLAN.md` | Steps 1–11, Step 6 Reference | **REQUIRED** (Isolated comparison protocol) | Ch. 17 (`17_Experimental_DAG.md`) |
| **Evaluation Protocols** | `evaluation/clean_and_robust.py`<br>`evaluation/compactness_report.py` | `run_eval_suite`, `measure_compactness` | **REQUIRED** (Matched norm-space, AutoAttack) | Ch. 18 (`18_Evaluation.md`) |
| **Gen-0 Evidence & Confounds** | `report/lens_e1_analysis/E1_FULL_AUDIT_REPORT.md`<br>`MASTER_PLAN.md` Part 0 | `_nx_trainer` SBR confound | **REQUIRED** (Confound documentation;<br>AIS-v2 & HPC-Belief effects UNKNOWN) | Ch. 19 (`19_Gen0_Evidence.md`) |
| **Belief Stability Diagnostic $L_{\text{stab}}$** | `noesis_vision/beliefs/drift.py` | `drift_to`, Gate 9 | **EXPERIMENTAL CANDIDATE** (Diagnostic-only; Responsiveness floor) | Ch. 20 / Ch. 21 |
| **Formal Decision Records** | `noesis_vision/RHAN_NXA/docs/19_Decision_Records.md` | DR-001 through DR-010 | **LOCKED** (Historical decision logs) | Ch. 21 (`21_Decision_Records.md`) |
| **Common Misunderstandings** | `noesis_vision/RHAN_NXA/docs/17_Common_Misunderstandings.md` | FAQs | **REQUIRED** (Disambiguation rules) | Ch. 22 (`22_Common_Misunderstandings.md`) |
| **Technical Glossary** | `noesis_vision/RHAN_NXA/docs/18_Glossary.md` | Terminology definitions | **REQUIRED** (Exact vocabulary) | Ch. 23 (`23_Glossary.md`) |
| **Scope Boundaries** | `noesis_vision/core/schema.py` | `REJECTED_OUTRIGHT` | **LOCKED** (Non-scope items) | Ch. 24 (`24_Scope_Boundaries.md`) |
| **Future System Directions** | Master Context Specification §22 | Future RHAN + LLM interface | **DEFERRED** (Future research only) | Ch. 25 (`25_Future_Directions.md`) |
| **Reproducibility Pipeline** | `scripts/build_rhan_nxa_handbook.py` | Handbook build script | **REQUIRED** (Automated PDF build & QA) | Ch. 26 (`26_Reproducibility.md`) |
