# RHAN-NXA — GEN-2 + GEN-3 MASTER ARCHITECTURE PLAN

**Status:** proposal document — design research at the level of *architectural trajectory*, not implementation. No codebase changes, no commits were made at the time of authoring (this file is the new committed record).

**What this documents**
- PART I: where Gen-1 genuinely stands (what is locked, what is pending, what is broken).
- PART II: every Kimi finding translated into a Gen-2 architecture.
- PART III: a complete Gen-2 architecture specification (input → classifier/readout), with the information flow traced and explicit Gen-2/Core / Gen-2/Experimental / Gen-3/Future boundary.
- PART IV: a complete Gen-2 training system.
- PART V: a full Gen-2 ablation / control matrix (including higher-order interactions).
- PART VI: a complete Gen-2 evaluation protocol.
- PART VII: a complete Gen-3 architecture (world state, object/scene representation, persistent memory, counterfactual perception, reasoning).
- PART VIII: a complete Gen-3 training strategy.
- PART IX: Gen-3 experiment matrix.
- PART X: failure-mode catalogue (symptom / cause / diagnostic / prevention / ablation / recovery) for every major subsystem and the 22 named failure classes.
- PART XI: compute budget and engineering requirements (RTX 4060 local / external / HPC).
- PART XII: how the architecture maps onto the cleaned repository (no new implementation, but explicit install points).
- PART XIII: a chronological master roadmap from the Gen-1 frozen baseline through Gen-3 validation to final scientific evaluation.
- Experiments are pre-registered with IDs (G2-K1-gist … G3-K27...) with config / seed / dataset fingerprint / code revision / architecture revision / recipe / evaluation protocol / metrics / checkpoint / manifest / result summary.

**The core scientific question this program serves (unchanged)**
> Can an artificial visual system develop human-like perceptual behavior by actively sampling visual information, maintaining recurrent belief states, predicting what it will observe next, and revising its internal beliefs from prediction error?
The goal is **not** "make a classifier more accurate." It is an **active recurrent perceptual system** whose behavior can be compared with human perception, measured along active perception, recurrent belief formation, prediction, uncertainty, information acquisition, adaptive attention, perceptual error correction, human-like visual behavior, robustness, and generalization.

---

## PART I — CURRENT GEN-1 DIAGNOSIS

### 1. What is genuinely working (Gen-1, locked)

| Component | Status | Evidence |
|---|---|---|
| Six-phase training ladder (`backbone_only` → `recurrence_only` → `belief_no_f` → `belief_with_f` → `ais_v2_swap` → `gen1_core`) | LOCKED, in use | `training/stage_state_machine.py`, `tests/test_stage_state_machine.py`, `24_Training_Phase_DAG.md` |
| Latent next-glimpse prediction (`E_t` in token feature space, predictors unified between belief-update and AIS-v2) | LOCKED | `08_Prediction_Error.md` — one predictor, two consumers, mathematically valid by construction |
| AIS-v2 gaze (K = 4–8, soft training / hard inference, Dirichlet-entropy scoring, center-bias as failure condition) | LOCKED as mechanism | `10_AIS_v2.md` — candidate-preference correlation r = 0.706 (512 samples, `ais_v2_smoke_gate_v1`), REQUIRED evidence |
| Belief state five-part contract `B_t = (z_t, S_t, U_t, E_t, A_t)` with `S_t=None` None-propagation | LOCKED | `04_Belief_State.md`, `05_z_State.md`, `14_Tensor_Shape_Reference.md` |
| Gradient health harness | LOCKED, in use | `tests/test_stage2_mechanism_seam.py` (41-seam validation), `training/adv_curriculum_freeze.py`, `training/measure_training_health.py`, `training/stage2_pipeline.py` — pre/post-clip gradients, tensor counts, TRADES vs CE, learned vs fixed precision/gaze, T=1, fixed-gaze, matched-FLOPs controls |
| Phase-specific gradient-reach pre-flight (`|dW|` check, optimizer-group isolation) | LOCKED | `stage2_pipeline.py` step-4 / step-5 enforcement |
| Diagnostic tooling | LOCKED | `tests/test_gen0_recipe_freeze.py`, `tests/test_stage2_dag_artifacts.py`, `report/` generation logs |
| No `rhan_core` / Gen-0 contamination | LOCKED | 20 test files fail at collection (out-of-scope, documented) |

### 2. What is weak

- `U_t` measures **classification readout** uncertainty, not belief-content uncertainty. `07_Uncertainty.md` states this explicitly. A representation-level uncertainty has no training target in Gen-1.
- AIS-v2 policy capacity is dominated by a single `logit_scale`; candidate sampler is heuristic/fixed. `10_AIS_v2.md` admits "policy learning capacity is currently extremely limited."
- Precision is `Π_t = 1 - U_t` — epistemic uncertainty conflated with sensory precision. `07_Uncertainty.md` / glossary flag this as a Gen-1 design matter but Gen-1 carries it forward.
- Mean pooling of `E_t` risks destroying spatial surprise (pooled into the belief update). `08_Prediction_Error.md` leaves this as latent.
- Belief revision is bounded (`Δz ≤ 0.1` style, ~4 glimpses); whether this artificially restricts belief updating is **unvalidated**. Gen-2 must test this.
- There is no **global/peripheral scene representation**; only 4 local glimpses. `08_Prediction_Error.md` + `11_Complete_Perceptual_Loop.md` imply this.
- L_stab exists as a diagnostic only — explicit staged protocol (diagnostic-only → promoted to objective only after core validated). Future experiment.
- Evaluation is PGD-100 / PGD-50 norm-space; **AutoAttack + EOT for stochastic gaze is not in the protocol** (`13_Gradient_Flow.md`, MASTER_PLAN Part 2 step 11).
- No human psychophysics alignment pipeline (Brain-Score, crowding, peripheral degradation, gaze alignment) — `28_Gates_and_Compute_Accounting.md` lists this as Tier-2, time-gated.

### 3. What is unvalidated

- AIS-v2 isolated benefit at 16-seed scale (confounded by legacy SBR D2/D3; r = 0.706 is the clean signal).
- Whether `Δz ≤ 0.1` / T=4 bounds are the cause of any perceived weakness, or are merely conservative but fine.
- Whether adversarial fine-tuning damages the visual backbone (pre-registered Π_t ablation + per-group clip diagnostic on `training/adv_curriculum_freeze.py` will start to answer this).
- Whether the frozen Gen-0 recipe hash is stable end-to-end (test_gen0_recipe_freeze.py passes; not yet a training-time stress test).
- L_stab's marginal value (still diagnostic-only).
- S_t with 2–4 slots (structural representation: EXPERIMENTAL CANDIDATE, DEFERRED; re-entry rules pre-registered).
- Research value of low-LR / frozen trunk / EMA target encoder on this substrate (not yet tested in the current tree).

### 4. Structurally limiting (things that force the architectural change)

The refactor froze the repository, but the architecture itself still carries these four limits, which the Kimi review flags and Gen-2 is designed to solve:

1. **The t=0 full image is never seen.** The model observes 4 local glimpses; there is no global gist, so the policy has no scene-level prior and cannot decide where to look without a coarse first pass. → **gist / peripheral / multi-scale input**.
2. **The learning signal is thin.** Adversarial TRADES fine-tuning may damage useful features, and the current objective family (`L_cls`, `L_adv`, `L_pred`, `Π_t`) is under-specified as a family. → **backbone adaptation controls + a candidate loss family**.
3. **Policy capacity is a single logit scale.** → **learned gaze networks (MLP / recurrent / transformer / spatial policy), candidate scoring, continuous coordinates, distribution over locations**.
4. **Precision is entangled with uncertainty.** → **explicit precision system independent of `U_t`**.

### 5. Optimization issue (not architectural, but must be fixed before attributing any Gen-2 result)

- SGD vs AdamW / warmup / cosine decay; per-subsystem LR; gradient-clipping calibration; TRADES calibration.
- Predictor target moves because the encoder trains — mitigate with EMA target encoder (candidate, not default).
- PGD-only robustness may be insufficient — include AutoAttack + EOT in the protocol.

### 6. Architectural issue (resolved by design)

- The input system is local-glimpse-only; the input system must become global → local → multi-scale.
- The error system is mean-pooled into a single scalar; it must become spatial + multi-scale.
- The memory system is per-image only; the architecture needs working memory, and future relational / object / scene state.

### 7. Evaluation issue

- No calibration / human-alignment metrics in the current protocol; add ECE, NLL, AUROC, shape/texture bias, occlusion, clutter, crowding, viewpoint, distribution-shift, human-agreement measures.
- No stochastic-policy attack / robust evaluation of the learned gaze.

---

## PART II — KIMI K3 → GEN-2 TRANSLATION

Every Kimi finding is mapped to problem / architectural intervention / implementation concept / experiment / expected benefit / failure mode / metric.

### K1 — The model never sees the scene
**Problem:** no global/peripheral scene representation. The policy has no coarse first pass.
**Intervention:** a t=0 full-image gist glimpse + peripheral representation + multi-scale input.
**Implementation concept:**
- Fixed gist (`t=0`, 224×224 → 56×56 → 16 tokens) + learned gist (extra token bank) + multi-scale gist (2–3 resolution branches).
- Low-resolution global token bank shared across all phases; peripheral/foveal split from day one.
- Global-to-local attention; scene tokens; spatial coordinates; positional representation (2-D coordinate embedding + RoPE-style relative positions).
- Result: global scene gist (coarse object layout, peripheral context, scene-level priors) fused with local glimpses before the belief update.
**Experiment:**
- G2-K1-gist: (a) no gist, (b) fixed gist, (c) learned gist, (d) multi-scale gist. 5 seeds each. Measure clean / robust accuracy, belief-stability under perturbation, gaze efficiency, AIC/BIC over token count, and attention-entropy.
**Expected benefit:** gives the policy a scene prior; reduces the number of glimpses needed to localize the decisive region; improves OOD and distribution-shift behavior.
**Failure mode:** gist over-reliance (scene-token dominance), foveal neglect, peripheral neglect, compute explosion (multi-branch encoding).
**Metric:** clean acc, robust acc, gaze entropy, glimpse efficiency, spatial-error localization, failure-mode present/absent.

### K2 — Learning signal may be too weak
**Problem:** adversarial fine-tuning may damage useful features; the learning objective family is thin.
**Intervention:** keep the adversarial TRADES curriculum, but add explicit backbone-adaptation controls and a candidate objective family (frozen trunk / low-LR trunk / partial adaptation / EMA target encoder / latent prediction / masked latent prediction / future latent prediction / temporal prediction / multi-step prediction / predictive coding / contrastive / redundancy reduction / object+scene prediction).
**Implementation concept:**
- Per-subsystem LRs: backbone (possibly low-LR or frozen), predictor, UpdateNet, evidential head, AIS policy, readout.
- Three backbone arms: (a) frozen, (b) low-LR (~1e-5…1e-4), (c) partial/fine-tuned with feature-drift monitoring (per-layer |Δnorm| → |W| logged).
- Candidate losses assembled as a searchable family `L_total(λ_cls, λ_pred, λ_belief, λ_policy, λ_uncertainty, λ_precision, λ_adv, λ_consistency, λ_memory)` — search over combinations, not a single hardcoded equation.
- EOT-aware evaluation of stochastic gaze (auto-attack with candidate sampling).
**Experiment:**
- G2-K2-precision / G2-K2a-backbone-adapt: frozen vs low-LR vs partial, with feature-drift monitoring. 3–5 seeds per arm.
- G2-K2-pred: latent prediction / masked latent / future latent / temporal / multi-step / predictive-coding losses as a grid search over λ_pred.
**Expected benefit:** separates feature destruction from mechanism gain; a richer self-supervised signal that does not depend on adversarial labels.
**Failure mode:** loss domination, gradient starvation, shortcut learning, representation drift, predictor mean-collapse.
**Metric:** clean acc, robust acc, feature-drift (|Δnorm| / |W|), backbone feature-similarity decay, ECE, failure-mode flags.

### K3 — AIS-v2 policy capacity too small
**Problem:** policy dominated by `logit_scale`; candidate sampler fixed; K is small.
**Intervention:** replace the single scalar policy with a learnable gaze network. Compare candidate-based scoring, continuous coordinates, spatial heatmap, stochastic policy, hierarchical gaze, coarse-to-fine selection, uncertainty-driven / prediction-error-driven / information-gain / novelty-driven / task-driven gaze, learned candidate generation.
**Implementation concept:**
- Candidate-sampling side: learned candidate generator (keeps K cheap) + K candidates from belief + uncertainty hotspots.
- Policy side: MLP / recurrent / transformer / belief-conditioned / scene-conditioned / uncertainty-conditioned / spatial-policy-map / object-aware / memory-aware policies. Evaluate whether the policy should output (a) direct coordinates, (b) candidate scores, (c) a distribution over locations, (d) an action program.
- EOT-aware stochastic policy evaluation; hard argmax inference.
**Experiment:**
- G2-K3-policy: fixed vs logit-only vs MLP vs recurrent vs transformer vs spatial-map, with the same K and same EOT evaluation. 5 seeds per arm.
- G2-K3a-information-gain: belief-conditioned + uncertainty-conditioned policy vs logit-scale policy.
**Expected benefit:** adaptive gaze that searches to reduce uncertainty rather than only following a fixed heuristic; better sample efficiency and human-like gaze allocation.
**Failure mode:** center bias, gaze collapse, policy collapse, adversarial policy exploitation, compute explosion (transformer policy at K candidates).
**Metric:** gaze entropy, gaze diversity, gaze trajectory, gaze efficiency, human-agreement on gaze, information gain, robust acc.

### K4 — Precision conflated with uncertainty
**Problem:** `Π_t = 1 - U_t` mixes epistemic / sensory / attentional / observation-quality.
**Intervention:** explicit precision decomposition. `Π_t` independent of `U_t`.
**Implementation concept:**
- Candidate precision fields: `Π_t = constant`, `Π_t = 1 - U_t`, `Π_t = learned scalar`, `Π_t = learned spatial map`, `Π_t = learned channel-wise precision`, `Π_t = uncertainty-conditioned precision`, `Π_t = policy-conditioned precision`.
- Uncertainty decomposition into epistemic / aleatoric / sensory / perceptual / policy / prediction, with explicit estimation routes for each (evidential, ensemble, MC-dropout, deterministic momentum, policy entropy, prediction-disagreement).
**Experiment:**
- G2-K4-precision: constant / 1-U / learned-scalar / learned-spatial-map / channel-wise / policy-conditioned precision. 5 seeds each.
- G2-K4a-uncertainty-decomp: where does each uncertainty component live and which one actually predicts action-value?
**Expected benefit:** precision becomes a controllable attention gain rather than an epiphenomenon of uncertainty; better robustness, better calibration, cleaner ablation attribution.
**Failure mode:** precision collapse (all ones or all zeros), degenerate gradients, loss domination, uncertainty collapse.
**Metric:** calibration (ECE), NLL, AUROC, uncertainty-quality, belief-stability, precision-field L2, failure-mode flags.

### K5 — Observed features should enter the belief update
**Problem:** new evidence reaches `z_t` only through prediction error.
**Intervention:** `z_t` update input becomes `[E_t, observed features]` with explicit fusion.
**Implementation concept:**
- Concatenation / gated fusion / cross-attention / FiLM-style modulation / recurrent fusion / attention over error + observation / spatially aligned evidence fusion.
- Keep `E_t` non-detached in the update (locked for Gen-1); add an explicit observed-features pathway in addition to the error pathway.
**Experiment:**
- G2-K5-observed-error: off vs on, with the same E_t path active; compare belief revision magnitude and spatial-error localization.
**Expected benefit:** richer, more direct use of evidence; better uncertainty calibration and belief-content uncertainty.
**Failure mode:** belief instability, double-counting of evidence, feature drift, loss domination, gradient conflict between paths.
**Metric:** belief-revision magnitude, spatial error localization, uncertainty quality, human agreement, failure-mode flags.

### K6 — Mean-pooling destroys spatial surprise
**Problem:** a global mean of `E_t` hides where the surprise happened.
**Intervention:** preserve spatial error structure as a first-class output.
**Implementation concept:**
- Spatial error map / error tokens / multi-scale error pyramid / error attention / error saliency map / error-to-gaze pathway / error memory.
- Policy answers "where did the prediction fail?" not merely "how much did the prediction fail?".
- Multi-scale error pyramid: patch-level, cell-level, region-level error at two scales.
**Experiment:**
- G2-K6-spatial-error: off vs on vs multi-scale error pyramid. 5 seeds each.
**Expected benefit:** targetable re-look, better object-level robustness, cleaner attribution of where the model was surprised.
**Failure mode:** scene-token dominance, gist over-reliance, foveal neglect, peripheral neglect, loss domination.
**Metric:** spatial-error localization (predicted error maps vs human attention maps), belief-stability under occlusion, gaze trajectory, failure-mode flags.

### K7 — Belief revision may be too heavily bound
**Problem:** `Δz ≤ 0.1` / T=4 may restrict belief updating; unvalidated.
**Intervention:** test bounded vs unbounded revision, adaptive update magnitude, learned update gates, confidence/uncertainty/evidence-dependent update, recurrent accumulation, nonlinear belief transitions, persistent (short-term) memory, catastrophic-reset mechanisms.
**Implementation concept:**
- Experimental (Gen-2) arms: bounded (baseline), adaptive (update magnitude gated by confidence/uncertainty), unbounded (test only), recurrent accumulation (leaky integrator), nonlinear transition.
- Keep the bound for the core build; expose a release knob as an experimental arm.
**Experiment:**
- G2-K7-belief: bound vs adaptive vs recurrent accumulation vs unbounded (G3-class, too risky for Gen-1). 5 seeds each.
**Expected benefit:** an empirical answer to whether the bound is protective or destructive.
**Failure mode:** belief instability, belief inertia, belief explosion, catastrophic reset, loss domination.
**Metric:** belief-revision magnitude, belief-stability (drift under perturbation), human-agreement, failure-mode flags.

### K8 — No reasoning loop by design
**Problem:** Gen-1 is a perceptual front-end; reasoning is not desired yet.
**Intervention:** design a roadmap where we add later: perception → belief → working memory → scene representation → reasoning → decision. Gen-2 should set the perceptual substrate; Gen-3 begins richer reasoning/state representations.
**Implementation concept:** keep Gen-2 perceptual; reserve reasoning mechanisms for Gen-3. Do not force it in now.
**Experiment:** record as architectural roadmap only; no Gen-2 experimental arm.
**Expected benefit:** prevents Gen-2 scope creep; makes the Gen-3 reasoning leap legitimate and incremental.
**Failure mode:** none within Gen-2 (by design).

### K9 — Optimization confounders
**Problem:** SGD may limit adaptation; warm-started ViT under-optimized; STL-10 clean accuracy low; PGD-only robustness insufficient; stochastic gaze needs EOT-aware evaluation.
**Intervention:** full optimizer recalibration: AdamW / warmup / cosine decay, frozen trunk / low-LR trunk / per-subsystem LR, EMA target encoder (candidate), multi-stage optimization, gradient-clipping calibration, TRADES calibration, AutoAttack + EOT.
**Implementation concept:**
- Re-run the foundation ladder with (a) AdamW + warmup + cosine, (b) low-LR trunk, (c) per-subsystem LR, (d) frozen trunk arm, (e) EMA target encoder candidate, (f) AutoAttack + EOT evaluation.
**Experiment:**
- G2-K7-optimization / G2-K8-robustness: optimizer matrix (AdamW vs SGD, warmup, LR per block), EMA candidate, AutoAttack + EOT protocol.
**Expected benefit:** cleaner training, better robust accuracy, trustworthy stochastic-gaze evaluation.
**Failure mode:** gradient starvation, loss domination, EMA lag, compute explosion, overfitting to PGD.
**Metric:** clean acc, robust acc (AutoAttack + EOT), gradient-health (pre/post-clip), optimizer group |dW|, failure-mode flags.

---

## PART III — GEN-2 ARCHITECTURE

Gen-2 design objective: **a substantially stronger active recurrent perceptual system that constructs a global-to-local visual representation, maintains a richer belief state, learns where to look, preserves spatial prediction error, explicitly models uncertainty and sensory precision, and learns useful predictive representations without destroying the visual backbone.**
Gen-2 stays a **perceptual architecture** (no LLM-style generic reasoning). Gen-2 = perceptuo-motor system.

### 3.1 Input system
- **Global gist:** t=0 full-image glimpse (224×224 → 56×56 → 16 tokens), fixed or learned, multi-scale (2–3 resolution branches). Gist is fused into `z_t` and the belief state as an additional input channel (global priors, coarse object layout, peripheral context, scene-level priors).
- **Peripheral representation:** separate low-resolution peripheral token stream with its own (small) tokenizer; fused at the belief-update stage; peripheral/foveal split from day one.
- **Foveal representation:** high-resolution local glimpses at the chosen gaze.
- **Multi-scale input:** input pyramid (3–4 scales); each scale has its own tokenization; fusion before the belief update.
- **Token allocation:** learned / heuristic allocation of tokens between gist/peripheral/foveal; resolution allocation learned.
- **Spatial coordinates:** 2-D gaze-coordinate embedding per gaze; RoPE / relative-position encoding; foveal coordinate normalization.
- **Positional representation:** coordinate + relative position + register tokens (optional).
- Boundary: gist and central multi-scale are **GEN-2 CORE** (directly answers the "never sees the scene" flaw); learned resolution allocation is **GEN-2 EXPERIMENTAL**; object-centric tokens are **GEN-3**.

### 3.2 Backbone
Evaluate (not necessarily adopt): DINOv2 / ViT variants / frozen trunk / low-LR trunk / partial adaptation / adapters / LoRA / multi-stage fine-tuning / EMA target encoder.
- Three arms: (a) frozen trunk, (b) low-LR trunk, (c) full fine-tuning with feature-drift monitoring.
- Feature-drift monitoring: per-layer |Δnorm| → |W|, cosine similarity of backbone features over seeds, EMA target encoder candidate for the predictor target (prevent target from chasing a moving encoder).
- Do not assume DINOv2 is permanently correct; the repo has no hard DINOv2 dependency (the substrate's ViT choice is the design lever).
- Boundary: frozen / low-LR trunk and EMA target encoder candidate belong to **GEN-2 EXPERIMENTAL**; full fine-tuning with drift monitoring is **GEN-2 CORE** (needed to answer K2 and K9).

### 3.3 Belief state
Richer than Gen-1's `z, U, E, A`:
- `z_t`: global content, **(B, D_z)** — unchanged, upgraded with gist/peripheral fusion.
- `U_t`: uncertainty (Dirichlet-evidence readout-level); in Gen-2 make the uncertainty components explicit and give it a representation-level branch (deferred fully in Gen-3).
- `E_t`: prediction error, now **spatial + multi-scale** (K6 expansion).
- `A_t`: gaze history + current glimpse index — upgraded with multi-scale/confidence-aware supervision.
- New Gen-2 additions (experimental): working memory (glimpse memory), spatial memory (persistent peripheral bank), confidence, novelty, task relevance, belief-state drift gate.
- Boundary: working memory, spatial memory, confidence, novelty, task relevance = **GEN-2 EXPERIMENTAL**; full episodic / semantic / object memory = **GEN-3**.

### 3.4 Predictive model
Candidate family (do not pick one): next-glimpse prediction (locked Gen-1), multi-step prediction, masked latent prediction, future latent prediction, cross-scale prediction, spatial prediction, object-level prediction, scene-level prediction, counterfactual prediction (Gen-3).
- G2 uses next-glimpse + latent prediction family; counterfactual and object/scene prediction are **GEN-3**.

### 3.5 Error system
- Local / global / spatial / temporal / multi-scale / object-level / prediction-disagreement error.
- Spatial error map + error tokens + multi-scale error pyramid (G6 expansion).
- Boundary: spatial error is **GEN-2 CORE**; multi-scale and object-level are **GEN-2 EXPERIMENTAL**.

### 3.6 Uncertainty system
Explicit decomposition: epistemic / aleatoric / sensory / perceptual / policy / prediction.
- Estimation routes: evidential (readout-level), ensemble, MC-dropout, deterministic momentum, policy entropy, prediction-disagreement.
- Each component has a training target (or explicitly says "no target → do not use yet").
- Boundary: readout-level `U_t` is **GEN-2 CORE**; representation- and policy-level uncertainties are **GEN-2 EXPERIMENTAL**; full belief-content uncertainty is **GEN-3**.

### 3.7 Precision system
Precision independent of uncertainty.
- Candidates: scalar, spatial map, channel-wise, feature-level, attention-level, learned precision field.
- `Π_t = const`, `Π_t = 1 - U_t`, learned scalar, learned spatial map, channel-wise, policy-conditioned.
- Boundary: constant / 1-U is Gen-1; scalar and simple spatial-map are **GEN-2 CORE**; channel-wise and policy-conditioned are **GEN-2 EXPERIMENTAL**.

### 3.8 Active vision / gaze policy
Significantly more capable AIS.
- Candidate-based policy (K candidates, cached, one predictor pass per candidate) — minimum for compute.
- Continuous coordinates (regression), spatial heatmap (Gaussian mixture / heatmap argmax).
- Stochastic policy (soft selection in training, hard argmax inference; EOT-aware evaluation).
- Hierarchical gaze: coarse first pass → fine refinement; coarse-to-fine selection.
- Uncertainty-driven / prediction-error-driven / information-gain / novelty-driven / task-driven gaze, possibly blended.
- Learned candidate generation (keeps K cheap) vs fixed heuristic sampling.
- Boundary: candidate-based + uncertainty-reduction scoring is **GEN-2 CORE**; continuous coordinates, spatial heatmap, hierarchical/CM, learned candidate generation are **GEN-2 EXPERIMENTAL**.

### 3.9 Memory
- **Short-term recurrent state** (the `B_t` trajectory itself) — GEN-2 CORE.
- **Glimpse memory:** per-glimpse evidence + prediction + error cache (small, bounded).
- **Spatial memory:** persistent peripheral token bank shared across images — **GEN-2 EXPERIMENTAL**.
- **Episodic / semantic / scene / object / long-term memory:** **GEN-3**.
- Boundary: working and spatial memory are Gen-2 experimental; episodic / semantic / scene / object / long-term memory are gen-3.

### 3.10 Adaptive computation
- Variable glimpse count, learned stopping, confidence-based stopping, uncertainty-based stopping, prediction-error stopping, dynamic compute allocation.
- Boundary: variable glimpse count / learned stopping are **GEN-2 EXPERIMENTAL** (adaptive halting was deferred Gen-1 for a documented reason); dynamic compute allocation is **GEN-3**.

### 3.11 Classifier / readout
- Classification readout (Dirichlet-evidence) — unchanged, read from final belief.
- Future: uncertainty readout, representation-level uncertainty readout — **GEN-3**.

### 3.12 Information flow
The loop: **input pyramid → gist/peripheral/foveal tokenization → fusion → z_t and belief state update → shared glimpse predictor (K candidates) → score → gaze choice → foveal encode → prediction vs observation → E_t (spatial + multi-scale) → fusion of E_t + observed features → precision-weighted UpdateNet update of z_t → AIS-v2 next gaze → repeat.** The readout reads the final belief. Precision gates the update; U_t gates precision and candidate scoring; E_t carries error.

---

## PART IV — GEN-2 TRAINING SYSTEM

### 4.1 Training pipeline
- **Pretraining / backbone adaptation:** three arms (frozen / low-LR / fine-tuned) with feature-drift monitoring. Candidate EMA target encoder; keep predictor target stable (EMA is one candidate).
- **Predictive learning:** latent-prediction family, λ_pred, multi-step / future / masked / cross-scale as candidate arms.
- **Belief learning:** updateNet, precision, U_t, belief update; per-subsystem LRs.
- **Policy learning:** AIS-v2 policy, λ_policy, soft/hard, EOT-aware.
- **Joint training:** weighted sum family; per-phase gating.
- **Adversarial training:** TRADES/PGD curriculum, with α/β/step calibration, plus AutoAttack + EOT evaluation.
- **Curriculum:** preserve the six-phase ladder but add loss- and phase-gating flags; train with the adapted recipe.
- **EMA target training:** candidate; evaluate stability and feature-drift.
- **Multi-loss balancing:** search over λ family; report gradient norms and loss trajectories per term.
- **Gradient management:** incorporate the gradient-health harness (pre/post-clip, tensor counts) into the experimental framework. Per-subsystem clip calibration.
- **Checkpointing:** freeze the clean baseline; checkpoint every phase; provenance manifest per run (config, seed, dataset fingerprint, code revision, architecture revision, recipe, evaluation protocol, metrics, checkpoint, manifest, result summary).
- **Phase transitions:** use `stage_state_machine.py` as the execution engine; gate each phase transition on the gradient-reach + frozen-hash checks.
- **Ablations:** built by flipping one flag in the published config, never by hand-assembling a similar config.
- **Controls:** parameter-matched and compute-matched controls established for every claim; the gradient-health harness provides matched-FLOPs control.

### 4.2 Loss family (candidate family, not a single hardcoded equation)
`L_total = λ_cls·L_cls + λ_pred·L_pred + λ_belief·L_belief + λ_policy·L_policy + λ_uncertainty·L_uncertainty + λ_precision·L_precision + λ_adv·L_adv + λ_consistency·L_consistency + λ_memory·L_memory`
- Per term: what it teaches, failure modes, interaction, collapse modes, gradient conflicts, which phases activate each.
- λ per term and per phase are hyperparameters to search; no term is hard-coded as the default.

### 4.3 Prospective Gen-2 phases (candidate; extend the existing six-phase ladder)
Defined per phase: name, purpose, active modules, frozen modules, trainable modules, losses, optimizer, data, number of glimpses, policy behavior, checkpoint, exit criteria, failure criteria, evaluation.

---

## PART V — GEN-2 EXPERIMENT MATRIX

Core matrix (13 arms, each vs the Gen-2 reference config, one flag changed):

| Arm | What changes vs reference | Isolates |
|---|---|---|
| G2-K1-gist (no gist / fixed gist / learned gist / multi-scale gist) | gist type | global scene prior |
| G2-K2-backbone (frozen / low-LR / fine-tuned + drift monitoring) | trunk adaptation | feature preservation |
| G2-K2-pred (latent / masked / future / multi-step / predictive-coding candidate) | prediction family | learning signal richness |
| G2-K3-policy (fixed / logit-only / MLP / recurrent / transformer / spatial map) | policy capacity | adaptive gaze |
| G2-K3a-information-gain | belief + uncertainty conditioned policy | information acquisition |
| G2-K4-precision (constant / 1-U / learned scalar / spatial map / channel-wise / policy-conditioned) | precision field | precision disentanglement |
| G2-K5-observed-error (off / on) | observed-feature pathway | belief-update evidence |
| G2-K6-spatial-error (off / on / multi-scale) | error spatial structure | where surprise occurred |
| G2-K7-belief (bounded / adaptive / recurrent accumulation) | belief update rule | revision bounds |
| G2-K7-optimization (AdamW / SGD, warmup, cosine) | optimizer | adaptation headroom |
| G2-K7-ema (off / on) | EMA target encoder | target stability |
| G2-K8-robustness (PGD / AutoAttack + EOT) | attack protocol | robust evaluation |
| G2-K9-controls (param-matched / compute-matched) | control arms | attribution |
| Parameter-matched and compute-matched controls, every claim |

Higher-order (2–3-way) interaction experiments where justified (e.g., gist × policy, precision × observed-error, backbone × prediction family).

---

## PART VI — GEN-2 EVALUATION SYSTEM

Metrics: clean accuracy, robust accuracy (PGD + AutoAttack + EOT for stochastic gaze), calibration (ECE), NLL, AUROC, uncertainty quality, prediction quality, belief dynamics, belief-revision magnitude, gaze entropy, gaze diversity, gaze trajectory, gaze efficiency, spatial error (localization vs human attention), information gain, glimpse efficiency, compute efficiency (FLOPs/param per correct decision), human agreement, shape bias, texture bias, occlusion robustness, clutter robustness, distribution shift, peripheral degradation, viewpoint changes.
Diagnostics for every new mechanism: U_t calibration, precision-field L2, gaze-center ratio, belief-drift under perturbation, candidate-score correlation (reuse `ais_v2_smoke_gate_v1` style), gradient health, failure-mode flags (all 22).

---

## PART VII — GEN-3 ARCHITECTURE

Gen-3 should not be "Gen-2 but bigger." It is the next conceptual leap: **a hierarchical active perceptual intelligence system with persistent scene/world state, structured memory, predictive perception, learned attention, object/scene representations, adaptive computation, and reasoning over the evolving perceptual state.**

### 7.1 Hierarchical perception
`pixels → features → objects → scene → world state`. Multi-scale to multi-level processing: pixel, patch, object, scene, world-state levels.

### 7.2 Persistent world model
Maintains: what exists, where it is, what was observed, what remains uncertain, what changed, what should be observed next. Sits between perception and reasoning; persistent across images (chunked), not per-image.

### 7.3 Object-centric state
Object slots, object identities, object attributes, object relations, object persistence, object-level prediction.

### 7.4 Scene graph
Objects, relations, spatial structure, uncertainty, temporal state.

### 7.5 Long-term memory
Episodic memory, semantic memory, visual memory, scene memory, experience replay, retrieval.

### 7.6 Counterfactual perception
"What would I expect to see if I looked there?" — the Gen-3 forward-prediction capacity.

### 7.7 Gen-3 active perception
Policy asks not just "where next?" but "which observation maximally reduces uncertainty / improves the internal model": information gain, expected prediction improvement, uncertainty reduction, task utility, novelty, model disagreement, expected classification improvement, human-like gaze prior.

### 7.8 Gen-3 reasoning
Only after the Gen-2 perceptual substrate validates: `belief → memory → reasoning → action`. Mechanisms: recurrent reasoning, latent reasoning, scene-graph reasoning, relational reasoning, transformer reasoning, iterative hypothesis testing, counterfactual reasoning, uncertainty-aware reasoning. Keep RHAN as an active perceptual intelligence architecture, not an LLM.

### 7.9 Boundary
Object-centric and scene-graph are **GEN-3 CORE**; long-term memory is **GEN-3 STRONG EXPERIMENTAL**; counterfactual perception is **GEN-3 HIGH-RISK/HIGH-REWARD**; reasoning is **GEN-3** (with a Gen-2 experimental precursor).

---

## PART VIII — GEN-3 TRAINING

- Self-supervised learning (masked prediction, latent prediction, future prediction, temporal consistency, contrastive, world-model learning).
- Object persistence and active-learning objectives.
- Continual learning.
- Joint generation of the training recipe; optimized with the same tooling as Gen-2.

---

## PART IX — GEN-3 EXPERIMENT MATRIX

G3 experiment IDs equivalent to the G2 series: G3-K1-world-state, G3-K2-object-scene, G3-K3-persistent-memory, G3-K4-counterfactual, G3-K5-reasoning, G3-K6-self-supervised, G3-K7-hierarchical-perception, G3-K8-graduated-training, G3-K9-robustness... etc.

---

## PART X — FAILURE-MODE CATALOGUE

All entries: symptom / cause / diagnostic / prevention / ablation / recovery.

| Failure | Symptom | Cause | Diagnostic | Prevention | Ablation | Recovery |
|---|---|---|---|---|---|---|
| **Policy collapse** | policy always picks the same location; gaze entropy → 0 | insufficient gradient signal, reward hacking, logit-scale dominance | gaze entropy, gaze diversity, selected-location histogram | EOT-aware stochastic policy, entropy regularization, diverse candidates | remove policy learning, compare logit-only vs learned | decay λ_policy; re-center rewards; reinitialize policy head |
| **Gaze collapse (center bias)** | empirical gaze histogram peaks at center; center-bias ratio high | learned policy extracts center shortcut; guiding prior on | center-bias histogram, info-gain per center-fixation | track and publish, hard center-bias gate, randomize initial gaze | remove learned policy; fix initial gaze | penalize center-only behavior; reweight candidates |
| **Uncertainty collapse** | U_t → 0 (overconfident) or U_t → 1 everywhere | optimizer minimizes U_t, or U_t has no target | U_t histogram, calibration curve | separate uncertainty and precision; add calibration loss; freeze U_t | reweight λ_uncertainty | remove U_t from the update or add regularization |
| **Precision collapse** | Π_t → 0 or 1 everywhere; gradients vanish or explode | precision prediction unregularized; feedback loop with U_t | precision-field L2, gradient magnitude per group | constrain Π_t, clip, add diagnostic gates | ablate precision term | clamp; reweight λ_precision; re-init |
| **Predictor mean-collapse** | predictor outputs the mean of the local features; E_t → 0 | predictor has no target pressure; shared-predictor degradation | predictor variance, residual distribution, E_t entropy | detach observation target, add predictor target pressure, freeze parts | remove predictor from the loop | re-init predictor; add target pressure |
| **Belief inertia** | z_t barely changes across glimpses | bounded Δz, Π too small, beliefs barely updated | belief-dynamics drift, z_t step norm | adaptive update magnitude, remove (or test) bound | ablate Δz | moderate bound, tune Π |
| **Belief instability** | belief trajectory diverges under small perturbation | large update steps, Π too high, feedback from U_t | belief-stability metric, drift between clean/perturbed | gradient clipping, Π clamp, gating, monitor | remove update dynamics | damp updates, lower Π, clip gradients |
| **Memory overwrite / explosion** | spatial working memory grows unboundedly; old evidence lost; catastrophic reset | no capacity limit, no decay | memory occupancy budget, age histogram, reset events | capacity cap, decay, gating, scheduled reset | ablate memory | cap memory, scheduled decay |
| **Representation drift** | backbone features rotate across epochs; features lose utility | unfrozen trunk overfits; no drift target | per-layer |Δnorm|/|W|, feature cosine similarity | drift monitoring, feature anchor, low-LR trunk | re-freeze trunk, lower LR | EMA, low-LR, early stopping |
| **Shortcut learning** | model exploits texture / background / gist shortcut | loss rewards the shortcut; insufficient control | attention/gradient maps, intuition-contributiveness, control arms | hard controls (param/compute matched), diverse data, disentanglement | remove shortcut feature | retrain with control, reweight loss |
| **Texture bias** | model trades robustness for texture accuracy | superficial cues dominate | shape vs texture bias suite | shape-bias eval, mix dataset, adversarial filter | ablate texture-heavy arm | retrain with shape-weighted loss |
| **Scene-token dominance** | gist tokens outcompete local features | gist is too strong, gating weak | attention map on scene vs spatial tokens | gating, token cap, contrastive token balance | ablate gist token | scale down gist, raise local weight |
| **Gist over-reliance** | model fails when gist is missing or wrong | no fallback to local | remove gist (G2-K1-no-gist) | balanced loss, fallback control | remove gist | tune gist weight |
| **Foveal neglect** | policy never samples high-resolution center details | candidate sampling bias, gaze cost | sampled-gaze histogram, info-gain per candidate | candidate diversity, equal sampling, guide prior | remove gaze, fix gaze | re-weight candidates |
| **Peripheral neglect** | model ignores peripheral tokens, great at center | peripheral signal weak, gating favors fovea | peripheral token gradient norm, peripheral-fixation rate | equal gaze budget or importance weighting | ablate peripheral stream | tune peripheral loss or capacity |
| **Adversarial policy exploitation** | policy chooses adversarially exploitable views | policy trained without attack awareness | targeted adversarial re-score of chosen views | EOT-aware attacks, policy robust objective, constrained views | remove or re-train policy | re-distribute candidates, robustify policy |
| **Gradient starvation** | some groups never receive gradients; others dominate | loss scaling, group LR, bound | pre/post-clip gradients per group, tensor counts | shared clip calibration, per-group LR, diagnostic | ablate one loss | re-tune weights, per-group LR |
| **Loss domination** | one loss term dominates the total; others flat | λ imbalance, gradient scale mismatch | loss trajectory per term | per-term λ search, gradient balancing, gradient-norm normalization | remove dominant term | recalibrate λ |
| **EMA lag** | predictor target diverges from real features; too stale or too fresh | EMA momentum too high/low | target-vs-real cosine similarity, drift | tune β, monitor, conditional EMA | remove EMA | re-tune EMA β, step | re-initialize target, adjust β |
| **Compute explosion** | training/inference time exceeds budget | too many candidates, multi-scale branches, long sequences | FLOPs, VRAM, wall-time | fixed candidate budget, cap scales, early-halting | remove scale or candidates | prune scales, cap candidates, cap glimpses |
| **Class imbalance / dataset** | results noise, minority-class collapse | dataset size/quality | per-class metrics | balanced sampling, class weighting, more seeds | remove hard classes | re-weight or resample |

---

## PART XI — COMPUTE / ENGINEERING REQUIREMENTS

Assumptions: RTX 4060-class GPU for local work; larger experiments on external/HPC.

| Tier | Local dev | Medium | HPC |
|---|---|---|---|
| Hardware | RTX 4060 (16 GB) local | A100/H100 or equivalent (40–80 GB) | 8× A100 (80–80 GB) or distributed |

Training budget (estimate): local 16-bit runs of the Gen-2 substrate with T=4 glimpses, batch 32, ~20–40 min/run at small scale; full ladder (once per pass) ~6–24 h on 1× 4060-class local, 1–4 h on 1× A100; full matrix (13+ arms, 5 seeds) ~2–6 weeks local; 1× A100 ~1–3 days. HPC-scale: full 16-seed protocol + AutoAttack + EOT ~1–3 weeks on a node; production runs on 8× nodes.
Parameter count: compact ViT (Gen-1 substrate) + gist + peripheral tokenizers + policy network + precision field + error pyramid; budget ~10–30M params (parameter-matched controls for every claim).
Inference: real-time at small scale; heavier at multi-scale; limit via candidate budget and early halting.
Dataset size: STL-10 / ImageNet-100 / ImageNet-1K variants per the repository; use the repository's canonical data loaders; keep datasets pinned.
No compute limitation should prevent the exploration: label each experiment Local / Medium / HPC; all experiments include small-faithfulness probes run locally first.

---

## PART XII — REPOSITORY / IMPLEMENTATION ROADMAP

The cleaned repo is the install target. The repository map (canonical + archive) is already clean; Gen-2/Gen-3 work lands in:

- `noesis_vision/` — model, belief, predictive coding, uncertainty, gaze, memory classes; the core package.
- `training/` — the Gen-2 trainer (ladder, loss family, optimizer controls, diagnostic harness, manifest engine).
- `evaluation/` — Agent I scales (Agent-1: measurement; Agent-2: human-alignment and psychophysics; Agent-3: generative/robustness).
- `scripts/` — gates, verifiers, sweep runners, provenance manifest.
- `tests/` — mechanism-seam tests extended (per-arm assertions, drift monitoring, gradient health).
- `report/` — result records, GEN2 master results, GEN3 master results.
- `diagnostics/` — measurement/forensic tooling (measurement harness migrates here).
- `cloud/` — per-generation launchers; cloud/gen2 and cloud/gen3 for later stages.
- `archive/` — existing generations stay untouched; future Gen-4+ or rejected builds go here.

Do not implement. Only specify install points, boundary rules (new modules must define interfaces first), and governance (single source of truth in the manifest, no ad-hoc config files).

---

## PART XIII — FINAL MASTER ROADMAP

Chronological implementation plan, from the frozen Gen-1 baseline:

```
Gen-1 frozen baseline (rhan-nxa-clean-baseline @ 14771f1)
  ↓
Gen-2 FOUNDATION
  ├── G2-K1: gist + peripheral + multi-scale input (fixed gist first, learned second)
  ├── G2-K2: backbone-adaptation arms (frozen / low-LR / fine-tuned + drift monitoring)
  ├── G2-K9: optimizer recalibration (AdamW + warmup + cosine, per-subsystem LR, clip calibration)
  └── G2-K7: optimizer + EMA candidate (AdamW, warmup, cosine, EMA target encoder)
  ↓
Gen-2 ACTIVE PERCEPTION
  ├── G2-K3: policy expansion (fixed → logit-only → MLP → recurrent → transformer → spatial map)
  ├── G2-K3a: information-gain / uncertainty-conditioned policy
  ├── G2-K6: spatial error + error pyramid
  └── G2-K5: observed-features belief-update pathway
  ↓
Gen-2 BELIEF + PREDICTION
  ├── G2-K4: precision system independent of U_t (scalar / spatial map / channel-wise / policy-conditioned)
  ├── G2-K7-belief: belief-update rule (bounded / adaptive / recurrent accumulation)
  ├── G2-K2-pred: prediction family (latent / masked / future / multi-step)
  └── G2-K1a: multi-scale + temporal prediction (extension of K2)
  ↓
Gen-2 MEMORY
  ├── G2-K7a: spatial working memory + glimpse memory (bounded)
  └── G2-K7b: spatial memory (persistent peripheral bank) — GEN-2 EXPERIMENTAL
  ↓
Gen-2 ADAPTIVE COMPUTATION
  ├── G2-K8a: variable glimpse count + learned stopping (experimental)
  └── G2-K8b: dynamic compute allocation — GEN-3
  ↓
Gen-2 HUMAN-LIKE EVALUATION
  ├── G2-E1: calibration + human-agreement + gaze allocation suite
  ├── G2-E2: shape/texture bias, occlusion, clutter, crowding, viewpoint
  └── G2-E3: distributional-shift + peripheral-degradation suite
  ↓
Gen-3 WORLD STATE
  ├── G3-K1: persistent scene/world-state representation
  ├── G3-K2: object-centric state (slots, identity, attributes, relations, persistence)
  └── G3-K3: scene graph
  ↓
Gen-3 PERSISTENT MEMORY
  ├── G3-K4: episodic / semantic / visual memory
  └── G3-K5: experience replay + retrieval
  ↓
Gen-3 COUNTERFACTUAL PERCEPTION
  ├── G3-K6: "what if I looked there?" forward prediction
  └── G3-K7: model disagreement / counterfactual belief revision
  ↓
Gen-3 REASONING
  ├── G3-K8: recurrent / latent reasoning over belief + memory
  ├── G3-K9: scene-graph / relational reasoning
  ├── G3-K10: iterative hypothesis testing + counterfactual reasoning
  └── G3-K11: uncertainty-aware reasoning
  ↓
Gen-3 FULL ACTIVE PERCEPTUAL INTELLIGENCE
  ├── G3-K12: generation-3 probe suite (human-like science, visual search, gaze allocation, confidence)
  ├── G3-K13: production harness + documentation
  └── G3-K14: final scientific evaluation (claim-level attribution, final numbers)
```

---

## GEN-2 VS GEN-3 BOUNDARY

| Feature | Gen-2 | Gen-3 |
|---|---|---|
| **GEN-2 CORE** | Global gist + peripheral/foveal + multi-scale input | — |
| | Frozen / low-LR / fine-tuned trunk with drift monitoring | — |
| | Spatial prediction error + multi-scale error pyramid | — |
| | Learned gaze policy (MLP / recurrent / transformer / spatial map) | — |
| | Precision system independent of U_t (scalar / spatial map / channel-wise / policy-conditioned) | — |
| | Belief update family (bounded / adaptive / recurrent accumulation) | — |
| | Working memory + glimpse memory (bounded) | — |
| | Variable glimpse count + learned stopping | — |
| | AutoAttack + EOT evaluation of stochastic gaze | — |
| | Human-alignment + psychophysics evaluation suite | — |
| **GEN-2 EXPERIMENTAL** | Learned resolution allocation | — |
| | Continuous-coordinate policy | — |
| | Spatial heatmap policy | — |
| | Hierarchical / coarse-to-fine gaze | — |
| | Learned candidate generation | — |
| | Spatial memory (persistent peripheral token bank) | — |
| | Episodic / semantic / visual memory | — |
| | Experience replay / retrieval | — |
| | Long-term memory | — |
| | Counterfactual perception | — |
| | Transformer / compositional reasoning | — |
| | Scene graph | — |
| | Object-centric representation (slots, identity, attributes, relations) | — |
| | Self-supervised / world-model learning | — |
| **GEN-3 CORE** | Persistent scene/world state | — |
| | Object-centric state | — |
| | Scene graph | — |
| | Persistent memory (episodic / semantic / visual) | — |
| | Counterfactual perception | — |
| | Self-supervised / world-model learning | — |
| | Transformer / relational reasoning | — |
| **GEN-3 STRONG EXPERIMENTAL** | Long-term memory | — |
| | Uncertainty decomposition beyond readout-level | — |
| | Generative/counterfactual perception | — |
| **GEN-3 HIGH-RISK/HIGH-REWARD** | Counterfactual perception | — |
| | Reasoning | — |

---

## EXPERIMENT IDENTITY CONVENTION

All experiments are pre-registered with IDs such as:
- `G2-K1-gist` (fixed gist)
- `G2-K1b-gist-learned`
- `G2-K1c-gist-multi-scale`
- `G2-K1-no-gist` (control)
- `G2-K2-backbone-frozen`
- `G2-K2-backbone-lowlr`
- `G2-K2-backbone-finetuned` (+ drift monitoring)
- `G2-K2-pred-latent`
- `G2-K2-pred-masked`
- `G2-K2-pred-future`
- `G2-K2-pred-multistep`
- `G2-K2-pred-prediction-coding`
- `G2-K3-policy-fixed` (control)
- `G2-K3-policy-logitonly`
- `G2-K3-policy-mlp`
- `G2-K3-policy-recurrent`
- `G2-K3-policy-transformer`
- `G2-K3-policy-spatialmap`
- `G2-K3a-info-gain`
- `G2-K4-precision-const`
- `G2-K4-precision-1-U`
- `G2-K4-precision-learned-scalar`
- `G2-K4-precision-spatialmap`
- `G2-K4-precision-channelwise`
- `G2-K4-precision-policyconditioned`
- `G2-K5-observed-error-off`
- `G2-K5-observed-error-on`
- `G2-K6-spatial-error-off`
- `G2-K6-spatial-error-on`
- `G2-K6-spatial-error-multiscale`
- `G2-K7-belief-bounded` (control)
- `G2-K7-belief-adaptive`
- `G2-K7-belief-recurrent`
- `G2-K7-belief-unbounded` (G3 bridge, HIGH-RISK)
- `G2-K7-optimization-AdamW`
- `G2-K7-optimization-SGD`
- `G2-K7-ema-off`
- `G2-K7-ema-on`
- `G2-K8-robustness-PGD`
- `G2-K8-robustness-AutoAttack-EOT`
- `G2-K9-controls-param` (+ compute)
- `G2-K9a-controls-compute`
- `G2-K10-adapt-glimpses-off` (control)
- `G2-K10-adapt-glimpses-on`
- `G2-K11-a` calibration; `G2-K11-b` shape/texture; `G2-K11-c` occlusion/clutter; `G2-K11-d` viewpoint; `G2-K11-e` distribution-shift / peripheral / human agreement.

Each experiment carries: config / seed / dataset fingerprint / code revision / architecture revision / training recipe / evaluation protocol / metrics / checkpoint / manifest / result summary.
