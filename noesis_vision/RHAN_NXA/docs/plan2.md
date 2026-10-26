# RHAN-NXA — Proposed Next Architecture and Falsifiable Roadmap

**Status:** research and architecture proposal for joint review. **Nothing in this document is authorized for implementation.** No registered experiment, checkpoint, artifact, or configuration is modified or proposed for modification by this document itself; §18 lists the implementation plan separately and explicitly marks what would need separate authorization.

**Inputs used**

| Input | How it was used |
|---|---|
| Revision 1 forensic audit (`docs/revision1.md`, audit date 2026-10-17, L1–L28 ledger) | Primary account of verified behavior. Cited throughout as audit §x / L# / D5.#. |
| `plan.md` v3.0 (2026-10-14) | Historical context and registered experiment IDs. Treated as context, not as binding spec (per brief §6 and audit §16.6, which documents places where the plan is wrong). |
| This session's prior critique and W1.5 interpretation | Background; wherever the audit supersedes it, the audit wins. |

**Conventions.** Confidence labels follow the audit exactly: VERIFIED, SUPPORTED, PROVISIONAL, HYPOTHESIS, UNRESOLVED, CONTRADICTED, NOT TESTED. Evidence tags: [M] measured, [R] reported, [I] inferred. Every numeric threshold proposed in this document is marked **[REQUIRES APPROVAL]** — none is preregistered. New experiment IDs use the prefix `NX-` to avoid collision with the `G2-`/`G3-` registry; where an experiment is identical to a registered plan.md ID (e.g. `G2-CTL2`), both names are given and the registered ID is respected.

**Requirement map.** The brief's 16 required deliverable sections appear as §1–§4, §7–§18 below, in order. Two inserted sections cover material the brief demands but that has no home in the 16: §5 (the five immediate questions A–E) and §6 (candidate-mechanism triage). Appendix A shows the full mapping.

---

## 1. Executive recommendation

**Do not authorize an architecture build yet. This proposal deliberately separates what should be decided by two cheap experiments from what should be committed to now.**

Three statements from the audit are established strongly enough to constrain everything that follows (audit §17):

1. The 56-px substrate cannot test information acquisition: every observation is a translation of one downsampled tensor, and a gaze-0 crop is bit-identical to the input (L3, audit §7.1). **VERIFIED.**
2. The measured "recurrence" benefit is multiview averaging plus extra compute; the best checkpoint in the ladder contains no recurrent belief machinery at all (L2, audit §8). **VERIFIED.**
3. The uncertainty, gist-gate, fusion, EMA-target, and gaze-temperature mechanisms are inert as shipped, and the calibration metrics that were supposed to justify the uncertainty mechanism were mis-implemented (L4–L12, audit §6). **VERIFIED.**

Given those, the recommendation has four parts.

**First, repair the instrument before training anything (Stage T0, eval-only, ≈1 GPU-day).** The audit's ordered sequence E0.2 → E0.1 is adopted unchanged, plus two new eval-only probes this document adds because they are cheap and decide questions the audit leaves open: a **trunk-gist swap probe** (NX-T0.4 — replace the trunkless linear gist with the trunk's own CLS at evaluation time; isolates "representation of the initial view" from "input resolution") and a **belief-revision probe** (NX-T0.5 — constructed contradiction sequences that measure whether any existing checkpoint revises a decision when later evidence contradicts earlier evidence; this is the behavioral test the audit says is missing, audit §9.3). Until T0 reports, every absolute robustness number in the historical record should be treated as unestablished, and every calibration number except clean ECE should be treated as withdrawn.

**Second, run the two deciders (Stage T1, ≈25 GPU-h).** `G2-RES1` (frozen-trunk 112-px linear probe) and `G2-CTL2` with its epoch-matched EMA control. These two numbers decide, respectively, whether the next architecture is substrate-first or representation-first, and whether state carry has any value beyond matched multiview aggregation. No Gen-3 build is justified before both report.

**Third, commit now to the architecture skeleton, not to the mechanism verdicts.** The recommended design (A1 "RHAN-Rec", §7–§10) is a corrected belief loop: trunk-processed gist as a persistent prior, belief-conditioned observation readout, a bounded GRU-style update that consumes prediction error, softmax-based uncertainty as the default action signal, fixed-T operation, and the code-verified TRADES objective with one additional ablatable predictor term. Its substrate extension (A2 "RHAN-HR": native-resolution glimpses from a high-res source with a 112-px trunk-processed gist) is the only branch in which the plan's P1–P4 principles and CLAIM-2/5/7 become testable as stated. The choice between substrate-first and belief-first execution is made by the T1 deciders, not by preference.

**Fourth, adopt the audit's six constraints as hard gates on every future claim** (audit §14): observations must come from a source with more information than the initial view; any gaze policy must beat **random** at matched compute, not merely fixed; any uncertainty signal must pass true-NLL, ECE, error-detection AUROC, and a realized-vs-predicted information correlation before driving actions; any recurrent update needs three separate matched controls (epochs, views, parameters); any stochastic component is evaluated with registered-strength EOT with the deterministic control alongside; precision/update gains are bounded and measured, never assumed.

If forced to commit today with no further measurement: build **A2**. The audit verified that the current substrate cannot express the core hypothesis, and no amount of state machinery can be validated on it. The belief program (A1) is then tested *on* the new substrate under the same matched-control doctrine, and the 56-px ladder is reported as the negative-control study the audit suggests it may already be.

---

## 2. Interpretation of the forensic findings

### 2.1 Six-way classification

**A. Verified implementation defects** (each changes what some reported number means)

| # | Defect | Audit cite | What it invalidates |
|---|---|---|---|
| A1 | CAL2 "NLL" is `−log max p`, label never used | L10, §6.1 | All reported NLL values; the "NLL falls under attack for gen1_core" anomaly is definitional, not empirical |
| A2 | fp16 probability underflow → NaN entropy → AUROC computed on NaNs for `recurrence_only` | L11, §6.2 | The `recurrence_only` error-detection AUROC (0.7708) is uninterpretable |
| A3 | Evidential head never scored (`has_evidential` set, never read); both models scored with softmax entropy | L12, §6.3 | CLAIM-4 status is NOT TESTED, not "unsupported"; reported AUROCs compare two checkpoints, not two uncertainty mechanisms |
| A4 | `gen1_core` stochastic at eval (global-RNG candidates in forward; two forwards differ by max\|Δlogit\| = 0.603) | L9, D5.1 | Every single-draw PGD number on `gen1_core` is a weak-attack estimate; obfuscated-gradient setting by construction |
| A5 | PGD-50 step rule α = ε/steps ⇒ α₅₀ = 0.00188 (weakest of the four attacks; 1/53 of ε per step) | L15, D5.2 | The 32.40 % PGD-50 figure is optimistic; plan E14's "PGD-10 ≈ PGD-50 ⇒ no masking" inference is unsupported (both share the weak step rule) |
| A6 | "APGD-CE-100" is a hand-rolled momentum loop, not APGD/AutoAttack | L17, D5.3 | The 22.67 % figure is not an AutoAttack number; Gate A remains unsatisfied |
| A7 | Gist fusion gate `fuse()` has no call site; zero gradient | L4 | "Gate-0 preserves bit-exact parity" is vacuously true; the documented parity mechanism never runs |
| A8 | `BeliefUpdaterV2` computes `fused` and discards it; `concat`≡`gated`≡`film`≡`error_only` | L5, §11 | The registered K5 fusion modes are one mode; BEL2-style claims about fusion are meaningless as trained |
| A9 | EMA target encoder delegates to the online predictor; shadow weights never used | L6 | All EMA behavior is a no-op; BEL3-style claims meaningless as trained |
| A10 | `logit_scale` divides then multiplies (cancellation to 1.2e-7); gaze-policy group has zero nonzero grads | L7, §16.4 | The "learned gaze policy" learned nothing; AIS-v2 at eval is a fixed deterministic function of its inputs |
| A11 | Refinement block inert (mean\|γ\| 2.5e-4–4.1e-4 vs trunk 1.31; bypass +0.2 clean / +0.6 robust) | L8 | Phase-2/3 checkpoints carry dead capacity; "tied refinement" contributes nothing measurable |
| A12 | Precision unbounded (`softplus`; measured 6.94–7.29) vs documented `[1e-4, 1]` clamp | L14 | The documented precision semantics are not what runs; first update is 421 % of ‖z₀‖ (§9.2) — the prior is overwritten at t=1 |
| A13 | `min(num_glimpses, len(schedule))` caps the loop at 4; D5's T=6/8 rows duplicate T=4 for `recurrence_only` | L18, §10.5 | Gate L's second clause was never evaluated; only T ∈ {1,2,4} are real for the deterministic path |
| A14 | `PositionSensitiveErrorPool` detaches the error map (`err_map.detach()`) | §11 table | The spatial error signal carries no gradient into the predictor; the predictor cannot learn from spatial error structure |
| A15 | Registered eval harness unused by W1.5 scripts; manifests describe a different run (2026-09-25, git `cf6ce8a`); config hash identical across five arms; W1.5 artifacts untracked | L19–L21, D5.7, §16.1 | Provenance for the W1.5 wave does not exist in-repo; results cannot be tied to a config by hash |

**B. Verified experimental-protocol defects**

Registered-strength violations: EOT executed at N=5/20 steps against registered N≥20/50 steps; D8 executed at n=300 on two checkpoints instead of n=500 on six finals; no PGD-50 in the D8 run; no large-ε collapse check; no AutoAttack (D5.1, D5.5, L27). Sample/protocol inconsistencies: nested 300/500 subsets (fair for comparison, but precision differs); 8×300 "seed spread" is subset noise, not stability (plan E16, confirmed by audit §5.1's nesting check). Structural gaps: no select split, so "best" checkpoints are selected on the reporting split (§16.3, plan B5); single training seed 41 everywhere (B8); phase 1/3/4/5 finals absent locally, so the registered "all six finals" evaluation cannot be run as written (B4). Documentation drift: plan §4's ε-ramp description (two linear stages) contradicts the code (three rungs 0.031/0.062/0.094) (§16.6 #7); the loss-form discrepancy is resolved in the code's favor: `w·(CE + β·KL)` = 0.55·CE + 1.375·KL (§3.2, plan E12 settled by code read).

**C. Observed empirical findings that survive** (mechanism-independent)

1. T=1→4 adds +2.4 pp clean / +5.6 pp robust on `recurrence_only`, saturating at T=4 [M, D5]. Real, and attributable to extra trunk passes + view averaging — not to information acquisition (impossible on this substrate, L3) and not to recurrence (fixed EMA, L2).
2. Relative ordering `recurrence_only` > `gen1_core` holds under PGD-50 (32.4 vs 28.2) and EOT (24.8 vs 20.4) on the same 500 images [M, §5.1]. Internally consistent; survives the audit.
3. Random gaze ≥ learned gaze on robustness for both checkpoints (+5.8 pp `recurrence_only`, +4.4 pp `gen1_core`) while costing 4–5 pp clean [M, D7]. Direction real; **magnitude suspect** — random-gaze evaluation makes the model stochastic, so single-draw PGD flatters it exactly as it flatters `gen1_core` (the audit flags mechanism as remaining uncertainty, §10.2). The honest test is EOT over gaze draws, not yet run.
4. Clean ECE ≈ 0.05 for both checkpoints; under PGD-10, ECE 0.21–0.33 [M, §6.4]. The one calibration result that used a verified-correct metric. Both models are reasonably calibrated clean and badly miscalibrated under attack.
5. Evidential state: U ∈ [0.990, 0.994] at z₀, α ≈ 1.01 → ≈1.10 after four updates — a near-uniform Dirichlet with almost no dynamic range; `ev_readout` is 12.8 % of logit norm, 0/16 argmax flips when dropped, 8/8 flips when evidence is replaced by a constant (sensitive to evidence *scale*, but the learned evidence is nearly flat) [M, §6.5].
6. Update trajectory: first update 421 % of ‖z₀‖ (glimpse 1 overwrites the prior), then 10–16 % per step; precision 7.15 → ≈1.2 [M, §9.2]. The state integrates sequentially for phases ≥ 3, but the "belief" does not behave like a persistent prior being refined.
7. Predictor is computationally connected but weak: zeroing its output moves logits by max 0.71, flips 0/8 decisions [M, §9.2].
8. Gate W (frozen DINOv2 probe) 58.52 % [R, plan E1]; the warm start remains the single biggest verified win of the program.
9. Ladder finals: phase 2 best on both axes; every later phase worse on clean, phases 5–6 worse on both [M, plan E10; audit §8 table].

**D. Plausible causal explanations** (hypotheses, ranked)

- Phase-2's +4.3 pp robust over phase 1 = multiview trunk ensemble + 15 extra full-ε epochs (audit: CONTRADICTED as a recurrence attribution).
- Ladder regression after phase 2 = per-phase LR restart on an adapted trunk + non-identity-initialized new modules + (phases 5–6) a gaze distribution the belief loop was not trained for — an **optimization** confound, not an architectural one (audit §8 endorses plan E4/E10's inference).
- Random-gaze robustness edge = input-space stochasticity acting as a weak randomized defense, partially inflated by single-draw evaluation.
- Near-uniform Dirichlet = the evidential objective never being applied to this head during these phases, or gradient domination by the classification path, or a 15-epoch budget too short — unresolved (B11); a gradient-norm diagnostic would separate these.
- Precision ≈ 7 = unbounded parameterization with no pressure toward small values; the `tanh` bound on Δz is the only brake.
- 56-px substrate verdict: "too easy or too redundant for active acquisition to demonstrate value" (audit §7.5) — with the qualification that this indicts the substrate, not active perception.

**E. Unresolved scientific questions** — (1) recurrent integration vs matched multiview averaging (E1.1); (2) resolution vs representation as the binding constraint (E1.2 + NX-T0.4); (3) whether U_t is informative under correct metrics (E0.1 + realized-vs-predicted correlation); (4) whether the predictor earns its parameters (trained ablation); (5) why the evidential head collapsed to uniform (gradient diagnostic); (6) whether any of the belief machinery shows value where observations add information — nothing in the record speaks to this; (7) the exact split of the 7.6 pp PGD-50→EOT gap between step-size weakness and stochasticity (E0.2 controls); (8) whether the refinement block would activate with longer training (probably moot: bypassing already helps).

**F. Architectural hypotheses still requiring evidence** — evidence-dependent gated updates > fixed EMA; prediction error as a gaze signal; uncertainty-driven stopping; belief revision as measurable behavior; and the entire acquisition program on a native-resolution substrate (G3-HRV1 lineage). None is decided; all are testable under §14.

### 2.2 How the defects change the interpretation of the historical record

| Historical statement | Status after audit | Corrected reading |
|---|---|---|
| "recurrence_only is the best model: 63.44 clean / 34.77 robust" | Absolute robust number unestablished | Clean stands [M]. Robust was measured with an under-stepped PGD; the honest floor is bounded above by the strongest executed attacks (24.0–24.8), themselves caveated (fake APGD, n=300). Quote as "honest robust floor ≈ 22–25, pending E0.2" |
| "recurrence gives +4.3 pp robust" | CONTRADICTED as attribution | "Four trunk passes with fixed EMA + 15 extra full-ε epochs beat one trunkless linear pass" |
| "NLL barely moves under attack (gen1_core)" | Withdrawn (A1) | Metric was −log max p; anomaly is definitional. True-NLL recomputation is NX-T0.1 |
| "recurrence_only error AUROC 0.7708" | Withdrawn (A2) | Computed on NaN-corrupted scores |
| "Dirichlet head vs softmax" | NOT TESTED (A3) | Never scored; CLAIM-4 remains open with a corrected script |
| "PGD-10 ≈ PGD-50 ⇒ no gradient masking" (plan E14) | Unsupported (A5) | Both share the weak step rule; the inference is void. Masking status unknown until a strong-stepped PGD is compared against MI-FGSM/APGD on identical images |
| "Stochastic gaze is a resilient active defense" | CONTRADICTED | EOT (21.0–24.8) sits below the deterministic control's PGD-10 (34.2–34.8) |
| "T-scaling measured P1 up to T=8" | Withdrawn for `recurrence_only` (A13) | Only T ∈ {1,2,4} real; Gate L unevaluable as run |
| "AIS-v2 adaptive gaze ≈ fixed" | Understated | The policy learned nothing (A10); the comparison as trained is fixed-schedule vs a fixed deterministic scorer |
| "52 passing tests" | Correct count, weak evidence | No test covers model composition, gate-in-path, fusion output differences, EMA usage, or metric correctness (§11.9) — the exact gaps that let A7–A12 ship |
| "Gate A satisfied" | CONTRADICTED (L27) | AutoAttack never run; EOT under-powered |

---

## 3. Current architecture: verified behavior versus intended behavior

Condensed verified forward path (audit §3.1; abbreviations: trunk = DINOv2-S-layout 12-block ViT, 384-d, 16 tokens at 56 px):

```
x_src ─Resize(56,56)→ x56 ─┬─ GIST:  z₀ = mean_16tokens( patch_embed(x56) )     ← linear, NO trunk
                           │
                           ├─ recurrence_only:  for t=0..3:
                           │      crop_t = translate(x56, gaze_t)   (scale=1.0, ±8.4 px)
                           │      g_t    = trunk(crop_t).cls
                           │      z      ← 0.5·z + 0.5·g_t          [fixed EMA]
                           │      logits = cls_head(z)
                           │
                           └─ belief_*/gen1_core:  per step:
                                  d_t = EvidentialHead(z_t)          (α ≈ 1.0 ⇒ near-uniform)
                                  prec_t = softplus(MLP)             (unbounded; measured ≈ 7)
                                  crop_t = translate(x56, gaze_t)    (t=0 fixed; t≥1 AIS-v2 if enabled)
                                  (pooled_t, tok_t) = trunk(crop_t)
                                  pred_t = predictor({z,U,e}, gaze_t)
                                  err_t  = pred_t − tok_t            (B,16,384; spatial path detached)
                                  Δz_t   = update_net(z_t, meanpool(err_t))   (tanh, bound 0.1)
                                  z_{t+1} = z_t + prec_t · Δz_t
                                  logits = cls_head([z_T ; e_T]) + ev_readout(e_T)
```

| Component | Intended (docs) | Verified actual | Verdict |
|---|---|---|---|
| Gist | "low-res global prior, 16 tokens, gated parity fusion" | Linear projection of block-averaged frame; **no trunk**; spatial structure discarded by mean; gate never called | Defect (A7) + representation gap the docs never stated |
| Glimpses | "foveated samples that reveal detail" | Pure translations of the same 56-px tensor (scale=1.0); gaze-0 crop bit-identical to input; **no zoom exists in the running system** | Substrate defect (L3) |
| Recurrence | "predictive belief dynamics" | `recurrence_only`: fixed EMA, no predictor/error/precision/uncertainty | Mechanism–implementation mismatch (L2) |
| Belief update | "precision-weighted evidence integration" | z_t + prec·tanh(MLP([z; pooled])); first update overwrites prior (421 %); fusion discarded; spatial error detached | Trains, but not the documented mechanism (A8, A12, A14) |
| Predictor | "predict next observation; error drives update" | Connected; weak behavioral footprint (0/8 flips on bypass) | Present, value unproven |
| Precision | "bounded [1e-4, 1]" | softplus, 6.94–7.29 | Defect (A12) |
| Uncertainty | "Dirichlet epistemic doubt drives gaze/precision" | Near-uniform (U ≈ 0.99); never scored by the eval; readout contribution minor (12.8 % of logit norm, 0/16 flips) | Untrained mechanism + untested claim (A3) |
| Gaze policy | "learned active information sampling" | K=6 candidates (anchor + N(0,0.12) jitter); argmax at eval; only learnable parameter cancels; zero grads | Inert as a learned mechanism (A10) |
| Stopping | (Gen-1 heritage) | Does not exist in Gen-2; T fixed | Absent |
| Adversarial recipe | "TRADES 0.45·CE + 0.55·β·KL" | `w·(CE+β·KL)` = 0.55·CE + 1.375·KL; PGD-4 train; ε_norm 0.094 ≈ 5.4/255 | Code verified; docs wrong (now settled) |
| Eval stack | "registered harness, registered attack strengths" | Bespoke scripts; under-stepped PGD; EOT N=5; fake APGD; no AutoAttack; no large-ε; stochastic eval on gen1_core | Protocol defects (B) |

The single most consequential verified fact for architecture design is the **gist/glimpse asymmetry is compute, not information** (audit §7.3): the initial state skips the trunk entirely while every glimpse enjoys all 12 blocks. The system as built cannot distinguish "the prior is weak because it is low-resolution" from "the prior is weak because it is a linear probe." Any next architecture must fix this before it can interpret its own ablations.

---

## 4. Root-cause analysis of the principal failures

**F1. The recurrence claim dissolved.** Chain: the 5-tuple belief semantics existed in documentation (Technical Handbook Part II), while the code shipped a fixed EMA for the flagship phase and a partially-wired loop for later phases; no test covers `Gen2FoundationModel.forward` composition (audit §11.9 #1), so mechanism–implementation divergence was invisible; no matched controls existed (CTL1/CTL2 appear only as plan IDs), so the confounded gain was the only number available; the number was then narrated as mechanism evidence. Root cause: **mechanisms were claimed at the documentation level and validated only at the "does it run" level.**

**F2. The ladder regressed after phase 2.** Chain: every phase restarts the LR schedule from zero on an adapted trunk (plan E4 troughs −2.3 to −8.3 pp); new modules are not identity-initialized at boundaries (CE spike 5.16 at phase-4 start, plan E8); phases run only 15 epochs, insufficient to recover; phases 5–6 show the underfit signature (train loss 0.64–0.66 vs 0.51–0.55, plan E11). The audit concurs: an optimization confound, not an architectural one. Root cause: **a phase protocol that changes three things at once (schedule, init, data distribution) while the narrative reads the result as mechanism evidence.**

**F3. Robustness numbers inflated and fragmented.** Chain: attack helpers written for training-time monitoring were reused for headline evaluation (α = ε/steps); stochastic evaluation of `gen1_core` was compared against deterministic attacks as if commensurable; the registered strengths (EOT N≥20/50 steps, AutoAttack, large-ε) were never executed; no provenance fields in the JSONs. Root cause: **the instrument was never validated against a registered standard, so every absolute number inherited an unknown bias of several points.**

**F4. Calibration science never happened.** Chain: the metric code computes −log max p instead of true NLL (A1); fp16 probabilities underflow so entropy is NaN for one checkpoint (A2); the evidential flag is accepted and ignored (A3); meanwhile the head itself trained to near-uniform, likely because no evidential loss reached it in these phases (B11, unresolved). Four independent failures, each sufficient to void the conclusion, stacked in one experiment. Root cause: **no metric-correctness tests existed, and the experiment was interpreted before the metric was audited.**

**F5. The substrate cannot test the core hypothesis.** Chain: `fovea_size = H = 56` makes scale 1.0 — the foveation mechanism is capable of zoom (the code raises only if fovea > H) but was configured so that no zoom occurs; every observation is a translation of one downsampled tensor; the initial state additionally discards trunk processing. The plan's own Gates B and S were designed to catch exactly this but were scheduled behind months of mechanism work. Root cause: **substrate chosen for compute economy without an information-flow audit; the cheapest possible experiment (does gaze-0 equal the input?) would have caught it on day one.**

**Meta-cause.** All five reduce to one ordering mistake: mechanisms before instrument, instrument before substrate. The corrective principle for the next generation is the reverse order — **instrument → substrate → mechanism** — with kill criteria at each layer (§16).

---

## 5. The five immediate questions (A–E)

### A. Information acquisition versus resolution

The audit already settles one part by construction: on the 56-px substrate, *ineffective selection* and *ineffective integration* cannot be cleanly probed because observations cannot add information (L3); what T=1→4 measures is compute and averaging. What remains genuinely open is a two-way decomposition, plus one factor the audit surfaced but did not name a test for:

| Factor | Decisive evidence | Cost | Status |
|---|---|---|---|
| F-i. Input resolution bound | `G2-RES1`: frozen-trunk linear probe at 112 px (64 tokens) vs Gate-W anchor 58.52 %. Adopt if ≥ 63.5 % **[REQUIRES APPROVAL — proposed +5 pp, anchored to plan E3's saturation argument]** | 1–2 GPU-h | Not run (E1.2) |
| F-ii. Initial-state representation (NEW) | **NX-T0.4 trunk-gist swap**: eval-only; set z₀ := trunk(x56).cls in `recurrence_only` (and gen1_core), leaving everything else frozen. If clean/robust jump without retraining, the trunkless linear gist was the bottleneck, not resolution; if flat, representation of the initial view is not binding at 56 px | ≈15 min | Not in any registered plan; proposed here |
| F-iii. Selection quality | Existing: D7 (random ≥ AIS/fixed). Decisive version needs the new substrate + known target locations (E2.2 / G3-VS1) | — | Deferred pending substrate |
| F-iv. Integration quality | `G2-CTL2` matched-feedforward vs epoch-matched EMA (E1.1) + the gated arm (§13) | ≈20 GPU-h | Not run |

The 2×2 that resolves A completely: {56, 112} px × {linear gist, trunk gist} at probe level (F-i × F-ii, both cheap), then integration (F-iv) only on the winning cell. **Do not assume the answer in advance**: E1.2 flat + NX-T0.4 large would mean the "resolution ceiling" story is wrong and representation was binding all along — a materially different architecture decision.

### B. The actual value of recurrence

The audit verified `recurrence_only`'s update is `z ← 0.5z + 0.5g_t` — UC1's fixed convex combination with α = 0.5 and no learning (audit §12.1 point 2). So the design space to separate is exactly four rungs, and the matched experiment is:

| Rung | Model | Aggregation | Views | State carry |
|---|---|---|---|---|
| R0 | single-view | none | 1 | none |
| R1 | multiview feedforward | concat 4 CLS + gist → matched-param head (`G2-CTL2`, audit §7.4 spec) | 4+gist | none (order-invariant) |
| R2 | fixed recurrent | `z ← 0.5z + 0.5g_t` (recurrence_only mechanism) | 4+gist | fixed EMA |
| R3 | evidence-dependent recurrent | gated update consuming prediction error (A1 design, §8) | 4+gist | learned, input-dependent |

Controls held across all rungs: identical parent (`recurrence_only_final` for R1 vs R2 per the audit's E1.1 design — both arms share the full phase-1+2 history and differ only in the aggregation head; R0/R3 lineage stated per arm in §13), identical epochs (15 at full ε), identical LR protocol (no restart — a single continuous schedule from the parent), identical views, parameter matching per the audit's spec, seed 41 (≥3 for headline claims later, audit B8), identical eval protocol (post-T0 suite).

From existing checkpoints (eval-only, no training): R0 vs R2 at T∈{1,2,4} (D5 rows are valid); R2 vs the belief phases (unpaired, confounded — suggestive only); random/fixed/AIS at eval (D7). Require new training: R1 (CTL2), the epoch-matched R2 control, R3. The belief_* and gen1_core checkpoints **cannot** serve as R3 representatives: they were trained under the broken protocol with unbounded precision, inert fusion, and detached spatial error — they test "the loop as shipped," not "evidence-dependent updates." That is precisely why E1.1's control must be `recurrence_only` continued, not phase 3/4.

Success criterion (proposed, **REQUIRES APPROVAL**): R3 > R1 by > 2σ paired on robust accuracy under the post-T0 suite, and R3 > R2 by > 2σ, with no clean-accuracy collapse > 2σ. Failure on both: state carry adds nothing at matched views/epochs/params — the honest conclusion the program must be willing to publish.

### C. Is the adversarial evaluation trustworthy?

Not yet, and the minimum safe sequence below is ordered so that correctness is established before strength, and strength before headline claims. Diagnostic work happens in isolated scripts under `scratch/` / new `eval_scripts/rev2/` namespace; registered scripts and artifacts are untouched (brief §2C; audit §16.5 precedent).

**Step 1 — metric correctness (isolated code, fp32).** True NLL = CE(logits, y); entropy from `log_softmax` in fp32 (never `p·log p` on fp16 probs); AUROC with NaN assertion that fails loudly; keep the verified-correct ECE; add Brier and risk–coverage. Output: NX-T0.1 recomputation table for both surviving checkpoints (audit E0.1 spec, ≈10 min).

**Step 2 — attack correctness (unit-level, already half-done by the audit).** Verified correct and not to be re-litigated: ε-projection (measured 0.03100002 / 0.09400010), random start, gradient target, CE-on-logits loss family, EOT loss averaging, accuracy accounting (§5.2). To add: (i) full-fp32 eval forwards (the model currently emits fp16 logits under autocast; at these model sizes fp32 eval is affordable and removes a whole class of precision artifacts); (ii) per-step loss-decrease assertion along a PGD trajectory on a fixed image (attack optimizes); (iii) a fixed eval-time generator seed for stochastic models — a diagnostic-mode change, reported as such, with registered-strength EOT remaining the primary objective.

**Step 3 — attack strength (differential, identical images).** Strength is *not* established by correctness. Operational criteria, all **[REQUIRES APPROVAL]**: (i) large-ε collapse — at ε_norm ∈ {0.25, 0.5} accuracy → < 5 %, else the attack or the eval is suspect; (ii) plateau — increasing steps/samples/restarts moves accuracy < 1 pp (PGD-50 strong-stepped vs 100 steps; EOT N=10 vs 20 vs 40); (iii) cross-family agreement — strong-stepped PGD-50 within ~2σ of a real APGD-CE (implemented per Croce & Hein 2020 or via a maintained library) on the deterministic model; (iv) for stochastic models, EOT-N curve flattens. Until (i)–(iii) pass, no number from the suite is quotable.

**Step 4 — registered-strength re-run (audit E0.2, adopted unchanged).** Both surviving checkpoints × {clean, PGD-10, PGD-50 with α=ε/4 and 2.5ε/steps, MI-FGSM-100, real APGD-CE-100, AutoAttack (APGD-CE, APGD-T, FAB-T, Square) on the deterministic model, EOT N=20/50 steps for stochastic modes, large-ε check} on the fixed 500-image seed-42 slice plus a fresh 1,000-image slice (binomial σ ≈ 1.5 pp); per-image flip matrix between attacks; report old and new artifacts side by side, originals preserved.

**Step 5 — provenance fields in every new artifact**: git SHA, checkpoint SHA-256, config hash (new hash function, future runs only — audit B6), seed, command, dataset fingerprint. Backfilling old artifacts is forbidden.

**Trust boundary after Steps 1–4:** headline robust accuracy = minimum over the suite; obfuscation gap = old-protocol PGD-50 − new minimum, reported as the inflation of historical numbers; the relative ordering `recurrence_only` > `gen1_core` is expected to survive (it is internally consistent across attacks on identical images) but its magnitude will move.

### D. What should uncertainty do?

Four separable properties, each with its own gate — conflation of these is what made CAL2 uninterpretable:

| Property | Definition | Test | Current status |
|---|---|---|---|
| P-cal (calibration) | predictive probabilities match frequencies | ECE, true NLL, Brier, clean and under attack | Softmax: clean SUPPORTED (0.05), under attack CONTRADICTED (0.21–0.33). Evidential: NOT TESTED |
| P-rank (error ranking) | score ranks errors above correct predictions | AUROC of score vs error, paired | Softmax entropy: one value NaN-corrupted, one plausible-unpaired. U_t: never scored |
| P-dec (decision relevance) | conditioning a *decision* on the score improves a task metric | risk–coverage curves, selective accuracy at fixed coverage, abstention-with-cost | Not run |
| P-act (behavioral utility) | the score improves an *action policy* (gaze/stop) | realized-vs-predicted information correlation (audit §12.2), then policy ablation ± score input | Not run; U_t has almost no dynamic range to drive anything (α ≈ 1.0) |

Prerequisites before uncertainty drives any action: (1) NX-T0.1 correct metrics; (2) a score beating softmax max-prob on P-rank in a paired test; (3) nontrivial P-dec improvement at matched accuracy; (4) for the evidential head specifically, a gradient diagnostic separating "loss never applied" from "architecture cannot represent" (B11) and, if the former, a retrained head with the evidential objective actually active — or abandonment; (5) for any information-style score, the predicted-vs-realized correlation check (NX-T0.6) before it touches the policy. **The Dirichlet output is not assumed to be a Bayesian posterior**: it is a moment-matched approximation whose posterior interpretation is contested in the literature, and at α ≈ 1.0 it is empirically uninformative here. Interim design position (§11): softmax margin/entropy is the default action signal; the evidential head is an experimental arm behind Gate U; precision-as-write-weight is replaced by a bounded gate with per-step statistics logged.

### E. What does meaningful belief revision require?

The audit states the structure can express revision but no experiment measures it (§9.3) and sketches a forced-gaze test. The following makes it quantitative, adds the controls that make it falsifiable, and — importantly — it is **constructible today at eval-only cost** on the existing checkpoints, because composites are pasted *before* the 56-px resize.

**NX-T0.5 Revision probe R1.**
- *Construction.* For each val image: select a donor patch whose frozen-trunk CLS logit-difference for class B exceeds a margin (donor strength pre-measured, generator seeded, manifest released). Composite = base object A (supports true class A) + donor patch B pasted at a disjoint location. Two forced gaze orders: [A-region, B-region] and [B-region, A-region], T=2 (plus T=4 with [A,A,B,B] vs [B,B,A,A]).
- *Measurements.* Final argmax; p(A) trajectory per step; **Revision Index** RI = P(final=B | A-first) − P(final=B | B-first) (order-symmetry breaking); flip latency (steps from the contradicting observation to the posterior crossing); gate-response correlation for gated models (does the write gate open at the contradiction step).
- *Controls.* (i) Frozen-state control: Δz_t ≡ 0 for t ≥ 1 — later evidence must not reach the decision at all; RI should vanish. (ii) Feedforward control (R1 rung): order-invariant by construction; RI must be ≈ 0 — a metric sanity check. (iii) Shuffled-donor control: donor patches from mismatched classes verify the model treats pasted content as evidence at all. (iv) Cue-strength sweep: donor margin varied to separate "no revision capacity" from "insufficient cue strength."
- *What distinguishes revision from averaging.* A mean-pool/EMA state is order-symmetric (EMA weakly asymmetric through optimization artifacts only). Genuine revision shows order sensitivity aligned with cue strength, a sharp trajectory break at the contradiction (not gradual drift), and gate-opening in gated models at that step. Success (**REQUIRES APPROVAL**): RI > 0 with 2σ significance, aligned order asymmetry, frozen-state control at RI ≈ 0. Falsification: RI ≈ 0 in every existing checkpoint → revision behavior is absent in the shipped system; the belief-revision hypothesis moves to A1's R3 arm as a *target* behavior with a pre-registered probe, and the program says so in the paper.
- *Cost.* ≈1 GPU-h. *Status.* Exploratory on existing checkpoints; pre-registered version accompanies the R3 arm.

---

## 6. Candidate-mechanism triage

Eight mechanisms, each assessed on the eight required points. Dispositions: **CORE** (enters the next architecture), **ARM** (experimental arm behind a flag), **BASELINE** (control only), **DEFER** (needs prerequisites), **REJECT**. Guiding rule from the brief and the audit: distinguish the validity of an idea from the validity of its present implementation; include nothing merely because it is theoretically appealing.

**6.1 Contraction-constrained state updates (UC1).**
(1) *Problem:* unstable/unbounded belief updates; adversarial perturbation accumulating through a high-gain update map. (2) *Why now:* measured precision ≈ 7 makes the current update anti-contractive (audit §12.1 point 4); the first update is 421 % of ‖z₀‖. (3) *Math:* for frozen observations the convex update contracts at factor (1−α); with state-dependent gaze, contraction requires L_φ·L_π < 1, and a discrete argmax policy makes L_π undefined — the decisive gap; steady-state error under persistent perturbation is L_φ‖δ‖ **independent of α**: contraction of state does not imply robustness of the classifier (audit §12.1 point 5). (4) *Evidence:* none in-project; UC1 is already instantiated as the α = 0.5 EMA without gating. (5) *Smallest experiment:* audit §12.1's eval-only α-clamp sweep {0.05, 0.2, 0.5, 1.0} on `belief_with_f`, paired, plus a revision metric because the risk is suppressing revision. (6) *Failure mode:* over-constraint freezes the belief (at α = 0.05, ~60 glimpses to move); false security from bounded state. (7) *Cost:* ≈1 GPU-h eval-only; but only meaningful after the precision parameterization is corrected. (8) *Disposition:* **DEFER** (E2.4 in the audit's plan); do not claim robustness from contraction — the write-gate in A1 gives bounded, measured updates (u-gate ∈ (0,1), tanh candidate) without a separate contraction mechanism.

**6.2 Expected-information-gain observation selection (EIG1).**
(1) *Problem:* choose observations that most reduce decision-relevant uncertainty. (2) *Why now:* AIS-v2's score is entropy reduction under a *point* prediction — with α ≈ 1.0 the Dirichlet entropy is nearly flat, so the score degenerates toward a function of predicted feature magnitude (audit §12.2); and the policy it drives learned nothing (A10). (3) *Math:* EIG(a) = E_{o∼p(o|z,a)}[H(p(h|z)) − H(p(h|z,o,a))] requires a likelihood over outcomes; a point feature predictor is not a likelihood; no cost model exists. (4) *Evidence:* none; random gaze ≥ policy in every executed comparison. (5) *Smallest experiment:* NX-T0.6 predicted-vs-realized reduction correlation on a fixed checkpoint (for each candidate, predicted entropy reduction vs realized reduction after actually observing; ≈1 GPU-h) — the precondition the audit demands. (6) *Failure mode:* proxy-objective mismatch; RL instability; gains vanish against cheap ensembles/random. (7) *Cost:* the honest version needs a Gaussian (or similar) likelihood head over observations — a real modeling commitment, sensible only on A2 where observations carry new information. (8) *Disposition:* **DEFER** to A2-era; run NX-T0.6 now since it is eval-only; the term "EIG" must not be used for any point-prediction proxy.

**6.3 Decision-preserving state objectives (DPS1).**
(1) *Problem:* keep decisions/latents stable under perturbation. (2) *Why now:* phases 5–6 degraded; maybe loop instability? (3) *Math:* preserving clean predictions as targets trains the model to reproduce 37–42 % wrong teacher outputs; preserving z preserves a first-glimpse-dominated latent; margin variants collapse toward TRADES' KL term. (4) *Evidence:* the loss already contains a decision-preserving term — KL(p_adv‖p_clean) at weight 1.375 (audit §12.3); phases 5–6, where the KL dominates, are the *worst* performers. (5) *Smallest experiment:* none worth running before E0.2 (the outcome metric is untrustworthy). (6) *Failure mode:* stubbornness (anti-revision); loss competition (the project already rejected a 9-term λ family for exactly this). (7) *Cost:* low to run, high to tune honestly. (8) *Disposition:* **REJECT for now**; if revisited, the control is tuned-weight TRADES-KL, not plain CE.

**6.4 Adaptive stopping.**
(1) *Problem:* spend observations only when needed; metacognition claim. (2) *Why now:* none — T is fixed and the T-sweep is invalid (A13); there is nothing for a stopping rule to trade. (3) *Math:* stop ⇔ posterior confidence > τ or predicted marginal gain < λ·cost; needs a per-sample stop rule, a fixed budget comparator, and a cost model distinguishing glimpse cost (one trunk pass) from gist cost (one convolution) (audit §10.4). (4) *Evidence:* none; Gen-1's halting gate (`step6_validated_result`) was never set (L28). (5) *Smallest experiment:* corrected T-sweep (NX-T0.3 / E0.3) then an accuracy-vs-T frontier for fixed/adaptive/random. (6) *Failure mode:* premature stopping on confidently-wrong predictions (depends on Gate U); always-stop gaming. (7) *Cost:* low after prerequisites. (8) *Disposition:* **DEFER** behind substrate + monotone T-curve (Gate L′) + Gate U; the cost model (trunk-pass counting) is defined in §9 so the frontier is measurable later.

**6.5 Spatially resolved prediction errors.**
(1) *Problem:* mean-pooled error discards *where* surprise lives; gaze and revision need localized contradiction signals. (2) *Why now:* the shipped spatial path is gradient-dead (A14) and the fusion is discarded (A8) — the idea was never actually trained. (3) *Math:* e_t(s) = ô_t(s) − o_t(s) on the token grid; consumed either by attention-pooling in the update (spatial read) or by mean-pooling (the ablation); no new loss terms. (4) *Evidence:* none trained; BEL2/PRE4 arms never ran. (5) *Smallest experiment:* ARM-2 (mean-pooled error) vs ARM-2s (spatial read, gradient-intact), paired, same parent — §13. (6) *Failure mode:* predictor learns dataset-mean features so the error map is uninformative; extra parameters destabilize a 15-epoch arm (phase-boundary lesson). (7) *Cost:* modest (a 384→384 predictor head on tokens; the GRU input grows). (8) *Disposition:* **ARM** — spatial read is the A1 default *only if* it beats pooled in the paired arm; the gradient-integrity tests (nonzero grads through the error path; fusion modes produce different outputs) are preconditions, not afterthoughts.

**6.6 Evidential and alternative uncertainty estimates.**
(1) *Problem:* a calibrated, decision-relevant uncertainty signal. (2) *Why now:* the shipped head is near-uniform (α ≈ 1.0), was never scored, and its metric suite was broken (A1–A3). (3) *Math:* Dirichlet EDL (Sensoy et al. 2018) is a moment-matched approximation; its posterior interpretation is contested; entropy reduction on α ≈ 1 Dirichlets is uninformative (audit §12.2). (4) *Evidence:* clean softmax ECE 0.05 (good), adversarial ECE 0.21–0.33 (bad) — the one solid calibration fact; evidential: NOT TESTED. (5) *Smallest experiment:* NX-T0.1 (correct metrics, both scorers) + the B11 gradient diagnostic (evidential-loss gradient norm vs CE gradient norm on a batch). (6) *Failure mode:* retraining the head still yields near-uniform evidence (then the mechanism is wrong for this scale, and that is a publishable negative); or the head passes P-rank but fails P-dec (then it is a diagnostic, not an action signal). (7) *Cost:* eval-only for the diagnostic; a retrain is one arm. (8) *Disposition:* **BASELINE = softmax margin/entropy** (free, verified computable); **ARM = evidential head behind Gate U** (four-property ladder in §11); bench alternatives: temperature-scaled softmax, 2-member deep ensemble or MC-dropout (epistemic proxy) if P-rank needs a stronger scorer.

**6.7 Multiscale input processing.**
(1) *Problem:* the 56-px ceiling (plan E3, audit §14 point 1). (2) *Why now:* the substrate verdict makes resolution the first-class question; but Gen-2 rejected a multi-branch input pyramid for tokenizer confound reasons — correctly. (3) *Math:* the clean version is one tokenizer, two *input scales*: trunk-processed 112-px gist (64 tokens) + native-resolution 56-px glimpse crops from the source (16 tokens) — the shared-tokenizer invariant that `gist.py` got right is preserved; no second tokenizer, no pyramid inside the trunk. (4) *Evidence:* E1.2 (G2-RES1) is the decider; the code already supports true zoom (`foveal_sample` allows fovea < H; scale = 0.25 ⇒ 4× zoom) — it was configured not to use it. (5) *Smallest experiment:* E1.2 probe, then NX-T0.4 (trunk-gist swap) to separate representation from resolution. (6) *Failure mode:* DINOv2 pos-embed interpolation degrades at unseen scales; augmentation mismatch; compute (the 112-px gist costs ~4× a 56-px pass). (7) *Cost:* 1–2 GPU-h for the probe; the full A2 rebuild ≈ 2–3× per-epoch cost (unmeasured — §15). (8) *Disposition:* **CORE for A2, gated on E1.2**; the per-layer pyramid stays REJECTED.

**6.8 Foundation-model representations and teachers.**
(1) *Problem:* the representation is the one thing verified to work (Gate W 58.52 %, E1; warm start = the program's biggest win). (2) *Why now:* the trunkless gist discards exactly the asset the program already owns (NX-T0.4 tests this at zero cost). (3) *Math:* no new math; optionally distillation L_kd = KL(student ‖ frozen teacher) on the select split — deferred, no evidence of need. (4) *Evidence:* E1 [measured]; D2 feature-drift arm unrun (how far the trunk has moved from DINOv2 is unknown). (5) *Smallest experiment:* NX-T0.4 (eval-only swap); then D2-style drift diagnostic if the trunk LR question reopens. (6) *Failure mode:* teacher bias transfer (texture bias); frozen-trunk gradient starvation — the Gen-1 autocast lesson, now covered by the per-group grad-norm assertion. (7) *Cost:* zero for the swap; a DINOv2-B or DINOv3 trunk swap is 3.5×+ trunk compute — T4-hostile, and DINOv3's license/weights availability must be verified before anything depends on it (candidate for the A2 rebuild, nothing more). (8) *Disposition:* **CORE = keep DINOv2-S init and extend it to the gist (trunk-gist)**; larger/different teachers **DEFER**.

**Summary of dispositions.** CORE: trunk-gist (6.8), bounded gated update with error input (new, §8), single-tokenizer dual-scale substrate (6.7, gated on E1.2). ARM: spatial error read (6.5), evidential head (6.6), predictor-loss weight (§10). BASELINE: softmax uncertainty, fixed EMA, concat-feedforward, random gaze, fixed grid. DEFER: contraction (post-precision-fix), EIG (post NX-T0.6 + likelihood head), stopping (post substrate + T-curve), teachers beyond DINOv2-S. REJECT for now: DPS1, per-layer pyramids, pixel reconstruction, 9-term λ families, RAG memory (unchanged from plan §6).

---

## 7. Candidate architectures and trade-offs

Three genuinely distinct alternatives. They are not three brands of the same design: A0 is the instrument, A1 is the mechanism testbed, A2 is the substrate bet.

### A0 — RHAN-Min: the minimal corrected baseline

*Purpose:* the null every future arm must beat; also the cheapest carrier of the two verified repairs (trunk-gist, no dead modules).

*Forward.* x_src → Resize(56 or 112) → trunk on the **full view** → z₀ = CLS (trunk-gist — replaces the trunkless linear mean); optionally T=4 fixed translations, each through the trunk; aggregation = concat (or mean) of {z₀, g₁..g_T} → linear → GELU → linear head (parameter-matched to the comparison arm where required by the audit's CTL2 spec). No state, no predictor, no uncertainty machinery, no policy.

*Reuses:* backbone, shared tokenizer, recipe/optimizer, flags infrastructure. *Deletes:* refinement block, gist gate, fusion modes, EMA target, evidential head, AIS policy, precision MLP. *Cost:* T=4 ≈ phase-2 cost (~10 GPU-h per 15-epoch arm); T=1 ≈ 1/4 of that.

*Strongest falsifiable prediction:* matches or exceeds `recurrence_only` (63.4 clean; honest robust floor set by E0.2) with strictly fewer mechanisms. *If it fails*, the fixed EMA was doing real work beyond views — itself a finding that reshapes A1. *Principal risk:* none scientific; it is a control. *Disposition:* **BASELINE — build first among the three, smallest run.**

### A1 — RHAN-Rec: recurrence-centered, evidence-integrated (the recommended core)

*Purpose:* test whether **evidence-dependent state updating** — the actual scientific hypothesis the program was named for — beats matched feedforward and fixed-EMA aggregation, and whether it exhibits belief revision. Runs on the 56-px substrate first (comparability with all existing results and the audit's matched-controls doctrine), then on A2's substrate where observations carry new information.

*Forward (56-px instantiation; A2 deltas in §9):*
1. z₀ = trunk(x56).cls — a trunk-processed prior (the repair).
2. For t = 1..T (T = 4, fixed schedule during controlled arms):
   crop_t = translate(x56, ℓ_t) → trunk → (CLS_t, Tok_t).
   r_t = AttnPool(query = W_q·x_{t−1}, keys/values = W_k·Tok_t) — the belief *reads* the observation; "what you extract depends on what you believe."
3. ô_t = P_ψ(x_{t−1}, emb(ℓ_t)) — predict the pooled observation from the belief and the gaze.
4. e_t = ô_t − r_t — prediction error.
5. x_t = GRUCell([r_t ; e_t], x_{t−1}) — bounded, input-dependent state update (gates are the precision mechanism, now in (0,1) by construction; per-step gate statistics logged).
6. logits_t = C([x_t ; z₀]) — the prior persists in the readout (directly addressing the audit's first-glimpse-overwrite finding, §9.2).

*Trainable:* trunk (layer-wise LR, no restarts), GRUCell (~1.8 M params at d=384), predictor head, attention pool, classifier. *Reuses:* trunk + warm start, tokenizer, recipe, EOT module, loader. *Deletes:* precision MLP, UpdateNet v1/v2 as shipped, evidential head (kept only as the Gate-U arm), AIS policy (deferred to A2-era behind its gate).

*Strongest falsifiable prediction:* at matched views/epochs/params/seed, R3 (A1) beats R1 (concat-feedforward) and R2 (EMA) by > 2σ paired on robust accuracy under the post-T0 suite, **and** shows order-asymmetric revision on probe NX-T0.5 (RI > 0 with gate-opening at the contradiction step) while R1 has RI ≈ 0 by construction.

*Principal scientific risk:* at 56 px with 4 fixed translations, the error signal may be uninformative (the predictor can solve the task from dataset statistics), making R3 ≈ R2 for reasons the substrate, not the mechanism, causes. Mitigation: the paired design plus the revision probe plus the explicit fallback — if R3 ≈ R2 on 56 px, the mechanism question moves to A2 where it can actually be expressed, and the 56-px result is reported as a null with the substrate caveat. *Cost:* ~10–15 GPU-h per arm (four arms ≈ 40–60 GPU-h).

### A2 — RHAN-HR: substrate-first hybrid (the bet the audit's verdict points to)

*Purpose:* build the substrate in which the acquisition claims (P1–P4, CLAIM-2/5/7) are expressible: observations that can add information, gaze that can matter, revision that can be forced.

*Forward.* Source at native resolution (≥ 224 px). Gist: full view at 112 px → trunk → z₀ = CLS (64 tokens available for readout). Glimpses: 56-px crops **from the source** (scale = 0.25 in `foveal_sample` — true 4× zoom; the mechanism exists and was never used), each a 16-token trunk pass. Belief loop identical to A1. Gaze: fixed grid (control) → candidate-scoring arm → only ever compared against random. Stopping: fixed T; HALT arm only behind Gate L′.

*Strongest falsifiable prediction:* on small-target synthetic scenes (G3-VS1 lineage, targets at known locations, 384–512 px), loop+policy at a fixed trunk-pass budget beats (i) a single full-view pass at matched FLOPs, (ii) fixed-grid gaze, and (iii) **random** gaze — the audit's bar — with gaze-to-target alignment above chance. This is CLAIM-7 in the regime the plan always intended.

*Principal risks:* (i) DINOv2-S feature quality at 112 px / interpolated pos-embeds is unverified (E1.2 decides cheaply); (ii) compute: 64-token gist + 4×16-token glimpses ≈ 2× the trunk FLOPs of the 56-px system per image (unmeasured — must be measured before authorization, plan §13 rule); (iii) the ladder-rewrite temptation that broke Gen-2 — A2 must ship as **one run with flag-gated ablations**, never as a phase ladder; (iv) if E1.2 is flat, the resolution story is wrong and A2's premise weakens — then representation/training is the bottleneck and the honest sequence is A0 + training-recipe work first. *Cost:* ≈2–3× per-epoch vs 56 px ⇒ 15-epoch arm ≈ 20–30 GPU-h (estimate with ±40 % uncertainty; measure first).

### Decision matrix (who builds what, when)

| Evidence outcome | Read | Next build |
|---|---|---|
| E1.2 large gain (≥ +5 pp **[RA]**) and NX-T0.4 small | Resolution was binding; the trunkless gist was not the bottleneck | A2 pulled forward; A1 tested *on* A2's substrate; 56-px ladder reported as negative-control study (audit §17 fork 1) |
| E1.2 large and NX-T0.4 large | Both bind; trunk-gist is free capacity | A2 with trunk-gist from epoch 0 (default anyway) |
| E1.2 flat, NX-T0.4 large | Representation was binding, not resolution | A0 + trunk-gist lineage first; A2 deferred; recipe/LR work prioritized (audit §17 fork 2) |
| E1.2 flat, NX-T0.4 flat | Neither binds at 56 px; optimization/recipe bound | Audit fork 2 verbatim: re-derive claims on the corrected metric stack; no new mechanism proposals until training-recipe arms (REC1-style) report |
| E1.1: R1 ≈ R2 (concat within 2σ of EMA) | State carry adds nothing beyond matched aggregation | A1's motivation narrows to the gated/error mechanism alone; the EMA lineage is retired either way |
| E1.1: R1 ≪ R2 | Even fixed state carry helps | Both R2 and R3 survive into A2-era testing |

---

## 8. Mathematical formulation of the recommended design (A1)

**State.** d = 384. x₀ = W_p · CLS(trunk(I_gist)) ∈ R^d, where I_gist is the full-view input (56 px on the current substrate; 112 px under A2).

**Step t = 1..T** (controlled phase: T = 4, ℓ_t from the fixed schedule [(0,0), (0.3,0.3), (−0.3,0.3), (0,−0.3)] — the same schedule as `recurrence_only` for comparability):

- Observation: (CLS_t, Tok_t) = trunk(φ(x, ℓ_t)), Tok_t ∈ R^{16×d} on 56 px.
- Belief-conditioned read:
  r_t = AttnPool(q_t, K_t, V_t), q_t = W_q x_{t−1}, K_t = V_t = W_k Tok_t,
  r_t = Σ_s softmax_s(q_tᵀ k_s / √d) v_s ∈ R^d.
- Prediction: ô_t = W_out · GELU(W_in [x_{t−1} ; emb(ℓ_t)]) ∈ R^d.
- Error: e_t = ô_t − r_t.
- Update (GRU cell, Cho et al. 2014 — chosen because its gates *are* the precision mechanism, bounded by construction):
  u_t = σ(W_u [r_t ; e_t ; x_{t−1}] + b_u)          (write gate — replaces unbounded precision)
  ρ_t = σ(W_ρ [r_t ; e_t ; x_{t−1}] + b_ρ)          (reset gate)
  ĥ_t = tanh(W_h [r_t ; ρ_t ⊙ e_t ; x_{t−1}] + b_h) (candidate — tanh-bounded, unlike the shipped tanh(Δz)·7)
  x_t = (1 − u_t) ⊙ x_{t−1} + u_t ⊙ ĥ_t
  Bounded per-step change: ‖x_t − x_{t−1}‖ ≤ u_t ‖ĥ_t − x_{t−1}‖ ≤ 2 u_t — a stability prior that is *measured and logged*, never claimed as robustness (audit §12.1 point 5).
- Readout: logits_t = C([x_t ; z₀]).

**What each quantity is for.** r_t: the content of the new observation *as read under the current belief* (evidence integration). e_t: the surprise (drives the gate; optionally the gaze signal later — after NX-T0.6 and only under the honest name "error-seeking," never "EIG"). u_t: how much to write (the precision analog, bounded, logged per step). z₀ in the readout: the prior is never discarded (the audit's 421 %-overwrite failure mode is structurally excluded).

**Initialization (the phase-boundary lesson, plan E8/E10).** New modules enter identity-safe: W_out of the predictor zero-initialized (⇒ ô_t = 0, e_t = −r_t, harmless), GRU candidate layer zero-initialized with b_u biased to ≈ 0 (moderate gating, not saturation), classifier head warm-started from the parent's and kept pinned for the first 100 steps if the parent's head shape is preserved. One continuous LR schedule from the parent (warmup 1 epoch → cosine); **no per-arm restarts**.

**A2 deltas.** I_gist at 112 px (Tok count 64; pos-embed antialiased interpolation as already done for 4×4); glimpse crops at 56 px from the native source, scale = 0.25; attention pool unchanged; the same GRU; an optional spatial bank M_t (last-visit map for inhibition of return) deferred behind an ARM flag.

---

## 9. Dataflow and observation semantics

```
                    ┌────────────────────────────────────────────────────────┐
                    │  x_src (native res; A2)  or  x56 (A1 controlled phase) │
                    └───────┬───────────────────────────────┬────────────────┘
                            │ full view                     │ crop φ(x, ℓ_t)
                            ▼                               ▼
                     ┌─────────────┐                 ┌─────────────┐
                     │ trunk (S/14)│                 │ trunk (S/14)│   same weights
                     └──────┬──────┘                 └──────┬──────┘
                            │ CLS                           │ CLS_t, Tok_t (16 or 64 tokens)
                            ▼                               ▼
                       x₀ = W_p·CLS ────────┐        r_t = AttnPool(q=x_{t−1}, Tok_t)
                                            │               │
                     ┌──────────────────┐   │        ô_t = P_ψ(x_{t−1}, ℓ_t)
   ℓ_t (fixed grid)──►  emb(ℓ_t)          │   │               │
                     └──────────────────┘   │        e_t = ô_t − r_t
                                            │               │
                                            ▼               ▼
                              x_t = GRUCell([r_t ; e_t], x_{t−1})
                                            │
                                            ▼
                          logits_t = C([x_t ; z₀])   →  ŷ, s_t = margin/entropy (logged)
```

**Observation semantics — what a new observation contributes.** Three distinct quantities, never conflated (the audit's §10.3 distinctions are enforced by naming): (i) *r_t*, the belief-read content — what the current hypothesis makes of the new evidence; (ii) *e_t*, the surprise — how badly the belief predicted it; (iii) the *gaze* ℓ_t, which selects where evidence is drawn from (fixed in the controlled phase). The write payload into state is ([r_t ; e_t]) through input-dependent gates; the *decision* is read from [x_t ; z₀].

**Cost model (the currency for every budget curve).** cost(image, T) = 1 (gist trunk pass) + T (glimpse trunk passes), in trunk-pass units; the gist costs 1 pass and the classifier/predictor/GRU are negligible (~1.8 M params total added). A feedforward T-view comparator has the same cost by construction — that is what makes E1.1 and the A2 budget curves fair.

**Uncertainty in the dataflow.** s_t = margin (p₁ − p₂) and entropy of softmax(logits_t) are computed and logged at every step but drive nothing until Gate U passes. If the evidential ARM is enabled, U_t is an *additional logged scorer*, never an input to gaze or stopping, until the four-property ladder (§11) is cleared.

**Stopping.** None. T is a constructor argument fixed at 4. The stopping decision interface (a halting head emitting STOP given [x_t ; s_t ; cost-so-far]) is *specified* here so the cost model exists, and *deferred* behind Gate L′ (a valid, monotone T-curve on a substrate where T can exceed the number of distinct fixed views). This ordering is deliberate: adaptive stopping on the current substrate would stop at T=4 always, because nothing beyond 4 is observable.

---

## 10. Minimal training objective

Two terms, one ablatable. The adversarial term is the **code-verified** TRADES form (audit §3.2), not the documented one:

```
L = 0.55 · [ CE(f(x_clean), y) + β · KL( p(·|x_adv) ‖ p(·|x_clean) ) ]        β = 2.5 at the top rung
  + λ_pred · (1/T) Σ_t ‖ ô_t − sg(r_t) ‖² / τ²                                λ_pred = 0.1 [RA], ablation {0, 0.1}
```

- The stop-gradient on r_t means the predictor learns to predict observations without dragging the observation pathway toward its own predictions (the failure mode the audit's EMA-target defect hints at — the shadow encoder exists precisely because plain self-prediction is degenerate; here the degeneracy is prevented by the stop-gradient instead of an unwired EMA).
- τ = 1 with r_t LayerScale-normalized, or a running mean of ‖r_t‖ logged and frozen at the end of epoch 1 (choose one at implementation review; either is defensible, neither is load-bearing).
- The KL direction and both forward passes follow the verified training code (PGD-4 adversarial stream; ε-ramp unchanged: 10 clean epochs then rungs 0.031/0.062/0.094 for arms trained from `g2_gist_only_final`; 15 epochs at full ε for arms branched from `recurrence_only_final`). The ramp description in future docs must match the code (three rungs), fixing plan §4's error (audit §16.6 #7).
- **No other loss terms.** No evidential loss in the core arms (it moves to the Gate-U ARM with its own registered objective if the B11 diagnostic says the head can learn). No L_stab (the bounded gate replaces it as a structural prior). No stopping loss. No distillation.
- λ_pred = 0 is the registered ablation that answers "does the predictor earn its parameters" for the new design (audit §15 question 4); if ARM-2(λ=0.1) ≈ ARM-2(λ=0), the predictor term is dropped from the core and the error input e_t degenerates to a novelty feature — reported honestly.

**Objective-health logging (new, from audit lessons):** per-group gradient norms every N steps with a nonzero assertion for every registered module (the audit found gaze_policy and gist_encoder with zero nonzero grads — that assertion would have caught A7/A10 at step 1); loss-term decomposition logged per batch (the E12 discrepancy was discoverable from logs alone); per-step gate means u_t and ‖x_t − x_{t−1}‖/‖x_{t−1}‖ logged (the 421 % signature must be visible in training, not discovered by audit).

---

## 11. Uncertainty and calibration strategy

**Default scorer:** softmax margin m_t = p_(1) − p_(2) (primary) and entropy (secondary), computed in fp32 from log_softmax. Rationale: it is free, it is the control every alternative must beat, and the audit shows softmax clean calibration is already good (ECE 0.05).

**The four-property ladder (Gate U)** — a score may drive actions only after clearing, in order, on held-out data (select split once it exists; until then a fixed 1,000-image val slice disjoint from any model selection):

1. **P-cal:** true NLL, Brier, ECE (15 bins, equal-mass preferred) clean and under PGD-10 — the adversarial ECE collapse (0.21–0.33) is the problem any action signal will inherit, so calibration *under attack* is the binding condition, not clean.
2. **P-rank:** error-detection AUROC paired between candidate scorers on identical images, with a NaN assertion in the metric code itself.
3. **P-dec:** risk–coverage curves and selective accuracy at fixed coverage (e.g. 90 %) for score-conditioned abstention vs random abstention at equal coverage.
4. **P-act:** for information-style scores only, the predicted-vs-realized reduction correlation (NX-T0.6); for any score, an A/B in which the policy consumes vs ignores the score at matched compute.

**Evidential ARM.** Pre-conditions before any retraining: the B11 gradient diagnostic (is evidential evidence even trained by the current loss mix?) and NX-T0.1's corrected scoring of the *existing* head (it may already beat softmax on P-rank — the audit could not tell because the AUROC was NaN). If retraining is authorized, the head gets its own registered objective and the four-property ladder from scratch. If it clears P-rank but not P-dec, its role is diagnostic (offline error detection), not control — say so and stop.

**Under-attack calibration** is a first-class result in its own right: the clean-good/adversarial-bad ECE split (0.05 vs 0.21–0.33) is one of the few surviving quantitative findings, and it says the uncertainty *problem* the program cares about is real. Any architecture that improves adversarial ECE while holding clean ECE has a result worth publishing regardless of what the gaze policy does.

---

## 12. Observation selection and stopping (justified deferral)

**Selection: fixed grid in all controlled arms.** Justification: (i) the audit verified the policy learned nothing (A10) and that random ≥ learned in every executed comparison (D7); (ii) on the 56-px substrate selection cannot add information (L3), so any learned policy would be optimizing something other than acquisition — uninterpretable by construction; (iii) the audit's bar (beat *random*, not merely fixed) has never been cleared by anything. The candidate-scoring machinery is rebuilt in the A2 era with: K candidates from a coarse spatial bank, scores = predicted error magnitude under the honest name, the random baseline in the same run, and gaze-to-target alignment as the primary metric on synthetic scenes (Gate S lineage).

**Stopping: fixed T = 4.** The interface is specified (§9) so the cost accounting exists; the mechanism is deferred behind Gate L′, which requires a substrate where T can exceed the number of distinct views (A2's zoomed glimpses qualify; the 56-px grid does not) and a monotone accuracy-vs-T curve. Running adaptive stopping earlier would produce a system that always stops at T=4 — a degenerate policy that proves nothing.

**Inhibition of return (plan G2-MEM2):** deferred with selection; its natural home is the A2 policy arm (the spatial bank M_t), and its acceptance test stays as registered (re-fixation rate falls, accuracy not worse).

---

## 13. Experimental controls and ablations

**The three-controls doctrine** (audit §14 point 4): any recurrent update is compared against (i) matched epochs, (ii) matched views, (iii) matched parameters — three separate controls, all present in the arm table below. Plus: matched parent lineage, matched LR protocol (no restarts), matched eval protocol (post-T0), seed 41 with ≥3 seeds only for headline claims.

**Arm table (Stage T2; all 15 epochs at full ε; parent = `recurrence_only_final` unless stated; all new checkpoints under a fresh namespace, e.g. `checkpoints/audit_rev2/`, never overwriting existing artifacts):**

| Arm | Design | Controls it provides | Primary comparison |
|---|---|---|---|
| ARM-0 (`G2-CTL2`) | Concat 4 CLS + trunk-gist → matched-param head; no state | R1 rung: views + compute + params, no carry | vs ARM-1: isolates state carry |
| ARM-1 | `recurrence_only` mechanism, continued 15 ep under corrected protocol | R2 rung: epoch-matched EMA | vs ARM-0: state carry value; vs historical phase-2: restart-protocol effect |
| ARM-2 | A1 core (GRU + predictor, λ_pred = 0.1, pooled error) | R3 rung | vs ARM-0 and ARM-1 (both must be beaten, > 2σ paired, **[RA]**) |
| ARM-2a | ARM-2 with λ_pred = 0 | Predictor-earns-its-parameters | vs ARM-2 |
| ARM-2s | ARM-2 with spatial error read (gradient-intact) | Spatial vs pooled error (mechanism 6.5) | vs ARM-2 |
| ARM-3 | A0 minimal (T=1 trunk-gist, no views) | The R0-era null + trunk-gist value at minimal cost | vs Gate-W 58.52 and vs ARM-0 |
| (eval-only) NX-T0.4 | trunk-gist swap on `recurrence_only` | F-ii without training | free |

**Prohibited comparisons** (documented so they are not silently revived): phase-k vs phase-k+1 from the historical ladder (confounded by restarts, init, epochs); anything vs `gen1_core` as a mechanism representative; any robustness comparison across different n or different attacks; "best" checkpoints (selection bias — finals only, and only after the select split exists, `G2-INF1`, which must precede Stage T2).

**Pre-registered probes attached to Stage T2 arms:** NX-T0.5 revision probe (pre-registered version: thresholds fixed before ARM-2 finishes); the T-sweep T ∈ {1,2,3,4} on every arm (valid on the 56-px grid only to T=4 — stated on every plot); per-step gate statistics; EOT evaluation for any arm with stochastic components (none in Stage T2 — all controlled arms are deterministic by design, which is itself a simplification the audit would endorse).

**Exploratory vs preregistered.** Everything in Stage T0/T1 is exploratory re-execution or new diagnostics (no registered thresholds existed to violate). Stage T2 arms carry the proposed thresholds marked [RA]; if the review approves them, they become preregistered *for those arms only*, in a dated preregistration note — never retroactively.

---

## 14. Staged validation roadmap

Ranked by information-about-the-architecture-decision ÷ cost, extending the audit's §13 tiers with the two new probes and the architecture stages. Every stage has an entry condition and a kill criterion. **No Stage T2+ run is authorized before T0 and T1 report** (the audit's own rule: do not measure with a broken ruler).

### Stage T0 — instrument repair and cheap probes (eval-only; ≈1 GPU-day total)

| ID | Hypothesis | Control | Data / lineage | Training | Protocol & metrics | Compute | Success / failure | Confounds | Status | Decision informed |
|---|---|---|---|---|---|---|---|---|---|---|
| NX-T0.1 (= audit E0.1) | True NLL/entropy/AUROC tell a different story than reported | softmax max-prob as scorer | both surviving ckpts, 500 img seed 42 | none | fp32 metrics from collected logits+labels: NLL, ECE, Brier, AUROC, risk–coverage | ~10 min | descriptive; output = which metric is trustworthy | none | exploratory | Gate M; all calibration claims |
| NX-T0.2 (= audit E0.2) | Robustness ordering survives registered-strength attacks | old PGD-50 (α=ε/50) result | same 500 + fresh 1,000 slice; same ckpts | none | attack matrix per §5 Step 4; large-ε collapse; flip matrix | 1–3 GPU-h | ordering holds under min-over-suite / reverses or large-ε fails ⇒ attack still suspect | gen1_core stochasticity (EOT is its only fair objective) | exploratory re-execution of G2-D3/D8 | Gate A; every robustness claim |
| NX-T0.3 (= audit E0.3) | Gains saturate before T=4, or continue | — | same ckpts | none | T ∈ {1,2,3,4} (rec) and {1,2,4,6,8} (gen1_core), binomial σ printed | 30–45 min | monotone with non-overlapping 2σ ends / flat after T=2 ⇒ "more views" is a first-pass artifact | view-count = compute | exploratory; G2-D5 preserved as-is with defect documented | Gate L (current status: unevaluable → re-established) |
| NX-T0.4 (NEW) | The trunkless linear gist, not resolution, limits the initial state | unmodified `recurrence_only` forward | both ckpts, 500 img | none | z₀ := trunk(x56).cls; clean + PGD-10 | ~15 min | large jump ⇒ representation binding / flat ⇒ not binding at 56 px | the swap feeds the loop an out-of-distribution state scale (LayerScale stats may mismatch) — report both raw and norm-matched variants | exploratory | Question A factor F-ii; A0/A1 design default |
| NX-T0.5 (NEW) | Some checkpoint revises under contradictory evidence | frozen-state Δz≡0; feedforward order-invariance; shuffled-donor | composites from val (generator seeded, manifest) | none | RI, flip latency, trajectory shape per §5E | ~1 GPU-h | RI > 0 (2σ) with order alignment / RI ≈ 0 everywhere ⇒ revision absent in shipped system | donor strength (handled by sweep); composite artifacts at 56 px differ from natural images | exploratory | Question E; A1's target behavior |
| NX-T0.6 (= audit E2.5, pulled forward) | The AIS entropy proxy predicts realized information | — | gen1_core ckpt, K candidates × N images | none | predicted vs realized entropy-reduction correlation | ~1 GPU-h | correlation materially > 0 / ≈ 0 ⇒ EIG unreachable from this head | α ≈ 1 near-uniform Dirichlet | exploratory | Mechanism 6.2 disposition; Gate U P-act |

**Entry:** none. **Exit / kill:** if NX-T0.2's large-ε check fails, the attack is still suspect — fix before proceeding; if NX-T0.1/0.2 contradict each other on ordering, the stalemate itself is the finding and T1 waits.

### Stage T1 — the two deciders (≈25 GPU-h)

| ID | Hypothesis | Control | Data / lineage | Training | Protocol & metrics | Compute | Success / failure | Confounds | Status | Decision informed |
|---|---|---|---|---|---|---|---|---|---|---|
| NX-T1.1 (= audit E1.2 = `G2-RES1`) | 56-px input is the binding constraint | Gate-W anchor 58.52 % | ImageNet-100, frozen trunk | 5 ep probe head at 112 px (64 tokens) | Gate-W protocol; clean only | 1–2 GPU-h | ≥ 63.5 % [RA] adopts substrate-first / flat ⇒ representation-first | pos-embed interpolation quality; probe-head capacity | exploratory (threshold proposed) | **G-SUB**: A2 now vs later |
| NX-T1.2 (= audit E1.1 = `G2-CTL2` + epoch-matched control) | Matched feedforward recovers the phase-2 gain | ARM-1 (EMA continued, identical recipe) | both arms branched from `recurrence_only_final`; select split carved first (`G2-INF1`) | 2 × 15 ep | post-T0 suite; paired, 2σ | ~20 GPU-h | R1 ≈ R2 ⇒ state carry adds nothing / R1 ≪ R2 ⇒ carry helps | concat head has different readout capacity (parameter-matched per audit spec mitigates) | **requires authorization** (new arm, new namespace) | **G-REC**: retirement of the EMA lineage either way; A1 motivation |

### Stage T2 — mechanism arms (≈40–60 GPU-h; two at a time on Kaggle + Colab, ≈2 weeks wall-clock)

ARM-3 → ARM-0 → ARM-1 → ARM-2 (→ 2a, 2s) per the §13 table. Entry: Gates M and A passed; `G2-INF1` done; T1 reported. Kill criteria: ARM-2 fails to beat ARM-0 **and** ARM-1 by 2σ paired ⇒ the belief core does not advance to A2; the program reports the null and proceeds with A0/A2-architecture-only. All thresholds [RA] at preregistration time.

### Stage T3 — substrate (A2; ≈60–80 GPU-h)

Entry: G-SUB adopted (or G-REC outcome makes A2 the only remaining branch for acquisition claims). Single run, flag-gated ablations, no phase ladder. Sequence: s/epoch smoke on 2k subset (stop if projection exceeds quota — plan §12 rule 2) → full run → corrected T-sweep including T ∈ {6,8} (now valid: zoomed glimpses are distinct) → budget-curve mini (G3-BUD1-lite on ~2k synthetic search scenes) → policy arm vs **random** (Gate L′). Kill: if the 112-px trunk-gist underperforms the 56-px probe badly (feature-quality failure), fall back to 112-px-everything (glimpses at 112 too) at 4× compute — re-budget before authorizing.

### Stage T4 — gated mechanisms

Policy (error-seeking), stopping (HALT), evidential control, IOR — each behind its gate (L′, U, S). Nothing here is scheduled; everything here is earned.

---

## 15. Compute budget and resource risks

**Measured anchors [M].** ≈39 min/epoch at 56 px (plan §13, agent-estimated [R] — treat as ±30 %); 15-epoch arm ≈ 10 GPU-h; W1.5 final protocol 170–252 s per checkpoint on the eval GPU (audit §13) ⇒ the full T0 suite is minutes-to-hours, not days; 52-test suite runs in 6.44 s.

| Stage | GPU-h | Wall-clock (Kaggle ~30 h/wk + Colab, 2 workers) | Uncertainty |
|---|---|---|---|
| T0 (all six probes + AutoAttack) | ≈ 5–8 | 2–3 days | low (eval-only) |
| T1 (probe + 2 arms) | ≈ 25 | ~1 week | ±30 % (epoch time [R]) |
| T2 (4–6 arms × 15 ep) | ≈ 40–60 | 2–3 weeks | ±30 % |
| T3 (A2 run + sweeps) | ≈ 60–80 | 3–4 weeks | **±40 % — unmeasured 112-px epoch time; measure first (plan §13 rule)** |
| Headline seeds (≥3 × winning arm) | ≈ 30 | 1–2 weeks | only for the final claim (audit B8) |

**Risks.** (1) *Quota starvation mid-arm* — mitigated by the existing checkpoint/resume discipline; verify HF sync each session (plan §12 rule 5). (2) *A2 compute overrun* — the smoke-test stop rule is mandatory; the 112-px-everything fallback quadruples trunk cost, so the fallback itself needs re-authorization. (3) *Single-seed variance* — all Stage T2 comparisons are same-seed paired (valid for mechanism attribution); only headline claims need the 3-seed campaign, which is budgeted separately and only for the winner. (4) *Session timers (11 h)* — 15-epoch arms fit in 2 sessions; the resume banner must show epoch + checkpoint sha256 (plan §12 rule 4). (5) *AutoAttack runtime* on CPU-bound components — cap Square queries at 500 and report the cap. (6) *The hidden cost of doing it right* — Stage T0+T1 ≈ 1 GPU-week before any new training; this is the price of the audit's findings and it is cheap relative to the 68 GPU-h already spent on a ladder whose conclusions did not survive.

---

## 16. Decision gates

| Gate | Condition to pass | Blocks | Kill behavior |
|---|---|---|---|
| **G-M** (metrics) | NX-T0.1 complete; metric suite frozen with NaN assertions; fp32 eval mandatory | every calibration claim; T1 | — |
| **G-A** (attack floor) | NX-T0.2 passes large-ε collapse + plateau + cross-family agreement **[RA criteria]** | every robustness claim; headline numbers | if unfixable, robustness claims are retired and the program pivots to clean/corruption/calibration axes |
| **G-SUB** (substrate) | NX-T1.1 ≥ 63.5 % [RA] | A2 authorization | flat ⇒ A2 deferred; recipe-first path (audit fork 2) |
| **G-REC** (recurrence) | NX-T1.2 pair reported, paired 2σ | A1's R3 arm; any "recurrence" language in writeups | R1 ≈ R2 ⇒ EMA lineage retired; mechanism claims require ARM-2 to restart from evidence |
| **G-CORE** (belief core) | ARM-2 > ARM-0 and > ARM-1 (> 2σ paired [RA]) **and** NX-T0.5-style RI > 0 with gate signature [RA] | belief-state claims anywhere; A2 belief loop | fail ⇒ A2 ships as multiview + (later) policy only; belief machinery shelved with the negative result documented — the paper is the negative-control study + substrate result, which the plan itself rates a graduation-level contribution (§8.1 outcome 1) |
| **G-U** (uncertainty) | four-property ladder cleared in order | any uncertainty-driven action (gaze, stopping, precision) | evidential head fails P-dec ⇒ diagnostic role only |
| **G-L′** (T-scaling on A2) | monotone accuracy-vs-T with valid distinct glimpses; policy beats **random** paired [RA] | HALT, EIG-style selection, IOR | fail ⇒ fixed-grid A2 remains the deliverable; selection declared open |

**Gate discipline.** Thresholds marked [RA] are proposals requiring approval at review; once approved they are preregistered for the named arms only. A failed gate stops the affected line of work; only the owner advances a gate (plan rule 5). No gate may be passed by analogy ("it worked on the other substrate") — each substrate re-runs its own gates.

---

## 17. Unresolved questions and the evidence that would answer them

| # | Question | Decisive evidence | Where |
|---|---|---|---|
| UQ1 | Is the 56-px input the binding constraint? | NX-T1.1 probe number | T1 |
| UQ2 | Is the trunkless linear gist binding independently of resolution? | NX-T0.4 swap | T0 |
| UQ3 | Does recurrent integration beat matched multiview averaging? | NX-T1.2 pair (R1 vs R2) | T1 |
| UQ4 | Does evidence-dependent updating beat both? | ARM-2 vs ARM-0/1 paired + revision probe | T2 |
| UQ5 | Is U_t informative under correct metrics? | NX-T0.1 scorers + B11 gradient diagnostic | T0 |
| UQ6 | Does the predictor earn its parameters? | ARM-2 vs ARM-2a | T2 |
| UQ7 | Why did the evidential head collapse to uniform? | gradient-norm ratio evidential:CE; loss-application audit of phases 3–6 | T0 diagnostic |
| UQ8 | What fraction of the 7.6 pp PGD-50→EOT gap is step size vs stochasticity? | NX-T0.2's step-size ladder on identical images | T0 |
| UQ9 | Is the random-gaze robustness edge genuine or single-draw obfuscation? | EOT over gaze draws: random vs fixed on `recurrence_only` | T0 add-on to NX-T0.2 |
| UQ10 | Does anything transfer to a substrate where observations add information? | A2 + corrected T-sweep + budget curve | T3 |
| UQ11 | How far has the trunk drifted from DINOv2? | D2-style feature-drift diagnostic (registered, unrun) | optional, T1 |
| UQ12 | Would the refinement block activate with longer training? | probably moot (bypass helps +0.6 robust); close by deletion | closed unless A0 surprises |
| UQ13 | Training-seed variance of every number above | ≥3-seed campaign, winner only | post-T2 |
| UQ14 | What is `gen1_core`'s defined delta from phase 5? | weight hashing + trainable-set diff (plan B4, still owed) | bookkeeping, T0 |

---

## 18. Implementation plan (explicitly separate; nothing authorized)

For review completeness only — execution requires its own authorization, in this order, and none of it touches registered files, checkpoints, or artifacts:

1. **Diagnostic scripts (T0).** `scratch/audit_rev2/` (following the audit's precedent): `metrics_fp32.py` (NX-T0.1), `attack_suite.py` (NX-T0.2 — imports a maintained AutoAttack implementation or implements APGD-CE per Croce & Hein 2020 with unit tests against known-correct behavior), `tsweep_fixed.py` (NX-T0.3), `trunk_gist_swap.py` (NX-T0.4), `revision_probe.py` + composite generator with seeded manifest (NX-T0.5), `proxy_realized_corr.py` (NX-T0.6). All read-only against checkpoints; all artifacts written to new `report/rev2/` directories with full provenance fields (§5 Step 5).
2. **Test-suite additions** closing audit §11.9's seven gaps: model-forward phase semantics; gate-in-path; fusion-mode output differences; EMA-usage; selection-changes-with-temperature; ε-projection; metric correctness (synthetic known-answer tests, including the confidently-wrong shift case from the audit's diagnostic).
3. **Infrastructure.** Config-hash function (future runs only, audit B6); select-split carve (`G2-INF1`); new checkpoint namespace `checkpoints/audit_rev2/`.
4. **Training arms (T1/T2)** behind new flags in a new namespace; parent/lineage/config recorded per arm; the loader change for A2 (native-resolution source) is the only change that would touch shared data code and is gated behind a default-off flag.
5. **Engineering effort estimate:** T0 diagnostics ≈ 1–2 agent-days; tests ≈ 1 day; arms ≈ 0.5 day each beyond the GPU time; A2 loader + flags ≈ 2–3 days including the smoke-test harness.

---

## Appendix A — Brief-requirement mapping

| Brief requires | Delivered in |
|---|---|
| 1. Executive recommendation | §1 |
| 2. Interpretation of the forensic findings (six-way classification; defect→interpretation changes) | §2.1, §2.2 |
| 3. Current architecture: verified vs intended | §3 |
| 4. Root-cause analysis | §4 |
| Questions A–E | §5A–§5E |
| 3. Candidate mechanisms (8-point assessments) | §6.1–§6.8 |
| 5. Candidate architectures and trade-offs | §7 (+ decision matrix) |
| 6. Mathematical formulation | §8 |
| 7. Dataflow and observation semantics | §9 |
| 8. Minimal training objective | §10 |
| 9. Uncertainty and calibration strategy | §11 |
| 10. Observation selection and stopping | §12 |
| 11. Experimental controls and ablations | §13 |
| 12. Staged validation roadmap | §14 |
| 13. Compute budget and resource risks | §15 |
| 14. Decision gates | §16 |
| 15. Unresolved questions | §17 |
| 16. Implementation plan, separated | §18 |

## Appendix B — What this proposal does not claim

- Not claiming the current architecture is validated — the audit shows the opposite, and this document adopts it.
- Not claiming recurrence causes any observed gain; the only attribution-safe statement is "four trunk passes + fixed EMA + 15 extra full-ε epochs beat one trunkless linear pass" (audit §8).
- Not claiming the evidential head improves calibration or error detection — CLAIM-4 is NOT TESTED, and this document keeps it that way until NX-T0.1 and the Gate-U ladder run.
- Not claiming EIG is implemented or that any entropy proxy is information gain — the term is reserved for the likelihood-based quantity (§6.2).
- Not claiming state contraction or bounded gates guarantee adversarial robustness — bounded updates are a measured stability prior, nothing more (§6.1, §8).
- Not treating plan.md as immutable — several of its statements are corrected here against the audit (§2.2, §16.6 of the audit); equally, plan.md's registered IDs, lineage, and integrity constraints are preserved and reused wherever they are sound.
- Not inventing preregistered thresholds — every numeric decision rule is marked [REQUIRES APPROVAL].
- Not hiding unresolved questions — §17 lists fourteen, including three this document's own designs depend on (UQ1–UQ3).

