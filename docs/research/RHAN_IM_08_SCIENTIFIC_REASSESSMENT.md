# RHAN scientific reassessment — read-only investigation

**Scope.** Architectural and scientific assessment before freezing Gate 1 or committing to RHAN-IM as RHAN's final identity. Nothing in the repository, the checkpoints, the preregistration draft, the model card or any experiment was modified. No code was run on a trained network. No dependency or external code was installed, imported, copied or integrated. This sentence describes the **original investigation**, which was read-only. After you approved it, this document was added to the repository by **documentation-only commits** (see the next paragraph); those commits changed nothing but this one file under `docs/research/`. No checkpoint, model, experiment, implementation or Gate-1 preregistration was changed by this assessment or by those commits.

**Branch status (as of the latest commit; please read).** `redesign/rhan-im` was built on `d81c4df` of `origin/stage0-closeout`. Its commits, oldest first: `c317703` and `fad8233` (the earlier RHAN-IM redesign work, dated 10 Nov, before this assessment); `943751f` (adds this document, 725 lines) and `8cc7365` (sampler result and class-split correction): both touch only `docs/research/RHAN_IM_08_SCIENTIFIC_REASSESSMENT.md` (`git diff --stat fad8233..8cc7365` lists that one file); and the commit that carries this correction, which is the branch HEAD when you read this (a commit cannot contain its own hash: run `git log -1 --oneline`). The branch is **not pushed** from the environment that made these commits: the push was denied by the environment's permission check and I did not try to route around it, so the remote has none of them until you push from your own machine. The bundle `rhan_im_redesign_fad8233.bundle` that I sent earlier preserves the branch only up to `fad8233` and does not contain this document or the later commits. Your original working tree was lost in a container reset; everything I read for this assessment came from a bundle restored into a scratch directory.

**Revision note (second version, after your review).** Changes against the version you read: (A) the Gate-1 "adequacy" claim now says exactly what Gate 1 tests; (B) the Learn-then-Test exchangeability unit is analysed (§E.5); (C) the 115-vs-102 discrepancy is narrowed with exact expectations from the real sampler rules (§E.1); (D) R1's sufficiency is derived, not just simulated (§C.3); (E) independent, global and hierarchical validity structures are separated and tested against each other (§C.4); (F) acquisition under uncertain validity is stated as **open** (§C.5); (G) the H-BR acceptance criteria add a no-revision baseline and a perfect-validity oracle, and the cue test now uses correct, flipped, sham and missing cues with direction-of-movement and accuracy effects (§D). Your final research structure is in §F. No implementation is approved or done; Gate 1 is not frozen.

**Third revision (after your run of the checks).** (1) **Sampler discrepancy resolved** (§E.1): your 50-seed run of the real sampler gives a size-4 mean of 114.88, matching the exact expectation 114.9, and the draft's 102 reproduces deterministically as a low-tail draw (seed 7). (2) **I got the class-split mechanism wrong, and §E.3 is corrected:** I analysed WNID order, but the script sorts *directory names*; on your list the actual split is 27 organisms / 23 artifacts (targets) against 17 / 33 (backgrounds), a moderate imbalance, not the 44-vs-0 confound I described. The recommendation (a recorded seeded or stratified split) stands on weaker grounds, stated in §E.3. Results (1) and the class-list counts come from your run and I have not reproduced them.

**Evidence labels used throughout**

| Tag | Meaning |
|---|---|
| **[M]** | mathematical identity or unit test in the RHAN-IM dossier; no trained network involved |
| **[S]** | my synthetic simulation from this round (scratch scripts outside the repo, fixed seeds, reproducible); a property of the stated generative model, **not** of any trained network |
| **[R]** | I read the source, README or page myself this round |
| **[P]** | a published claim I read; I did not reproduce it |
| **[L]** | located (title/abstract/listing) but not read in full |
| **[U]** | recalled from my own knowledge, not re-fetched; check before quoting |

Nothing here is a measured property of RHAN. No real-network d′, z-ROC slope, VIF, ρ or calibration number exists yet. Where a number appears, it carries a tag that says where it came from.

---

## 0. Answers in one page

**Your question.** *Can RHAN become a system that actively acquires visual evidence, understands the reliability and limitations of that evidence, maintains structured hypotheses about a scene, and revises its interpretation when the evidence changes — with measurable uncertainty, computational discipline and robustness?*

Clause by clause:

| Clause | Status | Why |
|---|---|---|
| actively acquires visual evidence | **Designed, untested on the real sensor** | C1 is built for this. Whether looking closer pays on the frozen `ff_112` trunk is exactly what Gate 1 measures. |
| understands the reliability of that evidence | **Half-supported** | C1 measures *sensor* reliability (d′ per unit, a fixed property of the sensor under calibration conditions). It has **no model of when a given look is unreliable** (blur, occlusion, corruption, distribution shift). That is a different quantity: d′ as a function of observable conditions, plus a validity variable per look or region. |
| maintains structured hypotheses about a scene | **Not supported by C1 as built** | C1's hypothesis space is *one object, one place* (H = 1 + N·C = 25,601). A multi-object scene is a different hypothesis space. This is a research problem, not an extension. |
| revises when the evidence changes | **Not supported as built; addable in a restricted form** | C1's state is a sum. A sum cannot say *which* earlier look to stop trusting. Clipping bounds the damage of one look but leaves it in the sum forever. A latent-validity layer with provenance (R1 below) is exact Bayes over a richer latent and does not change C1's foundation. |
| measurable uncertainty, computational discipline, robustness | **C1's strongest side** | Calibration-by-construction under a measured sensor, Learn-then-Test stopping, bounded influence. All conditional on Gate 1 passing, and all currently simulated. |

**Two kinds of "the evidence changes".** These need different mechanisms, and the project has been using one phrase for both:

1. **The scene is static; the meaning of earlier evidence changes** (a look is later found to have been blurred, occluded, mislabeled as in-region, or adversarial). The right object is a **validity variable with provenance**. Exact Bayesian inference over (hypothesis, validity). No forgetting is needed, and forgetting would be wrong.
2. **The scene itself changes between looks** (object moves, appears, is removed). The right object is a **dynamic latent with process noise** — a Kalman-type filter, or mutable slots. Forgetting is correct here.

C1's accumulated posterior (in "sum" mode, without the per-unit cap, which makes the recursion order-dependent) is the information-form filter for a **static** latent: contributions add and nothing is forgotten. C2's delta rule is the same filter with the covariance frozen and process-noise-like forgetting, no precision state and no calibration [S: exact algebraic identity checked to 4×10⁻¹⁶ in `belief_revision_toy.py`]. They are not competing answers to one question; they answer questions 1 and 2.

**Decision (detailed in §F).** Option 4, staged: keep C1 as the **sensor and acquisition foundation**; treat belief revision as a **separate, falsifiable hypothesis** (H-BR) to be tested first in simulation and then on the real sensor after Gate 1; test a C2-style mutable slot only against it as a controlled alternative; treat the C1+C2 hybrid as a hypothesis, not a design. **Do not freeze C1 as RHAN's final identity yet**, because the identity question depends on Stage 3, which cannot run before Gate 1.

**The external projects and papers do not justify changing direction.** They contribute two research principles worth adopting (provenance-tagged records separate from interpretation; hypothetical branches never write durable state), one idea worth a controlled experiment (contradiction-triggered local repair, implemented from our own math), and one test design that matters a great deal (representing reliability ≠ relying on it). Details in §11–§15.

**Gate 1.** It can be frozen as a *sensor-qualification* gate once four things are settled (§E): the class split (a moderate organism/artifact imbalance and a fragile sort rule, §E.3), the image-pool and LTT-unit decision (§E.5), the median-slope blind spot (§E.2), and the wording of the 2-of-3 rule (§E.4). The sampler discrepancy is resolved (§E.1). `n_test = 6,000` is adequate for the two gated scalars under my synthetic assumptions and should **not** be doubled; what it cannot support is listed in §E.

---

## A. Scientific direction assessment

### A.1 Capability classification

Classes: **Fundamental** = removing it removes RHAN-IM's reason to exist; **Extension** = addable without changing the foundation, with its own falsifiable hypothesis; **Alternative** = a different foundation that should be compared, not stacked; **Control** = exists to make a claim falsifiable.

| Capability | Class | Why |
|---|---|---|
| Measured sensor model (d′ measured, not learned; likelihood-adequacy test) | **Fundamental** | The one property that distinguishes C1 from "ViT + extra blocks"; also its first falsifier. Gate 1 tests the **likelihood and accumulation assumptions C1 uses** (equal-variance Gaussian per-unit evidence with measured d′, additivity of token evidence after the VIF correction, sum vs best across looks). It does **not** test whether Bayesian inference is applicable; that is a modelling choice, not a measurable property of a trunk. |
| Information-acquiring looks (recurrence earns cost only where information is acquired) | **Fundamental** | The thesis. The old recurrence-only run reprocessed shifted views of the *same* 112 px input, so its negative result says nothing about recurrence-as-perception. |
| Posterior over (what, where) + accumulated-precision map | **Fundamental** | State of the ideal searcher; closed-form policy and certificates depend on it (derived for the validity-free model only; see §C.5). |
| Learn-then-Test stopping | **Fundamental, separable** | Model-free; remains valid even if the Bayes claims fail (dossier §9 switch conditions). Evaluate independently. |
| Bounded influence (κ clip, per-unit cap) | **Fundamental to robustness claim, separable** | Needs local evidence; the unit of influence is the whole window look because a ViT window mixes its tokens. Evaluate independently from LTT. |
| Sensor reliability under varying conditions (d′ as a function of observable covariates) | **Extension (Stage 1b)** | Not in Gate 1. Without it, "measured sensor" means "measured under the Gate-1 stimulus distribution only". |
| Per-look validity (robustified likelihood, order-invariant) | **Extension** | Exact Bayes over a binary latent per look; no provenance needed [S]. |
| Region-level validity + provenance (region-indexed sufficient statistics) | **Extension (the minimum for revision)** | Needed as soon as corruption persists across looks. |
| Persistent per-look ledger | **Optional** | Only needed if retraction must be arbitrary rather than per-region. Cost scales with looks. |
| Mutable object-centric scene representation (slots, C2-style) | **Alternative / Extension, conditional** | Justified only if (i) scenes change between looks or (ii) content is continuous and not additive. Neither is demonstrated to matter. |
| Multi-object structured hypotheses | **Open research (Stage 4)** | Different hypothesis space; no support from current math. |
| Conditional computational specialization (CivoE-like) | **Unjustified at present** | RHAN-IM already has a computation discipline (policy + stopping). Revisit only if distinct uncertainty types demand distinct actions (Stage 5). |
| Frequency-coupled recurrence (C4) | **Control** | The recompute family; the correct negative control for the acquisition thesis. |
| Fast-weight field (C2) | **Alternative** | Compared on identical criteria in §B. |
| Persistent parametric knowledge edit (WSP-like) | **Out of scope** | Different problem: semantic memory in weights, not perceptual belief. §11–13. |

### A.2 The five-way separation, and what C1 has for each

| Quantity | Definition used here | C1 today | Gap |
|---|---|---|---|
| **Evidence** | what a look delivered: per-unit scores z_{u,c} plus the look's identity (where, when, scale, window) | scores are standardised and *summed into per-unit contributions*; identity is discarded after the sum | provenance |
| **Reliability** | how much a unit of evidence deserves trust: sensor d′ (long-run, calibrated) and look/region validity (this look, now) | d′ measured at Gate 1 (long-run only) | validity; d′ conditional on observable conditions |
| **Belief** | the current distribution over hypotheses | posterior over H_0, H_(b,c) | multi-object; hypotheses about *which looks are valid* |
| **Uncertainty** | what the belief says it does not know | posterior risk; ignorance Φ(−√D_b/2) per cell | uncertainty about reliability itself (the model's own assumptions) |
| **Revision** | change of belief that is not mere accumulation: reinterpreting earlier evidence | none | a mechanism that can lower the weight of evidence already absorbed |

**Does a posterior suffice?** For accumulation, yes: exact under the specified sensor model [M]. For revision, no — and this is a statement about *state*, not about Bayes. The posterior over hypotheses is a function of the sum of contributions; two different histories with the same sum are indistinguishable, so the state cannot answer "what would my belief be if look 3 had been invalid?". Bayes is not at fault. The latent in the model was too small: it contained the hypothesis but not the validity of each look. Enlarge the latent to (hypothesis, validity) and exact Bayes revises correctly [S]. This is why no non-Bayesian mechanism is needed (your constraint) — and why the mechanism's *cost* is state, not mathematics.

### A.3 Can C1 identify when earlier evidence should no longer be trusted?

Not as built. What it has is **bounded influence**, which is not belief revision:

| Scenario (synthetic [S], N = 20,000, K = 10 classes, 8 looks) | M0 plain sum (C1) | M1 clip κ=1 | M2 per-look validity | M3 region ledger + validity | M4 delta slot (β = 0.3) |
|---|---|---|---|---|---|
| **A** clean | acc 0.874, ECE 0.003 | 0.849, 0.061 | 0.864, 0.062 (prior π=0.8 mis-specified) | — | 0.747 |
| **B** isolated corrupted looks (p = 0.25) | 0.624, **0.146** | 0.620, 0.045 | **0.652, 0.004** (π = 0.75 correct) | — | 0.499 |
| **C** persistent corruption in region R (looks 1–3) + reliability cue | 0.518, **0.281** | 0.529, 0.145 | 0.539, 0.150 | **0.742, 0.007** | 0.639 |
| **D** as C, no cue | 0.525, **0.276** | 0.540, 0.138 | 0.543, 0.149 | **0.721, 0.003** | 0.644 |

(acc = accuracy; second number = ECE; logloss and conditional accuracies are in the scratch output. M1 κ=2 and κ=4 are between M0 and κ=1.) All methods are *given* the true generative parameters; the table shows what each **state representation can do at best**, not what a learned system will do.

What it shows, and what it does not:

1. **Isolated misleading looks (B):** a per-look validity mixture (M2) fixes calibration (ECE 0.146 → 0.004) and slightly improves accuracy, needs **no ledger**, and is order-invariant. Clipping helps logloss/ECE (0.146 → 0.045) but not accuracy.
2. **Persistent corruption (C, D):** neither clipping nor per-look validity helps (accuracy 0.518–0.543). Only a **region-indexed** state with a **shared** validity variable recovers (0.742 / 0.721); its own readout of "region R is corrupt" has AUROC 0.985 (C, with cue) and 0.948 (D, from disagreement with later valid looks alone). **Provenance is the missing ingredient**, not a cleverer sum.
3. **A recency baseline is surprisingly strong when corruption is early (C, D: 0.639 / 0.644 vs 0.52–0.54) and clearly worse when evidence is clean (A: 0.747 vs 0.874).** It wins by forgetting, not by knowing what to distrust, and it is uncalibrated. Any real claim of belief revision must beat it at matched clean-data accuracy, otherwise it is just forgetting.
4. **Identifiability:** if every look comes from the corrupted region and no cue exists, no method can detect it (information-theoretic). Scenario D succeeds only because *later valid looks disagree*.
5. **Cost of being robust when nothing is wrong:** M2 with a mis-specified prior π = 0.8 loses ECE on clean data (0.003 → 0.062). Reliability machinery must be calibrated, or it hurts.

### A.4 Representing, expressing, relying — and a stable representation with changing beliefs

Two distinctions from the two papers you attached, applied to RHAN:

* **Representing reliability ≠ relying on it** [P: Fel et al., *Shaping & Monitoring Models through their Internal Representations*]. A system can encode a quantity (a probe reads "this region is corrupted" at AUROC 0.985) and still make its decision from the corrupted evidence. The check that matters is *interventional*: change the reliability signal and see whether the **decision** changes. In the toy this is a counterfactual cue flip: replacing the cue with a draw from the opposite regime changes M3's decision in **21.1 %** of scenes (accuracy 0.752 → 0.625). M0 changes in 0 % — by construction, since it has no cue input; it also represents nothing about reliability. For an exact Bayes layer, reliance is guaranteed by the algebra; **for a learned system it must be measured**, and §D.3 designs that measurement.
* **A stable representation of the world is not a fixed belief** [P: Balestriero & LeCun, JEPA/Platonic representation]. Their theorem is about what an *encoder* converges to when positive pairs are two observations of one scene; it says nothing about whether downstream beliefs should change. RHAN's hypothesis is therefore well-formed: *representation (the sensor and its measured likelihood) stays fixed while beliefs (posterior over hypotheses and validity) are revised*. It is testable by comparing sensor-output statistics (frozen) with belief trajectories (changing) under evidence reassessment.

Separate the four layers the papers imply, because each has its own failure mode and test:

| Layer | Question | RHAN component | Test |
|---|---|---|---|
| Representation formation | does the sensor give stable, comparable evidence? | frozen `ff_112` trunk + head; measured d′ | Gate 1; Stage 1b |
| Evidence management | does each unit of evidence carry provenance and reliability? | **absent** (R1 region-indexed statistics + cue) | Stage 3 |
| Belief maintenance | are hypotheses mutable under revised evidence? | posterior (+ validity latent) | Stage 3 |
| Representation-level verification | do decisions *use* reliable evidence? | **absent** | counterfactual reliance, §D.3 |

---

## B. Architecture comparison (identical criteria)

All four are judged on the same rows. "Evidence in hand" is deliberately the last row: it says what has actually been shown, by which kind of evidence.

| Criterion | **C1** RHAN-IM (posterior + precision map) | **C2** fast-weight scene field | **C4** frequency-coupled two-clock recurrence | **C1 + C2 hybrid** (hypothesis) |
|---|---|---|---|---|
| Core state | posterior over H_0 and H_(b,c) (25,601 hypotheses) + per-unit accumulated precision; **no recurrent weights** | matrix S, d² = 147 k floats per image at d = 384; ~0.44 M projection parameters | slow state s, fast state f | slots with content *and* a measured precision; update = Kalman gain from measured d′ |
| Does recurrence **acquire** information? | **Yes**, through a look policy (new pixels at native resolution) | Only if looks are new pixels; on a fixed input it recomputes | **No**: LP and HP of one image are functions of the same pixels | Yes if driven by C1's policy |
| Update rule | Bayes (LLR add); per-unit cap makes it order-dependent | S ← S + β(v − Sk)kᵀ | coupled recurrence, stability by ρ(J) < 1 | Kalman gain β = P/(σ² + P kᵀk) with σ² = 1/d′² |
| Knows what it has **not** seen | **Yes** (precision map, ignorance Φ(−√D/2)) | **No** (linear state cannot represent "unseen") | No | Yes, via the precision |
| **Provenance** (which look contributed what) | **No** (summed away) | **No** (rank-1 writes mixed into S) | No | Only if writes are ledgered (information form permits exact retraction of a logged write) |
| **Retract** a specific earlier observation | No | Partial: overwrite along the same key; overlapping keys interfere (Cadence documents exactly this limit for its own delta-rule stores [R]) | No | Yes, **if** logged: subtract kkᵀ/σ² and kvᵀ/σ² |
| Can the state say a hypothesis is *dynamic* (scene changed)? | No (static latent) | Yes (forgetting by constant gain) | No | Yes (process noise) |
| Calibration of uncertainty | Exact **if** the sensor model is right [M]; ECE 0.002–0.007 with true d′, 0.08–0.13 when d′ is 30 % too high [dossier; simulated] | None for free | None | Inherits C1's, if the slot's likelihood is the measured one |
| Risk-controlled stopping | Yes (LTT) | No | No | Yes if the posterior is exposed |
| Cost | one look = 2.912 GFLOPs (112-px window, `ff_112` trunk); Bayes update ≈ 0.6 % of a look; closed-form policy ≈ 0.15 %; 38.5 k new trainable parameters with the trunk frozen | one outer product per token; 0.44 M parameters | ≈ 2 trunk passes per step plus filters | C1 cost + slot state; d² per slot is the issue: O(K·147 k) floats at d = 384 unless a diagonal/low-rank form is shown to suffice (untested) |
| Main assumptions | equal-variance Gaussian sensor (z-ROC slope ∈ [0.8, 1.25]); within-look dependence correctable by ÷√VIF; independent looks or "best" mode | linear associative memory is adequate; key geometry (orthogonality) | the frequency–robustness link holds (dataset-dependent) | all of C1's, plus that a linear slot content in the right feature space exists |
| Characteristic failure | d′ at native window ≤ gist d′; slope far from 1; vacuous certificate; **cannot retract** | interference, recomputation only, no calibration | acquires nothing; full-band attacker not bound by the frequency story | doubled state; may add parameters without adding information when the scene is static |
| **Evidence in hand** | math tests + simulations [M][S]; **no real-network number** | none beyond the algebra [S: identity with frozen-covariance Kalman] | none; old recurrence-only negative result concerns *recomputation* | none |

### B.1 What the old selection matrix does and does not show

`RHAN_IM_03_CANDIDATES.md` §8 scored C1 33/35 vs 22/35 for C2 and C4. Those were **my judgements against criteria I wrote**, one of which ("defensible hypothesis — acquires information") is satisfied by definition by C1 and failed by definition by C4. That is fair as a reason to *start* with C1, not as a finding that C1 is best. On the criterion this reassessment adds — *can it revise earlier interpretation* — C1 scores low (cannot retract), C2 scores partial (overwrites, with interference and no calibration), and the hybrid is an untested hypothesis. I therefore do not re-score numerically; that would add false precision.

### B.2 C4 as a control, stated correctly

C4 is the right control for the *acquisition* thesis only if it matches compute: same number of trunk passes, same pixels. But "recompute" is not worthless — a test-time-augmentation ensemble gives a small, known gain. The prediction should be *gain from acquisition > gain from matched recompute*, with a TTA-ensemble arm included, not "recompute gives nothing". The old recurrence-only result (shifted views of the same 112 px input) does **not** show that recurrence cannot function as perception; it shows that reprocessing the same pixels did not pay on that trunk.

### B.3 C1 + C2: what the hybrid would actually be

For a **static** scene, C1's per-unit accumulation already *is* an information-form Kalman filter over class evidence; a C2 slot placed on the same class-score space would reproduce C1's u_contrib with a weaker, frozen gain. A hybrid therefore adds value only if one of two things holds: (i) the scene **changes** between looks (process noise is real, forgetting is correct), or (ii) slot content is **not additive class evidence** (pose, part-whole, appearance attributes in feature space). Neither has been shown to matter for RHAN's current target (pasted squares; static canvas). Until Stage 3 shows a measurable gap that a mutable slot closes **and** R1 cannot, the hybrid is unjustified complexity.

---

## C. Belief-revision formulations (technically distinct)

Notation: looks t = 1…T, regions r (a window position/scale/time tag), hypotheses h = (b, c) or H_0, per-unit scores z, measured detectability d_u, region validity v_r ∈ {valid, corrupt}.

### R1 — Latent-validity Bayesian layer (exact Bayes over a richer latent) — **recommended to test first**

* **State.** Per region r: the region's per-unit, per-class contribution array (the region-indexed analogue of C1's `u_contrib`), a validity posterior q_r, and a prior π_r. Memory ≈ (units per region) × C per region: for a native window 64 × 100 = 6,400 floats; 25 windows + gist ≈ 5 × C1's 32,000-float `u_contrib` **[derived from shapes; not measured]**. Storing two full 25,601-wide hypothesis vectors (valid and corrupt) per region instead would be ≈ 1.3 M floats, ≈ 42 × C1's `u_contrib`, and is not needed.
* **Provenance.** Region tag = (window position, scale, time/order). Required to re-mix a region after a later cue; sufficiency is derived in §C.3.
* **Reliability estimation.** Two parts: (a) sensor d′ as a function of observable covariates (blur/noise/contrast indicators from a small separate head or from image statistics), fitted on calibration scenes with *injected, known* corruptions — **new measurement, Stage 1b**; (b) validity prior π_r and cue likelihoods from the same calibration. Nothing is learned inside the evidence path; the likelihood stays measured.
* **Contradiction handling.** Not a special mechanism: if region R's looks favour hypothesis h and later valid looks favour h′, the Bayes factor between "R valid" and "R corrupt" shifts q_r. Scenario D [S] recovers 0.721 accuracy from disagreement alone.
* **Update rule (independent-validity case, "R1-IND").** P(h | data) ∝ P(h) · Π_r [ π_r L_r^valid(h) + (1 − π_r) L_r^bad(h) ], where L_r^bad is a corruption model (in the toy, a decoy: an informative signal for a wrong class, averaged over the other classes). This product is exact **only if** region validities are independent a priori and region data are conditionally independent given (h, v). It is **one** dependency structure, not a universal rule: §C.4 separates independent, global and hierarchical validity and shows that assuming the wrong one can be worse than no revision. Both L_r^valid and L_r^bad are computable from the stored region contributions (§C.3 gives the sufficiency derivation). **Per-look validity (M2) is R1-IND with one look per region**; it needs no ledger and is order-invariant.
* **Uncertainty propagation.** The posterior is over (h, v); the decision risk uses the marginal over v. Acquisition under uncertain validity is **open** (§C.5): I do not claim that the Koopman gain, the ignorance map or the original closed-form policy remain optimal, near-optimal or unchanged once validity is a latent.
* **Complexity.** Update O(units·C) per look plus a re-mix O(R·units·C) when a cue arrives; memory as above. **Compute not measured.**
* **Failure conditions.** (1) The corruption model does not cover the real corruption (an "uninformative" model only protects against *vetoes*, not against false positives: mixture log(π e^{LLR} + 1−π) caps the negative side only); (2) mis-specified π costs calibration on clean data (ECE 0.003 → 0.062 in the toy); (3) unidentifiable when all looks share the corrupted region and no cue exists; (4) **new attack surface**: an attacker who controls the cue can make the system retract *true* evidence. This must be tested in Stage 5, not assumed away.
* **Tests.** Scenarios A–D on the real sensor using injected, known corruptions; validity AUROC; ECE after revision; counterfactual cue flip (§D.3); comparison with the recency baseline (M4) and clipped sum (M1) at matched clean-data accuracy.

### R2 — Reliability-gated mutable slots (C2-style, in information form)

* **State.** K slots; per slot an information matrix Λ_k = Σ_j k_j k_jᵀ/σ_j² and vector η_k = Σ_j k_j v_jᵀ/σ_j² (equivalently a mean with precision), plus a write ledger of (id, key, value, σ_j). Content μ_k = Λ_k⁻¹η_k.
* **Provenance.** The write ledger. **Retraction is exact for linear-Gaussian**: subtract the logged write's contribution. Without the ledger the delta rule cannot retract (it has only the current matrix).
* **Reliability estimation.** σ_j² = 1/d′² from the measured sensor (the thing C2 lacks); a validity-dependent inflation factor.
* **Contradiction handling.** Normalised-innovation test (χ²) on each new write: if the innovation is too large for the slot's predictive covariance, choose one of (a) hold the write, (b) open a private hypothetical slot and let evidence decide which survives (Cadence-like hypothetical branch), (c) inflate Λ's inverse (re-open the slot), (d) retract the logged writes of the lower-trust region.
* **Update rule.** Information-form additions; with a constant gain it degenerates to the delta rule [S: identity checked; the delta rule's gain stays near p/(σ² + p‖k‖²) while the Kalman/RLS gain decays — after 50 steps, error vs truth 1.081 for the delta rule and 0.486 for RLS].
* **Uncertainty propagation.** Predictive covariance from Λ⁻¹; no calibration claim unless the likelihood is the measured one.
* **Complexity.** Λ is d × d: **147 k floats per slot at d = 384** — K slots is K × that; a diagonal or low-rank form is untested.
* **Failure conditions.** Overlapping keys interfere (Cadence's own docs state: a fixed matrix cannot hold contradictory values or nearly parallel keys [R]); linear read-out too weak for real object content; a mis-specified σ² makes retraction confidently wrong; a dynamic-scene assumption (forgetting) is *wrong* for static scenes (toy: 0.747 vs 0.874 on clean data).
* **Tests.** Same scenarios; plus a changing-scene variant (the object moves between looks), which is the one setting where R2 has a reason to beat R1.

### R3 — Dempster–Shafer discounting (comparator only; **not recommended**)

Discount source r by α_r and re-mix with total ignorance. For simple-support evidence this is the same algebra as a mixture likelihood — R1 per look — with the added problems of a non-probabilistic combination rule and no calibration theory [U]. It is listed because it is the classic non-Bayesian alternative and the comparison closes the question: if it differs numerically from R1 it is because it is mis-specified, not because it knows more.

### C.1 Verdict on C1

| Question | Answer |
|---|---|
| Does C1's posterior + accumulated-precision map **support** belief revision with the current math? | **No.** It supports accumulation and bounded influence. |
| Can revision be **added without changing the foundation**? | **Yes**, in the restricted forms R1 (per-look and per-region validity) and R2 (ledgered slots) — same measured sensor, same Bayes core, a richer latent and a provenance store. Needs one new derivation (ignorance under uncertain validity). |
| Does it **require a different architecture**? | Only for **multi-object structured hypotheses** (Stage 4) and possibly for **dynamic scenes with continuous content**; neither is a requirement for the revision claim in the restricted single-object setting. |

### C.2 Theory to cite (all [U] unless noted)

Huber (1965) robust sequential tests for contaminated models; Saha & Ramdas robust e-values for adaptive contamination [L]; the switching-state-space / interacting-multiple-model filters and the normalised-innovation gating used in robust Kalman filtering; Dempster–Shafer discounting; AGM belief revision and truth-maintenance systems (Doyle; de Kleer) for the logical analogue of provenance-based retraction. None was re-read this round beyond identification.


### C.3 Sufficiency of R1's retained statistics (derivation)

**Setting.** Regions r = 1…R with data z_r (the per-unit, per-class scores of that region's looks); hypothesis h = (b, c); validity v ∈ {0, 1}^R (1 = corrupt); a late reliability cue s_r with likelihood g(s_r | v_r), conditionally independent of (z, h) given v. Assumptions:

* **(S1)** conditional independence of region data given (h, v): p(z_1…z_R | h, v) = Π_r p(z_r | h, v_r);
* **(S2)** valid model p(z_r | h, 0) = p₀(z_r) e^{ℓ_r(h)}, with ℓ_r(h) = Σ_{u∈U_r} F_{bu}(d_u z_{uc} − d_u²/2), the C1 likelihood ratio against the null density p₀ (a fixed linear within-region VIF correction is absorbed into the definition of a_r below); corrupt model p(z_r | h, 1) = p₀(z_r) m_r(h), where m_r is any function of z_r and h (the decoy model in the toy is m_r(h) = (C−1)⁻¹ Σ_{c′≠c} exp Σ_u F_{bu}(d^m_u z_{uc′} − (d^m_u)²/2));
* **(S3)** d_u > 0 and d^m_u > 0, known and fixed.

**Proposition 1 (sufficiency of region-indexed statistics).** Let a_r = {d_u z^{(r)}_{uc} − d_u²/2}_{u∈U_r, c}. Under S1–S3, P(h, v | z, s) depends on the data only through ({a_r}, s), and z_r ↔ a_r is a bijection, so storing a_r loses nothing.

*Proof.* By Bayes and S1, P(h, v | z, s) ∝ P(h) P(v) Π_r p₀(z_r) [ (1 − v_r) e^{ℓ_r(h)} + v_r m_r(h) ] g(s_r | v_r). The factors p₀(z_r) do not involve (h, v) and cancel in the normalisation. ℓ_r(h) is linear in a_r (the footprint F is known). m_r(h) is a function of z_r = (a_r + d²/2)/d, hence of a_r. By the Fisher–Neyman factorisation, a_r is sufficient. A cue enters only through g(s_r | v_r), a factor that does not involve z; re-mixing region r when its cue arrives therefore needs only a_r. ∎

**Proposition 2 (what is not sufficient).**

1. C1's summed statistic Σ_r a_r (`u_contrib`) is sufficient for the validity-free posterior (v ≡ 0) and for no model with P(v ≠ 0) > 0: two datasets with the same Σ_r a_r can have different posteriors. *Explicit instance* (K = 3, two single-look regions, independent-validity mixture, π = 0.7): both datasets have Σ a_r = (2, −1, −1); posteriors (0.528, 0.236, 0.236) and (0.722, 0.139, 0.139) [S, hand-checkable].
2. Any nonlinear per-unit operation applied **after** summing destroys sufficiency: the per-unit cap, and "best" mode (a max across looks). Clipping each look before summation preserves a sufficient statistic only for the *clipped* likelihood, which is a different model.
3. Storing the two H-vectors (ℓ_r(·), log m_r(·)) per region is sufficient for the specified pair of models, but not for a later change of the corruption model, of π, or of d^m. The a_r form is.
4. Memory: |U_r| × C per region (64 × 100 = 6,400 floats for a native window) [derived from shapes].

**What this does and does not show.** It shows the arrays *can* reconstruct both likelihoods after a late cue, relative to the model. It does not show the model is right: (i) S1 is an assumption (Gate 1 measures the cross-look correlation ρ of standardised scores; if ρ is not small, the product over regions is approximate and sufficiency holds for the *model*, not the data); (ii) if real corruption leaves signatures the model's p(z_r | ·) does not use (pixel-level blur statistics, say), the cue head must read them from the pixels, not from a_r. **Numerical confirmation, not proof:** posterior from the stored arrays versus a brute-force computation from the raw looks (loops over validity configurations, regions, looks, decoys) with a late cue: max absolute difference 8.9 × 10⁻¹⁶ over 40 scenes [S].

### C.4 Validity dependency structure: independent, global, hierarchical

All three use the **same stored a_r**; they differ only in the sum over v, i.e. in P(v):

| Model | P(v) | Cost of the sum |
|---|---|---|
| **IND** | v_r i.i.d. Bernoulli(0.3) | product of R two-term mixtures, O(R·\|H\|) |
| **GLOBAL** | one latent g; v_r = g for all r (P(g = 1) = 0.3) | two products, O(R·\|H\|) |
| **HIER** | g ~ Bernoulli(0.4); P(v_r = 1 \| g = 1) = 0.6, P(v_r = 1 \| g = 0) = 0.1; regions independent given g | two products with g-dependent π, O(R·\|H\|·\|g\|) |

All three have marginal P(bad) = 0.3, so none gets an advantage from the marginal. Setting [S]: K = 10, R = 4 regions × 2 looks, valid d = 1, persistent decoy d^m = 2 per corrupted region (independent across regions), N = 20,000, no cue. Rows: the structure that generated the data; columns: the model fitted. Accuracy / log-loss / ECE:

| Truth \ model | IND | GLOBAL | HIER | oracle (true v) | no revision (C1 sum) |
|---|---|---|---|---|---|
| **IND** | **0.620 / 1.146 / 0.007** | 0.454 / 2.006 / 0.099 | 0.615 / 1.178 / 0.031 | 0.768 / 0.674 / 0.007 | 0.528 / 1.995 / 0.259 |
| **GLOBAL** | 0.593 / 1.459 / 0.086 | **0.641 / 0.983 / 0.006** | 0.603 / 1.225 / 0.070 | 0.652 / 0.876 / 0.006 | 0.612 / 2.781 / 0.232 |
| **HIER** | 0.607 / 1.230 / 0.027 | 0.504 / 1.741 / 0.073 | **0.611 / 1.191 / 0.004** | 0.749 / 0.710 / 0.009 | 0.548 / 2.222 / 0.255 |

Readings: (1) the correct structure is best calibrated in every row (ECE 0.004–0.007); (2) **the wrong structure can be worse than doing nothing**: truth IND with a GLOBAL model has accuracy 0.454 against 0.528 for no revision, and truth GLOBAL with an IND model 0.593 against 0.612 (though log-loss and ECE are better than no revision); (3) IND is a tolerable approximation to HIER (ECE 0.027); (4) the GLOBAL ceiling is low because a global corruption leaves no valid region at all. **Consequences.** The product over regions must not be presented as universally valid. The structure is an empirical question: choose it on a labelled corruption suite by held-out log-likelihood, always include "no revision" as a candidate, and report all three. With a real network none of these three may hold; a fourth (spatially correlated validity, for instance) may be needed.

### C.5 Acquisition under uncertain validity — open

The Koopman policy and the ignorance map were derived for evidence that is valid with the measured d′. Once validity is a latent variable, at least four things are no longer settled: (i) the per-cell precision D_b is random rather than a fixed sum, so the ignorance factor Φ(−√D_b/2) is not the right object; (ii) the value of re-looking a region whose validity is uncertain includes information about v itself, a term the closed-form gain does not contain; (iii) Lemmas 4–5 (patch certificates, minimum looks) are statements about worst-case evidence under a valid sensor; when validity is inferred, an adversary who controls the cue changes *what is discounted*; (iv) the closed-form-versus-exact-EVOI check (within 1.6 pp) was run for the validity-free toy only.

**Position.** I do not claim that the original acquisition policy remains optimal, near-optimal or unchanged after introducing uncertain evidence validity, and Stage 2 results (validity-free) say nothing about acquisition under H-BR. **Design consequence:** evaluate H-BR first with a *fixed look schedule* (acquisition off), so the two hypotheses are not confounded; derive the modified value-of-information term, and test closed-form against exact lookahead in the toy, before combining them.

---

## D. Revised experimental roadmap

Rules for every stage: a hypothesis is stated before data; falsification criteria are written before the run; a negative result is kept and reported; compute is stated in FLOPs and parameters (you asked me not to estimate GPU hours); "simulated" never substitutes for "measured on the network". **Frozen `ff_112`** = the existing seed-17 checkpoint (sha256 `d00fb750…32bc`) read as it is, with at most the small linear head fitted on top; **new training** = anything else.

### D.1 Stage table

| Stage | Hypothesis (falsifiable) | Baselines | Ablations | Measurements | Falsified if | Network |
|---|---|---|---|---|---|---|
| **1 Sensor reliability / acquisition validity** | **H1a** token scores from `ff_112` + 384→51 head satisfy the equal-variance Gaussian model (z-ROC slope ∈ [0.8, 1.25], null sd ∈ [0.85, 1.2]). **H1b** centre d′ ≥ 1.5 and corrected block d′ ≥ 2.0. **H1c** native/gist d′ ratio ≥ 1.5. **H1d (new, Stage 1b)** d′ degrades measurably and *predictably* with observable quality covariates (blur, noise, contrast, occlusion). | gist-only look; ff_112 feedforward | per-size/per-opacity cells; "sum" vs "best" | Gate 1 quantities **plus** per-position slope distribution, per-class d′, d′ vs covariates | the dossier switch conditions (§9 of `03`): any of H1a–c fails → heteroscedastic likelihood / dense window CE / temporal looks / fallback to calibrated selective classification. H1d: if d′ is not predictable from observable covariates, reliability *cannot be inferred from the input*, and only look-to-look disagreement remains as a signal. | **Frozen `ff_112`**; linear head fitted (19.6 k parameters at 50 classes; 38.5 k at C = 100) |
| **2 Adaptive acquisition vs fixed coverage** | **H2** the Koopman policy beats fixed-lattice coverage by ≥ +2 pp at 4 looks, 95 % CI excluding 0, at matched FLOPs. | fixed lattice; random looks; entropy-greedy; saliency-greedy; **matched-FLOPs recompute control (C4 + TTA ensemble)** | policy off; ignorance map off; exact EVOI vs closed form (toy only; the exact lookahead is 3.17 GB unchunked at B = 64) | accuracy vs looks; accuracy vs FLOPs; calibration | gain < +2 pp or CI includes 0; **or** the recompute control matches it | **Frozen `ff_112`** + head |
| **3 Reliability and belief revision (H-BR)** | **H3a** per-look validity (R1-IND, one look per region) improves calibration under isolated injected corruption without losing clean-data accuracy. **H3b** region-indexed validity with a **selected** dependency structure (IND / GLOBAL / HIER, §C.4) recovers accuracy under persistent corruption beyond every baseline below. **H3c** decisions *depend* on the reliability signal in the correct direction (interventional test, §D.3). **H3d** R2 beats R1 only when the scene changes between looks. **All evaluated with acquisition off (fixed look schedule), §C.5.** | **no-revision (plain C1 sum)**; clipped sum κ ∈ {1, 2, 4}; per-look validity; **recency/forgetting slot**; R2; **perfect-validity oracle (ceiling)** | dependency structure IND / GLOBAL / HIER; corruption model and π mis-specified; cue correct / flipped / sham / missing; order of corruption (early vs late) | validity AUROC and calibration; accuracy, log-loss, ECE; clean-data cost; **fraction of the no-revision→oracle gap recovered**; cue-intervention table (§D.3); order-invariance; retract/restore audit; compute and memory | any acceptance criterion H-BR-1…6 (§D.3) fails; or the best-structure model does not beat **no revision** on held-out data | **Frozen `ff_112`** for the sensor; the corruption/cue head is **new (small) training** — flagged |
| **4 Object-centric scene representation** | **H4** a factorised K-slot hypothesis space beats K independent single-object C1 runs on multi-object scenes with occlusion/overlap. | independent C1 per object; slot-attention-style grouping [U] | number of slots; binding by position only vs by appearance | per-object accuracy, binding errors, calibration of count | no gain over independent runs on *separated* objects (then structure is unneeded); or only gains with oracle layouts | **New data and new training** (highest risk; defer) |
| **5 Robustness and adaptive computation** | **H5a** the whole pipeline (policy, stopping, validity layer) is no less robust than the `ff` baseline under adaptive white-box attack at ≤ equal FLOPs. **H5b** compute-exhaustion attacks (DeepSloth-type) do not inflate looks without bound. **H5c** the validity layer is not a new attack surface. **H5d** *conditional specialization* (CivoE-like router) beats an unconditional policy only at matched compute. | `ff` baseline; PatchGuard/Clipped-BagNet-style bounded aggregation | validity layer on/off; cue under attacker control; stopping rule on/off | robust accuracy worst-of-3; certified patch fraction; looks used under attack | any of H5a–c fails; or H5d shows no matched-compute gain | **Frozen + new head(s)**; attack code is evaluation only |

### D.2 Bounded influence and LTT are evaluated independently

* **LTT stopping** is model-free in the sense that its guarantee P(R(λ̂) ≤ α) ≥ 1 − δ does not depend on the sensor model, but it depends on a **unit of exchangeability** that the Gate-1 scene design violates by construction: scenes that share a source image are dependent, and at Gate-1 reuse a single dependence component holds most of the scenes (§E.5). Calibration and test items for LTT must therefore be image-disjoint in **both** roles (target and background). It survives a Gate-1 failure, but only with that split.
* **Bounded influence** depends on the sensor model and on local evidence; it is meaningful only if the head is local (the unit of influence is the whole window look because a ViT mixes its tokens). Test it with adaptive attacks and exact certificates on the real network, separately from stopping.
* Neither is evidence for belief revision, and belief revision is not evidence for either.

### D.3 Experiments that separate consistency, accumulation, reliability, revision and reliance

Five distinct quantities, five distinct tests (all pre-registered before running; all use the Stage-3 corruption suite):

| Quantity | Test | Distinguishes from |
|---|---|---|
| **Representation consistency** | frozen-sensor statistics (d′, null sd, slope) unchanged across the corruption suite *except* where the covariate model predicts | belief changes caused by a drifting sensor |
| **Evidence accumulation** | clean multi-look accuracy and calibration vs looks | revision |
| **Reliability assessment** | validity AUROC and calibration of p(corrupt) on injected, labelled corruptions | acting on it |
| **Belief revision** | decision after a *revealed* invalidation (cue arrives late) vs decision had the invalid region never been seen | mere forgetting (include the recency baseline) |
| **Downstream reliance** | **counterfactual intervention**: replace the reliability signal with one drawn from the opposite regime and measure the decision-change rate and accuracy change (toy: 21.1 %; 0.752 → 0.625) | representing the signal without using it |

**Interventional reliance test: correct, flipped, sham and missing cues.** A decision that changes after a cue intervention is not enough; the *direction* of posterior movement and the *effect on accuracy* must be measured. Four conditions, each compared with the missing-cue posterior (does the cue move belief the right way?) and with the factual correct-cue posterior:

* **correct** — the observed cue; **flipped** — a cue drawn from the opposite validity regime; **sham** — a cue drawn from the marginal cue distribution, independent of this scene's validity; **missing** — the cue omitted (marginalised out).

Synthetic toy [S] (N = 20,000, K = 10, R = 4 regions × 2 looks, cue_r ~ N(2·bad_r, 1) carrying no class information, model = independent validity + cue). Left block: relative to the **missing-cue** posterior; Δp(bad) is the change in the posterior probability that a region is corrupt, split by whether it truly is.

| Cue | accuracy | Δ accuracy vs missing | Δ log p(true class) | Δ p(bad), truly bad regions | Δ p(bad), truly valid regions |
|---|---|---|---|---|---|
| missing | 0.620 | 0 | 0 | 0 | 0 |
| correct | 0.696 | **+0.075** | **+0.240** | **+0.208** | **−0.089** |
| flipped | 0.239 | **−0.381** | −1.684 | −0.276 | +0.388 |
| sham | 0.511 | **−0.110** | −0.477 | −0.104 | +0.050 |

Against the factual (correct-cue) decision: flipping it changes **68.6 %** of decisions, of which 2.2 % of all scenes move wrong → right and **47.8 %** right → wrong; the sham cue changes 38.7 % (4.0 % helpful, 22.4 % harmful); dropping the cue changes 22.8 % (3.6 % helpful, 11.1 % harmful).

Readings. (1) A correct cue moves belief in the *right direction* for both truly bad and truly valid regions and raises accuracy; a flipped cue reverses both movements and drives accuracy far below the missing-cue level. (2) **The sham cue is not harmless**: it lowers accuracy by 0.110 against the missing-cue posterior because the model believes it. This is the reliance test measuring *gullibility*, and it is the same fact as the attack surface in Stage 5: reliance on a cue is justified only to the extent that the cue's likelihood g(s | v) is calibrated, so g must be measured, not assumed. (3) "The decision changed" would have scored all three interventions as positive evidence of reliance; the direction and accuracy columns separate them.

**H-BR acceptance criteria (strict; written before any real-network run; thresholds provisional and set before data, never moved to obtain a pass).**

1. **Beats every baseline at matched clean-data accuracy:** no-revision (plain C1 sum), clipping (κ ∈ {1, 2, 4}), per-look validity, recency/forgetting, on accuracy *and* log-loss/ECE under persistent corruption; the clean-data cost is reported and bounded by a pre-set margin.
2. **Reported against the ceiling:** the fraction of the gap between no-revision and the **perfect-validity oracle** that is recovered. Toy illustration of the quantity, not a target: (0.620 − 0.528)/(0.768 − 0.528) = 38 % without a cue and (0.696 − 0.528)/(0.768 − 0.528) = 70 % with the correct cue.
3. **Validity is assessable:** AUROC and calibration of p(bad) on injected, labelled corruptions.
4. **Interventional reliance in the correct direction:** correct cue raises accuracy over missing; flipped lowers it below missing and reverses the Δp(bad) signs; the sham cue is reported and its harm is bounded by the *measured* quality of g(s | v). A system whose decisions change under a flipped cue but not in the predicted direction fails.
5. **Order invariance and preserved contradictory evidence:** early-corrupt and late-corrupt scenes reach the same accuracy within the interval (the recency baseline does not); a late corrupt region does not displace earlier valid evidence; valid but surprising evidence is not retracted merely because it disagrees.
6. **Provenance audit:** retract(r) followed by restore(r) returns the original posterior to numerical precision, and the system can list which regions it discounted and by how much.

The best-structure model (§C.4) must also beat **no revision** on held-out data; if it does not, the answer is that revision is not warranted.

**Misleading-but-initially-predictive features (your experiments 1–6 from the shaping paper).**

1. Build scenes where a *feature* (background texture, border cue, a decoy patch) is predictive of the class for the first looks and is then invalidated by later observations (the feature's correlation flips).
2. **Encode?** Fit probes for the feature at the sensor tokens and at the belief state.
3. **Depend?** Intervene: ablate/patch the feature direction (or swap the feature between scenes) and measure the decision change.
4. Present the invalidating observations.
5. Compare reliability-aware (R1/R2) and reliability-blind (C1 sum, recency) mechanisms: does reliance on the invalidated feature fall, and does clean-data accuracy survive?
6. Monitor with **three probe classes**, as the shaping paper does: a *fixed* linear probe (frozen after fitting on the pre-invalidation state), a *refitted* linear probe, and a *nonlinear* probe. Interpretation, following the paper's three circumvention mechanisms (drift, erasure, nonlinear re-encoding) [P]: fixed fails + refitted succeeds → the concept **moved**, not left; refitted fails + nonlinear succeeds → **nonlinear re-encoding**; all fail **and** the intervention in step 3 shows no dependence → candidate genuine removal; all succeed yet decisions no longer depend → **represented but not relied upon** (the desired outcome for an invalidated feature).
7. **Goodhart control.** Never optimise a reliability mechanism against the probe used to monitor it. Use a disjoint monitor family and a held-out corruption family; the paper itself reports the original probe stays reliable when no gradient flows through its score [P] — preserve that property.

**What the toy does not establish.** M3 is exact Bayes with the true parameters, so reliance is guaranteed by algebra. All of the above becomes informative only when a *learned* component (the cue head, the validity prior) is in the loop.

### D.4 JEPA-motivated experiment (representation formation, later stage)

* **Hypothesis.** If two looks at the *same scene from independent observations* are used as positive pairs (good views in the paper's sense), the encoder's evidence space is more stable across looks than with crops of one image as positives.
* **Why it needs care.** The theorem is a **population statement over a finite world with injective renderings and matched marginals**; real modalities are lossy; the paper states that no experiment meets the hypotheses of Theorem 1 exactly and that its image runs mostly use crops as positives [P: Balestriero & LeCun, "Known limitations" paragraph and the discussion before it]. Crops of one image are *bad views* in their terms. A **native window and a gist look of one canvas are functions of the same pixels** and so are not independent observations of the scene either; independent re-observation needs genuinely new pixels (another frame, another sensor, a re-shot). That is a data requirement, not an architectural one.
* **Reliability-weighted pairs** are not covered by the theory (it has no notion of down-weighting a view). Treat as untested.
* **Status.** A later-stage idea; requires new training; not part of Stages 1–3.

---

## E. Gate-1 review

**Scope.** The draft (`RHAN_IM_07_GATE1_PREREG_DRAFT.md`) is *sensor qualification*: does the frozen `ff_112` trunk + 384→51 head satisfy the **likelihood and accumulation assumptions C1 uses** (equal-variance Gaussian per-unit evidence with measured d′; additive token evidence after the VIF correction; sum vs best across looks), is a native look at a small target worth more than a gist look, and can the tokens of one look be added as evidence. It says nothing about reliability, belief revision, natural objects, or any trained trunk. I recommend it stay that way. **Do not freeze before** the points below are decided; I re-read the measurement code and the draft (`gate1_measure_dprime.py`, `gate1.py`, `sdt.py`, `layout.py`, prereg §3–§5) but, after the container reset, only part of spec §4–§6/§11 and the migration plan — a final pre-freeze read of those two belongs to you or to a follow-up.

**What I ran.** Three throw-away scripts outside the repo [S]: a numpy port of the scene sampler and estimators (`torch` is not available here; a port can differ from the real code) with a **synthetic** sensor, a class-split analysis of a *common* ImageNet-100 class list, and the belief-revision toy above. The simulation's numbers are properties of the sampling design under assumed sensors (equal-variance Gaussian, assumed true d′ 2.0 at the centre and 1.2 elsewhere; R = 12 replicates per setting, so SDs are good to roughly ±20 %).

### E.1 Is `n_test = 6,000` supported? Yes for the two gated scalars; no for several things you might read into it

Design counts at the operating cell (size 4, opacity 1.0), 200 draws of the sampler [S]:

| Quantity | Value |
|---|---|
| operating-cell target scenes | mean 701 |
| distinct target images touched | mean 537 of 1,250 (50 classes × 25 val-half images) |
| size-biased scenes per target image m̃ | 1.56 |
| size-biased scenes per **background** image (all 6,000 scenes) | 5.80 |
| positives per token position, median over the 64 positions (size 4, opacity 1.0) | **exact expectation from the real sampler rules: 114.9** (min 51.7, max 175.8; all 64 positions ≥ 30); my sampled port: 115. The draft's 102 is **one replicate** of the operating-characteristics run (see below). |
| scenes with the whole target inside the window | mean 266 (5th–95th percentile 243–290) |
| design effect on a mean over the operating cell, ρ_img = 0.1 / 0.2 / 0.4 | 1.06 / 1.11 / 1.22 |

Estimator spread [S, synthetic Gaussian sensor; "fixed pool" = the same 1,250 images across runs, as in your three seeds; "fresh pool" = a new image pool each run]:

| Image effects | fixed pool: d_native_centre mean (SD) | fixed pool: slope median mean (SD) | fresh pool: d (SD) | fresh pool: slope (SD) |
|---|---|---|---|---|
| none | 2.009 (0.027) | 0.997 (0.014) | same | same |
| moderate (τ² = 0.2, ρ_bg = 0.3) | 1.973 (0.033) | 1.007 (0.015) | 2.000 (0.033) | 1.001 (0.017) |
| strong (τ² = 0.4, ρ_bg = 0.6) | 1.958 (0.040) | 1.014 (0.016) | 1.996 (0.040) | 1.002 (0.025) |

Reading: with these assumptions the run-to-run SD of the gated d is ≈ 0.03–0.04, so a threshold of 1.5 is many SDs from a sensor whose true d′ is ≈ 2.0; **`n_test = 6,000` is not the binding constraint, and doubling it would not help**, because the *image-pool* contribution (here a −0.03 to −0.04 shift of the fixed-pool mean, the same size as the run SD) **does not shrink with more scenes** — scenes are drawn with replacement from the same 25 test images per class — and is shared by all three primary runs, so averaging over seeds does not remove it. If you want a smaller pool error, the lever is more distinct images (a held-out slice of `train/` for test, which then cannot train the head), not more scenes.

**What `n_test = 6,000` does not support** (name any claim the gate cannot make):

1. **Per-class detectability.** ≈ 14 operating-cell target scenes per class (701 / 50). Class-level heterogeneity is not testable.
2. **Per-size cells other than the operating size.** The draft already notes size-2 is underpowered (§10.6).
3. **Position-level slope heterogeneity** (see E.2).
4. **Generalisation beyond this image pool**: 25 images/class, one fixed pool across the three primary runs.
5. **Natural objects.** Targets are pasted squares (draft §2 says so).
6. **Per-class leakage.** The leakage quantity averages over the 49 wrong classes, which dilutes a single strongly confusable pair; a mean cannot show it.
7. **Any claim about reliability.** Gate 1 measures d′ on clean pasted squares in clean backgrounds only.

### E.2 Does the median z-ROC slope capture heterogeneity? No

Synthetic case [S]: unit variance in the 4 × 4 centre and positive sd 1.6 (true slope 0.625) in the 28-position outer ring, i.e. 43 % of measured positions are non-Gaussian by the gate's own criterion. Result: **median-of-positions slope 0.900, inside [0.8, 1.25] in 100 % of runs**; outer-ring median 0.635; centre median 0.992. A median over positions hides up to just under half the positions being out of band. Because the gated quantity is the median, the gate would pass a sensor whose peripheral evidence violates the model. Two choices (yours, before freezing): add a report-only distribution (fraction of positions out of band, worst-ring median), or add a gated cap on that fraction. If you gate it, set the cap **before seeing data** and do not move it to obtain a PASS; my simulation only shows that a cap below ≈ 40 % would have caught this configuration, which is not a recommended value.

### E.3 Class split — a moderate imbalance and a fragile rule (corrected)

**Correction to the previous version.** I analysed the first 50 classes in **WNID** order (targets 44 organisms / 6 artifacts, 19 dog breeds; backgrounds 0 organisms / 50 artifacts-or-other; boundary `n02877765 | n02974003`) and called it a severe animal-vs-artifact confound. That is **not what the script executes**: `gate1_measure_dprime.py` (line 337) sorts the **directory names** (`sorted(d.name for d in (root/"train").iterdir())`), and I had not checked the sort key. Your run (reported to me, not reproduced by me) confirms the dataset is the CMC ImageNet-100 subset and gives the actual split: ASCII ordering puts uppercase directory names first and the boundary falls at `iron_smoothing_iron | jean_blue_jean_denim`; **targets = 27 organisms / 23 artifacts; backgrounds = 17 organisms / 33 artifacts.**

**What remains of the concern.** The imbalance is real but moderate (54 % organisms among targets against 34 % among backgrounds), so an organism-vs-artifact low-level cue is *partly* available to the head's `logit_c − logit_bg`, in the direction that could inflate `informative` and `looking_closer_pays`. Its size is **unmeasured**; I do not claim it matters. A second, independent reason to change the rule: it depends on directory naming and letter case, so the same 100 classes under different spellings or a different copy would produce a different split; a rule that silently changes with file naming is not a pre-registrable design. **Proposal for you to decide before freezing:** specify the split explicitly in the freeze file as a seeded random permutation (seed recorded) or a stratified one (organisms/artifacts balanced across targets and backgrounds), with the list of the 50 + 50 classes written out; this is a new draft version, not an edit after freezing. Optional measurement, if you want to know whether the current split matters: report d′ for the current split and for one or two recorded random splits (secondary, report-only).

### E.4 Other items

* **2-of-3 = engineering consistency.** The three primary runs use the *same checkpoint and the same image pool*; they test head-initialisation and scene-sampling reproducibility. They are not three independent replications of a scientific result. The draft should say so; the secondary runs (the seed-42 and seed-99 checkpoints) are the only evidence about trunk-to-trunk variation and are report-only.
* **Pooled-class scale mixture.** A pooled z-ROC over classes with different null spread is a scale mixture and can move the slope from 1 without any per-class violation; the pooled number alone cannot separate "non-Gaussian" from "heterogeneous".
* **Thresholds are provisional.** I recommend recording them as *design thresholds chosen before data*, never as properties of the world, and never adjusted to obtain a PASS; if they must change after seeing data, that is `rhan_im_gate1_v2`, reported next to v1.
* **Reused images ≠ independent information.** 6,000 scenes from 1,250 images are not 6,000 independent draws; the design effect above quantifies the part that matters for the gated mean under the assumed ρ_img.
* **Do not auto-double** any size (consistent with your instruction).
* **The 115-vs-102 discrepancy (item C) — resolved.** (1) The draft's 102 / 29 / 228 (sizes 4 / 2 / 6, opacity 1.0) are the values of one replicate (`reps[0]`, seed 7) of `gate1_operating_characteristics.py`, which also recorded 4,135 target scenes against an expected 4,200. (2) I computed the exact expectation from the sampler rules (no sampling): size 2: 32.8 (34 of 64 positions ≥ 30), size 4: 114.9, size 6: 224.8, at 700 expected scenes per cell. (3) **Your run of the real code resolves it** (reported to me, not reproduced by me): 50 seeds of `check_sampler_geometry.py` give size 2: 31.10 (SD 1.33), size 4: **114.88** (SD 5.91, range 102–132), size 6: 223.54 (SD 9.99), and 36.6 of 64 positions ≥ 30 at size 2 (expectation 34); the one-replicate run reproduces 102 / 29 / 228 and 4,135 target scenes deterministically; the size-4 minimum over the 50 seeds is exactly the draft's 102 (seed 7). So the sampler matches the rules I transcribed, my port was right, and the draft's 102 is a **low-tail draw**, not a bias. (4) **For the freeze document:** quote the expected values (114.9 / 32.8 / 224.8) and say the draft's single-replicate numbers are one draw; the size-2 conclusion (about half the positions below 30) is unchanged.
* **Not touched.** `prereg/rhan_im_gate1_v1.json` does not exist yet and I created nothing like it.

### E.5 The Learn-then-Test exchangeability unit (item B)

**What is already written.** The spec and `stopping.py` say the calibration losses must be independent draws "at the level of the unit that is resampled", counted by source image; `05_EXPERIMENTS` §3 says E5 uses *one scene per source image* from an image-level split into four disjoint sets (head training; measurement of d′/standardiser/VIFs; LTT calibration; final test). What is **not** pinned down, and what the Gate-1 design makes concrete:

1. **A scene has two images, so "one scene per source image" must hold for both roles** (target and background) and across the calibration and test sets. Disjointness in only one role leaves dependent pairs.
2. **Reuse is not a mild violation.** I built the dependence graph for scenes drawn with replacement from one 1,250-image-per-role pool (a scene links its background image to its target image). Largest dependence component [S]: **2 % of scenes at n = 1,250, 83 % at n = 3,000, 99 % at n = 6,000** (41 clusters at 6,000). At Gate-1 reuse there is effectively one cluster, so cluster-robust inference over images has nothing to work with. A design-effect correction needs the unknown image-level error correlation ρ_img: with the size-biased 3.40 scenes per background image at n = 3,000, n_eff = 2,679 / 2,420 / 2,027 / 1,531 for ρ_img = 0.05 / 0.1 / 0.2 / 0.4. That is not a valid replacement for disjointness, because the exact binomial tail needs independent items, not an inflated n.
3. **The cost of independence is small.** At α = δ = 0.10 the largest empirical risk the exact-binomial fixed-sequence test certifies is 0.0733 at n = 300, **0.0880 at n = 1,250**, 0.0927 at n = 3,000 and 0.0948 at n = 6,000 [S]. Image-disjoint n = 1,250 gives up 0.0047 in certifiable empirical risk against the nominal 3,000, in exchange for a guarantee that is actually valid.
4. **The Gate-1 pool cannot supply it beyond n ≈ 1,250** (1,250 images per role per half). Gate-1's own n_cal = 3,000 serves only to standardise scores (point estimates, no intervals), so reuse there costs a design effect, not a validity. **LTT calibration and test must not reuse Gate-1 scenes** (already stated in draft §10.2) and need an image-level split of `train/`-sized data, which is an image-pool decision for you (§E.1, item 3 of the decisions).
5. **The target of the guarantee must be named.** The calibration items must be drawn from the *deployment distribution*; if deployment means fresh images from the same class population, the calibration and test pools must be image-disjoint draws from it. A guarantee established on a fixed finite pool with reused images covers only that pool.
6. **The "violation rate over ≥ 500 resamples" check (E5)** is a Monte-Carlo over calibration subsets *conditional on a fixed test pool*; the 500 draws from one finite pool reuse images and are positively correlated, so the usual binomial standard error on the violation rate is too small. Partition the pool into disjoint blocks (one block per replicate) or report the between-block variance; and estimate the true risk R(λ̂) on image-disjoint test items large enough that its own error (≈ ±0.005 at m = 3,000 near α = 0.10) is small against the margin being tested.
7. **Residual dependence even with disjoint images** (near-duplicate photographs, same photographer or scene across classes in ImageNet) [U]: not quantified here; a small positive error correlation would reduce the effective n slightly and cannot be tested from this document.

**Recommendation for the freeze (your decision).** State the exchangeability unit as "(target image, background image) pair, with every image used at most once across calibration and test, both roles"; fix the four-way image-level split before any E5 number is seen; keep Gate-1 scenes out of LTT; and treat the image-pool size as the binding constraint on n_cal.

---

## F. Architecture decision

| Option | Verdict | Reason |
|---|---|---|
| 1. Keep C1 and extend it as one architecture | **Partly** | Extending C1 with R1 is cheap and keeps the foundation, but fusing revision into C1's identity now would commit to a hypothesis (H-BR) that no data can test before Gate 1. |
| 2. C2 as the primary architecture | **Not supported** | No acquisition guarantee, no "unseen" state, no calibration, no risk-controlled stopping; its one clear advantage (mutable content, forgetting) matters only for dynamic scenes, which are not RHAN's current target. |
| 3. C1 + C2 hybrid | **Hypothesis, not a design** | For a static scene C1 is already the information-form filter; a hybrid helps only for dynamic scenes or non-additive content. Doubled state, no evidence. |
| 4. C1 as sensor/acquisition foundation + a *separate* belief-revision layer with its own hypothesis and gates | **Recommended, staged** | If H-BR fails, the foundation stands unchanged; if it holds, it is a layer with a stable interface (region-tagged evidence in, posterior over (hypothesis, validity) out), not a new network. |

**Recommendation.** Option 4. Concretely:

1. **Keep C1 as the experimental foundation, not as a frozen final identity.** Its identity claim is "sensor + acquisition + risk-controlled stopping"; that claim is complete without belief revision.
2. **Treat belief revision as H-BR**, tested in order: simulation (done, §A.3, and only a toy) → real sensor after Gate 1 (Stage 3) → adversarial setting (Stage 5).
3. **C2 slots only as a controlled alternative (R2)** for the one setting where they have a reason to win: the scene changes between looks.
4. **Hybrid, specialization, multi-object structure**: not now. Each needs a demonstrated gap first.
5. **Prefer the simplest architecture.** The recency baseline (M4) is simpler than R1. R1 has to beat it at matched clean-data accuracy, or the right answer is forgetting.

**Formal research structure (adopted after your review).**

| Element | Status |
|---|---|
| **C1 — foundation** | Measured sensor, evidence acquisition, posterior accumulation, risk-controlled stopping. |
| **Stage 1b — reliability characterisation** | Do observable image-quality variables explain changes in sensor reliability? |
| **H-BR — separate hypothesis** | Can the system identify and selectively reduce reliance on earlier unreliable evidence while preserving provenance and useful contradictory observations? |
| **R1 — first revision hypothesis** | Latent validity with provenance, **subject to** the sufficiency (§C.3) and dependency-structure (§C.4) assumptions being demonstrated, not presumed. |
| **R2 — controlled alternative** | Mutable information-form slots with an auditable write ledger, for changing scenes or non-additive evidence. |
| **C2 — not adopted as the architecture** | Kept as a comparison, not a direction. |
| **Stage 4 — multi-object perception** | Deferred until the single-object foundation is established. |
| **Stage 5 — robustness and computational specialization** | Deferred until the preceding capabilities have independent evidence. |

**Governing principle.** RHAN does not become a collection of interesting mechanisms: each new capability must solve a demonstrated limitation of the existing system. No hybrid because it combines C1 and C2; no expert specialization because another project uses it; no claim of belief revision merely because a posterior changes. **Not approved and not done:** freezing Gate 1; implementing R1, R2 or any hybrid.

**Prerequisites before Gate 1 can be frozen:** the class-split rule written out explicitly (§E.3), the image-pool decision (§E.1/§E.5), and the LTT exchangeability unit (§E.5). *Resolved since the last version:* the sampler discrepancy (§E.1).

**Evidence required to commit** (written before any of it exists):

| Gate | Evidence | If it fails |
|---|---|---|
| Gate 1 (revised design, §E) | H1a–c pass on the real sensor | dossier §9 switch conditions (heteroscedastic likelihood → dense window CE → temporal looks → calibrated selective classification) |
| Stage 2 | acquisition beats fixed coverage **and** the matched-compute recompute control | the thesis "recurrence earns its cost only where information is acquired" is rejected for this trunk; keep the negative result |
| Stage 1b | d′ predictable from observable covariates | reliability cannot be read from the input; only contradiction between looks remains |
| Stage 3 | H-BR-1…6 (§D.3) all met: beats no-revision, clipping, per-look validity and recency at matched clean-data accuracy; recovers a reported fraction of the oracle gap; validity AUROC ≥ 0.7; correct/flipped/sham/missing cues move belief and accuracy in the predicted directions | belief revision is **not** part of the identity; the project is calibrated active perception with measured sensors |
| Stage 3 (changing scene) | R2 beats R1 on moving/changing scenes | hybrid enters as a hypothesis; otherwise C2 stays an alternative |

**What would change this recommendation.** (a) A real-network d′ with no native-vs-gist advantage: temporal looks become the natural setting and R2/C2 move up. (b) R1 loses to the recency baseline on real corruptions: forgetting is enough and the simpler architecture wins. (c) Single-object hypotheses prove too weak for every target worth studying: Stage 4 moves first. (d) A matched-compute ablation, from anyone, showing that conditional specialization helps a perception system.

---

## G. Revised research identity

> **RHAN: perception as measured-evidence search.** A system whose belief about a scene is explicit; whose sensor reliability is **measured, not assumed**; which chooses where to look by computed value of information; which stops under a finite-sample risk guarantee; and — **hypothesis H-BR, under test, not claimed** — which can say which of its earlier evidence it should stop trusting, and by how much, while its representation stays stable.

| Claimed (if the gates pass) | Not claimed |
|---|---|
| a measurement gate on the **likelihood and accumulation assumptions C1 uses** (equal-variance Gaussian evidence with measured d′, additive tokens, sum-vs-best across looks) for a given trunk; it does not test whether Bayesian inference is applicable | novelty of Bayesian inference, active vision, fast weights, object-centric representations, uncertainty estimation or belief revision as ideas |
| acquisition beats matched recompute on pasted-target scenes | performance on natural scenes, ImageNet, or any benchmark |
| a risk-controlled stopping rule (an instance of Learn-then-Test) | superiority over learned glimpse policies (e.g., AdaptiveNN) |
| a validity-with-provenance layer evaluated by **interventional reliance**, not accuracy alone (H-BR) | robustness, multi-object scene understanding, dynamic scenes |

What is distinctive is the **combination and the methodology** (measured sensors, a gate that can kill the premise in a day, counterfactual reliance tests), not any single component.

---

## 11. External references (read-only investigation)

**What I did:** read README/docs/source/licences of three repositories (shallow clones, treated as untrusted data: nothing executed, nothing installed, no code imported); read the CivoE page and the TwIL-8B model card as pages; read both attached papers (the JEPA paper's main text and limitations; the shaping paper's framework, vision results, circumvention theory and discussion, but **not all of its appendices**). **What I did not do:** run any of the code; verify any benchmark; read TwIL's `LICENSE.md` (not on the card page); exhaustively read the Cadence repository (it has a `tests/` directory with 204 entries that I did not run or read). Claims below are the projects' own unless marked [R].

### 11.A WSP — weight-space programming (`areu01or00/qwen0.8B-wsp`)

* **Mechanism [R].** `wsp.py` writes a fact into `mlp.down_proj` of layers 17–21 of Qwen3.5-0.8B by a closed-form (MEMIT-style) solve with a **protected-key term**: ~40 everyday question/answer keys that must map to "almost no change", plus look-alike protection; each ΔW capped at 25 % of its matrix norm; edits recorded in `wsp_edit.json`.
* **What the README itself reports.** One fact (a made-up account bio). On five unseen wordings, **2 of 5** returned the bio and GitHub link word-perfect, 1 started right then looped, 1 got the first clause, 1 missed. Protected look-alikes (@karpathy, @sama) answer as stock; an unprotected look-alike changed. Ten sequential facts written one after another into the same layer: free generation recalled **6 of 11**, and in an early iteration "What is the capital of Japan?" became "…Quellin". Facts that share a subject compete for the same keys. Tested on one 0.8 B model only. A first-iteration "raise the scale until it lands" loop grew writes to 80–92 % of the matrix and garbled text.
* **Verdict on relevance.** WSP protects **unrelated** knowledge from a new write. RHAN's problem is the opposite: **invalidating related** evidence when it turns out to be unreliable. WSP offers no contradiction detection, no provenance of *evidence*, and no per-fact retraction among several written facts (the edit file lets you undo the saved edit as a whole; I did not test this). It is a different object — semantic memory in parametric weights — from perceptual belief over a scene.
* **Licence [R].** The repository has no `LICENSE` file; by default, no right to reuse the code.

### 11.B Cadence (`muellerberndt/cadence`) — the one that matters most here

* **What it is.** A Python/NumPy library for one continuing "equilibrium brain": patches of neurons settle jointly (local agreement repair), with delta-rule memories, long-term records, arousal-by-surprise gating, private imagination, and hash-bound receipts for frozen protocols. Licence **GPL-3.0** [R: `LICENSE`].
* **Ambition vs implementation [R].** The manifesto says outright that the goals "do not imply that every capability is already implemented". Examples in its own docs: "selection that revises with the present interpretation remains a goal"; the opt-in episodic memory "uses fixed observation/history keys and adds its vote to the motor readout after the neural solve, so its measured benefits are component evidence"; "integrated camera behavior remains to be measured"; "settled-state reuse across moments, repair localized to declared dependencies and integrated learned consequences are open work".
* **Evidence it does have [R, self-reported].** A frozen-protocol bandit reversal benchmark (a four-odour world: rule A, then rule B, then A again, with a tabular learner and Elman/GRU actor-critics as comparators). On ten confirmation seeds the `live` arm met its gates: of 40 gated lives, 40 acquired rule A, 39 reversed, 39 returned, 40 kept the stable pair; median reversal lag 15.5, 24.5, 30 and 17 trials at exposures 300, 1,000, 3,000 and 10,000. The default-configuration arm sat near chance (0.48–0.53 optimal). There is **no vision result** and no contradiction-in-evidence test beyond reward reversal.
* **Mechanisms, individually (§12 records them).** (i) delta-rule writes `M += rate·outer(k, v − k·M)` with a stated interference law; (ii) `LongTermMemory` **revision**: a write whose first-choice slot holds a key at cosine ≥ `revise` **rewrites that slot's key and value in place and deepens it**, so "no second copy stays behind" — last-write-wins by similarity; (iii) `Brain.imagine`: a private branch with read-only durable memory and parameters; the docstring is explicit that it "imagines the brain's responses, not an unobserved environment's dynamics"; (iv) arousal gating (routine = settle once and learn nothing; surprise raises learning); (v) retained "actual events with their time, context and provenance" and a *missing-flag* convention (a missing feature differs from an observed zero).
* **Which transferable idea survives scrutiny.** Cadence's revision is **overwrite**, gated by similarity. In my toy that is the **recency baseline (M4)**: strong against early corruption (0.639 / 0.644 vs 0.52–0.54), clearly worse on clean data (0.747 vs 0.874), and uncalibrated [S]. As a belief-revision mechanism it is the baseline Stage 3 must beat, not an answer. What *is* transferable is **discipline**: provenance-tagged records kept apart from interpretation; hypothetical branches that never write durable state; missing ≠ zero; frozen, hash-bound protocols (which RHAN's preregistration already does).
* **Licence consequence.** GPL-3.0: do not copy or adapt code; the mathematics of the delta rule and Kalman filtering is textbook and already in our toy, so there is nothing to reuse that we could not derive. This is not legal advice.

### 11.C Meridian-1 CivoE (page reading only)

* **As claimed.** Experts specialised for named capabilities (personal-data detection, photo questions, contract-clause finding, table understanding, banking intent classification) on **one shared 12 B foundation**; "one expert engages at a time"; two to nine model calls per task; the page says it "selects the right one" but **does not describe the router**; disagreement and shared uncertainty are **not described**; the only conflict-resolution mention is a security workflow where "a Critic expert checks the work".
* **Evidence offered.** Internal benchmark: a composite on "a selected evaluation drawn from 14 public benchmark datasets" rising from **59.8 to 85.5**; frontier comparisons "not fully controlled" and not leaderboard submissions; 95 % bootstrap intervals not corrected for earlier task-family selection; family-level differences untested; **an ablation ("same workflow, no experts") only for prompt-injection benchmarks**; benign AgentDojo completion fell 63.9 % vs 76.3 %; CivoE followed a planted instruction in 50 % of 60 synthetic trap briefings; **no compute saving measured** ("lower total operating cost has not been established"); no code, weights or paper ("full research report: not yet published"); scaling to hundreds of experts not established.
* **Relevance to RHAN.** It addresses language-agent task routing; it is **not** an LLM-agent design we want. The abstract idea — *different uncertainty types should trigger different pathways* — is a hypothesis for Stage 5, but nothing on the page tests it, shares uncertainty between experts, or prevents unstable disagreement. Benchmarks here are **not** evidence for RHAN.

### 11.D TwIL-8B (model card reading only)

* **As stated.** Full-parameter SFT of IBM `granite-4.2-8b` (dense decoder-only, 40 layers, 8.79 B parameters) on reasoning traces for maths, STEM and instruction following; then GRPO/DAPO RL (the *verbose* variant); then a length-aware RL stage (the *balanced* default). Dataset names, sizes and training compute not stated. Reported scores are the authors' internal high-effort pass@1; competitor scores were taken from published sources and not re-run; AIME/HMMT sets have only 30–33 problems. Self-described weaknesses: knowledge benchmarks, ~19.5 k output tokens per AIME problem, no tool-use or agentic evaluation. Weights bf16 16.38 GiB; Q4_K_M GGUF 4.98 GiB. **Licence: "webAI Non-Commercial License ver. 1.0"** (full text in `LICENSE.md`, which I did not read); base model Apache-2.0.
* **Usefulness.** An optional *auxiliary* reasoner for checking derivations or drafting analysis scripts, with all outputs verified by code or hand. Its strength (contest maths) is not what RHAN's applied statistics need. Not a perception backbone; no role on the critical path.

### 11.E Teachable Figures (`ywEdAi/teachable-figures`)

* **What it is [R].** An agent *skill* that generates interactive teaching figures as self-contained HTML; MIT for code, CC BY 4.0 for content. A `tests/` directory with files such as `bayes.test.cjs`, `selection.test.cjs`, `equilibrium.test.cjs` checks the figures' invariants (I listed the files; I did not open or run them). The README's own audit note says a successful build is insufficient evidence for an exhaustive geometric proof in at least some figures. A Bayes/base-rate figure counts the sick among positives in a 1,000-person grid and updates a prior.
* **Relevance.** Communication, not mechanism: a way to explain evidence accumulation, contradictory evidence and belief revision to readers *after* experiments exist. Research tooling only; never evidence.

---

## 12. Cross-project synthesis

### 12.1 Transferable mechanisms, ten-point record

**T1 — Cadence: in-place revise-on-similarity long-term records (with delta-rule writes)**

| # | |
|---|---|
| 1 | `LongTermMemory`: a write first reads with its own key; if the first choice holds a key at cosine ≥ `revise`, rewrite that slot's key and value in place and deepen it; `fade` replaces the most faded slot when full. Delta-rule stores: `M += rate·outer(k, v − kM)`. |
| 2 | A fact that changed should be current, with no stale duplicate. |
| 3 | **Implemented** (module and docstring read; a `tests/` directory exists; nothing run by me). |
| 4 | Bandit reversal benchmark, self-reported, frozen protocol; no vision evidence. |
| 5 | Last-write-wins: cannot tell a corrupted re-telling from a correction; overlapping keys interfere (its own docs); no reliability or contradiction test beyond similarity. |
| 6 | Partly: gives a *mutable content* model for the dynamic-scene case C1 cannot represent. |
| 7 | Overlaps C2 (delta rule) and my recency baseline M4 exactly. |
| 8 | The math transfers; the similarity-gated overwrite does not give revision, only recency. |
| 9 | O(d²) per write for the matrix form; one read per moment. |
| 10 | GPL-3.0; do not copy. |

**T2 — Cadence: private hypothetical branch (`Brain.imagine`)**

| # | |
|---|---|
| 1 | A privately advanced branch state with durable memory and parameters read-only; refused phases end the branch and stay in the result. |
| 2 | Evaluate "what would I do if…" without corrupting durable state. |
| 3 | **Implemented** (docstring and code read; not run). |
| 4 | None specific to perception; the docstring says it imagines the brain's *responses*, not the environment's dynamics. |
| 5 | A response-level counterfactual, not a hypothesis about the world; cost grows with branch length. |
| 6 | Yes, as discipline: hypotheticals must not write durable state. |
| 7 | Overlaps what exact Bayes does for free: p(h | region R invalid) is a closed-form marginal in R1; R2 needs a private slot. |
| 8 | The *rule* (hypothetical state never writes durable state) transfers; the implementation does not. |
| 9 | Branch length × one settle per step. |
| 10 | GPL-3.0. |

**T3 — Cadence: patch-net local equilibrium (overlapping state, disagreement repaired locally)**

| # | |
|---|---|
| 1 | Patches hold local state and settle together over reciprocal synapses; learning is the same local repair driven by witnessed mismatch. |
| 2 | Perception, memory and action in one continuing system. |
| 3 | **Implemented as a research library; the whole-brain claims are goals.** |
| 4 | Frozen-protocol bandit results only. "Integrated camera behavior remains to be measured." |
| 5 | Equilibrium ≠ correctness: its docs state numerical settlement does not establish meaning; a unique stable interpretation needs its own measurement. |
| 6 | Not directly: it has no measured sensor model, no calibrated uncertainty, no look policy. |
| 7 | Overlaps C3 (equilibrium perception), not C1/C2. |
| 8 | No: it is a different research program, not a mechanism to bolt on. |
| 9 | "Sweeps" counted by the harness; not comparable to FLOPs per look without conversion. |
| 10 | GPL-3.0. |

**T4 — WSP: closed-form localized weight edit with protected keys**

| # | |
|---|---|
| 1 | Solve ΔW on `down_proj` in layers 17–21 so new keys map to new values while ~40 protected keys map to ≈ no change; cap each ΔW at 25 % of norm. |
| 2 | Writing a persistent fact into an LLM without training and without disturbing unrelated answers. |
| 3 | **Implemented** (code and results in the repo; not run by me). |
| 4 | Single fact, one 0.8 B model: 2 of 5 unseen wordings word-perfect; ten-fact sequence 6/11 recalled in free generation. |
| 5 | Facts sharing a subject compete; first tokens hardest; numbers and dates fragile; drift elsewhere. |
| 6 | No: RHAN's problem is revising related, invalidated evidence, not protecting unrelated knowledge. |
| 7 | Overlaps nothing in RHAN-IM; relates to C2's interference theory (keys and null spaces). |
| 8 | The protected-key idea is standard null-space editing; transferable math is generic, and not needed. |
| 9 | Closed-form solve plus refinement passes per layer. |
| 10 | No licence file: no reuse rights. |

**T5 — CivoE: conditional specialization over a shared foundation**

| # | |
|---|---|
| 1 | A router selects one specialised expert per task over a fixed 12 B foundation. |
| 2 | Better task performance at modest cost (claimed). |
| 3 | **Announced / internal results; no code or paper.** |
| 4 | Internal composite 59.8 → 85.5; ablation only on prompt-injection. |
| 5 | Router undescribed; no shared uncertainty; no scaling evidence. |
| 6 | Only as an abstract hypothesis: pathways by uncertainty *type*. |
| 7 | Overlaps RHAN-IM's policy-and-stopping as a computation-selection mechanism. |
| 8 | The principle only; no mechanism to transfer. |
| 9 | Two to nine model calls per task; savings not established. |
| 10 | Not available. |

**T6 — Mechanistic Shaping (Fel et al.): reliance vs expression, and a monitor hierarchy**

| # | |
|---|---|
| 1 | Train with a penalty on **attribution** to a concept direction (reliance) as well as on its expression; monitor with classes of probes. |
| 2 | Models that must not *use* a concept (not just not output it). |
| 3 | **Published, with released code** (`goodfire-ai/reliance`; I did not read it). |
| 4 | Across vision, medical imaging, language, up to 32 B parameters (paper's claim). |
| 5 | Training against a representation can modify it; three circumvention mechanisms (drift, erasure, nonlinear/product-code re-encoding); other mechanisms may arise. |
| 6 | **Yes, as test design**: it tells how to distinguish representing, expressing, relying. |
| 7 | New to RHAN-IM; no overlap with C1/C2. |
| 8 | The *evaluation* methodology transfers entirely; the training objective is optional and risks Goodhart. |
| 9 | Probe fitting and attribution passes; modest. |
| 10 | Paper; check the repository licence before use. |

**T7 — JEPA / Platonic representation (Balestriero & LeCun): good vs bad views**

| # | |
|---|---|
| 1 | JEPAs trained on any number of modalities learn the same embedding **up to one rotation** when positives are two observations of one scene; kernel mismatch bounds misalignment otherwise. |
| 2 | A guarantee and a recipe for aligned representations. |
| 3 | **Theorem + experiments** (145 k captions to texts of 940 M web images). |
| 4 | Scaling-law fits; text-JEPA reaches given alignment with 42× fewer samples than next-token prediction (paper's claim). |
| 5 | Population statements over a **finite world with injective renderings and matched marginals**; real modalities are lossy; no experiment meets Theorem 1's hypotheses exactly; image runs mostly use crops as positives. |
| 6 | Indirectly: tells what a good "same scene" pair is; says nothing about revising beliefs. |
| 7 | New to RHAN-IM (RHAN-IM never trains the trunk). |
| 8 | The view-quality principle; not the theorem. |
| 9 | Training-time only. |
| 10 | Paper. |

### 12.2 Mechanism-level comparison (not ambitions)

| Design | State unit | Update on new evidence | Provenance | Retraction | Uncertainty | Contradiction signal | Compute unit | Evidence in hand |
|---|---|---|---|---|---|---|---|---|
| **C1** | posterior + precision map | additive LLR (cap: order-dependent) | none | none | posterior, ignorance, LTT risk | none | one trunk pass per look | [M][S]; no real-network number |
| **C2** | matrix S | delta rule (constant-gain Kalman) | none | overwrite along a key | none | prediction error (implicit) | O(d²) per token | algebra only |
| **C4** | slow + fast state | coupled recurrence | none | none | none | none | ≈ 2 passes per step | none; control |
| **C1 + C2** | slots with precision | Kalman gain from measured d′ | write ledger (if kept) | exact (if logged) | Λ⁻¹ | innovation test | O(d²) per slot | none |
| **Cadence as implemented** | patch neurons + stores | local equilibrium repair; similarity-gated overwrite | moment records with context | last-write-wins | arousal by surprise (not a probability) | prediction error → arousal | sweeps | bandit reversal, self-reported |
| **Cadence-inspired (our math)** = R2 | slots + ledger | information-form writes | ledger | exact (linear-Gaussian) | Λ⁻¹ | χ² innovation | O(d²) per write | [S] algebra only |
| **CivoE-inspired** | shared posterior + uncertainty-type flag | unchanged | none | none | unchanged | type of uncertainty | conditional | none for perception |
| **WSP-inspired** | parameter edit | closed-form with protected keys | edit record | undo whole edit | none | none | solve per edit | single-model README |

### 12.3 Do the projects reveal a mechanism RHAN is missing?

**One, and it is not in any of the five projects as a mechanism: provenance with reliability.** Cadence states it as a convention and implements recency; the papers supply the test design. The mathematics that actually does the work (latent validity, information-form retraction) is classical and already in RHAN's own toolbox. No project justifies changing direction.

---

## 13. Integration decisions

| Reference | Decision | Why |
|---|---|---|
| **WSP** | **Reject as irrelevant** to perceptual belief revision; cite only as a separate persistent-memory line | protects unrelated knowledge; no retraction of related evidence; single-model evidence; no licence |
| **Cadence** | **Investigate through controlled experiment** (contradiction-triggered local repair = R2, from our own math); **adopt as research principles** (provenance separate from interpretation; hypotheticals never write durable state; missing ≠ zero) | its revision is the recency baseline; the principles are cheap and testable; no vision evidence; GPL-3.0 rules out code reuse |
| **CivoE** | **Investigate through controlled experiment, late (Stage 5)**, only if Stage 3 shows distinguishable uncertainty types that call for different actions | no router description, no perception evidence, no matched-compute ablation |
| **TwIL-8B** | **Use as external research tooling — optional, not adopted now** | auxiliary maths/analysis help only, verified by code; non-commercial licence; strengths mismatch RHAN's needs |
| **Teachable Figures** | **Use as external research tooling — optional, deferred** | communication of results after the experiments exist; MIT / CC BY 4.0 |
| **Shaping paper** | **Adopt as research principle** (represent ≠ express ≠ rely; monitor hierarchy) + experiments in Stage 3 | the single most useful test design for H-BR |
| **JEPA paper** | **Adopt as research principle** (view quality; independent re-observation) and **investigate later** | training-level; theory does not cover reliability-weighted pairs |

### 13.1 Capabilities for a candidate future architecture

| Capability | Status | Reason |
|---|---|---|
| Reliable visual evidence acquisition | **Necessary** | C1's core; Gate 1 and Stage 2 |
| Persistent provenance-aware observation records | **Necessary for any revision beyond isolated looks** | minimal form: region-indexed sufficient statistics; a full per-look ledger is optional |
| Mutable object-centric scene representation | **Optional; unjustified until a gap is shown** | only for dynamic scenes or non-additive content |
| Localised contradiction-driven belief revision | **Necessary for the fourth clause of your question; "localised" is unproven** | in R1 locality is per region; whether local repair beats global re-mixing is untested (global re-mix is cheap here) |
| Conditional computational specialization | **Unjustified complexity at present** | RHAN-IM already has policy + stopping; revisit only on evidence |
| Explicit uncertainty and selective stopping | **Necessary; already C1 (with LTT)** | |

None of these is combined because it is interesting; each row is conditional on its stage's gate.

---

## 14. Constraints — what I did and did not do

No installs, imports, copies or integration of any external project; no external code introduced as a dependency; Gate-1 preregistration not changed (recommendations in §E are proposals for you); other projects' claims treated as claims; mechanisms called "implemented" only where I read source or a docstring (not that I ran it); licences reviewed: WSP none, Cadence GPL-3.0, Teachable Figures MIT + CC BY 4.0, TwIL non-commercial (text unread); nothing implemented.

---

## 15. Two papers, joint implications, closest work

* **Reliance shaping (Fel et al.)** supplies the test I used in §A.4 and §D.3: representing, expressing and relying are three different quantities; decisions must be *intervened on* to establish reliance; a concept that escapes one probe may not be gone; the original probe stays reliable when no gradient flows through its score — a protection against Goodhart that I recommend preserving.
* **Good views (Balestriero & LeCun)** supplies the notion that "same scene" must mean independent observation, not crops of one image. A native window and a gist look are views of the same pixels and so are **not** good views; independence requires new pixels.
* **Joint reading.** Representation formation (stable evidence space), evidence management (provenance + reliability), belief maintenance (mutable hypotheses) and representation-level verification (do decisions use reliable evidence?) are four separate layers; RHAN-IM has the first by design (pending Gate 1) and not the middle two; §D.3 designs five tests that separate the five quantities you listed.

**Closest existing work for the joint hypothesis** (no component is claimed novel): Najemnik & Geisler; Akbas & Eckstein; Rashidi et al.; Naghshvar & Javidi; AdaptiveNN; Jazbec et al.; Clipped BagNet / PatchGuard [fetched in the earlier dossier, Tier A]; Huber's robust sequential tests, Kalman filtering with innovation gating, switching state-space models, Dempster–Shafer, AGM revision and truth-maintenance systems [U]; Saha & Ramdas robust e-values [L]; Test-time-training layers [Tier A in the dossier]; the two papers above [P]; Cadence for provenance-and-repair as a research stance [R].

**Genuine contribution vs engineering integration.** *Engineering integration:* ideal-searcher posterior, Koopman policy, LTT stopping, bounded influence, per-look validity mixtures. *Candidate contributions, none established:* (i) a **measurement gate on C1's likelihood and accumulation assumptions** for a given trunk; (ii) a **validity-with-provenance layer evaluated by interventional reliance** rather than accuracy alone; (iii) the **experimental separation** of representing and relying under evidence invalidation in an active-perception system. (ii) and (iii) are methodological.

---

## Disclosures

* **Simulation ≠ measurement.** Everything tagged [S] is a synthetic sensor with known parameters. Mathematical correctness is not practical effectiveness, and Bayesian accumulation is not belief revision.
* **Resolved by your run:** positives per position — expected 114.9 (size 4); 50 real-sampler seeds give 114.88; the draft's 102 is a low-tail draw (seed 7), reproduced deterministically (§E.1).
* **Not verified by me:** real-network d′, slope, VIF, ρ; the sampler and class-list results (they are from your run); the effect size of the organism/artifact imbalance; the shaping paper's appendices; Cadence beyond the files I list; TwIL's licence text; every [U] and [L] item.
* **Corrected error (mine):** the previous version analysed WNID order for the class split; the script sorts directory names (§E.3).
* **Repository state:** the investigation itself was read-only. The only repository changes made by this assessment are the documentation-only commits that add and correct this file (`943751f`, `8cc7365` and the commit carrying this correction). No checkpoint, model, experiment, implementation or Gate-1 preregistration was changed.
* **Not done:** pushing the branch (denied in the authoring environment; push from your own machine); any implementation.
* **Open math item (do not assume closed):** the Koopman gain, ignorance map and closed-form policy under uncertain validity (§C.5).
* **Assumptions not universal:** R1's independent-validity product is one of three structures tested (§C.4); sufficiency (§C.3) is relative to the model and to conditional independence of regions.

---

## Source ledger

| Source | What I read this round | Tag |
|---|---|---|
| Cadence | `README.md`, `MANIFESTO.md`, `docs/world-model.md`, `docs/memory.md` (the "One correction" section), `src/cadence/longterm.py` docstring, `memories.py` (`Moment`), `generic.py` (`imagine`), `benchmarks/reversal/README.md` (protocol, gates, ten-seed results), `LICENSE` (GPL-3.0). Not read: the rest of `src/`, the 204 entries in `tests/`, other benchmarks. Nothing run. | [R] |
| WSP | full `README.md`; greps of `wsp.py` (undo/save/edit-record); directory listing (no `LICENSE`) | [R] |
| Teachable Figures | `README.md`, `LICENSE`, `LICENSE-CONTENT`, `tests/` and `scripts/` listings | [R] |
| CivoE | the page (architecture, benchmarks, ablations, limitations) | [R, page] |
| TwIL-8B | the model card (training, benchmarks, licence name, limits); `LICENSE.md` not seen | [R, page] |
| Balestriero & LeCun, arXiv 2610.39302 | abstract, theory summary, Figure 1, "Known limitations" and the crops-as-positives caveat; proofs not checked | [P] |
| Fel et al., *Shaping & Monitoring Models through their Internal Representations* | abstract, framework, monitoring hierarchy, circumvention theory, vision results, discussion; **appendices and §2.2–5 not read in full** | [P] |
| Dossier literature (Najemnik & Geisler; Akbas & Eckstein; Rashidi et al.; Ahmad & Yu; Naghshvar & Javidi; AdaptiveNN; SASHA; Mukherjee et al.; Jazbec et al.; Learn-then-Test; Clipped BagNet; PatchGuard; DeepSloth; CNN-F; Singh et al.; Geiping et al.; Sun et al.) | fetched in the earlier dossier (`RHAN_IM_02`, Tier A); not re-fetched here | earlier [R] |
| Huber 1965; Kalman filtering and innovation gating; switching state-space models; Dempster–Shafer; AGM; Doyle/de Kleer truth maintenance; active inference | not re-fetched | [U] |
| Saha & Ramdas robust e-values | identified, not read | [L] |

I fetched three further arXiv abstracts after the last context reset that I can no longer list reliably; **I do not cite them.**

## Decisions and prerequisites

**You have decided:** approve the scientific direction, Option 4, and the separation of H-BR from C1; prepare this document for `docs/research/` after the corrections; **do not** freeze Gate 1; **do not** implement R1, R2 or any hybrid.

*Status note (13 Nov):* you have since recorded decisions on items 1, 2, 4 and 5 below; they are being checked against the repository in a separate Gate-1 freeze audit, which is not part of this document. Gate 1 is still **not frozen**, and the list below is kept as it stood when the document was written.

**Still open before Gate 1 can be frozen (yours to decide, with my evidence above):**

1. **Class split.** Choose an explicit rule for the freeze file: a seeded random permutation or a stratified split, with the 50 + 50 classes written out (§E.3). The list question is answered (CMC ImageNet-100).
2. **Image pool and the LTT unit.** 25 images per class per half, or hold out more from `train/`; state the unit as (target image, background image) with every image used once, both roles (§E.5).
3. **Sampler discrepancy — resolved** by your run; quote the expected values in the freeze (§E.1).
4. **Slope heterogeneity.** Report-only distribution, or a gated cap set before data (§E.2).
5. **Wording of the 2-of-3 rule** as engineering consistency (§E.4).

**Not before Gate 1 and Stage 1b results:** anything in Stages 3–5.
