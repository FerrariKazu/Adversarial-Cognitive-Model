# 03 — Candidate architectures and the decision

Seven candidates, each described under the same ten headings (A identity · B core mechanism · C mathematics · D information flow · E complexity · F expected advantages · G falsifiable predictions · H failure conditions · I implementation and reuse · J novelty against published work). Scores in §8 are my judgements against your selection criteria, not measurements. "Verified here" means executed in this branch; `RHAN_IM_RESULTS.md` has the numbers.

The organising idea comes from `RHAN_IM_01_FORENSIC_AUDIT.md` F5: **recurrence over a fixed input recomputes; it acquires nothing.** Candidates differ in whether they acquire information (C1, C2 with windowed looks, C5) or only recompute (C3, C4, C6, C7).

---

## C1 — RHAN-IM: perception as search with a stopping rule (CHOSEN)

**A. Identity.** RHAN is an *ideal searcher with measured sensors*: its recurrent state is a Bayesian posterior over *what is where* (exact for the specified sensor model; the optional per-unit cap makes it an explicit capped recursion instead) plus a map of *where it has not yet looked well enough*; it chooses where to look next by a computed rule, stops when a finite-sample-controlled risk is small, and can state how much any single region could have changed the decision. There are **no recurrent weights**; everything that is learned lives in the sensor.

**B. Core mechanism.** Hypotheses `H_0` (nothing there) and `H_(b,c)` (class `c` anchored at cell `b`). Each look delivers per-unit scores `z_{u,c}` and a **measured** detectability `d_u`. Evidence units `u` and hypothesis cells `b` are different sets, so a coarse whole-scene (gist) look and native-resolution windows feed one posterior. Next look: closed-form Koopman ignorance-reduction gain. Stop: when posterior risk ≤ λ, with λ chosen by Learn-then-Test. Influence: clip `κ`, per-unit cap, exact certificates.

**C. Mathematics.** With footprint `F ∈ {0,1}^{N×V}` and standardised scores `z_u ~ N(d_u F_{bu}·1[c = c*], 1)`:
`LLR(H_(b,c) : H_0) = Σ_u F_{bu} ( d_u z_{u,c} − d_u²/2 )`,
posterior `∝ prior · exp(LLR)`. *Independent looks* ("sum"): contributions accumulate; *nested deterministic views* ("best"): only the best-`d` view of a unit counts. *Within one look* tokens are treated as independent; for equicorrelated tokens the exact log-likelihood ratio is the independent sum divided by `VIF = 1 + (n − 1)ρ`, which the implementation reproduces by dividing `z` and `d` by `√VIF_null` (identity checked against the multivariate Gaussian; `VIF_null` is measured at gate 1). Scene-level *shared difficulty* is a different cause (an exact scale mixture), told apart by `VIF_pos / VIF_null` and flagged rather than shrunk. Ignorance of hypothesis `b`: `Φ(−√D_b / 2)` with `D_b = Σ_u F_{bu} D_u` (accumulated precision). Koopman gain of look `k`: `Σ_b p_b [Φ(−√D_b/2) − Φ(−√D_b'(k)/2)]`. LTT: ordered threshold grid, exact binomial p-values on calibration errors, fixed-sequence testing ⇒ `P(R(λ̂) ≤ α) ≥ 1 − δ`. Certificates: interval curtailment, exact patch enumeration, minimum looks to decide.

**D. Information flow.** canvas → {gist look, native windows} → trunk tokens → evidence head → standardise with measured null statistics → scatter to units → posterior update → ignorance map → policy → next look / stop → decision with certificate. The posterior never feeds back into the trunk; control flow is the only recurrence.

**E. Complexity (real trunk, `RHAN_IM_model_budget.json`).** New trainable parameters: 38,500 (a 384→100 linear head) with a frozen trunk; 21.59 M if the trunk is also tuned; 0 for state and policy. One look = 2.912 GFLOPs forward (112-px window, same trunk as `ff_112`). A scene with a gist look + 7 native looks ≈ 8× one feedforward pass. State: `u_contrib (B,320,100)`; Bayes update ≈ 0.6 % of a look; closed-form policy ≈ 0.15 %; exact lookahead (EVOI) is 3.17 GB unchunked at B = 64 and is therefore not used on the real model.

**F. Expected advantages.** Calibrated confidence by construction *when the sensor is measured and its likelihood is right* (ECE 0.002–0.007 with the true `d′`, 0.08–0.13 when `d′` is 30 % too high; **simulated** in the toy world, and on a real trunk the likelihood must first pass gate 1); a better accuracy-vs-looks curve than the strongest non-adaptive coverage schedule (+2.4 to +12.0 pp at 8 looks across the simulated worlds, depending on how fast visibility falls off; one non-significant −0.7 pp at 4 looks in one profile); a stopping rule whose error rate is guaranteed rather than hoped for, under the independence and split assumptions of Learn-then-Test (`04` §6) (toy world: plug-in thresholds violated a 10 % target in 27.6 % of calibration draws at n = 300; LTT in 1.3 %); an auditable ledger of what was and was not seen; benchmark-independent properties (calibration, risk control, certificates).

**G. Falsifiable predictions** (to be tested on the real sensor, gates in `05`/`07`):
1. *Sensor adequacy.* Token scores of the frozen `ff_112` trunk + linear head have z-ROC slope in [0.8, 1.25] and null sd in [0.85, 1.2]; the within-look variance-inflation factor is measured and corrected if above 1.25.
2. *Looking closer pays.* Native-window centre `d′` ≥ 1.5× the gist `d′` for the operating target size, and the corrected block-level detectability of one clean look is ≥ 2.0.
3. *Policy.* At 4 looks on composited scenes the Koopman policy beats fixed-lattice coverage by ≥ +2 pp (95 % CI excludes 0) under matched FLOPs.
4. *Calibration.* ECE ≤ 0.03 with measured `d′`; ≥ 0.08 when `d′` is replaced by a constant.
5. *Risk control.* LTT violation rate ≤ δ = 0.10 on held-out calibration draws; plug-in ≥ 2× δ at n_cal = 300.
6. *Robustness.* Against adaptive white-box attacks on the whole pipeline at ε = 5.417/255, worst-of-3 robust accuracy ≥ the ff baseline at ≤ equal FLOPs; certified patch fraction for m = 1 window ≥ 50 % on the operating profile.

**H. Failure conditions.** `d′` at the native window is no larger than at the gist look (no resolution advantage on this trunk/target size); z-ROC slope far from 1 (equal-variance model false and not repairable by a heteroscedastic likelihood); looks are nearly perfectly correlated (ρ ≈ 1) *and* regions never add new information; the tokens of one window are so dependent (`VIF` ≫ 1) that a look is worth only a few independent observations and the corrected block-level `d′` falls below 2 (then the look, not the token, is the unit of evidence); the certificate is vacuous at attainable confidence (`κ` against `G_up = κd − d²/2` — found in simulation); an adaptive attacker defeats the policy at equal budget; the sensor needs more than the head to be informative at 112 px (then tune the trunk with dense window CE — still inside C1).

**I. Implementation and reuse.** New package `rhan_im/` (additive). Reuses `CompactViTGen2` through `ff_token_fn` (exactly the ff path), the frozen `ff_112` checkpoint, the TRADES loop (for the later objective), the evaluation harness, `phase5_sdt` conventions. Tests include brute-force exact-Bayes and exhaustive-adversary checks (count in `RHAN_IM_00_README.md`). Prototype status: mathematics and simulations done; trunk adapter, budget and measurement script done; real-trunk numbers pending gate 1.

**J. Novelty.** Components are published (Najemnik & Geisler; Akbas & Eckstein; Rashidi et al.; Naghshvar & Javidi; Jazbec et al.; Clipped BagNet/PatchGuard). The combination, the trunk-measured sensor with the adequacy test, and the exposed design constraints are, to my knowledge, new; see `RHAN_IM_02_LITERATURE_SYNTHESIS.md` §5. Not claimed: optimality over multi-step plans, superiority over learned policies.

---

## C2 — Fast-weight scene field

**A.** RHAN's state is a small *model of the scene* (a matrix of fast weights) that is trained online, look by look, rather than a vector or a posterior.
**B.** Each look yields key/value pairs `(k_t, v_t)` from its tokens; the state absorbs them with a delta rule; a query reads the field.
**C.** `S_t = S_{t−1} + β (v_t − S_{t−1} k_t) k_tᵀ` — one step of online least squares on `½‖v − S k‖²`; read-out `r = S q`; with unit-norm keys `‖ΔS‖_F ≤ β‖v − S k‖`.
**D.** window tokens → (k, v) → S → readout → classifier.
**E.** State `d² = 147 k` floats per image at `d = 384`; projections ≈ 0.44 M parameters; one outer product per token.
**F.** Constant-size memory over arbitrarily many looks; a natural fit for temporal streams.
**G.** Held-out masked-patch reconstruction error falls monotonically with looks; accuracy at K looks ≥ gist-only by ≥ 3 pp.
**H.** No calibration or risk statement comes for free; the update is linear, so the state cannot represent "not yet seen"; on a fixed input it only recomputes.
**I.** New module; reuses trunk tokens. Cheap to try; no certificates.
**J.** Test-time-training layers (Sun et al.) and fast-weight programmers exist; not evaluated as a perception state with measured evidence.

## C3 — Contractive / equilibrium perception

**A.** RHAN's state is the *fixed point* of a contraction: perception is what remains once the dynamics settle, and the dynamics are provably stable.
**B.** `z* = f(z*, x)` with `Lip_z(f) ≤ L < 1`; iterate to the fixed point or solve implicitly.
**C.** Banach: unique `z*`, error `≤ L^t ‖z_1 − z_0‖ / (1 − L)`; sensitivity `‖z*(x) − z*(x′)‖ ≤ Lip_x(f) ‖x − x′‖ / (1 − L)`.
**D.** x injected at every step; state iterated; classifier on `z*`.
**E.** `T` applications of `f` (shared weights); implicit differentiation gives O(1) activation memory.
**F.** A certificate of state stability that does not depend on an attack.
**G.** Certified radius > 0 at `L ≤ 0.9`; clean accuracy within 3 pp of the unconstrained trunk.
**H.** Lipschitz-constrained transformers lose accuracy at ImageNet scale; a stable state is not a robust *decision*; recompute-only.
**I.** Hard to initialise from the DINOv2 warm start; needs spectral-norm layers.
**J.** Deep equilibrium models and Lipschitz networks exist; the perceptual-state framing adds little.

## C4 — Multi-timescale frequency-coupled dynamics (the ChatGPT proposal, treated as one candidate)

**A.** RHAN is a *two-clock system*: a slow state integrates coarse (low spatial-frequency) structure, a fast state refines detail, and the slow state gates which fine evidence is trusted.
**B.** `x = LP_σ x + HP_σ x`; `s_{t+1} = (1−α) s_t + α φ(LP x, f_t)`, `f_{t+1} = ψ(HP x, s_t)`.
**C.** Stability by the spectral radius `ρ(J) < 1` of the coupled Jacobian; the slow/fast ratio `α` sets the time-scale separation.
**D.** one image → two band-limited views → coupled recurrence → readout.
**E.** About 2 trunk passes per step plus filters; cheap.
**F.** A structural bias against perturbations concentrated at high frequency.
**G.** Fourier-sensitivity maps (Yin et al.) shift towards lower frequencies; robustness to high-frequency perturbations rises; full-band ℓ∞ PGD robustness changes by < 1 pp.
**H.** `LP` and `HP` of one image are functions of the same pixels — **it acquires no information** (F5). A full-band ℓ∞ attacker is not bound by the frequency story. The frequency-robustness link is dataset-dependent.
**I.** Easy: two filtered inputs through the existing trunk with a gating state. No executable frequency mechanism exists in the repo today (`RHAN_IM_01_FORENSIC_AUDIT.md` §3).
**J.** Multi-timescale recurrence and Fourier-robustness findings exist; the specific coupling is unevaluated. It deserves a **cheap test** (see §9) but not the project's centre.

## C5 — Severe testing: an internal adversary that tries to falsify the current belief

**A.** RHAN perceives by *attempting to refute its own best hypothesis* and reports only hypotheses that survived a severe test.
**B.** The next look maximises the expected evidence *against the leading hypothesis relative to its runner-up*.
**C.** For two Gaussian-SDT hypotheses differing on unit set `Δ`: `KL(h* ‖ h_r) = ½ Σ_{u∈Δ} d_u²` (Chernoff–Stein exponent); look `argmax_k Σ_{u∈Δ} d_u(k)²`; severity = probability the test would have discriminated had `h*` been false.
**D.** posterior → top-2 hypotheses → contrast set → look.
**E.** Same as C1 plus a top-2 contrast (negligible).
**F.** Efficient when the decision is a close call.
**G.** At equal looks it beats Koopman on scenes whose top-2 posterior is near 50/50, loses elsewhere.
**H.** It is a *policy* choice inside C1, not a separate architecture.
**I.** One function in `rhan_im/policy.py` (to add as an ablation).
**J.** Active sequential hypothesis testing (Chernoff; Naghshvar & Javidi). **Decision: absorbed into C1 as a policy ablation.**

## C6 — Byzantine-resilient token consensus (a cellular-automaton ViT)

**A.** Tokens are cells that agree by *robust aggregation* of their neighbours' messages; a corrupted minority cannot move the consensus.
**B.** `h_i ← h_i + η · TM_β({φ(h_j, h_i)}_{j ∈ N(i)})` with a trimmed mean (or geometric median) instead of a softmax-weighted sum.
**C.** With trimming fraction `β > f/|N|` the aggregate stays inside the coordinate-wise range of the honest messages: influence of `f` corrupted neighbours is bounded by the honest spread.
**D.** local diffusion over the token grid, many rounds.
**E.** `O(n² d log n)` for sorting-based aggregation; much slower than attention.
**F.** Bounded influence of a patch smaller than `β` of any neighbourhood.
**G.** Patch-attack success falls with `β`; clean accuracy within 2 pp.
**H.** Large patches exceed any neighbourhood's trim budget; coordinate-wise robust statistics lose the geometry; recompute-only; hard to warm-start.
**I.** A new attention variant; no checkpoint reuse.
**J.** Robust aggregation (distributed learning), PatchGuard-style masking, neural cellular automata exist; no evidence at ViT scale in the literature I read.

## C7 — Looped compute scaling

**A.** RHAN is a ViT whose *depth is elastic*: the same block group is applied `r` times with input injection and exits adaptively.
**B.** `z_{r+1} = z_r + R(z_r, e)`; exit when `KL(p_r ‖ p_{r−1}) < τ`.
**C.** Accuracy modelled as a function of `r` (log-linear until a plateau in language models, Geiping et al.).
**D.** one image → `e` → `r` loops → head.
**E.** `r ×` block-group cost; fits 8 GB comfortably.
**F.** Anytime inference.
**G.** Accuracy improves with `r` at matched FLOPs over the 12-layer baseline.
**H.** Recompute-only; it is the repository's tied refinement block under another name (`RHAN_IM_01_FORENSIC_AUDIT.md` F3, F5); DeepSloth-style attacks inflate `r`.
**I.** Already in the repo (`TiedRecurrence`).
**J.** Latent-recurrent depth (Geiping et al.); little new for vision.

---

## 8. Selection matrix (1–5; my judgement against your seven criteria)

| Criterion | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Differentiation from "ViT + extra blocks" | 4 | 3 | 3 | 4 | 3 | 3 | 1 |
| Mathematical mechanism that is not a re-labelling | 5 | 4 | 5 | 3 | 4 | 3 | 2 |
| Defensible hypothesis (acquires information) | 5 | 3 | 3 | 2 | 3 | 3 | 2 |
| Measurable performance, clear metrics | 5 | 3 | 4 | 4 | 3 | 4 | 4 |
| Contribution that survives small accuracy gains | 5 | 2 | 4 | 2 | 3 | 3 | 1 |
| Feasible on an RTX 4060 8 GB / Colab T4 | 4 | 4 | 2 | 4 | 4 | 3 | 4 |
| Clear validate/reject path | 5 | 3 | 4 | 3 | 3 | 3 | 4 |
| **Total / 35** | **33** | 22 | 25 | 22 | 23 | 22 | 18 |

## 9. Decision

**Commit to C1, RHAN-IM**, with C5 as a policy ablation, C2/C3/C6 retained as named alternatives, and C4 and C7 retained as *cheap controls* (they are the "recompute" family, which is the right negative control for the thesis).

Reasons, in order of weight:
1. It is the only candidate whose recurrence *acquires* information, which is the flaw in what exists today (F5).
2. Its contribution does not depend on a benchmark delta: calibration, risk control and certificates are properties that can be verified or refuted; the mathematics is verified by tests and its behaviour was simulated in toy worlds, not measured on a trained network.
3. It turns the lab's strongest habit — measure, pre-register, falsify — into the architecture (`d′` is measured, never learned).
4. It has a one-day falsification: gate 1 (measure `d′` on the frozen `ff_112` trunk) can kill it before any training.
5. It is feasible: 38.5 k new trainable parameters, a frozen trunk, one look = one `ff_112` forward pass.

**Switch conditions (written before the evidence).**
- Gate 1 *sensor adequacy* fails (slope outside [0.8, 1.25] after the head is fitted) ⇒ use a heteroscedastic likelihood or fit the head on windows with dense window CE; if it still fails the LTT stopping layer remains valid (it is model-free) but the exact-Bayes claims are dropped.
- Gate 1 *informative* fails (centre `d′` < 1.5) ⇒ unfreeze the last blocks with dense window CE (still C1); if `d′` stays below 1.5 the premise "this trunk resolves small targets in clutter at 112 px" is false ⇒ move to **temporal looks** (frames as looks; `RHANfuture.md` Gap 1) with the same posterior machinery, or C3 for certificates.
- Gate 1 *looking closer pays* fails (native/gist < 1.5 for every size tried and for a 448-px canvas) ⇒ resolution is not the bottleneck on this trunk; RHAN-IM degenerates to calibrated selective classification (publishable, smaller). Keep that fallback.
- Cross-look correlation ρ ≥ 0.6 ⇒ use "best" combination (conservative); not a failure.
- If the replication of the Stage-0 comparison (Stage A, optional) came out positive, run C4/C7 as controls to test whether a recompute mechanism explains it.

**What the choice does *not* do:** it does not invalidate any existing checkpoint, baseline or evaluation; they become the baseline arms of `RHAN_IM_05_EXPERIMENTS.md`.
