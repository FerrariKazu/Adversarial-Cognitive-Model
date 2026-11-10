# 04 — RHAN-IM: formal architecture specification

Status: **specification + working prototype of the mathematics and the sensor adapter; real-trunk detectabilities not yet measured** (gate 1, `RHAN_IM_07_GATE1_PREREG_DRAFT.md`). Every number in §9 is computed from the real `CompactViTGen2` at random init (`RHAN_IM_model_budget.json`) and the shapes are asserted by an end-to-end test at the real geometry (`tests/rhan_im/test_real_trunk_integration.py`); every behavioural statement is labelled *brute-force verified* (tests), *simulated* (toy worlds, mechanism-level only) or *not yet measured on the real trunk*.

## 1. Central research question

> On cluttered scenes in which the label-relevant object occupies a small, resolution-limited region, does a recurrent state that is a **Bayesian posterior over what-is-where, exact for the specified sensor model, with measured sensor reliabilities** (RHAN-IM) reach a prescribed error rate in **fewer looks** than fixed-coverage and feedforward-gist baselines at matched compute, while (i) its stopping rule's error rate stays below a chosen α with probability ≥ 1 − δ, (ii) its confidence is calibrated, and (iii) the influence of any single look on the decision is bounded and, where the geometry allows, certified?

Measured by: accuracy-vs-looks and looks-to-target-risk (paired, bootstrap CIs); LTT violation rate over calibration draws; ECE; certified fraction and adaptive-attack success (`RHAN_IM_05_EXPERIMENTS.md`). A "no" on any of (i)–(iii), or on the looks comparison, is a reportable result.

## 2. Thesis

*Recurrence earns its cost only where information is acquired, not merely recomputed.* In RHAN-IM every recurrent step is a **look** that exposes pixels at a resolution or location the previous looks did not (a native-resolution window of a larger canvas; the first look is a coarse whole-scene gist). The recurrent state records what has been exposed and how well.

## 3. Objects and notation

| Symbol | Meaning | Real-trunk value |
|---|---|---|
| canvas `H×H` | scene, patch `p = 14` | `H = 224`, `g = H/p = 16` fine units per side |
| window `w` | native look side = trunk input size | 112 px ⇒ `tok = 8`, 64 tokens |
| coarse block | fine units per gist token side | 2 |
| `N` | hypothesis anchors (one per fine unit) | 256 |
| `V` | evidence units = `N` fine + `tok²` coarse | 320 |
| `K` | candidate looks = native windows on a stride-2 lattice (`(g−tok)/2+1 = 5` per axis ⇒ 25) + 1 gist | 26 |
| `C` | classes | 100 (50 target classes in gate 1) |
| `F ∈ {0,1}^{N×V}` | footprint: units a target anchored at `b` occupies (fine units) or overlaps (coarse units) | built by `UnitLayout.footprint()` |
| `H_0`, `H_(b,c)` | no target; class `c` anchored at `b` | `1 + N·C` hypotheses |
| `z_{u,c}` | standardised score of class `c` at unit `u` from one look | |
| `d_u(k)` | **measured** detectability of unit `u` from look `k` | `d_tok (8×8)`, `d_gist` |
| `κ`, `u_cap` | score clip; per-unit accumulated-evidence cap | design parameters |
| `VIF_tok`, `VIF_gist` | within-look variance-inflation factor of the `d`-weighted token sum (1 = tokens are independent evidence) | measured at gate 1 |
| `λ` | stopping threshold on posterior risk | chosen by LTT |

## 4. Generative model (what the Bayes state assumes)

A scene contains no target (prob `p_0`, default 0.4) or one target `(b, c)` with prior `π(b,c)`. For a look `k`, scores are conditionally independent given the hypothesis:
`z_{u,c'} ~ N( d_u(k) · F_{bu} · 1[c' = c], 1 )` under `H_(b,c)`, and `N(0,1)` under `H_0`.
Hence `log [P(data | H_(b,c)) / P(data | H_0)] = Σ_u F_{bu} ( d_u z_{u,c} − d_u²/2 )`, exactly, for independent looks (**"sum" mode**). If the looks are nested views of the same pixels their noise is strongly correlated across looks, and **"best" mode** keeps only the best-`d` view of each unit (a better view *replaces* a worse one). "Best" is the exact Bayes combination only for a *degradation chain*: two Gaussian views whose noise correlation is ρ = `d_worse / d_best` (the worse view is the better one plus independent noise). At any other ρ the exact weights are Σ⁻¹d, not (0, `d_best`): for `d = (0.5, 1.5)` the exact information is 2.25 (= `d_best²`, what "best" keeps) only at ρ = 1/3, it is 2.5 at ρ = 0 (where "sum" is exact) and at ρ = 0.6, and it grows without bound as ρ → 1 (`tests/rhan_im/test_belief_bruteforce_extended.py`). So "best" is a **deliberate approximation that discards information**, not the Bayes rule. The posterior it gives is calibrated for the observation it keeps, provided the choice of look did not depend on the views it discarded; where it did (a worse view arrives first and steers the policy to a better one) a selection effect is possible and is not bounded here. In the simulated nested-view world, where the adaptive policy does exactly that, the ECE of "best" with the measured table was 0.008–0.035 over three profiles. Which mode applies is **measured** (cross-look correlation ρ, gate 1), not chosen, and ρ summarises a pair of looks; it does not prove the degradation structure. Mis-specifying the rule is costly: on the same nested evidence "sum" gave mean confidence 0.95 at accuracy 0.555 in the easiest profile after 8 looks, and ECE 0.15–0.64 over the three profiles (simulated image world).

*Within a look* the model also treats the tokens as independent. A ViT window is one global-attention computation, so that is an assumption about the trunk, and it is **measured** (gate 1, `gate1.block_vif`) as a variance-inflation factor `VIF = Var(Σ_u w_u z_u) / Σ_u w_u² Var(z_u)` over a block of tokens (1 = independent, `n` = fully dependent). It is measured twice: `VIF_null` on scenes without a target (every class channel; a clean `H_0` estimate) and `VIF_pos` on the own-class scores of scenes with a target (residual around the per-position mean), **using only looks that contain the whole target**: a VIF grows with the number of tokens summed, so pooling windows that cut the target at different places makes `VIF_pos` incomparable with the `block × block` `VIF_null` and biases it low (a first version did that and read low under strong dependence; `test_block_vif_pos_counts_only_looks_that_cover_the_whole_target…` pins the effect). The two separate two causes of dependence, which need different remedies:

- *Dependent noise* (the tokens of a window share context): `VIF_null > 1`. For equicorrelated tokens (correlation `ρ`) with equal `d` the exact log-likelihood ratio is `(1/VIF)·Σ_u (d z_u − d²/2)` with `VIF = 1 + (n − 1)ρ`, and the information is `n d² / VIF`. Dividing `z` and `d` by `√VIF_null` inside the unchanged update reproduces it exactly **when the `n` footprint tokens are the whole observation and share one `d`** (**Lemma 1b**, checked against the multivariate Gaussian in `tests/rhan_im/test_gate1.py`; `EvidenceCalibration.vif_tok / vif_gist` apply it). It is not exact outside that case, and it errs towards discarding information (`tests/rhan_im/test_dependence_calibration.py`): (a) with unequal `d` inside the block the exact weights Σ⁻¹d are not proportional to `d` (at ρ = 0.35, 16 tokens with `d` rising from 0.3 to 0.9, the exact information is 1.76 against 1.01 for the closed form); (b) when the same noise also sits on tokens outside the footprint, those tokens act as a noise reference (negative weights in Σ⁻¹d) that the block-only form never uses: for 16 footprint tokens at `d = 0.6` inside a window of 64 equicorrelated tokens the exact information is 5.0 at ρ = 0.1 and 6.7 at ρ = 0.35, against 2.3 and 0.9 for the closed form. The corrected posterior stays calibrated for the statistic it uses, but gate 1's `d_block` is read from the closed form, so it can fail a trunk that is more informative than `d_block` says when its noise is shared with the rest of the window. Calibration (E2) is where the rest is tested. Ignoring it costs in proportion to `VIF`: sixteen tokens at `ρ = 0.35` have `VIF = 6.25` and are worth `16 / 6.25 = 2.6` independent observations, so the independent sum overstates the log-odds 6.25-fold.
- *Shared difficulty* (**Lemma 1c**): a scene-level difficulty multiplier `s ≥ 0` (prior `π_s`, mean 1, coefficient of variation `cv`; not the clip `κ`) scales every unit's detectability, `z_u | s ~ N(s d_u, 1)` for `u` in the footprint: a target that is easy at one token is easy at its neighbours, so the dependence is between the target's tokens, not in the background noise. With the footprint statistics `S = Σ_u d_u z_u` and `D = Σ_u d_u²` (already divided by `VIF_null` if the noise is dependent too) the exact likelihood ratio against `H_0` is `∫ exp(sS − s²D/2) π_s(s) ds`, computed as `logsumexp_j [log π_j + s_j S − s_j² D/2]` on a quantile grid; it reduces to `S − D/2` for `s ≡ 1`. If `s` is a property of the target image it is shared across looks and `S, D` accumulate over them. Its symptom is `VIF_pos > VIF_null`, and for equal `d` and equicorrelated noise the spread is recoverable from the two VIFs: `cv² = (VIF_pos − VIF_null) / ((n − VIF_pos) d²)` (`gate1.difficulty_cv_from_vif`, inverted against the analytic VIF in `test_gate1.py`).

**Rule.** If `VIF_null > 1.25` the tokens are corrected by `1/√VIF_null`. If `VIF_pos / max(VIF_null, 1) > 1.25` shared difficulty is **flagged and not shrunk**: shrinking by `VIF_pos` is the wrong remedy for it (a first draft did that; ECE 0.046 against 0.034 unshrunk in the pure-difficulty world below), the right one is the mixture, which needs `(S, D)` per hypothesis instead of per-unit log-weights, a change to `belief.py` that has **not** been made. The mixture is therefore validated here only as a calculation on simulated scores, and its adoption is conditional on the flag firing on the real trunk. `tests/rhan_im/test_dependence_calibration.py` pins the claims; `scripts/rhan_im/dependence_calibration_check.py` produced the numbers (`docs/research/RHAN_IM_dependence_calibration.json`, SIMULATED scores, one block of 16 tokens, equal priors; "full rule" = correct by `VIF_null`, then the mixture with `cv` fitted from the measured VIFs if flagged):

| world (16 tokens, `d` = 0.6 per token, 200 000 scenes) | `VIF_null` | `VIF_pos` | fitted cv | ECE naive | ECE after `1/√VIF_null` | ECE full rule | ECE exact |
|---|---|---|---|---|---|---|---|
| independent | 1.00 | 1.01 | 0.00 | 0.001 | 0.001 | 0.001 | 0.001 |
| correlated noise (ρ = 0.35) | 6.24 | 6.23 | 0.00 | 0.235 | 0.002 | 0.002 | 0.002 |
| shared difficulty (cv 0.4) | 1.00 | 1.82 | 0.40 | 0.034 | 0.034 | 0.002 | 0.002 |
| both, mild noise dependence (ρ = 0.10) | 2.49 | 3.24 | 0.40 | 0.138 | 0.023 | 0.003 | 0.003 |
| both, strong noise dependence (ρ = 0.35) | 6.24 | 6.77 | 0.00 | 0.244 | 0.010 | 0.010 | 0.002 |

What the table shows and does not show. (i) Correlated noise is the larger danger (the independent sum is overconfident: mean confidence 0.92 at accuracy 0.69) and the `VIF_null` correction removes it. (ii) Difficulty alone is a smaller effect at this operating point (ECE 0.034, just above the E2 target of 0.03) and only the mixture removes it; the VIFs recover the true spread (fitted 0.40, true 0.40). (iii) **Blind spot:** with strong noise dependence (`VIF_null` 6.2) the ratio `VIF_pos / VIF_null` is only 1.09 at the same `cv`, the flag does not fire, and the residual ECE after the `VIF_null` correction alone is 0.010 against 0.002 exact; the first-draft rule happens to do better there (it over-shrinks), but it is worse in the pure-difficulty world, so it is not restored. The flag is deliberately conservative: the sampling noise of `VIF_pos` at the planned sizes can by itself move the ratio by a few percent, and a flag that fires on noise is worse than one that is blind to a 0.010 ECE. (iv) The mild combined world has a ratio of 1.30, a hair above the threshold. (v) Nothing here says the real trunk has any of these properties.

*Clip:* every score is clipped to `[−κ, κ]` at update time (deliberately mis-specifies the likelihood slightly in exchange for bounded influence). *Cap:* each unit's accumulated evidence is clipped to `[−u_cap, u_cap]` **after every look**, so with a cap the result depends on the order of the looks (scores +3, +3, −3 at `d = 1` and cap 2 end at −1.5 in that order and at +2.0 in the order −3, +3, +3; without the cap both end at +1.5): the cap is a saturation of the running state, not the posterior of an exchangeable-looks model.

**Lemma 1 (exactness).** Without the cap, and with the clip read as part of the model (the clipped score is treated as the observation, with an `N(d,1)` density under the hypothesis and `N(0,1)` otherwise), the implemented posterior equals brute-force enumeration over all `1 + N·C` hypotheses, with and without footprints, for `N ≠ V`, in both modes, with an active mask and with branching. With the cap it equals the explicit clipped-and-capped recursion written out in the test (not a Bayes posterior; see above). *Brute-force verified*: `tests/rhan_im/test_belief.py` (footprints, `N ≠ V`, both modes) and `tests/rhan_im/test_belief_bruteforce_extended.py` (clip, cap, mask, branching; an enumeration written independently of the implementation by a separate reviewer, agreeing to 1e-12).

## 5. State, policy, stopping

**State** (per scene): `log_prior (N,C)`, per-unit accumulated evidence `u_contrib (V,C)`, best detectability `d_best (V)`, accumulated precision `D2 (V)` (= `Σ d²` in sum mode, `d_best²` in best mode), influence `S1 (V)`, look counter. `log_cells = log_prior + F·u_contrib`; `H_0` has log weight `log p_0`. **No recurrent weights.** The **ignorance map** is `D2` projected through the footprint: `D_b = Σ_u F_{bu} D2_u`; a target at `b` would still be missed with probability `Φ(−√D_b / 2)` (the equal-prior Bayes error of telling `H_b` from `H_0`).

**Policy (computed, not learned).** For candidate look `k`, with `p_b` the marginal posterior mass of anchor `b`:
`G(k) = Σ_b p_b [ Φ(−√D_b/2) − Φ(−√D_b'(k)/2) ]`, `D_b'(k)` = precision after the look. This is the discrete analogue of Koopman's marginal-return rule; `O(B·K·N·V)` via the footprint einsum. The exact one-step lookahead `E[max_o P(o | data, new look)]` (posterior-predictive Monte-Carlo with common random numbers, **EVOI**) is the reference; baselines: MAP-cell fixation, fixed-lattice coverage ("tiling", the strongest non-adaptive rule), random, random without replacement. *Simulated (toy world, 2 000 paired scenes per profile, `RHAN_IM_RESULTS.md` A2):* EVOI minus Koopman is within 1.6 pp at every budget of 2, 4, 6 and 8 looks in both visibility profiles (largest: +1.55 pp at 4 looks, window profile), and every paired 95 % interval includes 0, the window profile's at 4 and 6 looks only just (lower bounds −0.23 and −0.02 pp). That is "no difference detected at this sample size", not equivalence. In the image world (B; 2 000 scenes; paired intervals are computed against tiling only, none for this pair) EVOI is ahead of Koopman at 4 looks by 2.9 and 2.0 pp in the two hard profiles and 0.2 pp behind in the easy one, and 2.3 pp behind at 8 looks in the sharp-blur profile: the closed form gives up up to about 3 pp of accuracy at some budgets for a policy that costs 0.15 % of a look (EVOI at `n_mc = 16` is 3.2 GB unchunked at `B = 64`).

**Stopping.** Stop at the first look `t ≥ k_min` (`k_min ≥ 1`, fixed before calibration) whose posterior risk `1 − max_o P(o)` is ≤ λ, else at the budget. `k_min` is implemented (`stopped_stats(…, k_min)`, `run_search(…, k_min)`); a **leave-one-look-out margin** (§7.4) is **not** implemented, so the rule LTT certifies has no such margin. Because the policy never depends on λ, the full-budget trace contains the trace of every stopping rule, so `stopped_stats` evaluates any λ offline from one run (*verified* against a run that really stops, for several λ including one above the prior risk, and for `k_min` = 1, 2, 3: `tests/rhan_im/test_agent.py`). λ is chosen by **Learn-then-Test** over an ordered grid (conservative → aggressive), exact binomial tail, fixed-sequence testing: `P(R(λ̂) ≤ α) ≥ 1 − δ` over the calibration draw **if the calibration scenes are independent draws from the deployment distribution (i.i.d.; exchangeability alone does not give the binomial tail), counted by source image rather than by scene, on a split disjoint from everything that was fitted upstream (evidence head, standardiser, `d′`, VIF, κ, policy settings), with the grid and `k_min` fixed before the calibration losses are seen.** A **selective** variant (loss = decided and wrong, with abstention) is used when no forced-answer λ is certifiable; it controls the *joint* rate `P(decided ∧ wrong)`, not the error rate among decided scenes `P(wrong | decided)`, so its coverage must be reported with it.

## 6. Mathematical guarantees implemented (all about the evidence channel; none about the trunk)

Assume every score is clipped to `[−κ, κ]`, every `d ≤ d_max ≤ 2κ`, and the same `κ` is used in updates.

- **Lemma 2 (swing).** One look moves any hypothesis' log-weight by at most `Γ = κ d_max + d_max²/2` (`Γ`, not the grid side `g` of §3). In best mode the final contribution of a unit is a single clipped view, so the swing over any number of remaining looks is ≤ `2Γ`; in sum mode ≤ `remaining·Γ`.
- **Lemma 3 (lossless curtailment, interval form).** If the outcome log-margin exceeds the largest admissible future change, the decision at the budget equals the decision now. Units already seen at `d_cap` are frozen in best mode. *Exhaustively verified on small instances (soundness), no false certificate found.*
- **Lemma 4 (patch certificate).** If an adversary may rewrite the evidence of any `m` units (in every look), the decision is unchanged when the margin exceeds the exact worst-case net shift, computed by enumerating all `C(V,m)` unit subsets against the clipped intervals `[−κ S1 − D2/2, κ S1 − D2/2] ∩ [−u_cap, u_cap]`. *Exhaustively verified against extreme rewrites.*
- **Lemma 5 (minimum looks).** No admissible evidence can raise a hypothesis' log-odds against `H_0` faster than `max_d (κd − d²/2)` per look, so a threshold cannot be reached — by honest data or by an attacker — in fewer looks than `min_fixations_to_reach`. *Verified tight against the fastest attack for footprint 1.*
- **Theorem (LTT).** As cited, under the assumptions listed in §5 (i.i.d. calibration scenes by source image, a split disjoint from everything fitted upstream, grid and `k_min` fixed in advance; `rhan_im/stopping.py` cannot check them). *Simulated:* violation 1.3 % (n_cal 300) / 0.1 % (1500) at α = δ = 0.10 against 27.6 % / 5.8 % for plug-in thresholds in the well-specified toy world (`RHAN_IM_RESULTS.md` A3), where the assumptions hold by construction.

## 7. Design implications discovered by building and testing it

1. **Sum vs best must be measured** (§4). Gate 1 measures the cross-look correlation ρ of the standardised score of the same (scene, unit, class): ρ ≤ 0.2 ⇒ sum; ρ ≥ 0.6 ⇒ best; in between ⇒ best (conservative).
2. **`d′` must be measured per position and per look type.** A constant `d′` gave ECE 0.20–0.41 against 0.008–0.035 with the measured table over the three simulated image-world profiles (the E2 target of 0.03 is met in the easy profile and missed by 0.004–0.005 in the two hard ones at some budgets). The real pipeline therefore carries position-specific null mean/sd (`EvidenceCalibration.mu0, sd0`: shape `(2, 64, C)`) and a measured `d_tok (8×8)` and `d_gist`.
3. **`κ` trades certified robustness against attainable confidence.** In best mode a single unit can contribute at most `G_up = κ d − d²/2` to a hypothesis' log-odds; if `κ` is small the posterior can never become confident. `κ` is therefore set from the *measured* score distribution (e.g. the 99.9th null percentile), not from the robustness target alone.
4. **A minimum-looks floor.** For strong evidence × large footprints the premature-exit lower bound is trivial (1 look): a few extreme scores could end the search. `k_min` is implemented (default 1, which only forbids answering from the prior; recommended ≥ 2 for the real model, to be fixed before calibration; the simulations in this dossier use `k_min = 1`). A **leave-one-look-out margin** (stop only if the decision survives removing any single look) is the next remedy if P7 fails; it is **not implemented**, so nothing in this dossier certifies or tests it.
5. **Sum-mode influence grows linearly with looks** unless `u_cap` is set; certificates need a cap or best mode.
6. **Lossless curtailment is vacuous at realistic `κ·d` and unit counts** (0 % early exits in both modes, 0 mismatches). The compute saving comes from *risk-controlled* (LTT) stopping, not from provably lossless exits. Reported as a negative result.
7. **Patch certificates need multi-unit targets.** Roughly `w > 1 + 2κ/d` and `w d²/2 > d² + κ d` for a target spanning `w` units (derived analytically, consistent with simulation: with `κ = 4`, cap 8, 3×3 footprint, `m = 1` certified 71.3 %, `m = 2` 41.3 %; without the cap 17.5 % and 0 %).
8. **Local evidence is a requirement for per-unit certificates, and the real trunk does not provide it.** A ViT with global attention inside a window mixes all of its tokens, so an adversarial patch inside a window can alter every token of that look: the *unit of influence on the real trunk is the look*. That is weak here: with 112-px windows on a 224-px canvas and a stride of 28 px, a unit-aligned patch intersects on average 25 % of the 25 windows for a 14-px patch (range 4–64 % depending on where it sits), 33 % for 28 px and 52 % for 56 px (computed from the lattice geometry), so an attacker confined to one patch can corrupt several of the ≈ 8 looks a scene receives. A look-granular certificate has to certify against `m` corrupted looks with `m` of that order, and I expect the certified fraction to be modest. It must be implemented and measured (experiment E7) before any robustness claim is made for the real model. Until then: **no certificate is claimed for the real trunk.** Mitigations to test, not assumed: PatchGuard-style token masking, smaller native windows, or a window-local evidence head.
9. **Tokens of one look are not independent evidence until measured to be.** The same global attention that makes the look the unit of influence also makes the tokens of one window share context, so their scores may be positively correlated, and a hypothesis whose footprint spans several tokens then collects less independent evidence than the sum says (§4, Lemma 1b). Gate 1 measures `VIF` on null windows (`VIF_null`: dependent noise) and on target windows (`VIF_pos`: with `VIF_null`, it separates *shared image difficulty*, Lemma 1c, from dependent noise). If `VIF_null > 1.25` the correction `1/√VIF_null` is applied and the informativeness test is made on the corrected block-level detectability `d_block = d_centre·√(n_op / VIF_null)` (≥ 2.0 ⇒ one clean look at the operating size has equal-prior error ≤ 15.9 %). If `VIF_pos / VIF_null > 1.25` shared difficulty is flagged, not shrunk (§4). A single constant `VIF` is exact only for equicorrelated tokens of one footprint size; a larger target has a larger `VIF`, so a constant fitted at the operating size is anti-conservative for bigger targets: E2 reports calibration by target size.

## 8. Information flow (one scene)

```
canvas (B,3,224,224)
   │  gist look: area-resample to 112 ─────────────┐
   │  native look k: crop window 112 px, NO resample│
   ▼                                                ▼
 trunk (ff path: prep → 12 blocks → norm), frozen unless stage S3
   │ tokens (B,64,384)      (CLS and register tokens are not used)
   ▼
 evidence head Linear(384 → C)                       → scores (B,64,C)
   ▼
 standardise with measured null mean/sd per (look type, token position, class), then ÷ √VIF_null (1 if tokens are independent)
   ▼
 scatter to evidence units via window geometry       → z (B,320,C), d (B,320) from measured d_tok / d_gist (÷ √VIF_null)
   ▼
 SearchPosterior.update (mode from ρ; clip κ; cap u_cap)
   ▼
 ignorance map D_b, marginals p_b, risk
   ├── risk ≤ λ̂ and k ≥ k_min? ── yes ──► decision over {absent, class c} + posterior prob + (look-granular) certificate
   └── no ──► Koopman gain G(k) over the K+1 candidates ──► next look (back to the top)
```
The posterior never feeds back into the trunk: control flow is the only recurrence.

## 9. Block-by-block, shapes, parameters, cost (real trunk; `RHAN_IM_model_budget.json`)

| # | Block | Input → output | Trainable parameters |
|---|---|---|---|
| 0 | `UnitLayout` | geometry; footprint `(256,320)`; windows `(25,2)` | 0 |
| 1 | look operator (`native_crop` / area resample) | `(B,3,224,224)` → `(B,3,112,112)` | 0 |
| 2 | trunk `CompactViTGen2` via `ff_token_fn` | → tokens `(B,64,384)` | 21,555,840 (frozen in S1/S2) |
| 3 | `EvidenceHead` | `(B,64,384)` → `(B,64,C)` | 38,500 at `C = 100` |
| 4 | `EvidenceCalibration` | buffers `mu0, sd0 (2,64,C)`, `d_tok (8,8)`, `d_gist`, `vif_tok`, `vif_gist` — measured, not trained | 0 |
| 5 | scatter | → `z (B,320,C)`, `d (B,320)` | 0 |
| 6 | `SearchPosterior` | `u_contrib (B,320,C)`, `log_cells (B,256,C)` | 0 |
| 7 | Koopman policy | → gain `(B,26)` | 0 |
| 8 | stopping + decision | → decision, risk, certificate | 0 |

Unused by this path (kept for the baseline arms): tied refinement block 1,775,232 parameters. Trunk used: patch embed 226,176; CLS+register 768; positional embedding (66×384) 25,344; 12 blocks 21,302,784; final norm 768.

**Cost.** One look = **2.912 GFLOPs** forward (FlopCounterMode; same as one `ff_112` forward). Gist + 7 native looks ≈ 8× one feedforward pass (23.3 GFLOPs). Bayes update ≈ 0.56 % of a look, Koopman policy ≈ 0.15 %. EVOI at `n_mc = 16` is 3.17 GB unchunked at `B = 64` — use the closed form on the real model.

**Memory.** Saved activations ≈ 21.2 MB per 112-px window in fp32 (autograd hooks, CPU math path; the GPU flash-attention path saves less). Trunk parameters 82.2 MB fp32; AdamW state 164.5 MB if the trunk is trained. Gate-1 head training holds one trunk forward without gradients. Adversarial training through `K = 8` looks costs ≈ 170 MB of saved activations per scene (arithmetic from the measurement), so micro-batches of ≤ 24 scenes are needed on an 8 GB card. Bayes state: `u_contrib` 7.8 MB, `log_cells` 6.3 MB per 64 scenes.

## 10. Training objective and schedule (staged; each stage can stop the project)

| Stage | What is trained | Objective | Gate |
|---|---|---|---|
| S0 | nothing | the validated `ff_112` checkpoint (seed 17, sha256 `d00fb750…`) is the frozen trunk | — |
| S1 | `EvidenceHead` (384→C+1 → exported as 384→C), 600 steps AdamW, lr 2e-3, cosine | token-level cross-entropy: class `c` on tokens overlapping a class-`c` target by ≥ 50 % (for unit-aligned targets that is every overlapping native token; for gist tokens 0 < overlap < 50 % is ignored), background elsewhere; background weight 0.25; export `score_c = logit_c − logit_bg` | **Gate 1** (measure `d′`, slope, ρ, VIF) |
| S2 | nothing (calibration only) | measure `mu0, sd0, d_tok, d_gist, vif_tok, vif_gist`; LTT on a calibration split; policy comparison on a held-out split | **Gate 2** (looks, ECE, risk control) |
| S3 | head + last 2–4 trunk blocks, only if gate 1 fails on `d′` | dense window CE with background class on composited scenes (sizes, opacities, backgrounds randomised); lr 1e-5 trunk / 1e-3 head; 5 epochs; early stop on held-out `d′`, **not** accuracy | re-run gate 1 |
| S4 | head + last blocks | TRADES on the **final posterior**: `L = 0.55·[CE(p_T(x), y) + β·KL(p_T(x_adv) ‖ p_T(x))]`, `β = 2.5`, PGD-4 on the canvas at ε = 5.417/255 ℓ∞ (the existing recipe), with the look sequence computed on `x` treated as non-differentiable | **Gate 3** (adaptive attacks) |
| optional | glimpse-JEPA (two overlapping looks as paired views) | embedding-consistency loss on overlap tokens | **risk:** raises ρ, which pushes towards "best" mode; test its effect on ρ first |

Reuses the existing AdamW grouping, warm-up/cosine, TRADES code and harness; the stage scripts are new files.

## 11. Falsifiable predictions and kill conditions

| # | Prediction | Kill / revise condition |
|---|---|---|
| P1 | sensor adequacy: z-ROC slope in [0.8, 1.25], null sd in [0.85, 1.2] after standardisation; within-look `VIF_null` is measured and the tokens corrected if > 1.25; shared difficulty (`VIF_pos / VIF_null` > 1.25) is flagged | slope outside range ⇒ heteroscedastic likelihood or S3; if still outside, drop exact-Bayes claims, keep LTT |
| P2 | centre `d′ ≥ 1.5` and corrected block-level `d_block ≥ 2.0` at the operating size; native/gist ≥ 1.5 | fail ⇒ S3; if still failing ⇒ temporal looks or fallback (selective classification) |
| P3 | Koopman beats fixed-lattice coverage by ≥ +2 pp at 4 looks, matched FLOPs (CI excludes 0) | not met ⇒ report; policy is not the contribution, calibration/risk control remain |
| P4 | ECE ≤ 0.03 with measured `d′`; ≥ 0.08 with constant `d′` | fail ⇒ likelihood misspecified; investigate before any claim |
| P5 | LTT violation ≤ δ over ≥ 500 calibration draws; plug-in ≥ 2δ at n_cal = 300 | LTT violation > δ ⇒ an LTT assumption is broken (i.i.d. scenes by source image, a split disjoint from everything fitted upstream, grid and `k_min` fixed in advance): find which |
| P6 | adaptive white-box worst-of-3 robust accuracy ≥ `ff` baseline at ≤ equal FLOPs; certified fraction (m = 1 look) ≥ 50 % | fail ⇒ no robustness claim; reported as measured |
| P7 | compute-exhaustion attack (force maximum looks) raises mean looks by < 2× at fixed ε | fail ⇒ add a leave-one-look-out margin (not implemented) and/or a cap on looks, and re-calibrate |

## 12. What is verified and what is not

| Claim | Evidence level |
|---|---|
| Posterior = the specified Bayes posterior (units ≠ cells, both modes, clip, mask, branching); with the cap it equals the explicit capped recursion, which is order dependent and not a Bayes posterior; "best" mode is exact only for a degradation chain | two brute-force enumerations (`test_belief.py`, `test_belief_bruteforce_extended.py`) and a Gaussian check of the "best" claim |
| Certificates sound (curtailment, patch, min-looks) | exhaustive-adversary tests on small instances |
| Policies, calibration, LTT behaviour | simulated (toy SDT world; Gabor-on-1/f image world with measured `d′`) — mechanism-level only |
| Sensor adapter geometry (crop, scatter, token ↔ unit) | unit tests incl. consistency with the compositing geometry |
| Checkpoint loading reproduces the `ff_112` token path | test with a freshly saved random-init checkpoint (bit-identical tokens) |
| Parameters, FLOPs, memory of the real trunk | measured on random-init `CompactViTGen2` (weight-independent) |
| The whole chain fits together at the real geometry (224 canvas, 112 window, 26 candidates, N = 256, V = 320, 100 classes) | `tests/rhan_im/test_real_trunk_integration.py` runs the real `CompactViTGen2` architecture (random weights) through the sensor, the Bayes state, the Koopman policy and the stopping rule; asserts the shapes of §9, that a look writes only to the units it exposes, batch independence, valid probabilities, determinism. It says nothing about perception |
| `d′`, slope, ρ, `VIF_null`, `VIF_pos` of the **real** frozen trunk | **not measured yet** (needs the checkpoint and ImageNet-100; script ready) |
| Within-look dependence estimators, the closed-form correction and the shared-difficulty mixture | analytic and exact-likelihood tests on synthetic data (`test_gate1.py`); a five-world simulation of the calibration effect (`RHAN_IM_dependence_calibration.json`). The correction is in the code path (`EvidenceCalibration`); the mixture is **not** implemented in `belief.py`. The toy worlds of A and B below have independent per-unit noise and do **not** test the assumption |
| Any accuracy, robustness or efficiency claim on real images | **none made** |
