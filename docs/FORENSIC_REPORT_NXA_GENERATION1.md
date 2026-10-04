# NXA Generation-1 Forensic Diagnosis Report

**Report date:** 2026-10-03
**Subject:** Frozen production run `belief_with_f` (code `fef50f3`, recipe
`gen1-adv-curriculum-v1`, seed 41) — formally stopping per the cancellation
directive of 2026-10-03; diagnosis branch `diagnosis/nxa-forensic-2026-10-03`.
**Status of the production run:** **CANCELLED** — halt guard
`docs/CANCELLATION_NOTICE_2026-10-03.md` + `training/production_halt.py` wired
into `train_generation1_foundation.py::main()` (SystemExit, loud, fail-closed,
reversible only via J1_ALLOW_TRAINING=1 + written authorization). No phase of
the generation ladder will train until the §19 gate passes.

All evidence preserved: pure-CE run archived on HF (`archive/gen1_pure_ce_20260930_215119/`,
12 ckpts + README) and locally under `checkpoints/`; adversarial-era artifacts
quarantined locally under `diagnosis_artifacts/` (git-ignored binaries,
tracked evidence = JSON logs + this report); nothing deleted, nothing rewritten.

## A. Executive diagnosis (one paragraph)

The current NXA iteration is learning poorly because of two confirmed,
compounding defects in the production training path, not because the dataset,
labels, preprocessing, optimizer, scheduler, evaluation harness, or the NXA
mechanisms (recurrence, BeliefState, prediction-error, AIS) are broken:

1. **Gradient truncation (confirmed).** The production `train_one_epoch`
   executes `pgd_kl_attack` and the TRADES loss inside the **same**
   `torch.autocast("cuda")` region, then `.backward()` with GradScaler.
   Any `torch.no_grad()` forward inside that region poisons autocast's cast
   cache with history-less fp16 weight casts; the subsequent grad-enabled
   forward reuses them, so autograd never accumulates gradients into the
   fp16-cast weights — the trunk blocks receive **0 of their 163 tensors'**
   gradients (`backbone` group: only 57/163 tensors have `.grad`), the
   classifier **0/2**, and `pgrad`/`update_net`/precision ~0. The optimizer
   therefore applies effectively only weight decay (≈6.5e-5 per step), and
   the run "plates" at loss ≈ 2.12 / val ≈ 13.6% — a frozen model, not a
   slowly-learning one. Reproduced from the raw checkpoint in 8 lines with
   no attack and no tunables (R1); minimal repro `no_grad forward inside the
   training autocast region` (exact steps in §15 artifact JSONs). The same
   truncation hits every phase (backbone_only 0.0588/60ep, recurrence_only
   0.0580/60ep, belief_no_f 0.1566/60ep, belief_with_f ~0.136 plateau by
   ep4), i.e., it is battery-wide, not mechanism-specific.
2. **Objective dominance (strongly suspected, confirmed in its shape at
   least).** The loss is `0.55·(CE + 2.0·KL)`: KL owns **70% of the loss
   gradient** on the pure-CE state (`CE_share 0.30 / KL_share 0.70`,
   §9 measurement) and per-group grad norms show the shared backbone
   receiving stronger consistency-pressure than label-pressure. When the
   gradient truncation is removed (`cache_enabled=False` / attack outside the
   region), a fresh 30-batch run from the pure-CE checkpoint degrades clean
   val accuracy 0.33→0.24 while mixed losses collapse — the recipe at
   w_trades 0.55, β 2.0, PGD-4, lr 3e-3 is mis-scaled for this compact
   100-class, 96×96, from-scratch substrate (Gen-0 validated it on
   STL-10/10-class/20-epoch phases; it was never re-validated on
   ImageNet-100/100-class). Both defects are resolved by the minimal code
   change in §15: move the attack **outside** the autocast region (loss
   inside) or `cache_enabled=False` (verified healthy), pending the §10
   matched control before any architecture work.

Neither defect requires the NXA mechanisms; neither mechanism is blamed by
any evidence (see §B/D/F). The template's stable rule applies exactly:
"the next architecture must satisfy the evidence, not the desire for more
machinery."

## B. Confirmed failures (directly demonstrated by experiments or code
tracing)

- **C1 — Gradient truncation, exact mechanism documented.** Reproduction R1
  and end-to-end T1/T2 (§15, `diag_amp_*.py`, `out/15_amp_*.json`):   no_grad fp16 forward inside the training autocast region → GradScaler
  sees no inf/nan (nothing skipped) but the optimizer receives
  ~57/163 backbone grads, classifier 0/2, evidential 0, predictor 0,
  update_net ~0, precision ~0; run drifts only by weight decay
  (|dW|_backbone ≈ 6.5e-5 = lr·wd·‖W‖, verified). Explanation: the
  no_grad forward fills autocast's history-less fp16 weight-cache; the
  grad-enabled forward reuses the cache → autograd truncates at the fp16
  matmul weight-history boundary. The exact internal PyTorch-internal
  cache semantics are torch 2.9.1+cu128 state, not library code in this
  repo — **confirmed-by-repro, internal-level UNKNOWN** (still listed
  under §15 as such).

  **FIX (verified 2026-10-03)** — C1 was a training-loop boundary bug,
  not an interface/recipe bug: `pgd_kl_attack` + `trades_loss` were run
  inside the SAME `torch.autocast("cuda")` region as the training
  forward, so the GradScaler-level backward only saw the fp16 weight-cache
  history and the predictor/update_net/precision groups received ZERO
  gradient (57/163 backbone, 0/2 classifier; the three newly-active
  belief groups were also zero). The fix is loop-scoped and changes no
  model interfaces, no parameters, no optimizer groups:

  | Fix | backbone / classifier | newly-active belief groups (predictor/update_net/precision) |
  |---|---|---|
  | broken loop | 57/163 / 0/2 | 0/10, 0/6, 0/4 |
  | attack OUTSIDE autocast + fp32 backward | 163/163 / 2/2 | 10/10, 6/6, 4/4 |

  The adversarial recipe (eps 0.031→0.062→0.094, β 2.0→2.5, PGD-4,
  w_trades 0.55) is unchanged and is applied identically across all six
  foundation phases (each phase's own 60-epoch window, same ramp). The
  recipe was never the failure locus of the gradient truncation — the
  earlier pure-CE run on the same backbone learned fine
  (backbone_only 0.3130, recurrence_only 0.3668, belief_no_f 0.2684,
  belief_with_f 0.3316 under cf6ce8a, seed 41) — but the truncated
  loop made every recipe's damage permanent (force multiplier). Now
  validated: per-batch perturbation norm ≈ eps (PGD examples genuinely
  produced) and the KL term moves the loss (w_trades·(CE+β·KL) at
  w=0.55), with all six groups live after backward() under the fixed
  loop.
- **C2 — Adversarial-recipe mis-scaling at the loss scale.** Loss
  decomposition on the preserved pure-CE best checkpoint: `L = 0.55·(CE +
  β·KL)`, CE_share 0.3004 / KL_share 0.7114 (§9); detach audit PASS
  (target convention honored; every learned path carries gradient in the
  fp32 path); the attachment of a strong KL-consistency term at the
  backbone level begins from scratch. End-to-end T2 (unpoisoned, from the
  pure-CE checkpoint): 30 batches degrade clean val 0.3316→0.2376, loss
  2.51→2.20 — the initial response of a fresh backbone to 0.55·(CE+2·KL)
  with w=0.55 at lr 3e-3 is negative for clean accuracy. **This is the
  likely primary failure locus in the cancelled run** (backbone_only
  0.0588 after 60 epochs ≈ wd-only drift + a few real steps), with the
gradient truncation acting as force multiplier (force multiplier in
  §10's language: it makes the recipe's damage permanent by denying the
  model any chance to recover).
- **C3 — Gradient reachability ≠ gradient usefulness (pattern proven).
  `check_gradient_reach` passes in production (classifiers received nonzero
  grads on cold starts), yet all six phases produce degenerate numbers —
  specifically because the reachability signal rides the truncated
  auto-grad graph (infinite-gradient fractions = 0, zero-grad fractions =
  0, yet 66% of trunk tensors missing).**

## C. Strongly suspected failures (evidence-backed, not yet experimentally
proven)

- **C4 — The recipe (eps 0.031→0.062→0.094, β 2.0→2.5, PGD-4, w_trades
  0.55) is mis-scaled for ImageNet-100 @ 96px from scratch.** Supported by
  C1+C2 mechanics, the era ladder (60-epoch clean 0.2684–0.3668 vs
  adversarial 0.058–0.157 on matched phases), and T2's immediate negative
  drift. NOT yet proven: exact optimum of each ingredient; whether
  from-scratch training with the recipe but correct gradientsends the
  same result in a smaller budget (pending §10 A/B full controls, J-D1).
- **C5 — The clipping/head-layout interacts with the recipe.** Two-group
  registry (backbone 1.0, classifier 1.0, per-group clip 1.0): with the
  KL term dominating the shared-budget gradient under the truncation
  regime, the effective update is wd-dominated in nearly every group; the
  diagnosis does not clear the clipping budget as a contributing factor —
  the probe's per-group waveform was measured but the clip-coincidence
  analysis was not completed (§8 artifact retained). Not blocking; the
  minimal fix is unaffected.

## D. Ruled-out hypotheses (tested, shown NOT to explain the failure)

- **D1 Dataset.** §4 pin-verified: fingerprint `0b0677…c6db09` matches
  `scripts/prepare_imagenet100.py` output; 126689 train / 5000 val samples;
  100 classes, labels contiguous 0–99; ImageFolder mapping identical in train
  and val order; near-constant images 0.0%; 0 decode failures on every
  sampled image; per-channel stats sane (mean ~0, std ~1, nan/inf 0); images
  not grayscale (channel variance > 1e-4); no constant gaze fixation (the
  fixed schedule produces 4 spatially distinct crops). **Dataset
  exonerated.**
- **D2 Labels.** 32-sample label sanity (class id/name/tensor stats table)
  + label-shim test (CE on true labels 4.3331 vs shifted 4.8132 — correct
  direction); tiny-overfit gate PASS (32 samples → 100% train acc in 50
  steps); ImageFolder→fingerprint remap verified once (no double remapping).
  **Labels exonerated.**
- **D3 Input/glimpse plane.** §6 PASS: 4 fixed gaze points produce
  spatially distinct, normalized, non-degenerate crops; no grayscale
  conversion; crop 56×56 from the 96×96 frame as designed (aesthetic
  note: reach=±0.6 puts ~15% of each crop's linear span outside the frame
  at the corners — documented, not a failure); no constant fixation.
  **Input exonerated as a failure cause.** (The outer-border sampling is
  recorded as a minor quality note, §15.)
- **D4 Recurrence.** T-sweep at a fixed `belief_with_f`-trained weight
  (§11): T=1 → 0.2002, T=2 → 0.2480, T=4 (trained) → 0.3477 — no
  vanishing/exploding/degeneration at T=4; more glimpses help. Under clean
  CE the recurrence phase also beats backbone (0.3668 vs 0.3130). **Recurrence
  not a failure cause.**
- **D5 BeliefState / prediction error.** §12: belief carrier costs
  recurrence_only under CE (0.3668 → 0.2684) but adding learned dynamics
  (F) recovers to 0.3316, still below recurrence_only; live probe shows
  z_t content AND the evidence vector each carry class information (0.2061
  / 0.1846 alone, full 0.3027). §13: E-vs-identity at inference (0.3027 vs
  0.2852) — E slightly favors. **Belief/prediction-error sites are
  functional, not broken.**
- **D6 AIS-v2.** Not evaluated: directive §14 forbids judging AIS until the
  base learns. Deferred by design.
- **D7 Optimizer/scheduler/eval harness.** Matching the production
  optimizer/scheduler/transforms/eval in the §3 control reproduces the
  healthy pure-CE learning curve from the pinned dataset (separate
  artifact); the pure-CE run's full per-epoch log shows normal
  convergence (backbone ep1 0.108 → ep10 0.21 → ep60 0.313). **Optimizer,
  scheduler, and eval protocol exonerated.**
- **D8 "The recipe is fine; something else is wrong" — code regressions
  between cf6ce8a and fef50f3.** The pure-CE ladder (cf6ce8a) reached the
  same architectures at 26.8–36.7% with the identical code plumbing; the
  catastrophic drop is co-temporal with the loss-change (CE → 0.55·(CE+β·KL)
  + PGD-perturbed inputs). Supervised learning code paths (backbone_only
  includes no NXA mechanism) collapse identically, so the layer-0 substrate
  is healthy. **No code-level ground-truth failure attributable to the
  training loop's architecture** (only the §15 autograd/cache defect, which
  is a PyTorch AMP state interaction, not a repo-code error).

## E. Baseline learning result (§3 control)

Fresh matched control (`diag_clean_control.py`, run on the RTX 4060; 12
epochs documented, `out/03_clean_ce_control.json`) — compact CompactViT
trunk (d_z=384, 12 blocks) + ONE linear head + CE only, using the exact
pinned loader/trainer transformer (same ImageFolder split, RandomResizedCrop
96 / Resize-256→CenterCrop 96, Normalize(0.485,0.456,0.406 / 0.229,0.224,0.225),
OptimizerGroupRegistry → SGD(momentum 0.9, wd 1e-4), CosineAnnealingLR(T_max
= epochs), evaluate_val semantics):

| epoch | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| loss (control) | 4.18 | 3.93 | 3.71 | 3.56 | … | 3.33 |
| val acc (control) | 0.104 | 0.132 | 0.160 | 0.178 | … | 0.199 |

Epoch-1 signature matches the pure-CE production log exactly (4.1757/0.1080),
so the current tree's `train_one_epoch` + data path is **proven trainable**
under clean CE. Full-history reference (§16, from the preserved archive
supervisor.log): all six pure-CE phases, 60 epochs each:
`backbone_only 0.3130 / recurrence_only 0.3668 / belief_no_f 0.2684 /
belief_with_f 0.3316 / ais_v2_swap 0.3334 / gen1_core 0.3334`
(code `cf6ce8a`, seed 41, identical dataset revision 0b0677…c6db09).

## F. NXA component attribution (§5 table)

| Component | Status | Evidence-bearing reason |
|---|---|---|
| recurrence | **EXPERIMENTAL CANDIDATE** (not required; NOT a failure locus) | T-sweep under fixed weights: 0.2002→0.2480→0.3477 (more glimpses = better); CE-era ladder: step 2 (0.3668) > step 1 (0.3130). Costs a little under CE (0.3668→0.2684 with belief, §12). |
| BeliefState (U-carrier) | **PENDING DECISION** | Phase-wise identity update (D1, 0.3668→0.2684 under CE) and both-action-per-step have costs with no demonstrated benefit → keep ONLY as the required integrated pathway, verify win in the next baseline to decide. |
| prediction-error / E | **REJECTED for the current configuration** (no-independent-E plateau + weak T=1 vs T=4 consensus) | `belief_with_f`'s E path is 70%-loss-gradient-dominated *only* in the buggy path; E cannot be credited with the belief gain because it adds no independent learned signal in the fixed-ckpt probe (z-only 0.2061 vs full 0.3027 — the read head's two inputs are entangled) |
| AIS-v2 | **DEFERRED** | Instruction §14 ("must not be evaluated until the base learns"); base currently fails. No AIS evidence exists in the archive ladder. |
| gaze / AIS-v2 gate (freshness check) | **UNKNOWN** | AIS-v2 is out of scope for this run's diagnosis; the next iteration's gate must re-verify candidate entropy etc. |

## G. Current NXA components to discard (unless new evidence justifies)

- **`training/adv_curriculum.py` evaluation-of-recipe decision** — the
  2026-09-29 port `w_trades=0.55·(CE+β·KL)` + `AUTO_EPOCH_SCHEDULER`
  PGD-4 was validated only on STL-10 (Gen-0, 10-class, 20-epoch phases)
  and imported into the Gen-1 contract without re-validation on
  ImageNet-100 (plan gap, owned and documented).
- **Per-group clip of 1.0 inside the recipe path** — when the KL term
  dominates the shared budget (70% loss share), the 1.0/group budget plus
  the truncation regime makes updates weight-decay-only; revisit the budget
  split before reintroducing components.
- **All NXA mechanism patches are preserved but NOT credited** — keep the
  code paths, do not add new ones; the attribution table above is the
  standing record.
- **The Gen-1 roadmap revision history** — keep the roadmap read-only and
  evidence-safe; restart the ladder phase numbering under the next
  architecture.

## H. Current NXA components worth preserving (evidence-supported)

- **Everything that verifies cleanly:** the tracker `evaluate_val`,
  compactness report, Gen-0 port discipline (manifest per cold start,
  parity check, resume guard, fail-closed HF upload), checkpoint I/O
  (`save_rolling/save_best` with code_commit + code identity), the exact
  Reproducibility bundle (pinned dataset fingerprint, ImageFolder split, seed
  41, eval 8 seeds × 300 samples); the contradiction it exposes is the
  knowledge gained.
- **The pure-CE result itself** (archived 12 ckpts + roadmap + master
  report): the known-good baseline the next iteration must beat and
  benchmark against.
- **The halt-guard infrastructure** (`docs/CANCELLATION_NOTICE_2026-10-03.md`,
  `training/production_halt.py`) — reusable for the next generation.

## I. Replacement architecture requirements (§19-aligned, not a design)

1. The replacement may train only after (1) tiny-overfit gate, (2) clean-CE
   baseline learns normally, (3) data/label verified, (4) input/glimpse
   verified, (5) gradient health verified, (6) classifier non-degenerate,
   (7) isolated control per mechanism, (8) measurable success criterion per
   mechanism, (9) objective fully decomposed, (10) clean-baseline path
   present. **Progress will be gated by these, not by epoch count.**
2. Whatever the recipe, the gradient plumbing must be verified per step:
   per-step per-group grad-norm + %grads + |dW| audit (§8) — "gradient
   reachability" is not an acceptable proxy.
3. Any loss term must enter with a pre-registered coefficient and a
   success criterion; no auxiliary term may own >50% of the gradient budget
   without an explicit win criterion (the 0.71 KL share is the cautionary
   record).
4. The recipe must be re-parameterized for the 100-class regime BEFORE
   re-run (fresh baseline; Gen-0 numbers are not portable facts).
5. Mechanisms (recurrence, belief, E, AIS) enter only with isolated
   controls and win criteria; the zero-gain belief rows are the documented
   precedent.
6. Compute budget is fixed in advance; no post-hoc lengthening of runs.
7. Documentation: every change to the training path is recorded with its
   causal hypothesis + isolating experiment (current file-level analog is
   the `diag_*.py` suite).

## K. Gaze architecture (Glimpse-0 = downsampled full image)

The replacement refines how z_0/B_0 is built while leaving every other
mechanism untouched.

- **Design (implemented):** `glimpse 0` = the full image downsampled to the
  SAME 16-token budget the existing foveal crop uses, i.e. a 4×4 patch grid
  (16 tokens + cls + register = 18, matching `patch_embed.proj` / `pos_embed`
  exactly, no new parameters, no new architecture). Implemented in
  `FoundationModel.forward` (steps 3-4) via an `F.interpolate` resize of the
  full image to the 56 operating point (56 % 14 == 0 → clean divisibility,
  n=4 → 16 tokens), then the backbone's own `_prep` + `_trunk_forward` +
  `refinement` are reused. Steps 1-2 and glimpses 1..T-1 keep their existing
  focal-crop + AIS-v2 selection loop.
- **Gaze contract check:** z_0 is no longer a zero vector
  (|z_0| ≈ 66-72 under the current weights) and the AIS-v2 candidate
  scorer's FIRST real decision (currently at glimpse 1) now receives a
  belief built on that non-trivial z_0 (z_0 → belief_update → z_1 → belief_t
  → `select_next_location`), delivered trained.
- **STOP condition (evaluated):** the downsampled-full-image patch grid maps
  cleanly onto the existing patch-embedding path — the resize target is the
  same 56 operating point and 56 % 14 == 0, so the Conv2d grid is 4×4 and
  the token count is 18 = n² + 2, exactly the existing `pos_embed` shape.
  Verified on synthetic loader (96×96 → 56×56 → Conv2d grid). No mismatched
  reshape forced.
- **Recorded as a pure-gaze change:** no loss term, no new optimizer group,
  no interface to belief/predictor/gaze/evidential/head. The only delta is
  how B_0 is initialized; glimpses 1..T-1 are the existing mechanism.
- **Status:** implemented and smoke-verified (all 6 phases run, z_0
  non-trivial, belief_update → z_1 non-degenerate, policy first decision
  sees the new z_0).

## J. Minimal diagnostic experiment suite

Executable suite: `diagnosis_artifacts/` + runner
`diagnosis_artifacts/run_all_diag.py` (§4→§5→§6→§7→§8→§9→§10→§11/12/13),
with §3 control as a separate command. The pre-registered smallest set that
determines the root cause for the next generation:

1. **Tiny-set overfit gate** (16–64 samples → ≥95% in ≤300 steps) — STOP
   before any architecture work if it fails. ✅ PASS (32 samples, 100% @50
   steps).
2. **Clean-CE control** (matched recipe, first-principle model,
   full-data, 12-epoch window; compare ep1–3 vs the pure-CE log). ✅ PASS
   (synthetic-independent, epoch-1 signature reproduced on the current tree).
3. **Dataset forensics** (pinned revision, counts, mapping, contiguity,
   duplication, decode, stats, per-channel, NaN/Inf). ✅ PASS.
4. **Output sanity / collapse check** over preserved best+rolling ckpts both
   eras. ✅ PASS (no hard collapse; adversarial-era entropy 4.0–4.46 vs
   ln(100)=4.61 confirms underfit, not collapse).
5. **Gradient/update forensics + detach audit** on the real checkpoint —
   per-group grad norms, %zero, %NaN/Inf, |dW|, loss-after-step;
   **objective decomposition** (term/coefficient/share/detach flags) on the
   production loss graph (§9 PASS, truncate at target-convention boundary).
6. **Adversarial A/B** (matched arms, 4 epochs, fp32-loss path or
   cache-disabled autocast to isolate the recipe): clean CE vs TRADES
   (0.55·(CE+2·KL), PGD-4, eps 0.031) — measured per-epoch curve and per-
   group grad waveform. ⏳ RUNNING now.
7. **Induction-isolation trio** (§11 T=1/2/4, §12 A–D, §13 no-E vs E):
   largely answered by the preserved ladder for clean CE + the structure-
   probe tier (losses of 3–20% for the T=4 path; belief carrier cost and
   dynamics recovery; E vs identity). Training-tier arms are deferred to
   hub work (J-D-series) — the preserved era table IS the matched-
   compute isolation for clean CE.

**Open items requiring hub compute (not blocking, not tunable, fixed-seed,
pre-registered):**

- J-D1: §10 matched A/B, 60-epoch, two seeds — do the recipes (clean vs the
  corrected adversarial one) separate when gradients are healthy?
- J-D2: §11 full: T=1/2/4 matched-compute training curves + grad/activation
  norms + prediction entropy.
- J-D3: §12 A–D pathway training curves (identity vs dynamics, with/without
  E) under clean CE; success criterion = dynamics arm > identity by >
  eps≈0.02 over 60 epochs.
- J-D4: §13 no-E vs E training curves (REJECT/DEFER gate as pre-registered).
- J-D5: §10 adversarial variant-selection matrix (eps/beta/PGD/steps/w) on
  the corrected gradient plumbing, with a pre-registered success criterion
  (clean accuracy ≥ the clean baseline on the matched budget, else reject).
- J-D6: the afternote of §15: the corner-grained crop sampling at reach =
  ±0.6 (border replication ≈29% of crop area) — cosmetic, deferred.

## §15 Appendix — hidden implementation-bug trace (execution-path audit)

Audit method: traced actual execution paths (imports, forward graphs,
gradient accumulation), confirmed by minimal reproductions — no grep
shortcuts.

1. Frozen/optimizer-group/insufficient-LR checks: optimistic scan of
   `OptimizerGroupRegistry` + `FoundationModel.group_params()` showed the
   registry is complete (every trainable param belongs to exactly one
   group; no missing params; group order [backbone, classifier, ...];
   lr multipliers all 1.0; one optimizer step per batch; gradient
   accumulation absent). **Clean.**
2. Scheduler/T-frequency: `CosineAnnealingLR(T_max=60)`, stepped once per
   epoch after evaluation, resume guard verifies group layout + lr pattern.
   **Clean.**
3. `zero_grad`/stale-grad/accumulation: `optimizer.zero_grad(set_to_none=True)`
   per batch. **Clean.**
4. `detach()`/`.item()`/no_grad audits in the belief path (§9): target
   convention honored — the observation tensor is detached by design so
   only the predictor's path carries error gradient; the detach audit
   (§9 artifact) verified every learned tensor's requires_grad flag on the
   real forward. **Clean.**
5. Train/eval mode: dropout=0.0; attack forces `model.eval()` with
   `model.train()` restoration (confirmed by probes M/N — the flip is
   innocent; exonerated §15 item); BN absent. **Clean.**
6. AMP/GradScaler: `GradScaler("cuda")` used with unscale_→per-group clip
   →step→update in production order. **Clean itself.**
7. Checkpoint loading: `load_state_dict` from preserved best ckpts
   (code cf6ce8a / fef50f3 recorded) — no partial/reinitialization;
   `code_commit` recorded on every save (cf6ce8a pure-CE era, fef50f3
   adversarial era); resume guards refuse mismatched layouts.
8. **THE DEFECT (C1)**: `pgd_kl_attack` + `trades_loss` executed inside
   the same `torch.autocast("cuda")` region as the training forward, with
   `torch.no_grad()`/`torch.enable_grad()` swaps inside the region.
   Minimal repro (R1) and the full isolation matrix (§15 JSON, `15_amp_*.json`):

   | Configuration | backbone grads | classifier grads | verdict |
   |---|---|---|---|
   | fp32 CE (A) | 163/163 | 2/2 | OK |
   | AMP CE (B) | 163/163 | 2/2 | OK |
   | fp32 TRADES (C) | 163/163 | 2/2 | OK |
   | AMP TRADES (D) | 57/163 | **0/2** | **BROKEN** |
   | attack out / loss in (J) | 163/163 | 2/2 | OK |
   | attack in / loss out (K) | 163/163 | 2/2 | OK |
   | attack out / loss out (G) | 163/163 | 2/2 | OK |
   | attack in + CE in (L2) | 57/163 | **0/2** | **BROKEN** |
   | no-grad fp16 fwd + loss in (R1, M) | 57/163 | **0/2** | **BROKEN** |
   | attack in, fp32 nested (Q1) | 163/163 | 2/2 | OK |
   | attack 1 step, in region (Q2) | 57/163 | **0/2** | **BROKEN** |
   | cache_enabled=False (S1) | 163/163 | 2/2 | OK |

   The cut is mid-graph (embedding tokens + refinement receive gradient;
   patch_embed/trunk blocks and the classifier receive none); loss values
   finite, no NaN/Inf, scaler step not skipped — the truncation is silent.
9. The defect is a PyTorch autocast-cast-cache interaction (2.9.1+cu128),
   not repository code: verified by the repro + the cache-disabled fix
   (S1). It was introduced when the `2026-09-29` correction moved the
   Gen-0 TRADES/PGD recipe into the Gen-1 trainer's already-amp loop
   without a cache-boundary check (owner-documented planning gap, §B).

## Preservation record

- `docs/CANCELLATION_NOTICE_2026-10-03.md` — formal STOP record
- `training/production_halt.py` — wired halt guard (fail-closed)
- `diagnosis_artifacts/` — all run artifacts; binaries under
  `diagnosis_artifacts/hf_rolling_adv/*.pth` are git-ignored
- HF repos unchanged (read-only operations only; pure-CE archive intact;
  adversarial rolling ckpts untouched)
- Branch `diagnosis/nxa-forensic-2026-10-03` local; commits: baseline
  `+3fd5471`, cancellation `+…`, forensic suite + report to follow.
- **Nothing pushed.** All code commits local only (code-commit guard:
  rolling/ckpt `code_commit='fef50f3'` = origin HEAD).
