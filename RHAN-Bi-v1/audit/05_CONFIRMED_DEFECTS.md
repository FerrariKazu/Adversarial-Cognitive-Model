# 5 — Confirmed defects

Severity: **S1** invalidates a reported number or a stated invariant; **S2** materially weakens a
claim; **S3** documentation or robustness issue. Every entry includes the location, the evidence, the
impact and a minimal fix. None of these fixes has been applied; this audit is read-only.

## F1 — Posterior is order-dependent (S1)

- Location: `rhan_bi/belief.py:73-121` (`from_ledger`), `rhan_bi/dependence.py:44-51`
  (`weight_for`), `docs/05_BELIEF_STATE_AND_REVISION.md:79`.
- Evidence: six permutations of the same three actions on one dev scene produced final MAP
  confidences 0.827–0.987 with different record weights (`audit/02` §2.3). The weight denominator is a
  prefix sum (`Σ_{j<i} ρ_ij`), so permuting records changes the weights.
- Impact: contradicts the documented invariant "the posterior is invariant to the order of records
  (addition commutes)". Any claim of order-invariant accumulation is false.
- Fix: replace prefix weights with a set-symmetric rule or a joint precision model; add a test that
  permutes records and asserts equal posterior.

## F2 — `n_eff` is reported but never used in the posterior (S1)

- Location: `rhan_bi/belief.py:109` (computed, not applied), `rhan_bi/ledger.py:89-96`,
  `rhan_bi/dependence.py:54-60`, `docs/04_EVIDENCE_QUALITY.md:92-99`.
- Evidence: `from_ledger` calls `ledger.n_eff()` only to populate `BeliefState.n_eff`; the posterior
  uses `Σ w_i`. The two differ (weight sum 1.991 vs n_eff 1.818 in the four-record example).
- Impact: `n_eff` is presented as an effective sample size but has no effect on inference. It should
  be labelled descriptive or removed from inferential claims.
- Fix: either derive a correction that actually scales the posterior, or relabel and stop reporting
  it as a correction.

## F3 — Duplicate records accumulate without bound (S1)

- Location: `rhan_bi/dependence.py:44-51`, `docs/05_BELIEF_STATE_AND_REVISION.md:81`.
- Evidence: for m identical actions, w_k = 1/k and Σw = H_m. At m = 50, posterior defect = 0.9996
  while n_eff = 1 (`audit/02` §2.5).
- Impact: contradicts "if all records are duplicates, n_eff → 1 and the posterior moves by at most
  one record's worth of evidence." The posterior moves by H_m records' worth.
- Fix: normalise duplicate influence by the full set, or use a joint model; add a duplicate
  saturation test.

## F4 — Ledger is not tamper-evident (S1)

- Location: `rhan_bi/ledger.py:98-109` (`verify`), `rhan_bi/types.py:115-128` (`record_id`),
  `rhan_bi/util.py:19-42` (`canon`).
- Evidence: mutating record 0's LLR to ±12, resetting its `record_id`, and re-chaining every
  downstream `dependence.relations` and `pair.with_record_id` made `verify()` return
  `{'records': 3, 'ok': True}` with posterior defect 1.0. Tail truncation of the last record also
  passes. (`audit/01` §1.4.)
- Impact: the "tamper-evident" claim is false for any attacker who re-chains. The hash chain is a
  content-addressed log, not an authenticated one.
- Fix: either remove the tamper-evidence claim, or add an authentication primitive (HMAC/signature)
  and a head commitment; at minimum, make `verify` check record count against an independent
  manifest.

## F5 — `add()` does not verify image bytes against `image_sha256` (S1)

- Location: `rhan_bi/ledger.py:26-37` (`add`), `rhan_bi/ledger.py:40-48` (`_remember_image`).
- Evidence: `add()` stores `image` under `record.image_sha256` without checking
  `sha256_image(image) == record.image_sha256`. The audit stored mismatched bytes and `verify()`
  returned ok.
- Impact: provenance can be silently replaced; a record can claim to depict one image while storing
  another.
- Fix: check `sha256_image(image) == record.image_sha256` in `add()` and raise on mismatch; store
  image only under its own hash.

## F6 — Gate A measures a train-fitted model, not the deployed model (S1)

- Location: `rhan_bi/training/stages.py:41-49` (`fit_all`), `stages.py:51-85` (`stage_a`),
  `training/data.py:47+` (`collect_calibration`).
- Evidence: `fit_all` fits on cal `rows_all` and is used by C/D/E; `stage_a` fits a separate logistic
  on train and evaluates on dev. Deployed dev accuracy 0.850 vs reported 0.817; deployed
  defect/reflection 0.883 vs reported 0.767.
- Impact: Gate A's headline number is not the model the rest of the system uses. The deployed model
  is also outside the required [0.55, 0.80] ambiguity band (0.883).
- Fix: report the deployed model's dev score; retrain on train+cal only; evaluate on dev then test;
  state whether 0.80 is met at 95% confidence.

## F7 — Gate C threshold scale mismatch (S1)

- Location: `rhan_bi/training/stages.py:169-203`, `rhan_bi/metrics.py:52-64`.
- Evidence: calibration posteriors built from raw LLRs (`180`), temperature fit on them (`182-183`),
  threshold selected from raw calibration confidences (`190-200`), dev evaluated on tempered
  posteriors (`184, 201`). Deployed: threshold 0.6751 raw; cal coverage 0.775; dev coverage raw 0.7667
  vs tempered 0.875; dev risk raw 0.1087 vs tempered 0.1143.
- Impact: the reported coverage 0.875 is not the coverage of the calibrated rule. The gate still
  clears its 0.40 floor, so the verdict does not flip, but the number is wrong.
- Fix: select the threshold on tempered calibration posteriors (or transform the threshold through
  the temperature); evaluate once on dev/test.

## F8 — Gate C pass rule omits dev risk at threshold (S2)

- Location: `rhan_bi/training/stages.py:239`.
- Evidence: `pass` checks ECE, selective risk at coverage 0.5, and `dev_cov >= 0.40`; it does not
  check `dev_selective_risk_at_threshold <= 0.20`.
- Impact: the gate can pass with arbitrarily bad risk at the certified coverage, as long as coverage
  is high.
- Fix: add `dev_selective_risk_at_threshold <= 0.20` to the pass condition; report both.

## F9 — Gate B threshold selected on dev (S1)

- Location: `rhan_bi/training/stages.py:88-160`, in particular `threshold_sweep` at `147-150` and
  the 0.97 quantile at `102`.
- Evidence: the sweep is hardcoded at q = {0.90, 0.95, 0.97, 0.99} and q = 0.97 is chosen after
  inspecting dev; the stored comment says pointing 0.9 at q = 0.97 while the measured pointing is 1.0.
  No `results/stage_b.json` exists.
- Impact: selection on the test set; the reported IoU/pointing are optimistic.
- Fix: select the threshold on cal `none` maps only; evaluate on dev once; then test.

## F10 — Revision metric is recovery rate, not revision quality (S2)

- Location: `rhan_bi/training/stages.py:272-331`, in particular `290-305` and `299-312`.
- Evidence: `revision_n[arm]` increments only when the final prediction is correct and the
  first-record prediction was wrong; reported as `revision_n / initial_wrong`. initial_wrong = 18;
  ledger, `no_dep` and `no_pair` each fix 2; `pooled` fixes 0 but changes 15. Wilson interval for
  2/18 is [0.031, 0.328]; McNemar p = 0.5.
- Impact: the 0.111 vs 0.000 contrast mainly shows that pooling averages away the disambiguating
  pair cue, not that the ledger revises better.
- Fix: use a larger paired sample (≥ 200 episodes); use a metric that separates justified from
  unjustified changes; report intervals.

## F11 — Gate E artifact supports only one policy (S2)

- Location: `rhan_bi/cli.py:70-80`, `results/episodes.jsonl`.
- Evidence: `episodes.jsonl` contains 12 episodes, all `falsification`; no `eig`, `confidence`,
  `random`, `fixed` or `oracle` episodes are saved. `cli.py:70` calls `run_gate_e` with one policy.
- Impact: the saved artifact cannot support the planner comparison; re-analysis is impossible.
- Fix: save episodes for every policy with full action sequences, seeds and model hashes.

## F12 — DINOv2 features are not consumed by the evidence model (S1)

- Location: `rhan_bi/perception/backbone.py:36-96` (in particular `encode` at `85-96`),
  `rhan_bi/evidence.py:41-94`.
- Evidence: `encode` returns `features = feats_h` (hand-crafted) at line 92 and puts CLS/patch tokens
  only in `meta` at line 94. `EvidenceBuilder.build` never reads `meta`. The path is optional and
  marked `# pragma: no cover`; no test exercises it.
- Impact: the system is not a visual system in any measured sense. The checkpoint exists but is
  unused.
- Fix: route DINOv2 CLS/patch tokens into `EvidenceBuilder`; add an integration test on real or
  rendered images; report a Gate A number on real features.

## F13 — Planner forward model ignores action identity beyond kind (S2)

- Location: `rhan_bi/planner.py:37-69` (`ForwardModel.fit`/`sample`), `planner.py:173-201`
  (`_simulate`).
- Evidence: rows are keyed by `(cause, kind)` only; `_simulate` samples from that key. All light
  actions at a cause share one distribution; all view actions share one distribution. The only
  action-specific terms are cost and the ρ-based redundancy penalty.
- Impact: the planner cannot prefer `light=2` over `light=0` for a reflection hypothesis, which is
  the discrimination the simulator was built to expose. "Active perception" is active only over action
  kinds.
- Fix: condition the forward model on action identity (light value, view angle, ROI); or state that
  the planner is kind-level only.

## F14 — Duplicate images accepted under new metadata (S2)

- Location: `rhan_bi/ledger.py:26-37` (`add`).
- Evidence: `add()` does not reject a record whose image bytes already appear under a different
  step/action. The audit appended a duplicate and `verify()` returned ok.
- Impact: combined with F3, this is the mechanism by which duplicate evidence accumulates. It is
  expected behaviour for a log, but it contradicts the "duplicate-aware" framing unless the
  dependence rule actually saturates (it does not).
- Fix: either reject exact duplicate images, or make the dependence rule saturate; document the
  choice.

## F15 — No schema validation on records (S2)

- Location: `rhan_bi/ledger.py:98-109`, `rhan_bi/util.py:19-42`, `rhan_bi/types.py:98-137`.
- Evidence: a record with NaN LLRs hashes and verifies (belief later raises `FloatingPointError`);
  a nested `provenance.cause` passes verify; float canonicalisation has a documented collision
  (`util.py:19-39`, six-decimal rounding); JSON keys coerce `1` and `'1'` to the same key.
- Impact: malformed or adversarial records survive integrity checks; canonicalisation is not
  injective for floats.
- Fix: validate finite LLRs and weights in `add()`; validate provenance schema; document the
  float-rounding collision as a known limitation or remove rounding.

## F16 — Documentation cites nonexistent artifacts and flags (S3)

- Location: `docs/03_EVIDENCE_LEDGER.md:105-125` (`verify --dependence-report`),
  `docs/09_EXPERIMENTS_AND_GATES.md` artifact rows, `rhan_bi/cli.py:97-115`.
- Evidence: `results/` contains only `demo_report.json`, `demo_report.md`, `episodes.jsonl`. `vif`
  and `effective_n` are defined but unused (`dependence.py:67-82`). The CLI has no
  `--dependence-report` flag.
- Impact: the documented verification workflow cannot be run; readers cannot reproduce the claimed
  checks.
- Fix: either implement the flag and generate the artifacts, or correct the docs to match the
  repository.

## F17 — Quality factor omitted from the documented weight formula (S3)

- Location: `rhan_bi/dependence.py:44-51`, `docs/04_EVIDENCE_QUALITY.md:80`.
- Evidence: the code computes `(0.5 + 0.5 * quality) / max(1, 1 + sum_rho)`; the docs formula
  omits the `(0.5 + 0.5 * quality)` factor. The first record's weight is 0.801, not 1.0.
- Impact: the documentation does not describe the implementation. Minor, but it obscures why a
  single high-quality record is down-weighted.
- Fix: update the docs formula, or remove the quality factor from the weight.

## F18 — Pair evidence is an ad-hoc score, not a likelihood ratio (S3, upgraded to S2 if claimed as evidence)

- Location: `rhan_bi/likelihood.py:33-34` (`PAIR_SCALE`, `PAIR_CLIP`), `likelihood.py:124-137`
  (`pair_llr_matrix`).
- Evidence: the score is a centred diagonal-Gaussian log-density, scaled by 0.5 and clipped to ±4,
  over three anomaly causes only. It sums to approximately zero across those causes and cannot
  express "no anomaly". Scale and clip are not fitted.
- Impact: it is a useful feature but not a probabilistic evidence term. Calling the sum a posterior
  requires the independence assumption the design otherwise denies.
- Fix: relabel as a feature, or fit it as a likelihood ratio and model its dependence on the parent
  record.

## Count by severity

S1: F1, F2, F3, F4, F5, F6, F7, F9, F12 (9). S2: F8, F10, F11, F13, F14, F15, F18 (7).
S3: F16, F17 (2). Total 18, matching `audit/01` §1.3.
