# 4 — Gate-by-gate classification

Legend for "class":
- **S** = software verification (code runs, types check, tests pass)
- **E** = exploratory synthetic (useful for debugging, no inferential claim)
- **D** = development-set result (tuned on the same data it is reported on)
- **C** = calibration result (properly held-out calibration, but no test set)
- **F** = confirmatory (pre-registered, held-out test, sufficient power) — none of A–E qualify
- **U** = unresolved (cannot be classified from the repository)

## Gate A — Evidence quality

- Class: **D** (development set), with a leakage defect.
- Reported: 0.817 accuracy, 0.767 defect/reflection balanced, ECE 0.08 (`README.md:43`,
  `docs/09:40`).
- Deployed (cal-fit model, dev): 0.850 accuracy, 0.883 defect/reflection, ECE 0.1075, Brier 0.2652,
  NLL 0.6471 (audit rerun).
- Defect: `collect_calibration` fits the deployed model on `rows_all` including dev
  (`data.py:47+`); `stage_a` reports a different, train-fit model. The two numbers are not comparable
  and neither is a held-out result.
- Fix before any claim: retrain on train+cal only, evaluate on dev once, then on test; report the
  deployed model's number, not the train-fit number; state whether 0.80 is met at 95% confidence.
- Minimum to promote to C: single pre-registered dev evaluation, no threshold tuning on dev.

## Gate B — Anomaly localisation

- Class: **D** (development set), with selection-on-test-set.
- Reported: IoU ≥ 0.5 on 24/30, pointing 30/30, threshold 0.97 quantile of train `none` maps
  (`README.md:44`, `docs/09:41`).
- Audit: pointing 30/30 (Wilson [0.886, 1.000]); IoU 24/30 (Wilson [0.627, 0.905]); threshold
  sweep hardcoded at q = {0.90, 0.95, 0.97, 0.99} (`stages.py:147-150`) and q = 0.97 chosen after
  dev inspection; measured pointing at q = 0.97 is 1.0 while the sweep comment says 0.9.
- Defects: threshold selected on dev; no `results/stage_b.json`; no test set; the 0.97 quantile is
  a round post-hoc choice.
- Fix: select threshold on cal `none` maps only; evaluate on dev once; report pointing and IoU with
  intervals; then run on test.
- Minimum to promote to C: threshold chosen on cal, evaluated on dev/test without re-tuning.

## Gate C — Calibrated belief and abstention

- Class: **C** (calibration result), with a scale defect and a missing risk check.
- Reported: ECE 0.072, selective risk 0.10 at 50% coverage, coverage at threshold 0.875
  (`README.md:45`, `docs/09:42`).
- Audit: temperature 0.715; ECE equal-frequency 0.0723 (equal-width 0.0585); selective risk at 50%
  coverage 0.1087 (raw) / 0.1143 (tempered); coverage at threshold 0.7667 raw vs 0.875 tempered;
  reliability AUC 0.642.
- Defects: threshold selected on raw cal, applied to tempered dev (§2.7); `pass` does not check
  `dev_selective_risk_at_threshold` (`stages.py:239`); ECE uses equal-frequency bins; Brier/NLL not
  reported.
- Fix: select threshold on tempered cal posteriors; evaluate risk at the certified coverage; report
  Brier and NLL; use equal-width ECE or both.
- Minimum to promote to F: pre-registered coverage target and risk target, evaluated on test.

## Gate D — Ledger vs pooling

- Class: **E** (exploratory synthetic). Useful, but not a controlled comparison.
- Reported: ledger 0.850 vs pooled 0.567 accuracy; 2/18 revisions
  (`README.md:46`, `docs/09:43`).
- Audit: ledger acc 0.850 ECE 0.1075 Brier 0.2652 NLL 0.6471; no_dep 0.8417; no_pair 0.850;
  pooled 0.5667 ECE 0.393. initial_wrong 18; ledger/no_dep/no_pair fix 2; pooled fixes 0 but changes
  15. McNemar on 2 discordant pairs, p = 0.5.
- Defects: the contrast is confounded by the pair cue (§3.3.4); the revision metric is a conditional
  recovery rate (§2.8); n = 18 is too small; no test set; the deployed model leaks dev.
- Fix: larger paired sample (≥ 200 episodes) or a pre-registered ablation on held-out scenes; a
  revision metric that separates justified from unjustified changes; report intervals.
- Minimum to promote to C: held-out scenes, paired design, pre-registered metric.

## Gate E — Planner comparison

- Class: **E** (exploratory synthetic) for the machinery; the equivalence claim is **unsupported**.
- Reported: "same error 0.192, 0/120 identical action sequences, same cost" (`README.md:47`,
  `docs/09:44`).
- Audit rerun (N = 120, seed 7): falsification and EIG both error 0.1917; 0/120 identical full
  action sequences, 1/120 identical kind sequences; 114/120 same final decision; first actions
  falsification zoom 74 / light 29 / view 17 vs EIG light 105 / zoom 11 / view 4; costs 3.525 vs 3.175;
  discordant (False,True): 3, (True,False): 3; McNemar p = 0.5.
- Defects: N = 120 gives ~5.7% minimum detectable difference at 80% power (§3.4); the policies
  chose different actions, so "same actions" is false; `episodes.jsonl` contains only 12
  falsification episodes; no test set; the forward model averages over action geometry (§2.9);
  `oracle` and `fixed` are baselines, not competitors.
- Fix: pre-registered non-inferiority margin and N ≥ 400 (≈ 3.1% MDD) or 1000 (≈ 2.0%); condition
  the forward model on action identity; record full action sequences, not only kinds; report cost with
  a calibrated cost model.
- Minimum to promote to F: pre-registered margin, sufficient N, held-out scenes, calibrated cost.

## Summary table

| Gate | Class | Reported headline | Audit correction |
|---|---|---|---|
| A | D (leakage) | 0.817 / 0.767 | deployed 0.850 / 0.883; lower bound 0.776 |
| B | D (selection on dev) | IoU 0.80, pointing 1.0 | intervals [0.627,0.905] / [0.886,1.000]; threshold not held out |
| C | C (scale defect) | ECE 0.072, cov 0.875 | tempered cov 0.767 at the selected threshold; risk not checked |
| D | E (exploratory) | ledger 0.850 vs pooled 0.567 | confounded by pair cue; n = 18; p = 0.5 |
| E | E (underpowered) | "same error, same cost" | different actions; 0.0 pp difference; MDD ≈ 5.7% at N = 120 |

No gate reaches **F**. Gates A and B are development-set results with defects. Gate C is a
calibration result with a scale defect. Gates D and E are exploratory. The 80% objective for Gate A
is not established at 95% confidence on either number.
