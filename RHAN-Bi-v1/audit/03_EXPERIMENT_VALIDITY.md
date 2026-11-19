# 3 — Experimental validity assessment

## 3.1 What the evidence base actually is

There is **one** real result artifact in the repository: `results/episodes.jsonl` (12 episodes, all
`falsification`) and `results/demo_report.json`/`results/demo_report.md`, which summarise those 12.
`results/` contains nothing else: no `stage_a.json`, `stage_b.json`, `stage_c.json`, `stage_d.json`,
`stage_e.json`, `risk_coverage.json`, `run_manifest.json`, `calibration.json`, `gate_e.json` (verified by
`find RHAN-Bi-v1/results -type f`). The README/docs describe a 120-episode benchmark with 12, 30, 30,
18, 30 episodes per gate. The code can produce those counts, but the committed artifacts do not.

Consequences:

- Every quantitative claim in `README.md` and `docs/09_EXPERIMENTS_AND_GATES.md` about the gates is
  either a code-expected number or a hand-written summary, not a reproduced artifact.
- There is no run manifest, no code hash, no checkpoint checksum, no seed list, no split file. The
  run is not reproducible from the repository alone.
- The `episodes.jsonl` file is only 12 episodes and only one planner (`falsification`). It cannot
  support Gate E's 0/120 comparison.
- The audit's own Gate E numbers (§01 §5.1) come from an in-memory rerun of the published code with
  seed 7 and N = 120, which is itself an experiment, not a committed artifact. It is included to give
  an independent check, not as a replacement artifact.

## 3.2 Splits and leakage

- Split seeds are fixed (`config.py:99-101`): train 11, cal 23, dev 37, test 53. The simulator is
  deterministic given a seed (`scene.py:112-152`).
- `stage_a` trains on train, evaluates on dev (`stages.py:51-85`).
- `collect_calibration` fits the evidence model on `rows_all` = train + cal + dev (`data.py:47+`)
  and uses `cal` rows for calibration. The model that is actually used by Gates C/D/E is therefore
  fitted on **all splits including dev**. This is the reason the deployed dev accuracy (0.850) exceeds
  the reported Gate A accuracy (0.817): Gate A reports a train-fit model's dev score; the deployed
  model is a cal-fit model whose dev score is 0.850. This is a split/leakage defect, not a
  measurement error, and it explains the discrepancy in `README.md:43` and `docs/09:40`.
- Temperature is fit on cal posteriors (`stage_c:182-183`), then dev is evaluated. That part is clean.
- Gate B threshold is chosen after inspecting dev (`stages.py:147-150` sweep; 0.97 quantile selected
  from dev behaviour). This is a selection-on-test-set defect.
- Gate C threshold is selected on cal and evaluated on dev; that part is clean in principle, but the
  scale mismatch (§2.7) makes the reported number wrong.
- No test set is evaluated anywhere in the code. Gates A–E all use train/cal/dev. There is no
  holdout. The 80% objective for Gate A is checked on dev only.

## 3.3 Confounds

1. **Causal structure.** `make_scene` (`scene.py:112-152`) generates cause from `scene_index % 4`, so
   the prior is exactly uniform by construction and the test is trivially balanced. A real encoder
   will face an unknown, likely skewed prior; the logistic's intercepts and the belief prior must
   jointly handle that, which has never been tested.
2. **Renderer determinism.** The evidence model is fitted on renderer outputs with fixed noise
   (`config.py:55-68`). The "noise" is structured by construction. Generalisation to a real image
   distribution is not addressed.
3. **Action geometry.** The action-specific discrimination (light/view/ROI) is encoded in the
   renderer; the planner's forward model averages over it (§2.9). The benchmark therefore tests
   action *kinds*, not the geometry that the physics claims to exploit.
4. **Pair cue.** `_pair` requires the previous record to have the same `ledger.image` and a ROI
   overlap of at least 25 px (`evidence.py:96-127`). It is a deterministic geometric cue. Gate D's
   `no_pair` arm removes it; the contrast (0.000 vs 0.111 revision) is as much about losing a cue as
   about the ledger.
5. **Planner confounds.** `oracle` uses the true cause in its score, so it is not a real planner and
   must not be compared to the others on equal footing. `fixed` always picks the same sequence. Both
   are baselines, not competitors, but they appear in the same tables.
6. **Cost.** Cost is action-kind fixed (`config.py:74-80`: light 0.1, view 0.25, zoom 0.2). No
   latency, no sensor cost model. Gate E's "cost" comparison (3.525 vs 3.175) is not meaningful
   without a calibrated cost.
7. **Multiplicity.** Five gates, multiple metrics, multiple planners, multiple arms. There is no
   family-wise error control, no pre-registration, no correction for the threshold sweep on Gate B.
   The 0.80/0.85/0.90/0.40 thresholds are post-hoc round numbers.

## 3.4 Uncertainty and power

All counts below are Wilson 95% intervals for proportions, computed on the audit's rerun or on the
reported counts.

| Quantity | Count / N | Point | Wilson 95% |
|---|---:|---:|---|
| Gate A accuracy (deployed) | 102 / 120 | 0.850 | [0.776, 0.903] |
| Gate A defect/reflection (deployed) | 53 / 60 | 0.883 | [0.778, 0.942] |
| Gate A accuracy (reported, train-fit) | 98 / 120 | 0.817 | [0.738, 0.876] |
| Gate A defect/reflection (reported) | 46 / 60 | 0.767 | [0.646, 0.856] |
| Gate B pointing | 30 / 30 | 1.000 | [0.886, 1.000] |
| Gate B IoU ≥ 0.5 | 24 / 30 | 0.800 | [0.627, 0.905] |
| Gate C revision recovery | 2 / 18 | 0.111 | [0.031, 0.328] |
| Gate C reliability AUC | — | 0.642 | not computed |
| Gate E discordant decisions | 6 / 120 | 0.050 | [0.023, 0.105] |

Gate E power. The audit reran Gate E with N = 120, seed 7 and found 6 discordant decisions out of
120, i.e. a 5% disagreement rate. McNemar's test on 6 discordant pairs has p = 0.5 (exact binomial).
The minimum detectable *accuracy* difference at 80% power and 5% type-I error for a paired binary
outcome with a 5% discordance rate is:

| N | Minimum detectable difference |
|---:|---:|
| 120 | ≈ 5.7% |
| 400 | ≈ 3.1% |
| 1000 | ≈ 2.0% |

The observed difference is 0.000 (both planners error 0.1917). The test is therefore underpowered to
detect differences below ~6%. "Gate E shows the planners are equivalent" is not a supported
conclusion; it is a failure to reject the null. The correct statement is: at N = 120 the data do not
distinguish the two planners' error rates at the 5% level.

Gate A's 80% objective. The deployed model clears 0.80 with point 0.850, but the lower bound of the
95% interval is 0.776. The "80%" is not established with 95% confidence. The reported train-fit
number (0.817) also has lower bound 0.738. Neither establishes the objective at the conventional
level. This matters because the objective is the only quantitative gate in the system.

## 3.5 What is not tested

- No test-set evaluation (no held-out split).
- No real images. The DINOv2 path is optional, untested (`# pragma: no cover`), and not part of any
  reported number.
- No repeated seeds. Every number is from a single seed per split.
- No ablation of the dependence constants ρ. They are fixed.
- No ablation of `PAIR_SCALE`/`PAIR_CLIP`.
- No calibration of `n_eff` against repeated observations.
- No experiment varies the class prior away from uniform.
- No experiment varies the number of available actions or the cost model.
- No adversarial or distribution-shift evaluation beyond the synthetic corruption in `simulator/`.
- No human or external baseline.

## 3.6 Validity verdict

The experiments are **internally consistent code exercises on a deterministic synthetic world**. They
are useful for debugging the belief and planner machinery. They are not, in their present form,
scientific evidence for the claims made in the README and docs, because: (a) the artifacts are
missing, (b) the deployed model leaks dev into training, (c) Gate B selects on dev, (d) Gate E is
underpowered and misreported as equivalence, (e) the 80% objective is not established at 95%
confidence, and (f) no real image has ever been processed by the evidence path.
