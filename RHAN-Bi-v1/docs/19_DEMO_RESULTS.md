# 19 — Demo results

> **Claim level: feasibility.** This page reports one demo-scale run of the code in this folder. It is
> not the pre-registered confirmatory experiment (docs/09, N = 400), it is not a result on real
> images, and no number below should be quoted as a property of RHAN-Bi in general. The purpose of the
> run is to show that the five gates execute end to end and to record what the mechanism actually did,
> including where it did not separate from its baselines.

The numbers below were produced by `python -m rhan_bi.cli demo` on CPU and are copied from
`results/demo_report.json`; the human-readable form is `results/demo_report.md`, the per-episode
traces are `results/episodes.jsonl`. The run is deterministic given the seed.

## 1. What was run

| Item | Value |
|---|---|
| package / config | `1.0.0` / `16bb095c3497` |
| evidence model hash | `532e86050daaafcd` |
| forward-model hash | `7148d4a06c303487` |
| code snapshot | git `ca268c135b37` (working tree at run time) |
| seed | 7 |
| N train / calib / dev / test | 240 / 120 / 120 / 120 scenes |
| observation budget | 3 |
| bootstrap resamples | 2000 |
| environment | Python 3.10.12, NumPy 2.2.6, CPU only |

## 2. Verdicts

| Gate | Question | Endpoint | Verdict |
|---|---|---|---|
| A | single-observation competence | accuracy 0.817, defect/reflection 0.767 | **PASS** |
| B | spatial evidence | IoU@0.3 0.800, pointing 1.000, loc-cond. 1.000 | **PASS** |
| C | reliability and coverage | ECE 0.072, coverage 0.875 at risk 0.20 | **PASS** |
| D | belief revision | revision 0.111 vs pooled 0.000 | **PASS** |
| E | adaptive observation | falsification 0.192 = EIG 0.192 | **INCONCLUSIVE** |

A pass here means "the acceptance rule in docs/09 was met at demo scale". It does not mean the
underlying hypothesis is confirmed; Gate E, which is the hypothesis, did not separate.

## 3. Gate A — foundation

- accuracy **0.817**, balanced accuracy **0.817** over the four classes
  (`defect`, `reflection`, `shadow`, `none`).
- defect-vs-reflection balanced accuracy **0.767**, inside the required ambiguity band [0.55, 0.80].
  The simulator therefore still leaves the central distinction genuinely underdetermined from one
  base look, which is the precondition for the rest of the programme.
- Confusion matrix (rows truth, columns predicted):

  |  | defect | reflection | shadow | none |
  |---|---|---|---|---|
  | defect | 26 | 1 | 0 | 3 |
  | reflection | 2 | 20 | 0 | 8 |
  | shadow | 0 | 0 | 29 | 1 |
  | none | 0 | 7 | 0 | 23 |

The error structure is the intended one: almost all mistakes are inside the ambiguity set
(defect ↔ reflection, reflection ↔ none), not shadow confusions. `shadow` is easy because a soft
low-contrast edge is a different feature regime, not because the task is easy.

## 4. Gate B — spatial evidence

The localizer is a learned dark-ridge template with a threshold set at the 0.97 quantile of the null
distribution, fitted on the calibration split and evaluated on dev (30 defect scenes).

| Endpoint | Value | Rule |
|---|---|---|
| IoU@0.3 | **0.800** | ≥ 0.25 |
| mean IoU | 0.346 | reported, not gated |
| pointing (mask pixel within 4 px) | **1.000** | ≥ 0.85 |
| peak pointing (stricter) | 0.900 | reported, not gated |
| localization-conditioned accuracy | **1.000** | ≥ 0.90 |

The threshold sweep is part of the record: q0.95 gives IoU@0.3 0.000 with pointing 0.900, q0.97 gives
0.800 / 1.000, q0.98 gives 0.733 / 0.900, q0.99 gives 0.533 / 0.733. The 0.97 setting was chosen on
the calibration/dev data after the sweep, which is legitimate model selection but means Gate B at demo
scale is not a held-out test of that one number. The confirmatory protocol fixes the threshold on
calibration only and evaluates once on test.

The apparent jump between q0.95 and q0.97 is a rasterisation effect: at the stricter threshold the
template's peak lands off the ridge on most scenes and the mask becomes a small blob that misses the
0.3 overlap. Mean IoU 0.346 is the more honest summary — the localizer knows *where*, but its masks
are not tight.

## 5. Gate C — reliability and coverage

| Endpoint | Before | After | Rule |
|---|---|---|---|
| ECE (15 bins) | 0.104 | **0.072** | ≤ 0.08 |
| Brier | — | 0.246 | reported |
| NLL | — | 0.532 | reported |
| selective risk @ coverage 0.5 | — | **0.050** | ≤ 0.25 |
| dev coverage at risk ≤ 0.20 | — | **0.875** | ≥ 0.40 |
| dev selective risk at that coverage | — | 0.114 | ≤ 0.20 |

Temperature **0.715** (under-confident base model, so temperature < 1 sharpens). The conformal
threshold is 0.675, giving 0.775 coverage on calibration and 0.875 on dev.

This is the gate where the earlier RHAN-IM programme failed: it met a risk target by abstaining on
about 89 % of scenes. Here the certificate is met while keeping 87.5 % of scenes, so the risk number
is not purchased with coverage. The coverage floor (0.40) is doing real work; see docs/09 for why the
risk target is 0.20 rather than 0.10 at this base accuracy.

## 6. Gate D — belief revision and dependence

Four arms, same episodes, same fixed schedule base → light → zoom:

| Arm | accuracy | ECE | revision accuracy | mean confidence | n_eff (median) |
|---|---|---|---|---|---|
| `ledger` | 0.850 | 0.107 | **0.111** | 0.878 | 1.731 |
| `no_pair` | 0.850 | 0.108 | 0.111 | 0.866 | — |
| `no_dep` | 0.842 | 0.105 | 0.111 | 0.946 | — |
| `pooled` | 0.567 | 0.393 | **0.000** | 0.955 | — |

- The revision endpoint passes: evidence accumulated across looks and recorded with a light change
  moves the decision to the truth in 11.1 % of episodes where the pooled baseline moves in 0.0 %.
- Dependence discounting changes behaviour: `no_dep` is 0.07 more confident at equal accuracy
  (0.946 vs 0.878), which is the direction H3 predicts.
- **Honest negative:** ledger ECE (0.107) is *not* better than `no_dep` ECE (0.105) in this run. Gate D
  passes on revision accuracy, not on that secondary clause; the clause is reported as not met in the
  demo. Dependence discounting reduced confidence but did not improve calibration at this scale, and
  the effective sample size (median 1.73 of 3 looks) is small enough that the distinction is fragile.

## 7. Gate E — matched-budget policy comparison

Budget 3, paired episodes, error is the primary endpoint; abstention counts as error and coverage is
reported separately.

| Policy | Error | Accuracy | Mean cost | Mean confidence | ECE |
|---|---|---|---|---|---|
| falsification | 0.192 | 0.808 | 3.52 | 0.911 | 0.109 |
| eig | 0.192 | 0.808 | 3.17 | 0.947 | 0.141 |
| confidence | 0.183 | 0.817 | 3.02 | 0.927 | 0.129 |
| random | 0.183 | 0.817 | 4.09 | 0.930 | 0.133 |
| fixed_light_first | 0.183 | 0.817 | 4.00 | 0.950 | 0.136 |
| oracle | 0.175 | 0.825 | 3.25 | 0.932 | 0.134 |

- falsification − EIG: **0.000**, 95 % CI [−0.042, +0.042], McNemar p = 1.000, 6 discordant pairs.
- falsification − confidence: +0.008, CI [−0.017, +0.033].
- falsification − random: +0.008, CI [−0.017, +0.042].
- Oracle error 0.175, best deployable 0.183: **headroom gap 0.008**, i.e. the whole policy space is
  worth less than one scene at this N.

Verdict **INCONCLUSIVE**, as pre-specified for the demo when the oracle headroom is below 0.03. With
8 % of the policy space separating the oracle from the best baseline, no paired test at N = 120 can
distinguish policies; the confidence intervals above are wider than the entire effect being sought.
The demo establishes that the planner runs, that the arms are matched on episodes, and that the
statistics compute — not that falsification-directed selection is better.

Two observations worth carrying forward to the confirmatory run:

- Falsification is *more expensive* (3.52 mean observations vs 3.17 for EIG, 3.02 for confidence) and
  no more accurate here. Its cost-adjusted case is currently negative.
- EIG is the least calibrated arm (ECE 0.141) despite being most confident (0.947). Information gain
  and correctness are different objectives, which is the reason docs/06 separates them.

## 8. What these results do and do not support

**Supported at demo scale.**

- The full loop executes: render → observe → build evidence → append to ledger → update belief →
  plan → observe again → revise → commit or abstain.
- The ledger's provenance and dependence machinery is not inert: removing it changes confidence and
  revision behaviour (Gate D).
- Risk can be certified at useful coverage on this task (Gate C), unlike the earlier RHAN-IM result.
- The simulator retains the intended defect/reflection ambiguity (Gate A band), so the experiment is
  about evidence rather than about a trivially separable cue.

**Not supported.**

- That falsification-directed planning beats EIG, confidence-seeking or random selection. Gate E is
  inconclusive with no measurable headroom.
- That dependence-aware updating improves calibration; at this scale it did not.
- Any claim about real images, real defects, or Unreal-rendered scenes. Nothing in this run touched
  them.
- Any claim of state of the art. The baselines are internal arms, not the literature.

**Known weaknesses of this run.**

1. Gate B's threshold was selected after looking at dev; the confirmatory protocol must fix it on
   calibration only.
2. Gate E's N is far too small for the effect size the oracle implies; the demo cannot answer H1.
3. The forward model and the stopping rule share errors (risk R5), so the planner's cost estimates are
   optimistic in a way the demo does not measure.
4. Only one seed. Seed variance is unmeasured here; the confirmatory run uses five.
5. The evidence model is an 8-feature logistic regression over handcrafted features. It is
   interpretable and fast, and it is not a perception backbone. Swapping in DINOv2 is an
   implementation change behind the same interface, not a new experiment.

## 9. Reproduction

```bash
cd RHAN-Bi-v1
python -m rhan_bi.cli demo        # writes results/demo_report.{json,md}, results/episodes.jsonl
python -m rhan_bi.cli verify      # ledger hashes, posterior normalisation, split integrity
python -m pytest tests -q         # unit and end-to-end tests
```

The demo prints the gate verdicts as it goes; `results/demo_report.md` is the same content in prose.
Because the run is seeded and the simulator is deterministic, a rerun on the same NumPy version should
reproduce these numbers exactly; a different NumPy/BLAS version can perturb the last digit through the
logistic fit.
