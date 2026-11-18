# 10 — Evaluation and statistics

## 1. Unit of analysis

The **scene** is the unit. Every policy sees every test scene with the same seed, so comparisons are
paired at the scene level. Observations within a scene are not independent and are never treated as
such: a per-observation standard error would be wrong by roughly the dependence factor the programme
is studying.

## 2. Primary endpoint

`decision_error(scene, policy) ∈ {0, 1}`: 1 if the committed cause differs from the ground truth, or
if the system abstains. Abstention counts as an error in the primary endpoint because the endpoint is
"did the system deliver the right answer"; abstention is reported separately as a secondary outcome,
never as a way to lower primary error.

The primary contrast is the paired difference

```
Δ = mean_scenes [ error(scene, falsification) − error(scene, EIG) ]
```

with a two-sided 95 % percentile bootstrap CI over scenes (10,000 resamples, resampling scenes with
replacement, both policies resampled together to preserve pairing). McNemar's exact test on the
discordant pairs is the secondary significance check; the CI is primary because the effect size
matters more than the threshold.

## 3. Secondary endpoints

| Endpoint | Definition | Reported with |
|---|---|---|
| error vs confidence | paired difference as above | CI |
| error vs random | paired difference | CI |
| cost-weighted error | `error + 0.05 · total_cost` per scene | CI |
| abstention rate | fraction of scenes with no commit | Wilson CI |
| coverage at risk | fraction committed at selective risk ≤ 0.20 | Wilson CI |
| ECE | 15 equal-mass bins, |accuracy − confidence| | bootstrap CI |
| Brier / NLL | standard | bootstrap CI |
| revision accuracy | among initially-wrong episodes, fraction corrected | Wilson CI |
| n_eff | effective sample size per episode | median [IQR] |
| predicted vs realised | Pearson r between planner's predicted displacement and realised | CI |

## 4. Multiplicity

Three confirmatory policy comparisons (EIG, confidence, random) plus the primary. Holm–Bonferroni at
family-wise α = 0.05 across the three secondary comparisons; the primary is tested at α = 0.05 and is
not included in the Holm family (it is the gate). Everything else is exploratory and labelled.

## 5. Power

For the confirmatory run, N = 400 scenes. Under McNemar's normal approximation with discordance
`π_d = p01 + p10` and effect `δ`, the variance of the paired difference is approximately
`(π_d − δ²)/N`. With `π_d = 0.30` and `N = 400`, an 8-point difference (δ = 0.08) gives
`z ≈ 0.08 / sqrt((0.30 − 0.0064)/400) ≈ 3.0`, i.e. ≈ 84 % power at α = 0.05; a 6-point difference
gives ≈ 60 % power. The pre-registration therefore states 8 points as the smallest effect the design
can reliably detect, and the Gate E decision rule requires a point estimate ≤ −0.05 with an upper CI
bound below 0 — a result that is both detectable and meaningful.

The power statement is an approximation, not a simulation result. Before the confirmatory run, a
simulation-based power check (`results/power_sim.json`, 2,000 synthetic paired tables at the observed
discordance) must confirm it; if the simulated power at δ = 0.06 is below 0.70, N is increased *before*
the freeze, not after.

## 6. Calibration metrics

- **ECE** with 15 equal-mass bins. Bins are formed on confidence, so empty bins do not occur. ECE is
  sensitive to binning; the bin count is fixed in every report and a reliability diagram is plotted.
- **Brier** is the mean squared error of the probability assigned to the true cause (multi-class
  version: sum over classes).
- **NLL** uses the temperature-scaled posterior, clipped at 1e-12.
- **Selective risk / coverage curve** is computed by sweeping the commit threshold over the sorted
  confidences, so the curve is exact for the dev set rather than interpolated.
- **Conformal risk control**: the threshold is chosen on calibration with the Learn-Then-Test upper
  bound; the dev coverage is reported, never used to re-tune the threshold.

## 7. Confidence intervals

- Proportions: Wilson interval.
- Paired differences: percentile bootstrap, 10,000 resamples, scenes resampled jointly.
- ECE and other smooth functionals: bootstrap over scenes.
- Medians: bootstrap percentile.
- No CI is reported for a quantity computed on fewer than 30 independent units; it is reported as a
  point estimate with `n` and labelled unstable.

## 8. Missing data and exclusions

The simulator produces no missing data. The only exclusions permitted are:

- an episode aborted by a simulator error (recorded, with the error, not silently dropped);
- a scene whose ground truth fails an integrity check (never observed in v1; would be recorded).

Exclusions are reported as a count with reasons. A policy comparison with a different exclusion set per
policy is invalid and the runner enforces the same scene set for every policy.

## 9. Reproducibility of a statistical claim

Every reported number carries:

- the config hash and git commit;
- the seed list hash;
- the N and the exclusion count;
- the interval method.

`results/stage_e.json` is the machine-readable record; `results/stage_e_report.md` is generated from
it by `rhan_bi.evaluation.report`, so the prose cannot drift from the numbers.

## 10. The claim ladder

A result may be reported at one of three levels, and the level must be stated:

1. **demonstrated** — measured on the test split, pre-registered, CI excludes no effect;
2. **feasibility** — measured at demo scale or on dev, no confirmatory claim;
3. **exploratory** — observed after looking, a hypothesis for the next run.

The demo is level 2. Anything computed while writing this dossier, before the freeze, is level 3.
Level 1 requires the frozen run.
