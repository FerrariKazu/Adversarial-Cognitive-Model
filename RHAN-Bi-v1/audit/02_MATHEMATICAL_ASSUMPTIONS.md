# 2 — Mathematical assumptions and derivations

Notation: causes `c ∈ C = {defect, reflection, shadow, none}`, observations `i = 1..n`, features
`z_i`, per-cause singleton evidence `ℓ_i(c)`, pair evidence `π_i(c)`, record weight `w_i`, prior
`p_0(c)`. All numbers below use the demo configuration (`config.py:41-72`).

## 2.1 The evidence model is a classifier, not a likelihood

`EvidenceModel.fit` trains a multinomial logistic regression on standardised features
(`likelihood.py:52-84`):

```
P(c | z) = softmax(W z + b)_c
```

and exposes

```
ℓ(c) = log P(c | z) - log P(none | z)          (likelihood.py:116-119)
```

The belief update then computes

```
logit(c) = log p_0(c) + Σ_i w_i [ℓ_i(c) + π_i(c)]          (belief.py:96-121)
```

For `ℓ(c)` to be a likelihood ratio `log p(z|c)/p(z|none)`, the class priors implicit in the
logistic fit must cancel. They do only if the calibration set is class-balanced and the belief prior
equals the calibration class prior. The demo is balanced by construction (`index % 4` in
`simulator/scene.py:122`), so the cancellation is numerically harmless **in this demo**. It is not a
property of the method. If the calibration split is imbalanced, the logistic intercepts contain the
calibration prior and the update counts a prior twice.

A second assumption is that `ℓ_i` is calibrated as a likelihood. It is fitted on calibration rows and
Gate C fits a temperature on *single-look* calibration posteriors (`stages.py:169-185`), but that
temperature is **never applied to the sequential belief** used by Gates D and E. The ECE 0.072 is a
single-look, temperature-scaled number; the Gate D ECE 0.107 is an untempered sequential number. They
are not the same object and should not be compared.

## 2.2 The pair term is not a likelihood ratio

`pair_llr_matrix` computes, for each cause in `{defect, reflection, shadow}`, a diagonal Gaussian
log-density over `(persistence, bright_shift)`, subtracts the mean log-density across those causes,
multiplies by `PAIR_SCALE = 0.5`, and clips to `±PAIR_CLIP = 4`
(`likelihood.py:124-137`). There are three consequences:

1. The scores sum to approximately zero across the three fitted causes. Pair evidence cannot express
   "there is no anomaly"; it only re-ranks the three anomaly causes relative to each other.
2. The scale and clip are arbitrary hyperparameters. They are not fitted by maximising a likelihood
   and have no calibration interpretation.
3. The term is added to `ℓ_i` and then multiplied by the same `w_i` as the singleton term, although
   the pair measurement is derived from the *previous* record (`evidence.py:96-127`). The dependence
   between the pair term and the earlier record is therefore not represented by the action-level ρ.

The pair term is a useful engineering feature. It is not a probabilistic evidence term, and calling
the sum a posterior requires the independence assumption that the rest of the design denies.

## 2.3 The dependence weight is prefix-dependent

The implemented weight is (`dependence.py:44-51`):

```
w_i = clip( (0.5 + 0.5 q_i) / max(1, 1 + Σ_{j<i} ρ_ij), 0, 1 )
```

where `q_i` is the quality score and `ρ_ij` is a **fixed structural constant** from
`EvidenceConfig` (`config.py:41-48`): 1.00 same action, 0.55 same view/different light, 0.60 same
light/different view, 0.45 same view+light/different ROI, 0.10 common scene.

The denominator depends on the **prefix** `j < i`, not on the whole set. Therefore the total evidence

```
E = Σ_i w_i (ℓ_i + π_i)
```

is not invariant to the order of the observations. Swapping two actions changes which ρ values are in
each prefix; it also changes whether a light observation has a pair parent.

**Measured counterexample.** One dev scene, the same three actions `{light=2, view=-1, zoom}` in all
six orders, deployed model, deterministic renderer:

| Order | Final MAP confidence | Record weights | `n_eff` |
|---|---:|---|---:|
| light, view, zoom | 0.912 | 0.801, 0.487, 0.434, 0.269 | 1.818 |
| light, zoom, view | 0.969 | 0.801, 0.487, 0.426, 0.251 | 1.818 |
| view, light, zoom | 0.906 | 0.801, 0.455, 0.447, 0.350 | 2.051 |
| view, zoom, light | 0.827 | 0.801, 0.455, 0.282, 0.330 | 1.818 |
| zoom, light, view | 0.987 | 0.801, 0.456, 0.400, 0.321 | 2.051 |
| zoom, view, light | 0.956 | 0.801, 0.456, 0.263, 0.330 | 1.818 |

The spread is 0.160 in final confidence for the same information. `docs/05:79` states "the posterior
is invariant to the order of records (addition commutes)". That statement is false for this
implementation. The first weight is 0.801 rather than 1 because the quality factor is included;
`docs/04:80` omits the quality factor from the formula.

## 2.4 Why `n_eff` is not a posterior correction

The formula (`ledger.py:89-96`, `dependence.py:54-60`) is

```
n_eff = n / (1 + (n - 1) ρ̄),     ρ̄ = mean over all recorded relations
```

This is the standard design effect for a sum of `n` exchangeable, equal-variance variables with a
common pairwise correlation ρ:

```
Var(Σ X_i) = n σ² + 2 σ² Σ_{i<j} ρ = n σ² (1 + (n-1) ρ)
Var(mean) = σ²/n_eff,   n_eff = n / (1 + (n-1) ρ)
```

The derivation requires: equal variances, a **common** pairwise correlation, linear (Pearson)
dependence, no higher-order structure, and a known/estimated ρ. None of these holds literally here:

- ρ is a **design constant**, not an estimate from observations. There is no sampling variability and
  no confidence interval. `docs/03:122` claims the constants are "checked against the empirical
  correlation of the feature residuals on the calibration split" and that the code prints it via
  `verify --dependence-report`. The `vif` function exists (`dependence.py:67-82`) but is **never
  called**, and the CLI has no such flag (`cli.py:97-115`). This is a documentation-code mismatch.
- With heterogeneous ρ, replacing every pair by the mean is only a first-order approximation to the
  variance of the sum; it is exact only under equicorrelation.
- ρ ∈ [0, 1] by construction. Negative dependence cannot be represented; `max(0.0, avg)` in the
  formula (`ledger.py:92`) clamps the average rather than modelling it.
- The formula is descriptive. The posterior does not use `n_eff`; it uses `w_i` from §2.3. There is no
  derivation that `Σ_i w_i` equals `n_eff`, and numerically they differ (the 4-record example above:
  weight sum 1.991, `n_eff` 1.818; order-dependent). `docs/04:97-99` admits "the number is
  descriptive, not a posterior parameter". If that is the intent, it should not be reported as an
  effective sample size correction, and the README's n_eff framing overstates it.

**The reported 1.29.** `verify` builds a two-record ledger: base (light=1) then light=2 at the same
view. The single relation has ρ = 0.55, so

```
n_eff = 2 / (1 + 1·0.55) = 2 / 1.55 = 1.290322…
```

It is exactly the formula, with no data estimation. It is not evidence that two looks are worth 1.29
independent looks; that interpretation would require estimating ρ from repeated observations and
validating the variance model.

## 2.5 Duplicate observations are unbounded

For `m` identical actions, the k-th duplicate has `k-1` prior relations of ρ = 1.0, so

```
w_k = 1 / (1 + (k-1)) = 1/k,     Σ_{k=1..m} w_k = H_m = 1 + 1/2 + ... + 1/m
```

The harmonic sum diverges logarithmically. With a per-record LLR of 2.0 for `defect` and zero for the
others, the defect logit after `m` duplicates is `log(0.25) + 2 H_m`. Measured with the real
`weight_for` rule:

| duplicates m | Σw | defect logit | posterior defect |
|---:|---:|---:|---:|
| 1 | 1.000 | 0.614 | 0.711 |
| 2 | 1.500 | 1.614 | 0.870 |
| 5 | 2.283 | 3.180 | 0.970 |
| 10 | 2.929 | 4.472 | 0.992 |
| 20 | 3.598 | 5.809 | 0.998 |
| 50 | 4.499 | 7.612 | 0.9996 |

Meanwhile `n_eff → 1`. The posterior claims near-certainty while the reported effective sample size
claims one observation. This directly contradicts `docs/05:81`: "if all records are duplicates,
`n_eff → 1` and the posterior moves by at most one record's worth of evidence." It moves by `H_m`
records' worth. If the dependence correction were intended to bound duplicate influence, the weight
would need to be normalised by the full set (or a joint precision matrix used), not by a growing
prefix sum.

## 2.6 Small-sample and correlation regimes

| Regime | Behaviour of the implemented rule | Assessment |
|---|---|---|
| Independent observations | ρ = 0.10 (common scene) still applies to every pair; weights fall as `1/(1+0.1k)`; total weight `≈ 10 ln n` | Not independent; a floor of 0.10 is a modelling choice, not an estimate |
| Highly correlated | ρ = 1.0 for exact duplicates; weight `1/k`; total `H_n` | Under-corrects the sum; `n_eff` says 1 |
| Partially correlated | ρ ∈ {0.45,0.55,0.60}; weights depend on order and on how many unrelated records precede | No unique correction; not exchangeable |
| Negative correlation | impossible (ρ ≥ 0) | Cannot represent alternating/contrasting evidence |
| Heterogeneous ρ | mean is used in `n_eff`; prefix sum used in weights | Two different aggregations; neither is the exact variance |
| n = 2–4 (the actual episodes) | `n_eff` formula is exact only under equicorrelation; weights are dominated by the first few terms | Small-n bias is not characterised; no simulation study |

There is no bias correction, no standard error, and no calibration of `n_eff` against repeated
observations. At n ≤ 4, the notion of an asymptotic effective sample size is anyway descriptive.

## 2.7 Gate C's threshold is on the wrong probability scale

`stage_c` constructs calibration posteriors from raw LLRs (`stages.py:180`), fits temperature on
them (`182-183`), and then selects the conformal threshold from **raw** calibration confidences
(`190-200`). It evaluates dev on **tempered** dev posteriors (`184, 201`). Temperature scaling is a
monotone transformation but not the identity, so a numeric threshold selected on the raw scale does
not select the same set on the tempered scale.

Measured with the deployed model:

| Quantity | Value |
|---|---:|
| temperature | 0.7152 |
| raw calibration threshold | 0.6751 |
| calibration coverage at threshold | 0.775 |
| dev coverage at threshold, **tempered** (reported) | **0.875** |
| dev coverage at threshold, **raw** (same scale as selection) | **0.7667** |
| dev selective risk, tempered | 0.1143 |
| dev selective risk, raw | 0.1087 |

The reported 0.875 is not the coverage of the calibrated rule. The gate still clears its 0.40 floor
with 0.767, so the verdict does not flip, but the number in the report is wrong. The correct
procedure is to select the threshold on tempered calibration posteriors (or transform the threshold
through the temperature), then evaluate once on dev/test.

Two further Gate C points:

- `pass` checks ECE, selective risk at coverage 0.5, and `dev_cov >= 0.40`; it does **not** check
  `dev_selective_risk_at_threshold <= 0.20` (`stages.py:239`). The gate can pass with an arbitrarily
  bad risk at the certified coverage, as long as coverage is high.
- `ece` uses 15 **equal-frequency** bins (`metrics.py:52-64`, `np.array_split` of sorted confidence),
  not equal-width bins. Equal-frequency ECE is noisier at n = 120 and is not comparable to
  equal-width ECE. On this dev set, equal-frequency gives 0.0723 and equal-width gives 0.0585. ECE is
  also not a proper scoring rule; Brier/NLL should carry the calibration conclusion.

## 2.8 The revision metric

`stage_d` counts `initial_wrong` from the first-record MAP and increments `revision_n[arm]` when the
final prediction is correct **and** the first prediction was wrong (`stages.py:290-305`). It reports

```
revision_accuracy = revision_n / initial_wrong
```

This is a conditional recovery rate. It does not measure whether a revision was justified, whether it
was reversible, whether contradictory evidence was preserved, or whether confidence moved
appropriately. In the demo, `initial_wrong = 18`; ledger, `no_dep` and `no_pair` each fix 2 (0.111),
and `pooled` fixes 0 but changes prediction on 15 of the 18. The 0.111 vs 0.000 contrast is therefore
mostly a statement that pooling averages away the disambiguating pair cue, not that the ledger
produces better revisions.

With 2/18, the Wilson 95% interval is roughly [0.03, 0.33]; a paired McNemar on the two discordant
episodes gives p = 0.5. The result is compatible with no effect. Any claim of a revision advantage
needs a larger paired sample and a metric that separates correct revisions from changes.

## 2.9 Planner objectives

Let `p` be the current posterior, `h*` the current leader, and `p'` the posterior after a hypothetical
action. The implemented scores (`planner.py:114-171`) are:

```
falsification:  E[1 - p'(h*)] + 0.5 (risk_before - E[1 - max p']) - penalty
eig:            H(p) - E[H(p')] - penalty
confidence:     E[max p'] - penalty
random:         uniform choice
fixed:          light=2, then view=-1, then light=0
oracle:         E[1(argmax p' = true cause)] under samples from the true cause
```

Because `risk_before = 1 - p(h*)` and `E[1 - max p'] = 1 - E[max p']`,

```
risk_gain = E[max p'] - p(h*)
```

so falsification is a weighted combination of "reduce the leader's probability" and "increase the
maximum probability". It is not a formal expected likelihood ratio against the leader, and it is not
the same functional as EIG. Under a saturated posterior both displacement and entropy gain approach
zero, so the objectives become numerically similar even when they are not algebraically equivalent.
That is what the demo shows: different actions, same error.

The larger problem is the forward model. `ForwardModel.fit` keys rows by `(cause, kind)` only
(`planner.py:37-54`), and `_simulate` samples features from that key (`173-201`). It does not condition
on **which** light, **which** view, or the ROI. Consequences:

- all light actions at a given cause have the same predicted feature distribution;
- all view actions have the same predicted distribution;
- the only action-specific term is the cost and the `rho`-based redundancy penalty;
- the planner cannot prefer `light=2` over `light=0` for a reflection hypothesis, which is precisely
  the physical discrimination the simulator was built to expose.

This is a design limitation, not a coding bug, but it means "active perception" is currently active
only over action *kinds*, not over the geometry that makes the causes identifiable.

## 2.10 What would make the mathematics defensible

1. Replace prefix weights with a set-symmetric dependence model (e.g. a joint Gaussian precision over
   records, or a normalised weight `w_i ∝ 1/(1+Σ_{j≠i}ρ_ij)` with a derivation), and prove order
   invariance in a test with permuted records.
2. Either estimate ρ from repeated calibration observations and report its uncertainty, or state
   explicitly that it is a prior/heuristic and remove `n_eff` from inferential claims.
3. Calibrate the sequential posterior (temperature on multi-look episodes), not only the single-look
   model, and report proper scores.
4. Replace the pair score with a fitted likelihood ratio (or clearly label it a feature) and model its
   dependence on the parent record.
5. Derive or drop the conformal coverage claim; if kept, use a threshold on the same scale and check
   risk at the certified coverage.
