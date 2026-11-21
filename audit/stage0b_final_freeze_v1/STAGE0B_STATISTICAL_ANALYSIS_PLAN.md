# Stage 0B — Statistical Analysis Plan

**Protocol:** `rhan-stage0b-dinov2-b14-temperature/1` version `1.0.0`  
**Status:** **FROZEN CANDIDATE — NOT EXECUTED**  
**Machine-readable config:** `work/statistical_config.json`  
**Execution authorization:** **NO**

---

## 1. Estimand and unit

- **Unit:** one ImageNetV2 image.
- **Fitting set:** 5,000 calibration images, five per class.
- **Confirmatory set:** 5,000 evaluation images, five per class.
- **Seeds:** 17, 42, 99; all reported; none selected or dropped.
- **Primary estimand:** equal-weight mean over the three seeds of the 15-bin ECE reduction:

\[
\bar{\Delta} = \frac{1}{3}\sum_{s\in\{17,42,99\}}
\left[\operatorname{ECE}_{15}(\text{raw}_s) -
\operatorname{ECE}_{15}(\text{calibrated}_s)\right].
\]

A positive value means lower ECE after temperature scaling.

The three seeds share images. They are not independent model replications, so images are never pooled across seeds as if they were independent.

---

## 2. Temperature fitting

### 2.1 Parameterization and positivity

\[
T_s = \exp(\theta_s), \qquad \theta_s \in [\log 0.05,\log 20].
\]

This guarantees \(T_s>0\). It also makes the bounded optimizer operate on an unconstrained-in-sign scale. No negative or zero temperature is representable.

### 2.2 Objective

\[
\hat{T}_s = \arg\min_{T_s} \frac{1}{n_{\text{cal}}}
\sum_{i=1}^{n_{\text{cal}}}
-\log \operatorname{softmax}(z_s(x_i)/T_s)_{y_i}.
\]

Only calibration images and labels enter this objective. Evaluation labels are never loaded during fitting.

### 2.3 Numerical procedure

- Convert original float32 logits to float64 for fitting.
- Use `scipy.optimize.minimize_scalar(method='bounded')` on \(\log T\).
- Bounds: \(\log 0.05=-2.995732273553991\), \(\log 20=2.995732273553991\).
- `xatol=1e-10`, `maxiter=1000`.
- Initialize at \(T=1\).
- Record NLL at \(T=1\), NLL at optimum, `nfev`, `success`, and `message`.
- A boundary hit is reported explicitly. It is not clipped, widened, or reparameterized after the fact.

### 2.4 Per-seed independence

Each seed is fitted separately. There is no pooled temperature, no shared temperature, no hierarchical shrinkage, and no seed selection. The aggregate is an equal-weight summary of three pre-specified fits.

### 2.5 Convergence and failure

All three fits must report `success=true` and be interior to the bounds for confirmatory acceptance. A failed or boundary fit yields **“calibration improvement not established under the frozen rule”** and a technical/model-condition report. No fallback optimizer, alternative bounds, or per-class temperature is permitted.

---

## 3. ECE definition

The primary metric is 15-bin expected calibration error:

- bins are equal-width on \([0,1]\);
- membership is \(lo < \text{confidence} \le hi\);
- empty bins contribute zero;
- ECE is weighted by bin count;
- confidence is the maximum softmax probability;
- correctness is exact Top-1 correctness.

No bin-count sensitivity analysis is confirmatory. Adaptive ECE with 15 equal-frequency bins is secondary and its bin counts are reported.

A positive ECE reduction is an improvement. A negative reduction is an increase.

---

## 4. Paired comparison

For each seed:

1. Raw probabilities: \(\operatorname{softmax}(z_s)\).
2. Calibrated probabilities: \(\operatorname{softmax}(z_s/T_s)\).
3. Compute raw and calibrated metrics on exactly the same evaluation rows.
4. Compute \(\Delta_s=\text{ECE}_{15}^{\text{raw}}-\text{ECE}_{15}^{\text{cal}}\).
5. Compute NLL and Brier differences with the same sign convention: positive means improvement.
6. Compare Top-1 and Top-5 index arrays exactly.

No row is dropped, reordered, or imputed. The evaluation row order is the frozen partition order.

---

## 5. Uncertainty

### 5.1 Class-stratified paired bootstrap

- 10,000 resamples.
- RNG seed `20261121`.
- Within each of the 1,000 classes, draw five images with replacement.
- Concatenate the 1,000 class resamples into one 5,000-row index vector.
- Reuse the same index vector for raw and calibrated predictions and for all three seeds.
- In each resample, compute the three-seed mean ECE reduction.
- Report the 2.5th and 97.5th percentiles as the 95% interval.

Why stratified: every class contributes exactly five images in each split. An unstratified image bootstrap can produce resamples with missing classes and unstable weighting. Stratification preserves the design.

### 5.2 What the interval means

The interval describes uncertainty of the three-seed mean ECE reduction over the fixed evaluation sample. It is not a posterior probability, not a guarantee for future datasets, and not a substitute for effect size.

### 5.3 No seed-level significance claims

Per-seed ECE deltas are reported. No per-seed p-values or “significant/not significant” claims are made. The primary decision uses the three-seed mean and the frozen per-seed guard.

---

## 6. Decision rule

Accept H1 only if all nine conditions hold:

| # | Condition | Threshold |
|---:|---|---|
| 1 | Mean ECE reduction | \(>0\) |
| 2 | 95% class-stratified paired-bootstrap interval | excludes zero |
| 3 | Worst per-seed ECE increase | \(\le 0.005\) |
| 4 | Top-1 prediction mismatches | exactly zero |
| 5 | Top-5 index-set mismatches | exactly zero |
| 6 | Worst per-seed NLL increase | \(\le 0.005\) |
| 7 | Worst per-seed Brier increase | \(\le 0.005\) |
| 8 | Temperature fits converged | all three |
| 9 | Temperature fits at bound | none |

The `0.005` margins are frozen engineering tolerances. They are not derived from a power analysis and are not claimed to be universal scientific standards. They may not be relaxed after outcomes are seen.

If the interval includes zero, report **“calibration improvement not established.”** Do not report “no effect.” If the estimate is positive but small, report the number, interval, and practical ambiguity.

---

## 7. Multiple metrics

The analysis has one primary endpoint and two confirmatory guard families:

- primary: mean ECE reduction;
- guard 1: accuracy invariance;
- guard 2: proper-score non-degradation (NLL and Brier).

All other metrics are secondary/descriptive:

- adaptive ECE;
- reliability bins;
- mean confidence;
- macro Top-1;
- per-class ECE and confidence–accuracy gap;
- selective risk, AURC, and coverage summaries.

No bin-count search, no metric shopping, no subset search, and no alternative temperature family is permitted under this protocol.

---

## 8. Accuracy and proper-score guards

### 8.1 Accuracy invariance

Single-scalar positive temperature scaling divides every logit by the same positive number. It cannot change the mathematical ordering of classes. Therefore:

- Top-1 predictions must be identical;
- Top-5 index sets must be identical;
- any non-tie mismatch is an implementation failure.

Ties use descending logit, then ascending class index. A calibration-induced tie that changes the frozen tie-break is treated as an implementation failure, not as an accuracy result.

### 8.2 Proper scores

NLL and Brier score are proper scores. Temperature scaling can improve or worsen them. The guard allows at most `0.005` absolute degradation per seed. If ECE improves but a proper score degrades beyond the margin, the result is **“ECE improvement with degraded proper score,”** not a uniformly better calibrator.

---

## 9. Selective prediction

- Compute risk–coverage curves from coverage 0.05 to 1.00.
- Report AURC.
- Report selective risk at coverage 0.50, 0.80, 0.90, 0.95.
- Report abstention fraction, correct-but-abstained fraction, and confidence tie count.

Expected invariant: raw and calibrated curves coincide except at numerical ties, because scalar temperature scaling preserves confidence ordering. A material difference is investigated as an implementation error. It is not reported as a temperature-scaling benefit.

---

## 10. Class-conditional analysis

The primary evaluation split has five images per class. Therefore:

- per-class accuracy, ECE, and confidence–accuracy gaps are **descriptive only**;
- no per-class confidence intervals are used for confirmatory claims;
- no 1,000-class significance testing is performed;
- no multiplicity-adjusted per-class “wins” are claimed;
- class-level summaries are reported with `n=5` visible.

A class-level bootstrap may be used only as a descriptive summary, not as evidence for individual classes.

The secondary 50,000-image validation benchmark, if used, still carries the “not independent” label and the same multiplicity caution.

---

## 11. Missing and invalid data

The analysis aborts, without imputation, on:

- a missing or unreadable image;
- a content-hash mismatch;
- a wrong tensor or logit shape;
- a non-finite logit or probability;
- a label outside `[0,999]`;
- a calibration/evaluation overlap;
- duplicate image identity;
- a checkpoint, backbone, config, or code hash mismatch;
- an unexpected file in the frozen split;
- a failed optimizer convergence.

No clipping, label smoothing, temperature floor, or synthetic replacement is permitted.

---

## 12. Reporting

Per seed:

- \(T_s\);
- NLL at \(T=1\);
- NLL at optimum;
- iterations, tolerance, convergence, boundary;
- raw and calibrated ECE;
- \(\Delta_s\);
- raw and calibrated NLL and Brier;
- Top-1 and Top-5;
- calibration/evaluation logit hashes.

Aggregate:

- mean ECE reduction;
- 95% class-stratified paired-bootstrap interval;
- per-seed direction;
- each of the nine decision checks;
- acceptance or non-acceptance wording.

The report must separate:

1. confirmed facts and frozen configuration;
2. raw Stage 0A baseline;
3. fitted-temperature results;
4. confirmatory decision;
5. secondary benchmark, if any;
6. descriptive/exploratory findings;
7. limitations.

---

## 13. Interpretation boundaries

- Global calibration is the confirmatory object.
- Class-conditional results are descriptive at five images per class.
- Selective prediction is a rank-preserving descriptive check.
- Accuracy preservation is an invariant and guard.
- ImageNetV2 is a distribution-shifted test set, not a random sample from ImageNet-1k training.
- No claim of adaptive perception, principled abstention, or general robustness follows from this experiment.
- No claim of exact Stage 0A image-level reproduction follows from the separately pinned Stage 0B pipeline.

**No Stage 0B outcome has been inspected.**
