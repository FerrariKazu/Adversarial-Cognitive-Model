# 04 — Evidence quality, reliability, dependence, conflict

Four quantities are routinely collapsed into one "confidence" number. RHAN-Bi keeps them separate
because they answer different questions and fail in different ways.

| Quantity | Question | Failure if collapsed |
|---|---|---|
| **Observation quality** | was this measurement taken under good conditions? | a sharp, high-contrast image of an irrelevant region looks "good" |
| **Predictive uncertainty** | how undecided is the posterior? | a confidently wrong model reports low uncertainty |
| **Evidence dependence** | how much new information does this record add given the others? | repeated looks inflate confidence without adding information |
| **Hypothesis conflict** | does this record contradict the current leader? | averaging hides the contradiction that should trigger revision |

## 1. Observation quality

Quality is a property of the image and the observation geometry, computed without labels. Six
components, each in [0, 1] after the transforms below:

| Component | Definition | Mapped to [0,1] by |
|---|---|---|
| `blur` | variance of the Laplacian of the ROI, normalised by ROI size | `min(1, lapvar / 0.002)` |
| `contrast` | (p95 − p5)/255 of the ROI | `min(1, contrast / 0.25)` |
| `exposure` | 1 − |mean/255 − 0.45| / 0.45 | clipped to [0,1] |
| `glare` | fraction of pixels ≥ 250 | `1 − glare` |
| `occlusion` | fraction of ROI that is flat and mid-grey | `1 − occlusion` |
| `snr` | anomaly amplitude / robust noise σ (MAD) | `min(1, snr / 8)` |

The composite quality score is the product of the six components, raised to no power and not
re-weighted:

```
q = blur * contrast * exposure * (1 - glare) * (1 - occlusion) * min(1, snr / 8)      ∈ [0, 1]
```

A product is used deliberately: one catastrophic defect (an occluder over the relevant region) should
dominate the score, and independence between components is not assumed for calibration — the product
is a heuristic index, validated as a *predictor of error* in Stage C, not as a probability.

Quality flags (booleans, stored with the record): `blurred` (blur < 0.25), `glared` (glare > 0.02),
`occluded` (occlusion > 0.15), `low_contrast` (contrast < 0.10), `saturated` (exposure < 0.4).

Quality enters the belief in two places: the record weight is multiplied by `(0.5 + 0.5 q)` so a
terrible observation still counts at half weight rather than being silently dropped, and quality below
0.35 raises a `LOW_QUALITY` debt item on every hypothesis the record touches.

## 2. Predictive uncertainty

The posterior over the four causes gives three scalars: entropy `H(p)` (nats), risk `1 − max_h p_h`,
and margin `p_(1) − p_(2)`. All three are read-outs of the same object and are reported separately
because they disagree in useful ways: a bimodal posterior has high entropy and high margin; a
four-way tie has high entropy and zero margin.

Uncertainty is *not* calibrated by construction. The calibration split measures ECE, Brier and NLL
after temperature scaling fitted on the calibration split only (`docs/07` Stage C). The temperature is
fitted on the *cause* posterior, not on the feature likelihoods, and is stored with the model so a
replay is reproducible.

## 3. Observation reliability

Reliability is the expected correctness of the evidence a record contributes. It is estimated by a
small model fitted on the calibration split:

```
P(correct record | quality features) = sigmoid(a + Σ_j b_j * q_j)
```

with `q_j` the six quality components and the target "the record's LLR sign agrees with the true
cause for its leading non-NONE hypothesis". This is a *measured* quantity, not a prior. Its only use
in v1 is to scale the record's weight by `clip(reliability, 0.25, 1.0)`; the posterior update itself
remains the likelihood model's job. Stage C reports the reliability model's own calibration, because a
miscalibrated reliability model is worse than none.

## 4. Evidence dependence

Two records are dependent when they share the conditions that generated them. The structural rule in
`docs/03` §5 gives `ρ(r, r')` from the action pair. Dependence enters three places:

### 4.1 Weight of a new record

```
w(r) = 1 / max(1, 1 + Σ_{r' < r} ρ(r, r'))        ∈ (0, 1]
```

A first record has weight 1; a duplicate of one record has weight ½; a look that shares geometry with
two earlier looks has weight ⅓. The weight can never exceed 1, so dependence can only reduce
influence.

### 4.2 Effective sample size

For `n` records with mean pairwise dependence `ρ̄`:

```
n_eff = n / (1 + (n - 1) ρ̄)
```

This is the standard design-effect correction used for clustered samples. It is reported per episode
and per hypothesis. In the demo, three records at the same view under three lights have
`ρ̄ = 0.55`, so `n_eff = 3 / (1 + 2·0.55) = 1.43`; a light change followed by two zooms at the same
light has higher `ρ̄` and a lower `n_eff`. The number is descriptive, not a posterior parameter.

### 4.3 Variance inflation

The variance inflation factor of the accumulated evidence is computed from the empirical covariance of
the feature residuals on the calibration split:

```
VIF_j = 1 / (1 - R_j²)      where R_j² is from regressing feature j on the other features
```

`VIF` is reported, not used to rescale features. RHAN-IM's VIF ≈ 11.6 is the motivating precedent; in
this simulator the expected VIF is lower because the features are less collinear by construction. If
the measured VIF is not materially above 1, H3's mechanism is absent in this laboratory and must be
created by the dependence rule, not by feature collinearity. That possibility is stated in
`docs/13` (risk R4).

## 5. Hypothesis conflict

A record conflicts with the current leader `h*` when

- its `llr` for `h*` is ≤ −log 3, **or**
- its pair evidence contradicts the persistence predicted by `h*` by more than the calibrated
  tolerance (a defect should persist; if it does not, either the defect hypothesis or the observation
  is wrong).

Conflict is not averaged away. Each conflicting record is listed in `belief.conflicts` with its id,
the hypothesis it supports, and the magnitude. A conflict does two things:

1. it adds a `CONFLICT` debt item (severity 0.5) to the leader;
2. if the record's own weight × conflict magnitude exceeds the leader's margin, the stopping rule is
   forced to continue if budget remains — a system may not commit while a visible contradiction is
   unresolved and an observation could resolve it.

A conflict is *resolved* only when a later record's evidence explains it (e.g. a second light change
confirms the reflection) or when the leader changes. The revision log records which record caused the
change.

## 6. What is measured, what is designed

| Quantity | Status | Where measured |
|---|---|---|
| dependence weights (0.45–1.00) | **designed constants** | compared with empirical residual correlation in `verify --dependence-report` |
| quality components | defined transforms | their predictive value measured in Stage C |
| reliability model coefficients | fitted | calibration split, reported with its own ECE |
| temperature | fitted | calibration split |
| VIF | measured | calibration split |
| n_eff | derived | per episode, printed in the trace |

The distinction matters: a designed constant that the data do not support is a weakness to report,
not a mechanism to claim. §5 of the demo results reports the empirical dependence comparison.

## 7. Failure cases the design accepts

- **Occlusion is not evidence of absence.** An occluded ROI raises debt; it does not support `NONE`.
- **Glare can masquerade as a bright anomaly.** Glare lowers quality and therefore weight; it does not
  invert the bright-ridge feature.
- **A changed object looks like a contradiction.** If the world changes between observations, the
  ledger will flag conflict. The stopping rule treats this as a reason to keep looking, not as a
  reason to discard the earlier record; the revision log will show the change. Detecting *that* the
  world changed (rather than the observation being bad) is Gate D's secondary question and is
  explicitly not solved in v1.
