# 05 — Belief state and revision

## 1. The hypothesis space

The belief is a distribution over **world causes**, not over the system's epistemic status:

| Hypothesis | Meaning |
|---|---|
| `DEFECT` | a physical surface defect is present at the candidate location |
| `REFLECTION` | the appearance is produced by a reflection of something else |
| `SHADOW` | the appearance is produced by illumination geometry |
| `NONE` | the surface is unremarkable; no candidate anomaly is present |

"Insufficient evidence" is not a fifth hypothesis. It is a state of the belief — high entropy, high
debt, low margin — that the stopping rule converts into an abstention. Making it a hypothesis would
confuse the world with the observer's knowledge, and would let the system "be right" about its own
ignorance while being wrong about the object.

The prior is uniform (0.25 each) in v1. It is not tuned. A scene-conditioned prior is a legitimate
extension, but it would need its own calibration split and is out of scope.

## 2. The evidence model

The likelihood model is a multinomial logistic classifier over the eight anomaly features, fitted on
the calibration rows:

```
log p(c | f) = softmax(W φ(f) + b)_c          φ = standardisation with the calibration mean/std
LLR_c(f) = log p(c | f) - log p(NONE | f)
```

Four causes × eight singleton features. A logistic model replaced an earlier diagonal Gaussian
naive Bayes for a measured reason: the Gaussian saturated. With eight correlated features, small
per-class mean differences divided by per-class standard deviations of about 0.2 produced single-look
LLRs of ±10, and the posterior collapsed to 1.0 on scenes where the evidence was weak (including
clean scenes being called reflections with p = 0.99). The logistic posterior is calibrated on its
fitting split by construction, and Stage C measures the residual miscalibration on held-out data
rather than assuming it away.

Pair evidence is separately fitted as per-cause Gaussians over `(persistence, bright_shift)`, centred
across causes, scaled by 0.5 and clipped to ±4. The scaling is deliberate: the pair measurement is
the single strongest cue in the task, and an unshrunk likelihood would let one pair record dominate
the posterior. The scale is a design constant; the ablation `no_pair` measures what happens without
the pair evidence at all.

## 3. Pair evidence

A pair record compares the current observation with an earlier one at the **same view** and a
**different light**. Two measurements:

- `persistence` — correlation of thresholded dark-anomaly maps (object-aligned);
- `bright_shift` — centroid displacement of the bright anomaly.

Their likelihoods are fitted per cause as truncated Gaussians on the calibration pairs:

| Cause | `persistence` mean (calibration) | Interpretation |
|---|---|---|
| DEFECT | high (≈ 0.8) | fixed to the object; survives a light change |
| REFLECTION | low (≈ 0.0) | tied to illumination geometry; moves |
| SHADOW | low–medium (≈ 0.2) | soft and moves with the light |
| NONE | undefined | no anomaly to persist; the pair contributes no LLR |

The pair's LLR is added with the same dependence weight as its record. This is the only feature that
can separate `DEFECT` from `REFLECTION` on this task, and it exists only if the planner chose a light
change. That is the mechanism the whole programme is built around.

## 4. Update rule

Given ledger records `r_1..r_n` with weights `w_i` and LLR vectors `ℓ_i`, and pair LLRs `π_i`:

```
log w_h = log prior_h + Σ_i w_i · (ℓ_i[h] + π_i[h])
p_h = softmax(log w)_h
```

Properties, each unit-tested:

- adding a record with `ℓ = 0` changes nothing (neutral evidence is neutral);
- the posterior is invariant to the order of records (addition commutes);
- a record with weight 0 has no effect;
- if all records are duplicates, `n_eff → 1` and the posterior moves by at most one record's worth of
  evidence;
- the posterior is normalised to 1 within 1e-9.

The recomputation is from the full ledger each step, O(n · H) with n ≤ 4 and H = 4 in v1. There is no
incremental state to drift out of sync. `BeliefState` exposes `log_weights` and `posterior`; a
`from_records()` classmethod exists for replay and is checked against the incremental path in tests.

## 5. Contradiction analysis

For each record, conflict against the current leader `h*` is defined in `docs/04` §5. The belief
stores:

```python
conflicts = [
    {"record_id": ..., "supports": "REFLECTION", "against": "DEFECT",
     "magnitude": 1.7, "kind": "llr" | "pair"},
]
```

Rules:

1. A conflict is *open* until a later record with non-trivial weight supports the leader on the same
   discriminating feature, or until the leader changes.
2. An open conflict with `magnitude * weight > margin` blocks commitment while budget remains.
3. Conflicts never delete a record; they are listed in the trace and in the report.
4. A conflict whose source record has quality < 0.35 is marked `weak_source` and does not block
   commitment, but is still reported.

## 6. Evidence debt

Debt is a per-hypothesis list of unresolved weaknesses, each with a code, a severity in [0, 1] and a
short human-readable reason. The list is computed from the ledger, not stored as free text.

| Code | Trigger | Severity |
|---|---|---|
| `OCCLUDED` | any record with `occlusion > 0.15` covering the candidate ROI | 0.6 · occlusion fraction |
| `GLARE` | any record with `glare > 0.02` near the anomaly | 0.5 · min(1, glare/0.1) |
| `LOW_QUALITY` | record quality < 0.35 | 0.5 |
| `CORRELATED_ONLY` | all supporting records have pairwise ρ ≥ 0.8 | 0.4 |
| `CONFLICT` | an open conflict against this hypothesis | 0.5 |
| `NO_DISAMBIGUATION` | leader's margin < 0.15 and no light change has been observed | 0.5 |
| `OUT_OF_RANGE` | a feature z-score exceeds 4 in absolute value | 0.3 |
| `SINGLE_SOURCE` | the leader is supported by exactly one record | 0.3 |

Total debt is the capped sum: `debt_h = min(1, Σ severity)`. The demo's commit rule uses
`debt < 0.35`; this threshold is a design choice and is reported with the sensitivity analysis in
Stage C (thresholds 0.2 / 0.35 / 0.5).

The `NO_DISAMBIGUATION` item is the one that turns "I am unsure" into "I should change the light":
it names *why* the uncertainty cannot be resolved by looking more carefully at the same conditions.

## 7. Revision log

Every change of the MAP hypothesis is recorded:

```python
revision = {"step": 2, "from": "DEFECT", "to": "REFLECTION",
            "caused_by": "<record_id>", "delta": -1.9,
            "reason": "pair persistence 0.06 under REFLECTION (LLR +1.9 vs DEFECT -1.4)"}
```

A revision is only valid if it names a record that was added since the previous belief state. The
`verify` command checks this for every episode trace. Revisions are reported as a first-class metric
(H2): *revision accuracy* is the fraction of episodes whose final MAP equals the ground truth among
episodes where an early MAP was wrong, i.e. the system's ability to recover.

## 8. Stop / commit / abstain

After each record:

```
if max_h p_h >= τ_commit and debt_{h*} <= τ_debt:      commit h*
elif budget exhausted and margin < τ_abstain_margin:   abstain
elif expected_value_of_next_action < cost:             commit/abstain (no more looking)
else:                                                  continue
```

Defaults: `τ_commit = 0.80`, `τ_debt = 0.35`, `τ_abstain_margin = 0.10`, budget 3. Thresholds are
fitted/frozen on the calibration split before the test split is touched (`docs/09` Gate C), and the
demo reports the sensitivity of the commit threshold.

A commit is not a claim of certainty. It is the system saying: on the evidence available, the
expected loss of stopping now is lower than the expected cost of another observation. The report
prints the posterior, the debt and the number of observations behind every commitment.

## 9. What the belief cannot represent (v1)

- **Continuous causes and parameter uncertainty.** Four discrete causes only; no "defect of severity
  0.3 at location (x, y)".
- **Object identity across scenes.** Each episode is one scene; the ledger is per-episode.
- **Time and state change.** `advance_scene` exists in the lab, but the belief has no explicit
  state-change model; a genuine change appears as a contradiction (see `docs/08` S4) and is reported
  as such.
- **Dependence learned from data.** ρ is a designed constant with an empirical check, not a fitted
  covariance.
- **Calibration under distribution shift.** Temperature scaling fitted on in-distribution calibration
  scenes does not guarantee calibration under shift; the OOD flag is a warning, not a certificate.
