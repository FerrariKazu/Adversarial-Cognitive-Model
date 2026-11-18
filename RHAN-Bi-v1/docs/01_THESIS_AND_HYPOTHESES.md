# 01 — Thesis and hypotheses

## 1. The claim under test

The programme tests one primary claim and three supporting claims. All four are stated so that a
specific result would falsify them.

**Primary (H1).** At a fixed observation budget, a falsification-directed observation policy that
accounts for evidence reliability and dependence achieves lower decision error than random,
confidence-driven, and expected-information-gain (EIG) selection on the inspection task.

**Supporting:**

- **H2 (belief revision).** A belief state that records quality and dependence revises toward the
  truth when a disambiguating observation arrives, and does so more reliably than a confidence-only
  baseline that pools observations without provenance.
- **H3 (dependence).** Discounting correlated observations prevents the overconfidence produced by
  repeated near-duplicate looks, improving calibration without reducing coverage.
- **H4 (identifiability ceiling).** Single-observation classification has a ceiling on
  reflection-versus-defect scenes that is caused by the physical ambiguity of the image, not by
  encoder capacity; additional *disambiguating* observations break the ceiling while additional
  *redundant* observations do not.

## 2. Operational definitions

| Term | Definition used in the experiments |
|---|---|
| decision error | fraction of test scenes whose final committed cause label ≠ ground-truth cause (abstentions counted as errors in the primary endpoint; reported separately) |
| fixed budget | every policy takes exactly *B* = 3 observations per scene in the primary comparison; a secondary analysis compares policies at the budget each stops at |
| falsification-directed | the policy of `docs/06_OBSERVATION_PLANNER.md` §4, with the default weights frozen in `configs/gate_e.yaml` before the run |
| confidence-driven | select the action expected to maximize the posterior of the current leader (confirmation-seeking), ignoring what the observation would reveal if the leader is wrong |
| EIG | select the action maximizing expected mutual information between the observation and the cause variable |
| random | uniform over the legal action set, excluding actions already taken in the same episode |
| confidence-only baseline | the same encoder and features, but evidence pooled as a single averaged feature vector with no dependence discount and no pair evidence (H2/H3 ablations) |
| disambiguating observation | an observation whose likelihood ratio between the leading and second hypothesis has magnitude ≥ log 3 under the fitted evidence model |
| redundant observation | an observation whose dependence with an existing record is ρ ≥ 0.8 |

## 3. Falsifiers

Each hypothesis names the result that would count against it. These are fixed; a post-hoc reframing
of the endpoint after seeing results is a deviation and must be reported as exploratory.

| Hypothesis | Disconfirmed if |
|---|---|
| H1 | the paired difference in decision error (falsification − best baseline) has a 95 % CI that includes 0, or is positive, at N = 400 scenes |
| H2 | revision accuracy after a disambiguating observation is not better than the confidence-only baseline by ≥ 5 percentage points (95 % CI excluding 0) |
| H3 | the dependence-discounted system is not better calibrated (ECE) than the undiscounted ablation by ≥ 0.02, or loses more than 2 points of coverage at matched risk |
| H4 | the single-observation ceiling on reflection-vs-defect is not materially above chance (≥ 0.75 balanced accuracy), i.e. there is no ambiguity to break |

The thresholds are design decisions, not discoveries. They are chosen to be large enough that a
statistically detectable effect is also a practically meaningful one, and small enough that the demo
can run in under a minute per gate.

## 4. Design logic

The experiment is a 2 × 2 × 3 factorial over mechanisms, not a single arm:

| Factor | Levels | Purpose |
|---|---|---|
| evidence model | provenance + dependence vs confidence-only pooled | tests H2, H3 |
| observation source | single observation vs sequential budget 3 | tests H4, and gives H1 its budget |
| policy | falsification, EIG, uncertainty, random | tests H1 |

The simulator's ground truth makes the first factor measurable without a real-world label problem.
The intervention structure (change light, change view, zoom) makes the second meaningful: light
change is the only action that produces the *persistence* measurement that separates a fixed defect
from a moving reflection.

## 5. Why the task is not trivial by construction

A reasonable objection is that a simulator built by the authors can be built so that the preferred
policy wins. Three design choices make that harder:

1. **The perception front-end is fixed before the planner is compared.** All policies see the same
   features, the same likelihood model and the same simulator. Only the selection rule differs.
2. **Single-observation evidence is deliberately ambiguous.** Reflection streaks and defect scratches
   share the dark-ridge feature; the discriminating feature exists only across a light change. If the
   ambiguity were removed, all policies would tie and the experiment would be uninformative.
3. **EIG is a strong baseline, not a straw man.** EIG also values the light change, because it also
   reduces uncertainty. The falsification policy has to earn its difference through *which*
   uncertainty it resolves and through its dependence and cost terms. It is entirely possible that
   EIG matches it; that outcome will be reported.

The simulator's realism is defended in `docs/08_SIMULATOR_SPEC.md` §9 and its limits are stated in
`docs/13_RISKS_ETHICS_LIMITS.md`. No claim from this simulator transfers to a factory floor without
the real-data check in Gate C/D.

## 6. Pre-registration skeleton

To be frozen (commit + `prereg.sh freeze`, or a lone first commit if the script is unavailable)
before the confirmatory Gate E run. The demo in this repository is explicitly **feasibility**, not
confirmation, and is labelled as such in `docs/19_DEMO_RESULTS.md`.

```
# Pre-registration: RHAN-Bi v1 Gate E — falsification-directed observation selection

Question: at B = 3 observations, does falsification-directed selection reduce decision error
          relative to EIG, uncertainty, and random selection?

H1 (directional): mean paired decision error (falsification) < mean paired decision error (EIG).

Primary analysis: paired difference in per-scene 0/1 decision error, falsification − EIG,
                  two-sided 95 % percentile bootstrap CI (10,000 resamples, scene-level),
                  McNemar exact test as a secondary check.

Prediction: difference ≤ −0.05 (falsification better by at least 5 points), based on the
            mechanism argument in docs/06 §4 and the pilot in docs/19.

Sample: N = 400 test scenes, fixed; no optional stopping. Scenes are generated from a fixed seed
        list held out from model fitting and calibration.

Decision rule: confirm H1 if the 95 % CI upper bound < 0 and the McNemar p < 0.05, Holm-corrected
               across the three policy comparisons (EIG, uncertainty, random).
               Disconfirm if the CI includes 0 or the point estimate is positive.

Multiplicity: 3 confirmatory comparisons; Holm–Bonferroni at family-wise α = 0.05.

Secondary (labelled exploratory): calibration (ECE), abstention rate, coverage at risk target,
                                 cost-weighted error, revision accuracy (H2), ECE ablation (H3).
```

The confirmatory run has not been executed. Until it is, no Gate E claim in this repository is
confirmatory; the demo numbers are feasibility evidence about whether the pipeline can run and
whether the effect is worth a confirmatory test.

## 7. Deviations policy

Any change to endpoint, N, policy weights, feature set, or stopping rule after the first confirmatory
data are observed is a deviation. It is recorded in `results/deviations.md` with the reason and the
affected analysis is relabelled exploratory. A deviation that changes the primary endpoint voids the
confirmatory claim for that run.
