# RHAN-Bi v1 — demo report

> **Claim level: feasibility.** This is a demo-scale run, not the pre-registered confirmatory N=400 Gate E. No result here is a confirmed claim.

- package `1.0.0`, config `16bb095c3497`, git `ca268c135b37`, python 3.10.12
- seeds: 7, N train/cal/dev/test = 240/120/120/120, budget 3

## Gate A — single-observation competence

- accuracy **0.817**, balanced **0.817**
- defect-vs-reflection balanced accuracy **0.767** (required band 0.55–0.80: the task must remain ambiguous)
- confusion (rows truth, cols predicted, labels ['defect', 'reflection', 'shadow', 'none']): `[[26, 1, 0, 3], [2, 20, 0, 8], [0, 0, 29, 1], [0, 7, 0, 23]]`
- gate: **PASS**

## Gate B — spatial evidence

- IoU@0.3 **0.800**, mean IoU 0.346, pointing **1.000**, localization-conditioned 1.000
- gate: **PASS**

## Gate C — reliability and coverage

- temperature **0.715**; ECE 0.104 → **0.072**; Brier 0.246; NLL 0.532
- selective risk at coverage 0.5: **0.050**
- conformal threshold 0.675: calibration coverage 0.775, dev coverage **0.875**, dev selective risk 0.114
- gate: **PASS** (coverage floor 0.40 at risk target 0.200)

## Gate D — belief revision and dependence

- `ledger`: accuracy 0.850, ECE 0.107, revision accuracy 0.111, mean confidence 0.878
- `pooled`: accuracy 0.567, ECE 0.393, revision accuracy 0.000, mean confidence 0.955
- `no_pair`: accuracy 0.850, ECE 0.108, revision accuracy 0.111, mean confidence 0.866
- `no_dep`: accuracy 0.842, ECE 0.105, revision accuracy 0.111, mean confidence 0.946
- revision gain vs pooled: **0.111**; confidence gap (ledger − no_dep) -0.068; median n_eff 1.731
- gate: **PASS**

## Gate E — matched-budget policy comparison

| policy | error | accuracy | mean cost | mean confidence | ECE |
|---|---|---|---|---|---|
| falsification | 0.192 | 0.808 | 3.52 | 0.911 | 0.109 |
| eig | 0.192 | 0.808 | 3.17 | 0.947 | 0.141 |
| confidence | 0.183 | 0.817 | 3.02 | 0.927 | 0.129 |
| random | 0.183 | 0.817 | 4.09 | 0.930 | 0.133 |
| fixed_light_first | 0.183 | 0.817 | 4.00 | 0.950 | 0.136 |
| oracle | 0.175 | 0.825 | 3.25 | 0.932 | 0.134 |

- falsification − eig: **0.000** (95% CI [-0.042, 0.042]), McNemar p = 1.0000
- falsification − confidence: **0.008** (95% CI [-0.017, 0.033]), McNemar p = 1.0000
- falsification − random: **0.008** (95% CI [-0.017, 0.042]), McNemar p = 1.0000
- oracle error 0.175, best deployable 0.183, headroom gap 0.008
- gate: **INCONCLUSIVE** (feasibility, not confirmatory)

## Limitations (required)

- synthetic laboratory only; no real-image result in this run
- defect-vs-reflection ambiguity is a property of the simulator's action set
- abstention is counted as an error in the primary endpoint; coverage is reported separately
- the planner's forward model and stopping rule share errors (risk R5)
- the evidence ledger records provenance, not truth
