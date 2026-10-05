# Chapter 18 — Evaluation Protocols: PGD Robustness and Verification Metrics

## 1. In one sentence
RHAN-NXA uses the Generation-0 validated 16-seed adversarial evaluation protocol: PGD-100 and PGD-50 at $\epsilon = 0.094$ in norm space, 300 samples/seed, 16 seeds (41–56), with adversarial accuracy as the primary metric and all comparators carried via a frozen registry.

## 2. The protocol specification

| Parameter | Value | Source |
|:---|:---|:---|
| Dataset | STL-10 test (Gen-0 / Gen-1 STL Phase) | Gen-0 validated |
| Samples per seed | 300 | Gen-0 protocol |
| Seeds | 16 (seeds 41–56) | Gen-0 protocol |
| Attack | PGD-100 (primary) and PGD-50 (secondary) | Gen-0 protocol |
| $\epsilon$ | 0.094 (normalized) | Gen-0 protocol |
| Primary metric | Adversarial accuracy (percentage) | |
| Secondary metric | Clean accuracy | |
| Significance test | $|\Delta| > 2\sqrt{\sigma_a^2 + \sigma_b^2}$ | Conservative crossover criterion |
| Comparators | Registered frozen (never re-evaluated) | Prevents seed-selection bias |

**Note**: AutoAttack evaluation is expected but details are pending.

## 3. Why 16 seeds?
Single-seed evaluation has high variance—a model can appear to "improve" by 3–5 percentage points on a single evaluation run just due to noise in the attack initialization. The 16-seed protocol provides:
- Standard deviation estimates across seeds ($\pm$ values in all reported numbers)
- A conservative crossover test that requires $|\Delta| > 2\sqrt{\sigma_a^2 + \sigma_b^2}$

This is what makes the Gen-0 headline numbers reliable: D vs TRADES (+9.79 pp adversarial, 2σ = 7.72) is real; ais_v2 vs D (+1.63 pp, 2σ = 8.26) is not significant.

## 4. The frozen comparator registry
All baseline comparators (D, TRADES, sbr2–sbr4, ais_v2, hpc_belief) are evaluated **once** and their numbers are stored in a registry. Subsequent arms compare against registry numbers; they never re-evaluate baselines. This prevents:
- Baseline numbers being updated after new arms reveal weakness
- Seed-selection bias in comparator evaluation

## 5. Evaluation scripts
The following evaluation scripts exist in the workspace:

| Script | Purpose |
|:---|:---|
| `eval_pgd_final.py` | Final PGD evaluation |
| `eval_aa_v2.py` | AutoAttack evaluation |
| `eval_pgd_sweep.py` | Sweep across configurations |
| `eval_quick_perclass.py` | Per-class accuracy breakdown |
| `evaluation/clean_and_robust.py` | Matched norm-space clean+robust evaluation |
| `evaluation/compactness_report.py` | Model compactness measurement |

## 6. Metric interpretation rules

### On adversarial accuracy
- Numbers are reported as mean ± std over 16 seeds.
- Only differences exceeding 2σ are described as "real."
- Smaller differences are reported as "not significant" and not interpreted causally.

### On clean accuracy
- A model should not purchase adversarial robustness improvements by sacrificing clean accuracy.
- The Gen-0 Pareto lesson: every SBR rung bought +4.5–8.4 pp clean and paid −6.3–10.4 pp adversarial.

### On gate results
- A gate pass means the pre-registered criterion passed.
- A gate pass does not mean the mechanism was validated as effective (see Chapter 22: sbr0 gate-pass example).

## 7. What the evaluation does NOT currently measure
- Robustness to adaptive attacks (AA second-order)
- Cross-dataset transfer
- Out-of-distribution generalization
- Computational cost at inference (though compactness is measured)

## 8. Scientific status
- **16-seed PGD protocol**: **REQUIRED** (matched to all Gen-0 numbers).
- **Frozen comparator registry**: **REQUIRED** (isolation rule).
- **Significance threshold $2\sqrt{\sigma_a^2 + \sigma_b^2}$**: **REQUIRED** (pre-registered conservative criterion).

## 9. Source references
- `evaluation/clean_and_robust.py`
- `evaluation/compactness_report.py`
- `noesis_vision/RHAN_NXA/docs/16_Gen0_Evidence_And_Confounds.md`
- Connected to Chapter 17 (experimental DAG) and Chapter 19 (Gen-0 evidence).
