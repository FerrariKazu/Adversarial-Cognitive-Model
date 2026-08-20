# Stage 3 Preregistration — D = AIS-v1 + HPC

**Date frozen:** 2026-08-20
**Frozen BEFORE any D training or evaluation results are observed.**

---

## 1. Objective

Determine whether combining AIS-v1 (halting-only variant) with HPC (hierarchical predictive coding, level 1) produces a meaningful improvement over either mechanism alone and over the TRADES baseline.

---

## 2. Experimental Configuration

### D = AIS-v1 + HPC

| Parameter | Value |
|---|---|
| `enable_ais` | True |
| `ais_halt_enabled` | True |
| `ais_precision_recon_enabled` | False |
| `enable_hpc` | True |
| `hpc_num_levels` | 1 |
| `hpc_error_weight` | 0.10 |
| `enable_sbr` | False |
| `enable_iwm` | False |
| `ais_v2` | NOT implemented |
| `sbr` | NOT implemented |

**Label:** `rhan_next_ais_hpc` (checkpoint: `checkpoints/rhan_next_ais_hpc_best.pth`)
**Arch:** `next`

---

## 3. Reference Systems

| Label | Config | Status |
|---|---|---|
| A: TRADES baseline | Static TRADES-Large, no recurrence | VALIDATED |
| B: AIS-v1 (halting-only) | `enable_ais=True, ais_halt_enabled=True, ais_precision_recon_enabled=False` | VALIDATED |
| C: HPC-only | `enable_ais=False, enable_hpc=True, hpc_num_levels=1, w_hpc=0.10` | VALIDATED |
| D: AIS-v1 + HPC | `enable_ais=True, ais_halt_enabled=True, ais_precision_recon_enabled=False, enable_hpc=True, hpc_num_levels=1, w_hpc=0.10` | TO BE TRAINED |

All systems share:
- Same evaluation dataset (STL-10 test split)
- Same sample-selection protocol (300 samples per seed, `datasets==4.7.0`)
- Same seeds (41–48)
- Same attack implementation (PGD, norm-space)
- Same epsilon convention (ε=0.094 applied directly to normalized inputs)
- Same PGD-50 / PGD-100 settings
- Same checkpoint/evaluation protocol

---

## 4. Primary Evaluation Protocol

| Parameter | Value |
|---|---|
| **Primary metric** | PGD-50 accuracy |
| **Primary epsilon** | ε = 0.094 (norm-space) |
| **Seeds** | 41–48 (8 seeds) |
| **Baseline** | trades_large_baseline |
| **Dataset dependency** | `datasets == 4.7.0` (pinned) |
| **Primary significance criterion** | Δ > 2 · σ_combined |
| **Secondary attack** | PGD-100, ε = 0.094 |
| **Secondary purpose** | Masking / attack-strength sanity check |
| **N samples per seed** | 300 |
| **Batch size** | 32 |

### Crossover Verdict Rule

For each comparison (D vs A, D vs B, D vs C):

| Condition | Verdict |
|---|---|
| Δ > 2 · σ_combined | CROSSOVER REAL |
| 0 < Δ ≤ 2 · σ_combined | positive but NOT significant |
| Δ ≤ 0 | NO improvement |

### Masking Check (PGD-50 → PGD-100)

| Gap | Verdict |
|---|---|
| ≤ 1.0 pp | GENUINE (no masking) |
| 1.0–2.5 pp | BORDERLINE (within documented ~1.5 pp cross-run nondeterminism) |
| > 2.5 pp | POTENTIAL MASKING |

---

## 5. D Interaction Hypothesis (Pre-registered)

### Existing Lens Evidence

**AIS-v1 belief drift:** ~0.060 → ~0.032 across recurrent steps.
Interpretation: AIS-v1's internal belief trajectory becomes more similar to the clean trajectory over recurrence.

**HPC-only belief drift:** ~0.063 → ~0.163 across recurrent steps.
Interpretation: HPC's belief trajectory diverges further from the clean representation under attack.

### Interaction Hypothesis for D

```
AIS-v1:    stabilizing belief drift
HPC-only:  increasing belief drift
D = AIS-v1 + HPC:  unknown interaction

Possible outcomes:

1. Synergistic:
   D reduces belief drift more strongly than AIS-v1.

2. Antagonistic:
   D inherits or amplifies HPC's destabilizing behavior.

3. Redundant:
   D resembles one parent without meaningful additional effect.

4. Non-monotonic interaction:
   drift decreases/increases in a qualitatively different trajectory.
```

The purpose is not to predict a favorable result. The purpose is to make the interaction falsifiable.

---

## 6. Required Comparison Matrix

```
                         Clean      PGD-50      PGD-100

A: TRADES baseline
B: AIS-v1
C: HPC-only
D: AIS-v1 + HPC
```

Then calculate:

```
B - A
C - A
D - A

D - B
D - C
```

The D-vs-B and D-vs-C comparisons are particularly important. They tell us whether the combined architecture actually adds something beyond simply combining modules in name.

---

## 7. Training Protocol

```
smoke test
    ↓
health gate
    ↓
isolation if required
    ↓
full 60-epoch training
    ↓
matched multi-seed evaluation
    ↓
mechanistic Lens analysis
    ↓
honest verdict
```

### Curriculum

| Epochs | ε | lr |
|---|---|---|
| 1–20 | 0.031 | 0.003 |
| 21–40 | 0.062 | 0.002 |
| 41–60 | 0.094 | 0.001 |

### Base Checkpoint

`checkpoints/rhan_next_ais_v1_halting_only_best.pth` (validated Stage 1 halting-only)

---

## 8. Lens Analysis Requirements

For the same fixed image subset, compare A/B/C/D under clean input and PGD ε=0.094.

Measure:
- Belief drift across recurrent steps: distance(belief_clean_step_t, belief_adversarial_step_t)
- Mean drift per step
- Distribution of drift
- Trajectory plot
- Per-model comparison
- Representative Lens visualizations

Central question: Does combining AIS and HPC produce a more stable internal belief trajectory than either mechanism alone?

---

## 9. Mechanistic Interaction Classification

Classify D as one of:

| Classification | External behavior | Internal belief drift |
|---|---|---|
| SYNERGISTIC | Significant robustness improvement | Reduced drift beyond AIS-v1 |
| ANTAGONISTIC | No improvement or degradation | Increased drift, possibly beyond HPC-only |
| REDUNDANT | Resembles one parent | Resembles one parent |
| MIXED / NON-MONOTONIC | Qualitatively different trajectory | Different drift pattern |
| INCONCLUSIVE | Measurements disagree | Measurements disagree |

---

## 10. Scientific Discipline Rules

1. Do not cherry-pick seeds.
2. Do not change the primary epsilon after seeing results.
3. Do not change the primary metric after seeing results.
4. Do not increase/decrease the number of seeds opportunistically.
5. Do not remove failed runs.
6. Do not call a positive directional result "significant."
7. Do not interpret correlation as causation.
8. Do not infer mechanism from accuracy alone.
9. Do not call AIS-v1 "information seeking" without evidence.
10. Do not call HPC "belief stabilizing" because that was the original hypothesis; the existing evidence actually points in the opposite direction.
11. Do not silently change dependencies.
12. Do not overwrite historical results.
13. Do not optimize the architecture specifically against the observed test outcome.

If something unexpected happens, stop and investigate it before continuing.

---

## 11. Negative Results That Must Be Preserved

- AIS-v1 did not establish significant superiority (+8.5 pp, NOT significant).
- HPC-only did not establish significant superiority (+3.92/+4.29 pp, NOT significant).
- HPC belief drift increased under attack (0.063 → 0.163).
- AIS-v1 belief drift decreased under attack (0.060 → 0.032).
- AIS-v1 was not demonstrated to perform genuine information-seeking.
- Earlier apparent crossovers were affected by an evaluation dependency issue.
- PGD-50 → PGD-100 behavior does not indicate masking in any checkpoint.

---

## 12. SBR and AIS-v2 Lock Status

**SBR:** LOCKED during Stage 3. Do not implement `StructuredBeliefState`, object slots, relations, distributional belief, or μ/Σ hypothesis distributions.

**AIS-v2:** REMAINS SEPARATE. Do not implement AIS-v2 as part of D. AIS-v2 should eventually replace AIS-v1 as an isolated experiment with its own validation sequence.

---

## 13. Traceability Requirements

Every headline result must be traceable back to:
- Exact git commit
- Exact environment (PyTorch, CUDA, datasets version)
- Exact dataset version
- Exact configuration
- Exact seeds
- Exact checkpoints (SHA-256)
- Exact attack parameters
- Raw per-seed results
- Aggregate results
- Lens results
- Interpretation
- Limitations
- Final verdict
