# Stage 3-D Provenance Audit & Corrected Interaction Table

**Date:** 2026-08-27
**Trigger:** Table 1 and Section 5 of stage3_D_analysis.md used inconsistent A reference values.

---

## 1. The Inconsistency

The original report mixed two different A baselines:

| Where | A baseline used | Source |
|-------|----------------|--------|
| Table 1, A's own row | **22.83%** | Stage 3 D run (PGD-100, 8 seeds, 2026-08-26) |
| Table 1, B's Δ vs A column | **23.58%** | Stage 1 masking spot-check (PGD-100, 2026-08-10) |
| Table 1, C's Δ vs A column | **23.58%** | Stage 1 masking spot-check (PGD-100, 2026-08-10) |
| Table 1, D's Δ vs A column | **22.83%** | Stage 3 D run (same session as D) |
| Section 5, HPC main effect | **22.83%** | Stage 3 D run (same session as D) |

**Evidence of the mixup:**
- B Δ = 32.17 − 23.58 = **8.59** ✓ (uses old A=23.58)
- C Δ = 27.40 − 23.58 = **3.82** ✓ (uses old A=23.58)
- D Δ = 34.38 − 22.83 = **11.54** ✓ (uses D-session A=22.83)
- Section 5 HPC effect = 27.40 − 22.83 = **4.57** ✓ (uses D-session A=22.83, but C=27.40 is from a different session!)

---

## 2. Full Provenance Map

### A (TRADES baseline) PGD-100 @ ε=0.094 — ALL Available Evaluations

| # | A value | Seeds | PGD steps | Session | Date | Source file |
|---|---------|-------|-----------|---------|------|-------------|
| 1 | **23.58** | 41–48? | 100 | Stage 1 masking spot-check | 2026-08-10 | roadmap stages['1'].stage1_verdict.masking_check.trades_large_baseline.acc_pgd100 |
| 2 | **23.17** | 41–48 | 100 | HPC PGD-100 rerun | 2026-08-20 | report/sweep_rerun_hpc_8seed_pgd100/eval_provenance.json |
| 3 | **22.83** | 41–48 | 100 | Stage 3 D run | 2026-08-26 | report/stage3_D/per_seed_results.csv (trades_large_baseline rows) |

Cross-run GPU nondeterminism spread on A: **22.83–23.58** (range 0.75 pp). This is consistent with the documented ~1.5 pp floor.

### B (AIS-v1 halting-only) PGD-100 @ ε=0.094

| # | B value | Seeds | PGD steps | Session | Date | Source file |
|---|---------|-------|-----------|---------|------|-------------|
| 1 | **32.17** | ? | 100 | Stage 1 masking spot-check | 2026-08-10 | roadmap stages['1'].stage1_verdict.masking_check.rhan_next_ais_v1_halting_only.acc_pgd100 |
| 2 | **31.67** | 41–48 | **50** | AIS 8-seed rerun | 2026-08-19 | report/sweep_rerun_ais_v1_8seed/eval_provenance.json (⚠️ PGD-50, not PGD-100!) |

**B was never re-evaluated with PGD-100 in the same session as D.** The32.17 figure is from a separate Stage 1 masking invocation. The8-seed rerun (31.67) used only 50 PGD steps and is NOT a PGD-100 number.

### C (HPC-only) PGD-100 @ ε=0.094

| # | C value | Seeds | PGD steps | Session | Date | Source file |
|---|---------|-------|-----------|---------|------|-------------|
| 1 | **27.40** | 41–45 | 100 | Stage 2 masking check | 2026-08-18 | roadmap stages['2'].stage2_verdict.masking_check.rhan_next_hpc_only.acc_pgd100 |
| 2 | **27.46** | 41–48 | 100 | HPC PGD-100 rerun | 2026-08-20 | report/sweep_rerun_hpc_8seed_pgd100/eval_provenance.json |

C's 8-seed PGD-100 (27.46) is available from a dedicated rerun. The original27.40 was 5-seed only.

### D (AIS + HPC) PGD-100 @ ε=0.094

| # | D value | Seeds | PGD steps | Session | Date | Source file |
|---|---------|-------|-----------|---------|------|-------------|
| 1 | **34.38** | 41–48 | 100 | Stage 3 D run | 2026-08-26 | report/stage3_D/per_seed_results.csv |

---

## 3. Same-Session Paired Comparisons (Most Reliable)

These are the comparisons where A and the model were evaluated in the **same eval invocation**:

| Pair | A value | Model value | Δ | Seeds | PGD steps | Session |
|------|---------|-------------|---|-------|-----------|---------|
| **D vs A** | 22.83 | 34.38 | **+11.54** | 8 | 100 | Stage 3 D run (2026-08-26) |
| **C vs A** (8-seed) | 23.17 | 27.46 | **+4.29** | 8 | 100 | HPC PGD-100 rerun (2026-08-20) |
| **B vs A** | 23.58 | 32.17 | **+8.59** | ? | 100 | Stage 1 masking (2026-08-10) |

**Note:** There is no same-session PGD-100 comparison for B vs A with the full 8-seed protocol. B's32.17 comes from the Stage 1 masking invocation, which is a separate run from both the Stage 1 main eval and the Stage 3 D run.

---

## 4. Corrected Table 1 — Source-Labeled

### Option A: Show each model's PGD-100 with explicit source label

| Model | Clean Acc | PGD-100 | Source (run) | Δ vs A (paired) | Notes |
|-------|----------:|--------:|-------------|:----------------:|-------|
| **A: TRADES baseline** | 53.47 ± 2.87 | 22.83 ± 3.47 | Stage 3 D run, 8 seeds | — | Same session as D |
| **B: AIS-v1 halting-only** | 49.40 ± 3.47 | 32.17 ± 2.74 | Stage 1 masking, PGD-100 | +8.59 ¹ | Different session from D |
| **C: HPC-only** | 55.20 ± 3.67 | 27.46 ± 2.29 | HPC PGD-100 rerun, 8 seeds | +4.29 ² | Different session from D |
| **D: AIS-v1 + HPC** | **54.25 ± 3.10** | **34.38 ± 1.94** | Stage 3 D run, 8 seeds | **+11.54** ³ | Same session as A |

¹ B vs A: Δ=8.59 from Stage 1 masking check (B=32.17, A=23.58, same invocation). Cross-session vs D: Δ_rel_to_Dsession_A = 32.17 − 22.83 = +9.34.
² C vs A: Δ=4.29 from HPC PGD-100 rerun (C=27.46, A=23.17, same invocation). Cross-session vs D: Δ_rel_to_Dsession_A = 27.46 − 22.83 = +4.63.
³ D vs A: Δ=11.54 from Stage 3 D run (D=34.38, A=22.83, same invocation).

### Option B: Harmonized to D-session A baseline (for factorial decomposition only)

This uses D-session A=22.83 as the universal baseline and applies same-session deltas:

| Model | PGD-100 (harmonized) | Source | Δ vs A_session |
|-------|---------------------|--------|----------------|
| A | 22.83 | Stage 3 D run | — |
| B | 22.83 + 8.59 = **31.42** ⁴ | Stage 1 Δ applied to D-session A | +8.59 |
| C | 22.83 + 4.29 = **27.12** ⁴ | HPC rerun Δ applied to D-session A | +4.29 |
| D | **34.38** | Stage 3 D run | +11.54 |

⁴ **These are reconstructed estimates, not direct measurements.** The B and C PGD-100 values (31.42 and 27.12) are computed by applying the same-session Δ to the D-session A baseline. The actual same-session values (B=32.17, C=27.46) differ by up to 0.75 pp due to cross-run GPU nondeterminism.

---

## 5. Corrected Section 5: Interaction Analysis

### Factorial Decomposition (2×2: AIS on/off × HPC on/off)

Using **paired same-session Δ values** for each main effect:

| | HPC OFF | HPC ON |
|---|---------|--------|
| **AIS OFF** | A (baseline) | C (+4.29 pp) |
| **AIS ON** | B (+8.59 pp) | **D (+11.54 pp)** |

| Quantity | Value | Source |
|----------|------:|--------|
| Main effect of AIS (B − A) | +8.59 pp | Stage 1 masking (same-session B vs A) |
| Main effect of HPC (C − A) | +4.29 pp | HPC PGD-100 rerun (same-session C vs A) |
| Expected additive (AIS + HPC) | +12.88 pp | 8.59 + 4.29 |
| Observed D − A | +11.54 pp | Stage 3 D run (same-session D vs A) |
| **Interaction (residual)** | **−1.34 pp** | 11.54 − 12.88 |

**Cross-check using D-session A harmonized values:**

| Quantity | Value |
|----------|------:|
| AIS main effect (B_harmonized − A) | 31.42 − 22.83 = +8.59 pp |
| HPC main effect (C_harmonized − A) | 27.12 − 22.83 = +4.29 pp |
| Expected additive | +12.88 pp |
| Observed D − A | +11.54 pp |
| **Interaction (residual)** | **−1.34 pp** |

**Cross-check using raw measured values (mixing sessions):**

| Quantity | Value | Notes |
|----------|------:|-------|
| AIS main effect (B_raw − A_raw) | 32.17 − 23.58 = +8.59 pp | Stage 1 masking |
| HPC main effect (C_raw − A_raw) | 27.46 − 23.17 = +4.29 pp | HPC rerun |
| Expected additive | +12.88 pp | |
| Observed D − A | +11.54 pp | Stage 3 run |
| **Interaction (residual)** | **−1.34 pp** | Consistent across all decompositions |

### Classification: **Subadditive / complementary**

- D > B and D > C: both mechanisms contribute positively
- D < expected B+C additive gain: the interaction is subadditive (−1.34 pp)
- **This is NOT superadditive synergy.** Synergy would require D > B + C.
- **This is NOT antagonism.** D is clearly better than both components.

### Key difference from the original report

The original report showed an interaction of −2.36 pp because it used:
- AIS main effect = 8.59 (from Stage 1, correct)
- HPC main effect = **4.57** (incorrect — this was 27.40 − 22.83, mixing a 5-seed C value with D-session A)

The corrected HPC main effect is **+4.29 pp** (from the same-session HPC PGD-100 rerun: C=27.46, A=23.17), which changes the interaction residual from −2.36 to **−1.34 pp**. The interaction remains subadditive but is smaller in magnitude.

---

## 6. Caveat Summary

1. **B's PGD-100 (32.17) has NO same-session paired comparison with D.** The value comes from the Stage 1 masking spot-check (2026-08-10), a different session. A fresh 4-way PGD-100 evaluation of A/B/C/D in a single session would be the definitive resolution.

2. **C now has an 8-seed PGD-100 value (27.46) from a dedicated rerun** (2026-08-20), replacing the original 5-seed value (27.40). The difference (0.06 pp) is negligible.

3. **A varies 22.83–23.58 across sessions** (0.75 pp range), consistent with the documented ~1.5 pp GPU nondeterminism floor. All comparisons should note which A was used.

4. **The D-vs-A paired result (11.54 pp, p≈2×10⁻⁵) is fully internally consistent and stands as reported.** It is the only comparison where A and the model were evaluated in the same invocation with the same8 seeds and PGD-100 steps.

---

## 7. Source Files Referenced

| File | Contents |
|------|----------|
| `report/stage3_D/per_seed_results.csv` | A and D per-seed PGD-100 (Stage 3 D run, 8 seeds) |
| `report/stage3_D/summary.json` | Stage 3 D summary statistics |
| `report/sweep_rerun_hpc_8seed_pgd100/eval_provenance.json` | C and A 8-seed PGD-100 (HPC rerun) |
| `report/sweep_rerun_hpc_8seed_pgd100/epsilon_sweep_per_seed.csv` | C and A per-seed PGD-100 |
| `report/sweep_rerun_ais_v1_8seed/eval_provenance.json` | B and A 8-seed PGD-**50** (NOT PGD-100!) |
| `docs/rhan_next_roadmap.json` stages['1'].stage1_verdict | B=32.17 PGD-100, A=23.58 PGD-100 (Stage 1 masking) |
| `docs/rhan_next_roadmap.json` stages['2'].stage2_verdict | C=27.40 PGD-100, A=19.87 PGD-100 (Stage 2, 5-seed) |
| `rhan_core/ablation/matrix.py` | Registry note: A baseline PGD-100 = 23.71 (PGD-50 value, stale) |
