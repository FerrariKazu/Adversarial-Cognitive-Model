# Chapter 19 — Generation-0 Evidence: Empirical Record and Infrastructure Confounds

## 1. In one sentence
Generation-0 produced a 16-seed verified table establishing D's +9.79 pp robustness advantage over TRADES and the AIS-v2 candidate-preference correlation ($r = 0.706$), but the D2 (ais_v2) and D3 (hpc_belief) rows are confounded by an unintentionally-active SBR runner flag, making their isolated mechanism contributions UNKNOWN.

## 2. The Gen-0 ladder
Gen-0 ran a sequence of experiments on STL-10:

> **gen0 → sbr0 → sbr1 → sbr2 → sbr3 → sbr4 → ais_v2 (D2) → hpc_belief (D3)**

Each rung added or swapped one mechanism, gated by pre-registered criteria. Results were evaluated under the 16-seed protocol (Chapter 18).

## 3. Verified headline numbers

| Checkpoint | Clean acc | Adv acc PGD-100 @0.094 |
|:---|:---:|:---:|
| TRADES Large baseline | 54.81 ± 2.35 | 24.23 ± 1.94 |
| **D (donor: ais_hpc)** | **54.96 ± 2.37** | **34.02 ± 3.24** |
| E1 recon | 58.06 ± 2.38 | 33.12 ± 2.64 |
| E2 SBR (intentional) | 45.06 ± 3.30 | 33.42 ± 2.93 |
| E3 T6 | 57.25 ± 2.54 | 30.77 ± 2.98 |
| sbr2 | 63.35 ± 2.78 | 23.58 ± 2.45 |
| sbr3 | 59.44 ± 2.78 | 27.69 ± 3.11 |
| sbr4 | 61.79 ± 2.71 | 25.06 ± 2.95 |
| ais_v2 (D2) | 48.25 ± 2.20 | 35.65 ± 2.56 |
| hpc_belief (D3) | 45.06 ± 2.64 | 33.96 ± 2.26 |

*Source: `report/Gen0.md`, machine-verified against HF per-seed CSVs.*

## 4. What is VERIFIED (16-seed, matched protocol)

| Finding | Status |
|:---|:---:|
| D beats TRADES by +9.79 adv at equal clean (2σ = 7.72) | **VERIFIED** |
| ais_v2 vs TRADES: +11.42 adv (2σ = 6.42) — largest margin | **REAL, but confounded** |
| ais_v2 vs D: +1.63 adv (2σ = 8.26) — not significant | **NOT SIGNIFICANT** |
| hpc_belief vs D: −0.06 adv — nothing | **NOT SIGNIFICANT** |
| Every SBR rung vs D: clean +4.5–8.4pp (real), adv −6.3–10.4pp (real) | **VERIFIED** |
| E1 recon vs D: adv −0.90pp — not significant; clean +3.10pp — not significant | **NOT SIGNIFICANT** |

## 5. The confound: `_nx_trainer` SBR flag
**Affected**: ais_v2 (D2), hpc_belief (D3). Both evaluated via `sweep_rhan_nx_*` / the `_nx_trainer` runner, which **hardcodes `--enable-sbr`** for every stage. These were supposed to be isolated swap tests (AIS-v2 alone; belief-HPC alone). They are not. Both carry legacy SBR (16 slots, 512-dim) unintentionally.

**Not affected**: D, E1, E2, E3 — evaluated via the older `sweep_stage3_d_*` / `sweep_stage4_*` provenance path. E2's SBR is intentional. sbr2–sbr4 run through `_nx_trainer`, but SBR being active is correct by design.

**The tell**: D3's clean accuracy (45.06 ± 2.64) and E2's clean accuracy (45.06 ± 3.30) are **identical to two decimal places**, from experiments testing supposedly different mechanisms. This is the SBR-legacy signature dominating both results.

**Consequence**: AIS-v2's and belief-HPC's TRUE isolated effect on 16-seed clean/robust accuracy is **UNKNOWN**. Not "weak"—unknown. Everything else in the table stands as reported.

## 6. The salvageable result: AIS-v2 correlation ($r = 0.706$)
The AIS-v2 smoke-gate result was measured via a **narrower, more targeted diagnostic** (`ais_v2_smoke_gate_v1`, 512 samples) that measured whether the candidate-scoring head's predictions correlate with policy choice—largely orthogonal to whether SBR is active. This gives:

- g1 (gaze-shift): 0.109 ≥ 0.02 threshold ✓
- g2 (candidate-preference): **r = 0.706** (1536 pairs) ≥ 0.05 threshold ✓

This is graded **REQUIRED** evidence for the AIS-v2 mechanism. It is "the single most valuable new fact Gen-0 produced."

## 7. The Pareto lesson
Sorted by clean accuracy, adversarial accuracy comes out inverted across all Gen-0 rungs. Adding structured representational capacity (SBR) slides models along the clean/robust frontier but **never moves the frontier itself**. This is why Gen-1's matched-control requirement exists: any future "mechanism helps" claim must exclude "more parameters/compute helped."

## 8. Gate verdicts

| Gate | Result | Notes |
|:---|:---:|:---|
| sbr0 (`sbr0_gate_v1`) | Passed criteria | BUT: per-slot accuracies ≈ chance; slot vacuity not excluded |
| sbr1 (one-sided collapse) | Passed | |
| ais_v2 (`ais_v2_smoke_gate_v1`) | Passed | Strongest mechanism result of Gen-0 |
| hpc_belief gate | **NOT RECORDED** | Verdict not on roadmap; UNVERIFIED |
| sbr2 masking gate | Computed but not official | UNVERIFIED as artifact |

## 9. Provenance
- Per-seed CSVs: HF `FerrariKazu/rhan-eval-sweep`
- Aggregation script: `scratch/gen0_pull_hf_evals.py`
- Raw cache: `report/_gen0_pull.json`
- Full tables and contradiction log: `report/Gen0.md`
- Roadmap: `FerrariKazu/rhan-checkpoints-rolling/rhan_next_roadmap.json`

## 10. Source references
- `noesis_vision/RHAN_NXA/docs/16_Gen0_Evidence_And_Confounds.md`
- [`report/Gen0.md`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/report/Gen0.md)
- `report/lens_e1_analysis/E1_FULL_AUDIT_REPORT.md`
- Connected to Chapters 17 (experimental DAG) and 18 (evaluation protocols).
