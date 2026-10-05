# RHAN-NX Generation-1 consolidated report

Generated 2026-09-08T06:14:34Z — every cell traceable to a fresh eval or an explicitly-labeled comparator DONOR row (rule 1c: asserted against its source CSV before this table was written).


## sbr2 — SBR-2 (adversarial curriculum) (PENDING)

No fresh eval CSV yet — eval not complete. Donor rows below are available for comparison.

| checkpoint | eps | acc_mean | acc_std | source |
|---|---|---|---|---|
| rhan_next_ais_hpc | 0.000 | 54.96 | 2.37 | DONOR |
| rhan_next_ais_hpc | 0.094 | 34.02 | 3.24 | DONOR |
| trades_large_baseline | 0.000 | 54.81 | 2.35 | DONOR |
| trades_large_baseline | 0.094 | 24.23 | 1.94 | DONOR |

  _(DONOR row from report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv, validated 2026-08-31, NOT re-evaluated this session.)_
  _(DONOR row from report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv, validated 2026-08-31, NOT re-evaluated this session.)_

## sbr3 — SBR-3 (relational evidence) (PENDING)

No fresh eval CSV yet — eval not complete. Donor rows below are available for comparison.

| checkpoint | eps | acc_mean | acc_std | source |
|---|---|---|---|---|
| rhan_next_ais_hpc | 0.000 | 54.96 | 2.37 | DONOR |
| rhan_next_ais_hpc | 0.094 | 34.02 | 3.24 | DONOR |
| trades_large_baseline | 0.000 | 54.81 | 2.35 | DONOR |
| trades_large_baseline | 0.094 | 24.23 | 1.94 | DONOR |

  _(DONOR row from report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv, validated 2026-08-31, NOT re-evaluated this session.)_
  _(DONOR row from report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv, validated 2026-08-31, NOT re-evaluated this session.)_

## sbr4 — SBR-4 (uncertainty first-class) (PENDING)

No fresh eval CSV yet — eval not complete. Donor rows below are available for comparison.

| checkpoint | eps | acc_mean | acc_std | source |
|---|---|---|---|---|
| rhan_next_ais_hpc | 0.000 | 54.96 | 2.37 | DONOR |
| rhan_next_ais_hpc | 0.094 | 34.02 | 3.24 | DONOR |
| trades_large_baseline | 0.000 | 54.81 | 2.35 | DONOR |
| trades_large_baseline | 0.094 | 24.23 | 1.94 | DONOR |

  _(DONOR row from report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv, validated 2026-08-31, NOT re-evaluated this session.)_
  _(DONOR row from report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv, validated 2026-08-31, NOT re-evaluated this session.)_

## ais_v2 — D2 = AIS-v2 swap test (PENDING)

No fresh eval CSV yet — eval not complete. Donor rows below are available for comparison.

| checkpoint | eps | acc_mean | acc_std | source |
|---|---|---|---|---|
| rhan_next_ais_hpc | 0.000 | 54.96 | 2.37 | DONOR |
| rhan_next_ais_hpc | 0.094 | 34.02 | 3.24 | DONOR |
| trades_large_baseline | 0.000 | 54.81 | 2.35 | DONOR |
| trades_large_baseline | 0.094 | 24.23 | 1.94 | DONOR |

  _(DONOR row from report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv, validated 2026-08-31, NOT re-evaluated this session.)_
  _(DONOR row from report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv, validated 2026-08-31, NOT re-evaluated this session.)_

## hpc_belief — D3 = belief-HPC swap test (PENDING)

No fresh eval CSV yet — eval not complete. Donor rows below are available for comparison.

| checkpoint | eps | acc_mean | acc_std | source |
|---|---|---|---|---|
| rhan_next_ais_hpc | 0.000 | 54.96 | 2.37 | DONOR |
| rhan_next_ais_hpc | 0.094 | 34.02 | 3.24 | DONOR |
| trades_large_baseline | 0.000 | 54.81 | 2.35 | DONOR |
| trades_large_baseline | 0.094 | 24.23 | 1.94 | DONOR |

  _(DONOR row from report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv, validated 2026-08-31, NOT re-evaluated this session.)_
  _(DONOR row from report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv, validated 2026-08-31, NOT re-evaluated this session.)_

## Final summary — all models, clean / PGD-50 / PGD-100

| checkpoint | eps | acc_mean | acc_std | source |
|---|---|---|---|---|
| rhan_next_ais_hpc | 0.000 | 54.96 | 2.37 | DONOR |
| rhan_next_ais_hpc | 0.094 | 34.02 | 3.24 | DONOR |
| rhan_next_ais_v1_halting_only | 0.000 | — | — | UNAVAILABLE |
| rhan_next_ais_v1_halting_only | 0.094 | — | — | UNAVAILABLE |
| rhan_next_hpc_only | 0.000 | 55.20 | 3.67 | DONOR |
| rhan_next_hpc_only | 0.094 | 27.73 | 2.28 | DONOR |
| trades_large_baseline | 0.000 | 54.81 | 2.35 | DONOR |
| trades_large_baseline | 0.094 | 24.23 | 1.94 | DONOR |

  _Legend: DONOR = validated comparator rows (byte-verified against their cited source CSV, NOT re-evaluated); FRESH = evaluated this Generation-1 run._
  _UNAVAILABLE: rhan_next_ais_v1_halting_only donor rows could not be loaded (comparator 'rhan_next_ais_v1_halting_only' source CSV not found locally (report/sweep_stage1_ais_v1_halting_only_merged/epsilon_sweep_per_seed.csv) and could not be restored from HF (FerrariKazu/rhan-eval-sweep/sweep_stage1_ais_v1_halting_only_merged/epsilon_sweep_per_seed.csv). The rows it would cite do not exist — refusing to fabricate a comparison.) — cells marked UNAVAILABLE, NOT fabricated, NOT re-evaluated (rule 1b)._
  _NOTE: rhan_next_hpc_only rows are 5-seed validated (41..45) vs D/baseline's 16 — any comparison mixing them must flag the seed-count mismatch, never silently average across mismatched n (rule 1b)._

## What this tells us about Generation 2

_(Populated after all stages report. Candidates: SBR-4+AIS-v2, SBR-4+belief-HPC, AIS-v2+belief-HPC, all three combined, IWM — a recommendation, not a decision.)_
