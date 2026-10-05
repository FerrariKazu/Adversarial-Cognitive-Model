# GEN1_RESULTS_MASTER.md — RHAN-NXA Generation-1 Foundation: Complete Extracted Results

Generated 2026-09-29 20:29 UTC from the frozen Gen-1 production-run artifacts (local tree at commit `0762a5e`). Every number below is read directly from the listed artifact files; derived statistics are labeled with their exact computation. **No interpretation is offered beyond what the artifacts support.** Sections marked ABSENT record metric families the artifacts do not contain.

## 1. Run identity & frozen protocol

| Field | Value | Source |
|---|---|---|
| Frozen config sha256 | `902bf0c1757a801fc8c3b56cb8609637364d622791b73acdff062e914503d136` | runs/production_launch_manifest.json |
| Frozen at | 2026-09-25T01:23:36Z | runs/production_launch_manifest.json |
| Code at freeze | git a4444dbde054 (branch feature/rhan-next) | production_launch_manifest.json |
| Trainer commit (per-phase manifests & checkpoints) | `cf6ce8a` | runs/foundation_*/manifest.json |
| Dataset | ImageNet-100, HF `clane9/imagenet-100` @ pinned `0519dc2f402a3a18c6e57f7913db059215eee25b`; 126,689 train / 5,000 val images; 100 classes/split; aggregate listing sha256 `0b06779f7b89…` | production_launch_manifest.json dataset.fingerprint |
| Dataset root at freeze | `/home/ferrarikazu/Adversarial Cognitive Model/data/imagenet100` (absolute; part of the hashed config) | production_launch_manifest.json |
| Training seed | 41 | protocol.seed_policy |
| Eval seeds | 41–48 (8), subsets drawn per seed via `torch.randperm(seed)` | protocol.seed_policy |
| Optimizer | SGD(momentum), base_lr 0.003, momentum 0.9, weight_decay 1e-4, CosineAnnealingLR(T_max=60) | protocol.optimizer |
| Optimizer groups | per-phase (see §10) | runs/foundation_*/manifest.json |
| Recurrence | T=4 glimpses, 2 within-glimpse iters, fovea 56, image 96, K=4 candidates | protocol.recurrence |
| Gaze | steps 1–4 `PLACEHOLDER_FIXED_GAZE`; steps 5–6 `AIS_V2` (Agent F AISv2GazePolicy); backbone_only `single_center_fixation` | protocol.gaze + roadmap |
| Uncertainty design | carrier = Agent D EvidentialHead Dirichlet (the ONE U_t); S_t = null; L_stab diagnostic-only (not a training objective) | protocol.uncertainty |
| batch / epochs / AMP / rolling | 48 / 60 per phase / true / every epoch | protocol |
| Eval | 8 seeds × 300 samples, ε ∈ {0, 0.031, 0.062, 0.094} (pixel-norm space), PGD-10, Agent I `run_clean_and_robust` (summary asserted against per-seed CSV before write) | protocol.eval + eval_provenance.json |
| Launch | `J1_DATA_ROOT=/home/ferrarikazu/Adversarial Cognitive Model/data/imagenet100 ./cloud_setup/run_j1_local.sh` | production_launch_manifest.json |
| Destinations | HF `FerrariKazu/rhan-nxa-checkpoints` (best), `FerrariKazu/rhan-nxa-checkpoints-rolling` (rolling+roadmap), both private datasets | destinations |

## 2. Phase ladder status (the roadmap is the single source of truth)

| Phase | Depends on | Status | started_utc | finished_utc | best_val_acc |
|---|---|---|---|---|---|
| `backbone_only` | — | done | 2026-09-25T04:38:18Z | 2026-09-25T05:36:10Z | 0.3130 |
| `recurrence_only` | backbone_only | done | 2026-09-25T09:47:58Z | 2026-09-25T11:27:48Z | 0.3668 |
| `belief_no_f` | recurrence_only | done | 2026-09-25T15:20:00Z | 2026-09-25T16:23:04Z | 0.2684 |
| `belief_with_f` | belief_no_f | done | 2026-09-25T22:10:21Z | 2026-09-26T00:40:53Z | 0.3316 |
| `ais_v2_swap` | belief_with_f | done | 2026-09-26T00:40:54Z | 2026-09-26T08:25:12Z | 0.3334 |
| `gen1_core` | ais_v2_swap | done | 2026-09-26T08:25:13Z | 2026-09-26T13:18:09Z | 0.3334 |

All six phases `done`; Phase-11 completion verification: **exit 0** (report/run_completion_report.json — parity, manifests, eval artifacts, frozen config hash, HF sync all pass; `problems: []`).

## 3. Checkpoint identifiers

Naming: `foundation_<phase>_{best,rolling}.pth`. Local: `checkpoints/`. HF best repo: `FerrariKazu/rhan-nxa-checkpoints`; HF rolling repo: `FerrariKazu/rhan-nxa-checkpoints-rolling`. Best/rolling parity verified per phase (`verify_best_rolling_parity` = ok, code_commit `cf6ce8a`).

| Phase | eval-time ckpt sha256 (best file, from eval_provenance.json) | best saved_at_utc |
|---|---|---|
| `backbone_only` | `673f94840741cec28fe40770738c5abe615092391b0b4a53830c318f24a8e431` | see §2 |
| `recurrence_only` | `5184c54f27999f34042f43751d4b89c6e4fbafea4cb7d09f3f777031e8d61332` | see §2 |
| `belief_no_f` | `d97c99e9cb401cc3084783b4064fdfeda852585fbc377a0f8d71d81414727a95` | see §2 |
| `belief_with_f` | `c04fe1b59e50bc5bc89dbf28171b1b2e56647efc80fbc8e633e3038f4f7e7cad` | see §2 |
| `ais_v2_swap` | `ee4c970e9664266484e003f81316a24c9696f7a190407cf83790298bfdd015a8` | see §2 |
| `gen1_core` | `e92e3c3708c4f2a081434c849a2efd199216a104e6614caddd4a6907ad1402be` | see §2 |

Checkpoints store: epoch, model, optimizer, scheduler, code_commit, saved_at_utc, kind, phase, gaze_scheme. They contain **no training history and no metric telemetry**.

## 4. Compactness / architecture cost per phase

| Phase | params_total | params_trainable | est MACs/image (96px) |
|---|---|---|---|
| `backbone_only` | 23,341,156 | 23,341,156 | 24,884,736 |
| `recurrence_only` | 23,341,156 | 23,341,156 | 113,579,520 |
| `belief_no_f` | 23,485,516 | 23,485,516 | 114,095,136 |
| `belief_with_f` | 27,730,509 | 27,730,509 | 126,801,984 |
| `ais_v2_swap` | 27,730,510 | 27,730,510 | 149,880,896 |
| `gen1_core` | 27,730,510 | 27,730,510 | 149,880,896 |

## 5. Training convergence (trainer stdout, preserved in report/supervisor.log)

Loss = the trainer's reported training loss; val_acc/best = the trainer's per-epoch validation metric (fraction).

| Phase | ep1 loss/val | ep30 loss/val | ep59 loss/val | ep60 loss/val | best_val_acc (epochs completed) |
|---|---|---|---|---|---|
| `backbone_only` | 4.1757 / 0.1080 | 3.0187 / 0.2764 | 2.8461 / 0.3104 | 2.8446 / 0.3116 | 0.3130 (60/60) |
| `recurrence_only` | 4.1340 / 0.1072 | 2.7505 / 0.3218 | 2.4845 / 0.3668 | 2.4852 / 0.3666 | 0.3668 (60/60) |
| `belief_no_f` | 4.3029 / 0.0820 | 3.3185 / 0.2340 | 3.1136 / 0.2682 | 3.1187 / 0.2684 | 0.2684 (60/60) |
| `belief_with_f` | 4.2589 / 0.0908 | 2.9760 / 0.2994 | 2.7612 / 0.3292 | 2.7570 / 0.3298 | 0.3316 (60/60) |
| `ais_v2_swap` | 4.1928 / 0.1138 | 2.9727 / 0.3000 | 2.7630 / 0.3334 | 2.7579 / 0.3326 | 0.3334 (60/60) |
| `gen1_core` | 4.1928 / 0.1138 | 2.9727 / 0.3000 | 2.7630 / 0.3334 | 2.7579 / 0.3326 | 0.3334 (60/60) |

Epoch-line inventory in supervisor.log: backbone_only 60, recurrence_only 59, belief_no_f 59, belief_with_f 60, ais_v2_swap 60, gen1_core 60 (no duplicate epoch lines; the 59s are phases where the final line is epoch 59).

## 6. Compute / runtime

| Phase | started → finished (roadmap, final pass) | wall time |
|---|---|---|
| `backbone_only` | 2026-09-25T04:38:18Z → 2026-09-25T05:36:10Z | 0h57m52s |
| `recurrence_only` | 2026-09-25T09:47:58Z → 2026-09-25T11:27:48Z | 1h39m50s |
| `belief_no_f` | 2026-09-25T15:20:00Z → 2026-09-25T16:23:04Z | 1h03m04s |
| `belief_with_f` | 2026-09-25T22:10:21Z → 2026-09-26T00:40:53Z | 2h30m32s |
| `ais_v2_swap` | 2026-09-26T00:40:54Z → 2026-09-26T08:25:12Z | 7h44m18s |
| `gen1_core` | 2026-09-26T08:25:13Z → 2026-09-26T13:18:09Z | 4h52m56s |

Environment (frozen): RTX 4060 8 GB, torch 2.9.1+cu128, CUDA runtime 12.8, cuDNN 91002, Python 3.10.12, host `FerrariKazu` (WSL2), driver 616.92. Supervisor: 2026-09-25T01:17:47Z (first pass header in log) → 2026-09-26T13:18:18Z (verification) ≈ 36 h; 32 `=== supervisor pass` headers across invocations; crash-restart counter reached 13/60 (transient CUDA errors; GPU verified healthy); phases are single-run dispatch, so clean rc=0 passes between phases are normal. Roadmap timestamps cover each phase's final successful pass; wall time spent in crashed pre-restart attempts is not attributable per phase from these artifacts.

## 7. Robustness evaluation protocol (identical for all six phases)

From eval_provenance.json (all phases): attack `generic_pgd`, PGD-10, eps_space `norm`, ε ∈ {0, 0.031, 0.062, 0.094}, 300 samples/seed, seeds 41–48, self_test false. **CSV semantics (from evaluation/clean_and_robust.py):** each per-seed CSV row is one eval batch-chunk accuracy (loader yields 48-image batches; 6×48 + 1×12 = 300 per seed; `take = min(batch, remaining)`), so 56 rows per (phase, ε): 8 seeds × 7 chunks. The published `summary_table.csv` aggregates those 56 chunk rows (mean, SD ddof=1) — reproduced exactly in this document's check. (PGD step size α uses the harness default; not recorded in provenance.)

## 8. Results — accuracy vs ε

### 8.1 As published (aggregation unit = eval batch-chunk, 56 rows, SD ddof=1) — verbatim from summary_table.csv (2 dp; exact values in the CSVs)

| Phase | ε=0 clean | ε=0.031 | ε=0.062 | ε=0.094 |
|---|---|---|---|---|
| `backbone_only` | 31.73 ± 6.96 | 3.05 ± 3.37 | 0.41 ± 1.01 | 0.04 ± 0.28 |
| `recurrence_only` | 36.64 ± 7.09 | 1.93 ± 2.54 | 0.15 ± 0.54 | 0.00 ± 0.00 |
| `belief_no_f` | 26.60 ± 7.32 | 3.76 ± 2.90 | 0.82 ± 1.47 | 0.11 ± 0.47 |
| `belief_with_f` | 33.15 ± 7.88 | 3.31 ± 3.20 | 0.26 ± 0.69 | 0.11 ± 0.47 |
| `ais_v2_swap` | 33.33 ± 6.54 | 3.09 ± 2.92 | 0.56 ± 1.09 | 0.07 ± 0.39 |
| `gen1_core` | 33.33 ± 6.54 | 3.09 ± 2.92 | 0.56 ± 1.09 | 0.07 ± 0.39 |

### 8.2 Seed-level (exact 300-sample accuracy per seed; mean ± SD over the 8 eval seeds, ddof=1)

| Phase | ε=0 clean | ε=0.031 | ε=0.062 | ε=0.094 |
|---|---|---|---|---|
| `backbone_only` | 31.92 ± 2.69 | 2.92 ± 0.85 | 0.46 ± 0.50 | 0.04 ± 0.12 |
| `recurrence_only` | 36.29 ± 2.01 | 1.92 ± 0.56 | 0.17 ± 0.25 | 0.00 ± 0.00 |
| `belief_no_f` | 26.42 ± 2.23 | 3.83 ± 0.96 | 0.92 ± 0.43 | 0.12 ± 0.17 |
| `belief_with_f` | 32.75 ± 2.22 | 3.21 ± 1.22 | 0.29 ± 0.28 | 0.12 ± 0.17 |
| `ais_v2_swap` | 33.33 ± 2.24 | 3.21 ± 0.89 | 0.62 ± 0.33 | 0.08 ± 0.24 |
| `gen1_core` | 33.33 ± 2.24 | 3.21 ± 0.89 | 0.62 ± 0.33 | 0.08 ± 0.24 |

Seed-level vs chunk-level means differ by construction (chunk weighting); per-seed-value difference ≤ 2.39 pp (chunk-macro vs sample-weighted). The two conventions agree on every conclusion below; the single sign-level discrepancy between conventions is a null test, noted in §9.

### 8.3 Per-seed clean accuracy (ε=0, exact 300-sample convention, %)

Exact 300-sample accuracies are multiples of 1/3 %; shown at 2 dp.

| seed | `backbone_only` | `recurrence_only` | `belief_no_f` | `belief_with_f` | `ais_v2_swap` | `gen1_core` |
|---|---|---|---|---|---|---|
| 41 | 31.00 | 36.67 | 28.33 | 33.67 | 31.33 | 31.33 |
| 42 | 28.00 | 37.00 | 28.67 | 31.00 | 33.33 | 33.33 |
| 43 | 29.33 | 35.67 | 23.00 | 32.67 | 31.33 | 31.33 |
| 44 | 31.33 | 35.67 | 26.33 | 35.67 | 33.67 | 33.67 |
| 45 | 31.33 | 40.67 | 27.00 | 33.00 | 31.67 | 31.67 |
| 46 | 34.67 | 34.33 | 23.00 | 28.67 | 32.00 | 32.00 |
| 47 | 33.67 | 34.33 | 27.67 | 35.00 | 36.33 | 36.33 |
| 48 | 36.00 | 36.00 | 27.33 | 32.33 | 37.00 | 37.00 |

### 8.4 Per-seed robust accuracy, PGD-10 (exact 300-sample convention, %)

Exact 300-sample accuracies are multiples of 1/3 %; shown at 2 dp. Zeros are exact.

**ε = 0.031**

| seed | `backbone_only` | `recurrence_only` | `belief_no_f` | `belief_with_f` | `ais_v2_swap` | `gen1_core` |
|---|---|---|---|---|---|---|
| 41 | 2.33 | 1.33 | 5.00 | 3.33 | 4.00 | 4.00 |
| 42 | 3.33 | 2.33 | 2.67 | 3.00 | 3.33 | 3.33 |
| 43 | 2.67 | 2.00 | 3.00 | 3.00 | 3.00 | 3.00 |
| 44 | 4.00 | 1.67 | 3.67 | 1.33 | 3.00 | 3.00 |
| 45 | 3.33 | 2.00 | 3.33 | 3.67 | 3.33 | 3.33 |
| 46 | 1.33 | 1.33 | 3.33 | 2.00 | 1.67 | 1.67 |
| 47 | 2.67 | 1.67 | 4.33 | 4.00 | 2.67 | 2.67 |
| 48 | 3.67 | 3.00 | 5.33 | 5.33 | 4.67 | 4.67 |

**ε = 0.062**

| seed | `backbone_only` | `recurrence_only` | `belief_no_f` | `belief_with_f` | `ais_v2_swap` | `gen1_core` |
|---|---|---|---|---|---|---|
| 41 | 0.67 | 0.33 | 1.33 | 0.33 | 0.67 | 0.67 |
| 42 | 0.00 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| 43 | 0.33 | 0.00 | 1.00 | 0.67 | 0.33 | 0.33 |
| 44 | 1.33 | 0.00 | 0.67 | 0.00 | 0.67 | 0.67 |
| 45 | 0.00 | 0.00 | 0.67 | 0.00 | 1.00 | 1.00 |
| 46 | 0.33 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 |
| 47 | 1.00 | 0.00 | 1.00 | 0.33 | 1.00 | 1.00 |
| 48 | 0.00 | 0.33 | 1.67 | 0.33 | 0.67 | 0.67 |

**ε = 0.094**

| seed | `backbone_only` | `recurrence_only` | `belief_no_f` | `belief_with_f` | `ais_v2_swap` | `gen1_core` |
|---|---|---|---|---|---|---|
| 41 | 0.00 | 0.00 | 0.33 | 0.33 | 0.00 | 0.00 |
| 42 | 0.00 | 0.00 | 0.33 | 0.33 | 0.00 | 0.00 |
| 43 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 |
| 44 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.67 |
| 45 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 46 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 47 | 0.33 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 |
| 48 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## 9. Statistical comparisons between consecutive phases

Method: values are the exact 300-sample accuracy per seed (multiples of 1/3 %, reconstructed from the chunk counts in the per-seed CSVs; reconstruction round-trip verified for every row), paired across the 8 shared eval seeds. Paired t-test + two-sided Wilcoxon signed-rank (Pratt, zeros retained); SD of Δ with ddof=1. n=8 is small — p-values are indicative. No multiplicity correction except where stated. `ais_v2_swap → gen1_core` is **not computable**: the two phases' eval outputs are identical bit-for-bit (§11). The chunk-macro convention (§8.1) reaches the same significance decision (p<0.05 vs p≥0.05) for every computable test except `belief_with_f → ais_v2_swap` at ε=0.031 (macro Δ −0.223 pp vs seed-level +0.000 pp; both null).

| Pair | ε | Δmean (pp) | SD of Δ | paired t | p (t) | p (Wilcoxon) |
|---|---|---|---|---|---|---|
| `backbone_only` → `recurrence_only` | 0.0 | +4.375 | 3.901 | +3.172 | 0.0157 | 0.0312 |
| `backbone_only` → `recurrence_only` | 0.031 | -1.000 | 0.667 | -4.243 | 0.0038 | 0.0156 |
| `backbone_only` → `recurrence_only` | 0.062 | -0.292 | 0.653 | -1.263 | 0.2470 | 0.2812 |
| `backbone_only` → `recurrence_only` | 0.094 | -0.042 | 0.118 | -1.000 | 0.3506 | 1.0000 |
| `recurrence_only` → `belief_no_f` | 0.0 | -9.875 | 2.423 | -11.527 | <0.0001 | 0.0078 |
| `recurrence_only` → `belief_no_f` | 0.031 | +1.917 | 1.035 | +5.237 | 0.0012 | 0.0078 |
| `recurrence_only` → `belief_no_f` | 0.062 | +0.750 | 0.427 | +4.965 | 0.0016 | 0.0156 |
| `recurrence_only` → `belief_no_f` | 0.094 | +0.125 | 0.173 | +2.049 | 0.0796 | 0.2500 |
| `belief_no_f` → `belief_with_f` | 0.0 | +6.333 | 2.404 | +7.452 | 0.0001 | 0.0078 |
| `belief_no_f` → `belief_with_f` | 0.031 | -0.625 | 1.015 | -1.742 | 0.1250 | 0.3125 |
| `belief_no_f` → `belief_with_f` | 0.062 | -0.625 | 0.415 | -4.255 | 0.0038 | 0.0156 |
| `belief_no_f` → `belief_with_f` | 0.094 | +0.000 | 0.178 | +0.000 | 1.0000 | 1.0000 |
| `belief_with_f` → `ais_v2_swap` | 0.0 | +0.583 | 2.683 | +0.615 | 0.5580 | 0.4375 |
| `belief_with_f` → `ais_v2_swap` | 0.031 | +0.000 | 0.909 | +0.000 | 1.0000 | 1.0000 |
| `belief_with_f` → `ais_v2_swap` | 0.062 | +0.333 | 0.436 | +2.160 | 0.0676 | 0.1250 |
| `belief_with_f` → `ais_v2_swap` | 0.094 | -0.042 | 0.330 | -0.357 | 0.7318 | 0.6250 |
| `ais_v2_swap` → `gen1_core` | 0.0 | +0.000 | 0.000 | — | — | — (identical evals) |
| `ais_v2_swap` → `gen1_core` | 0.031 | +0.000 | 0.000 | — | — | — (identical evals) |
| `ais_v2_swap` → `gen1_core` | 0.062 | +0.000 | 0.000 | — | — | — (identical evals) |
| `ais_v2_swap` → `gen1_core` | 0.094 | +0.000 | 0.000 | — | — | — (identical evals) |

Holm step-down across the four computable clean-accuracy (ε=0) consecutive tests (family defined here, exploratory):

| Comparison | raw p | Holm-adjusted p |
|---|---|---|
| `recurrence_only` → `belief_no_f` | <0.0001 | <0.0001 |
| `belief_no_f` → `belief_with_f` | 0.0001 | 0.0004 |
| `backbone_only` → `recurrence_only` | 0.0157 | 0.0313 |
| `belief_with_f` → `ais_v2_swap` | 0.5580 | 0.5580 |

Omnibus Friedman test across all six phases (n=8 seeds, k=6 → df=5):

| ε | χ²_F(5) | p |
|---|---|---|
| 0.0 | 24.019 | 0.0002 |
| 0.031 | 20.508 | 0.0010 |
| 0.062 | 16.972 | 0.0046 |
| 0.094 | 6.081 | 0.2984 |

## 10. Metrics that DIFFER between phase implementations (from the artifacts)

| Dimension | backbone_only | recurrence_only | belief_no_f | belief_with_f | ais_v2_swap | gen1_core |
|---|---|---|---|---|---|---|
| gaze_scheme | single_center_fixation | PLACEHOLDER_FIXED_GAZE | PLACEHOLDER_FIXED_GAZE | PLACEHOLDER_FIXED_GAZE | AIS_V2 | AIS_V2 |
| part2_steps (manifest) | 1-4 | 1-4 | 1-4 | 1-4 | 5-6 | 5-6 |
| ais_v2 flag (manifest) | false | false | false | false | true | true |
| optimizer groups | backbone, classifier | + (per manifest) | + evidential_head | + predictor, update_net, precision | + gaze_policy | + gaze_policy |
| params_total | 23,341,156 | 23,341,156 | 23,485,516 | 27,730,509 | 27,730,510 | 27,730,510 |
| est MACs/image | 24,884,736 | 113,579,520 | 114,095,136 | 126,801,984 | 149,880,896 | 149,880,896 |

Optimizer-group cell values are read from runs/foundation_<phase>/manifest.json (`optimizer_config.groups`); each phase's manifest lists its cumulative groups (recurrence_only's manifest groups are backbone, classifier — same as backbone_only; the recurrence loop adds compute (MACs) but the manifest records no new named group). Eval protocol, seeds, ε grid, PGD steps, sample counts and aggregation are identical across all six phases (eval_provenance.json), so the phase mechanism is the only protocol difference.

## 11. Artifact identity: `gen1_core` vs `ais_v2_swap`

Facts, all verified directly:

- The two `*_best.pth` files have different file sha256 (`ee4c970e…` vs `e92e3c37…` — metadata/timestamps differ: best saved_at 2026-09-26T07:14:25Z vs 2026-09-26T12:32:23Z).
- Their **model state is bitwise identical** (all 206 tensors equal; state-dict payload hash `4814d7c70bfce254` for both).
- Their eval outputs are identical bit-for-bit (per-seed and summary CSVs, all four ε).
- Their epoch-1/2 training-log lines are identical (loss=4.1928 val_acc=0.1138; loss=3.9153 val_acc=0.1424).
- Both phases ran full 60-epoch windows per the supervisor log line inventory, and both roadmap entries record best_val_acc 0.3334.

The artifacts establish that the `gen1_core` best checkpoint **is** the `ais_v2_swap` best weights (the new-best never surpassed the inherited 0.3334 at any point its trajectory is visible in the artifacts). The precise mechanism (exact tie-handling at the per-epoch new-best decision, or an earlier resume identity) is **not determinable from these artifacts alone**; the checkpoints and log lines are the complete evidence. Any gen1_core-specific performance claim is therefore unsupported by this run.

## 12. Negative / null findings (artifact-supported)

1. **No phase achieves meaningful adversarial robustness under the frozen protocol.** PGD-10 accuracy at ε=0.094 is 0.00–0.12% across all phases (chunk-level published values; recurrence_only is exactly 0.00 ± 0.00). At ε=0.062 the range is 0.15–0.82%.
2. **The AIS-v2 swap (steps 5–6) shows no measurable clean or robust benefit over `belief_with_f` in this run.** Clean Δ +0.583 pp (p_t=0.558); all robust |Δ| ≤ 0.333 pp with p ≥ 0.068 (exploratory, n=8).
3. **`belief_no_f` regressed clean accuracy** vs `recurrence_only` by −9.875 pp (paired t p<0.0001, Wilcoxon p=0.0078, Holm-adjusted p<0.0001 within this family) — the largest effect in the ladder, and it is negative. `belief_with_f` recovered +6.333 pp over `belief_no_f` (Holm-adj p=0.0004) but remained below `recurrence_only`.
4. **`recurrence_only` traded clean accuracy for nothing measurable at this PGD budget**: clean +4.375 pp over `backbone_only` (Holm-adj p=0.0313) but robust accuracy *decreased* at ε=0.031 (−1.000 pp, p_t=0.0038) and was already near-chance elsewhere.
5. **No gen1_core-specific effect is measurable** (bit-identical artifacts, §11).

## 13. ABSENT metric families (requested but not present in any Gen-1 artifact)

The following were requested and are **not recorded anywhere** in the Gen-1 artifacts (trainer stdout logs only `loss/val_acc/best`; checkpoints store no history; the eval harness records accuracy only):

- **d′ / perceptual separability measures** — no artifact computes d′ or any signal-detection/perceptual metric.
- **Uncertainty metrics** — no per-sample or aggregate readout of the Dirichlet evidential head (U_t, α_k, S_t), precision head, or L_stab exists in the artifacts. The design roles are recorded (§1) but no values.
- **Prediction-error metrics** — no artifact logs predictor/UpdateNet prediction-error values or belief-space errors.
- **Gaze / AIS-v2 behavior metrics** — no artifact logs gaze positions, candidate selection (K=4), policy entropy, info-gain, or any AISv2GazePolicy behavioral telemetry.
- Per-class / confusion results, calibration, ECE — not present.
- EMA / weight-norm trajectories — not present (no EMA in checkpoints).

Obtaining these requires a re-instrumented eval/instrumentation pass on the checkpoints (they are not recoverable from the existing artifacts).

## 14. Operational caveats (facts about the record, not the science)

- The 2026-09-26 'VERIFICATION FAILED (1 problems)' was a verifier bug: it hashed the config with the default relative `data_root` while the freeze hashed the absolute frozen dataset root. Fixed 2026-09-29 (commit `0762a5e`); verification now exits 0. The training run itself was never affected.
- The frozen launch manifest was uploaded to the rolling HF repo post-hoc (2026-09-29) together with per-phase provenance manifests, eval CSVs, result/compactness jsons, the completion report and supervisor.log (27 files). `scripts/freeze_run_manifest.py` now performs that sync at freeze time.
- `git_status` at freeze recorded dirty files (T4x2.ipynb modified; untracked scratch items) — recorded in production_launch_manifest.json `code.dirty_files`.

## 15. Artifact index (everything this document is derived from)

Local (and mirrored to HF as noted):

- `runs/foundation_backbone_only/manifest.json` → HF best repo `runs/foundation_backbone_only/manifest.json`
- `report/foundation_backbone_only_result.json`, `report/foundation_backbone_only_compactness.json` → HF best repo
- `report/foundation_backbone_only_eval/summary_table.csv`, `epsilon_sweep_per_seed.csv`, `eval_provenance.json` (CSVs → HF best repo)
- `runs/foundation_recurrence_only/manifest.json` → HF best repo `runs/foundation_recurrence_only/manifest.json`
- `report/foundation_recurrence_only_result.json`, `report/foundation_recurrence_only_compactness.json` → HF best repo
- `report/foundation_recurrence_only_eval/summary_table.csv`, `epsilon_sweep_per_seed.csv`, `eval_provenance.json` (CSVs → HF best repo)
- `runs/foundation_belief_no_f/manifest.json` → HF best repo `runs/foundation_belief_no_f/manifest.json`
- `report/foundation_belief_no_f_result.json`, `report/foundation_belief_no_f_compactness.json` → HF best repo
- `report/foundation_belief_no_f_eval/summary_table.csv`, `epsilon_sweep_per_seed.csv`, `eval_provenance.json` (CSVs → HF best repo)
- `runs/foundation_belief_with_f/manifest.json` → HF best repo `runs/foundation_belief_with_f/manifest.json`
- `report/foundation_belief_with_f_result.json`, `report/foundation_belief_with_f_compactness.json` → HF best repo
- `report/foundation_belief_with_f_eval/summary_table.csv`, `epsilon_sweep_per_seed.csv`, `eval_provenance.json` (CSVs → HF best repo)
- `runs/foundation_ais_v2_swap/manifest.json` → HF best repo `runs/foundation_ais_v2_swap/manifest.json`
- `report/foundation_ais_v2_swap_result.json`, `report/foundation_ais_v2_swap_compactness.json` → HF best repo
- `report/foundation_ais_v2_swap_eval/summary_table.csv`, `epsilon_sweep_per_seed.csv`, `eval_provenance.json` (CSVs → HF best repo)
- `runs/foundation_gen1_core/manifest.json` → HF best repo `runs/foundation_gen1_core/manifest.json`
- `report/foundation_gen1_core_result.json`, `report/foundation_gen1_core_compactness.json` → HF best repo
- `report/foundation_gen1_core_eval/summary_table.csv`, `epsilon_sweep_per_seed.csv`, `eval_provenance.json` (CSVs → HF best repo)
- `runs/production_launch_manifest.json`, `report/run_completion_report.json`, `report/supervisor.log` → HF rolling repo
- `report/generation1_foundation_roadmap.json` → HF rolling repo (trainer-synced)
- `checkpoints/foundation_<phase>_{best,rolling}.pth` → HF best/rolling repos (trainer-synced)

HF datasets: `FerrariKazu/rhan-nxa-checkpoints` (7 ckpts + 24 per-phase artifacts), `FerrariKazu/rhan-nxa-checkpoints-rolling` (6 rolling ckpts + roadmap + frozen manifest + completion report + supervisor log).

---

Statistics: SciPy 1.15.3 (`scipy.stats.ttest_rel`, `wilcoxon` Pratt, `friedmanchisquare`), ddof=1 SDs. Definitions of every derived quantity are stated inline. No other computation was applied to the artifacts.
