# 7 — Unsupported claims

Each row: claim as stated, where it appears, why it is unsupported, and what evidence would support it.

| # | Claim | Location | Why unsupported | Evidence needed |
|---|---|---|---|---|
| 1 | "80% defect detection accuracy" | `README.md:43`, `docs/09:40` | Deployed model gives 0.850 with lower bound 0.776; reported 0.817 has lower bound 0.738. Neither establishes 0.80 at 95% confidence. | Held-out test evaluation with a pre-registered threshold and interval. |
| 2 | "0.850 accuracy" for the deployed model (this is actually correct) | `README.md:46` | Supported on dev, but dev is leaked into training (`data.py:47+`). | Retrain on train+cal, evaluate on dev once, then test. |
| 3 | "ECE 0.072" | `README.md:45`, `docs/09:42` | Single-look, temperature-scaled, equal-frequency 15 bins, on dev; not the sequential posterior. | Multi-look calibration on a held-out set; report Brier/NLL. |
| 4 | "coverage 0.875 at threshold" | `README.md:45`, `docs/09:42` | Threshold selected on raw scale, applied to tempered dev; raw coverage is 0.767. | Threshold on tempered cal; evaluate on dev/test. |
| 5 | "ledger 0.850 vs pooled 0.567" | `README.md:46`, `docs/09:43` | Confounded by the pair cue; n = 18; deployed model leaks dev; revision p = 0.5. | Larger paired ablation on held-out scenes with pre-registered metric. |
| 6 | "falsification and EIG have the same error 0.192" | `README.md:47`, `docs/09:44` | Underpowered (MDD ≈ 5.7% at N = 120); policies chose different actions; 6 discordant pairs. | N ≥ 400 with pre-registered non-inferiority margin. |
| 7 | "same cost" (3.525 vs 3.175) | `README.md:47` | Cost model is fixed round numbers, not calibrated. | Calibrated cost model. |
| 8 | "posterior is invariant to record order" | `docs/05:79` | False: 0.827–0.987 spread on the same three actions. | Order-invariance test. |
| 9 | "if all records are duplicates, posterior moves by at most one record's worth" | `docs/05:81` | False: harmonic sum H_m; posterior defect 0.9996 at m = 50. | Duplicate saturation test. |
| 10 | "tamper-evident ledger" | `docs/03`, README | Tail truncation and full re-chain forgery pass `verify()`. | Authentication primitive or corrected claim. |
| 11 | "n_eff corrects the posterior for dependence" | `docs/04:92-99` | `n_eff` is computed and reported but never used in inference. | Either a real correction or relabelling. |
| 12 | "dependence constants checked against empirical correlation on cal" | `docs/03:122` | `vif()` and `effective_n()` are defined but never called; no `--dependence-report` flag. | Implement the check or delete the claim. |
| 13 | "real visual encoder ready to connect" (implied) | README, `perception/` | DINOv2 path returns hand-crafted features; CLS/patch tokens unused. | Integration test on real/rendered images with a Gate A number. |
| 14 | "9-gate / 5-gate experimental programme" | `docs/09` | Only 3 result files exist; no stage JSONs, no manifest, no checksums. | Reproducible run with artifacts. |
| 15 | "revision accuracy 0.111 (2/18)" | `README.md:46` | Recovery rate, not revision quality; p = 0.5 vs no_dep/no_pair which also score 0.111. | Larger paired sample and a proper revision metric. |
| 16 | "reliability AUC 0.642" | audit rerun | Computed on dev, single seed; not pre-registered. | Test-set AUC with interval. |
| 17 | "no overclaiming of Gate E" | `README.md:47` | True and worth noting, but the write-up still says "same error" and "same cost" without power. | Restate as "no detectable difference at N = 120". |
| 18 | Novelty claims | none in repo (good) | The repo makes no novelty claims; keep it that way. | A comparison against published baselines before any novelty statement. |

## Claims that are supported

- The simulator is deterministic and seed-disjoint (`scene.py:112-120`).
- Records are immutable and ids are deterministic (`types.py:98-131`, `util.py:19-50`).
- Canonical serialisation is deterministic for finite values (`util.py:9-42`).
- The control loop runs end to end (`evaluation/runner.py:22-118`).
- Temperature scaling improves single-look ECE on dev (0.104 → 0.072, equal-frequency).
- The ledger arm has better proper scores than `no_dep` (Brier 0.265 vs 0.277, NLL 0.647 vs 0.928).
- Gate E is reported as inconclusive rather than as a falsification win.
- The DINOv2 checkpoint exists at `checkpoints/dinov2_vits14_pretrain.pth`.

## Rule of thumb for the next write-up

Every number in the README must be traceable to a file in `results/` that contains the seed, the
code hash, the model hash, and the split. If a number cannot be traced, it must be labelled
"exploratory" or removed. The current README mixes exploratory, development-set and calibration
numbers without labels.
