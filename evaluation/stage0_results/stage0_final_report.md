# Stage 0 Closeout Report
**Protocol:** consolidated-flagship-112/1  
**Prereg:** prereg/stage0_decision_v1.json  
**Generated:** 2026-11-01T07:35:59Z  
**Harness SHA256:** `6a35fcbb943128c410c7ed802568b0700c5b314a7517a74eab4c60c738d67482`

---

## Verdict

> **INFERIOR**

min(rec_112) = 30.18% < min(ff_112) = 35.44%.

---

## Per-arm Results (5,000 images, ε = 5.417/255)

### Flagship — recurrence_112

| arm | clean (5k) | worst-case | pgd50 | apgd-ce | apgd-dlr | sha8 | gate |
|-----|-----------|------------|-------|---------|----------|------|------|
| recurrence_112_seed42 | 76.12% | 37.72% | 39.40% | 38.98% | 40.14% | eb40bf84 | PASS |
| recurrence_112_seed17 | 67.06% | 30.18% | 31.82% | 31.50% | 31.66% | 636e4b1c | PASS |

### Baseline — ff_112

| arm | clean (5k) | worst-case | pgd50 | apgd-ce | apgd-dlr | sha8 | gate |
|-----|-----------|------------|-------|---------|----------|------|------|
| ff_112_seed17 | 74.30% | 36.04% | 38.18% | 37.66% | 38.10% | d00fb750 | PASS |
| ff_112_seed42 | 74.58% | 35.88% | 37.48% | 36.86% | 38.34% | 2920df54 | PASS |
| ff_112_seed99 | 74.60% | 35.44% | 37.00% | 36.52% | 37.40% | abde98a0 | PASS |

---

## Statistical Test (Pooled McNemar, pre-registered)

| Metric | Value |
|--------|-------|
| b (rec✓, ff✗) | 401.0 |
| c (rec✗, ff✓) | 602.0 |
| Discordant pairs | 1003.0 |
| Δ worst-case (pp) | -2.01 |
| p (exact two-sided) | 2.3613e-10 |
| 95% CI | [-2.63, -1.39] pp |
| Threshold | p < 0.01, CI_low > 0 |
| Pass | NO |

---

## Descriptive Diagnostics

### T4a — Crop check
ImageNet-100 fingerprint verified via `data/imagenet100/fingerprint.json`.  
All val images confirmed: 5000 samples, hash chain intact.

### T4b — T-sweep (T=1..4)
Evaluations performed at canonical T=4 per protocol.  
T-sweep metrics available in future amendment if required.

### T4c — Learning curve slope (epochs 10–15)

| Phase | slope_clean (pp/ep) | slope_robust (pp/ep) |
|-------|---------------------|----------------------|
| recurrence_112 | 0.310 pp/epoch | 0.156 pp/epoch |
| ff_112 | N/A | N/A |

### T4d — Cost table (RTX 4060, measured)

| Metric | Value |
|--------|-------|
| recurrence_112 step wall (T=4) | 1630.6 ms |
| ff_112 single trunk pass | 25.15 ms |
| Trunk ratio (T=4 vs T=1) | 4.76× |
| img/sec | 31.37 |

---

## Checkpoint Integrity

| arm | sha8 | epoch | monitor_clean | gate_gap_pp |
|-----|------|-------|---------------|-------------|
| recurrence_112_seed42 | eb40bf84 | 15 | 76.10% | 0.02 |
| ff_112_seed17 | d00fb750 | 15 | 74.30% | 0.00 |
| ff_112_seed42 | 2920df54 | 15 | 74.60% | 0.02 |
| ff_112_seed99 | abde98a0 | 15 | 74.58% | 0.02 |
| recurrence_112_seed17 | 636e4b1c | 15 | 67.68% | 0.62 |

---

## T6 Housekeeping

- All per-sample arrays saved to `evaluation/stage0_results/<arm_id>_per_sample.npz`
- Harness run log: `evaluation/stage0_results/harness_run.log`
- Verdict JSON: `evaluation/stage0_results/verdict.json`
- Pre-registration: `prereg/stage0_decision_v1.json`
- Intake manifest: `checkpoints/intake_manifest.json`

---

## T7 Hypothesis Status

| Hypothesis | Verdict |
|------------|---------|
| H1: recurrence_112 worst-case > ff_112 worst-case (all seeds) | NOT ESTABLISHED |
| H2: Statistically significant advantage (McNemar p<0.01, CI>0) | NOT ESTABLISHED |
| Overall Stage 0 | **INFERIOR** |

*Numbers sourced from harness JSON only. No manual transcription.*
