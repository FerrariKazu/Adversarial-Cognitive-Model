# Gate-2 Verdict: Selective Classification, Calibration, and Learn-Then-Test Risk Control

Status: **OFFICIAL VERDICT, 18 Nov 2026**
Protocol: `rhan-im-gate2/1.2`
Rules SHA-256: `3a2632ca60f02972dc335ff38a47332eb727c1f00e591611b483e236f57bb011`
Aggregator SHA-256: `17569c435411ee1e484a383baae4bffdf8dfd9386cb282be4b328f7bf267dc14`
Overall Verdict: **FAIL**

---

## 1. Executive Summary

Under the pre-registered `rhan-im-gate2/1.2` protocol, the three independent measurement runs on the frozen foundation trunk (`foundation_ff_112_seed17_final_d00fb750.pth`) across seeds 17, 42, and 99 have executed to completion and were evaluated by the locked aggregator (`scripts/rhan_im/gate2_aggregate.py`).

| Test / Criterion | Rule | Seed 17 | Seed 42 | Seed 99 | 2-of-3 Verdict | Overall |
|---|---|---|---|---|---|---|
| **E2: Posterior Calibration** | ECE-15 at K=8 $\le 0.03$ | 0.2897 | 0.2892 | 0.2835 | 0 / 3 PASS | **FAIL** |
| **E5: LTT Risk Validity** | Resample violation rate $\le 0.10$ | 0.002 | 0.000 | 0.002 | 3 / 3 PASS | **PASS** |
| **E5: LTT Stopping Efficiency** | Mean looks saved $\ge 25\%$ ($\ge 2.0$ looks) | 0.538 (6.72%) | 0.536 (6.70%) | 0.513 (6.42%) | 0 / 3 PASS | **FAIL** |
| **Gate 2 Aggregate Status** | E2 PASS AND E5 PASS | FAIL | FAIL | FAIL | 0 / 3 PASS | **FAIL** |

---

## 2. Protocol Integrity and Code Lock Verification

All measurement code, upstream libraries, and configuration files were strictly locked in `prereg/gate2_code_lock_v1_2.json` and validated by `scripts/rhan_im/gate2_aggregate.py`.

- **Protocol ID:** `rhan-im-gate2/1.2` (unanimous across runs)
- **Code Hashes:** All 14 target files matched the locked hashes exactly with zero deviations.
- **Data Partitions:**
  - Class split: `prereg/class_split.json` (SHA-256 `c2e05641...`)
  - 3-Way Val split: `prereg/gate2_val_split.json` (SHA-256 `b8c4543f...`)
    - Fit pool (part 0): 850 null scenes (no images reused)
    - Calibration pool (part 1): 425 target + 425 null scenes
    - Test pool (part 2): 400 target + 400 null scenes
  - Train holdout: `prereg/gate2_train_holdout.json` (SHA-256 `9a66bc48...`)
    - 800 target + 1600 null holdout images for sensor $d'$ measurement
    - 124,289 images for linear evidence head training (600 steps)
- **Provenance Sidecars:** Every run's provenance sidecar (`gate2_seed*.provenance.json`) verified with valid link SHA-256 hashes.

---

## 3. Detailed Results by Experiment

### 3.1 Upstream Sensor Fitting

The sensor operating point was fixed at size 4 units, opacity 1.0.

| Metric | Seed 17 | Seed 42 | Seed 99 | Mean |
|---|---|---|---|---|
| $d_{\text{gist}}$ | 2.161 | 2.185 | 2.229 | 2.192 |
| $\text{VIF}_{\text{tok}}$ (null) | 11.60 | 11.58 | 11.67 | 11.62 |
| $\text{VIF}_{\text{gist}}$ (null) | 3.33 | 3.33 | 3.29 | 3.32 |
| $\kappa$ (99.9th percentile) | 1.864 | 1.859 | 1.860 | 1.861 |
| $\rho_{\text{null}}$ (cross-window correlation) | 0.407 | 0.417 | 0.420 | 0.415 |
| Combination mode ($\rho_{\text{null}} > 0.2$) | `best` | `best` | `best` | `best` |

### 3.2 Experiment E2: Posterior Calibration

The test pool comprises 800 independent scenes (400 target, 400 null) with balanced target sizes (2, 4, 6 units).

| Metric | Seed 17 | Seed 42 | Seed 99 |
|---|---|---|---|
| **Primary ECE-15 ($K=8$)** | **0.2897** | **0.2892** | **0.2835** |
| Top-1 Accuracy ($K=8$) | 43.12% | 43.00% | 44.12% |
| Mean Confidence ($K=8$) | 72.10% | 71.92% | 72.48% |
| Negative Log-Likelihood (NLL) | 3.545 | 3.698 | 3.558 |
| Brier Score | 0.868 | 0.877 | 0.863 |
| ECE Size 2 Subgroup | 0.2129 | 0.2053 | 0.2285 |
| ECE Size 4 Subgroup | 0.2599 | 0.2567 | 0.2539 |
| ECE Size 6 Subgroup | 0.4237 | 0.4306 | 0.4054 |

**Diagnosis:** The posterior is systematically overconfident across all look steps and all target sizes (mean confidence ~72% vs accuracy ~43%). As observed in Gate 1, the presence of token-level correlations and non-Gaussian spatial noise violates the independent-Gaussian likelihood assumptions, resulting in severe miscalibration (ECE ~0.285 vs threshold $\le 0.03$).

### 3.3 Experiment E5: Learn-Then-Test Risk Control

LTT was calibrated over 850 calibration scenes using $B=500$ bootstrap resamples ($n_{\text{cal}} = 300$). Primary target: selective committed error $P(\text{decided AND wrong}) \le \alpha = 0.10$ with tolerance $\delta = 0.10$.

| Metric | Seed 17 | Seed 42 | Seed 99 | Gate Criterion | Verdict |
|---|---|---|---|---|---|
| **Test Error Violation Rate** | **0.002** (1/500) | **0.000** (0/500) | **0.002** (1/500) | $\le 0.10$ | **PASS** |
| Mean Test Committed Risk | 0.0610 | 0.0615 | 0.0501 | $\le 0.10$ | PASS |
| Mean Test Coverage | 11.47% | 11.84% | 10.29% | Diagnostic | — |
| Fixed-Horizon Risk | 0.5688 | 0.5700 | 0.5587 | Diagnostic | — |
| **Mean Looks** | 7.462 | 7.464 | 7.487 | — | — |
| **Looks Saved** | **0.538** | **0.536** | **0.513** | — | — |
| **Mean Efficiency** | **6.72%** | **6.70%** | **6.42%** | $\ge 25.0\%$ | **FAIL** |
| Plug-in Violation Rate | 0.142 | 0.054 | 0.080 | Diagnostic | — |

**Diagnosis:**
1. **Validity is fully established:** Learn-Then-Test rigorously guarantees finite-sample risk control. The selective committed error was held strictly under $\alpha = 0.10$ (mean risk $\sim 0.058$) with virtually zero violations across resamples (violation rate $\le 0.002 \ll 0.10$).
2. **Efficiency fails:** Because the posterior is miscalibrated and discriminative signal is weak across random looks, the policy is unable to reach high confidence early. It almost always consumes all 8 looks (mean looks ~7.47, saving only ~0.53 looks or ~6.6%), falling well short of the required 25% look-saving threshold.

---

## 4. Scientific Conclusions

1. **Gate 2 Fails as Scoped:** The exact-Bayes sequential sensor on the frozen foundation trunk fails both calibration (E2 ECE ~0.285) and stopping efficiency (E5 efficiency ~6.6%).
2. **Distribution-Free Guarantee Holds:** Despite severe miscalibration of the underlying posterior probabilities, the Learn-Then-Test calibration layer successfully bounded the committed error at $\le 0.06$ (well below $\alpha = 0.10$). This demonstrates that distribution-free conformal / LTT stopping works even when parametric Bayes assumptions fail, but cannot compensate for weak information gain without sacrificing coverage or efficiency.
3. **Implications for Architecture:** S3 dense-window tuning and Gate 1 v2 previously established that high-resolution native looks do not provide sufficient discriminative advantage over gist looks on this trunk architecture. Gate 2 confirms that this limitation directly impairs sequential multi-look foraging efficiency.

