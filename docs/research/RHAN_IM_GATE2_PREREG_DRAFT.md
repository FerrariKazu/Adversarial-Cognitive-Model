# Gate-2 Pre-Registration Draft: Selective Classification, Calibration, and Learn-Then-Test Risk Control

Status: **DRAFT written on 16 Nov 2026 for review prior to freeze.**
Scope: **INTERNAL QUALIFICATION ONLY.**
Trunk Target: Frozen baseline trunk `foundation_ff_112_seed17_final_d00fb750.pth` (SHA-256 `d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc`).
Freeze Timing: Frozen **before** Stage S3 results are known; independent of the Gate 1 v2 outcome.

---

## 1. Questions and Scope Boundary

### 1.1 Research Questions
1. **E2 (Posterior Calibration):** Given the SDT-adequate sensor validated in Gate 1 v1, is the multi-look posterior well-calibrated (ECE ≤ 0.03, NLL, Brier score) under the measured noise and dependence parameters?
2. **E5 (Selective Classification & Risk-Controlled Stopping):** Does Learn-Then-Test (LTT) risk control bound empirical error below $\alpha = 0.10$ with probability at least $1 - \delta = 0.90$ across deployment resamples, while saving looks compared to fixed-horizon evaluation?

### 1.2 Strict Scope Boundaries
- **No Spatial Resolution Claims:** Gate 1 v1 established that whole-scene gist views provide detectability comparable to native window crops. Therefore, Gate 2 **makes no claims** that adaptive looking beats whole-scene gist, nor that spatial resolution confers an advantage. Gate 2 focuses exclusively on selective classification, Bayesian belief accumulation, posterior calibration, and statistical risk control.
- **Internal Qualification Only:** All validation images were part of ImageNet-100 `val/`. Because foundation self-supervised models saw `train/` and Gate 1 saw `val/`, Gate 2 claims are explicitly designated as internal qualification of the Bayesian stopping engine on natural clutter, not out-of-distribution generalization.
- **Trunk Independence:** Frozen strictly against the validated `ff_112_seed17` trunk. No reliance on Stage S3 fine-tuning.

---

## 2. Frozen Inputs and Data Partitions

### 2.1 Pinned Files and Artifacts

| Item | Pinned Reference | Purpose |
|---|---|---|
| Backbone Trunk | `foundation_ff_112_seed17_final_d00fb750.pth` (SHA-256 `d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc`) | Frozen feature extractor |
| Class Split | `prereg/class_split.json` (SHA-256 `289f2818f0eae4dd7b654627d4bdf197cc11d43923300c52df7b4c0bdecc1241`) | 50 target classes, 50 background classes |
| Validation Split | `prereg/gate2_val_split.json` (SHA-256 `b8c4543f7b7272cae43ebfd1a9ad2095908b65b36c0ae2577c9e8e3b8f92bed0`) | Pinned 3-way split of 5,000 val images |
| Code Files | `rhan_im/stopping.py`, `rhan_im/belief.py`, `rhan_im/window_head.py`, `rhan_im/gate1.py` | Pinned by git commit hash |

### 2.2 Three-Way Disjoint Split (`prereg/gate2_val_split.json`)
Doc 05 line 34 requires that LTT calibration items must be strictly disjoint from everything fitted upstream. Gate 2 cannot reuse Gate 1's 2-way pools file. Instead, the 5,000 ImageNet-100 `val/` images (50 per class × 100 classes) are partitioned into three strictly disjoint sets:
1. **Fit Pool (`fit_pool_part_0_of_3`):** 17 images/class (1,700 images total). Used exclusively to fit the readout head and measure Gate-2-specific noise parameters ($VIF_{\text{null}}$, null mean, null variance).
2. **Calibration Pool (`calibration_pool_part_1_of_3`):** 17 images/class (1,700 images total). Used exclusively to calibrate the LTT risk-controlling thresholds $\hat{\lambda}$ and minimum look floor $k_{\min}$.
3. **Test Pool (`test_pool_part_2_of_3`):** 16 images/class (1,600 images total). Used exclusively to evaluate empirical risk violation rates, coverage, and look-saving efficiency.

---

## 3. Sensor Setup and Upstream Parameter Fitting

### 3.1 Head Refit Procedure
Gate 1 did not export a persistent head checkpoint. Gate 2 fits its own readout head from scratch:
- **Architecture:** `Linear(384, 51)` exporting `score_c = logit_c - logit_bg`.
- **Training Data:** Drawn exclusively from `fit_pool_part_0_of_3`.
- **Optimization:** 600 steps, AdamW, learning rate 2e-3, cosine decay, batch size 48, seed matching the evaluation seed.
- **Export:** Readout projection copied to `EvidenceHead(384, 50)`.

### 3.2 Noise and Dependence Parameters
Gate 2 does **not** copy Gate 1's $VIF_{\text{null}} = 11.605$.
- $VIF_{\text{null}}$ and token standardizers ($\mu_0, \sigma_0$) are measured directly on `fit_pool_part_0_of_3` null scenes.
- If $VIF_{\text{null}} > 1.25$, token scores and detectabilities are scaled by $1 / \sqrt{VIF_{\text{null}}}$ (equicorrelated correction).

---

## 4. Experimental Protocols and Decision Rules

### 4.1 E2: Posterior Calibration
- **Procedure:** Evaluate Bayesian belief accumulation over look sequences $K \in \{1, \dots, 8\}$. Compute predicted posterior confidence vs empirical accuracy across 15 calibration bins on `test_pool_part_2_of_3`.
- **Primary Metrics:**
  - Expected Calibration Error (ECE, 15 equal-frequency bins).
  - Negative Log-Likelihood (NLL).
  - Brier score.
  - Per-target-size ECE (sizes 2, 4, 6 units).
- **Decision Rules:**
  - **PASS:** $\text{ECE} \le 0.03$ with measured sensor parameters.
  - **DIAGNOSE / FAIL:** $\text{ECE} > 0.05 \implies$ likelihood misspecified; stop and diagnose before any deployment claim.

### 4.2 E5: Risk-Controlled Stopping (Learn-Then-Test)
- **Framework:** Learn-Then-Test (Angelopoulos et al., 2021) bounding family-wise error rate / false discovery rate under monotonic risk.
- **Primary Criterion (Doc 05 E5):**
  - Target error rate: $\alpha = 0.10$.
  - Target violation probability: $\delta = 0.10$.
  - Evaluation resamples: $\ge 500$ independent bootstrap splits from the calibration and test pools.
- **Exploratory Additions (explicitly labeled exploratory):**
  - $\alpha \in \{0.01, 0.05\}$ with $\delta = 0.10$. (Labeled exploratory because $\alpha = 0.01$ on $n_{\text{cal}} \approx 1,700$ operates near the sample size boundary).
- **Decision Rules:**
  - **Validity:** Empirical violation rate $\le \delta = 0.10$ across the $\ge 500$ resamples.
  - **Efficiency:** Mean looks saved vs fixed maximum budget ($K = 8$) $\ge 25\%$ at matched or lower error rate.
  - **Conservative Control:** Fixed-threshold plug-in rule violates target error by $\ge 2\delta$ at low sample counts, confirming the necessity of LTT bounds.
  - **KILL CONDITION:** Empirical violation rate $> \delta \implies$ exchangeability or independence broken; invalidate risk control claims and diagnose.

---

## 5. Aggregation and Reproducibility

- **Seeds:** Three primary evaluation seeds (17, 42, 99) governing scene generation and bootstrap resampling.
- **Provenance:** Every run writes `<run>.json` and `<run>.provenance.json` side by side, recording the exact source images and scene parameters.
- **Code Locking:** All measurement scripts and libraries pinned by SHA-256 in the final freeze JSON.
