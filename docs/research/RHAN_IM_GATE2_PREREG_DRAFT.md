# Gate-2 Pre-Registration: Selective Classification, Calibration, and Learn-Then-Test Risk Control (Rules & Protocol Freeze)

Status: **RULES & PROTOCOL FROZEN on 17 Nov 2026 prior to Stage S3 execution.**
Scope: **INTERNAL QUALIFICATION ONLY.**
Trunk Target: Frozen baseline trunk `foundation_ff_112_seed17_final_d00fb750.pth` (SHA-256 `d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc`).
Freeze Phase: **Rules and Protocol Freeze.** (Measurement code hashes will be pinned in a second step prior to executing on data; no full code freeze is claimed yet).

---

## 1. Questions and Scope Boundary

### 1.1 Research Questions
1. **E2 (Posterior Calibration):** Given the SDT-adequate sensor validated in Gate 1 v1, is the multi-look posterior well-calibrated (ECE ≤ 0.03, NLL, Brier score) under the measured noise and dependence parameters?
2. **E5 (Selective Classification & Risk-Controlled Stopping):** Does Learn-Then-Test (LTT) risk control bound empirical error below $\alpha = 0.10$ with probability at least $1 - \delta = 0.90$ across deployment resamples, while saving looks compared to fixed-horizon evaluation?

### 1.2 Strict Scope Boundaries
- **No Spatial Resolution Claims:** Gate 1 v1 established that whole-scene gist views provide detectability comparable to native window crops **at size 4 (56 px)**. Therefore, Gate 2 **makes no claims** that adaptive looking beats whole-scene gist, nor that spatial resolution confers an advantage. Gate 2 focuses exclusively on selective classification, Bayesian belief accumulation, posterior calibration, and statistical risk control.
- **Internal Qualification Only:** All validation images are from ImageNet-100 `val/`. The foundation backbone was trained (supervised TRADES fine-tuning) on `train/`, and Gate 1 evaluated on `val/`. Gate 2 claims are explicitly designated as internal qualification of the Bayesian stopping engine on pasted-squares-on-natural-backgrounds scenes, not out-of-distribution generalization.
- **Trunk Independence:** Frozen strictly against the validated `ff_112_seed17` trunk. No reliance on Stage S3 fine-tuning.

---

## 2. Frozen Inputs and Data Partitions

### 2.1 Pinned Files and Artifacts

| Item | Pinned Reference | Purpose |
|---|---|---|
| Backbone Trunk | `foundation_ff_112_seed17_final_d00fb750.pth` (SHA-256 `d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc`) | Frozen feature extractor |
| Class Split | `prereg/class_split.json` (SHA-256 `289f2818f0eae4dd7b654627d4bdf197cc11d43923300c52df7b4c0bdecc1241`) | 50 target classes, 50 background classes |
| Validation Split | `prereg/gate2_val_split.json` (SHA-256 `b8c4543f7b7272cae43ebfd1a9ad2095908b65b36c0ae2577c9e8e3b8f92bed0`) | Pinned 3-way split of 5,000 val images |
| Code Files | `rhan_im/stopping.py`, `rhan_im/belief.py`, `rhan_im/window_head.py`, `rhan_im/gate1.py` | To be pinned by SHA-256 in code-lock step before data execution |

### 2.2 Three-Way Disjoint Split (`prereg/gate2_val_split.json`)
Doc 05 line 34 requires that LTT calibration items must be strictly disjoint from everything fitted upstream. Gate 2 cannot reuse Gate 1's 2-way pools file. Instead, the 5,000 ImageNet-100 `val/` images (50 per class × 100 classes) are partitioned into three strictly disjoint sets:
1. **Fit Pool (`fit_pool_part_0_of_3`):** 17 images/class (1,700 images total). Used exclusively to measure Gate-2-specific noise parameters ($VIF_{\text{null}}$, null mean, null variance, and $\kappa$). The readout head is trained upstream on `data/imagenet100/train/` per §3.1.
2. **Calibration Pool (`calibration_pool_part_1_of_3`):** 17 images/class (1,700 images total). Used exclusively to calibrate the LTT risk-controlling thresholds $\hat{\lambda}$ and minimum look floor $k_{\min}$.
3. **Test Pool (`test_pool_part_2_of_3`):** 16 images/class (1,600 images total). Used exclusively to evaluate empirical risk violation rates, coverage, and look-saving efficiency.

### 2.3 Scene Generation and Disjoint Unit Contract
Per the 13 Nov decision, one image pair is the strict unit:
- **Fit Pool:** Generates 1,700 null calibration scenes (for noise standardisation, $VIF_{\text{null}}$, and $\kappa$).
- **Calibration Pool:** Generates 1,700 composited scenes (50% null, 50% target). In each bootstrap resample, a sub-sample of $n_{\text{cal}} = 300$ scenes is drawn.
- **Test Pool:** Generates 1,600 composited scenes (50% null, 50% target).
- **Zero Cross-Pool Reuse:** No background or target image in `calibration_pool_part_1_of_3` appears in `test_pool_part_2_of_3` or `fit_pool_part_0_of_3`.

---

## 3. Sensor Setup and Upstream Parameter Fitting

### 3.1 Head Refit Procedure
Gate 1 did not export a persistent head checkpoint. Gate 2 fits its own readout head from scratch:
- **Architecture:** `Linear(384, 51)` exporting `score_c = logit_c - logit_bg`.
- **Training Data:** `data/imagenet100/train/` (~126,000 images, 600 steps, batch 48), exactly as Gate 1 did. This matches the Gate 1 head-fitting procedure identically.
- **Optimization:** 600 steps, AdamW, learning rate 2e-3, cosine decay, batch size 48, seed matching the evaluation seed.
- **Export:** Readout projection copied to `EvidenceHead(384, 50)`.
- **Head File:** Pinned by SHA-256 hash in freeze JSON, or refitted deterministically from the pinned seed and training procedure.

### 3.2 Noise and Dependence Parameters
Gate 2 does **not** copy Gate 1's $VIF_{\text{null}} = 11.605$. That value was measured on pasted squares on Gate 1's scene configuration; Gate 2 must measure its own.
- $VIF_{\text{null}}$, token standardizers ($\mu_0, \sigma_0$), and noise-calibration parameter $\kappa$ are measured directly on `fit_pool_part_0_of_3` (1,700 images) null scenes.
- If $VIF_{\text{null}} > 1.25$, token scores and detectabilities are scaled by $1 / \sqrt{VIF_{\text{null}}}$ (equicorrelated correction).

### 3.3 Look Policy Sequence
Gate 2 evaluates the **fixed look budget** policy with up to $K = 8$ sequential looks per scene:
- **Look 1:** Whole-scene gist view (resampled 112×112 view of the entire 224×224 canvas).
- **Looks 2–8:** Pseudo-random draws from the 25 candidate native 112×112 window locations, deterministically determined by scene seed and look step.
- Each look updates the Bayesian belief accumulation state. The "looks saved" metric evaluates how many looks the stopping rule skips relative to the fixed $K = 8$ horizon. No adaptive spatial selection is claimed.

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
  - **INCONCLUSIVE:** $\text{ECE} \in (0.03, 0.05]$: explicitly reported as inconclusive — insufficient evidence to pass, but not necessarily misspecified.
  - **DIAGNOSE / FAIL:** $\text{ECE} > 0.05 \implies$ likelihood misspecified; stop and diagnose before any deployment claim.

### 4.2 E5: Risk-Controlled Stopping (Learn-Then-Test)
- **Framework:** Learn-Then-Test (Angelopoulos et al., 2021) bounding family-wise error rate / false discovery rate under monotonic risk.
- **Primary Criterion (Doc 05 E5):**
  - Target error rate: $\alpha = 0.10$.
  - Target violation probability: $\delta = 0.10$.
  - Resampling scheme: Subset $n_{\text{cal}} = 300$ from the 1,700-image calibration pool over $B \ge 500$ overlapping bootstrap resamples.
- **Risk Evaluation on Test Pool:**
  - In each resample $b \in \{1, \dots, B\}$, the calibration subset identifies stopping threshold $\hat{\lambda}_b$ and floor $k_{\min, b}$.
  - The empirical risk of $(\hat{\lambda}_b, k_{\min, b})$ is evaluated on the full independent 1,600-scene test pool (`test_pool_part_2_of_3`).
  - An empirical violation occurs if $\text{Risk}_{\text{test}}(\hat{\lambda}_b, k_{\min, b}) > \alpha$.
  - The empirical violation rate across all $B$ resamples is $\frac{1}{B} \sum_{b=1}^B \mathbb{I}(\text{Risk}_{\text{test}}(\hat{\lambda}_b, k_{\min, b}) > \alpha)$.
- **Exploratory Additions (explicitly labeled exploratory):**
  - $\alpha \in \{0.01, 0.05\}$ with $\delta = 0.10$. (Labeled exploratory because $\alpha = 0.01$ on $n_{\text{cal}} \approx 300$ operates near the sample size boundary for LTT coverage guarantees).
- **Decision Rules:**
  - **Validity:** Empirical violation rate $\le \delta = 0.10$ across the $\ge 500$ resamples.
  - **Efficiency:** Mean looks saved vs fixed maximum budget ($K = 8$) $\ge 25\%$ at matched or lower error rate.
  - **Conservative Control:** Fixed-threshold plug-in rule violates target error by $\ge 2\delta$ at low sample counts, confirming the necessity of LTT bounds.
  - **KILL CONDITION:** Empirical violation rate $> \delta \implies$ exchangeability or independence broken; invalidate risk control claims and diagnose.

---

## 5. Aggregation and Reproducibility

- **Seeds:** Three primary evaluation seeds (17, 42, 99) governing head initialisation, scene generation, and bootstrap resampling.
- **Aggregation Rule:** 2-of-3 seeds must pass each criterion for the criterion to pass.
- **Provenance:** Every run writes `<run>.json` and `<run>.provenance.json` side by side, recording the exact source images and scene parameters.
- **Code Locking:** All measurement scripts and libraries to be pinned by SHA-256 hash in the final code-lock JSON prior to evaluation runs on data.

## 6. Tuned-Trunk Gate 2

If Gate 1 v2 passes (S3-tuned trunk restores $d_{\text{native}} / d_{\text{gist}} \ge 1.50$), a separate subsequent Gate 2 version may be registered against the tuned trunk with spatial resolution claims. That is not this document.

