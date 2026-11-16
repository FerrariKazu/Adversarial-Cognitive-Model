# 07 — Gate-1 pre-registration (DRAFT v2, not frozen)

Status: **DRAFT v2 (15 Nov 2026).** Successor to Draft v1 (12 Nov 2026). Incorporates owner decisions on OI-01 through OI-08 and protocol amendments C1 through C9. Nothing in this document has been run on the real checkpoint. It becomes binding only when `scripts/rhan_im/gate1_prereg.py freeze` has written `prereg/rhan_im_gate1_v1.json` from a committed, pristine repository state and that file has been committed *before* the first real run (§11). After the freeze, a change of any threshold, configuration value, data listing, or code file in the hash list is a **new version** (`rhan_im_gate1_v2`), never an in-place edit.

---

## 1. Question

Does the frozen `ff_112` trunk (seed 17, sha256 `d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc`), read through a 384→50 linear head fitted on its tokens, behave like the equal-variance Gaussian sensor that the RHAN-IM Bayes state assumes (`RHAN_IM_04_ARCHITECTURE_SPEC.md` §4), is a native-resolution look at a small target worth more than a coarse look at the whole scene, and can the tokens of one look be added as evidence?

---

## 2. What the gate decides, and what it does not

It decides whether gate 2 (calibration of the posterior, next-look policy comparison, Learn-then-Test stopping; `05` E2–E4) is worth running on this trunk **with the head and nothing else trained**, which evidence-combination rule ("sum" or "best") and which token-evidence treatment (`1/√VIF_null`; shared-difficulty flag) gate 2 uses, and which element of the plan to revise if the answer is no.

**Claim scope and boundaries:**
- The evaluation and any resulting claims are strictly bounded to the specific image pools fixed by this pre-registration (`prereg/pools.json` on ImageNet-100 `val/`), the 50 designated target classes, and synthetic scenes formed by unit-aligned pasted squares at sizes 2, 4, and 6 units (28, 56, 84 px) on 224×224 px canvases.
- It does **not** show:
  1. That natural objects in complex photographic context behave like pasted squares.
  2. Full-scale classification accuracy on ImageNet-1k or ImageNet-100.
  3. Adversarial robustness (AutoAttack or adversarial corruptions).
  4. Any computational efficiency or latency claim.
  5. Performance of a trunk that has undergone end-to-end recurrent or foveated fine-tuning (stage S3).
  6. Performance at target sizes other than 28, 56, and 84 px or canvas sizes other than 224 px.
- **Language-layer probe (E5):** Experiment E5 (probing alignment with language representations / Platonic Representation Hypothesis) is **outside Gate 1**. It is an independent post-gate investigation that does not affect Gate 1 thresholds, criteria, or progression decisions.

---

## 3. Frozen inputs

| Input | Value | How it is pinned |
|---|---|---|
| Primary trunk | `foundation_ff_112_seed17_final_d00fb750.pth` (sha256 `d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc`) | `checkpoint_hashes` in the freeze file; `freeze` refuses any other primary file unless `--allow-other-primary` is explicitly given; every run JSON records the sha256 of the checkpoint it actually loaded; a mismatch is an INVALID deviation (§7). |
| Secondary trunks | The `ff_112` seed-42 and seed-99 checkpoints | Hashed at freeze; each secondary run is verified against its frozen hash; reported only (§7). |
| Data | ImageNet-100 `train/` and `val/` | sha256 of sorted relative file paths of each split (`train_path_listing` and `val_path_listing`), recorded at freeze (`--data-root` required). Every run records its data listing hashes; any difference is an INVALID deviation. Validation data status is **fine-tuning-held-out** (used for Stage 0 monitoring, abort gates, best checkpoints, and baseline comparisons; not in the fine-tuning cross-entropy loss). *Validation use in hyper-parameter selection during Stage 0 training is UNVERIFIED; the claim wording holds either way.* Pretraining independence from DINOv2 LVD-142M is unverified; no pretraining-independence claim is asserted. |
| Class split | Explicit file: `prereg/class_split.json` (C1, OI-05, OI-10) | Pinned by `class_split_sha256` and `class_lists_sha256`. File must have status `FINAL`, containing exactly 50 target classes and 50 background classes with verified WordNet IDs (WNIDs) mapped to folder names. In the legacy folder name sorting, the boundary falls between `iron_smoothing_iron` and `jean_blue_jean_denim`. Disclosed subcategory breakdown: targets contain 27 organisms (animals) and 23 artifacts; backgrounds contain 17 organisms and 33 artifacts (honeycomb `n03530642` placed in non-animal/artifact in targets). No balanced-food or balanced-subcategory claim is asserted. |
| Calibration and test pools | Explicit file: `prereg/pools.json` (C8, OI-05, OI-11) | Bound to `class_split_sha256` and `val_repo_listing_sha256`. Pinned by `pool_assignment_sha256`. Divides the 50 images per class in `val/` into two disjoint banks: `calibration_pool_part_0_of_2` (2,500 images, even indices) and `test_pool_part_1_of_2` (2,500 images, odd indices). Every run checks that its image bank matches the frozen pools file exactly. |
| Head training scenes | `train/` | Drawn from the 50 target and 50 background folders defined in `prereg/class_split.json`. |
| Per-scene provenance sidecar | `<run>.provenance.json` (C7, OI-04) | Written alongside each run JSON. Records scene index, target class, background class, source image file paths, pasted coordinates, unit size, opacity, and window coordinates. Its sha256 (`provenance_sha256`) is stored inside the run JSON and verified by the aggregator. Report-only; does not enter the decision rule. |
| Original-split comparator | `original_split_report_only` run (C9, OI-06) | One report-only run on the primary checkpoint using measurement seed 17, evaluating the legacy directory-sorted split defined in `prereg/legacy_split.json` without pool enforcement. Recorded under `original_split_report_only`. Cannot alter the gate verdict, deviations, or pass counts. |
| Code | `rhan_im/gate1.py`, `scripts/rhan_im/gate1_measure_dprime.py`, `scripts/rhan_im/gate1_prereg.py`, `scripts/rhan_im/gate1_slope_heterogeneity.py`, `layout.py`, `trunk_adapter.py`, `window_head.py`, `sdt.py`, `gen2_foundation/backbone.py`, `noesis_vision/models/recurrent_block.py`, `noesis_vision/models/foveation.py` (`MEASUREMENT_CODE_FILES`) | sha256 of each file recorded at freeze. Every run JSON records the code hashes it executed (`code_sha256`). Aggregator verifies matches against freeze and analysis-time state. `freeze` refuses uncommitted modifications unless `--allow-dirty` is used for non-binding dry runs. |
| Runtime environment | Python version, PyTorch version, CUDA version, OS/platform, pip packages (C5) | Recorded in `runtime` block of every run JSON; checked by aggregator for consistency across runs. |
| Thresholds and aggregation rule | Defined in §6 and §7 | Embedded verbatim in the freeze file; aggregator executes frozen thresholds, ignoring post-freeze code alterations. |

---

## 4. Protocol

### Scene and Window Sampling
- **Canvas:** 224×224 px.
- **Background:** Centre-cropped ImageNet-100 image from a background class.
- **Target:** With probability 0.7, a square patch from a target-class image is pasted onto the canvas at a random unit-aligned position (grid unit = 14 px; 16×16 unit grid). With probability 0.3, the scene is null (no target pasted; size and opacity draws are stored as placeholders).
- **Target sizes:** 2, 4, or 6 units (28, 56, 84 px).
- **Target opacity $\alpha$:** 1.0 or 0.6.
- **Operating point:** Size 4 (56 px), opacity $\alpha = 1.0$.
- **Native look:** 112×112 px window cropped without resampling (25 candidate unit positions, stride 2 units = 28 px; 70% of sampled windows contain the target centre).
- **Gist look:** Entire 224×224 px canvas area-resampled to 112×112 px. Both looks are evaluated by the identical frozen trunk and head.

### Evidence Head Training
- **Architecture:** Linear(384, 51) fitted on token representations; exported as `score_c = logit_c − logit_bg` (50 outputs).
- **Optimization:** 600 steps, AdamW, learning rate 2e-3, cosine decay, batch size 48, using scenes synthesized from `train/`.
- **Head initialisation seed rule (C2, OI-01 Option A):**
  - For the three primary runs: `head_init_seed` equals the measurement seed (17, 42, 99 respectively).
  - For the two secondary runs: `head_init_seed = 17` (measurement seed 17, C5).

### What Differs Between Runs (§10.3)
- **Primary runs (seeds 17, 42, 99):** Run on the identical primary checkpoint. They differ in:
  1. Evidence head weight initialisation (`head_init_seed = 17, 42, 99`).
  2. Scene generation stream (stochastic placement, size/opacity draws, null-scene indicator).
  3. Image file draws from the disjoint calibration and test pools.
- **Secondary runs (checkpoints seed 42, seed 99):** Run with `head_init_seed = 17` and measurement seed 17. They isolate trunk differences from head initialisation and sampling variations.

### Sample Sizes and Semantics
- **`n_cal = 3000`:** 3,000 scenes synthesized from `calibration_pool_part_0_of_2` used exclusively to fit per-position standardisers (null mean $\mu$ and standard deviation $\sigma$ for each token position, look type, and class).
- **`n_test = 6000`:** 6,000 scenes synthesized from `test_pool_part_1_of_2` used to measure standardised scores, detectability, and dependency metrics.
- **`n_pairs = 2000`:** The first 2,000 test scenes receive a second, independently sampled native window (disjoint from target where applicable) to compute the cross-look noise correlation $\rho$. *This refers strictly to the number of dual-window scenes, not to paired source images.*

### Execution Command
```bash
python scripts/rhan_im/gate1_measure_dprime.py \
  --ckpt checkpoints/foundation_ff_112_seed17_final_d00fb750.pth \
  --data-root data/imagenet100 \
  --class-split prereg/class_split.json \
  --pools prereg/pools.json \
  --device cuda \
  --canvas 224 --window 112 --stride-units 2 --classes 50 \
  --sizes 2 4 6 --alphas 1.0 0.6 --op-size 4 --p-null 0.3 \
  --steps 600 --batch 48 --lr 2e-3 \
  --n-cal 3000 --n-test 6000 --n-pairs 2000 --chunk-scenes 200 \
  --seed <17|42|99> \
  --head-init-seed <17|42|99> \
  --out runs/gate1_primary_seed<17|42|99>.json
```

---

## 5. Quantities (per run, at the operating point)

- **Lower-median convention:** In accordance with PyTorch behavior (`torch.median`), for an even number of values, the median is defined as the lower of the two middle values ($k/2$-th element, 0-indexed). All positional medians throughout Gate 1 follow this lower-median convention.
- **Detectability $d$ of a token position:** Mean standardised own-class score on tokens overlapping the target minus the null token mean, expressed in null-sd units ($d = (\mu_{\text{pos}} - \mu_{\text{null}}) / \sigma_{\text{null}}$).
- **Gist minimum threshold (`GIST_MIN_POS = 30`, C6, OI-03):** Token positions with fewer than 30 positive instances are unmeasured (`None`). This 30-instance floor is enforced identically across both native and gist token analysis sites.
- **`d_native_centre`:** Positive-count-weighted mean detectability over the 4×4 centre block of the 8×8 native window tokens (grid indices 2–5 on both axes).
- **`d_gist` and `d_gist_full` in the operating cell (C3, OI-02):**
  - `d_gist` is the mean detectability over all coarse tokens overlapping the target.
  - `d_gist_full` is the mean detectability over coarse tokens fully covered ($\ge 99\%$) by the target.
  - **Operating cell restriction (C3):** `d_gist_full` and `partially_covered_token` are evaluated strictly within the operating cell (size 4, opacity 1.0). The null mean is computed from all null scenes of the run (~1,800 scenes) plus the operating-cell target scenes.
  - **Instance floor:** Both quantities require at least 30 token instances (`GIST_FULL_MIN_INSTANCES = 30`); if fewer than 30 instances occur, the metric is unmeasured (`None`).
- **z-ROC slope:** Slope of the z-transformed ROC curve of own-class scores versus null scores per position; summarized as the lower median over measured positions (theoretical value for equal-variance Gaussian evidence is 1.0). If empirical hit/false-alarm rates prevent numerical fitting ($d'$ too large), the standard deviation ratio $\sigma_{\text{null}} / \sigma_{\text{pos}}$ is used as fallback.
- **Slope heterogeneity check (C4):** Evaluated via `scripts/rhan_im/gate1_slope_heterogeneity.py`. Report-only analysis inspecting per-position slope dispersion across the token grid. Has no pass/fail threshold and does not affect the gate decision.
- **`null_sd`:** Standard deviation of standardised null scores, summarized as the lower median over positions. Confirms standardiser transfer across image halves; expected to be close to 1.0 by construction.
- **Cross-look correlation $\rho$:** Pearson correlation between standardised null scores of identical grid units across two independently sampled native windows of the same scene.
- **Within-look dependence:**
  - $\text{VIF}_{\text{null}}$: Variance Inflation Factor on 4×4 token blocks of target-free scenes across all class channels.
  - $\text{VIF}_{\text{pos}}$: Variance Inflation Factor on target own-class scores (residualized around per-position means) on scenes where the window contains the complete target (requires $\ge 50$ whole-target scenes).
  - Defined as $\text{VIF} = \text{Var}(\sum w_i z_i) / \sum w_i^2 \text{Var}(z_i)$; equals 1.0 for mutually independent tokens.
- **Spillover:** Mean own-class score shift on uncovered tokens adjacent to and distal from the target, measured in null-sd units. **Report-only**; does not enter the decision rule.

---

## 6. Decision rule (per run)

1. **`sdt_adequate`:**
   Lower-median z-ROC slope $\in [0.80, 1.25]$ **and** lower-median `null_sd` $\in [0.85, 1.20]$. When z-ROC slope cannot be directly fitted, the $\sigma_{\text{null}} / \sigma_{\text{pos}}$ ratio is substituted. If neither can be computed, the quantity is unmeasured (triggering the fail-closed rule).
2. **`informative`:**
   `d_native_centre` $\ge 1.5$ **and** `d_block` $\ge 2.0$.
   Here $d_{\text{block}} = d_{\text{native\_centre}} \cdot \sqrt{16 / \text{VIF}_{\text{used}}}$, where 16 is the token count of the 4×4 operating block, and $\text{VIF}_{\text{used}} = \text{VIF}_{\text{null}}$ if $\text{VIF}_{\text{null}} > 1.25$, else 1.0.
   **Fail-closed rule:** If $\text{VIF}_{\text{null}}$ or operating block token count is missing, non-finite, or non-positive, $d_{\text{block}}$ is undefined, `informative` evaluates to `False`, the metric is recorded under `unmeasured`, and the aggregate outcome is INCONCLUSIVE (not FAIL).
   *Scope note:* $d_{\text{block}}$ represents a conservative lower bound under spatial noise correlation; a failure with elevated $\text{VIF}_{\text{null}}$ is reported as "not shown informative by this estimator."
3. **`looking_closer_pays`:**
   $d_{\text{native\_centre}} / \max(d_{\text{gist}}, d_{\text{gist\_full}}) \ge 1.5$.
   The comparator is the best-case coarse look within the operating cell. If neither gist metric can be computed, the criterion evaluates to `False` and `d_gist` is listed under `unmeasured`.
4. **`pass_all`:**
   Evaluates to `True` iff `sdt_adequate` $\wedge$ `informative` $\wedge$ `looking_closer_pays`.
   *Fail-closed enforcement:* Any missing required quantity (`vif_null_op`, `op_block_tokens`, `d_native_centre`, `d_gist`, `null_sd_median`, `zroc_slope_median`) forces `pass_all = False` and renders the aggregate INCONCLUSIVE.
5. **Evidence combination policy (report-only):**
   $\rho \le 0.20 \implies \text{"sum"}$; $\rho \ge 0.60 \implies \text{"best"}$; $0.20 < \rho < 0.60 \implies \text{"best"}$ (sum is unjustified under intermediate correlation).
6. **Token evidence treatment (report-only):**
   - If median $\text{VIF}_{\text{null}} > 1.25$: divide token scores and detectabilities by $\sqrt{\text{VIF}_{\text{null}}}$.
   - If $\text{VIF}_{\text{pos}} / \max(\text{VIF}_{\text{null}}, 1.0) > 1.25$: **flag shared scene difficulty** (remedied by scale-mixture likelihood in S2, not shrinkage).

---

## 7. Aggregation and consequences

The gate evaluates three primary runs (seeds 17, 42, 99) on the primary checkpoint. The outcome label is determined strictly by priority:

1. **INVALID (protocol deviations):**
   Triggered if any primary run exhibits deviations visible to the harness:
   - Hash mismatches in primary checkpoint, code files (`MEASUREMENT_CODE_FILES`), data listings (`train_path_listing`, `val_path_listing`), class-split file, or pools file.
   - Missing or mismatched per-scene provenance sidecar hash (`provenance_sha256`).
   - Missing runtime environment record.
   - Seeds other than integers 17, 42, 99 or incorrect run count.
   - Secondary runs executed with seeds other than measurement seed 17 and head seed 17.
   - Any execution flagged as `--self-test`.
   - Malformed run JSON files.
   Every deviation is enumerated in `deviations`. The aggregator reports the nominal verdict under `gate1_if_deviations_and_unmeasured_quantities_were_ignored` for diagnostic transparency, which has no procedural standing.
2. **INCONCLUSIVE (unmeasured required quantities):**
   Triggered in the absence of deviations if any primary run has an unmeasured required quantity (§6.4). An unmeasured quantity in a single run cannot be overridden by other runs.
3. **PASS:**
   Triggered iff at least 2 of the 3 primary runs achieve `pass_all == True`.
4. **FAIL:**
   Triggered if 2 or more primary runs fail one or more criteria without deviations or unmeasured quantities.

### Secondary and Comparator Runs
- **Secondary checkpoints (seeds 42, 99):** Run with seed 17. **Reported only**. If the primary checkpoint passes but a secondary fails, the result is designated *trunk-dependent*, restricting claims to the primary trunk.
- **Original-split run (C9):** Run on primary checkpoint with seed 17 and `prereg/legacy_split.json`. **Reported only** under `original_split_report_only`; does not affect pass counts or verdicts.

| Outcome | Procedural Consequence |
|---|---|
| **INVALID** | Gate undecided. Correct operational defect (e.g. data path, file mismatch) or issue a revised pre-registration (`v2`) before reviewing results. |
| **INCONCLUSIVE** | Re-execute runs with unmeasured quantities. If unmeasurable due to low positive counts, scene counts must be enlarged under a new pre-registration version. |
| **PASS** | Advance to Gate 2 (`05` E2–E4) using the determined combination rule and token evidence treatment. |
| **`sdt_adequate` fails** | Implement heteroscedastic likelihood (`04` §11 P1) or one stage S3 tuning cycle; if slope remains outside $[0.80, 1.25]$, drop exact-Bayes claim and retain distribution-free Learn-then-Test layer. |
| **`informative` fails** | Execute Stage S3 fine-tuning once (tune last 2–4 blocks with dense window cross-entropy, stopping on held-out $d'$); re-test under a new pre-registration. If failure persists, trigger architectural fallback (`03` §9). |
| **`looking_closer_pays` fails** | Native resolution confers no advantage over gist. RHAN-IM active foveation is discontinued in favour of calibrated selective classification. |
| **Shared difficulty flagged** | Implement scale-mixture likelihood (`04` §4) in S2; stratify Experiment E2 by difficulty tercile. |

---

## 8. Prior distribution (subjective baseline)

Recorded prior probabilities (subjective, uncalibrated):
- $P(\text{looking\_closer\_pays}) \approx 0.80$ (2× linear resolution advantage of 56 px target).
- $P(\text{informative}) \approx 0.65$.
- $P(\text{sdt\_adequate}) \approx 0.30$ (neural network representations frequently exhibit wider positive variance than null variance, driving slopes below 1.0).
- Joint $P(\text{pass\_all}) \approx 0.20$.
The expected primary path is the heteroscedastic likelihood branch of §7.

---

## 9. Operating characteristics (simulation study)

Evaluated via synthetic token scores matching sensor properties passed through `gate1_measure_dprime.analyse()` with nominal protocol parameters ($N_{\text{cal}} = 3000$, $N_{\text{test}} = 6000$, $N_{\text{pairs}} = 2000$).
Deterministic verification: running `gate1_operating_characteristics.py --only ideal_equal_variance --replicates 1` yields a size-4 median positive count of **102.0** per position.

Summary across 11 synthetic conditions (6 replicates each, resolution 1/6):
- **Ideal equal-variance sensor:** Passes 6/6; neither dependence flag fires ($\text{VIF}_{\text{null}} = 1.00$, $\text{VIF}_{\text{pos}} = 1.00$ [0.91, 1.10]); z-ROC slope median 0.98.
- **Correlated noise ($\rho = 0.30$):** Passes 6/6; $\text{VIF}_{\text{null}} = 5.44$; tokens corrected 6/6; difficulty flag 0/6; corrected $d_{\text{block}} = 4.01$.
- **Shared difficulty (cv 0.2, 0.4):** Passes 6/6; difficulty flagged 6/6; slopes depressed to 0.92 and 0.82 respectively.
- **Unequal variance ($s = 1.15, 1.25, 1.40$):** Slope degrades from 0.86 (passes 6/6) to 0.80 (passes 1/6) and 0.71 (passes 0/6).
- **Weak sensor ($d = 1.2$):** Fails `informative` 6/6; passes others.
- **No resolution advantage ($d_{\text{gist}} = 2.4$):** Fails `looking_closer_pays` 6/6.
- **Extreme noise dependence ($\rho = 0.80, d = 1.7$):** $\text{VIF}_{\text{null}} = 12.83$; corrected $d_{\text{block}} = 1.78 < 2.0$; fails `informative` 6/6.

---

## 10. Limitations and structural constraints

1. **Pasted synthetic targets:** Targets are planar, unit-aligned squares with artificial border discontinuities. Performance does not imply equivalence on natural, embedded objects.
2. **Image reuse across scenes:** 6,000 test scenes are generated by sampling from 2,500 test pool images (1,250 target, 1,250 background). Scenes are not statistically independent draws. Nominal sample sizes overstate degrees of freedom; no confidence intervals are computed for decision criteria.
3. **Primary run dependence:** Primary runs share the identical checkpoint and image pools; they differ in evidence head initialisation and synthetic scene composition. They protect against optimization and sampling variance, not dataset sampling artifacts.
4. **`null_sd` criterion power:** Evaluates standardiser transfer across halves of the identical validation dataset; it serves as a sanity check rather than a stringent test of invariance.
5. **Constant-VIF bound:** $d_{\text{block}}$ assumes equicorrelated tokens with uniform detectability; it acts as a conservative bound when detectability is peaked at the target centre.
6. **Sample size at size 2:** Size-2 targets (4 tokens) yield a median of 29 positive samples per position at $N_{\text{test}} = 6000$, leaving some peripheral token positions unmeasured. Gate decisions are anchored to the size-4 operating point where positive counts exceed 100 per position.
7. **Fixed linear head:** Evidence is extracted by a 600-step linear head. Results reflect this specific read-out architecture.

---

## 11. Pre-registration freeze checklist

The binding pre-registration requires completion of the following sequence:

1. **Verification of code and environment:**
   - All code in `MEASUREMENT_CODE_FILES` committed with clean git working tree.
   - Full test suite passing (`pytest tests/rhan_im`).
   - Synthetic operating-characteristics study verified.
2. **Data and split artifacts committed under `prereg/`:**
   - `prereg/class_split.json`: status `FINAL`, verified WNIDs, 50/50 target/background split.
   - `prereg/legacy_split.json`: status `FINAL`, directory-sorted baseline split.
   - `prereg/pools.json`: status `FINAL`, bound to `val_repo_listing_sha256` and `class_split_sha256`.
3. **Dry-run freeze validation:**
   Execute `scripts/rhan_im/gate1_prereg.py freeze` with `--allow-dirty` targeting a temporary file; inspect all embedded hashes and configuration keys.
4. **Production freeze:**
   Execute freeze from a clean git commit:
   ```bash
   python scripts/rhan_im/gate1_prereg.py freeze \
     --primary checkpoints/foundation_ff_112_seed17_final_d00fb750.pth \
     --secondary checkpoints/foundation_ff_112_seed42_final.pth checkpoints/foundation_ff_112_seed99_final.pth \
     --data-root data/imagenet100 \
     --class-split prereg/class_split.json \
     --pools prereg/pools.json \
     --out prereg/rhan_im_gate1_v1.json
   ```
5. **Commit pre-registration artifact:**
   Commit `prereg/rhan_im_gate1_v1.json` to git repository **prior to executing any measurement on the real checkpoint**. The commit SHA and timestamp establish the binding pre-registration baseline.
6. **Execution and aggregation:**
   Execute the three primary runs, two secondary runs, and original-split comparator. Run `scripts/rhan_im/gate1_prereg.py aggregate` to compute the official verdict. Every generated JSON is retained and published without post-hoc exclusion.

