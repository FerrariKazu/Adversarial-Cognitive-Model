# Stage 0B — Final Preregistration

**Protocol ID:** `rhan-stage0b-dinov2-b14-temperature/1`  
**Protocol version:** `1.0.0`  
**Status:** **FROZEN CANDIDATE FOR HUMAN PROTOCOL REVIEW — NOT LAUNCHED**  
**Date:** 2026-11-21  
**Parent:** `rhan-stage0a-dinov2/1`, version `1.0.0`, closeout **CONDITIONAL PASS**  
**Supersedes for Stage 0B planning:** `audit/stage0b_preparation_v1/STAGE0B_FINAL_PREREGISTRATION_PROPOSAL.md`  
**Stage 0B launch:** **NOT AUTHORIZED**  
**Real temperature fitted:** **NO**  
**Stage 0B evaluation outcomes inspected:** **NO**

> This document freezes a candidate protocol for human review. It does not authorize fitting a temperature, scoring the evaluation split, or uploading any image-derived artifact. A separate explicit approval is required before execution.

---

## 1. Authority and preservation

This directory is additive. It does not modify or reinterpret:

- `audit/stage0a_closeout_v1/`
- the original Stage 0A checkpoints or their sidecars
- `audit/stage0b_preparation_v1/`
- `audit/stage0b_data_gate_v1/`
- RHAN-IM or RHAN-Bi-v1 experimental records

The historical Stage 0B proposal is retained unchanged. Where it conflicts with the completed data gate, this preregistration explicitly supersedes it for Stage 0B. The superseded items are:

| Historical proposal | Frozen Stage 0B decision |
|---|---|
| Path-based split: `SHA-256(protocol|seed|class_index|relative_path)` | **Content-hash split:** `SHA-256(ASCII(seed) || ASCII(WNID) || raw_32_byte_image_SHA256)` |
| “Paired bootstrap over 5,000 evaluation images” | **Class-stratified paired bootstrap**, 5 images resampled within each of 1,000 classes |
| Data gate pending | Data gate completed **CONDITIONAL PASS**; archive, audit, mapping, and partition are bound below |
| Head/feature-level image replay caveat | Raw-image inference path verified for Stage 0B on eight fixed calibration images |

The obsolete path-based algorithm must not be used. The frozen source of truth is `audit/stage0b_data_gate_v1/PARTITION_MANIFEST.json`.

---

## 2. Confirmed facts

These are evidence-backed, not assumptions:

1. The ImageNetV2 MatchedFrequency archive was downloaded from the pinned revision `d626240be2538720e83103a0e1178d24aca8b12c`.
2. Archive size is `1,264,079,360` bytes; SHA-256 is `f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c`.
3. The archive is an **uncompressed POSIX tar** despite its `.tar.gz` name; all 11,001 members were read with zero size mismatches and extraction passed.
4. The real dataset audit passed: 10,000 JPEG/RGB images, 1,000 classes, 10 per class, all decoded, no unexpected entries, no exact duplicate content.
5. All 1,000 numeric ImageNetV2 folders map to the canonical ImageNet WNIDs and Stage 0A class order. The Stage 0A class-name digest is `385e0240499426c022b773be1ae4da780f057b89c9694c7f47199f62746b6edd`.
6. The deterministic partition passed: 5,000 calibration / 5,000 evaluation, five per class per split, zero overlap, all image hashes rematched, recomputed twice identically.
7. The partition canonical hash is `557e10f1b94c161bc5e57926c137faaa27208410743a0a039b5b43be6da52578`; the JSON file hash is `fa7f23032d6b953664bc573a53d7d90b4d84be3d46fe7faed445e288168135e6`. Both are frozen.
8. The three Stage 0A ViT-B/14 `final.pt` files and sidecars match:
   - seed 17: `320f79a79b7110c7f8120eaedaa122049f8e2b12ba2fe765e867f1e1f3af308e`
   - seed 42: `f5e7d42d3a901b93b0fb0d67d2cf3d25a0fa34f8dc5d59b9530c2ff40b23c357`
   - seed 99: `cb91a2b14c96b514c977e2c09a87deb7002ea3d87450f1a04d2de45a7aac452d`
9. The pinned DINOv2 ViT-B/14 weights file hash is `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b`; the reconstructed runtime state-dict hash is `375b4c89151489e61df6f1f3b3fe1f9fa8b815c52ae0f7117b23e2e201fbb4fe`.
10. The raw-image inference test passed on eight fixed calibration images: loading, RGB conversion, Stage 0A transform equivalence, backbone state hash, 768-D CLS features, three 1,000-way heads, and finite float32 logits.
11. The synthetic calibration gate passed 17/17; the synthetic evaluation-core gate passed 7/7. Neither used real predictions or real temperatures.
12. The compute probe measured approximately **267 images/s** for the shared backbone at batch size 8, with **0.544 GiB** peak CUDA allocation. This is an engineering observation, not a Stage 0B result.

---

## 3. Frozen design

### 3.1 Question

For three frozen Stage 0A ViT-B/14 linear heads, does fitting one positive scalar temperature on new calibration images make confidence more representative of accuracy on disjoint new images, without changing classification decisions or degrading proper scores?

This is a calibration study. It is not a backbone, classifier, robustness, or adaptive-perception study.

### 3.2 In scope

- One scalar temperature per seed: seeds 17, 42, 99.
- Fit on the 5,000-image ImageNetV2 calibration split only.
- Assess on the disjoint 5,000-image ImageNetV2 evaluation split.
- Raw-versus-calibrated comparison, per seed and as an equal-weight three-seed mean.
- 15-bin ECE as the primary endpoint.
- NLL and Brier score as proper-score guards.
- Accuracy preservation as an invariant.
- Reliability, selective prediction, and descriptive class-conditional summaries.

### 3.3 Out of scope

- Backbone or head training.
- Feature extraction from a different backbone.
- Per-class, vector, matrix, ensemble, Dirichlet, or neural calibration.
- Best-seed selection or seed dropping.
- Hyperparameter searches over temperature bounds, bin counts, bootstrap settings, or evaluation subsets.
- Any claim of adaptive perception, principled abstention, or general robustness.
- Any Stage 0B outcome claim before a frozen, approved run.

---

## 4. Data

### 4.1 Primary source

| Field | Frozen value |
|---|---|
| Dataset | ImageNetV2 MatchedFrequency |
| Source revision | `d626240be2538720e83103a0e1178d24aca8b12c` |
| Archive SHA-256 | `f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c` |
| Images | 10,000 |
| Classes | 1,000 |
| Images per class | 10 |
| Local root | `audit/stage0b_data_gate_v1/work/extracted/imagenetv2-matched-frequency-format-val` |
| Licensing | Local, non-commercial educational/research use only; no redistribution or upload |

The archive-extension discrepancy is documented in `STAGE0B_DATA_PROVENANCE.md`. The bytes and digest, not the filename, are authoritative.

### 4.2 Partition

Algorithm, frozen:

```text
rank = SHA-256(ASCII("20261121") || ASCII(WNID) || raw_32_byte_image_SHA256)
within each class: ascending rank
positions 0-4  -> calibration
positions 5-9  -> evaluation
```

Partition invariants:

- exactly 5,000 calibration and 5,000 evaluation images;
- exactly five images per class in each split;
- calibration and evaluation are disjoint by path and content hash;
- no image is added, removed, or reassigned after freeze;
- class index is parsed from the numeric folder name, never from alphabetical `ImageFolder` order.

The partition is not re-drawn. The calibration set is the only set used to fit temperatures. The evaluation set is not used for fitting, model selection, threshold selection, feature selection, implementation choice, or protocol amendment.

### 4.3 Secondary benchmark

Official ImageNet-1k validation is retained only as an optional secondary fixed benchmark. It was already evaluated in Stage 0A, is not independent, and cannot override the primary decision. It must never be used to fit or select a temperature. If the verified Stage 0A feature cache is unavailable or fails hash verification, the secondary analysis is omitted, not reconstructed from unverified data.

### 4.4 Data-use constraints

- Non-commercial educational/research use only.
- Local storage only.
- No redistribution of source images.
- No upload of raw images.
- No upload of features, logits, probabilities, predictions, or other image-derived artifacts until permitted use is established.
- No public Hugging Face dataset repository containing the source data.
- Supervisor/institutional confirmation is a launch prerequisite where required.

No legal assurance beyond the documented terms is made.

---

## 5. Models

### 5.1 Backbone

| Field | Frozen value |
|---|---|
| timm name | `vit_base_patch14_dinov2.lvd142m` |
| HF repository | `timm/vit_base_patch14_dinov2.lvd142m` |
| HF revision | `4685c99dabffe5affac90bd99dbffd25801ae58d` |
| Weights file | `model.safetensors` |
| Weights file SHA-256 | `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b` |
| State-dict SHA-256 | `375b4c89151489e61df6f1f3b3fe1f9fa8b815c52ae0f7117b23e2e201fbb4fe` |
| Architecture | DINOv2 ViT-B/14, patch 14, width 768, depth 12, 12 heads |
| Pooling | pooled CLS token, `global_pool='token'`, `num_classes=0` |
| Feature dimension | 768 |

The backbone is frozen in eval mode with gradients disabled. The three heads share one backbone forward pass per image.

### 5.2 Heads

| Seed | Artifact | SHA-256 |
|---:|---|---|
| 17 | `audit/stage0a_closeout_v1/artifacts/best/stage0a/vit-b14/seed17/final.pt` | `320f79a79b7110c7f8120eaedaa122049f8e2b12ba2fe765e867f1e1f3af308e` |
| 42 | `audit/stage0a_closeout_v1/artifacts/best/stage0a/vit-b14/seed42/final.pt` | `f5e7d42d3a901b93b0fb0d67d2cf3d25a0fa34f8dc5d59b9530c2ff40b23c357` |
| 99 | `audit/stage0a_closeout_v1/artifacts/best/stage0a/vit-b14/seed99/final.pt` | `cb91a2b14c96b514c977e2c09a87deb7002ea3d87450f1a04d2de45a7aac452d` |

Each artifact contains `model = {'fc.weight': [1000,768], 'fc.bias': [1000]}`. `final.pt` is used. `best.pt` is not used because the Stage 0A audit found it stale for seeds 42 and 99.

### 5.3 Inference configuration

Stage 0B inference is a **separately pinned configuration**, not a claim of byte-identical reconstruction of Stage 0A's unrecorded training-time software:

- Pillow `Image.open`, context-managed, then `convert('RGB')`.
- Resize shorter side to 256 with bicubic interpolation and antialias.
- Center crop 224.
- `ToTensor`.
- Normalize with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
- CUDA autocast float16, channels-last, eval mode, inference mode.
- Backbone output cast to float16 for the cache and float32 for the head.
- Head output float32 `[N,1000]`.

The documented transform was tested for exact tensor equality against `stage0a.data.make_transform(224)`. The backbone state-dict hash and all three head hashes were re-verified. Full evidence is in `STAGE0B_INFERENCE_PIPELINE_AUDIT.md` and `STAGE0B_INFERENCE_CONFIG.json`.

---

## 6. Temperature method

For each seed \(s\), independently:

\[
p_{T_s}(y=c \mid x) = \operatorname{softmax}\left(\frac{z_s(x)}{T_s}\right), \qquad T_s>0.
\]

- \(z_s(x)\) are the frozen Stage 0A logits.
- \(T_s\) is fitted by minimizing mean NLL on the 5,000 calibration images only.
- Parameterization: \(\theta_s=\log T_s\), with \(T_s=\exp(\theta_s)\).
- Bounds: \(T_s \in [0.05,20]\).
- Initialization: \(T_s=1\).
- Fitting precision: float64; original logits preserved as float32.
- Optimizer: `scipy.optimize.minimize_scalar`, method `bounded`, bounds on \(\log T\), `xatol=1e-10`, `maxiter=1000`.
- Record NLL at \(T=1\), NLL at the optimum, iteration count, convergence flag, and boundary flag.
- A boundary solution is reported and is **not** silently clipped, widened, or replaced. For confirmatory acceptance, all fits must converge and no fit may be at a bound.
- No per-class temperature.
- No calibration on evaluation labels.
- No evaluator may inspect evaluation outcomes before all fits and all code hashes are finalized.

---

## 7. Primary endpoint

**Primary endpoint:** mean 15-bin ECE reduction over the three seeds on the held-out 5,000-image ImageNetV2 evaluation split.

For seed \(s\):

\[
\Delta_s = \operatorname{ECE}_{15}(\text{raw}_s) - \operatorname{ECE}_{15}(\text{calibrated}_s).
\]

The primary estimate is:

\[
\bar{\Delta} = \frac{1}{3}\sum_{s=1}^{3}\Delta_s.
\]

ECE definition:

- 15 equal-width bins on \([0,1]\);
- membership \(lo < confidence \le hi\);
- count-weighted;
- report raw ECE, calibrated ECE, each \(\Delta_s\), \(\bar{\Delta}\), absolute and relative reduction.

A positive \(\Delta\) means lower ECE after calibration.

### 7.1 Uncertainty

- Class-stratified paired percentile bootstrap.
- 10,000 resamples.
- Resample five images with replacement within each of the 1,000 classes, so every resample has five images per class.
- Use the same resampled indices for raw and calibrated predictions and for all three seeds.
- Compute the three-seed mean ECE reduction in each resample.
- Report the point estimate and 95% percentile interval \([\text{2.5th},\text{97.5th}]\).
- Fixed bootstrap RNG seed: `20261121`.

This replaces the historical unstratified image bootstrap. The change is required by the completed partition's five-per-class structure and is frozen before any outcome.

### 7.2 Acceptance rule

Accept H1 only if **all** hold:

1. \(\bar{\Delta}>0\).
2. The 95% class-stratified paired-bootstrap interval excludes zero.
3. No seed has an ECE increase greater than `0.005`.
4. Every raw and calibrated Top-1 prediction is identical; zero mismatches.
5. Every raw and calibrated Top-5 index set is identical; zero mismatches.
6. No seed has an NLL increase greater than `0.005`.
7. No seed has a Brier-score increase greater than `0.005`.
8. All three temperature fits report convergence success.
9. No temperature fit is at a bound.

The `0.005` margins are frozen engineering tolerances, not established scientific standards. They may not be relaxed after seeing outcomes.

If the interval includes zero, the result is **“calibration improvement not established,”** not “no effect.” If the point estimate is positive but small, report the effect size and interval; do not equate statistical detectability with practical importance.

---

## 8. Metrics and interpretation

All metrics are reported raw and calibrated, per seed and as an equal-weight three-seed mean, with the evaluation-set definition and count.

### 8.1 Classification

- Top-1 accuracy.
- Top-5 accuracy.
- Macro Top-1 accuracy.
- Per-class accuracy.
- Correct count.
- Exact Top-1 and Top-5 prediction agreement.

A non-tie change in Top-1 or Top-5 is an implementation failure, not a calibration result. Stop and diagnose.

### 8.2 Calibration

- 15-bin ECE (primary).
- Adaptive ECE with 15 equal-frequency bins; report bin counts.
- NLL.
- Multi-class Brier score.
- Mean confidence.
- Reliability diagram with 15 fixed bins, counts, and Wilson 95% intervals.
- Per-class confidence–accuracy gap and per-class ECE, descriptive only.

### 8.3 Selective prediction

- Risk–coverage curve from coverage 0.05 to 1.00.
- AURC.
- Selective risk at coverage 0.50, 0.80, 0.90, and 0.95.
- Abstention fraction.
- Correct-but-abstained fraction.
- Number of tied confidence values.

Single-scalar temperature scaling is strictly rank-preserving. Raw and calibrated curves should coincide except for numerical ties. Any material difference is investigated as an implementation error, not reported as a selective-prediction improvement.

### 8.4 Per-seed reporting

For each seed, report:

- fitted \(T_s\);
- NLL at \(T=1\);
- NLL at the optimum;
- iterations, tolerance, convergence, boundary;
- raw and calibrated ECE;
- \(\Delta_s\);
- raw and calibrated NLL and Brier;
- Top-1 and Top-5;
- calibration and evaluation logit-array hashes.

No seed is dropped or selected. The three-seed mean is an equal-weight summary, not a replacement for per-seed display.

---

## 9. Statistical interpretation

- **Global calibration** is the only confirmatory object. It is measured over 5,000 evaluation images and 1,000 classes.
- **Class-conditional results** are descriptive. Five images per class cannot support 1,000 confirmatory per-class claims.
- **Selective prediction** is a separate descriptive reliability outcome, not evidence that temperature scaling improves ranking.
- **Accuracy preservation** is an invariant and a guard, not a claim that calibration improves accuracy.
- **Distribution shift** is inherent: ImageNetV2 is a new test distribution, not a random sample from ImageNet-1k training.
- **Independence caveat:** the evaluation images were not used in Stage 0A, but the backbone's pretraining corpus is broad. No claim of absolute distributional independence is made.

---

## 10. Missing, invalid, and tied data

- Missing image, unreadable image, hash mismatch, wrong shape, duplicate index, or unexpected file: **abort**; no imputation.
- Non-finite logit or probability: **abort**; no clipping.
- Label out of range or length mismatch: **abort**.
- Calibration/evaluation overlap: **abort**.
- Failed checkpoint, backbone, config, or code hash: **abort**.
- Top-k ties: use descending logit, then ascending class index; compare exact index arrays.
- If calibration introduces a new tie or changes an ordering beyond the frozen tie policy: **stop**, treat as implementation failure.
- Failed optimizer convergence: **no confirmatory result**.
- Boundary temperature: report it; **no confirmatory acceptance** under this protocol.
- No optional stopping, no evaluation extension, no re-partitioning, no post-hoc threshold changes.

---

## 11. Execution boundary

This preregistration freezes the intended analysis. It does **not** authorize execution. Before launch:

1. Human protocol approval is recorded.
2. Data-use/supervisor confirmation is recorded where required.
3. All artifact hashes are re-verified against `STAGE0B_PROTOCOL_MANIFEST.json`.
4. The runtime fingerprint matches the frozen environment, or the deviation is explicitly approved and versioned.
5. The inference test and synthetic gates are re-run.
6. The evaluation split is not inspected.
7. The one-shot execution order is followed.

The evaluation split may be scored only after all three temperatures are fitted and all outputs are hashed. Once evaluation outcomes are seen, no protocol change is permitted in this version.

---

## 12. Explicit non-claims

This document does not claim:

- that Stage 0B has been run;
- that temperature scaling will improve calibration;
- that Stage 0A is fully image-level verified;
- that ImageNetV2 is free of near-duplicate overlap with ImageNet-1k;
- that the historical Stage 0A software environment has been exactly reconstructed;
- that any result is publishable, redistributable, or cleared for upload;
- that the separate Stage 0B launch approval has been granted.

**Stage 0B remains NOT LAUNCHED.**
