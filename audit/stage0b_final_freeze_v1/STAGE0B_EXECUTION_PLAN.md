# Stage 0B — Execution Plan

**Protocol:** `rhan-stage0b-dinov2-b14-temperature/1` version `1.0.0`  
**Status:** **PLAN ONLY — NOT AUTHORIZED — NOT EXECUTED**  
**Real temperature fitted:** **NO**  
**Evaluation split inspected:** **NO**

---

## 1. Boundary

This plan describes what an approved Stage 0B run would do. It does not start the run. The run requires recorded human protocol approval and the launch checklist.

All new outputs would go under a new versioned directory, `audit/stage0b_run_v1/`. Historical Stage 0A/0B directories would remain read-only. No Stage 0A checkpoint, manifest, log, or report would be overwritten.

---

## 2. Host and storage

Recorded development host:

| Field | Value |
|---|---|
| Platform | WSL2, Ubuntu 22.04.5, kernel 6.6.87.2-microsoft-standard-WSL2 |
| CPU | Intel i5-12400F, 8 logical cores |
| RAM | 8,328,015,872 bytes (7.76 GiB) |
| GPU | NVIDIA GeForce RTX 4060, 8,585,216,000 bytes (8 GiB) |
| Driver / CUDA | 621.01 / 12.8 |
| Python | 3.10.12 |
| PyTorch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| timm | 1.0.27 |
| NumPy / SciPy | 2.2.6 / 1.15.3 |
| Free disk at freeze | ~687 GiB |

Before execution, re-record the actual runtime and compare it to `work/environment.json`. A software-version change requires explicit approval and a new protocol version. GPU model or driver differences must be recorded; whether they invalidate the run is a human decision, not an automatic “pass.”

Storage plan:

| Artifact | Approx. size |
|---|---:|
| 10,000 × 768 float16 features | 15.36 MB |
| 5,000 × 1,000 float32 logits per split per seed | 20 MB |
| Both splits, three seeds, raw logits | 120 MB |
| Calibrated probabilities, if materialized | 120 MB |
| Metrics, bootstrap samples, reports | < 100 MB |
| Working total | < 1 GB |

No Hugging Face storage is assumed or authorized. No raw images, features, logits, probabilities, or predictions may be uploaded while image-derived-artifact rights are unresolved.

---

## 3. Shared-backbone optimization

All three heads use the same frozen backbone and the same preprocessing. Therefore:

1. Extract one 768-D pooled CLS feature per image for the calibration split.
2. Extract one 768-D pooled CLS feature per image for the evaluation split.
3. Cache features as float16 with SHA-256 sidecars.
4. Apply each of the three float32 heads independently to the same features.
5. Fit one temperature per seed on the calibration logits only.

This is a compute-efficiency optimization, not a methodological change. It is equivalent to three independent backbone passes up to floating-point non-associativity. The calibration comparison uses the same features for raw and calibrated probabilities, so any shared numerical error cancels from the raw-versus-calibrated contrast.

The compute probe measured approximately **267 images/s** and **0.544 GiB** peak CUDA allocation for backbone-only inference at batch size 8. The 10,000-image backbone pass is therefore expected to take roughly **40 seconds** on the development host before I/O and serialization. This is an estimate, not a guarantee, and not a result.

---

## 4. Run sequence

### Phase 0 — Preflight, no data outcomes

1. Verify `STAGE0B_PROTOCOL_MANIFEST.json` and all referenced hashes.
2. Verify the three head hashes and sidecars.
3. Verify the backbone weights file and reconstructed state-dict hash.
4. Verify archive size/SHA-256 and the extracted dataset audit.
5. Verify the partition canonical and file hashes.
6. Verify the inference config, calibration core, evaluation core, and statistical config hashes.
7. Re-run the eight-image inference pipeline test.
8. Re-run the synthetic calibration and evaluation tests.
9. Confirm the evaluation split has not been read.
10. Confirm output directory is new and empty.
11. Confirm no HF/Kaggle upload path is enabled.

Any mismatch aborts. No “repair by substitution.”

### Phase 1 — Feature extraction

1. Read the frozen calibration and evaluation relative paths from the partition manifest.
2. Load each image with Pillow, convert RGB, apply the frozen 224 transform.
3. Run the frozen backbone in eval/inference mode, autocast float16, channels-last, batch size 8.
4. Store float16 features in the frozen row order; never shuffle.
5. Write SHA-256 sidecars and a completion record.
6. Verify every feature row is finite and every expected image appears exactly once.

No logits are fitted or scored in this phase.

### Phase 2 — Raw logits

1. Load the three `final.pt` heads.
2. Apply each head to the calibration features and evaluation features independently.
3. Store float32 logits `[5000,1000]` per seed per split with hashes.
4. Verify shapes, finiteness, and head hashes before and after.
5. Do not compute ECE, NLL, Brier, accuracy, or confidence summaries yet.

### Phase 3 — Temperature fitting, calibration only

1. Load only calibration logits and calibration labels.
2. Fit one temperature per seed with the frozen bounded scalar optimizer.
3. Record NLL at \(T=1\), optimum, iterations, tolerance, convergence, and boundary.
4. Write temperature records and their hashes.
5. Assert calibration/evaluation disjointness again.
6. Stop if any fit fails or is at a bound; report a technical failure, not a calibration result.

No evaluation label or evaluation logit may enter Phase 3.

### Phase 4 — Freeze fitted artifacts

1. Hash the three temperature records and raw logit arrays.
2. Write a `fit_complete.json` record binding protocol, data, model, and code hashes.
3. Only after this record exists may evaluation scoring begin.
4. Do not change any code, threshold, bin count, bootstrap setting, or partition after this point.

### Phase 5 — One-shot evaluation

1. Load evaluation logits and labels exactly once.
2. Compute raw and calibrated probabilities for all three seeds.
3. Compute Top-1/Top-5, NLL, Brier, ECE-15, adaptive ECE, reliability bins, mean confidence, and selective curves.
4. Verify Top-1 and Top-5 exact agreement.
5. Compute the nine decision checks.
6. Run the 10,000-resample class-stratified paired bootstrap of the three-seed mean ECE reduction.
7. Write per-seed, aggregate, and bootstrap outputs with hashes.

### Phase 6 — Reporting

1. Write the Stage 0B report with raw/calibrated tables, per-seed values, interval, guard checks, and decision.
2. Label class-conditional results descriptive.
3. Label the secondary benchmark non-independent, if used.
4. Record all limitations.
5. Do not edit the frozen protocol in place.

---

## 5. Determinism and ordering

- Feature row order: frozen partition order (`class_index`, `within_class_rank`).
- Batch size: 8.
- No shuffling.
- `torch.manual_seed(0)`.
- `cudnn.deterministic=True`, `cudnn.benchmark=False`.
- Inference mode, no gradients.
- Temperature fitting in float64; original logits preserved as float32.
- Bootstrap seed `20261121`.
- BLAS/thread settings recorded. If thread settings change bootstrap output, the configured thread setting is frozen with the run; the interval is not silently recomputed under a different setting.

A rerun must either reproduce the recorded hashes or produce a new run ID. It may not overwrite a completed run.

---

## 6. Output layout (future, not created)

```text
audit/stage0b_run_v1/
  config.json
  environment.json
  preflight.json
  features/
    calibration_features.npy
    calibration_features.npy.sha256
    evaluation_features.npy
    evaluation_features.npy.sha256
  logits/
    seed17_calibration_raw.npy
    seed17_evaluation_raw.npy
    seed42_calibration_raw.npy
    seed42_evaluation_raw.npy
    seed99_calibration_raw.npy
    seed99_evaluation_raw.npy
  temperatures/
    seed17.json
    seed42.json
    seed99.json
  fit_complete.json
  metrics/
    raw_eval_metrics.json
    calibrated_eval_metrics.json
    per_seed_deltas.json
    bootstrap_summary.json
    decision.json
  reports/
    STAGE0B_REPORT.md
  completion.json
  MANIFEST.sha256
```

All artifacts remain local under the unresolved data-use constraints.

---

## 7. Abort conditions

Abort immediately, write an incident record, and do not report a calibration result if:

- any frozen hash differs;
- any head or backbone fails to load;
- any image is missing, unreadable, duplicated, or hash-mismatched;
- calibration/evaluation overlap is nonzero;
- any tensor is non-finite or wrongly shaped;
- the evaluation split is opened before `fit_complete.json`;
- any temperature fit fails or is at a bound;
- Top-1 or Top-5 changes beyond the frozen tie policy;
- raw and calibrated selective curves differ materially;
- a code or config hash changes after fitting;
- an unexpected file appears in the frozen split;
- an upload or external write is attempted.

No threshold relaxation, no seed substitution, no re-partitioning, no “quick look” at evaluation outcomes.

---

## 8. Post-run boundary

A completed run is a **Stage 0B result** only if all of the following are bound:

- protocol version and manifest hash;
- the three head hashes;
- backbone revision and state-dict hash;
- ImageNetV2 archive and partition hashes;
- inference configuration hash;
- calibration and evaluation implementation hashes;
- runtime fingerprint;
- statistical configuration hash;
- per-seed temperatures and artifact hashes;
- the one-shot evaluation record.

Otherwise it is an exploratory run and must be labelled as such.

**Stage 0B remains NOT LAUNCHED.**
