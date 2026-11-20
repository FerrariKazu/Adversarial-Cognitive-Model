# RHAN Stage 0A — Execution Plan and Pre-Training Audit

**Protocol:** `rhan-stage0a-dinov2/1`

**Status:** implementation complete; full execution is gated on Kaggle T4 x2 and an attached ImageNet-1k dataset. No full Stage 0A training was started on this machine.

**Scope:** frozen DINOv2-S/14 and DINOv2-B/14, independently trained linear heads, three seeds, ImageNet-1k, official 50k validation, target top-1 ≥ 80%. No backbone training, no integrator, no CEE, no changes to previous experimental branches or results.

---

## 1. Exact consolidated reference files identified

| Role | Exact path | Why it matters |
|---|---|---|
| Kaggle consolidated training runner | `cloud_setup/Kaggle_consolidated.py` | Established Kaggle bootstrap, secret resolution, branch checkout, dedicated HF best/rolling split, per-epoch sync, restore-on-restart, and fail-fast training composition. |
| Colab consolidated training runner | `cloud_setup/colab_consolidated.py` | Same protocol and checkpoint format for Colab; documents the single-GPU counterpart of the Kaggle path. |
| Kaggle foundation ladder runner | `cloud_setup/Kaggle_J1_FOUNDATION.py` | Shows the dedicated-namespace partitioning and automatic mid-ladder resume convention used for foundation runs. |
| Colab foundation ladder runner | `cloud_setup/colab_j1_foundation.py` | Colab counterpart of the same foundation conventions. |
| Shared durable checkpoint primitive | `noesis_vision/core/checkpoint.py` | `atomic_torch_save` (.tmp → fsync → rename), SHA sidecars, and the “never silently restart” resume principle. |
| Existing multi-GPU convention | `phase2_attacks/shard_2gpu.py` | Confirms the repository’s T4-safe convention: independent single-GPU subprocesses, not DataParallel/DDP. |

`T4x2.ipynb` is a scratch notebook and is explicitly marked “never commit” in `docs/REPOSITORY_MAP.md`; it was not used as the canonical runner.

Stage 0A does **not** modify any of these files. It adds a separate `stage0a/` package and two new launchers.

---

## 2. How the existing resume and HF synchronization systems are reused

### 2.1 Reused semantics

- **Atomic writes:** Stage 0A uses `noesis_vision.core.checkpoint.atomic_torch_save` for every rolling/best/final checkpoint.
- **Never a silent restart:** a local rolling checkpoint is mandatory if present; if remote state cannot be verified, the run aborts instead of starting at epoch zero.
- **Dedicated remote partitions:** best/final artifacts and rolling state live in separate repositories, exactly as the consolidated runners do.
- **Automatic restart procedure:** re-open the notebook, run the preparation and dispatch cells, and the trainer discovers the remote rolling checkpoint and resumes.
- **Per-checkpoint sync:** rolling state and JSONL logs sync to the rolling repository; best/final artifacts sync to the best repository.

### 2.2 Stage 0A additions required by the assignment

The existing `save_rolling` primitive stores model/optimizer/scheduler/epoch, but not the full Stage 0A resume contract. Stage 0A adds:

- AMP GradScaler state.
- Python, NumPy, PyTorch CPU, and **all CUDA device** RNG states.
- `epoch`, `batch_in_epoch`, and `global_step`.
- Deterministic epoch permutation seed (`seed * 100003 + epoch`) and next-batch index, so the next data order is reproducible without storing a 1.28M-element permutation.
- Full resolved config and canonical experiment hash.
- Training seed.
- Backbone identity, timm name, and SHA-256 of the backbone state dict.
- Dataset fingerprint (relative path + file size + label manifest hash) and train/holdout split fingerprint.
- Training-code fingerprint (Stage 0A package + launchers), plus Git commit.
- Hardware and precision metadata.

### 2.3 Dedicated Hugging Face namespaces

- Best/final: `FerrariKazu/rhan-stage0a-dinov2-checkpoints`
- Rolling/logs/cache: `FerrariKazu/rhan-stage0a-dinov2-checkpoints-rolling`

These are new, isolated repositories. Nothing is written to `rhan-nxa-checkpoints`, `rhan-nxa-g2-checkpoints`, `rhan-checkpoints`, or any RHAN-IM/RHAN-Bi namespace.

Every checkpoint has a `.sha256` sidecar. Upload is synchronous and verified by remote size and, where the hub exposes an LFS SHA-256 ETag, by digest. A missing/unreadable sidecar is a hard resume failure, not a fresh start.

---

## 3. Proposed Stage 0A notebook structure

### Kaggle: `cloud_setup/Kaggle_STAGE0A.py`

1. Environment, dependency check, Kaggle secret resolution.
2. Clone `rhan-exp`, record exact Git SHA.
3. Verify two visible T4s, CUDA, VRAM, driver, and mounted storage.
4. Locate ImageNet-1k; refuse ImageNet-100 substitution.
5. Create/verify dedicated HF repositories.
6. Build deterministic train/val indices and the 50-per-class train holdout.
7. Run the checkpoint-interruption/resume drill.
8. Run the dual-T4 smoke benchmark; fail unless both GPUs participate.
9. Write the required audit report (`stage0a_audit.md` / `.json`) before any full extraction.
10. Extract pooled CLS features for both backbones, sharded across both GPUs.
11. Train six linear heads (2 backbones × 3 seeds), two independent single-GPU processes at a time; rolling/best sync after every checkpoint/evaluation.
12. Aggregate the three seeds and apply the pre-registered 80% rule.
13. Verify every final artifact and SHA sidecar is remotely present.

### Colab: `cloud_setup/colab_STAGE0A.py`

Same package, same config, same HF repositories, same checkpoint format. Differences:

- Single-GPU sharding.
- Dual-T4 gate is replaced by an explicit `--allow-single-gpu` warning; Colab results may not be described as dual-T4 results.

The two notebooks cannot silently diverge in training objective, split, evaluation, or checkpoint semantics because both call the same `stage0a.cli` commands.

---

## 4. Dataset availability and storage audit

### 4.1 This machine

Measured with the repository's loader and filesystem:

| Item | Value |
|---|---|
| Local ImageNet-1k | **Not present** |
| Local ImageNet-100 | Present at `data/imagenet100`, 14 GB |
| ImageNet-100 train | 126,689 images, 100 classes |
| ImageNet-100 val | 5,000 images, 100 classes |
| `/kaggle/input` | Not mounted on this machine |
| Free disk | ≈ 690 GiB |

The Stage 0A loader **refuses** to build an index from ImageNet-100: `ImageNet-1k requires 1000 train classes`. This is deliberate; the assignment is ImageNet-1k.

### 4.2 Kaggle requirement

Attach an ImageNet-1k dataset anywhere under `/kaggle/input/<dataset-slug>`; the notebook scans that mount instead of requiring one fixed slug. A root resolves when it contains one of:

- `<root>/train` + `<root>/val` (class-foldered), or
- `<root>/ILSVRC/Data/CLS-LOC/train` + `val`, or
- flat `val/` plus `LOC_val_solution.csv` at the root or an ancestor.

The notebook checks:

- 1,000 train class directories.
- 50,000 official validation images.
- Labels present for flat validation.

If the dataset is not found, the notebook aborts before training.

### 4.3 Storage strategy

Only the **pooled CLS feature** is cached. Full spatial patch-token maps are not persisted.

| Backbone | Train fp16 | Val fp16 |
|---|---:|---:|
| ViT-S/14 (384-d) | 938 MiB | 37 MiB |
| ViT-B/14 (768-d) | 1.83 GiB | 73 MiB |
| Both | **2.86 GiB** | — |

Head artifacts (six runs) are estimated at **≈ 53 MiB** including best/final/rolling and prediction files. Working peak for features + head artifacts is **≈ 2.91 GiB**, excluding the read-only ImageNet mount.

The dataset is never copied into `/kaggle/working`. The feature cache is uploaded to HF so a new session can skip extraction.

---

## 5. Dual-T4 hardware smoke-test results

**Not executed on this machine.** The local host has one RTX 4060 (8.6 GB) and 8.3 GB RAM, not Kaggle T4 x2. Reporting local numbers as dual-T4 results would be false.

What was executed locally:

- Both official timm DINOv2 backbones loaded and ran at 224×224.
- A single-GPU smoke path completed on a synthetic 1,000-class fixture.
- Per-GPU batch 16; peak VRAM 0.17 GB (ViT-S) and 0.49 GB (ViT-B); the run was intentionally short and is **not** a throughput claim.
- The Kaggle notebook's `smoke` cell requires two visible GPUs and aborts otherwise. It records both GPUs' utilization, VRAM, per-GPU and global batch size, images/s, dataloader wait, feature-cache write throughput, and head-training throughput.

Required Kaggle smoke report: `stage0a/smoke/smoke_report.json` with `both_gpus_participated = true`.

---

## 6. Verified checkpoint-resume test

Local state-machine drill command:

```bash
python3 -m stage0a.cli resume-drill \
  --data-root /tmp/nonexistent \
  --work-root /tmp/stage0a_drill \
  --no-hf
```

Result: **PASS**

- Phase 1 trained 7 steps of a 20-step synthetic run, wrote a complete rolling checkpoint, and exited `partial`.
- Phase 2 detected the local rolling checkpoint, restored model/optimizer/scheduler/scaler/RNG/sampler state, logged a `resume` event, and completed at global step 20.
- The final artifact was written and the official-eval path ran.
- No epoch-zero restart occurred.

Remote durability probe: a file was uploaded to the rolling repository, its SHA-256 sidecar verified, downloaded, and re-hashed successfully. The probe files were then deleted.

The real Kaggle resume test is the same mechanism after a notebook restart: the trainer calls `HFStore.restore_rolling`, validates config/code/dataset/split/backbone hashes, and refuses to continue on mismatch.

---

## 7. Resolved training configuration

| Item | ViT-S/14 | ViT-B/14 |
|---|---|---|
| timm checkpoint | `vit_small_patch14_dinov2.lvd142m` | `vit_base_patch14_dinov2.lvd142m` |
| Embedding dim | 384 | 768 |
| Backbone trainable params | 0 (frozen) | 0 (frozen) |
| Reference linear top-1 (external) | 81.1% | 84.5% |
| Input | 224×224, resize 256 + center crop | same |
| Feature | pooled CLS token, fp16 cache | same |
| Head | one linear layer, 1000 outputs | same |
| Optimizer | SGD, momentum 0.9, Nesterov | same |
| LR | 0.001 | same |
| Weight decay | 0.0 | same |
| Batch size | 1024 | same |
| Steps | 12,500 | same |
| Warmup | 500 steps | same |
| Schedule | cosine | same |
| Precision | AMP float16 | same |
| Seeds | 17, 42, 99 | 17, 42, 99 |
| Selection | 50/class holdout from train | same |
| Official test | 50k val, touched once | same |

Pre-registration file: `prereg/stage0a_decision_v1.json`.

Acceptance:

- Mean official-val top-1 across three seeds ≥ 0.80 for each backbone.
- No seed below 0.79.
- Report top-5, NLL, ECE (15 bins), and per-class accuracy.
- No tuning on the official validation split.

---

## 8. Estimated artifact requirements

| Artifact | Estimate |
|---|---:|
| Pooled train/val features, both backbones | 2.86 GiB |
| Six head best/final/rolling artifacts | ≈ 53 MiB |
| Predictions + per-class CSVs | < 6 MiB |
| Configs, manifests, logs, completion records | < 10 MiB |
| Working peak excl. read-only ImageNet mount | ≈ 2.91 GiB |

Remote repositories store:

- Rolling: rolling checkpoint, SHA sidecar, JSONL log, run manifest, feature cache.
- Best: best/final checkpoints, SHA sidecars, official metrics, predictions, per-class CSV, config snapshot, completion record.

---

## Verification performed locally

| Check | Result |
|---|---|
| `python3 -m py_compile stage0a/*.py cloud_setup/Kaggle_STAGE0A.py cloud_setup/colab_STAGE0A.py` | PASS |
| `python3 -m pytest tests/test_stage0a.py -q` | 4 passed |
| Frozen ViT-S/14 feature extraction on a 1,000-class synthetic fixture | PASS |
| Frozen ViT-B/14 forward path | PASS |
| 20-step linear-head training + official-eval path | PASS |
| Checkpoint interruption/resume drill | PASS (7 → 20 steps, resume logged) |
| HF upload → SHA sidecar → download verification | PASS |
| ImageNet-100 rejection by the ImageNet-1k loader | PASS (expected refusal) |
| Dual-T4 smoke | **NOT RUN — no second GPU** |
| Full ImageNet-1k training | **NOT RUN — blocked on attached dataset and dual-T4 gate** |

---

## Execution status

The implementation and local validation are complete. The remaining work is intentionally gated:

1. Run `cloud_setup/Kaggle_STAGE0A.py` on Kaggle T4 x2 with ImageNet-1k attached.
2. Inspect `stage0a_audit.md` and `stage0a_audit.json`; the status must be `ready_for_training`.
3. Only then allow the extraction and six-head training cells to proceed.
4. Aggregate and verify remote finalization.

No full training result is claimed here. No backbone was trained. No integrator or CEE code was added. Previous experimental branches and checkpoint repositories were not modified.
