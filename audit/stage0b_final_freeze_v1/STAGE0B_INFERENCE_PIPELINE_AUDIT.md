# Stage 0B — Inference Pipeline Audit

**Protocol:** `rhan-stage0b-dinov2-b14-temperature/1` version `1.0.0`  
**Status:** **PASS for the separately pinned Stage 0B inference configuration**  
**Stage 0B execution:** **NOT AUTHORIZED**  
**Real temperature fitted:** **NO**  
**Evaluation images used:** **0**

---

## 1. Why this audit exists

The Stage 0A closeout reproduced metrics from **verified feature caches and final heads**, but it did not independently reproduce the entire raw-image-to-prediction path. Stage 0B uses actual ImageNetV2 images, so the complete path must be pinned before freezing.

This audit reconstructs the path from:

- the Stage 0A source at training commit `a23dc0f76f5ba2514a3c6a42035d28b44a4ae17e`;
- the Stage 0A configuration and run manifests;
- the three final Stage 0A ViT-B/14 heads;
- the pinned DINOv2 ViT-B/14 weights at HF revision `4685c99dabffe5affac90bd99dbffd25801ae58d`;
- eight fixed ImageNetV2 calibration images.

It does not fit a temperature, score the evaluation split, or compute calibration metrics.

---

## 2. Frozen path

```text
ImageNetV2 JPEG
  -> Pillow Image.open(...).convert("RGB")
  -> Resize(shorter side = 256, bicubic, antialias=True)
  -> CenterCrop(224)
  -> ToTensor()                         # float32, [0,1]
  -> Normalize(ImageNet mean/std)
  -> float32 tensor [3,224,224]
  -> frozen DINOv2 ViT-B/14
       patch 14, width 768, depth 12, 12 heads
       forward_features -> final norm -> CLS token
  -> float16 feature cache [N,768]
  -> float32 head input
  -> Linear(768 -> 1000) fc.weight/fc.bias
  -> float32 logits [N,1000]
```

The backbone is shared. For each split, features are extracted once and passed independently through seeds 17, 42, and 99. This is a compute optimization and does not change the model, data, or statistical design.

---

## 3. Thirteen required checks

| # | Item | Frozen result | Evidence |
|---:|---|---|---|
| 1 | Image loading and RGB conversion | Context-managed `Image.open`, then `convert("RGB")` before transform | `stage0a/data.py::IndexedImageDataset.__getitem__`; file SHA-256 `63cb07b5…`; matches training commit |
| 2 | Input resolution | 224 × 224 | Stage 0A config `img_size=224`; inference config |
| 3 | Resize procedure | Shorter side to 256, aspect ratio preserved | `make_transform(224)` computes `round(224*256/224)=256`; no stretch-to-square |
| 4 | Center crop | 224 × 224, centered, no padding | `torchvision.transforms.CenterCrop(224)` |
| 5 | Interpolation | Bicubic, `antialias=True` | Effective torchvision 0.24.1 default; explicitly restated in test |
| 6 | Pixel normalization | mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)` | Stage 0A config and source |
| 7 | DINOv2 revision | `4685c99dabffe5affac90bd99dbffd25801ae58d` | Pinned HF model revision; downloaded metadata |
| 8 | Backbone architecture | DINOv2 ViT-B/14, patch 14, width 768, depth 12, 12 heads, MLP ratio 4, `num_classes=0`, `dynamic_img_size=True` | Runtime introspection; state-dict hash |
| 9 | Feature extraction layer | `VisionTransformer.forward_features` output after final LayerNorm | Runtime equality of `model(x)` and `forward_head(forward_features(x), pre_logits=True)` |
| 10 | CLS/pooling | `global_pool='token'`; CLS token only | Runtime introspection; feature shape `[8,768]` |
| 11 | Feature dimensions | 768; cached float16; upcast float32 for head | Test output |
| 12 | Head loading | `final.pt['model']` has exactly `fc.weight [1000,768]` and `fc.bias [1000]`; float32 | All three heads loaded and hashed |
| 13 | Final logits | `features_float32 @ fc.weight.T + fc.bias`, float32 `[N,1000]`, finite | Test output |

No item is marked PASS on documentary evidence alone. The executable test covers items 1–13 on the fixed sample.

---

## 4. Hash verification

### 4.1 Backbone

| Check | Expected | Observed |
|---|---|---|
| Weights file SHA-256 | `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b` | match |
| Runtime state-dict SHA-256 | `375b4c89151489e61df6f1f3b3fe1f9fa8b815c52ae0f7117b23e2e201fbb4fe` | match |
| HF revision | `4685c99dabffe5affac90bd99dbffd25801ae58d` | match |
| Positional embedding at runtime | `[1,257,768]` | match |

The weights file hash and state-dict hash are different objects: one hashes the serialized `.safetensors` bytes, the other hashes the ordered model tensors. Both are recorded.

### 4.2 Heads

| Seed | `final.pt` SHA-256 | Sidecar | Keys | Shapes |
|---:|---|---|---|---|
| 17 | `320f79a7…308e` | match | `fc.weight`, `fc.bias` | `[1000,768]`, `[1000]` |
| 42 | `f5e7d42d…c357` | match | `fc.weight`, `fc.bias` | `[1000,768]`, `[1000]` |
| 99 | `cb91a2b1…452d` | match | `fc.weight`, `fc.bias` | `[1000,768]`, `[1000]` |

No `best.pt` is used.

---

## 5. Deterministic pipeline test

**Script:** `audit/stage0b_final_freeze_v1/scripts/verify_inference_pipeline.py`  
**Output:** `work/inference_pipeline_test.json`  
**Status:** **PASS**

Sample selection:

- eight calibration images only;
- one image per class for class indices 0–7;
- each is the `within_class_rank=0` image from the frozen partition;
- no evaluation image is opened;
- no labels are scored.

Results:

| Check | Result |
|---|---|
| Sample images | 8 |
| Images decode as RGB | 8/8 |
| Documented transform equals `stage0a.data.make_transform(224)` | **exact tensor equality** |
| Transformed shape | `[3,224,224]` |
| Transformed dtype | float32 |
| Backbone feature shape | `[8,768]` |
| Feature dtype | float16 |
| `model(x)` equals explicit `forward_head(forward_features(x))` | **exact equality** |
| Heads evaluated | 3 |
| Logits shape per head | `[8,1000]` |
| Logits dtype | float32 |
| All logits finite | yes |
| Evaluation images used | 0 |
| Temperature fitted | no |
| Calibration metrics computed | no |

Per-image content hashes and per-head logit hashes are in `work/inference_pipeline_test.json`.

---

## 6. Runtime record

The test ran with:

- Python `3.10.12`
- torch `2.9.1+cu128`
- torchvision `0.24.1+cu128`
- timm `1.0.27`
- NumPy `2.2.6`
- Pillow `12.2.0`
- CUDA `12.8`, cuDNN `91002`
- NVIDIA GeForce RTX 4060, compute capability 8.9, 8,585,216,000 bytes
- `torch.backends.cudnn.deterministic=True`, `benchmark=False`
- inference mode, autocast float16, channels-last, batch size 8

The environment fingerprint is recorded in `work/environment.json`.

---

## 7. Compute probe

**Script:** `audit/stage0b_final_freeze_v1/scripts/probe_inference_cost.py`  
**Output:** `work/compute_probe.json`  
**Scope:** calibration images only; backbone inference only; no logits, labels, temperatures, or evaluation images.

| Measurement | Value |
|---|---:|
| Batch size | 8 |
| Images timed | 56 (after one warm-up batch) |
| Throughput | ~267 images/s |
| Peak CUDA allocated | 0.544 GiB |
| Extrapolated 10,000-image backbone extraction | ~37.4 s |

This is a development-host observation. It does not include disk I/O, feature serialization, head evaluation, bootstrap, or reporting. It is not a Stage 0B result.

The shared-feature plan is valid: one 768-D feature vector per image is consumed by all three independent heads. There is no methodological reason to run the backbone three times.

---

## 8. Historical reconstruction limitation

The Stage 0A **training-time** versions of `timm` and `torchvision` were not recorded. The current environment reproduces:

- the pinned weights file hash;
- the exact Stage 0A state-dict hash;
- the exact Stage 0A transform source and tensor output on the test images;
- the exact head contract and logits shape.

It does **not** prove that the historical training-time software produced byte-identical features or predictions. Stage 0B is therefore defined as a **separately pinned inference configuration**, not as an exact software reconstruction of Stage 0A.

This limitation does not weaken the Stage 0B calibration comparison: raw and calibrated logits are produced by the same frozen Stage 0B pipeline. It does mean Stage 0B cannot be used to claim exact Stage 0A image-level replay.

---

## 9. Verdict

**Inference pipeline: PASS for the frozen Stage 0B configuration.**

The path is executable, hash-bound, dimensionally correct, and tested on real ImageNetV2 images without touching the evaluation split. Stage 0B remains **not launched**.
