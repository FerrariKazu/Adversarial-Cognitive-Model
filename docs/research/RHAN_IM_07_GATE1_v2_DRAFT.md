# 07 — Gate-1 Pre-Registration v2 (Single Stage-S3 Attempt)

Status: **DRAFT written on 16 Nov 2026 for review prior to freeze.**
Replaces: `prereg/rhan_im_gate1_v1.json` (commit `6592b9a`, published results `ab882d1`).
Binding mechanism: When approved, frozen via `prereg/rhan_im_gate1_v2.json` committed before any v2 run is executed.

---

## 1. Context and Motivation

Gate 1 v1 evaluated the frozen feedforward trunk `foundation_ff_112_seed17_final_d00fb750.pth` across three primary seeds (17, 42, 99) and two secondary seeds.
Results aggregated to **FAIL** (`runs/gate1_verdict.json`):
- `sdt_adequate`: **PASS** (3/3 runs, z-ROC slope 0.899–0.969 ∈ [0.80, 1.25], null sd 0.999–1.002 ∈ [0.85, 1.20])
- `informative`: **PASS** (3/3 runs, d_centre 2.871–2.877 ≥ 1.5, d_block 3.336–3.398 ≥ 2.0 with VIF_null 11.56–11.63 corrected)
- `looking_closer_pays`: **FAIL** (0/3 runs, native/gist ratio 1.063–1.113 < 1.50)

The root cause was that coarse whole-scene gist views produced detectability ($d' \approx 2.56\text{--}2.70$) nearly as high as native-resolution crops ($d' \approx 2.87$). Resolution provided negligible advantage over whole-scene resampling.

Per Doc 03 §9 and Doc 07 §7, the pre-registered protocol remedy for a failure on `looking_closer_pays` is **Stage S3 once**: fine-tune the top trunk blocks on dense window crops, early-stop on held-out native-over-gist ratio, discard the training head, and freeze a new Gate 1 v2.

---

## 2. Stage S3 Tuning Specification

| Parameter | Value | Rationale |
|---|---|---|
| Script | `scripts/rhan_im/train_s3_dense_window.py` | Pinned by git SHA-256 |
| Base Checkpoint | `foundation_ff_112_seed17_final_d00fb750.pth` | Seed 17 primary trunk (SHA-256: `d00fb750...`) |
| Trainable Parameters | `trunk.blocks[9:12]` (5,325,696) + `trunk.norm` (768) = **5,326,464** | Blocks 9, 10, 11 and final LayerNorm. Blocks 0..8 and patch_embed remain strictly frozen. |
| Train Mode Isolation | Blocks 0..8 and patch_embed in `eval()` mode; blocks 9..11 and norm in `train()` mode | Enforced via `set_trunk_train_mode()`. Prevents stochastic dropout/drop-path drift in frozen layers. |
| Co-training Head | `nn.Linear(384, 51)` | Used only to provide cross-entropy loss gradients during trunk updates. **Discarded at checkpoint save.** |
| Image Partition | `data/imagenet100/train/` partitioned into disjoint `s3_train` and `s3_heldout` (50 held-out images per class) | Documented and hash-pinned in `prereg/s3_split.json`. **Strictly disjoint from val/ (asserted).** |
| Training Budget | 5 epochs × 300 steps/epoch × batch 48 = **1,500 total steps** | ~72,000 crops trained. Cosine annealing schedule from lr 1e-5 (trunk) and 1e-3 (head). |
| Loss Function | $0.5 \cdot \mathcal{L}_{\text{native}} + 0.5 \cdot \mathcal{L}_{\text{gist}}$ | Equal weighting on 25 native candidate windows and resampled gist view. Native-only training is banned. |
| Early-Stopping Metric | Held-out native-over-gist ratio at operating cell (size 4, alpha 1.0) | $d_{\text{native\_centre}} / \max(d_{\text{gist}}, d_{\text{gist\_full}})$, evaluated with a freshly refitted 600-step head via Gate 1 estimators. |
| Evaluation Scenes | 1,000 calibration scenes + 1,500 test scenes generated once from fixed seeds (`seed + 100`, `seed + 200`) | Cached in `uint8` on CPU (~380 MB). Zero scene-sampling noise between baseline and epochs. |
| Baseline Measurement | Refit fresh 600-step head on untuned trunk before training; measure baseline ratio | Prevents noise from untrained random weights setting early-stopping target. |
| No-Improvement Rule | If no epoch beats the baseline ratio, return untuned trunk (`best_epoch = -1`) | Stored in output checkpoint metadata. |
| Output Checkpoint | `checkpoints/foundation_ff_112_s3_tuned.pth` | Follows standard format (`phase="ff_112_s3"`, `arm="s3_tuned"`, `backbone.*`). Head is discarded. |
| Single Attempt Boundary | S3 is run **exactly once** | If S3 fails to lift the ratio above 1.50 in Gate 1 v2, the protocol branch terminates; take the fallback (Gate 2 selective classification). |

### Evidence Boundary Note on Early Stopping
The held-out ratio measured during S3 on `data/imagenet100/train/` held-out scenes is an internal selection signal only and is **NOT evidence**. All train images were seen during foundation self-supervised pre-training. Confirmatory evidence is produced only by the Gate 1 v2 evaluation on the unseen validation test pool.

---

## 3. Gate 1 v2 Pre-Registration

### 3.1 Unchanged Elements from v1
All core protocol specifications from `docs/research/RHAN_IM_07_SUCCESSOR_DRAFT.md` remain identical word for word:
- **Class Split:** Pinned FINAL `prereg/class_split.json` (seed 20261113, SHA-256 `289f2818f0eae4dd7b654627d4bdf197cc11d43923300c52df7b4c0bdecc1241`).
- **Validation Pools:** Pinned FINAL `prereg/pools.json` (SHA-256 `095d5e9975e6bdbe82d4ba2720e66e6e7f8122bc4105d09605aa7f18299b729d`).
- **Operating Cell:** Size 4 (56 px), opacity 1.0, 30% null probability.
- **Head Refitting:** Linear(384, 51) fitted inside every run from scratch from `head_init_seed = seed` (600 steps, AdamW, lr 2e-3, cosine, batch 48).
- **Measurement Protocol:** 3,000 cal scenes, 6,000 test scenes, 2,000 paired looks.
- **Thresholds:**
  1. `sdt_adequate`: z-ROC slope ∈ [0.80, 1.25] and `null_sd` ∈ [0.85, 1.20]
  2. `informative`: $d_{\text{native\_centre}} \ge 1.5$ and $d_{\text{block}} \ge 2.0$ (with $VIF_{\text{null}}$ correction if $> 1.25$)
  3. `looking_closer_pays`: $d_{\text{native\_centre}} / \max(d_{\text{gist}}, d_{\text{gist\_full}}) \ge 1.50$
  4. `pass_all`: $1 \land 2 \land 3$
- **Aggregation:** 3 primary runs (seeds 17, 42, 99) on the primary checkpoint. PASS requires 2/3 runs passing all criteria.

### 3.2 What Changes in v2
- **Primary Checkpoint:** `checkpoints/foundation_ff_112_s3_tuned.pth` (replaces `foundation_ff_112_seed17_final_d00fb750.pth`).
- **Validation Reuse Note:** The 5,000 validation images in `prereg/pools.json` were used to evaluate Gate 1 v1. S3 training NEVER touches `val/` (verified and asserted). Re-evaluating on `val/` is permitted as a single pre-registered confirmation of S3 tuning under §9 switch conditions, but is noted as post-v1 evaluation.
- **Decision Consequences:**
  - If Gate 1 v2 **PASSES**: The tuned trunk restores the resolution advantage ($d_{\text{native}} / d_{\text{gist}} \ge 1.50$). Proceed to Gate 2 with adaptive looking policy claims.
  - If Gate 1 v2 **FAILS**: The hypothesis that "looking closer pays" is falsified for this trunk architecture. Do NOT attempt further trunk tuning. Conclude honestly and take the pre-registered fallback: Gate 2 scoped strictly to selective classification, calibration, and risk control without spatial resolution claims.
