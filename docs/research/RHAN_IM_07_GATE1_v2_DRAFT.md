# 07 — Gate-1 Pre-Registration v2 (Single Stage-S3 Attempt)

Status: **DRAFT written on 16 Nov 2026 for review prior to freeze.**
Replaces: `prereg/rhan_im_gate1_v1.json` (commit `6592b9a`, published results `ab882d1`).
Binding mechanism: When approved, frozen via `prereg/rhan_im_gate1_v2.json` committed before any v2 run is executed.

---

## 1. Context and Motivation

Gate 1 v1 evaluated the frozen feedforward trunk `foundation_ff_112_seed17_final_d00fb750.pth` across three primary seeds (17, 42, 99) and two secondary seeds.
Results aggregated to **FAIL** (`runs/gate1_verdict.json`):
- `sdt_adequate`: **PASS** (3/3 runs, z-ROC slope 0.899–0.969 ∈ [0.80, 1.25], null sd 0.993–1.013 ∈ [0.85, 1.20])
- `informative`: **PASS** (3/3 runs, d_centre 2.844–2.888 ≥ 1.5, d_block 3.336–3.398 ≥ 2.0 with VIF_null 11.56–11.63 corrected)
- `looking_closer_pays`: **FAIL** (0/3 runs, native/gist ratio 1.063–1.113 < 1.50)

The root cause was that coarse whole-scene gist views produced detectability ($d' \approx 2.56\text{--}2.70$) nearly as high as native-resolution crops ($d' \approx 2.87$). Resolution provided negligible advantage over whole-scene resampling.

Per Doc 04 §10–11 and Doc 07 verdict table (`runs/gate1_verdict.json` → `next_step`), the pre-registered protocol remedy for a failure on `looking_closer_pays` is **Stage S3 once**: fine-tune the top trunk blocks on dense window crops, early-stop on held-out native-over-gist ratio, discard the training head, and freeze a new Gate 1 v2. The 448-px canvas test specified in Doc 03 §9 is **waived** for v2, since S3 is the primary architectural remedy on the standard 224-px canvas; if S3 fails, the 448-px path remains open as an independent future investigation.

---

## 2. Stage S3 Tuning Specification

| Parameter | Value | Rationale |
|---|---|---|
| Script | `scripts/rhan_im/train_s3_dense_window.py` | Pinned by git SHA-256 |
| Base Checkpoint | `foundation_ff_112_seed17_final_d00fb750.pth` | Seed 17 primary trunk (SHA-256: `d00fb750...`) |
| Trainable Parameters | `trunk.blocks[9:12]` (5,325,696) + `trunk.norm` (768) = **5,326,464** | Blocks 9, 10, 11 and final LayerNorm. Blocks 0..8 and patch_embed remain strictly frozen. |
| Train Mode Isolation | Blocks 0..8 and patch_embed in `eval()` mode; blocks 9..11 and norm in `train()` mode | Enforced via `set_trunk_train_mode()`. Prevents stochastic dropout/drop-path drift in frozen layers. |
| Co-training Head | `nn.Linear(384, 51)` warm-started from baseline head | Used only to provide cross-entropy loss gradients during trunk updates. Warm-started from the baseline head fitted on the untuned trunk. **Discarded at checkpoint save.** |
| Image Partition | `data/imagenet100/train/` partitioned into disjoint `s3_train` and `s3_heldout` (50 held-out images per class) | Documented and hash-pinned in `prereg/s3_split.json`. **Isolation guarantee: S3 exclusively reads from data/imagenet100/train/ (val/ is never accessed).** Verified against disk per-class counts. |
| Training Budget | 5 epochs × 300 steps/epoch × batch 48 = **1,500 total steps** | ~72,000 crops trained. Cosine annealing schedule from lr 1e-5 (trunk) and 1e-3 (head). |
| Loss Function | $0.5 \cdot \mathcal{L}_{\text{native}} + 0.5 \cdot \mathcal{L}_{\text{gist}}$ | Equal weighting on 25 native candidate windows and resampled gist view. Native-only training is banned. |
| Early-Stopping Metric | Held-out native-over-gist ratio at operating cell (size 4, alpha 1.0) | $d_{\text{native\_centre}} / \max(d_{\text{gist}}, d_{\text{gist\_full}})$, evaluated with a freshly refitted 600-step head via Gate 1 estimators. |
| Evaluation Scenes | 1,000 calibration scenes + 1,500 test scenes generated once from fixed seeds (`seed + 100`, `seed + 200`) | Cached in `uint8` on CPU (~380 MB). Zero scene-sampling noise between baseline and epochs. |
| Baseline Measurement | Refit fresh 600-step head on untuned trunk before training; measure baseline ratio | Prevents noise from untrained random weights setting early-stopping target. Non-finite baseline aborts run immediately. |
| No-Improvement Rule | If no epoch beats the baseline ratio, return untuned trunk (`best_epoch = -1`) | Stored in output checkpoint metadata. If `best_epoch = -1`, S3 made no gain: Gate 1 v2 runs are not executed and the fallback is taken immediately. |
| Output Checkpoint | `checkpoints/foundation_ff_112_s3_tuned.pth` | Follows standard format (`phase="ff_112_s3"`, `arm="s3_tuned"`, `backbone.*`). Head is discarded. S3 log (`.json`) and checkpoint SHA-256 are published regardless of outcome. |
| Single Attempt Boundary | S3 is run **exactly once** | If S3 fails to lift the ratio above 1.50 in Gate 1 v2, the protocol branch terminates; take the fallback (Gate 2 selective classification). |

### Evidence Boundary Note on Early Stopping
The held-out ratio measured during S3 on `data/imagenet100/train/` held-out scenes is an internal selection signal only and is **NOT evidence**. All train images were seen during the supervised TRADES fine-tuning of the foundation backbone. The S3 held-out ratio is not comparable to v1's 1.063–1.113 (measured on validation images that the foundation was never trained on). Confirmatory evidence is produced only by the Gate 1 v2 evaluation on the validation test pool, which is held out from S3 training but was used to evaluate Gate 1 v1 and was used for monitoring and abort gates per OI-13. It is not "unseen."

### Exploratory Observation (Post-v1)
After v1, an exploratory check on size-2 targets (28 px) showed a native/gist ratio of approximately 2.0, suggesting that resolution advantage exists at smaller target scales. However, Stage S3 was specified by the pre-registered protocol (§9 switch conditions) before the size-2 check occurred; the exploratory size-2 observation played no role in choosing S3 and provides no confirmatory evidence. The **operating cell remains locked at size 4 (56 px), opacity 1.0** for all v2 confirmatory evaluation.

### Trunk Seed Dependence
S3 fine-tunes seed-17 trunk only. Trunk seed dependence (whether seeds 42 and 99 produce similar ratios) is **not assessed** in v2. The three v2 evaluation seeds (17, 42, 99) vary only the head initialisation and scene sampling, not the trunk weights.

### Freeze Tool Extension
The freeze tool extension is completed: `scripts/rhan_im/gate1_prereg.py freeze --protocol-version rhan_im_gate1_v2` requires and pins the SHA-256 hashes of the S3 training script, the `s3_split.json` partition manifest, and the base checkpoint, and asserts that the tuned checkpoint's embedded `base_checkpoint_sha256` matches.

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
- **Validation Reuse Note:** The 5,000 validation images in `prereg/pools.json` produced the v1 result and were used for monitoring and abort gates (OI-13). S3 training NEVER touches `val/` (verified and asserted). Re-evaluating on `val/` is permitted as a single pre-registered confirmation of S3 tuning under the Doc 04 §10–11 switch conditions. These images are "held out from S3 training" but are **not unseen**; this is a single-attempt evaluation. If it fails, take the fallback.
- **Decision Consequences:**
  - If Gate 1 v2 **PASSES**: The tuned trunk restores the resolution advantage ($d_{\text{native}} / d_{\text{gist}} \ge 1.50$). Proceed to Gate 2 with adaptive looking policy claims.
  - If Gate 1 v2 **FAILS**: The hypothesis that "looking closer pays" is **not supported at the pre-registered cell (size 4, 224 canvas) after S3; other target sizes and 448 px remain untested**. Do NOT attempt further trunk tuning on this canvas. Conclude honestly and take the pre-registered fallback: Gate 2 scoped strictly to selective classification, calibration, and risk control without spatial resolution claims.
  - **No-Improvement Guard:** If S3 finishes with `best_epoch = -1` (no epoch improved over baseline), the tuned checkpoint is identical to the untuned trunk. Gate 1 v2 primary runs are NOT executed (as they would simply duplicate v1), and the protocol transitions immediately to the Gate 2 fallback.

### 3.3 Historical v1 Reproducibility Note
`scripts/rhan_im/gate1_prereg.py` changed when adding support for the v2 freeze and S3 provenance (hash changed from `cd5c98c6…` to `681d9691…`). Because v1's frozen pre-registration (`prereg/rhan_im_gate1_v1.json`) pinned `gate1_prereg.py` at `cd5c98c6…`, running `gate1_prereg.py aggregate` on historical v1 runs with the updated script would report a code hash deviation (`INVALID`). Historical Gate 1 v1 runs reproduce strictly at commit `ab882d1` or `3a03de2`.

