# 8 — DINOv2 assessment and integration readiness

## 8.1 What exists

- Checkpoint: `checkpoints/dinov2_vits14_pretrain.pth` (present).
- Backbone wrapper: `rhan_bi/perception/backbone.py`, class `DinoV2Backbone` (lines 36-96).
- `encode(image)` (lines 85-96) returns a `BackboneOutput` with:
  - `features = feats_h` (hand-crafted, line 92),
  - `meta = {"cls": cls, "patch": patch}` (line 94).
- `EvidenceBuilder.build` (`evidence.py:41-94`) consumes `features` only. It never reads `meta`.
- The path is optional and marked `# pragma: no cover`; no test exercises it. It is not part of any
  reported number.

## 8.2 What is missing

1. **No feature routing.** CLS and patch tokens are computed and discarded. The evidence model has
   never seen a DINOv2 feature.
2. **No training on real features.** `EvidenceModel` is fitted on renderer features. There is no
   calibration set of real images.
3. **No per-cause accuracy on real images.** The ambiguity band [0.55, 0.80] for defect/reflection
   has never been measured on real data. Q12 in `audit/06`.
4. **No resolution/patch-size study.** DINOv2 ViT-S/14 produces 16×16 patches at 224 px; the
   renderer's ROI geometry (≥ 25 px overlap in `_pair`, `evidence.py:96-127`) may not map to patch
   coordinates.
5. **No cost/latency measurement.** The planner's cost model is fixed (light 0.1, view 0.25, zoom
   0.2). Real encoding cost is unmeasured.
6. **No robustness study.** Lighting, viewpoint and occlusion in real inspection images differ from
   the renderer.

## 8.3 Is the architecture sound enough to connect a real encoder?

**Conditionally yes, for a bounded integration experiment only.** Reasons:

- The control loop is real and end to end (`evaluation/runner.py:22-118`); the belief update,
  quality scoring, planning and stopping are implemented and testable.
- The record/ledger data structures are sound enough to carry real features (immutable, hashed ids).
- The simulator gives a deterministic baseline against which a real-encoder arm can be compared.

But **not** for confirmatory or real-world claims, because:

- F1–F7 (order dependence, duplicate accumulation, tamper-evidence, Gate A leakage, Gate C scale,
  Gate B selection, DINOv2 non-use) must be fixed first; otherwise a real-encoder result would be
  measured by a broken ruler.
- The deployed model's defect/reflection accuracy (0.883) is outside the required ambiguity band,
  so the 80% gate is not currently calibrated to the intended difficulty.
- No real image has ever been processed.

## 8.4 Minimum integration experiment (before any claim)

1. Route DINOv2 CLS + mean-pooled patch tokens into `EvidenceBuilder` as features (replace or
   augment `feats_h`).
2. Build a small real-image calibration set (≥ 200 scenes, balanced causes, multiple views/lights)
   and fit `EvidenceModel` on it.
3. Report per-cause accuracy on a held-out real-image dev set with Wilson intervals; check whether it
   falls in [0.55, 0.80].
4. Run the existing Gates A–E on real features and compare against the synthetic baseline. Do not
   overwrite synthetic artifacts; write to a new `results/real_*` namespace.
5. Measure encoding latency and memory; update the cost model.
6. Only if (3) holds and (4) does not regress, consider a confirmatory test set.

## 8.5 What must not be claimed after integration

- That DINOv2 "works" because the synthetic gates pass (they are synthetic).
- That the 80% objective is met without a held-out real-image test.
- That the ledger is tamper-evident (F4).
- That the posterior is order-invariant (F1) or duplicate-bounded (F3).
- That Gate E shows policy equivalence (underpowered).
- Any novelty claim without a comparison to published active-perception baselines.
