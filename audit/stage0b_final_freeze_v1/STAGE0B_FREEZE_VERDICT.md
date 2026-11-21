# Stage 0B — Freeze Verdict

**Verdict:** **READY FOR HUMAN PROTOCOL FREEZE**  
**Execution status:** **NOT AUTHORIZED — NOT LAUNCHED**  
**Real temperature fitted:** **NO**  
**Evaluation outcomes inspected:** **NO**  
**Protocol:** `rhan-stage0b-dinov2-b14-temperature/1`, version `1.0.0`

---

## 1. Decision

The Stage 0B protocol is sufficiently specified, data-bound, model-bound, code-bound, and statistically frozen for a human to review and either approve or reject the freeze.

This is **not** a READY-TO-RUN verdict. It does not authorize:

- fitting a temperature;
- scoring the evaluation split;
- publishing any result;
- uploading any image or image-derived artifact.

A separate explicit execution approval is required after human protocol approval and closure of the launch checklist.

---

## 2. Why the verdict is READY rather than CONDITIONAL

The technical and methodological gates that can be closed without running Stage 0B are closed:

| Gate | Verdict | Evidence |
|---|---|---|
| Dataset identity and archive digest | PASS | `STAGE0B_DATA_PROVENANCE.md`; pinned revision and SHA-256 |
| Archive integrity and extraction | PASS | 11,001 members, zero size mismatches; uncompressed-tar caveat documented |
| Real dataset audit | PASS | 10,000 images, 1,000 classes, 10/class, no exact duplicates |
| Class mapping | PASS | All 1,000 numeric folders mapped to canonical WNIDs and Stage 0A order |
| Deterministic partition | PASS | 5,000/5,000, 5/class, zero overlap, recomputed twice |
| Three Stage 0A heads | PASS | All `final.pt` hashes and sidecars match |
| Backbone pin | PASS | HF revision, weights-file hash, and state-dict hash match |
| Raw-image inference path | PASS | Eight-image test; transform, 768-D features, three 1,000-way heads, finite logits |
| Synthetic calibration gate | PASS | 17/17 |
| Synthetic evaluation gate | PASS | 7/7 |
| Compute/memory plan | PASS | Shared features once; ~267 images/s; 0.544 GiB peak in probe |
| Statistical specification | PASS | Primary endpoint, guards, bootstrap, ties, missing data, and decision rule frozen |
| Artifact binding | PASS | `STAGE0B_PROTOCOL_MANIFEST.json` with file hashes and payload hash |
| Evaluation leakage prevention | PASS by design | Evaluation split not opened; fitting restricted to calibration |

The remaining items are **launch prerequisites**, not unresolved protocol defects:

- human protocol-freeze approval;
- supervisor/institutional data-use confirmation;
- separate execution approval;
- runtime re-verification on the execution host;
- no upload rights for image-derived artifacts.

Those are explicitly tracked in `STAGE0B_UNRESOLVED_BLOCKERS.md` and the sign-off block in `STAGE0B_LAUNCH_CHECKLIST.md`.

---

## 3. Accepted limitations

These limitations do not block freeze, but must travel with every Stage 0B report:

1. **Historical software is not exactly reconstructible.** Stage 0A did not record its training-time `timm`/`torchvision` versions. Stage 0B is a separately pinned inference configuration, not an exact software replay. The weights-file hash, state-dict hash, transform source, transform tensor output, and head contract are verified.
2. **Stage 0A image-level replay remains blocked.** Stage 0B's pipeline test does not repair Stage 0A's missing-image replay.
3. **Five images per class.** Class-conditional Stage 0B results are descriptive only.
4. **Distribution shift.** ImageNetV2 is a new test distribution, not a random sample from ImageNet-1k training.
5. **Near-duplicate overlap was not independently reconstructed.** Exact duplicates were checked; the authors' near-duplicate screening is inherited.
6. **No Stage 0B outcome exists.** This freeze is prospective.

If a reviewer judges the historical-software limitation or the inherited near-duplicate screening to be disqualifying, the correct disposition becomes **CONDITIONAL — SPECIFIC ITEMS REQUIRE DECISION**, not a silent PASS. The evidence for that decision is documented rather than hidden.

---

## 4. What is frozen

- Design A: ImageNetV2 MatchedFrequency only.
- Content-hash 5,000/5,000 partition, seed `20261121`.
- Three ViT-B/14 `final.pt` heads, seeds 17/42/99.
- One positive scalar temperature per seed.
- Calibration-only fitting; evaluation-only scoring.
- 15-bin ECE primary endpoint.
- Top-1/Top-5 exact invariance.
- NLL and Brier guard margins `0.005`.
- Class-stratified paired bootstrap, 10,000 resamples, seed `20261121`.
- Nine-condition acceptance rule.
- Descriptive class-conditional and selective-prediction analyses.
- No optional stopping, no re-partitioning, no seed selection.

See `STAGE0B_STATISTICAL_ANALYSIS_PLAN.md` and `work/statistical_config.json`.

---

## 5. What must happen next

1. Human reviewer records approval or rejection of the protocol freeze.
2. Data-use supervisor/institution records confirmation where required.
3. Execution approver records separate authorization.
4. An approved operator re-verifies the protocol manifest and runtime.
5. The launch checklist is completed and signed.
6. Only then may Stage 0B fit temperatures.

Until then, **Stage 0B remains NOT LAUNCHED.**
