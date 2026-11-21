# Stage 0B — Launch Checklist

**Protocol:** `rhan-stage0b-dinov2-b14-temperature/1` version `1.0.0`  
**Status:** **UNCHECKED — STAGE 0B NOT AUTHORIZED**  
**Use:** Complete only after human protocol approval. Checking this list does not authorize execution by itself.

---

## A. Human and institutional gates

- [ ] Human reviewer has read `STAGE0B_FINAL_PREREGISTRATION.md`, `STAGE0B_STATISTICAL_ANALYSIS_PLAN.md`, and `STAGE0B_INFERENCE_PIPELINE_AUDIT.md`.
- [ ] Human reviewer records explicit **PROTOCOL FREEZE APPROVAL** with name/date/reference.
- [ ] Design A is approved; Design B is not silently substituted.
- [ ] Data-use review is accepted as local, non-commercial educational/research use only.
- [ ] Supervisor/institutional confirmation is recorded where required.
- [ ] No redistribution, publication, or upload of images or image-derived artifacts is authorized.
- [ ] A separate **EXECUTION APPROVAL** is recorded. Protocol freeze approval alone is not execution approval.

---

## B. Artifact and hash gates

- [ ] `STAGE0B_PROTOCOL_MANIFEST.json` parses.
- [ ] `MANIFEST.sha256` verifies every listed file.
- [ ] All three `final.pt` SHA-256 values match the manifest.
- [ ] All three `final.pt.sha256` sidecars match.
- [ ] Backbone weights file SHA-256 matches `55cbb5d8…8c7b`.
- [ ] Runtime backbone state-dict SHA-256 matches `375b4c89…b4fe`.
- [ ] ImageNetV2 archive size is `1,264,079,360` bytes.
- [ ] ImageNetV2 archive SHA-256 is `f0c37fdf…ca7c`.
- [ ] Dataset audit is `PASS` and its artifact hash matches.
- [ ] Class-mapping verification is `PASS` and its artifact hash matches.
- [ ] Partition canonical hash is `557e10f1…2578`.
- [ ] Partition JSON file hash is `fa7f2303…35e6`.
- [ ] Calibration count is 5,000 with 5/class.
- [ ] Evaluation count is 5,000 with 5/class.
- [ ] Partition overlap is zero.
- [ ] Inference config hash matches.
- [ ] Calibration implementation hash matches.
- [ ] Evaluation implementation hash matches.
- [ ] Statistical config hash matches.

Any mismatch stops the launch. Do not repair by editing the manifest or substituting a similar file.

---

## C. Environment gate

- [ ] Actual host fingerprint is recorded.
- [ ] Python is `3.10.12`.
- [ ] torch is `2.9.1+cu128`.
- [ ] torchvision is `0.24.1+cu128`.
- [ ] timm is `1.0.27`.
- [ ] NumPy is `2.2.6`.
- [ ] SciPy is `1.15.3`.
- [ ] Pillow is `12.2.0`.
- [ ] CUDA is `12.8`.
- [ ] GPU is visible and has at least the recorded 8 GiB memory class; any difference is documented.
- [ ] At least 5 GiB free disk is available beyond the expected <1 GB working set.
- [ ] No unexplained process is consuming GPU memory.
- [ ] BLAS/thread settings are recorded.

A software-version change requires explicit approval and a new protocol version, not an unrecorded “close enough.”

---

## D. Data gate

- [ ] Dataset root is the frozen extracted root.
- [ ] All 10,000 images are present; no extras.
- [ ] No symlinks or non-regular files in the split.
- [ ] Calibration and evaluation paths are read from the partition manifest, not re-derived.
- [ ] No image is added, removed, re-encoded, or replaced.
- [ ] Evaluation images have not been opened in this run.
- [ ] Evaluation labels have not been loaded in this run.
- [ ] Official-validation data, if used, is only the pre-existing hash-verified secondary cache and is labelled non-independent.

---

## E. Pipeline gates

- [ ] Eight-image inference pipeline test is re-run and `PASS`.
- [ ] Documented transform equals Stage 0A `make_transform(224)` exactly.
- [ ] Backbone feature shape is `[N,768]`.
- [ ] All three heads produce `[N,1000]` float32 logits.
- [ ] All logits are finite.
- [ ] Synthetic calibration test is re-run and `PASS` 17/17.
- [ ] Synthetic evaluation-core test is re-run and `PASS` 7/7.
- [ ] No real temperature has been fitted during preflight.
- [ ] No calibration or evaluation metric has been computed during preflight.

---

## F. Execution-order gates

- [ ] Output directory is new and empty.
- [ ] Feature extraction produces both splits once, with hashes.
- [ ] Raw logits are produced for three seeds and hashed.
- [ ] Temperature fitting reads only calibration logits/labels.
- [ ] `fit_complete.json` is written and hashed.
- [ ] Only after `fit_complete.json` may evaluation logits be scored.
- [ ] Evaluation is one-shot; no repeated “peeking” runs.
- [ ] Bootstrap seed is `20261121`.
- [ ] No optional stopping, no re-partitioning, no threshold change.
- [ ] No seed is selected or dropped.
- [ ] No best-seed claim.
- [ ] Per-seed and three-seed mean results are both produced.

---

## G. Output and publication gates

- [ ] All outputs remain local.
- [ ] No raw images uploaded.
- [ ] No features uploaded.
- [ ] No logits or probabilities uploaded.
- [ ] No predictions uploaded.
- [ ] No public HF dataset repository created.
- [ ] Report labels class-conditional results descriptive.
- [ ] Report labels secondary benchmark non-independent.
- [ ] Report states the five-per-class limitation.
- [ ] Report states the separately pinned Stage 0B software limitation.
- [ ] Report does not claim Stage 0A exact image-level replay.
- [ ] Report does not claim adaptive perception or general robustness.

---

## H. Stop conditions

Stop and do not report a Stage 0B result if:

- [ ] any hash fails;
- [ ] any image or label is missing or duplicated;
- [ ] any tensor is non-finite or wrongly shaped;
- [ ] evaluation is opened before fit completion;
- [ ] a fit fails to converge or hits a bound;
- [ ] Top-1 or Top-5 changes beyond the frozen tie policy;
- [ ] raw and calibrated selective curves differ materially;
- [ ] code or config changes after fitting;
- [ ] an unauthorized upload or external write is attempted.

---

## I. Sign-off block

| Role | Name / reference | Decision | Date |
|---|---|---|---|
| Protocol reviewer |  |  |  |
| Data-use supervisor |  |  |  |
| Execution approver |  |  |  |
| Environment verifier |  |  |  |

**No sign-off is pre-filled. Stage 0B is not authorized.**
