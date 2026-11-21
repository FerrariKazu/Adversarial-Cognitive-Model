# Stage 0B — Unresolved Blockers

**Protocol:** `rhan-stage0b-dinov2-b14-temperature/1` version `1.0.0`  
**Status:** **NOT LAUNCHED**  
**Freeze state:** Candidate protocol prepared for human review.

---

## Blockers to execution (not defects in the frozen analysis)

| ID | Blocker | Why it matters | Closure condition | Blocks freeze? | Blocks launch? |
|---|---|---|---|---|---|
| B1 | No recorded human protocol-freeze approval | The protocol is a candidate, not an approved freeze | Reviewer records approval against this directory and manifest | Yes, for “approved” status | Yes |
| B2 | No recorded supervisor/institutional data-use confirmation | ImageNetV2 images are not redistributable; local research use must be accepted by the responsible institution | Written confirmation is filed and referenced in the checklist | No | Yes, where required |
| B3 | No separate execution approval | Preparing a protocol does not authorize fitting or scoring | A second, explicit execution approval is recorded | No | Yes |
| B4 | No public-release or upload rights for image-derived artifacts | Features, logits, predictions, and metrics may be derivative artifacts | Rights review explicitly permits the intended artifact use, or outputs remain local | No | Yes for any upload/publication |
| B5 | Stage 0A image-level replay remains blocked | No local ImageNet validation images, no HF token, no Kaggle mount; gated sources returned 401 | Authorized validation access is obtained and a separate replay is completed, or the limitation is formally waived | No | No, for Stage 0B primary; yes for any Stage 0A replay claim |
| B6 | Historical Stage 0A training-time `timm`/`torchvision` versions were not recorded | Exact software reconstruction of Stage 0A cannot be proven | Recover original environment record, or accept the separately pinned Stage 0B configuration | No; documented limitation | No, if separately pinned config is approved |
| B7 | ImageNetV2 near-duplicate overlap with ImageNet-1k was not independently reconstructed | Exact-duplicate audit passed, but near-duplicate screening is inherited from the dataset authors | Independent near-duplicate study, or accept inherited screening as a limitation | No; documented limitation | No, if accepted in report |
| B8 | Official ImageNet validation is not independent and images are unavailable | It was fully evaluated in Stage 0A; only a feature cache may exist | Use only as labelled secondary benchmark, or omit it | No | No for primary |
| B9 | Hugging Face storage is not assumed or authorized | Uploading features/logits could violate data-use constraints | Keep all artifacts local; no HF write path enabled | No | Yes for any upload |
| B10 | Runtime may differ from the recorded development host | Feature/logit reproducibility can depend on GPU, driver, CUDA, cuDNN, and BLAS | Re-record environment; approve any material difference or reproduce on the recorded host | No | Yes if unapproved difference |

---

## Accepted limitations

These are not silently treated as resolved:

1. **Five images per class.** The primary evaluation supports a global calibration estimate, not stable per-class inference. Class-conditional results are descriptive.
2. **Distribution shift.** ImageNetV2 MatchedFrequency is a new test distribution, not a random sample from ImageNet-1k training.
3. **Backbone provenance is broader than the evaluation set.** DINOv2 pretraining used a large corpus. No absolute independence claim is made.
4. **Stage 0B software is separately pinned.** The state-dict hash and transform match Stage 0A evidence, but the historical training-time software is unknown.
5. **Archive extension mismatch.** The upstream `.tar.gz` is an uncompressed POSIX tar. The digest and member integrity pass; the naming caveat remains.
6. **No Stage 0B outcomes exist.** The absence of results is intentional, not a hidden failure.
7. **No legal conclusion.** The documentation records operating constraints; it does not provide legal advice or a redistribution license.

---

## Not blockers

The following are deliberate protocol choices, not unresolved problems:

- one temperature per seed rather than a shared temperature;
- no per-class temperature;
- no seed selection;
- no optional stopping;
- no evaluation-set re-partitioning;
- no secondary benchmark override;
- no upload of raw or derived data.

---

## Launch disposition

The freeze bundle can be reviewed without closing B1–B10. Execution remains blocked until the launch checklist is signed and every applicable gate is closed.

**Stage 0B is NOT LAUNCHED.**
