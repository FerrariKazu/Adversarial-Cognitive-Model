# Publication and Storage Restrictions

Stage 0B artifacts are local-only until separate publication or upload authorization is granted.

## Prohibited without separate authorization

- Raw ImageNetV2 images.
- Image-derived features.
- Logits.
- Predictions.
- Other derived artifacts whose publication or storage rights are unresolved.
- Frozen bundle copies containing the above artifacts.

## Permitted local execution

- Local temperature fitting on the calibration split.
- Local one-shot evaluation on the evaluation split.
- Local storage of calibration features, evaluation features, temperatures, metrics, bootstrap results, and decision records under `audit/stage0b_execution_v1/`.

## Separation of authorizations

- Execution authorization does not imply publication authorization.
- Upload authorization does not imply execution authorization.
- Data-use confirmation does not imply publication rights.
- Publication or upload rights must be resolved and recorded separately.

## Required before any upload or publication

- [ ] Rights holder or institutional permission identified.
- [ ] Permitted artifact list defined.
- [ ] Destination and retention policy defined.
- [ ] Security and access controls defined.
- [ ] Explicit approval recorded.