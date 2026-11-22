# Final Runner Validation

Runner: `audit/stage0b_execution_v1/scripts/stage0b_runner.py`
Runner SHA-256: `16a1c136ebf5da883cb9d40ce2ea1b4f87fb9578ce23d4fd15f94caef1a43077`

## Validation tests run

1. Preflight:
   - Command: `python3 audit/stage0b_execution_v1/scripts/stage0b_runner.py preflight --out audit/stage0b_execution_v1/preflight`
   - Result: technical status PASS; launch_ready false; 3 launch blockers.
   - Archive: PASS, 1,264,079,360 bytes.
   - Dataset: PASS, 10,000 files, 1,000 classes, 10 images per class, 5,000/5,000 split, zero overlap.
   - Class mapping: PASS.
   - Three heads: PASS.
   - Backbone: PASS.
   - 8-image inference smoke: PASS, feature shape `[8, 768]`.
   - Evaluation images opened for outcomes: 0.

2. Fit refusal without approval:
   - Command: `python3 .../stage0b_runner.py fit --out /tmp/scratchpad/fit_test`
   - Result: exit code 1; `execution approval required`.

3. Score refusal without approval:
   - Command: `python3 .../stage0b_runner.py score --out /tmp/scratchpad/score_test`
   - Result: exit code 1; `execution approval required`.

4. Invalid approval refusal:
   - Temporary copy of `execution_approval.example.json` used as `execution_approval.json`.
   - Result: exit code 1; `invalid execution approval: approved and execution_approved must be true`.
   - Temporary approval file removed; no approval file remains.

## Code-level checks

- Protocol hash: `load_protocol()` verifies `MANIFEST.sha256` (27 entries), recomputes the protocol payload hash, verifies partition JSON and canonical hashes, inference config hash, all three head hashes, and backbone state-dict hash.
- Checkpoint hashes: `verify_heads()` and `load_protocol()` check seeds 17, 42, and 99 against the approved hashes.
- Calibration isolation: `cmd_fit()` calls `extract_features("calibration", ...)` only; no evaluation feature extraction occurs during fitting.
- Evaluation gating: `cmd_score()` requires `fit_complete.json`, verifies calibration feature/meta/labels and temperature hashes, refuses overwrite of `score_complete.json`, and writes `score_started.json` before opening evaluation images.
- Mandatory seeds: fitting and scoring iterate over `(17, 42, 99)`; no subset path exists.
- Integrity failures: missing or mismatched artifacts raise `RuntimeError` and stop execution.
- Resumability: completed fit or score outputs are not overwritten; interrupted runs can be resumed from the remaining phase.
- Protocol immutability: the runner reads frozen artifacts and never edits them.

## Non-substantive runner correction

- `APPROVED_ENV_FINGERPRINT` was corrected to the fingerprint recorded in `audit/stage0b_final_freeze_v1/work/environment.json`.
- Preflight was rerun after the correction and remained PASS with the same 3 launch blockers.

## Validation conclusion

The runner is technically functional and safe for authorized execution. Execution remains blocked by missing human authorization, missing data-use confirmation, and unresolved publication/upload rights.