# Authorization-Gate Correction

## Scope

Operational authorization-gate correction only. The frozen Stage 0B protocol was not changed.

- Frozen protocol payload: `0bcf8bf8e37e3d8929bdf778bd443f41517e54597afb15a31d8e869367959672`
- Freeze commit: `a1fc47c996b5a6c31446d108f14499626c8254ea`
- No temperature fitting was performed.
- No evaluation outcomes were inspected.

## Readiness states

Preflight now reports:

- `technical_status`: `PASS` or `FAIL`
- `local_execution_ready`: true only when technical preflight passes, execution approval is valid, and data-use confirmation is valid.
- `publication_ready`: false until publication/upload rights are separately established.
- `execution_blockers`: blockers for local execution.
- `publication_blockers`: blockers for publication/upload only.

The ambiguous unconditional `launch_ready=false` field was removed. No external consumers were found outside the runner and generated preflight report.

## Data-use enforcement

`fit` and `score` now call `validate_execution_gates()`, which requires:

1. Valid execution approval.
2. Valid data-use confirmation.
3. Frozen protocol integrity via `load_protocol()`.
4. Existing checkpoint, archive, dataset, and partition integrity checks.
5. Stage-specific conditions:
   - `score` requires `fit_complete.json`.
   - `score` verifies calibration artifact and temperature hashes.
   - `score` refuses overwrite of completed scoring.

Machine-readable data-use template:

- Schema: `rhan-stage0b-data-use-confirmation/v1`
- Template: `audit/stage0b_execution_v1/authorization/DATA_USE_CONFIRMATION_TEMPLATE.json`
- Completed record path: `audit/stage0b_execution_v1/data_use_confirmation.json`

## Publication separation

Publication/upload authorization is not a prerequisite for `fit` or `score`. Raw images, features, logits, predictions, and other image-derived artifacts remain local. `publication_ready` remains false until separately authorized.

## Tests

Tests: `tests/test_stage0b_authorization_gate.py`

Result: `12 passed`.

Coverage:

| Execution approval | Data-use confirmation | Local execution |
|---|---|---|
| Missing | Missing | BLOCK |
| Valid | Missing | BLOCK |
| Missing | Valid | BLOCK |
| Valid | Valid | ALLOW |

Additional tests:

- Invalid execution approval: BLOCK.
- Invalid data-use confirmation, including publication rights asserted: BLOCK.
- Invalid protocol hash: BLOCK.
- Invalid checkpoint hash: BLOCK.
- Invalid partition hash: BLOCK.
- Publication status unresolved: does not block local execution gates.
- `score` without `fit_complete.json`: BLOCK.
- `score` without fitted temperatures: BLOCK.

## Execution implementation hash

- Runner SHA-256: `16a1c136ebf5da883cb9d40ce2ea1b4f87fb9578ce23d4fd15f94caef1a43077`
- Test file: `tests/test_stage0b_authorization_gate.py`

## Final unresolved blockers

1. Human execution approval not recorded.
2. Supervisor/institutional data-use confirmation not recorded.
3. Publication/upload rights unresolved (does not block local execution).

## Verdict

CONDITIONAL — HUMAN APPROVALS MISSING.