# Final Authorization Checklist

Stage 0B protocol freeze commit: `a1fc47c996b5a6c31446d108f14499626c8254ea`
Protocol payload hash: `0bcf8bf8e37e3d8929bdf778bd443f41517e54597afb15a31d8e869367959672`

## Freeze preservation

- [ ] Freeze commit verified and unchanged.
- [ ] Working tree contains no changes to frozen protocol files.
- [ ] All 27 manifest entries verified.
- [ ] Protocol payload hash verified.
- [ ] No push to GitHub has been made.

## Execution approval

- [ ] `EXECUTION_APPROVAL_TEMPLATE.json` copied to `audit/stage0b_execution_v1/execution_approval.json`.
- [ ] Approver identity and role recorded.
- [ ] Authorization date recorded.
- [ ] `approved` set to `true`.
- [ ] `execution_approved` set to `true`.
- [ ] Protocol payload hash matches exactly.
- [ ] Execution runner path and SHA-256 match exactly.
- [ ] All three approved `final.pt` checkpoint hashes match exactly.
- [ ] Partition JSON and canonical hashes match exactly.
- [ ] Approved local environment recorded.
- [ ] Permitted scope reviewed and accepted.

## Data-use confirmation

- [ ] Supervisor or institutional confirmation recorded.
- [ ] Confirming person's role recorded.
- [ ] Confirmation date recorded.
- [ ] Scope and restrictions recorded.

## Publication and storage

- [ ] Local-only artifact storage accepted.
- [ ] Raw images, features, logits, predictions, and unresolved derived artifacts will not be uploaded.
- [ ] Publication rights decision recorded separately from execution authorization.

## Runner readiness

- [ ] Preflight completed with technical status PASS.
- [ ] Fit refusal without approval verified.
- [ ] Score refusal without approval verified.
- [ ] All three seeds mandatory in runner.
- [ ] Failed integrity checks stop execution.

## Final authorization statement

- [ ] Separate execution authorization granted.
- [ ] Ready to run `fit`, then `score`, with no protocol changes.