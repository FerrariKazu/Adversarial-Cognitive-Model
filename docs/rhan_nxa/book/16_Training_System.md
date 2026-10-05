# Chapter 16 — Training System: Checkpoint Durability and Resumption

## 1. In one sentence
RHAN-NXA's training infrastructure enforces atomic checkpoint saves, code-identity-gated resume, best/rolling checkpoint parity, and mandatory HF sync, eliminating the silent restart, stale-resume, and parity-gap incidents that disrupted Generation-0 training.

## 2. Five durability rules (each motivated by a real Gen-0 incident)

### Rule 1: Never a silent restart
When a rolling checkpoint exists—locally or on Hugging Face—resume is **mandatory**. A trainer that cannot restore the checkpoint must raise `CheckpointResumeError` and abort. It must never fall through to a fresh run. This closes the 2026-08-12 stale-resume bug where a Colab session restart silently began training from epoch 0, discarding 40+ epochs of state.

### Rule 2: Code-identity guard
Every checkpoint records `code_commit` (the git HEAD SHA at save time). Before restoring any checkpoint, `resume_commit_ok(ckpt, current_sha)` verifies the saved SHA matches the current code. Legacy checkpoints without a recorded SHA are **always refused**. This prevents silently resuming training with changed semantics.

### Rule 3: Atomic saves
Every write is `tmp_file → fsync → rename`. A runtime crash mid-save cannot produce a partial checkpoint that becomes the only artifact on disk.

### Rule 4: Best/rolling parity from commit one
Both the rolling (resume) artifact and the best-eval artifact are written by the same `save_state` call. Before any evaluation cites either artifact, `verify_best_rolling_parity()` checks that they agree on code identity and that the rolling checkpoint is at least as recent as the best. A parity gap—best artifact from a different code commit than the rolling—is flagged before it corrupts a reported metric.

### Rule 5: Mandatory HF sync
`save_rolling` and `save_best` accept an optional `uploader` callable. When provided, the artifact is pushed to Hugging Face immediately after the local write. A local-only checkpoint is one runtime reset away from loss.

## 3. Implementation

### `atomic_torch_save(target_path, obj)`
`tempfile.mkstemp → torch.save → fsync → os.rename`

### `save_rolling(path, *, epoch, model, optimizer, scheduler, extra, uploader)`
Records: `epoch`, `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`, `code_commit`, `saved_at_utc`, `kind="rolling"`.

### `save_best(path, *, model, config, metric_value, uploader)`
Records: `model.state_dict()`, `config.to_dict()`, `metric_value`, `code_commit`, `saved_at_utc`, `kind="best"`. The embedded config prevents eval/train config drift.

### `resume_or_abort(rolling_path, hf_repo_id, hf_filename, force_restart)`
Decision table:
| Condition | Action |
|:---|:---|
| `force_restart=True` | Return None (cold start, loud warning) |
| Local rolling exists | Load, check code identity, return state |
| No local, HF verified | Download, load, check code identity, return state |
| No local, HF unverifiable | **Raise `CheckpointResumeError`** |
| No local, HF not configured | Return None (genuinely fresh run) |

### `verify_best_rolling_parity(best_path, rolling_path)`
Returns `(bool, message)`. Eval scripts must call this before citing either artifact.

## 4. The optimizer resume guard
`OptimizerGroupRegistry.resume_guard(state_dict, saved_scheduler)` protects against restoring an optimizer state from a different group layout. It verifies group count, group names, and the LR-ratio pattern. See Chapter 14.

## 5. Port status
The checkpoint module is **ADAPTED** from Gen-0's `phase1_training/train_rhan_next.py` and `checkpoint_utils.py`:
- Core atomic save and all five rules: carried unchanged.
- Equivalence-edge registry (notebook-only commit equivalences): stays in Gen-0 file; Gen-1 trainers call it explicitly when needed.

## 6. Scientific status
- **Atomic saves**: **LOCKED** (infrastructure requirement; three incidents).
- **Code-identity guard**: **LOCKED** (infrastructure requirement).
- **Best/rolling parity**: **LOCKED** (infrastructure requirement).

## 7. Source references
- [`checkpoint.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/checkpoint.py): `atomic_torch_save`, `save_rolling`, `save_best`, `resume_or_abort`, `verify_best_rolling_parity`, `CheckpointResumeError`
- Illustrated in Figure 8 (`checkpoint_resume.svg`).
- Connected to Chapter 14 (gradient flow / optimizer resume guard) and Chapter 17 (experimental DAG).
