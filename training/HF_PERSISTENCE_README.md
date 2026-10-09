# HF Persistence for J1 Foundation Training

This document describes the Hugging Face checkpoint persistence system for J1 foundation training.

## Overview

The HF persistence system provides runtime-reset tolerance for J1 foundation training by persisting every completed epoch to Hugging Face. This ensures that if a Colab runtime disconnects, resets, or crashes, the training can resume from the newest verified epoch without retraining already completed work.

## Architecture

### Two HF Repositories

1. **Immutable Archive**: `FerrariKazu/rhan-nxa-checkpoints`
   - Every completed epoch creates a unique immutable artifact
   - Structure: `j1_foundation/<run_id>/trades/backbone_only/epoch_XXX/`
   - Contains: `checkpoint.pt`, `metadata.json`, `sha256.txt`
   - Never overwritten after commit

2. **Rolling Recovery Mirror**: `FerrariKazu/rhan-nxa-checkpoints-rolling`
   - Fast recovery mirror with latest state
   - Structure: `j1_foundation/<run_id>/trades/backbone_only/latest/`
   - Contains: `checkpoint.pt`, `metadata.json`, `sha256.txt`, `state.json`
   - Updated after every successful commit

### Configuration

Environment variables control HF persistence:

```bash
RHAN_HF_CHECKPOINT_REPO=FerrariKazu/rhan-nxa-checkpoints      # Immutable archive
RHAN_HF_ROLLING_REPO=FerrariKazu/rhan-nxa-checkpoints-rolling  # Recovery mirror
RHAN_HF_REPO_TYPE=dataset                                      # Repo type
HF_TOKEN=<your_hf_token>                                       # Mandatory for persistence
```

Defaults are set in `training/hf_persistence.py`.

### HF_TOKEN Requirement

When HF persistence is enabled:
- `HF_TOKEN` must be present
- If missing: training refuses to start with an error message
- Opt-out: use `--no-hf` flag for local development/testing

When `--no-hf` is used:
```
WARNING: HF persistence disabled.
Runtime-reset recovery is NOT guaranteed.
```

### Checkpoint Contents

Every checkpoint captures all state required to resume training:

- `model_state_dict` - model weights
- `optimizer_state_dict` - optimizer state
- `scheduler_state_dict` - learning rate scheduler state
- `scaler_state_dict` - AMP gradient scaler state
- `epoch` - current epoch number
- `global_step` - total training steps
- `best_metric` - best validation metric value
- `best_epoch` - epoch where best metric was achieved
- `training_history` - list of epoch records
- `config` - full configuration
- `config_hash` - SHA-256 of config
- `seed` - random seed
- `python_rng_state` - Python RNG state
- `numpy_rng_state` - NumPy RNG state (if enabled)
- `torch_cpu_rng_state` - PyTorch CPU RNG state
- `torch_cuda_rng_state` - PyTorch CUDA RNG state (if CUDA available)
- `code_commit` - git commit SHA
- `saved_at_utc` - timestamp

### Checkpoint Lifecycle

```
TRAINING
  ↓
LOCAL_CHECKPOINT_WRITTEN (atomic: tmp → fsync → rename)
  ↓
CHECKSUMMED (SHA-256 computed)
  ↓
METADATA_WRITTEN (metadata.json + sha256.txt sidecars)
  ↓
HF_ARCHIVE_UPLOADED (to immutable archive)
  ↓
HF_ARCHIVE_VERIFIED (remote download + SHA-256 verification)
  ↓
ROLLING_UPDATED (to recovery mirror)
  ↓
COMMITTED (manifest updated, event logged)
```

### State Transitions

- `LOCAL_ONLY` - checkpoint written locally but not yet uploaded
- `COMMITTED` - checkpoint uploaded, verified, and in manifest

A checkpoint that has only been written locally is **NOT** considered committed.

### Resume Algorithm

On fresh Colab/Kaggle runtime:

1. Authenticate `HF_TOKEN`
2. Identify `run_id` (stable across reconnects)
3. Download/inspect manifest from HF
4. Discover immutable epoch checkpoints from HF archive
5. Verify metadata/checksums of discovered checkpoints
6. Determine newest verified committed epoch
7. Reconcile manifest if necessary (in case of manifest lag)
8. Restore checkpoint state (model, optimizer, scheduler, RNG, etc.)
9. Continue with next epoch

Example:
```
Remote archive:
  epoch 1 ✓
  epoch 2 ✓
  ...
  epoch 43 ✓
  epoch 44 ✓
  epoch 45 partial/corrupt ✗

Resume from: epoch 44 → begin epoch 45
Do NOT start from epoch 1.
```

### Best Checkpoint Persistence

When validation produces a new best metric, an immutable best artifact is created:

```
best/
  epoch_001/
    checkpoint.pt
    metadata.json
    sha256.txt
  epoch_005/
    ...
```

Each best checkpoint has metadata identifying it as a best:
```json
{
  "kind": "best",
  "epoch": 5,
  "metric": "val_acc",
  "metric_value": 0.2718,
  "previous_best": 0.2632
}
```

### Corruption Handling

If the newest checkpoint is corrupt:

1. Search backward through committed epochs
2. Find newest verified valid checkpoint
3. Restore from that checkpoint
4. Log `CHECKPOINT_CORRUPT` and `RECOVERY_FALLBACK` events
5. Do NOT delete the corrupt artifact

### Config Integrity

Before resuming, the system compares:
- `config_hash` - configuration hash
- `git_commit` - code commit SHA
- `dataset_revision` - dataset version
- `architecture` - model architecture
- `phase` - training phase
- `arm` - experiment arm
- `seed` - random seed

If configuration differs from persisted run:
```
ERROR:
Existing J1 run has configuration hash ABC.
Current configuration hash XYZ.
Refusing to resume because this would mix experimental conditions.
Use --new-experiment for a separate run.
```

### Preflight

Before training starts, an HF persistence preflight runs:

```
J1 HF persistence preflight
HF_TOKEN: present
Archive repo: FerrariKazu/rhan-nxa-checkpoints
Rolling repo: FerrariKazu/rhan-nxa-checkpoints-rolling
Repo type: dataset
Authentication: OK
Remote access: OK
Write test: OK
Read-back verification: OK
```

The preflight performs a tiny non-destructive write/read test in a dedicated temporary namespace and cleans up afterward.

### Events

All significant events are logged to `events.jsonl` (append-only):

- `RUNTIME_STARTED`
- `CHECKPOINT_LOCAL_COMMITTED`
- `HF_ARCHIVE_UPLOAD_STARTED`
- `HF_ARCHIVE_UPLOAD_COMPLETED`
- `HF_ARCHIVE_VERIFIED`
- `ROLLING_UPDATED`
- `CHECKPOINT_COMMITTED`
- `BEST_CHECKPOINT_COMMITTED`
- `RECOVERY`
- `CHECKPOINT_CORRUPT`
- `RESUME`
- `EPOCH_COMPLETED`
- `TRAINING_COMPLETED`

Each event includes:
```json
{
  "timestamp": "2026-10-08T12:00:00Z",
  "run_id": "J1_backbone_only_...",
  "epoch": 44,
  "event": "CHECKPOINT_COMMITTED",
  "sha256": "abc123..."
}
```

### Verification Tool

Use `training/verify_hf_checkpoint_chain.py` to verify checkpoint chain integrity:

```bash
python3 training/verify_hf_checkpoint_chain.py \
  --repo FerrariKazu/rhan-nxa-checkpoints \
  --rolling-repo FerrariKazu/rhan-nxa-checkpoints-rolling \
  --run-id J1_backbone_only_gen1-adv-curriculum-v1_abc123
```

Verifies:
- Epoch continuity
- Immutable artifacts
- Checkpoint metadata
- SHA-256 integrity
- Config hash
- Git commit
- Dataset revision
- Best checkpoint chain
- Rolling pointer
- Manifest consistency
- No missing committed epochs
- No duplicate run identities

Output:
```
HF CHECKPOINT CHAIN: VERIFIED
```
or
```
HF CHECKPOINT CHAIN: FAILED
```
(with exact reason)

## Integration

### Trainer Changes

The trainer (`training/train_generation1_foundation.py`) now:

1. Imports `HFPersistenceCoordinator` from `training.hf_persistence`
2. Creates a coordinator when HF is enabled
3. Runs preflight before training
4. Discovers committed epochs from HF on resume
5. Reconciles manifest with discovered state
6. Persists every epoch after completion
7. Logs events to `events.jsonl`
8. Logs training completion

### Harness Changes

The harness (`cloud/gen1/colab_j1_foundation_gate.py`):

1. Removed `--no-hf` from canonical commands (HF persistence is now default)
2. HF persistence is enabled when `HF_TOKEN` is available
3. `--no-hf` still available for local development/testing

### CLI Usage

For the real J1 experiment with HF persistence enabled:

```bash
python3 training/train_generation1_foundation.py \
  --phase backbone_only \
  --data-root /path/to/imagenet100 \
  --batch-size 64 \
  --num-workers 0 \
  --device cuda
```

With `HF_TOKEN` set in environment:

```bash
export HF_TOKEN=hf_...
python3 training/train_generation1_foundation.py --phase backbone_only ...
```

For local development without HF:

```bash
python3 training/train_generation1_foundation.py \
  --phase backbone_only \
  --no-hf \
  --force-fresh
```

## Files Changed

- `training/hf_persistence.py` - New HF persistence module
- `training/train_generation1_foundation.py` - Integrated HF persistence
- `training/verify_hf_checkpoint_chain.py` - Verification tool
- `training/test_hf_persistence.py` - Unit tests
- `cloud/gen1/colab_j1_foundation_gate.py` - Updated harness

## Backward Compatibility

- Local checkpoint paths (`checkpoints/foundation_*_rolling.pth`, etc.) are preserved
- Existing local checkpoints remain usable
- `--no-hf` flag preserved for local-only mode
- Smoke mode still disables HF writes

## Current Run Status

The currently running T4 J1 experiment at epoch 44/60 was NOT disrupted. The new persistence system is ready for the next safe execution/resume. If needed, the local epoch-44 checkpoint can be uploaded as an explicitly recovered artifact.

## Acceptance Criteria

> **If Colab or Kaggle disappears immediately after any completed epoch, a fresh runtime can discover the newest verified epoch from Hugging Face and continue without retraining already committed work.**

This is verified by:
1. Unit tests for atomic checkpoint writes (Test A)
2. Tests for upload interruption handling (Test B)
3. Tests for runtime reset recovery (Test C)
4. Tests for corrupt checkpoint fallback (Test D)
5. Tests for manifest lag reconciliation (Test E)
6. Tests for config mismatch refusal (Test H)
7. Tests for HF disabled behavior (Test I)

All 28 tests pass.
