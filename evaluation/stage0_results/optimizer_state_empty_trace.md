# Mechanism Trace: Empty Optimizer State at Saved Epoch 4

**Artifact:** `checkpoints/foundation_recurrence_112_seed17_rolling.pth`
**Scope:** read-only inspection; no GPU, no training.
**Date:** 2026-11-04

---

## 1. Observed artifact

| Field | Value |
| :--- | :--- |
| File size | 93,555,205 bytes |
| `phase` | `consolidated_112` |
| `arm` | `v1` |
| `epoch` | 4 |
| `val_clean_acc` | 0.576171875 |
| `val_rob_acc` | 0.27734375 |
| `optimizer.state` entries | 0 |
| `optimizer.param_groups` | 5 |
| `scheduler.last_epoch` | 4 |
| `scaler` / `rng` / `commit_hash` / `source_hashes` | absent |

The epoch-4 metrics match the `recurrence_112_seed17` epoch-4 monitor row in `forensics_seed17.md` (57.62% clean, 27.73% robust), so the model weights and metrics are real epoch-4 state. Only the optimizer moments are missing. A normal full-state checkpoint from this family is about 280.7 MB; the 93.6 MB file is the expected size after the optimizer `state` dictionary is emptied while the five parameter groups are preserved.

The same seed's epoch-15 final checkpoint (`foundation_recurrence_112_seed17_final.pth`) has 191 optimizer state entries and an AdamW step counter of 21,762. That is 11 epochs of steps at 1,979 steps per epoch (11 x 1,979 = 21,769, short by 7), matching a fresh optimizer starting at epoch 5 and running through epoch 15. The clean/robust final metrics (67.04% / 33.40%) are therefore the result of training epochs 5-15 after a reset at the epoch 4/5 boundary.

## 2. What the canonical save path does

`git log -S"consolidated_112" --oneline`:

```text
4d6b5f9 release-prep: assemble non-code artifacts onto the hardened resume base
83383f1 stage0 closeout part 3: T5 AutoAttack battery, McNemar reconciliation, forensics, model card
a1bf81f stage0 closeout: Kaggle_stage0_closeout.py + T2-T7 scripts (T2 resume-safe, arm-isolated)
b37f6c1 feat(colab): adapt Kaggle_consolidated.py to colab_consolidated.py with live sync and resume support
512a562 feat(consolidated-112): integrate ff_112 engine, lock preregistration, and prepare 5k worst-case evaluation
5c5b217 feat: consolidated 112px recurrence flagship training and locked pre-registration
```

The phase name enters the training engine at `5c5b217` and is integrated at `512a562`. At both commits the per-epoch save is a direct `torch.save` of a dictionary containing `optimizer.state_dict()`; there is no branch that empties `state`:

```python
# 512a562 training/train_gen2_foundation.py
ckpt_state = {
    "phase": phase, "arm": arm, "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "val_clean_acc": val_clean_acc,
    "val_rob_acc": val_rob_acc,
    "best_clean_acc": max(best_clean_acc, val_clean_acc),
    "best_rob_acc": max(best_rob_acc, val_rob_acc),
}
torch.save(ckpt_state, rolling_path)
```

The 11-hour cutoff block at the same commit writes the same full optimizer dictionary but omits `val_clean_acc` / `val_rob_acc`. The local artifact has both metrics, so it is not an 11-hour cutoff save.

At `6399601` the optimizer load was wrapped in a try/except that can continue with a fresh optimizer after a parameter-group mismatch. That fallback is a resume-time choice; it does not write an empty optimizer state. It also does not change the fact that a subsequent completed epoch would save a populated optimizer state.

## 3. Hugging Face sync and restore

At `6399601`, `cloud_setup/Kaggle_consolidated.py` had no `sync_to_hf` / `restore_checkpoints_from_hf`; `cloud_setup/colab_consolidated.py` did. At `19c5ea9` and at the current tip both files have both functions.

In both files and at all three revisions, `restore_checkpoints_from_hf` only downloads (`hf_hub_download`) and copies bytes (`shutil.copyfile`) into the local checkpoint directory. `sync_to_hf` only uploads local files (`api.upload_file`). Neither function loads a checkpoint, edits a dictionary, or re-saves a checkpoint. Therefore neither can create an optimizer entry with an empty `state`.

## 4. History search for strip/slim/re-save code

Commands run over `cloud_setup/` and `scripts/`, plus `git log -S` over `cloud_setup`, `scripts`, and `training`:

```text
rg -n 'torch\.save|"state"\s*:\s*\{\}|slim|strip\(|optimizer\.state_dict|param_groups' cloud_setup scripts
git log --all --oneline -S'torch.save' -- cloud_setup scripts training noesis_vision
git log --all --oneline -S'optimizer.state_dict' -- cloud_setup scripts training noesis_vision
git log --all --oneline -S'param_groups' -- cloud_setup scripts training noesis_vision
git log --all --oneline -S'"state": {}' -- cloud_setup scripts training noesis_vision
```

Findings:

- No current launcher, notebook, or script contains a checkpoint re-save, slim, strip, or `"state": {}` transformation.
- `torch.save` in current `cloud_setup/` and `scripts/` writes synthetic image/label payloads only, not checkpoints.
- The only `-S'"state": {}'` hit is `7c8ca52` ("Kaggle fix 10/9"), which adds a test fixture under `training/test_hf_persistence.py`; it is not production save code and predates the `consolidated_112` phase.
- Historical `training/hf_persistence.py` (added at `7c8ca52`, removed later) is not imported by the consolidated launchers and predates `consolidated_112`.

## 5. What can and cannot be established

**Can be established:**

1. The local epoch-4 file contains real epoch-4 model weights and metrics.
2. Its optimizer dictionary has the expected five parameter groups but zero state entries.
3. The canonical training save path writes `optimizer.state_dict()` directly and cannot produce an empty state after epochs 1-4 of optimizer steps.
4. The Hugging Face sync/restore paths only move bytes and cannot rewrite a checkpoint.
5. No current or historical launcher/script in the searched paths contains a checkpoint slim/strip/re-save step.
6. The epoch-15 seed-17 final checkpoint is consistent with a fresh optimizer starting at epoch 5 (11 x 1,979 = 21,769 expected, 21,762 observed).

**Cannot be established from this tree:**

1. Which external process, machine, or manual action rewrote the epoch-4 rolling file and emptied `optimizer.state`.
2. Whether the rewrite was a deliberate recovery step or an accident.
3. Whether the seed-17 reset caused the lower seed-17 accuracy; the census shows the reset happened, not that it caused the accuracy difference.
4. Anything about epochs 1-4 beyond the saved epoch-4 weights and metrics.

## 6. Consequence for resume safety

Before the guard added in `e487dff`, `optimizer.load_state_dict({"state": {}, "param_groups": [...]})` succeeds silently, so a resume from this file would continue with fresh AdamW moments at epoch 5 and no warning in the fallback log. The new guard aborts unless `allow_fresh_optimizer=True`, and records `{"type": "optimizer_state_empty"}` when the fresh path is explicitly allowed.
