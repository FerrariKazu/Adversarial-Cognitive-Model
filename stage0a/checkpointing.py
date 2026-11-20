"""Durable Stage 0A checkpoints.

A rolling checkpoint is the authoritative resume state and contains every
piece of state the assignment requires: model, optimizer, scheduler, AMP
GradScaler, epoch/global step, RNG for Python/NumPy/PyTorch CPU/all CUDA
devices, deterministic sampler information, full config, seed, backbone and
dataset identities, code fingerprint, and hardware/precision metadata.

Writes are atomic.  Best/final artifacts are model-only and are never used as
resume state.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from noesis_vision.core.checkpoint import atomic_torch_save
from stage0a.config import Stage0AConfig
from stage0a.utils import atomic_write_text, capture_rng_state, restore_rng_state, sha256_file, utc_now

CHECKPOINT_FORMAT_VERSION = 1


class ResumeValidationError(RuntimeError):
    """The checkpoint exists but cannot safely resume this run."""


def epoch_seed(seed: int, epoch: int) -> int:
    return int(seed) * 100_003 + int(epoch)


def _sidecar(path: str | Path) -> Path:
    return Path(str(path) + ".sha256")


def write_sidecar(path: str | Path) -> str:
    digest = sha256_file(path)
    atomic_write_text(_sidecar(path), digest + "\n")
    return digest


def save_rolling(
    path: str | Path,
    *,
    cfg: Stage0AConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    history: Dict[str, Any],
    training_fingerprint: str,
    dataset_fingerprint: str,
    split_fingerprint: str,
    backbone_weights_sha256: str,
    hardware: Dict[str, Any],
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "rolling",
        "run_id": cfg.run_id,
        "protocol_id": cfg.to_dict()["protocol_id"],
        "protocol_version": cfg.to_dict()["protocol_version"],
        "seed": cfg.seed,
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "global_step": int(global_step),
        "epoch_seed": epoch_seed(cfg.seed, epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng_state(),
        "sampler": {
            "kind": "torch.randperm",
            "epoch_seed": epoch_seed(cfg.seed, epoch),
            "batch_in_epoch": int(batch_in_epoch),
            "note": "permutation is regenerated from epoch_seed; skipping batch_in_epoch reproduces the next batch",
        },
        "config": cfg.to_dict(include_runtime=True),
        "config_canonical_hash": cfg.canonical_hash(),
        "training_fingerprint": training_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
        "backbone": cfg.backbone,
        "backbone_id": cfg.backbone_info["timm_name"],
        "backbone_weights_sha256": backbone_weights_sha256,
        "hardware": hardware,
        "precision": {"amp": True, "amp_dtype": cfg.amp_dtype},
        "history": history,
        "saved_at_utc": utc_now(),
    }
    atomic_torch_save(str(path), state)
    write_sidecar(path)
    return state


def save_model_artifact(
    path: str | Path,
    *,
    cfg: Stage0AConfig,
    model: torch.nn.Module,
    kind: str,
    metric: Dict[str, Any],
    training_fingerprint: str,
    dataset_fingerprint: str,
    split_fingerprint: str,
    backbone_weights_sha256: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": kind,
        "run_id": cfg.run_id,
        "config": cfg.to_dict(include_runtime=True),
        "config_canonical_hash": cfg.canonical_hash(),
        "seed": cfg.seed,
        "model": model.state_dict(),
        "metric": metric,
        "training_fingerprint": training_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
        "backbone": cfg.backbone,
        "backbone_id": cfg.backbone_info["timm_name"],
        "backbone_weights_sha256": backbone_weights_sha256,
        "saved_at_utc": utc_now(),
    }
    if extra:
        state.update(extra)
    atomic_torch_save(str(path), state)
    write_sidecar(path)
    return state


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    sidecar = _sidecar(path)
    if not sidecar.exists():
        raise ResumeValidationError(f"{path} has no SHA-256 sidecar; refusing unverifiable resume")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(path)
    if expected != actual:
        raise ResumeValidationError(f"{path} hash mismatch: file={actual} sidecar={expected}")
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ResumeValidationError(f"{path} did not contain a checkpoint dictionary")
    return state


def validate_resume(
    state: Dict[str, Any],
    *,
    cfg: Stage0AConfig,
    training_fingerprint: str,
    dataset_fingerprint: str,
    split_fingerprint: str,
    backbone_weights_sha256: str,
) -> None:
    if int(state.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ResumeValidationError(
            f"checkpoint format {state.get('format_version')} != {CHECKPOINT_FORMAT_VERSION}"
        )
    if state.get("run_id") != cfg.run_id:
        raise ResumeValidationError(f"run id mismatch: {state.get('run_id')} != {cfg.run_id}")
    if state.get("config_canonical_hash") != cfg.canonical_hash():
        raise ResumeValidationError("resolved experiment config hash changed; refusing resume")
    if state.get("training_fingerprint") != training_fingerprint:
        raise ResumeValidationError("training-code fingerprint changed; refusing resume")
    if state.get("dataset_fingerprint") != dataset_fingerprint:
        raise ResumeValidationError("dataset fingerprint changed; refusing resume")
    if state.get("split_fingerprint") != split_fingerprint:
        raise ResumeValidationError("train/holdout split fingerprint changed; refusing resume")
    if state.get("backbone") != cfg.backbone:
        raise ResumeValidationError(f"backbone mismatch: {state.get('backbone')} != {cfg.backbone}")
    if state.get("backbone_weights_sha256") != backbone_weights_sha256:
        raise ResumeValidationError("backbone weights fingerprint changed; refusing resume")
    if int(state.get("seed", -1)) != cfg.seed:
        raise ResumeValidationError(f"seed mismatch: {state.get('seed')} != {cfg.seed}")
    for key in ("epoch", "batch_in_epoch", "global_step", "epoch_seed"):
        if key not in state:
            raise ResumeValidationError(f"checkpoint missing required resume field {key!r}")
    if state.get("epoch_seed") != epoch_seed(cfg.seed, int(state["epoch"])):
        raise ResumeValidationError("epoch_seed is inconsistent with seed/epoch; refusing resume")


def restore_training_state(
    state: Dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
) -> None:
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    restore_rng_state(state.get("rng") or {})
