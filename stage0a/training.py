"""Linear-head training with step-level durable resume."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage0a.checkpointing import (
    ResumeValidationError,
    epoch_seed,
    load_checkpoint,
    restore_training_state,
    save_model_artifact,
    save_rolling,
    validate_resume,
)
from stage0a.config import Stage0AConfig
from stage0a.evaluation import evaluate_head, save_eval_artifacts
from stage0a.features import FeatureCache, cache_summary, load_complete_cache
from stage0a.hf_sync import HFProvenanceError, HFStore, HFUnavailable
from stage0a.utils import (
    append_jsonl,
    atomic_write_json,
    capture_rng_state,
    seed_everything,
    sha256_file,
    state_dict_sha256,
    utc_now,
)


class LinearHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _run_dir(cfg: Stage0AConfig) -> Path:
    return Path(cfg.output_root) / "runs" / cfg.backbone / f"seed{cfg.seed}"


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = min(1.0, max(0.0, (step - warmup) / max(total - warmup, 1)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_holdout(
    head: nn.Module,
    cache: FeatureCache,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> Dict[str, float]:
    head.eval()
    features = cache.store()
    labels = np.load(cache.labels, mmap_mode="r")
    correct = 0
    loss_sum = 0.0
    n = len(indices)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        x = torch.from_numpy(np.array(features[idx], copy=True)).float().to(device)
        y = torch.from_numpy(np.array(labels[idx], copy=True)).long().to(device)
        logits = head(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
        correct += int((logits.argmax(dim=-1) == y).sum().item())
    head.train()
    return {"n": n, "top1": correct / max(n, 1), "loss": loss_sum / max(n, 1)}


def _resolve_backbone_sha(cfg: Stage0AConfig, train_cache: FeatureCache) -> str:
    sha = train_cache.meta.get("backbone_weights_sha256")
    if sha:
        return str(sha)
    # Fallback for an older cache: load the official model once on CPU.
    import timm

    m = timm.create_model(
        cfg.backbone_info["timm_name"],
        pretrained=True,
        num_classes=0,
        img_size=cfg.img_size,
        dynamic_img_size=True,
    )
    sha = state_dict_sha256(m.state_dict())
    del m
    return sha


def _write_run_files(cfg: Stage0AConfig, run_dir: Path, manifest: Dict[str, Any]) -> None:
    cfg.save(str(run_dir / "config.json"))
    atomic_write_json(run_dir / "run_manifest.json", manifest)


def train_run(
    cfg: Stage0AConfig,
    *,
    train_cache: FeatureCache,
    val_cache: FeatureCache,
    index_dir: str | Path,
    split_info: Dict[str, Any],
    training_fingerprint: str,
    backbone_weights_sha256: str,
    hardware: Dict[str, Any],
    hf_store: HFStore,
    force_fresh: bool = False,
    confirm_overwrite_remote: bool = False,
    max_new_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Train one (backbone, seed) run, auto-resuming from HF when present."""
    run_dir = _run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    rolling_path = run_dir / "rolling.pt"
    best_path = run_dir / "best.pt"
    final_path = run_dir / "final.pt"
    metrics_path = run_dir / "metrics.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_fp = str(split_info["dataset_fingerprint"])
    split_fp = str(split_info["split_fingerprint"])
    manifest = {
        "protocol_id": "rhan-stage0a-dinov2/1",
        "run_id": cfg.run_id,
        "config": cfg.to_dict(include_runtime=True),
        "config_canonical_hash": cfg.canonical_hash(),
        "training_fingerprint": training_fingerprint,
        "dataset_fingerprint": dataset_fp,
        "split_fingerprint": split_fp,
        "backbone_id": cfg.backbone_info["timm_name"],
        "backbone_weights_sha256": backbone_weights_sha256,
        "features": {"train": cache_summary(train_cache), "val": cache_summary(val_cache)},
        "hardware": hardware,
        "created_at_utc": utc_now(),
    }
    _write_run_files(cfg, run_dir, manifest)
    hf_store.sync_json(cfg.run_id, run_dir / "config.json", "config.json")
    hf_store.sync_json(cfg.run_id, run_dir / "run_manifest.json", "run_manifest.json")

    if final_path.exists():
        final = load_checkpoint(final_path)
        if final.get("config_canonical_hash") == cfg.canonical_hash() and final.get("model"):
            hf_store.sync_final(cfg.run_id, final_path)
            return {"status": "already_complete", "run_id": cfg.run_id, "final": str(final_path), "metric": final.get("metric")}

    remote_name = hf_store.remote_name(cfg.run_id, "rolling.pt")
    remote_status = hf_store.status(hf_store.rolling_repo, remote_name)
    if force_fresh and remote_status == "present" and not confirm_overwrite_remote:
        raise HFProvenanceError(
            f"refusing --fresh: remote rolling state exists at {hf_store.rolling_repo}/{remote_name}. "
            "Pass explicit confirmation to overwrite a previous run."
        )
    if force_fresh:
        archive = run_dir / "archive" / f"fresh_{utc_now().replace(':', '').replace('-', '')}"
        archive.mkdir(parents=True, exist_ok=True)
        for p in (rolling_path, best_path):
            if p.exists():
                p.rename(archive / p.name)
        append_jsonl(metrics_path, {"event": "force_fresh", "archive": str(archive), "at_utc": utc_now()})
    restored_path, source = hf_store.restore_rolling(cfg.run_id, rolling_path, force_fresh=force_fresh)
    state: Optional[Dict[str, Any]] = None
    if restored_path is not None:
        state = load_checkpoint(restored_path)
        try:
            validate_resume(
                state,
                cfg=cfg,
                training_fingerprint=training_fingerprint,
                dataset_fingerprint=dataset_fp,
                split_fingerprint=split_fp,
                backbone_weights_sha256=backbone_weights_sha256,
            )
        except ResumeValidationError as err:
            if os.environ.get("STAGE0A_STRICT_RESUME", "0") == "1":
                raise
            print(
                f"[train_run] Incompatible rolling checkpoint for {cfg.run_id} at {restored_path}: {err}. "
                "Archiving and starting fresh run to preserve provenance.",
                flush=True,
            )
            archive = run_dir / "archive" / f"incompatible_{utc_now().replace(':', '').replace('-', '')}"
            archive.mkdir(parents=True, exist_ok=True)
            for p in (rolling_path, best_path):
                if p.exists():
                    try:
                        p.rename(archive / p.name)
                    except OSError:
                        pass
                sidecar = p.with_name(p.name + ".sha256")
                if sidecar.exists():
                    try:
                        sidecar.rename(archive / sidecar.name)
                    except OSError:
                        pass
            append_jsonl(
                metrics_path,
                {"event": "incompatible_checkpoint_fresh_start", "reason": str(err), "archive": str(archive), "at_utc": utc_now()},
            )
            state = None
    elif best_path.exists():
        raise HFProvenanceError(
            f"best checkpoint exists at {best_path} but no rolling checkpoint exists; "
            "cannot resume and will not silently restart"
        )

    seed_everything(cfg.seed)
    head = LinearHead(cfg.embed_dim, cfg.num_classes).to(device)
    optimizer = torch.optim.SGD(
        head.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, cfg.warmup_steps, cfg.max_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.amp_dtype == "float16")

    history: list[Dict[str, Any]] = []
    best_metric: Dict[str, Any] = {}
    start_epoch = 0
    start_batch = 0
    global_step = 0
    if state is not None:
        restore_training_state(state, model=head, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        start_epoch = int(state["epoch"])
        start_batch = int(state["batch_in_epoch"])
        global_step = int(state["global_step"])
        history_payload = state.get("history") or {}
        if isinstance(history_payload, dict):
            history = list(history_payload.get("events") or [])
            best_metric = dict(history_payload.get("best_metric") or {})
        else:
            history = list(history_payload)
            best_metric = dict(state.get("best_metric") or {})
        append_jsonl(
            metrics_path,
            {"event": "resume", "source": source, "epoch": start_epoch, "batch_in_epoch": start_batch, "global_step": global_step, "at_utc": utc_now()},
        )
    else:
        append_jsonl(metrics_path, {"event": "fresh_start", "epoch": 0, "global_step": 0, "at_utc": utc_now()})

    # The best file is authoritative if the rolling checkpoint was written
    # immediately before an evaluation that then improved the best metric.
    if best_path.exists():
        best_state = load_checkpoint(best_path)
        if best_state.get("config_canonical_hash") == cfg.canonical_hash() and not best_metric:
            best_metric = dict(best_state.get("metric") or {})

    train_idx = np.asarray(split_info["train_indices"], dtype=np.int64)
    holdout_idx = np.asarray(split_info["holdout_indices"], dtype=np.int64)
    n_train = len(train_idx)
    steps_per_epoch = int(math.ceil(n_train / cfg.head_batch_size))
    n_epochs = int(math.ceil(cfg.max_steps / steps_per_epoch))
    stop_step = cfg.max_steps if max_new_steps is None else min(cfg.max_steps, global_step + max_new_steps)
    features = train_cache.store()
    labels = np.load(train_cache.labels, mmap_mode="r")

    def save_and_sync(epoch: int, batch_in_epoch: int, step: int) -> None:
        save_rolling(
            rolling_path,
            cfg=cfg,
            model=head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            global_step=step,
            history={"events": history[-25:], "best_metric": best_metric},
            training_fingerprint=training_fingerprint,
            dataset_fingerprint=dataset_fp,
            split_fingerprint=split_fp,
            backbone_weights_sha256=backbone_weights_sha256,
            hardware=hardware,
        )
        hf_store.sync_rolling(cfg.run_id, rolling_path, log_path=metrics_path, required=False)

    def eval_and_maybe_best(step: int, epoch: int, batch_in_epoch: int) -> None:
        nonlocal best_metric
        m = evaluate_holdout(head, train_cache, holdout_idx, device=device)
        m.update({"global_step": step, "epoch": epoch, "at_utc": utc_now(), "kind": "holdout"})
        history.append(m)
        append_jsonl(metrics_path, {"event": "holdout_eval", **m})
        if not best_metric or (m["top1"], -m["loss"]) > (best_metric.get("top1", -1), -best_metric.get("loss", 1e9)):
            best_metric = dict(m)
            save_model_artifact(
                best_path,
                cfg=cfg,
                model=head,
                kind="best",
                metric=best_metric,
                training_fingerprint=training_fingerprint,
                dataset_fingerprint=dataset_fp,
                split_fingerprint=split_fp,
                backbone_weights_sha256=backbone_weights_sha256,
                extra={"global_step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch},
            )
            hf_store.sync_best(cfg.run_id, best_path, required=False)
            append_jsonl(metrics_path, {"event": "best_saved", **best_metric})

    cur_epoch = start_epoch
    cur_batch = start_batch
    try:
        for epoch in range(start_epoch, n_epochs):
            cur_epoch = epoch
            cur_batch = 0 if epoch != start_epoch else start_batch
            if global_step >= stop_step:
                break
            gen = torch.Generator()
            gen.manual_seed(epoch_seed(cfg.seed, epoch))
            perm = torch.randperm(n_train, generator=gen).numpy()
            for batch_idx, offset in enumerate(range(0, n_train, cfg.head_batch_size)):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue
                if global_step >= stop_step:
                    break
                idx = train_idx[perm[offset : offset + cfg.head_batch_size]]
                x = torch.from_numpy(np.array(features[idx], copy=True)).float().to(device, non_blocking=True)
                y = torch.from_numpy(np.array(labels[idx], copy=True)).long().to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                    logits = head(x)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                batch_in_epoch = batch_idx + 1
                cur_batch = batch_in_epoch

                if global_step % cfg.eval_every_steps == 0:
                    eval_and_maybe_best(global_step, epoch, batch_in_epoch)
                if global_step % cfg.checkpoint_every_steps == 0:
                    save_and_sync(epoch, batch_in_epoch, global_step)
                if global_step >= stop_step:
                    save_and_sync(epoch, batch_in_epoch, global_step)
                    break
            # End of epoch checkpoint.
            if global_step < stop_step:
                eval_and_maybe_best(global_step, epoch + 1, 0)
                save_and_sync(epoch + 1, 0, global_step)
    except KeyboardInterrupt:
        save_and_sync(cur_epoch, cur_batch, global_step)
        raise

    if global_step < cfg.max_steps:
        append_jsonl(metrics_path, {"event": "partial_stop", "global_step": global_step, "max_steps": cfg.max_steps, "at_utc": utc_now()})
        hf_store.sync_rolling(cfg.run_id, rolling_path, log_path=metrics_path, required=False)
        return {
            "status": "partial",
            "run_id": cfg.run_id,
            "global_step": global_step,
            "max_steps": cfg.max_steps,
            "best_metric": best_metric,
        }

    # Official validation is touched once, after training and holdout selection.
    if best_path.exists():
        best = load_checkpoint(best_path)
        head.load_state_dict(best["model"])
    official, y, preds = evaluate_head(head, val_cache, device=device, batch_size=cfg.eval_batch_size)
    official.update({"seed": cfg.seed, "backbone": cfg.backbone, "global_step": global_step})
    artifacts = save_eval_artifacts(
        run_dir,
        metrics=official,
        labels=y,
        preds=preds,
        class_names=traincache_class_names(val_cache),
    )
    save_model_artifact(
        final_path,
        cfg=cfg,
        model=head,
        kind="final",
        metric=official,
        training_fingerprint=training_fingerprint,
        dataset_fingerprint=dataset_fp,
        split_fingerprint=split_fp,
        backbone_weights_sha256=backbone_weights_sha256,
        extra={"best_holdout": best_metric, "official_val": official, "artifacts": artifacts, "global_step": global_step},
    )
    hf_store.sync_final(cfg.run_id, final_path)
    hf_store.sync_json(cfg.run_id, Path(artifacts["metrics"]), "official_val_metrics.json")
    hf_store.sync_json(cfg.run_id, Path(artifacts["per_class"]), "official_val_per_class.csv")
    hf_store.sync_json(cfg.run_id, Path(artifacts["predictions"]), "official_val_predictions.npz")
    completion = {
        "run_id": cfg.run_id,
        "status": "complete",
        "global_step": global_step,
        "official_val": official,
        "final_sha256": sha256_file(final_path),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(run_dir / "completion.json", completion)
    hf_store.sync_json(cfg.run_id, run_dir / "completion.json", "completion.json")
    return {"status": "complete", "run_id": cfg.run_id, "final": str(final_path), "metric": official}


def traincache_class_names(cache: FeatureCache) -> Optional[list[str]]:
    """Class names are stored in the feature cache metadata's source index."""
    # Avoid carrying a 1000-string list in every checkpoint; read it from the
    # cache metadata when available, else let the evaluator use indices.
    names = cache.meta.get("class_names")
    return list(names) if names else None
