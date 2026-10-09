"""
Durable per-epoch JSONL log and offline summarizer for the Gen-1 foundation
trainer (Agent J1). LOGGING/DIAGNOSTICS ONLY — no training math changed.

One JSON line per epoch, appended and fsynced, synced to HF at the same cadence
as the rolling checkpoint. On resume, prints a banner and verifies the JSONL's
terminal epoch matches the checkpoint epoch.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry

from training.run_report import (
    EPOCH_BLOCK_LEGEND,
    HEALTH_LEGEND,
    compute_health_flags,
    default_phase_jsonl_path as _default_phase_jsonl_path,
    format_epoch_block,
    snap_model_state,
    model_update_stats,
    group_update_stats,
    feature_snapshot,
    ResumedBestState,
    _gradient_share_from_trades,
    _val_subset_hash,
    _feature_snapshot_for_block,
)

# Re-export the one shared class so the trainer and tests can import
# 'GradNormCollector' from either module and get the same object.
from training.run_report import GradNormCollector



# ── JSONL helpers ────────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def epoch_jsonl_record(
    phase: str,
    epoch: int,
    total_epochs: int,
    point: Any,
    lr: float,
    tr_loss: float,
    ce_clean: float,
    kl_trades: float,
    w_trades: float,
    train_clean_acc: float,
    train_adv_acc: float,
    val_clean: float,
    val_robust: float,
    clean_n: int,
    robust_n: int,
    robust_secs: float,
    best_clean: float,
    best_clean_epoch: int,
    best_robust: float,
    best_robust_epoch: int,
    criterion: str,
    grad: Dict[str, Any],
    epoch_seconds: float,
    img_per_sec: float,
    peak_vram_gb: Optional[float],
    config_sha256: str,
    git_commit: str,
    session_id: str,
    timestamp_utc: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "phase": phase,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "epsilon": float(getattr(point, "eps", 0.0)),
        "beta": float(getattr(point, "beta", 0.0)),
        "pgd_steps": int(getattr(point, "pgd_steps", 0)),
        "lr": float(lr),
        "train_loss_total": float(tr_loss),
        "train_ce_clean": float(ce_clean),
        "train_kl_trades": float(kl_trades),
        "w_trades": float(w_trades),
        "train_acc_clean": float(train_clean_acc),
        "train_acc_adv": float(train_adv_acc),
        "val_acc_clean": float(val_clean),
        "val_acc_robust": float(val_robust),
        "val_clean_n": int(clean_n),
        "val_robust_n": int(robust_n),
        "val_robust_secs": float(robust_secs),
        "best_clean": float(best_clean),
        "best_clean_epoch": int(best_clean_epoch),
        "best_robust": float(best_robust),
        "best_robust_epoch": int(best_robust_epoch),
        "checkpoint_criterion": str(criterion),
        "grad": grad,
        "epoch_seconds": float(epoch_seconds),
        "img_per_sec": float(img_per_sec),
        "peak_vram_gb": float(peak_vram_gb) if peak_vram_gb is not None else None,
        "config_sha256": str(config_sha256),
        "git_commit": str(git_commit),
        "session_id": str(session_id),
        "timestamp_utc": str(timestamp_utc or utc_now_iso()),
    }


def append_epoch_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(record, sort_keys=True, default=str) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def last_epoch_in_jsonl(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    last: Optional[int] = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            e = int(row.get("epoch", -1))
            if last is None or e > last:
                last = e
    return last


def default_phase_jsonl_path(report_dir: str, phase: str) -> str:
    return os.path.join(report_dir, f"{phase}_epoch_log.jsonl")


# ── resume banner + verification ─────────────────────────────────────────────


def resume_banner(
    phase: str,
    start_epoch: int,
    best_clean: float,
    best_clean_epoch: int,
    best_robust: float,
    best_robust_epoch: int,
    loaded_checkpoint_path: Optional[str],
    loaded_checkpoint_sha256: Optional[str],
    jsonl_path: str,
) -> str:
    jsonl_last = last_epoch_in_jsonl(jsonl_path)
    mismatch = (
        jsonl_last is not None
        and jsonl_last != start_epoch - 1
        and jsonl_last != start_epoch
    )
    parts = [
        f"=== {phase} RESUME ===",
        f"resumed from epoch {start_epoch}",
        f"best_clean={best_clean*100:.2f}% @ep{best_clean_epoch}",
        f"best_robust={best_robust*100:.2f}% @ep{best_robust_epoch}",
    ]
    if loaded_checkpoint_path and os.path.exists(loaded_checkpoint_path):
        parts.append(f"loaded checkpoint: {loaded_checkpoint_path}")
        parts.append(f"checkpoint sha256: {loaded_checkpoint_sha256 or 'n/a'}")
    else:
        parts.append("loaded checkpoint: none (cold start)")
        parts.append("checkpoint sha256: n/a")
    parts.append(f"jsonl last epoch: {jsonl_last if jsonl_last is not None else 'none'}")
    if mismatch:
        parts.append(
            f"!!! MISMATCH: jsonl last epoch {jsonl_last} != expected "
            f"{start_epoch-1}; audit before trusting persisted bests")
    return "\n".join(parts)


def validate_jsonl_terminal_epoch(
    jsonl_path: str,
    expected_last_epoch: int,
) -> Tuple[bool, str]:
    last = last_epoch_in_jsonl(jsonl_path)
    if last is None:
        return True, f"jsonl empty or missing ({jsonl_path}) — cold start"
    if last != expected_last_epoch:
        return False, (
            f"jsonl last epoch {last} != expected {expected_last_epoch} "
            f"({jsonl_path})")
    return True, f"jsonl terminal epoch matches: {last}"


# ── instrumented train-one-epoch (Step 1 TRAIN decomposition) ────────────────

def _train_one_epoch_instrumented(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    registry: OptimizerGroupRegistry,
    device: torch.device,
    scaler: Optional[Any],
    epoch: int,
    total_epochs: int,
    clean_only: bool,
    w_trades: float,
    grad_collector: 'GradNormCollector',  # runtime type; imported from training.run_report
    point: Any,
) -> Tuple[float, float, float, float, float]:
    """Returns (loss_total, ce_clean, kl_trades, train_clean_acc, train_adv_acc).

    Reuses the existing training loop shape EXACTLY for backward/clip/step, and
    adds per-batch clean/adv accuracy plus loss decomposition by observing the
    SAME attack/loss inputs used by the training step. Training math is not
    changed.
    """
    import torch.nn.functional as F
    from training.adv_curriculum import (
        curriculum_for_epoch, pgd_kl_attack, trades_loss,
    )
    pt = point if point is not None else curriculum_for_epoch(epoch, total_epochs)
    model.train()
    total_loss = 0.0
    n_batches = 0
    ce_sum = 0.0
    kl_sum = 0.0
    clean_correct = 0
    adv_correct = 0
    n_samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        n_samples += y.numel()
        # train-batch accuracy: same forward + same attack contract, no-grad
        model.eval()
        try:
            with torch.no_grad():
                clean_pred = model(x).argmax(dim=1)
                clean_correct += (clean_pred == y).sum().item()
                if not clean_only:
                    x_adv_for_acc = pgd_kl_attack(
                        model, x, eps=pt.eps, steps=pt.pgd_steps)
                    adv_pred = model(x_adv_for_acc).argmax(dim=1)
                    adv_correct += (adv_pred == y).sum().item()
        finally:
            model.train()
        # loss decomposition (same attack/loss as training step)
        if clean_only:
            loss = F.cross_entropy(model(x), y)
            ce_sum += float(loss.item())
            kl_sum += 0.0
        else:
            x_adv = pgd_kl_attack(model, x, eps=pt.eps, steps=pt.pgd_steps)
            loss, _ = trades_loss(model, x, y, x_adv, beta=pt.beta)
            loss = w_trades * loss
            ce_sum += float(F.cross_entropy(model(x), y).item())
            raw_trade = loss / w_trades
            kl_sum += float(
                (raw_trade - F.cross_entropy(model(x), y)).item())
        # real training step (existing path)
        if scaler is not None:
            with torch.autocast("cuda", enabled=True):
                sl = loss
            scaler.scale(sl).backward()
            scaler.unscale_(optimizer)
            pre = grad_collector.record_pre()
            registry.clip_grad_per_group()
            post = grad_collector.record_post()
            if grad_collector.any_clipped_this_step(pre, post):
                grad_collector.steps_any_clipped += 1
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            pre = grad_collector.record_pre()
            registry.clip_grad_per_group()
            post = grad_collector.record_post()
            if grad_collector.any_clipped_this_step(pre, post):
                grad_collector.steps_any_clipped += 1
            optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    n = max(n_batches, 1)
    return (
        total_loss / n,
        ce_sum / n,
        kl_sum / n,
        clean_correct / max(n_samples, 1),
        adv_correct / max(n_samples, 1),
    )


# ── phase header (Step 4) ────────────────────────────────────────────────────

def _print_phase_header(
    phase: str,
    cfg: Any,
    registry: OptimizerGroupRegistry,
    first_point: Any,
) -> None:
    print(
        f"══ {phase} phase header ══\n"
        f"  dataset: {cfg.data_root} (fingerprint: see data root fingerprint.json)\n"
        f"  img_size={cfg.img_size} fovea_size={cfg.fovea_size} "
        f"num_classes={cfg.num_classes}\n"
        f"  epochs={cfg.epochs} batch_size={cfg.batch_size} "
        f"num_workers={cfg.num_workers}\n"
        f"  w_trades={cfg.w_trades} clean_only={cfg.clean_only}\n"
        f"  eval_seeds={list(cfg.eval_seeds)} n_eval_samples={cfg.n_eval_samples} "
        f"pgd_steps={cfg.pgd_steps}\n"
        f"  val_clean_subset={getattr(cfg,'val_clean_subset',None)} "
        f"val_robust_subset={getattr(cfg,'val_robust_subset',512)}\n"
        f"  optimizer groups: {registry.group_names}\n"
        f"  current position in eps/beta curriculum: epoch 1 -> "
        f"eps={first_point.eps:.3f} beta={first_point.beta:.1f} "
        f"pgd={first_point.pgd_steps}\n"
        f"  epoch block columns: {EPOCH_BLOCK_LEGEND}\n"
        f"  health flags: {HEALTH_LEGEND}\n"
        f"  (printed {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})\n"
        f"══\n",
        flush=True,
    )


def _gpu_peak_vram_gb(device: torch.device) -> Optional[float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return float(torch.cuda.max_memory_allocated(device)) / 1e9
    except Exception:
        return None


def _sync_jsonl_to_hf(coordinator: Any, jsonl_path: str) -> None:
    """Append-only sync of the phase JSONL to the HF rolling repo, at the same
    cadence as the rolling checkpoint."""
    if not coordinator or not getattr(coordinator, "enabled", False):
        return
    if not os.path.exists(jsonl_path):
        return
    rid = getattr(coordinator, 'run_id', 'local)')
    phase = getattr(coordinator, 'phase', '?')
    repo_base = f"j1_foundation/{rid}/trades/{phase}/epoch_log.jsonl"
    try:
        from training.hf_persistence import upload_to_hf
        upload_to_hf(
            jsonl_path,
            repo_base,
            coordinator.checkpoint_repo,
            coordinator.repo_type,
            coordinator.hf_token,
            f"epoch log: {getattr(coordinator,'phase','?')}",
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[{phase}] WARNING: jsonl HF sync "
            f"failed: {e}",
            flush=True,
        )


# ── offline summarizer (Step 5) ─────────────────────────────────────────────


def summarize_phase_from_jsonl(
    jsonl_path: str,
) -> str:
    """Read a phase JSONL log and print a table + plain-text trend verdict.

    Works from the HF-synced file alone (no GPU needed).
    """
    if not os.path.exists(jsonl_path):
        return f"no log at {jsonl_path}"
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return f"{jsonl_path}: empty"
    rows.sort(key=lambda r: int(r["epoch"]))
    header = (
        "epoch | eps | beta | lr | loss | ce | kl | w | "
        "trClean | trAdv | valClean | valRobust | bestClean@ep | "
        "bestRobust@ep | sec | img/s | vram | flags"
    )
    out = [header]
    for r in rows:
        grad = r.get("grad", {})
        per = grad.get("per_group", {})
        pg = ", ".join(
            f"{n}:{g['post_clip_norm']:.2f}" for n, g in per.items())
        flags = compute_health_flags_string_only(
            epoch=int(r["epoch"]),
            total_epochs=int(r["total_epochs"]),
            num_classes=100,
            val_clean=float(r["val_acc_clean"]),
            val_robust=float(r["val_acc_robust"]),
            kl_trades=float(r["train_kl_trades"]),
            grad=grad,
            best_clean=float(r["best_clean"]),
            best_robust=float(r["best_robust"]),
            best_clean_at_epoch=int(r["best_clean_epoch"]),
            best_robust_at_epoch=int(r["best_robust_epoch"]),
        )
        vclean = float(r.get('val_acc_clean', 0))
        vrobust = float(r.get('val_acc_robust', 0))
        bc = float(r.get('best_clean', 0))
        br = float(r.get('best_robust', 0))
        out.append(
            f"{int(r['epoch']):>3} | {r.get('epsilon',0):.3f} | "
            f"{r.get('beta',0):.1f} | {r.get('lr',0):.2e} | "
            f"{r.get('train_loss_total',0):.4f} | "
            f"{r.get('train_ce_clean',0):.4f} | "
            f"{r.get('train_kl_trades',0):.4f} | "
            f"{r.get('w_trades',0):.2f} | "
            f"{r.get('train_acc_clean',0)*100:.1f} | "
            f"{r.get('train_acc_adv',0)*100:.1f} | "
            f"{vclean*100:.1f} | {vrobust*100:.1f} | "
            f"{bc*100:.1f}@{int(r['best_clean_epoch'])} | "
            f"{br*100:.1f}@{int(r['best_robust_epoch'])} | "
            f"{r.get('epoch_seconds',0):.0f} | "
            f"{r.get('img_per_sec',0):.0f} | "
            f"{r.get('peak_vram_gb',0) if r.get('peak_vram_gb') is not None else 0:.1f} | "
            f"{flags} | pg[{pg}]"
        )
    first = rows[0]
    last = rows[-1]
    delta_clean = last["val_acc_clean"] - first["val_acc_clean"]
    delta_robust = last["val_acc_robust"] - first["val_acc_robust"]
    improved_clean = last["best_clean"] > first["best_clean"]
    improved_robust = last["best_robust"] > first["best_robust"]
    verdict = (
        f"epochs: {len(rows)} | clean Δ={delta_clean*100:+.1f} pp | "
        f"robust Δ={delta_robust*100:+.1f} pp | "
        f"best_clean {first['best_clean']*100:.1f}->"
        f"{last['best_clean']*100:.1f} "
        f"({'improved' if improved_clean else 'flat'}) | "
        f"best_robust {first['best_robust']*100:.1f}->"
        f"{last['best_robust']*100:.1f} "
        f"({'improved' if improved_robust else 'flat'}) | "
        f"config_sha256={rows[0].get('config_sha256','n/a')} | "
        f"git_commit={rows[0].get('git_commit','n/a')} | "
        f"session_id={rows[0].get('session_id','n/a')}")
    return "\n".join(out + ["", verdict])


def compute_health_flags_string_only(**kwargs) -> str:
    from training.run_report import compute_health_flags
    return compute_health_flags(**kwargs)


def compute_health_flags_string_only(**kwargs) -> str:
    from training.run_report import compute_health_flags
    return compute_health_flags(**kwargs)


# ── offline summarizer (Step 5) ─────────────────────────────────────────────

def summarize_phase_from_jsonl(
    jsonl_path: str,
) -> str:
    """Read a phase JSONL log and print a table + plain-text trend verdict.

    Works from the HF-synced file alone (no GPU needed).
    """
    if not os.path.exists(jsonl_path):
        return f"no log at {jsonl_path}"
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return f"{jsonl_path}: empty"
    rows.sort(key=lambda r: int(r["epoch"]))
    header = (
        "epoch | eps | beta | lr | loss | ce | kl | w | "
        "trClean | trAdv | valClean | valRobust | bestClean@ep | "
        "bestRobust@ep | sec | img/s | vram | flags"
    )
    out = [header]
    for r in rows:
        grad = r.get("grad", {})
        per = grad.get("per_group", {})
        pg = ", ".join(
            f"{n}:{g['post_clip_norm']:.2f}" for n, g in per.items())
        flags = compute_health_flags_string_only(
            epoch=int(r["epoch"]),
            total_epochs=int(r["total_epochs"]),
            num_classes=100,
            val_clean=float(r["val_acc_clean"]),
            val_robust=float(r["val_acc_robust"]),
            kl_trades=float(r["train_kl_trades"]),
            grad=grad,
            best_clean=float(r["best_clean"]),
            best_robust=float(r["best_robust"]),
            best_clean_at_epoch=int(r["best_clean_epoch"]),
            best_robust_at_epoch=int(r["best_robust_epoch"]),
        )
        out.append(
            f"{int(r['epoch']):>3} | {r.get('epsilon',0):.3f} | "
            f"{r.get('beta',0):.1f} | {r.get('lr',0):.2e} | "
            f"{r.get('train_loss_total',0):.4f} | "
            f"{r.get('train_ce_clean',0):.4f} | "
            f"{r.get('train_kl_trades',0):.4f} | "
            f"{r.get('w_trades',0):.2f} | {r.get('train_acc_clean',0)*100:.1f} | "
            f"{r.get('train_acc_adv',0)*100:.1f} | {r.get('val_acc_clean',0)*100:.1f} | "
            f"{r.get('val_acc_robust',0)*100:.1f} | {r.get('best_clean',0)*100:.1f}@"
            f"{int(r['best_clean_epoch'])} | {r.get('best_robust',0)*100:.1f}@"
            f"{int(r['best_robust_epoch'])} | {r.get('epoch_seconds',0):.0f} | "
            f"{r.get('img_per_sec',0):.0f} | "
            f"{r.get('peak_vram_gb',0) if r.get('peak_vram_gb') is not None else 0:.1f} | "
            f"{flags} | pg[{pg}]"
        )
    first = rows[0]
    last = rows[-1]
    delta_clean = last["val_acc_clean"] - first["val_acc_clean"]
    delta_robust = last["val_acc_robust"] - first["val_acc_robust"]
    improved_clean = last["best_clean"] > first["best_clean"]
    improved_robust = last["best_robust"] > first["best_robust"]
    verdict = (
        f"epochs: {len(rows)} | clean Δ={delta_clean*100:+.1f} pp | "
        f"robust Δ={delta_robust*100:+.1f} pp | "
        f"best_clean {first['best_clean']*100:.1f}->"
        f"{last['best_clean']*100:.1f} "
        f"({'improved' if improved_clean else 'flat'}) | "
        f"best_robust {first['best_robust']*100:.1f}->"
        f"{last['best_robust']*100:.1f} "
        f"({'improved' if improved_robust else 'flat'}) | "
        f"config_sha256={rows[0].get('config_sha256','n/a')} | "
        f"git_commit={rows[0].get('git_commit','n/a')} | "
        f"session_id={rows[0].get('session_id','n/a')}")
    return "\n".join(out + ["", verdict])


def compute_health_flags_string_only(**kwargs) -> str:
    from training.run_report import compute_health_flags
    return compute_health_flags(**kwargs)
