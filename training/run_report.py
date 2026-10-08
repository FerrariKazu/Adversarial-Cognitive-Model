"""
Run-time reporting and diagnostics for the Gen-1 foundation trainer (Agent J1).

LOGGING AND DIAGNOSTICS ONLY — this module does NOT change any training math,
loss, optimizer, schedule, or checkpoint contents. It adds:
 * an explicit clean + robust per-epoch validation path (the current epoch
   line only ever printed clean top-1 under the name "val_acc");
 * a compact per-epoch report block with every number labeled;
 * health flags derived from logged numbers;
 * a durable per-phase JSONL log and a resume banner that verifies it.

Reuses existing harness contracts where possible:
 * PGD attack / TRADES loss from training/adv_curriculum.py (ported Gen-0).
 * Per-group gradient norms via the existing OptimizerGroupRegistry
   (noesis_vision.core.multi_group_optimizer), whose
   clip_grad_per_group() already computes per-group norms.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry

# ── column-meaning legend (printed once per phase) ─────────────────────────

EPOCH_BLOCK_LEGEND = """
Per-epoch block fields (every number explicitly labeled):
  TRAIN
    loss_total   = mean training loss over all train batches this epoch
                   (under TRADES: w_trades * mean(CE(clean) + beta*KL(adv||clean));
                    under --clean-only: mean(CE(clean)))
    ce_clean     = mean CE(clean) over train batches
    kl_trades    = mean KL(adv||clean) over train batches (0.0 under --clean-only)
    w_trades     = TRADES scalar multiplier in effect this epoch
    train_acc[clean] = mean per-batch clean top-1 on the TRAINING loader
    train_acc[adv]   = mean per-batch PGD-attacked top-1 on the TRAINING loader
  VAL
    val_acc[CLEAN]   = clean top-1 on the (full or subset) VAL loader, eval mode
    val_acc[ROBUST]  = PGD-attacked top-1 on a FIXED deterministic val subset,
                       same images + seed every epoch, eval mode
  BEST
    best_clean   = best val_acc[CLEAN] seen so far this phase, and the epoch
    best_robust  = best val_acc[ROBUST] seen so far this phase, and the epoch
    checkpoint criterion in use = which metric currently selects best.pth
  GRAD
    pre_clip_norm / post_clip_norm = global L2 norm of all .grad before/after
                                      the existing per-group clip
    clipped = whether ANY group was clipped this step (Y/N; aggregate over epoch)
    per group = per-group pre/post clip norms from the existing registry
  SPEED   = wall seconds for the epoch block (sec/epoch), images/sec over the
           block (img/s), peak VRAM GB at the end of the block, ETA to end of
           current phase
  HEALTH  = flags that fired this epoch (see Step 2 legend); "ok" if none
""".strip()

HEALTH_LEGEND = """
Health flags (only the ones that fire are printed; the threshold is printed
next to each flag):
  CHANCE-LEVEL   val_acc[CLEAN] within 2x of 1/num_classes after epoch 5
                 (threshold: clean_val <= 2.0 / num_classes at epoch > 5)
  CLIP-SATURATED post_clip_norm == the group's clip norm in >90% of steps this
                 epoch (the historical starvation signature). Reported per group.
  TRADES-DEAD    kl_trades < 1e-6, or its gradient share ~0
                 (threshold: epoch-mean kl_trades < 1e-6)
  STALL          no improvement in either best_* for 8 epochs
                 (threshold: max(best_clean, best_robust) unchanged for 8 epochs)
  GAP            val_acc[CLEAN] - val_acc[ROBUST] shown always; flagged only if
                 robust > clean (impossible, indicates a bug)
""".strip()


# ── explicit clean + robust validation ──────────────────────────────────────

def robust_val_accuracy(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    eps: float,
    pgd_steps: int,
    *,
    n_subset: int = 512,
    seed: int = 17,
    batch_size: int = 64,
) -> Tuple[float, int, float]:
    """Robust (PGD) val accuracy on a fixed deterministic subset, eval mode.

    Returns (robust_top1, n_used, seconds). Subset is rebuilt from a pinned seed
    every call so epoch-to-epoch numbers are comparable. No permanent state change.
    """
    was_training = model.training
    model.eval()
    try:
        ds = val_loader.dataset
        n_total = len(ds)
        if n_total <= n_subset:
            chosen = list(range(n_total))
        else:
            g = torch.Generator().manual_seed(seed)
            chosen = torch.randperm(n_total, generator=g)[:n_subset].tolist()
        loader = DataLoader(
            Subset(ds, chosen), batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False)
        n_used = len(loader.dataset)
        t0 = time.perf_counter()
        correct = total = 0
        from evaluation.clean_and_robust import generic_pgd
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_adv = generic_pgd(model, x, y, eps=eps, steps=pgd_steps)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(total, 1), n_used, time.perf_counter() - t0
    finally:
        if was_training:
            model.train()


def clean_val_accuracy(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    n_subset: Optional[int] = None,
    seed: int = 17,
    batch_size: int = 64,
) -> Tuple[float, int, float]:
    """Clean val accuracy on full val loader (or a deterministic subset).

    Returns (clean_top1, n_used, seconds). Same state-safety as robust path.
    """
    was_training = model.training
    model.eval()
    try:
        ds = val_loader.dataset
        n_total = len(ds)
        if n_subset is not None and n_total > n_subset:
            g = torch.Generator().manual_seed(seed)
            chosen = torch.randperm(n_total, generator=g)[:n_subset].tolist()
            loader = DataLoader(
                Subset(ds, chosen), batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False)
            n_used = len(loader.dataset)
        else:
            loader = val_loader
            n_used = n_total
        t0 = time.perf_counter()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(total, 1), n_used, time.perf_counter() - t0
    finally:
        if was_training:
            model.train()


# ── gradient norms (reuse existing registry; do NOT reimplement clipping) ───

class GradNormCollector:
    """Captures pre-clip/post-clip per-group norms using OptimizerGroupRegistry.

    The registry already computes per-group clip_grad_norm_ inside
    clip_grad_per_group(); we observe norms before and after that call so we
    can report pre/post clipped per group without reimplementing the clip.
    """

    def __init__(self, registry: OptimizerGroupRegistry) -> None:
        self.registry = registry
        self.steps_total = 0
        self.steps_any_clipped = 0
        self.pre_total_sum = 0.0
        self.post_total_sum = 0.0
        self.per_group: Dict[str, Dict[str, float]] = {}

    def _norm(self, params: Sequence[nn.Parameter]) -> float:
        ps = [p.grad.detach() for p in params if p.grad is not None]
        if not ps:
            return 0.0
        return float(torch.stack([t.norm(2) for t in ps]).norm(2).item())

    def record_pre(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for g in self.registry.registered:
            name = g["name"]
            pre = self._norm(g["params"])
            out[name] = pre
            stats = self.per_group.setdefault(
                name, {"pre_sum": 0.0, "post_sum": 0.0,
                       "clipped_count": 0, "steps": 0})
            stats["pre_sum"] += pre
            stats["steps"] += 1
        self.pre_total_sum += self._norm(
            [p for g in self.registry.registered for p in g["params"]])
        return out

    def record_post(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for g in self.registry.registered:
            name = g["name"]
            post = self._norm(g["params"])
            out[name] = post
            stats = self.per_group.setdefault(
                name, {"pre_sum": 0.0, "post_sum": 0.0,
                       "clipped_count": 0, "steps": 0})
            stats["post_sum"] += post
        self.post_total_sum += self._norm(
            [p for g in self.registry.registered for p in g["params"]])
        self.steps_total += 1
        return out

    def any_clipped_this_step(
        self, pre: Dict[str, float], post: Dict[str, float]) -> bool:
        for name in pre:
            if name not in post:
                continue
            if not math.isclose(pre[name], post[name], rel_tol=1e-6, abs_tol=1e-6):
                return True
        return False

    def finish(self) -> Dict[str, Any]:
        per_group_out: Dict[str, Any] = {}
        for name, s in self.per_group.items():
            steps = max(s["steps"], 1)
            per_group_out[name] = {
                "pre_clip_norm": s["pre_sum"] / steps,
                "post_clip_norm": s["post_sum"] / steps,
                "clipped_frac": s["clipped_count"] / steps,
            }
        return {
            "steps": self.steps_total,
            "steps_any_clipped": self.steps_any_clipped,
            "pre_clip_norm": self.pre_total_sum / max(self.steps_total, 1),
            "post_clip_norm": self.post_total_sum / max(self.steps_total, 1),
            "clipped_frac_any": self.steps_any_clipped / max(self.steps_total, 1),
            "per_group": per_group_out,
        }


# ── epoch report block ───────────────────────────────────────────────────────

def format_epoch_block(
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
    eta_phase_end: Optional[float],
    health: str,
) -> str:
    adv_tag = (
        "clean-only"
        if (point is None or getattr(point, "adversarial", True) is False)
        else f"eps={point.eps:.3f} beta={point.beta:.1f} pgd={point.pgd_steps}"
    )
    eps = getattr(point, "eps", 0.0)
    lines = [
        f"── EPOCH {epoch}/{total_epochs} · phase={phase} · ε={eps:.3f} · "
        f"β={getattr(point,'beta',0.0):.1f} · lr={lr:.2e} ──",
        f"TRAIN  loss_total={tr_loss:.4f}  ce_clean={ce_clean:.4f}  "
        f"kl_trades={kl_trades:.4f}  (w_trades={w_trades:.3f})",
        f"       train_acc[clean]={train_clean_acc*100:.2f}%   "
        f"train_acc[adv]={train_adv_acc*100:.2f}%",
        f"VAL    val_acc[CLEAN]={val_clean*100:.2f}% ({clean_n} imgs, eval mode)",
        f"       val_acc[ROBUST, PGD-{robust_n}@ε={eps:.3f}]={val_robust*100:.2f}%",
        f"       (robust: fixed {robust_n}-img subset, seed-pinned, took {robust_secs:.1f}s)",
        f"BEST   best_clean={best_clean*100:.2f}% @ep{best_clean_epoch}   "
        f"best_robust={best_robust*100:.2f}% @ep{best_robust_epoch}",
        f"       checkpoint criterion in use: {criterion}",
        f"GRAD   pre_clip_norm={grad['pre_clip_norm']:.3f}  "
        f"post_clip_norm={grad['post_clip_norm']:.3f}  "
        f"clipped={'Y' if grad['clipped_frac_any'] > 0.0 else 'N'}",
    ]
    for name, g in grad.get("per_group", {}).items():
        lines.append(
            f"       {name}: pre_clip_norm={g['pre_clip_norm']:.3f}  "
            f"post_clip_norm={g['post_clip_norm']:.3f}  "
            f"clipped_frac={g['clipped_frac']:.2f}")
    vram = f"{peak_vram_gb:.1f}" if peak_vram_gb is not None else "n/a"
    eta = f"{eta_phase_end:.0f}s" if eta_phase_end is not None else "n/a"
    lines += [
        f"SPEED  {epoch_seconds:.1f}s/epoch · {img_per_sec:.0f} img/s · "
        f"peak VRAM {vram}GB · ETA to phase end {eta}",
        f"HEALTH {health}",
    ]
    return "\n".join(lines)


def compute_health_flags(
    epoch: int,
    total_epochs: int,
    num_classes: int,
    val_clean: float,
    val_robust: float,
    kl_trades: float,
    grad: Dict[str, Any],
    best_clean: float,
    best_robust: float,
    best_clean_at_epoch: int,
    best_robust_at_epoch: int,
) -> str:
    flags: List[str] = []
    if epoch > 5 and val_clean <= 2.0 / num_classes:
        flags.append(
            f"CHANCE-LEVEL(clean_val_acc={val_clean*100:.2f}% <= "
            f"2/num_classes={2.0/num_classes:.4f})")
    if grad.get("clipped_frac_any", 0.0) >= 0.9:
        flags.append(
            f"CLIP-SATURATED(any_group clipped_frac="
            f"{grad['clipped_frac_any']:.2f} >= 0.90)")
    for name, g in grad.get("per_group", {}).items():
        if g["clipped_frac"] >= 0.9:
            flags.append(
                f"CLIP-SATURATED(group={name} clipped_frac="
                f"{g['clipped_frac']:.2f} >= 0.90)")
    if kl_trades < 1e-6:
        flags.append(
            f"TRADES-DEAD(epoch_mean_kl_trades={kl_trades:.2e} < 1e-6)")
    if (epoch - best_clean_at_epoch) >= 8 and (
        epoch - best_robust_at_epoch) >= 8:
        flags.append(
            f"STALL(no improvement in best_clean @ep{best_clean_at_epoch} or "
            f"best_robust @ep{best_robust_at_epoch} for >= 8 epochs)")
    if val_robust > val_clean + 1e-6:
        flags.append(
            f"GAP(robust>clean, robust={val_robust*100:.2f}% "
            f"clean={val_clean*100:.2f}%, Δ={(val_robust-val_clean)*100:.2f}pp — "
            f"impossible under correct eval; indicates a bug)")
    return " · ".join(flags) if flags else "ok"


def default_phase_jsonl_path(report_dir: str, phase: str) -> str:
    return os.path.join(report_dir, f"{phase}_epoch_log.jsonl")
