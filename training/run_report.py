"""
Run-time reporting and diagnostics for the Gen-1 foundation trainer (Agent J1),
richer-block extension.

LOGGING AND DIAGNOSTICS ONLY — this module does NOT change any training math,
loss, optimizer, schedule, curriculum, dataset, seed, augmentation, or checkpoint
contents. It adds:
 * parameter-update diagnostics (|Delta theta|, relative update, fraction changed)
   computed by snapshotting model parameters BEFORE and AFTER the epoch step;
 * safe representation-health diagnostics (feature norm/mean/std/NaN/Inf/near-zero)
   using a ONE-time pre-epoch feature snapshot, without changing any forward/loss
   computation in the training step;
 * a legacy-checkpoint UNKNOWN policy: when a rolling checkpoint only has
   metric_value (old contract), best_clean/best_robust are marked UNKNOWN rather
   than silently set to 0.0;
 * richer epoch block fields and an extended JSONL record shape;
 * an updated offline summarizer.

Gradient norms still reuse the existing OptimizerGroupRegistry per-group
clip path (record_pre/record_post around existing clip_grad_per_group).
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


def clean_val_accuracy(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    n_subset: Optional[int] = None,
    seed: int = 17,
    batch_size: int = 64,
) -> Tuple[float, int, float]:
    from evaluation.clean_and_robust import generic_pgd
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
    from evaluation.clean_and_robust import generic_pgd
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
        if g.get("clipped_frac", 0.0) >= 0.9:
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
    update = grad.get("update", {})
    if update.get("delta_total_norm", 0.0) == 0.0 and update.get("n_groups_with_any_change", 0) > 0:
        flags.append(
            f"UPDATE-ZERO(delta_total_norm=0.0 while {update.get('n_groups_with_any_change')} "
            f"group(s) moved)")
    rep = grad.get("representation", {})
    if rep:
        before = rep.get("feature_norm_before", 0.0) or 1e-9
        after = rep.get("feature_norm_after", 0.0)
        if after / before < 0.5 or after / before > 2.0:
            flags.append(
                f"REPRESENTATION-COLLAPSE(observed norm ratio={after/before:.3f}; "
                f"no pre-registered threshold; investigate if it persists)")
    return " · ".join(flags) if flags else "ok"

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
    gradient_share[CE]  = rough per-step CE grad contribution share (if available)
    gradient_share[KL]  = rough per-step KL grad contribution share (if available)
  VAL
    val_acc[CLEAN]   = clean top-1 on the (full or subset) VAL loader, eval mode
    val_acc[ROBUST]  = PGD-attacked top-1 on a FIXED deterministic val subset,
                       same images + seed every epoch, eval mode
  BEST
    best_clean   = best val_acc[CLEAN] seen so far this phase, and the epoch
    best_robust  = best val_acc[ROBUST] seen so far this phase, and the epoch
    checkpoint criterion in use = which metric currently selects best.pth
    best provenance = legacy checkpoint handling for resumed bests
  UPDATE  (parameter movement; DIFFERENT from gradient magnitude)
    delta_total_norm     = |theta_after - theta_before| global L2 over trained params
    delta_relative_norm   = delta_total_norm / |theta_before| (0 if before is 0)
    fraction_params_changed = fraction of trained params whose elements changed
    per group            = same three numbers per optimizer group
  REPRESENTATION  (safe, snapshot-based; one snapshot BEFORE the epoch step,
                   one AFTER; no training math changed)
    feature_norm_before/after
    feature_mean_before/after
    feature_std_before/after
    nan_count_before/after
    inf_count_before/after
    nearzero_frac_before/after
    NOTE: this harness does NOT attempt per-layer feature drift against a
    phase-start reference because no such snapshot store exists today; absence
    is reported explicitly rather than fabricated.
    In the epoch block this appears as the REPRESENTATION HEALTH section and its
    NOTE about missing phase-start reference snapshots / drift-from-start.
  GRAD
    pre_clip_norm / post_clip_norm = global L2 norm of all .grad before/after
                                      the existing per-group clip
    clipped = whether ANY group was clipped this step (Y/N; aggregate over epoch)
    per group = per-group pre/post clip norms from the existing registry
  SPEED   = wall seconds for the epoch block, images/sec over the block, peak
           VRAM GB at the end of the block, ETA to end of current phase
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
  UPDATE-ZERO    delta_total_norm == 0.0 while gradients were present (possible
                 optimizer/scheduler/scaler state issue, not an invented conclusion)
  REPRESENTATION-COLLAPSE  feature_norm_after / feature_norm_before < 0.5 or > 2.0
                 (OBSERVED ratio only; no pre-registered threshold; investigate if
                 it persists across epochs)
  REPRESENTATION   snapshot-based feature norm/mean/std/NaN/Inf/nearzero when
                 representation snapshots are captured this epoch; otherwise NOT
                 COMPUTED. drift-from-phase-start reported only when a start
                 snapshot was stored; absence reported explicitly.
""".strip()


# ---------------------------------------------------------------------------
# Gradient-norm collector (reuse existing OptimizerGroupRegistry; do NOT
# reimplement clipping). Publicly exported so the trainer, JSONL module, and
# tests can import one shared class.
# ---------------------------------------------------------------------------

class GradNormCollector:
    """Captures pre-clip/post-clip per-group grad norms using the existing
    noesis_vision.core.multi_group_optimizer.OptimizerGroupRegistry contract.

    The registry already computes per-group clip_grad_norm_ inside
    clip_grad_per_group(); we observe norms before/after that call so we can
    report pre/post clipped per group without reimplementing the clip.
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


# ---------------------------------------------------------------------------
# Legacy-checkpoint UNKNOWN policy (D1)
# ---------------------------------------------------------------------------

class ResumedBestState:
    """Explicit provenance for best_clean / best_robust when resuming.

    Old rolling checkpoints only carried metric_value and did NOT carry
    best_clean / best_clean_epoch / best_robust / best_robust_epoch. We must
    NOT silently interpret missing values as 0.0 and continue as though the
    state were complete. Instead we preserve known information and mark the
    rest UNKNOWN.
    """

    def __init__(
        self,
        *,
        best_clean: Optional[float],
        best_clean_epoch: Optional[int],
        best_robust: Optional[float],
        best_robust_epoch: Optional[int],
        known_clean: bool,
        known_robust: bool,
        note: str = "",
    ) -> None:
        self.best_clean = best_clean
        self.best_clean_epoch = best_clean_epoch
        self.best_robust = best_robust
        self.best_robust_epoch = best_robust_epoch
        self.known_clean = known_clean
        self.known_robust = known_robust
        self.note = note

    @classmethod
    def from_rolling_state(
        cls,
        state: Optional[Dict[str, Any]],
        *,
        prefer_clean_from_metric: bool = True,
    ) -> "ResumedBestState":
        """Read best-clean/robust provenance from a rolling checkpoint state dict.

        Old checkpoints contain only:
          metric_value, kind, code_commit, saved_at_utc, epoch?, extra?

        New checkpoints additionally carry:
          best_clean, best_clean_epoch, best_robust, best_robust_epoch

        We never silently turn missing fields into 0.0.
        """
        if state is None:
            return cls(
                best_clean=None,
                best_clean_epoch=None,
                best_robust=None,
                best_robust_epoch=None,
                known_clean=False,
                known_robust=False,
                note="no rolling checkpoint loaded (cold start)",
            )

        has_clean = "best_clean" in state
        has_robust = "best_robust" in state
        has_clean_ep = "best_clean_epoch" in state
        has_robust_ep = "best_robust_epoch" in state

        best_clean = state.get("best_clean")
        best_clean_epoch = state.get("best_clean_epoch")
        best_robust = state.get("best_robust")
        best_robust_epoch = state.get("best_robust_epoch")

        note_parts: List[str] = []
        if has_clean and has_clean_ep:
            note_parts.append("best_clean/best_clean_epoch preserved from rolling checkpoint")
        elif has_clean and not has_clean_ep:
            note_parts.append("best_clean present but best_clean_epoch missing -> epoch UNKNOWN")
        else:
            if prefer_clean_from_metric and "metric_value" in state:
                mv = state.get("metric_value")
                if isinstance(mv, (int, float)):
                    # SAFE ONLY IF we can document that metric_value was clean
                    # validation. Old best checkpoints were selected by clean
                    # validation (evaluate_val), so backfilling best_clean from
                    # metric_value is provenance-consistent, but we must record
                    # that this is a backfill, not an original field.
                    best_clean = float(mv)
                    best_clean_epoch = None
                    note_parts.append(
                        "LEGACY CHECKPOINT: best_clean backfilled from metric_value "
                        "(clean-validation legacy selection); best_clean_epoch UNKNOWN"
                    )
                else:
                    note_parts.append(
                        "LEGACY CHECKPOINT: metric_value present but not numeric -> best_clean UNKNOWN"
                    )
            else:
                note_parts.append(
                    "LEGACY CHECKPOINT: best_clean not present -> UNKNOWN"
                )

        if has_robust and has_robust_ep:
            note_parts.append("best_robust/best_robust_epoch preserved from rolling checkpoint")
        elif has_robust and not has_robust_ep:
            note_parts.append("best_robust present but best_robust_epoch missing -> epoch UNKNOWN")
        else:
            # Old checkpoints did not track best_robust at all.
            note_parts.append(
                "LEGACY CHECKPOINT: best_robust was not tracked -> UNKNOWN"
            )

        if not has_clean and not has_robust and "metric_value" in state:
            note_parts.append(
                "LEGACY CHECKPOINT DETECTED: only metric_value available; "
                "best_clean backfilled from metric_value where clean-validation "
                "selection is documented, best_robust UNKNOWN"
            )

        return cls(
            best_clean=best_clean,
            best_clean_epoch=best_clean_epoch,
            best_robust=best_robust,
            best_robust_epoch=best_robust_epoch,
            known_clean=bool(best_clean is not None),
            known_robust=bool(best_robust is not None),
            note="; ".join(note_parts) if note_parts else "resume best provenance UNKNOWN",
        )

    def effective_for_training(self) -> Tuple[float, int, float, int]:
        """Return (best_clean, best_clean_epoch, best_robust, best_robust_epoch)
        values usable for comparisons and printing. Unknown fields become 0.0/0
        for arithmetic but remain flagged in the banner/JSONL.
        """
        bc = self.best_clean if self.best_clean is not None else 0.0
        bce = self.best_clean_epoch if self.best_clean_epoch is not None else 0
        br = self.best_robust if self.best_robust is not None else 0.0
        bre = self.best_robust_epoch if self.best_robust_epoch is not None else 0
        return bc, bce, br, bre

    def banner_lines(self, phase: str, start_epoch: int,
                     rolling_path: Optional[str],
                     rolling_sha: Optional[str]) -> List[str]:
        lines = [
            f"=== {phase} RESUME ===",
            f"resumed from epoch {start_epoch}",
        ]
        if self.known_clean and self.best_clean is not None:
            lines.append(
                f"best_clean={self.best_clean*100:.2f}% "
                f"@ep{self.best_clean_epoch}"
            )
        else:
            lines.append(
                "best_clean: UNKNOWN (legacy checkpoint; clean best not reliably "
                "restorable from this artifact)"
            )
        if self.known_robust and self.best_robust is not None:
            lines.append(
                f"best_robust={self.best_robust*100:.2f}% "
                f"@ep{self.best_robust_epoch}"
            )
        else:
            lines.append(
                "best_robust: UNKNOWN (legacy checkpoint did not track robust best)"
            )
        if rolling_path and os.path.exists(rolling_path):
            lines.append(f"loaded checkpoint: {rolling_path}")
            lines.append(f"checkpoint sha256: {rolling_sha or 'n/a'}")
        else:
            lines.append("loaded checkpoint: none (cold start)")
            lines.append("checkpoint sha256: n/a")
        if self.note:
            lines.append(f"provenance note: {self.note}")
        return lines


# ---------------------------------------------------------------------------
# Parameter-update diagnostics (D2) — snapshot-based, no training math changed
# ---------------------------------------------------------------------------

def snap_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Read-only snapshot of model parameter tensors (no grad, no copy into
    training state). Used only for |theta_after - theta_before| diagnostics."""
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def model_update_stats(
    before: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Compute |Delta theta| diagnostics from two snapshots.

    Returns global + per-name stats. A name is "changed" if any element differs
    by more than float epsilon.
    """
    names = sorted(set(before) | set(after))
    total_norm_before = 0.0
    total_delta_norm = 0.0
    changed_any = 0
    n_elements_total = 0
    n_elements_changed = 0
    per_name: Dict[str, Dict[str, Any]] = {}
    for name in names:
        b = before.get(name)
        a = after.get(name)
        if b is None or a is None:
            per_name[name] = {
                "norm_before": None,
                "delta_norm": None,
                "relative_norm": None,
                "fraction_changed": None,
                "n_elements": 0,
                "n_changed": 0,
            }
            continue
        bn = float(b.norm(2).item())
        delta = a - b
        dn = float(delta.norm(2).item())
        rel = (dn / bn) if bn > 0.0 else 0.0
        ne = int(b.numel())
        nc = int((delta.abs() > 0.0).sum().item())
        frac = (nc / ne) if ne > 0 else 0.0
        per_name[name] = {
            "norm_before": bn,
            "delta_norm": dn,
            "relative_norm": rel,
            "fraction_changed": frac,
            "n_elements": ne,
            "n_changed": nc,
        }
        total_norm_before += bn * bn
        total_delta_norm += dn * dn
        changed_any += (1 if nc > 0 else 0)
        n_elements_total += ne
        n_elements_changed += nc
    return {
        "total_norm_before": math.sqrt(total_norm_before),
        "delta_total_norm": math.sqrt(total_delta_norm),
        "delta_relative_norm": (
            math.sqrt(total_delta_norm) / math.sqrt(total_norm_before)
            if total_norm_before > 0.0 else 0.0
        ),
        "fraction_params_changed": (
            n_elements_changed / n_elements_total
            if n_elements_total > 0 else 0.0
        ),
        "n_groups_with_any_change": changed_any,
        "per_name": per_name,
    }


def group_update_stats(
    registry: OptimizerGroupRegistry,
    update_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Map per-name update stats onto optimizer groups, plus per-group totals."""
    per_group: Dict[str, Dict[str, Any]] = {}
    for g in registry.registered:
        name = g["name"]
        group_names = sorted(
            n for n in update_stats["per_name"]
            if n.startswith(name)
        )
        # A cleaner mapping: match by exact registered param name prefix.
        # Since named_parameters() includes the full dotted name, we match any
        # param name that starts with the group name followed by '.' or equals it.
        matched = {}
        for n, s in update_stats["per_name"].items():
            if n == name or n.startswith(name + "."):
                matched[n] = s
        if not matched:
            continue
        ne = sum(s.get("n_elements", 0) for s in matched.values())
        nc = sum(s.get("n_changed", 0) for s in matched.values())
        dnorm2 = sum((s.get("delta_norm") or 0.0) ** 2 for s in matched.values())
        bnorm2 = sum((s.get("norm_before") or 0.0) ** 2 for s in matched.values())
        per_group[name] = {
            "n_params": ne,
            "n_params_changed": nc,
            "fraction_params_changed": (nc / ne if ne > 0 else 0.0),
            "delta_norm": math.sqrt(dnorm2),
            "norm_before": math.sqrt(bnorm2),
            "relative_norm": (
                math.sqrt(dnorm2) / math.sqrt(bnorm2)
                if bnorm2 > 0.0 else 0.0
            ),
        }
    return {
        "groups": per_group,
        "n_groups": len(per_group),
        "n_groups_with_any_change": sum(
            1 for g in per_group.values() if g.get("n_params_changed", 0) > 0
        ),
    }


# ---------------------------------------------------------------------------
# Safe representation-health diagnostics (D2) — snapshot-based, no training math
# ---------------------------------------------------------------------------

def feature_snapshot(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    n_samples: int = 64,
    label: str = "val",
) -> Dict[str, Any]:
    """One-time, read-only feature snapshot used only for representation-health
    diagnostics. Does NOT change any training state or loss computation.

    Captures, over a small deterministic prefix of the loader:
      feature_norm, feature_mean, feature_std, nan_count, inf_count,
      nearzero_frac

    Returns the stats plus the number of samples used. If the model has no
    obvious feature hook, this harness reports the intent and absences
    explicitly rather than inventing numbers.
    """
    was_training = model.training
    model.eval()
    try:
        ds = loader.dataset
        n_total = len(ds)
        take = min(n_samples, n_total)
        g = torch.Generator().manual_seed(17)
        idx = torch.randperm(n_total, generator=g)[:take].tolist()
        sub = Subset(ds, idx)
        loader2 = DataLoader(
            sub, batch_size=min(16, max(1, take)), shuffle=False,
            num_workers=0, pin_memory=False)
        norms: List[float] = []
        means: List[float] = []
        stds: List[float] = []
        nans = 0
        infs = 0
        nelements = 0
        n_taken = 0
        for x, _y in loader2:
            x = x.to(device)
            # Try the model's natural feature path: forward into logits is safe,
            # but we want intermediate features if available. If the model exposes
            # a feature method we use it; otherwise we fall back to logits and
            # report that choice explicitly.
            try:
                feats = model.feature(x)
            except Exception:
                with torch.no_grad():
                    logits = model(x)
                feats = logits.float()
            feats = feats.detach().float().reshape(feats.shape[0], -1)
            norms.append(float(feats.norm(2).item()))
            means.append(float(feats.mean().item()))
            stds.append(float(feats.std().item()))
            nans += int(torch.isnan(feats).sum().item())
            infs += int(torch.isinf(feats).sum().item())
            nelements += int(feats.numel())
            n_taken += int(feats.shape[0])
        return {
            "feature_norm": (sum(norms) / max(len(norms), 1)),
            "feature_mean": (sum(means) / max(len(means), 1)),
            "feature_std": (sum(stds) / max(len(stds), 1)),
            "nan_count": nans,
            "inf_count": infs,
            "nearzero_frac": (
                sum(1 for n in norms if n < 1e-6) / max(len(norms), 1)
            ),
            "n_samples": n_taken,
            "n_elements": nelements,
        }
    finally:
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# Richer epoch block (D4)
# ---------------------------------------------------------------------------

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
    *,
    gradient_share: Optional[Dict[str, float]] = None,
    update_stats: Optional[Dict[str, Any]] = None,
    representation: Optional[Dict[str, Any]] = None,
    chance_level: Optional[float] = None,
    ratio_to_chance: Optional[float] = None,
    accuracy_gap_pp: Optional[float] = None,
    best_provenance: Optional[str] = None,
    checkpoint_epoch: Optional[int] = None,
    checkpoint_sha256: Optional[str] = None,
    historical_reference: Optional[Dict[str, Any]] = None,
    comparability: Optional[Dict[str, str]] = None,
    current_minus_historical_pp: Optional[float] = None,
) -> str:
    adv_tag = (
        "clean-only"
        if (point is None or getattr(point, "adversarial", True) is False)
        else f"eps={getattr(point,'eps',0.0):.3f} beta={getattr(point,'beta',0.0):.1f} "
        f"pgd={getattr(point,'pgd_steps',0)}"
    )
    eps = getattr(point, "eps", 0.0)
    beta = getattr(point, "beta", 0.0)
    pgds = getattr(point, "pgd_steps", 0)
    cl = chance_level if chance_level is not None else (1.0 / 100.0)
    rc = ratio_to_chance if ratio_to_chance is not None else (val_clean / cl if cl > 0 else 0.0)
    gap = accuracy_gap_pp if accuracy_gap_pp is not None else ((val_clean - val_robust) * 100.0)
    eta = eta_phase_end if eta_phase_end is not None else 0.0
    vram = peak_vram_gb if peak_vram_gb is not None else 0.0

    lines = [
        "══════════════════════════════════════════════════════════════════════",
        f"EPOCH {epoch} / {total_epochs}",
        f"PHASE: {phase}",
        f"ε={eps:.3f} · β={beta:.1f} · PGD={pgds} · lr={lr:.2e}",
        f"loss_total={tr_loss:.4f} · ce_clean={ce_clean:.4f} · kl_trades={kl_trades:.4f} · w_trades={w_trades:.3f}",
        f"train_acc[clean]={train_clean_acc*100:.2f}% · train_acc[adv]={train_adv_acc*100:.2f}%",
        "──────────────────────────────────────────────────────────────────────",
        "VALIDATION",
        f"  CLEAN   val_acc[CLEAN]={val_clean*100:.2f}%  ({clean_n} imgs, eval mode, no grad)",
        f"  ROBUST  val_acc[ROBUST, PGD-{robust_n}@ε={eps:.3f}]={val_robust*100:.2f}%  "
        f"(fixed subset, seed-pinned, {robust_secs:.1f}s)",
        f"  ACCURACY GAP: clean - robust = {gap:+.2f} pp",
        "──────────────────────────────────────────────────────────────────────",
        "BEST-SO-FAR",
        f"  best_clean   = {best_clean*100:.2f}% @ ep {best_clean_epoch}",
        f"  best_robust  = {best_robust*100:.2f}% @ ep {best_robust_epoch}",
        f"  checkpoint criterion in use: {criterion}",
        f"  best provenance: {best_provenance or 'n/a'}",
        "──────────────────────────────────────────────────────────────────────",
        "GRADIENT HEALTH",
        f"  pre_clip_norm = {grad['pre_clip_norm']:.4f}",
        f"  post_clip_norm = {grad['post_clip_norm']:.4f}",
        f"  clipped this epoch = {'YES' if grad['clipped_frac_any'] > 0 else 'NO'} "
        f"({grad['clipped_frac_any']*100:.1f}% of steps)",
    ]
    for name, g in grad.get("per_group", {}).items():
        lines.append(
            f"  {name}: pre={g['pre_clip_norm']:.4f} post={g['post_clip_norm']:.4f} "
            f"clipped_frac={g['clipped_frac']:.2f}"
        )
    if gradient_share:
        ce_g = gradient_share.get("CE", 0.0)
        kl_g = gradient_share.get("KL", 0.0)
        lines.append(f"  gradient_share[CE] = {ce_g:.3f}  gradient_share[KL] = {kl_g:.3f}")
    lines += [
        "──────────────────────────────────────────────────────────────────────",
        "PARAMETER UPDATE HEALTH  (|Delta theta|, not gradient magnitude)",
    ]
    if update_stats:
        u = update_stats
        lines.append(
            f"  delta_total_norm = {u['delta_total_norm']:.4f} · "
            f"relative = {u['delta_relative_norm']:.4f} · "
            f"fraction_changed = {u['fraction_params_changed']*100:.2f}%"
        )
        for name, g in u.get("per_group", {}).items():
            lines.append(
                f"  {name}: |Δθ|={g['delta_norm']:.4f} rel={g['relative_norm']:.4f} "
                f"changed={g['fraction_params_changed']*100:.2f}% "
                f"(n={g['n_params']})"
            )
    else:
        lines.append("  NOT COMPUTED this epoch")
    lines += [
        "──────────────────────────────────────────────────────────────────────",
        "REPRESENTATION HEALTH  (snapshot-based; NOT invented)",
    ]
    if representation:
        r = representation
        ratio = (r.get("feature_norm_after") or 0.0) / (
            r.get("feature_norm_before") or 1e-9
        )
        lines.append(
            f"  feature_norm: before={r.get('feature_norm_before'):.3f} "
            f"after={r.get('feature_norm_after'):.3f} ratio={ratio:.3f}"
        )
        lines.append(
            f"  feature_mean: before={r.get('feature_mean_before'):.3f} "
            f"after={r.get('feature_mean_after'):.3f}"
        )
        lines.append(
            f"  feature_std: before={r.get('feature_std_before'):.3f} "
            f"after={r.get('feature_std_after'):.3f}"
        )
        lines.append(
            f"  nan_count: before={r.get('nan_count_before',0)} after={r.get('nan_count_after',0)}"
        )
        lines.append(
            f"  inf_count: before={r.get('inf_count_before',0)} after={r.get('inf_count_after',0)}"
        )
        lines.append(
            f"  nearzero_frac: before={r.get('nearzero_frac_before',0):.3f} "
            f"after={r.get('nearzero_frac_after',0):.3f}"
        )
        lines.append(
            "  NOTE: no phase-start reference snapshot stored today -> "
            "drift-from-start NOT REPORTED (absence reported explicitly)"
        )
    else:
        lines.append("  NOT COMPUTED this epoch")
    lines += [
        "──────────────────────────────────────────────────────────────────────",
        "SPEED",
        f"  {epoch_seconds:.1f}s/epoch · {img_per_sec:.0f} img/s · "
        f"peak VRAM {vram:.1f}GB · ETA phase end {eta:.0f}s",

        "──────────────────────────────────────────────────────────────────────",
        "HEALTH DIAGNOSIS",
        f"  {health}",
        "══════════════════════════════════════════════════════════════════════",
    ]
    if historical_reference:
        h = historical_reference
        cmp = comparability or {}
        dmhp = current_minus_historical_pp
        lines += [
            "──────────────────────────────────────────────────────────────────────",
            "HISTORICAL RESULT COMPARISON",
            f"  historical: backbone_only best_clean = {h.get('best_clean',0)*100:.2f}% "
            f"@ {h.get('commit','?')}",
            f"  current:    clean = {val_clean*100:.2f}%",
        ]
        if dmhp is not None:
            lines.append(f"  difference: {dmhp:+.2f} pp")
        lines += [
            "  comparability:",
            f"    dataset   : {cmp.get('dataset','UNKNOWN')}",
            f"    config    : {cmp.get('config','UNKNOWN')}",
            f"    code      : {cmp.get('code','UNKNOWN')}",
            f"    architecture: {cmp.get('architecture','UNKNOWN')}",
            f"    seed      : {cmp.get('seed','UNKNOWN')}",
            f"    metric    : {cmp.get('metric','UNKNOWN')}",
            f"    checkpoint lineage: {cmp.get('checkpoint_lineage','UNKNOWN')}",
            f"  overall   : {cmp.get('overall','UNKNOWN')}",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Richer JSONL record shape (D4)
# ---------------------------------------------------------------------------

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
    *,
    config_sha256: str = "local",
    git_commit: str = "local",
    session_id: str = "local",
    timestamp_utc: Optional[str] = None,
    gradient_share: Optional[Dict[str, float]] = None,
    update_stats: Optional[Dict[str, Any]] = None,
    representation: Optional[Dict[str, Any]] = None,
    val_robust_seed: int = 17,
    val_robust_subset_hash: Optional[str] = None,
    best_provenance: Optional[str] = None,
    checkpoint_epoch: Optional[int] = None,
    checkpoint_sha256: Optional[str] = None,
    dataset_fingerprint: Optional[Dict[str, Any]] = None,
    historical_reference: Optional[Dict[str, Any]] = None,
    comparability: Optional[Dict[str, str]] = None,
    current_minus_historical_pp: Optional[float] = None,
    chance_level: Optional[float] = None,
    ratio_to_chance: Optional[float] = None,
    accuracy_gap_pp: Optional[float] = None,
    timestamp_utc_override: Optional[str] = None,
) -> Dict[str, Any]:
    ts = timestamp_utc_override or timestamp_utc or _utc_now_iso()
    return {
        "timestamp_utc": ts,
        "session_id": str(session_id),
        "environment": "KAGGLE/Colab (undifferentiated log path)",
        "phase": phase,
        "seed": 0,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "epsilon": float(getattr(point, "eps", 0.0)),
        "beta": float(getattr(point, "beta", 0.0)),
        "pgd_steps": int(getattr(point, "pgd_steps", 0)),
        "learning_rate": float(lr),
        "train_loss_total": float(tr_loss),
        "train_ce_clean": float(ce_clean),
        "train_kl_trades": float(kl_trades),
        "train_acc_clean": float(train_clean_acc),
        "train_acc_adv": float(train_adv_acc),
        "gradient_share": gradient_share,
        "val_acc_clean": float(val_clean),
        "val_acc_robust": float(val_robust),
        "val_clean_n": int(clean_n),
        "val_robust_n": int(robust_n),
        "val_robust_seed": int(val_robust_seed),
        "val_robust_subset_hash": str(val_robust_subset_hash or ""),
        "val_robust_secs": float(robust_secs),
        "best_clean": float(best_clean),
        "best_clean_epoch": int(best_clean_epoch),
        "best_robust": float(best_robust),
        "best_robust_epoch": int(best_robust_epoch),
        "best_provenance": str(best_provenance or ""),
        "checkpoint_criterion": str(criterion),
        "checkpoint_epoch": int(checkpoint_epoch) if checkpoint_epoch is not None else None,
        "checkpoint_sha256": str(checkpoint_sha256 or ""),
        "gradients": grad,
        "parameter_updates": update_stats,
        "representation": representation,
        "timing": {
            "epoch_seconds": float(epoch_seconds),
            "img_per_sec": float(img_per_sec),
            "peak_vram_gb": float(peak_vram_gb) if peak_vram_gb is not None else None,
        },
        "config_sha256": str(config_sha256),
        "git_commit": str(git_commit),
        "dataset_fingerprint": dataset_fingerprint,
        "historical_reference": historical_reference,
        "comparability": comparability,
        "current_minus_historical_pp": float(current_minus_historical_pp)
        if current_minus_historical_pp is not None else None,
        "chance_level": float(chance_level) if chance_level is not None else None,
        "ratio_to_chance": float(ratio_to_chance) if ratio_to_chance is not None else None,
        "accuracy_gap_pp": float(accuracy_gap_pp) if accuracy_gap_pp is not None else None,
    }


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_phase_jsonl_path(report_dir: str, phase: str) -> str:
    return os.path.join(report_dir, f"{phase}_epoch_log.jsonl")


def _val_subset_hash(loader: DataLoader) -> str:
    try:
        n = len(loader.dataset)
        idxs = sorted(loader.dataset.indices) if hasattr(loader.dataset, "indices") else None
        h = hashlib.sha256()
        h.update(str(n).encode())
        if idxs is not None:
            h.update(str(idxs).encode())
        return h.hexdigest()[:16]
    except Exception:
        return ""


def _feature_snapshot_for_block(
    model: nn.Module,
    loaders: Dict[str, Any],
    device: torch.device,
    label: str,
) -> Dict[str, Any]:
    """Thin wrapper that captures a feature snapshot for the requested loader
    label and returns the dict used by the richer epoch block."""
    loader = loaders.get(label)
    if loader is None:
        return {}
    return feature_snapshot(model, loader, device, n_samples=64, label=label)


def _gradient_share_from_trades(
    ce_clean: float,
    kl_trades: float,
    kl_weight_beta: float,
    w_trades: float,
) -> Optional[Dict[str, float]]:
    """Rough per-step gradient-share estimate for the TRAIN line.

    This is an OBSERVED decomposition of the scalar loss contributions, NOT a new
    gradient computation and not a pre-registered scientific threshold. If kl is
    effectively zero we report it as such.
    """
    if kl_trades < 1e-12 and ce_clean < 1e-12:
        return None
    total = ce_clean + kl_trades
    if total <= 0.0:
        return None
    return {
        "CE": float(ce_clean / total),
        "KL": float(kl_trades / total),
    }
