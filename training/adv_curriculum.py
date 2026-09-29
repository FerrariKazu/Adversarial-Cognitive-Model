"""
Gen-0 TRADES/PGD curriculum for the Gen-1 foundation trainer (2026-09-29).
================================================================================

The 2026-09-25->26 foundation run trained with PURE cross-entropy — the
frozen trainer had no adversarial term at all (confirmed 2026-09-29 by
reading train_one_epoch; the loss trajectory sat exactly where CE-on-100-
classes sits). Every Gen-0 robustness result (D, AIS-v1, HPC-only, SBR)
was produced by the TRADES/PGD curriculum below; it was never carried
into the Gen-1 training contract (a planning gap, owned in the plan).

This module is a PORT, not a redesign: the curriculum table, the PGD
attack used to build training-time adversarial examples, and the TRADES
loss are carried from Gen-0's canonical trainer
(phase1_training/train_rhan_next.py — the D / AIS-v1 / HPC-only / SBR
entrypoint, itself a strict superset of train_rhan_v12.py) with ONLY
these adaptations:

  * the 60-epoch table is sliced PER PHASE: each foundation phase runs
    its own 60-epoch window through the same eps/beta/steps ramp
    (Gen-0's phases were 20 epochs each; the foundation phases are all
    60, so the ramp is spread over each phase's window 1:1 by thirds —
    the same relative exposure, matched-compute across phases);
  * the attack/loss operate on the foundation model's forward(x) -> logits
    contract (Agent I's harness contract) — no trajectory API;
  * epsilon/beta are in NORMALIZED space per the Gen-0 protocol's hard
    rule (eps_space="norm" everywhere; there is no pixel-space path).

Defaults (w_trades=0.55, beta schedule, PGD steps, alpha=eps/steps,
random start 0.001*randn) are the validated Gen-0 values — NOT new
hyperparameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# ── The Gen-0 curriculum (train_rhan_next.py lines 686-688, verbatim) ───────
# (start_epoch, end_epoch, eps, beta_base, pgd_steps) over 60 epochs.
#: The canonical 60-epoch ramp: eps 0.031 -> 0.062 -> 0.094, beta 2.0 -> 2.5.
CURRICULUM_60 = (
    (1, 20, 0.031, 2.0, 4),
    (21, 40, 0.062, 2.0, 4),
    (41, 60, 0.094, 2.5, 4),
)

#: The validated Gen-0 TRADES weight (train_rhan_next.py --w-trades default).
W_TRADES_DEFAULT = 0.55

#: Gen-0's adversarial random-start magnitude (train_rhan_next.py training
#: loop: x_adv = clamp(imgs + 0.001 * randn_like)). Kept for parity even
#: though the eval harness's generic_pgd uses a uniform start.
RAND_START_MAG = 0.001


@dataclass(frozen=True)
class CurriculumPoint:
    """The adversarial recipe for one training epoch."""
    eps: float          # normalized-space (norm-space protocol)
    beta: float         # TRADES KL weight
    pgd_steps: int      # attack steps used to BUILD x_adv
    adversarial: bool   # False only for an explicit clean-only window


def curriculum_for_epoch(epoch: int, total_epochs: int) -> CurriculumPoint:
    """Slice the 60-epoch Gen-0 ramp onto this phase's window.

    The ramp is defined over relative thirds of the window (Gen-0: epochs
    1-20 / 21-40 / 41-60 of 60), so any total_epochs > 0 gets the same
    relative exposure. epoch is 1-based.
    """
    if total_epochs <= 0:
        raise ValueError(f"total_epochs must be > 0, got {total_epochs}")
    if not 1 <= epoch <= total_epochs:
        raise ValueError(f"epoch {epoch} out of 1..{total_epochs}")
    n = total_epochs
    # Window boundaries at ceil(n/3) and ceil(2n/3): the exact Gen-0 split
    # at n=60 (20/20/20), and a 1-epoch window (smoke) still lands in the
    # ramp's FIRST segment (never skips the gentle eps).
    if epoch <= math.ceil(n / 3.0):
        start, end, eps, beta, steps = CURRICULUM_60[0]
    elif epoch <= math.ceil(2 * n / 3.0):
        start, end, eps, beta, steps = CURRICULUM_60[1]
    else:
        start, end, eps, beta, steps = CURRICULUM_60[2]
    return CurriculumPoint(eps=eps, beta=beta, pgd_steps=steps,
                           adversarial=True)


def curriculum_table(total_epochs: int) -> Dict[str, CurriculumPoint]:
    """The full per-epoch table (introspection + provenance printing)."""
    return {str(e): curriculum_for_epoch(e, total_epochs)
            for e in range(1, total_epochs + 1)}


# ── The Gen-0 training-time attack (PGD-KL; train_rhan_next.py, verbatim
#    mechanics under the foundation model's logits-only contract) ────────────
def pgd_kl_attack(model, x: torch.Tensor, eps: float, steps: int,
                  alpha: float = 0.0) -> torch.Tensor:
    """Build x_adv by PGD on the KL(x_adv || clean) objective — Gen-0's
    training attack exactly (random start 0.001*randn, alpha = eps/steps,
    sign steps, Linf ball around x0, [-4, +4] normalized-validity box to
    match the eval harness's clip_range convention).

    Runs the model in eval() internally (BN-free ViT trunk; purely
    functional) and restores nothing — the caller re-enables train() as
    Gen-0's loop does (raw_model.eval() ... model.train()).
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            probs_c = F.softmax(model(x).float(), dim=1)
        x_adv = torch.clamp(
            x.clone().detach() + RAND_START_MAG * torch.randn_like(x),
            -4.0, 4.0)
        if alpha <= 0.0:
            alpha = eps / max(steps, 1)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                logits_a = model(x_adv)
                loss = F.kl_div(F.log_softmax(logits_a.float(), dim=1),
                                probs_c, reduction="batchmean")
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv.detach() + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
                x_adv = x_adv.clamp(-4.0, 4.0)
        return x_adv.detach()
    finally:
        if was_training:
            model.train()


def trades_loss(model, x_natural: torch.Tensor, y: torch.Tensor,
                x_adv: torch.Tensor, beta: float
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The Gen-0 TRADES term: l_trades = CE(clean) + beta * KL(adv||clean).

    Carried from train_rhan_next.dynamic_trades_loss_next with beta_dynamic
    collapsed to beta_base: the foundation model has no trajectory API and
    no precision sequence OUTSIDE the belief path yet — the Gen-1
    uncertainty machinery is diagnostic-only by frozen protocol (L_stab is
    NOT a training objective), so there is nothing to modulate beta with.
    Gen-0's per-image weighting collapses to the unweighted batch mean here
    (weights == 1): per-image KL sums averaged over the batch ARE KL
    batchmean, so the math is identical for the plain ImageNet-100 loaders.
    Returns (loss, beta_used).
    """
    logits_c = model(x_natural)
    logits_a = model(x_adv)
    ce = F.cross_entropy(logits_c.float(), y)
    kl = F.kl_div(
        F.log_softmax(logits_a.float(), dim=1),
        F.softmax(logits_c.float().detach(), dim=1),
        reduction="batchmean")
    loss = ce + beta * kl
    return loss, float(beta)


# ── Per-phase windows: each foundation phase runs the FULL ramp ─────────────
#: Foundation phases are matched-compute (each trains its own 60-epoch
#: window from scratch — training/stage_state_machine.py DEPENDENCIES doc),
#: so every phase traverses the same eps/beta ramp. gen1_core's differentiation
#: is enforced structurally by the trainer (rebuild-from-parent gate), not by
#: a different loss recipe.
PHASE_WINDOW: Dict[str, Tuple[int, int]] = {
    "backbone_only": (1, 60),
    "recurrence_only": (1, 60),
    "belief_no_f": (1, 60),
    "belief_with_f": (1, 60),
    "ais_v2_swap": (1, 60),
    "gen1_core": (1, 60),
}


def phase_curriculum(phase: str, epoch: int, total_epochs: int
                     ) -> CurriculumPoint:
    """The curriculum point for a foundation phase's 1-based epoch."""
    if phase not in PHASE_WINDOW:
        raise ValueError(
            f"unknown foundation phase {phase!r} — expected one of "
            f"{sorted(PHASE_WINDOW)}")
    return curriculum_for_epoch(epoch, total_epochs)
