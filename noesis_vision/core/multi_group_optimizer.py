"""
Multi-group optimizer registry — PORTED VERBATIM from Gen-0.
================================================================================

Source: rhan_core/optim/multi_group_optimizer.py (the STL-10 project),
ported per MASTER_PLAN Part 5 (PORT VERBATIM — "Validated across 3+
Gen-0 components; zero reason to touch"). The class below is carried
unchanged; two Gen-0-only pieces were intentionally NOT ported (documented
here rather than silently dropped):

  * the HPC/SBR/AIS LR-multiplier constants — Gen-0 component names;
  * default_group_spec(model, ...) — derives groups from the GEN-0 model's
    named_parameters. Gen-1's group derivation belongs to the substrate
    (Agent C), using the group names in
    noesis_vision.core.schema.OPTIMIZER_GROUP_NAMES (one group per
    independently gradient-isolated component; UpdateNet and the predictor
    each get their own, per Part 1.B).

Why this exists (the Stage 2 fix, 2026-08-13, now a standing architectural
rule): backbone lr = phase_lr, auxiliary-head group at phase_lr * 6.67, with
PER-GROUP grad clipping — the global clip_grad_norm_ over all params let the
backbone's TRADES gradient own the shared budget and crushed the HPC head's
per-step |dW| to ~1e-5 (the ratio-1.00 freeze). Every new trainable
component registers its own named group here.

Contract:
  * register(name, params, lr_multiplier, clip_norm) — a parameter may only
    belong to ONE group (double-registration raises loudly).
  * build_optimizer(base_lr) -> torch.optim.SGD, one param group per
    registered name, group lr = base_lr * lr_multiplier, in REGISTRATION
    order (backbone first).
  * clip_grad_per_group() applies clip_grad_norm_ to EACH group's parameters
    SEPARATELY (never a global budget) — the exact mechanism that fixed HPC's
    starvation.
  * resume_guard(optimizer_state_dict, saved_scheduler=None) returns True
    only when the saved state's group count, group NAMES, and lr-ratio
    pattern match the currently-registered groups — a checkpoint written by
    a different group layout can never be silently restored onto this
    optimizer. False means the caller falls back to a fresh optimizer with a
    loud warning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

#: Default per-group grad clip budget (matches the Stage 2 fix's max_norm=1.0).
DEFAULT_CLIP_NORM = 1.0


class OptimizerGroupRegistry:
    """
    Named, ordered parameter groups -> one torch.optim.SGD with per-group lr
    and per-group grad clipping.

    "backbone" is always registered first, implicitly, at lr_multiplier=1.0 —
    every parameter not claimed by a named auxiliary group lands there, so the
    registry can never silently drop a parameter from the optimizer.
    """

    def __init__(self):
        #: Ordered list of {name, params, lr_multiplier, clip_norm}.
        self._groups: List[Dict] = []
        self._backbone: List[nn.Parameter] = []
        self._claimed: set = set()          # id(param) -> owned by a named group

    # ── registration ────────────────────────────────────────────────────────
    def register(self, name: str, params: Sequence[nn.Parameter],
                 lr_multiplier: float = 1.0, clip_norm: float = DEFAULT_CLIP_NORM):
        """Add a named parameter group.

        Args:
            name: stable group label (recorded in the param group dict so the
                resume guard can verify group IDENTITY, not just count).
            params: parameters of this group. A parameter may only belong to
                one group — registering it twice raises loudly (a param in two
                groups would get two momentum buffers and double updates).
            lr_multiplier: group lr = base_lr * lr_multiplier (backbone = 1.0).
            clip_norm: this group's OWN clip_grad_norm_ budget (default 1.0).
        """
        if not params:
            return
        owned = [p for p in params if p.requires_grad]
        for p in owned:
            if id(p) in self._claimed:
                raise ValueError(
                    f"parameter {_param_name(p)} registered in group '{name}' "
                    f"but already claimed by another group — a parameter may "
                    f"only belong to ONE optimizer group (double momentum "
                    f"buffers otherwise).")
        for p in owned:
            self._claimed.add(id(p))
        self._groups.append({
            "name": str(name),
            "params": owned,
            "lr_multiplier": float(lr_multiplier),
            "clip_norm": float(clip_norm),
        })

    def register_backbone(self, params: Sequence[nn.Parameter],
                          clip_norm: float = DEFAULT_CLIP_NORM):
        """Register the backbone group (lr_multiplier=1.0) FIRST, implicitly.

        Must be called before any named auxiliary group so the optimizer's
        group order is [backbone, aux1, aux2, ...] — the same order the
        pre-migration Stage 2 builder produced (backbone, hpc).
        """
        if self._groups:
            raise RuntimeError(
                "register_backbone() must be the FIRST registration — the "
                "backbone group is always group 0 (lr_multiplier=1.0).")
        self.register("backbone", params, lr_multiplier=1.0, clip_norm=clip_norm)

    # ── construction ────────────────────────────────────────────────────────
    def build_optimizer(self, base_lr: float, momentum: float = 0.9,
                        weight_decay: float = 1e-4) -> torch.optim.SGD:
        """torch.optim.SGD with one param group per registered group.

        Group lr = base_lr * lr_multiplier. Group dicts carry 'name' and
        'clip_norm' (arbitrary keys are preserved by torch) so per-group
        clipping and the resume guard work without external bookkeeping.
        """
        if not self._groups:
            raise RuntimeError(
                "no groups registered — call register_backbone() first")
        param_groups = []
        for g in self._groups:
            param_groups.append({
                "params": list(g["params"]),
                "lr": float(base_lr) * g["lr_multiplier"],
                "name": g["name"],
                "clip_norm": g["clip_norm"],
            })
        return torch.optim.SGD(param_groups, momentum=momentum,
                               weight_decay=weight_decay, foreach=True)

    # ── per-group clipping ──────────────────────────────────────────────────
    def clip_grad_per_group(self, max_norm: Optional[float] = None):
        """Clip EACH group's gradients to ITS OWN budget.

        The Stage 2 root cause: under the old global clip, the backbone's
        TRADES gradient owned the shared norm and the HPC head's tiny grad was
        crushed to ~1e-5 updates. Groups with no gradients are skipped.
        """
        for g in self._groups:
            params = [p for p in g["params"] if p.grad is not None]
            if not params:
                continue
            budget = max_norm if max_norm is not None else g["clip_norm"]
            nn.utils.clip_grad_norm_(params, budget)

    # ── resume guard ────────────────────────────────────────────────────────
    def resume_guard(self, optimizer_state_dict, saved_scheduler=None) -> bool:
        """Can a checkpoint's saved optimizer state be restored onto a
        registry-built optimizer for the CURRENT group layout?

        Checks, in order:
          1. saved state is a dict with param_groups;
          2. group COUNT matches the currently-registered groups;
          3. group NAMES (recorded in both dicts) match, when the saved state
             carries them — catches a parameter reordering that keeps the
             count identical but silently misassigns momentum by position;
          4. the lr-RATIO pattern matches: each saved group's lr must equal
             base_lr * that group's lr_multiplier for some common base_lr.
             When `saved_scheduler` is provided its base_lrs (the flag-derived
             phase-start lrs, invariant to cosine decay) are used instead —
             the saved param_groups carry the CURRENT (decayed) lrs at the
             checkpoint's epoch, which would falsely refuse every legitimate
             mid-phase session continuation (same semantics as the Stage 2
             optimizer_restore_compatible guard).

        Returns True only when every check passes; False means the caller must
        fall back to a fresh optimizer (the trainer warns loudly).
        """
        if not isinstance(optimizer_state_dict, dict):
            return False
        saved_groups = optimizer_state_dict.get("param_groups", [])
        if len(saved_groups) != len(self._groups):
            return False

        # Names (when recorded).
        saved_names = [g.get("name") for g in saved_groups]
        cur_names = [g["name"] for g in self._groups]
        if all(isinstance(n, str) for n in saved_names):
            if saved_names != cur_names:
                return False

        # lr-ratio pattern.
        expected = None
        if isinstance(saved_scheduler, dict):
            base = saved_scheduler.get("base_lrs")
            if (isinstance(base, (list, tuple))
                    and len(base) == len(self._groups)):
                expected = [float(x) for x in base]
        if expected is None:
            expected = [g.get("lr") for g in saved_groups]
        if not all(isinstance(x, (int, float)) for x in expected):
            return False

        # base_lr consistent across all groups: lr_i / mult_i must be equal.
        base_lr = None
        for g, lr in zip(self._groups, expected):
            if g["lr_multiplier"] <= 0:
                return False
            b = float(lr) / g["lr_multiplier"]
            if base_lr is None:
                base_lr = b
            elif abs(b - base_lr) > 1e-9 * max(1.0, abs(b)):
                return False
        return True

    # ── introspection ───────────────────────────────────────────────────────
    @property
    def group_names(self) -> List[str]:
        return [g["name"] for g in self._groups]

    @property
    def registered(self) -> List[Dict]:
        return [dict(g, params=list(g["params"])) for g in self._groups]

    def group(self, name: str) -> Dict:
        for g in self._groups:
            if g["name"] == name:
                return dict(g, params=list(g["params"]))
        raise KeyError(f"no group registered under '{name}' — "
                       f"registered: {self.group_names}")

    def __len__(self):
        return len(self._groups)

    def __repr__(self) -> str:
        groups = [(g["name"], len(g["params"]), g["lr_multiplier"])
                  for g in self._groups]
        return f"OptimizerGroupRegistry(groups={groups})"


def _param_name(p: nn.Parameter) -> str:
    """Best-effort human name for a parameter in an error message."""
    return getattr(p, "_name", str(id(p)))
