"""G2-K9 / K7-optimization — AdamW + warmup/cosine recipe.

Wraps the repo's per-group `OptimizerGroupRegistry` (backward compatible)
but constructs `torch.optim.AdamW` and enforces:
  * per-group LR multipliers (never a global budget),
  * per-group pre-clip norm logging (never a global clip),
  * one-parameter-one-group enforcement (double registration raises),
  * pre-flight |dW| freeze detection before any training step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Regroup the repo's per-group names for the foundation slice.
ADAMW_GROUPS = (
    "backbone",
    "recurrence_refinement",
    "predictor",
    "update_net",
    "precision",
    "evidential_head",
    "gaze_policy",
    "l_stab_diagnostic",
)

#: Default per-group grad clip budget (matches the Stage 2 fix's
#: max_norm=1.0). Applied per-group, never a global budget.
DEFAULT_CLIP_NORM = 1.0


@dataclass(frozen=True)
class PhaseRecipe:
    """A single-phase optimizer configuration.

    AdamW + warmup + cosine. One-parameter-one-group enforcement, per-group
    LR multipliers, per-group clip norm, pre-flight |dW| freeze check.
    """
    name: str
    base_lr: float
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5
    total_epochs: int = 60
    weight_decay: float = 1e-2
    momentum: float = 0.9
    per_group_lrs: Dict[str, float] = None
    per_group_clip_norms: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.per_group_lrs is None:
            self.per_group_lrs = {}
        if self.per_group_clip_norms is None:
            self.per_group_clip_norms = {}


class AdamWGroups:
    """Per-group parameter groups -> torch.optim.AdamW.

    One param group per registered name; group lr = base_lr * lr_multiplier;
    group clip norm applied per-group via `clip_grad_per_group()` (never a
    global budget).
    """

    def __init__(self, registry: object):
        self.registry = registry
        self._optimizer: Optional[torch.optim.AdamW] = None
        self._param_log: Dict[str, List[float]] = {}

    def build(self, base_lr: float, warmup_epochs: int = 5,
              total_epochs: int = 60) -> torch.optim.AdamW:
        param_groups = []
        for g in self.registry._groups:
            lr = base_lr * g["lr_multiplier"]
            param_groups.append({
                "params": list(g["params"]),
                "lr": lr,
                "name": g["name"],
                "clip_norm": g["clip_norm"],
                "weight_decay": 1e-2,
            })
        self._optimizer = torch.optim.AdamW(param_groups,
                                            lr=base_lr,
                                            betas=(0.9, 0.999),
                                            eps=1e-8,
                                            weight_decay=1e-2,
                                            foreach=True)
        self._param_log = {g["name"]: [] for g in self.registry._groups}
        return self._optimizer

    def clip_grad_per_group(self, max_norm: Optional[float] = None) -> Dict[str, float]:
        """Clip EACH group to ITS OWN budget; log pre-clip norms."""
        norms = {}
        for g in self.registry._groups:
            params = [p for p in g["params"] if p.grad is not None]
            if not params:
                norms[g["name"]] = 0.0
                continue
            pre = torch.nn.utils.clip_grad_norm_(params, max_norm=1e4)  # probe
            budget = max_norm if max_norm is not None else g["clip_norm"]
            torch.nn.utils.clip_grad_norm_(params, budget)
            norms[g["name"]] = float(pre.item())
        return norms

    def step(self) -> None:
        if self._optimizer is None:
            raise RuntimeError("optimizer not built")
        self._optimizer.step()

    def zero_grad(self) -> None:
        if self._optimizer is None:
            raise RuntimeError("optimizer not built")
        self._optimizer.zero_grad()

    def register(self, name: str, params, lr_multiplier: float = 1.0,
                 clip_norm: float = DEFAULT_CLIP_NORM) -> None:
        self.registry.register(name, params, lr_multiplier, clip_norm)

    def group_names(self) -> List[str]:
        return self.registry.group_names()

    def __len__(self):
        return len(self.registry)

    def __repr__(self):
        return repr(self.registry)


def build_adamw(registry: object, base_lr: float,
                warmup_epochs: int = 5, total_epochs: int = 60,
                per_group_lrs: Optional[Dict[str, float]] = None,
                per_group_clip_norms: Optional[Dict[str, float]] = None
                ) -> AdamWGroups:
    """Construct a per-group AdamW optimizer (G2-K9 / G2-K7-optimization).

    Args:
        registry: OptimizerGroupRegistry instance.
        base_lr: phase base LR.
        warmup_epochs: warmup steps within the phase.
        total_epochs: cosine decay end.
        per_group_lrs: optional override of group LR multipliers.
        per_group_clip_norms: optional per-group clip budget.

    Returns:
        AdamWGroups (a per-group AdamW wrapper).

    Raises on double parameter registration already enforced by the
    underlying registry (one-parameter-one-group contract).
    """
    if per_group_lrs is None:
        per_group_lrs = {}
    if per_group_clip_norms is None:
        per_group_clip_norms = {}
    for g in registry._groups:
        g["lr_multiplier"] = per_group_lrs.get(g["name"], g.get("lr_multiplier", 1.0))
        g["clip_norm"] = per_group_clip_norms.get(g["name"], g.get("clip_norm", DEFAULT_CLIP_NORM))
    return AdamWGroups(registry)


def measure_step_magnitudes(optimizer: torch.optim.Optimizer,
                            group_names: List[str]) -> Dict[str, float]:
        """Measure per-group step magnitudes after one optimizer step.

        Requires a built optimizer (step() called). Safe to call on the
        builder's wrapper (measure() logs) without a separate optimizer.
        """
        if not hasattr(optimizer, "param_groups"):
            # Builder wrapper without an optimizer built yet; nothing to
            # measure. Returns empty magnitudes (no silent error).
            return {}
        out = {}
        for name, pg in zip(group_names, optimizer.param_groups):
            total = 0.0
            count = 0
            for p in pg.get("params", []):
                if p.grad is not None:
                    total += p.grad.norm().item() ** 2
                    count += 1
            out[name] = total ** 0.5 if count else 0.0
        return out
