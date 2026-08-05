"""Uncertainty-gated halting policy (Pillar 2)."""
from __future__ import annotations

import torch
import torch.nn as nn

from rhan_core.beliefs.base import BeliefState


class EntropyGatedHalting(nn.Module):
    """
    Halts evidence gathering once belief uncertainty drops below a threshold.

    Uncertainty is the belief's own signal (the model injects u = 1 - Pi_D).
    Halting is NOT a step-count penalty: there is no `steps_used / max_steps`
    term anywhere in the loss (asserted by a static scan in
    tests/test_gradient_flow.py). This directly resolves the historical
    contradiction in which a deleted `halt_efficiency` loss penalized step
    count, opposing the project's Banach-contraction argument that more
    steps should improve robustness.

    Two views of the same gate:
      * should_halt()  — hard (B,) bool, for diagnostics and eval;
      * continuation() — soft differentiable sigmoid gate used to weight
        belief accumulation and freeze gaze for halted samples (keeps the
        batch graph stable; hard per-sample early exit is deferred, see
        docs/rhan_next_roadmap.json).
    """

    def __init__(self, threshold: float = 0.35, softness: float = 8.0):
        super().__init__()
        self.threshold = float(threshold)
        self.softness = float(softness)

    def _effective_threshold(self, history) -> torch.Tensor:
        """Per-sample threshold, optionally modulated by the precision
        modulator (recurrence-depth wiring): history[-1]['halt_threshold'].

        Fallback: a (1,) tensor of the configured threshold.
        """
        if history and isinstance(history[-1], dict):
            ht = history[-1].get('halt_threshold', None)
            if ht is not None:
                return ht
        return torch.tensor(self.threshold)

    def should_halt(self, belief: BeliefState,
                    history: list) -> torch.Tensor:
        """(B,) bool — True when uncertainty < effective threshold."""
        u = belief.uncertainty()                          # (B,)
        return u < self._effective_threshold(history).to(u.device)

    def continuation(self, belief: BeliefState,
                     history: list, softness: float = None) -> torch.Tensor:
        """(B,) differentiable continuation weight in [0, 1].

        sigma(softness * (u - threshold)): uncertain samples keep
        gathering evidence (~1), confident samples stop (~0).
        """
        u = belief.uncertainty()                          # (B,)
        soft = float(softness) if softness is not None else self.softness
        thr = self._effective_threshold(history).to(u.device)
        return torch.sigmoid(soft * (u - thr))

    def __repr__(self) -> str:
        return (f"EntropyGatedHalting(threshold={self.threshold}, "
                f"softness={self.softness})")
