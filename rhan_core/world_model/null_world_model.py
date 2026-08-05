"""Safe no-op world model (Pillar 4 default)."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from rhan_core.beliefs.base import BeliefState
from rhan_core.world_model.base import WorldModel

_logger = logging.getLogger(__name__)


class NullWorldModel(WorldModel, nn.Module):
    """
    PILLAR 4 — SCAFFOLD. Safe passthrough.

    simulate() returns the input unchanged and logs a debug-level notice on
    first use. It has NO parameters and NO buffers, so wiring it into
    RHANNext adds nothing to the state dict (default-config state dicts stay
    byte-identical to RHANv12's).
    """

    def __init__(self):
        super().__init__()
        self._notified = False

    def simulate(self, belief: BeliefState, action: torch.Tensor) -> torch.Tensor:
        """Passthrough: returns the belief tensor unchanged.

        Args:
            belief: BeliefState — returned via as_tensor().
            action: (B, A) — ignored (no real world model).
        Returns:
            (B, D) the belief tensor itself, untouched.
        """
        if not self._notified:
            _logger.debug(
                "NullWorldModel: no real world model is wired in — simulate() "
                "returns its input unchanged. (Pillar 4 scaffold.)")
            self._notified = True
        return belief.as_tensor()
