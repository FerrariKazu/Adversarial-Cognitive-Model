"""Dense vector belief state used by Pillars 1 & 2 (and v12 compatibility)."""
from __future__ import annotations

import math
from typing import Optional

import torch

from rhan_core.beliefs.base import BeliefState


class VectorBeliefState(BeliefState):
    """
    Dense (B, D) belief state — the concrete belief used by Pillars 1 & 2 and
    identical in spirit to v12's belief vector `s`.

    The uncertainty is normally injected from upstream evidence (e.g. the
    ImageSpacePrecision signal: uncertainty = 1 - pi_D). When no uncertainty
    is supplied we fall back to a cheap magnitude proxy that is documented as
    such — it exists only so the class is usable standalone in unit tests;
    the model always passes a precision-derived uncertainty.

    Args:
        tensor:    (B, D) belief embedding.
        uncertainty: (B,) optional per-sample uncertainty in [0, 1].
                     Higher = less certain. If None, a magnitude proxy is used.
    """

    def __init__(self, tensor: torch.Tensor, uncertainty: Optional[torch.Tensor] = None,
                 uncertainty_source: str = "external"):
        super().__init__()
        if tensor.dim() != 2:
            raise ValueError(f"VectorBeliefState expects a (B, D) tensor, got {tuple(tensor.shape)}")
        self.tensor = tensor
        if uncertainty is not None:
            if uncertainty.shape[0] != tensor.shape[0]:
                raise ValueError(
                    f"uncertainty batch {tuple(uncertainty.shape)} != tensor batch {tuple(tensor.shape)}")
            uncertainty = uncertainty.to(tensor.device)
        self._uncertainty = uncertainty
        self.uncertainty_source = uncertainty_source

    # ── BeliefState interface ────────────────────────────────────────────────
    def as_tensor(self) -> torch.Tensor:
        """(B, D) — flattened belief for the classifier head / legacy callers."""
        return self.tensor

    def uncertainty(self) -> torch.Tensor:
        """(B,) per-sample uncertainty.

        Injected signal when available (the model passes 1 - pi_D); otherwise
        a documented magnitude proxy: 1 - ||s|| / sqrt(D), clipped to [0, 1].
        """
        if self._uncertainty is not None:
            return self._uncertainty
        norm = self.tensor.norm(dim=-1) / math.sqrt(self.tensor.shape[-1])
        return (1.0 - norm.clamp(0.0, 1.0))

    # ── Convenience ──────────────────────────────────────────────────────────
    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    def with_uncertainty(self, uncertainty: torch.Tensor) -> "VectorBeliefState":
        """Return a copy carrying an explicit uncertainty signal (B,)."""
        return VectorBeliefState(self.tensor, uncertainty=uncertainty)

    def detach(self) -> "VectorBeliefState":
        """Detached copy for storing in trajectory/history without graph leaks."""
        return VectorBeliefState(self.tensor.detach(),
                                 uncertainty=None if self._uncertainty is None
                                 else self._uncertainty.detach())

    def __repr__(self) -> str:
        return (f"VectorBeliefState(shape={tuple(self.tensor.shape)}, "
                f"uncertainty={'injected' if self._uncertainty is not None else 'magnitude-proxy'})")
