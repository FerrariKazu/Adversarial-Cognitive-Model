"""
Structured belief representation — PILLAR 3, SCAFFOLD ONLY.

This module contains a REAL, importable, instantiable interface for the
future object-slot + relational-graph belief (Pillar 3). It never errors on
import or instantiation; only *calling* the unimplemented core operations
raises a clear, documented NotImplementedError. `enable_sbr` MUST remain
False in RHANNextConfig until Pillar 3 is implemented.
"""
from __future__ import annotations

from typing import Optional

import torch

from rhan_core.beliefs.base import BeliefState

_PILLAR3_MSG = (
    "StructuredBeliefState is PILLAR 3 (Structured Belief Representation) "
    "scaffold-only in RHAN-Next. The slot/relation machinery is NOT "
    "implemented; only the interface exists so callers can code against it. "
    "Pillar 3 is gated behind RHANNextConfig.enable_sbr which MUST remain "
    "False. No gradient, loss, or forward path may call this method until "
    "Stage-3 of the SBR roadmap lands."
)


class StructuredBeliefState(BeliefState):
    """
    PILLAR 3 SCAFFOLD — object slots + relational graph belief.

    When implemented this will represent the scene as `num_slots` object
    slots (each a dense embedding) plus an explicit relational graph, and
    will expose the same two BeliefState methods (as_tensor / uncertainty)
    so every existing caller keeps working.

    Representation state (placeholder only):
        slots: (num_slots, slot_dim) — object-slot embeddings (NOT wired to
               the autograd graph; Pillar 3 does not exist yet).
    """

    def __init__(self, num_slots: int = 16, slot_dim: int = 512):
        super().__init__()
        if num_slots < 1 or slot_dim < 1:
            raise ValueError("num_slots and slot_dim must be >= 1")
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        # Placeholder representation — never fed into the network.
        self.slots = torch.zeros(num_slots, slot_dim)

    # ── BeliefState interface (documented NotImplementedError for now) ───────
    def as_tensor(self) -> torch.Tensor:
        """Raises NotImplementedError — Pillar 3 is scaffold-only."""
        raise NotImplementedError(_PILLAR3_MSG)

    def uncertainty(self) -> torch.Tensor:
        """Raises NotImplementedError — Pillar 3 is scaffold-only."""
        raise NotImplementedError(_PILLAR3_MSG)

    # ── Future Pillar 3 operations (interface only) ──────────────────────────
    def update_slots(self, evidence: torch.Tensor, edges: Optional[torch.Tensor] = None):
        """Raises NotImplementedError — declared for interface stability only."""
        raise NotImplementedError(_PILLAR3_MSG)

    def message_passing(self, steps: int = 1):
        """Raises NotImplementedError — declared for interface stability only."""
        raise NotImplementedError(_PILLAR3_MSG)

    def __repr__(self) -> str:
        return (f"StructuredBeliefState(num_slots={self.num_slots}, "
                f"slot_dim={self.slot_dim}) [PILLAR 3 SCAFFOLD — not implemented]")
