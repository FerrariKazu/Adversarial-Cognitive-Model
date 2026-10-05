"""
Inter-slot message passing — SBR-3 relational evidence.
================================================================================

Implements the relational component of the SBR ladder: attention-based message
passing between slot pairs. The source plan's Section 9 calls for a strictly
finer-grained robustness signal than whole-belief drift — the relational
layer exposes its per-pair attention so the Lens layer can measure
**relation drift** D(R_clean, R_adv): whether inter-slot relationships
specifically destabilize under attack.

Deliberately a SIMPLE attention-based relation (start with a tractable
mechanism, per the task: \"start with a simple attention-based relation, not a
full graph network\"). A single self-attention block over the slot set with a
residual + MLP — each slot collects messages from every other slot weighted
by their pairwise relevance.

Gradient contract (project lesson #1): `forward` returns the updated slots
fully attached to the computation graph — `message_passing` is a loss-bearing
path and tests/test_sbr_gradient_flow.py asserts gradients reach these
parameters (relational.* optimizer group, Generation 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SlotRelationalLayer(nn.Module):
    """
    Attention-based message passing between slot pairs.

    Args:
        slot_dim: slot embedding dimension (512 in RHANNext).
        num_heads: kept for interface parity with the slot-attention module;
            this first pass uses single-head dot-product attention over the
            full slot dimension (tractable; multi-head is a future refinement).
    """

    def __init__(self, slot_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.num_heads = int(num_heads)
        self.scale = slot_dim ** -0.5

        self.norm_in = nn.LayerNorm(slot_dim)
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(slot_dim, slot_dim)
        self.to_v = nn.Linear(slot_dim, slot_dim)
        # Residual MLP (same shape as the slot-attention MLP).
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, 2 * slot_dim),
            nn.GELU(),
            nn.Linear(2 * slot_dim, slot_dim),
        )
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, slots: torch.Tensor):
        """
        Run one round of inter-slot message passing.

        Args:
            slots: (B, K, D) current slot states.

        Returns:
            (updated_slots (B, K, D), attn (B, K, K)):
                updated_slots — residuals over attention-aggregated messages;
                attn — the per-pair relation matrix (softmax over source
                slots), consumed by the Lens relation-drift metric.
        """
        B, K, D = slots.shape
        h = self.norm_in(slots)
        q = self.to_q(h)                                  # (B, K, D)
        k = self.to_k(h)
        v = self.to_v(h)

        attn = torch.einsum("bkd,bld->bkl", q, k) * self.scale   # (B, K, K)
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = attn.softmax(dim=-1)                       # row-softmax over src

        msgs = torch.einsum("bkl,bld->bkd", attn, v)      # (B, K, D)
        out = slots + msgs                                # residual
        out = out + self.mlp(self.norm_mlp(out))
        return out, attn

    def __repr__(self) -> str:
        return (f"SlotRelationalLayer(slot_dim={self.slot_dim}, "
                f"num_heads={self.num_heads})")