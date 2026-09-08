"""
Structured belief representation — PILLAR 3.
================================================================================

Object-slot belief representation using Slot Attention (Locatello et al. 2020).
Replaces VectorBeliefState's flat 512-dim vector with structured object slots
that compete for input features via iterative attention refinement.

When enable_sbr=True in RHANNextConfig:
  - BeliefState is StructuredBeliefState (num_slots object slots)
  - Classifier and gaze policy receive pooled_slots (B, D) — drop-in compatible
  - Generative prior receives raw slots (B, K, slot_dim)
  - Uncertainty is derived from slot attention entropy

SBR wiring mode (`sbr_stage`, RHAN-NX):
  * "legacy" (default, E2-equivalent): slots bind over a SINGLE pooled vector
    (N=1 position) — the wiring every pre-RHAN-NX SBR checkpoint was trained
    with. The model must not reinterpret old checkpoints under new wiring.
  * "gate_only" .. "uncertainty" (SBR-0..4): slots bind over the SPATIAL stem
    feature map (B, H*W, D) + positional encoding. The SBR-0 gate metrics
    (occupancy entropy, pairwise attention-map cosine, per-slot linear
    probes) REQUIRE multiple spatial positions and are meaningless under the
    legacy N=1 wiring — this is the binding mode the gate is designed for.
  * SBR-3/4 add the relational layer (attention-based inter-slot message
    passing) and the per-slot evidence decomposition, built here as
    submodules (structured_belief.relational.* / structured_belief.evidence.*)
    so the belief stays self-contained.

When enable_sbr=False (default):
  - VectorBeliefState is used (identical to current behavior)
  - No code path touches StructuredBeliefState
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from rhan_core.beliefs.base import BeliefState


class StructuredBeliefState(BeliefState, nn.Module):
    """
    Structured belief: object-slot representation via Slot Attention.

    Each slot is a learned embedding that competes for input features.
    The resulting slots provide structured object-level representations
    that are more robust to adversarial perturbation than flat vectors,
    because perturbations affect specific slots rather than the entire
    belief state uniformly.

    Args:
        num_slots: Number of object slots (K). More slots = finer decomposition.
        slot_dim: Dimension of each slot embedding (D). Must match model proj_dim.
        iters: Number of slot attention refinement iterations.
        max_steps: Foraging steps (used for slot temporal aggregation).
        num_heads: Number of attention heads in slot attention.
        input_dim: Optional input feature dimension. When given and !=
            slot_dim, an input projection (input_proj) maps the bound features
            into slot space (the spatial stem tap is 768-dim vs slot_dim 512).
        use_relational: build the SBR-3 inter-slot message-passing layer.
        use_evidence: build the SBR-3 per-slot evidence decomposition heads.
        uncertainty_mode: SBR-4 — evidence decomposition adds the
            hypothesis/supporting/contradictory/uncertainty output, and the
            belief's uncertainty() becomes the decomposition's uncertainty.
    """

    def __init__(self, num_slots: int = 16, slot_dim: int = 512,
                 iters: int = 3, max_steps: int = 4, num_heads: int = 4,
                 input_dim: Optional[int] = None,
                 use_relational: bool = False,
                 use_evidence: bool = False,
                 uncertainty_mode: bool = False):
        BeliefState.__init__(self)
        nn.Module.__init__(self)
        if num_slots < 1 or slot_dim < 1:
            raise ValueError("num_slots and slot_dim must be >= 1")
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.max_steps = max_steps
        self.num_heads = num_heads

        self.scale = slot_dim ** -0.5

        # Learnable slot initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.slots_sigma = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)

        # Slot attention components
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(slot_dim, slot_dim)
        self.to_v = nn.Linear(slot_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, 2 * slot_dim),
            nn.ReLU(),
            nn.Linear(2 * slot_dim, slot_dim),
        )
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_input = nn.LayerNorm(slot_dim)

        # Spatial input projection (SBR-0..4): maps the 768-dim stem tap into
        # slot space when the caller binds over spatial features. Absent in
        # the legacy N=1 wiring so E2's state dict loads unchanged.
        self.input_proj: Optional[nn.Module] = None
        if input_dim is not None and input_dim != slot_dim:
            self.input_proj = nn.Linear(input_dim, slot_dim)

        # SBR-3: relational evidence (inter-slot message passing).
        from rhan_core.beliefs.relational import SlotRelationalLayer
        self.relational: Optional[nn.Module] = None
        if use_relational:
            self.relational = SlotRelationalLayer(slot_dim=slot_dim,
                                                  num_heads=num_heads)

        # SBR-3/4: per-slot evidence decomposition.
        from rhan_core.beliefs.evidence_decomposition import (
            EvidenceDecomposition)
        self.evidence: Optional[nn.Module] = None
        if use_evidence:
            self.evidence = EvidenceDecomposition(
                slot_dim=slot_dim, uncertainty_mode=uncertainty_mode)

        # Temporal aggregation: weighted combination across foraging steps
        if max_steps > 1:
            self.step_gate = nn.Sequential(
                nn.Linear(slot_dim * 2, slot_dim),
                nn.Sigmoid(),
            )

    def init_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize slots for a new forward pass: (B, K, D)."""
        mu = self.slots_mu.expand(batch_size, -1, -1)
        sigma = self.slots_sigma.expand(batch_size, -1, -1)
        return mu + sigma * torch.randn_like(mu)

    def attend(self, slots: torch.Tensor, features: torch.Tensor,
               prev_slots: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """One step of slot attention.

        Args:
            slots: (B, K, D) current slot states
            features: (B, N, D) input features
            prev_slots: (B, K, D) previous slot states for GRU

        Returns:
            updated_slots: (B, K, D)
            attn: (B, K, N) attention masks
        """
        B, K, D = slots.shape
        if prev_slots is None:
            prev_slots = slots

        slots_normed = self.norm_slots(slots)
        k = self.to_k(self.norm_input(features))
        v = self.to_v(self.norm_input(features))

        attn = torch.einsum("bkd,bnd->bkn", self.to_q(slots_normed), k) * self.scale
        attn = attn - attn.max(dim=1, keepdim=True).values
        attn = attn.softmax(dim=1)

        updates = torch.einsum("bkn,bnd->bkd", attn, v)
        slots = self.gru(updates.reshape(-1, D), prev_slots.reshape(-1, D)).reshape(B, K, D)
        slots = slots + self.mlp(self.norm_slots(slots))

        return slots, attn

    def forward(self, features: torch.Tensor,
                prev_state: Optional[Dict[str, torch.Tensor]] = None,
                run_relational: bool = True) -> Dict[str, torch.Tensor]:
        """Run slot attention and return structured belief.

        Args:
            features: (B, N, D) input features (e.g. the spatial stem tap,
                or a single pooled vector under the legacy E2 wiring).
            prev_state: optional dict with 'slots' key for temporal continuation
            run_relational: when a relational layer exists (SBR-3/4), run one
                round of inter-slot message passing after slot refinement.

        Returns:
            dict with:
                'slots': (B, K, D) updated slot states
                'attn': (B, K, N) attention masks from last iteration
                'entropy': (B,) slot attention entropy
                'pooled': (B, D) pooled representation (mean of slots)
                'prev_slots': (B, K, D) slot states from previous step (for temporal gate)
                'relation_attn': (B, K, K) inter-slot relation matrix (SBR-3+)
                'evidence': evidence-decomposition dict (SBR-3+)
        """
        B, N, D = features.shape

        if self.input_proj is not None:
            features = self.input_proj(features)          # (B, N, slot_dim)

        if prev_state is not None and 'slots' in prev_state:
            slots = prev_state['slots']
        else:
            slots = self.init_slots(B, features.device)

        prev_slots = slots.detach()

        for _ in range(self.iters):
            slots, attn = self.attend(slots, features, prev_slots if _ == 0 else slots)

        # SBR-3: inter-slot message passing (relational evidence).
        relation_attn = None
        if self.relational is not None and run_relational:
            slots, relation_attn = self.relational(slots)

        # SBR-3/4: per-slot evidence decomposition.
        evidence = None
        if self.evidence is not None:
            evidence = self.evidence(slots)

        # Attention entropy: high = slots are diffuse (undecided), low = focused
        attn_clamp = attn.clamp(min=1e-8)
        entropy = -(attn_clamp * attn_clamp.log()).sum(dim=1).mean(dim=1)  # (B,)

        # Pooled representation for backward-compatible consumers
        pooled = slots.mean(dim=1)  # (B, D)

        # Temporal aggregation: gate current pooled with previous step
        if prev_state is not None and 'pooled' in prev_state and hasattr(self, 'step_gate'):
            prev_pooled = prev_state['pooled']
            gate = self.step_gate(torch.cat([pooled, prev_pooled], dim=-1))
            pooled = gate * pooled + (1 - gate) * prev_pooled

        # Store for legacy interface (as_tensor / uncertainty)
        self._last_pooled = pooled.detach()
        self._last_slots = slots.detach()
        if evidence is not None and 'uncertainty' in evidence:
            # SBR-4: uncertainty is the decomposition's first-class output.
            self._last_entropy = evidence['uncertainty'].detach()
        else:
            self._last_entropy = entropy.detach()

        out = {
            'slots': slots,
            'attn': attn,
            'entropy': entropy,
            'pooled': pooled,
            'prev_slots': slots,
            # The belief's operative uncertainty — evidence-decomposition
            # uncertainty under SBR-4 (first-class belief output), slot
            # attention entropy otherwise (SBR-0..3). The model's halting
            # gate reads this key.
            'uncertainty': self._last_entropy,
        }
        if relation_attn is not None:
            out['relation_attn'] = relation_attn
        if evidence is not None:
            out['evidence'] = evidence
        return out

    # ── BeliefState interface ────────────────────────────────────────────────
    def as_tensor(self) -> torch.Tensor:
        """Flattened (B, K*D) view for legacy classifier compat."""
        if not hasattr(self, '_last_pooled'):
            raise RuntimeError("StructuredBeliefState has not been called yet — "
                               "call forward() before as_tensor()")
        return self._last_pooled

    def uncertainty(self) -> torch.Tensor:
        """(B,) per-sample uncertainty.

        SBR-4 (uncertainty_mode): the evidence decomposition's pooled
        uncertainty — uncertainty as a first-class belief output. Otherwise:
        slot attention entropy.
        """
        if not hasattr(self, '_last_entropy'):
            raise RuntimeError("StructuredBeliefState has not been called yet — "
                               "call forward() before uncertainty()")
        return self._last_entropy

    def update_slots(self, evidence: torch.Tensor, edges=None):
        """Update slot representations from new evidence."""
        return self.forward(evidence)

    def message_passing(self, steps: int = 1):
        """Run inter-slot message passing (SBR-3 relational evidence).

        Promoted from the Stage-0 NotImplementedError scaffold: with the
        relational layer built (use_relational=True), runs `steps` rounds of
        attention-based message passing over the CURRENT slots and returns
        (updated_slots, last_relation_attn). Without the layer (legacy / E2
        wiring), raises a clear error — the layer is only built under
        sbr_stage in {'relational', 'uncertainty'}.
        """
        if self.relational is None:
            raise RuntimeError(
                "message_passing() called but no relational layer is built — "
                "the layer only exists under sbr_stage in {'relational', "
                "'uncertainty'} (SBR-3/4). Legacy/E2 wiring has no "
                "inter-slot relations.")
        slots = self._last_slots if hasattr(self, '_last_slots') else None
        if slots is None:
            raise RuntimeError(
                "message_passing() requires a forward pass first (the slots "
                "it should relate do not exist yet).")
        attn = None
        for _ in range(int(steps)):
            slots, attn = self.relational(slots)
        return slots, attn

    def __repr__(self) -> str:
        extra = []
        if self.relational is not None:
            extra.append("relational")
        if self.evidence is not None:
            extra.append(f"evidence(uncertainty={self.evidence.uncertainty_mode})")
        suffix = f" [{','.join(extra)}]" if extra else ""
        return (f"StructuredBeliefState(num_slots={self.num_slots}, "
                f"slot_dim={self.slot_dim}, iters={self.iters}){suffix}")