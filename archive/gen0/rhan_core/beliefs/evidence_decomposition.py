"""
Per-slot evidence decomposition — SBR-3/4.
================================================================================

Two stages of the SBR ladder live here:

  * SBR-3 (`uncertainty_mode=False`): per-slot shape / texture / spatial
    evidence heads + a combination layer. Each slot's 512-dim vector is
    projected through three complementary evidence heads, and the combined
    evidence is what downstream consumers see. This gives the structured
    belief an explicit, attributable decomposition of WHAT each slot is
    evidence for.

  * SBR-4 (`uncertainty_mode=True`): adds the explicit
    hypothesis / supporting-evidence / contradictory-evidence / uncertainty
    output structure — uncertainty as a FIRST-CLASS belief output. Each slot
    produces a class hypothesis (logits over the 10 STL-10 classes) plus
    scalar support / contradiction / uncertainty scores; these are pooled
    across slots into the belief-level decomposition. This is the mechanism
    behind the final SBR checkpoint (SBR-4) and the Belief Stability under
    Perturbation (BSP) Lens metric.

Gradient contract: every head's output is attached to the graph — the
evidence combination feeds the belief `s` (so the classifier's gradient
reaches these params) and tests/test_sbr_gradient_flow.py asserts it.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class _EvidenceHead(nn.Module):
    """Small MLP head producing one evidence modality from a slot vector."""

    def __init__(self, slot_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, slot_dim),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return self.net(slots)


class EvidenceDecomposition(nn.Module):
    """
    Per-slot shape/texture/spatial evidence heads + combination (SBR-3),
    plus the SBR-4 hypothesis/supporting/contradictory/uncertainty output.

    Args:
        slot_dim: slot embedding dimension (512 in RHANNext).
        num_classes: STL-10 class count (10) for the hypothesis head.
        uncertainty_mode: when True, build the SBR-4 decomposition heads.
    """

    def __init__(self, slot_dim: int = 512, num_classes: int = 10,
                 uncertainty_mode: bool = False):
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.num_classes = int(num_classes)
        self.uncertainty_mode = bool(uncertainty_mode)

        # ── SBR-3: three complementary per-slot evidence heads ──────────────
        # shape: object-part geometry; texture: surface statistics; spatial:
        # position-aware binding evidence. All three are learned projections
        # of the slot vector; the combination layer merges them so the
        # structured belief carries an attributable decomposition.
        self.shape_head = _EvidenceHead(slot_dim)
        self.texture_head = _EvidenceHead(slot_dim)
        self.spatial_head = _EvidenceHead(slot_dim)
        self.combiner = nn.Linear(3 * slot_dim, slot_dim)

        # ── SBR-4: uncertainty as a first-class belief output ───────────────
        if self.uncertainty_mode:
            self.hypothesis_head = nn.Linear(slot_dim, num_classes)
            self.support_head = nn.Linear(slot_dim, 1)
            self.contradict_head = nn.Linear(slot_dim, 1)
            self.uncertainty_head = nn.Linear(slot_dim, 1)

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose a slot set into per-slot evidence + (SBR-4) belief output.

        Args:
            slots: (B, K, D) slot states (after message passing, if any).

        Returns:
            dict with:
                evidence:        (B, K, D) combined per-slot evidence vectors
                pooled_evidence: (B, D) mean evidence across slots
                shape_evidence / texture_evidence / spatial_evidence:
                                 (B, K, D) per-head evidence vectors
            and, when uncertainty_mode=True (SBR-4):
                hypothesis_logits:        (B, K, C) per-slot class hypotheses
                hypothesis:               (B, C) pooled hypothesis logits
                supporting_evidence:      (B,) pooled support scores
                contradictory_evidence:   (B,) pooled contradiction scores
                uncertainty:              (B,) pooled uncertainty in [0, 1]
        """
        e_shape = self.shape_head(slots)      # (B, K, D)
        e_tex = self.texture_head(slots)
        e_spat = self.spatial_head(slots)
        combined = self.combiner(
            torch.cat([e_shape, e_tex, e_spat], dim=-1))     # (B, K, D)

        out: Dict[str, torch.Tensor] = {
            "evidence": combined,
            "pooled_evidence": combined.mean(dim=1),
            "shape_evidence": e_shape,
            "texture_evidence": e_tex,
            "spatial_evidence": e_spat,
        }

        if self.uncertainty_mode:
            hyp = self.hypothesis_head(combined)             # (B, K, C)
            support = self.support_head(combined).squeeze(-1)      # (B, K)
            contra = self.contradict_head(combined).squeeze(-1)    # (B, K)
            unc = torch.sigmoid(self.uncertainty_head(combined)
                                .squeeze(-1))                      # (B, K)
            out.update({
                "hypothesis_logits": hyp,
                "hypothesis": hyp.mean(dim=1),               # (B, C)
                "supporting_evidence": support.mean(dim=1),
                "contradictory_evidence": contra.mean(dim=1),
                "uncertainty": unc.mean(dim=1),
            })
        return out

    def __repr__(self) -> str:
        return (f"EvidenceDecomposition(slot_dim={self.slot_dim}, "
                f"uncertainty_mode={self.uncertainty_mode})")