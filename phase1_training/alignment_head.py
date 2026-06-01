"""
Alignment Head for Neural Representation Alignment (Improvement #3)
=====================================================================
Projects RHAN's CLS token features into a shared space for alignment with
biological (CORnet-S IT) features.

The alignment loss encourages RHAN's representations to organize the way
biological vision does, using CORnet-S IT cortex features as a surrogate
for fMRI responses.

Loss: L_align = 1 - cosine_similarity(rhan_proj, bio_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentHead(nn.Module):
    """Projects RHAN features to a shared space for alignment loss."""

    def __init__(self, rhan_dim=512, bio_dim=512, hidden_dim=256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(rhan_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bio_dim),
        )
        self.norm = nn.LayerNorm(bio_dim)

    def forward(self, rhan_features):
        """
        Args:
            rhan_features: (B, 512) — RHAN CLS token
        Returns:
            (B, 512) — L2-normalized projected features
        """
        projected = self.projector(rhan_features)
        return F.normalize(projected, dim=-1)


def alignment_loss(rhan_cls, bio_features, alignment_head):
    """Cosine alignment loss between RHAN and biological features.

    Args:
        rhan_cls: (B, 512) — RHAN CLS token
        bio_features: (B, 512) — frozen biological reference features
        alignment_head: AlignmentHead module
    Returns:
        loss: scalar — 1 - mean cosine_similarity
    """
    rhan_proj = alignment_head(rhan_cls)
    bio_norm = F.normalize(bio_features, dim=-1)
    cos_sim = (rhan_proj * bio_norm).sum(dim=-1)
    return 1.0 - cos_sim.mean()
