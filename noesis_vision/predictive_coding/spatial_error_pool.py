"""Spatial error pool — Agent E / Adjustment 1, Part 4.

The belief-update error E_t is a (B, N, D_feat) token-feature surface:
local, per-patch-token, relative to the shared predictor's prediction.
The update net, however, consumes (z_t, E_t) in the GLOBAL content space
— z_t is (B, D_z). A flat mean of E_t is dimensionality reduction, not a
learned mapping, so it silently assumes the two spaces are already
compatible. The plan's standing rule is that the spaces are NOT assumed
compatible without a *learned* mapping.

This module replaces that flat mean with a small learned attention pool:

    E_map  (B, G, G)                      # spatial error surface
    err    = tanh( E_map * pool_logits )  # (B, G, G), learned weights
    pooled = MNLP(err)                    # spatial mean-pooling
    z_hat  = Linear(pooled -> D_z)        # local -> global projection

Each of the G x G spatial positions gets a learned scalar weight from a
single-pooling-logit head over the pooled error features (not over the
raw per-token features, so the head is cheap and the head's gradients
stay confined to `error_pool`), and the pooled error is then projected to
z_t's dimension with a Linear. The whole thing is a tiny MLP + one Linear
— well under the "small, new, independently gradient-isolated" contract,
plus its own optimizer group `error_pool` and a pre-flight |dW| check
before any smoke test.

The observed-token path is detached (target convention, Part 1.A); only
the predicted path and the pool's own parameters carry gradient.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class SpatialErrorPool(nn.Module):
    """Learned spatial pooling of the per-token error surface.

    Args:
        d_z: dimension of z_t (the global content space).
        n_tokens: token count per glimpse (16 for the 4x4 grid).
        pool_hidden: hidden width of the single-pooling-logit head.
        projection_hidden: hidden width of the local->global projection.
    """

    def __init__(
        self,
        d_z: int = 384,
        n_tokens: int = 16,
        pool_hidden: int = 64,
        projection_hidden: int = 128,
    ) -> None:
        super().__init__()
        if n_tokens <= 0 or n_tokens * n_tokens <= 0:
            raise ValueError(f"n_tokens must be positive; got {n_tokens}")
        self.d_z = int(d_z)
        self.n_tokens = int(n_tokens)
        self.grid_size = int(n_tokens ** 0.5)
        if self.grid_size * self.grid_size != self.n_tokens:
            raise ValueError(
                f"n_tokens ({n_tokens}) must be a perfect square "
                f"(4x4 = 16 tokens for the substrate)"
            )

        # Learned weights per spatial position of the error map, squeezed
        # through a tiny MLP so the head's parameters live ONLY in this
        # module (its own optimizer group).
        self.pool_logits = nn.Sequential(
            nn.Linear(1, pool_hidden),
            nn.GELU(),
            nn.Linear(pool_hidden, 1),
            nn.Tanh(),
        )

        # Local->global projection: pooled error -> z_t's space.
        # input = pooled error (B, G, G) after the learned spatial
        # attention, flattened to (B, G*G) ; output (B, d_z).
        self.projection = nn.Sequential(
            nn.Linear(self.n_tokens, projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, d_z),
        )

    # ── forward ────────────────────────────────────────────────────────────────
    def forward(
        self, err_map: torch.Tensor, predicted_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Pool the spatial error surface -> (B, D_z) in z_t's space.

        Args:
            err_map: (B, G, G) spatial error surface, e.g. the
                token_error_map output (predictions squared vs observed).
                ``predicted_tokens`` is the (B, N, D_feat) prediction that
                the surface describes — it only exists to keep the function
                signature discoverable and is NOT part of the forward graph.
                The observed path is detached (target convention).
            predicted_tokens: (B, N, D_feat) prediction feeding the map.
                Detached here on purpose: only the predicted path carries
                gradient, and the pool's own parameters do.

        Returns:
            (B, d_z) — a learned mapping of the local token-feature error
            into the global content space z_t, ready for
            ``UpdateNet(z_t, err)``.
        """
        if err_map.dim() != 3:
            raise ValueError(
                f"err_map must be (B, G, G); got {tuple(err_map.shape)}"
            )
        if err_map.shape[1] != self.grid_size or err_map.shape[2] != self.grid_size:
            raise ValueError(
                f"err_map grid {tuple(err_map.shape[1:])} != "
                f"(G, G) = ({self.grid_size}, {self.grid_size})"
            )

        # Observed path is detached (target convention): the error map is a
        # target surface, the pool is a learned mapping. Only the predicted
        # path and the pool's own parameters carry gradient, so the pool
        # learns to map the *prediction error* without training on the
        # observed tokens.
        observed = err_map.detach()
        # Pool through the learned spatial attention (not a flat mean): each
        # of the G*G positions gets a learned weight, then a spatial mean,
        # so the positions that matter most for the update dominate.
        w = self.pool_logits(observed.unsqueeze(-1))               # (B, G, G, 1)
        w = w.squeeze(-1)                                          # (B, G, G)
        attn = w / (w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6))
        pooled = (attn * observed).flatten(1)                      # (B, G*G)
        z_hat = self.projection(pooled)                            # (B, d_z)
        return torch.tanh(z_hat)                                   # bounded update

    # ── contract helpers ───────────────────────────────────────────────────────
    def expected_input(self) -> str:
        return f"(B, {self.grid_size}, {self.grid_size})"

    def expected_output(self) -> str:
        return f"(B, {self.d_z})"
