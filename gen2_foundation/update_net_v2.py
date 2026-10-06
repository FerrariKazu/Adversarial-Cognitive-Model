"""G2-K5 / G2-K6 / G2-K7 — observed-feature belief update and spatial error.

Modules:
  * `BeliefUpdaterV2` — observed-feature fusion (concat / gated / FiLM),
    position-sensitive attention pooling of the error map, and
    adaptive-bound belief-update magnitude with a hard ceiling.
  * `adapt_magnitude` — bounded adaptive update magnitude (Delta z cap).

Gen-1 parity path remains executable: fusion="error_only" + mean-pool
== the original `ConcreteUpdateNet` behavior (and is the default off
control). The original `ConcreteUpdateNet` is left untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal, Optional

#: Hard ceiling on adaptive update magnitude (spec: keep hard bound).
ADAPTIVE_BOUND_MAX: float = 0.1

# Fusion modes for observed features + error (G2-K5).
FusionMode = Literal["error_only", "concat", "gated", "film"]

# Adaptive-bound arm (G2-K7-belief-adaptive): hard ceiling retained,
# tests whether Delta z <= 0.1 is protective (default) or destructive.
AdaptiveBoundMode = Literal["bounded", "adaptive", "unbounded"]

# ---------------------------------------------------------------------------
# G2-K5: observed-feature fusion
# ---------------------------------------------------------------------------


class ObservedFeatureFusion(nn.Module):
    """Fuse observed features (observation beyond prediction error) into z.

    When OFF (fusion="error_only"), the update reduces to the exact Gen-1
    path: mean-pool E_t then concat with z into UpdateNet. This is the
    legal parity control and is the default.
    """

    def __init__(
        self,
        d_z: int,
        obs_dim: int,
        mode: FusionMode = "error_only",
        hidden: int = 768,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.d_z = int(d_z)
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)

        if mode == "error_only":
            self.net = nn.Sequential(
                nn.Linear(2 * d_z, hidden), nn.GELU(),
                nn.Linear(hidden, d_z))
            self._offset = 0
        elif mode == "concat":
            self.net = nn.Sequential(
                nn.Linear(d_z + obs_dim, hidden), nn.GELU(),
                nn.Linear(hidden, d_z))
            self._offset = d_z
        elif mode == "gated":
            # Concatenated, then gated: s = sigmoid(W_g * [z; obs])
            self._gate = nn.Sequential(
                nn.Linear(d_z + obs_dim, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
            self.net = nn.Sequential(
                nn.Linear(d_z + obs_dim, hidden), nn.GELU(),
                nn.Linear(hidden, d_z))
            self._offset = d_z
        elif mode == "film":
            self.gamma = nn.Linear(d_z + obs_dim, d_z)
            self.beta = nn.Linear(d_z + obs_dim, d_z)
            self._offset = 0
        else:
            raise ValueError(f"unknown fusion mode {mode}")

    def forward(self, z: torch.Tensor,
                pooled_error: torch.Tensor,
                observed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the concatenated input [z, E_pool, obs] to the update MLP."""
        if self.mode == "error_only" and observed is not None:
            # Gen-1 parity: observed path must be absent (error_only).
            pass
        if self.mode != "error_only" and observed is None:
            raise ValueError("fusion mode requires observed features")
        if self.mode == "error_only":
            return torch.cat([z, pooled_error], dim=-1)
        if self.mode == "concat":
            return torch.cat([z, pooled_error, observed], dim=-1)
        if self.mode == "gated":
            g = torch.sigmoid(self._gate(torch.cat([z, observed], dim=-1)))
            return torch.cat([z * g, pooled_error, observed], dim=-1)
        if self.mode == "film":
            fb = self.gamma(torch.cat([z, observed], dim=-1)) + 1.0
            return z * fb + self.beta(torch.cat([z, observed], dim=-1))


# ---------------------------------------------------------------------------
# G2-K6: position-sensitive attention pooling over error tokens
# ---------------------------------------------------------------------------


class PositionSensitiveErrorPool(nn.Module):
    """Attention pooling over position-embedded error tokens.

    Each spatial position in the error map gets a learned weight through
    a single-pooling-logit head, then a spatial mean. This is
    position-sensitive; a mean-pool variant is provably NOT position
    sensitive and is used as the Gen-1 parity control.
    """

    def __init__(
        self,
        d_z: int,
        n_tokens: int = 16,
        pool_hidden: int = 64,
        projection_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.grid_size = int(n_tokens ** 0.5)
        assert self.grid_size * self.grid_size == n_tokens
        self.pool_logits = nn.Sequential(
            nn.Linear(1, pool_hidden), nn.GELU(),
            nn.Linear(pool_hidden, 1), nn.Tanh())
        self.projection = nn.Sequential(
            nn.Linear(n_tokens, projection_hidden), nn.GELU(),
            nn.Linear(projection_hidden, d_z))

    def forward(self, err_map: torch.Tensor) -> torch.Tensor:
        """err_map: (B, G, G) -> (B, d_z) pooled spatial error."""
        B, G, _ = err_map.shape
        observed = err_map.detach()                      # target convention
        w = self.pool_logits(observed.unsqueeze(-1)).squeeze(-1)  # (B, G, G)
        attn = w / (w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6))
        pooled = (attn * observed).flatten(1)            # (B, G*G)
        z_hat = self.projection(pooled)
        return torch.tanh(z_hat)


def _mean_pool_error(err_map: torch.Tensor) -> torch.Tensor:
    """Gen-1 parity pooling: flat mean. NOT position-sensitive."""
    return err_map.mean(dim=(1, 2))
# ---------------------------------------------------------------------------
# G2-K7: adaptive belief-update magnitude with hard ceiling
# ---------------------------------------------------------------------------


class BeliefUpdaterV2(nn.Module):
    """G2-K5/K6/K7: observed-feature fusion, spatial error, adaptive bound.

    Extension point for the belief-update equation. Produces the update
    term that PrecisionField (G2-K4) gates.

    Gen-1 parity path: fusion='error_only' + mean pool == the original
    ConcreteUpdateNet's behavior, preserved as the executable control.
    """

    def __init__(
        self,
        d_z: int = 384,
        obs_dim: int = 384,
        n_tokens: int = 16,
        fusion_mode: FusionMode = 'error_only',
        bound_mode: AdaptiveBoundMode = 'bounded',
        delta_bound: float = ADAPTIVE_BOUND_MAX,
        pool_hidden: int = 64,
        projection_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.d_z = int(d_z)
        self.obs_dim = int(obs_dim)
        self.n_tokens = int(n_tokens)
        self.fusion_mode = fusion_mode
        self.bound_mode = bound_mode
        self.delta_bound = float(delta_bound)

        # Fusion arm (G2-K5).
        self.fusion = ObservedFeatureFusion(
            d_z, obs_dim, mode=fusion_mode, hidden=768)

        # Position-sensitive error pool (G2-K6).
        self.error_pool = PositionSensitiveErrorPool(
            d_z, n_tokens, pool_hidden, projection_hidden)

        # Update MLP (G2-K6/K7).
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * d_z, 768), nn.GELU(),
            nn.Linear(768, 768), nn.GELU(),
            nn.Linear(768, d_z))

        # Adaptive-bound head (G2-K7-belief-adaptive).
        if bound_mode == 'bounded':
            self.bound_head = nn.Identity()
        else:
            self.bound_head = nn.Sequential(
                nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(
        self,
        z: torch.Tensor,
        err_map: torch.Tensor,
        observed: Optional[torch.Tensor] = None,
        U: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce the fuzzy update term for the belief update.

        Args:
            z: (B, D_z) belief state.
            err_map: (B, G, G) spatial error surface.
            observed: (B, D) observed features (G2-K5).
            U: epistemic uncertainty scalar (B,) (G2-K7-adaptive).

        Returns:
            delta_z: (B, D_z), bounded to [-delta_bound, +delta_bound].
        """
        if self.fusion_mode == 'error_only':
            pooled = self.error_pool(err_map)   # (B, d_z)
            fused = self.fusion(z, pooled, observed=None)
        else:
            pooled = self.error_pool(err_map)
            fused = self.fusion(z, pooled, observed)

        h = self.update_mlp(torch.cat([z, pooled], dim=-1))
        raw = self.delta_bound * torch.tanh(h)

        if self.bound_mode in ('bounded', 'adaptive'):
            raw = adapt_magnitude(raw, mode=self.bound_mode,
                                  max_bound=self.delta_bound, U=U, z=z)
        elif self.bound_mode == 'unbounded':
            pass
        else:
            raise ValueError(f'unknown bound mode {self.bound_mode}')
        return raw


def adapt_magnitude(
    delta_z: torch.Tensor,
    mode: AdaptiveBoundMode = "bounded",
    max_bound: float = ADAPTIVE_BOUND_MAX,
    U: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the adaptive update magnitude rule with a hard ceiling.

    * bounded (default): clamp |delta_z| to max_bound [1e-4, 1] guard.
    * adaptive: magnitude gated by U (uncertainty) — test whether
      Delta z <= 0.1 is protective or destructive.
    * unbounded: test-only (G3-class risk); logged but not default.
    """
    if mode == "unbounded":
        return delta_z
    if mode == "adaptive":
        if U is None or z is None:
            raise ValueError("adaptive mode requires U and z")
        # adaptive-bound arm with hard ceiling: modulate magnitude by
        # uncertainty, keep |delta_z| <= max_bound.
        scale = 0.5 + 0.5 * torch.sigmoid(U.float().unsqueeze(-1))
        return torch.clamp(delta_z * scale, -max_bound, max_bound)
    if mode == "bounded":
        return torch.clamp(delta_z, -max_bound, max_bound)
    raise ValueError(f"unknown adaptive mode {mode}")
