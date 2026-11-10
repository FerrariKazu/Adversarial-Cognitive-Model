"""Unit layout: how canvas pixels, native windows and a coarse peripheral view map onto the
evidence units and hypothesis cells of rhan_im.belief.SearchPosterior.

    fine units    one per patch of a g x g grid over the canvas (g = canvas / patch). A native-resolution
                  window of `window` px covers tok x tok of them (tok = window / patch) and gives each of
                  them its own token-level evidence.
    coarse units  one per token of the *gist* look (the whole canvas resampled to `window` px): a gist
                  token covers coarse_block x coarse_block fine units.
    hypotheses    N = g * g anchors; a target anchored at (r, c) occupies target_units x target_units fine
                  units (clipped at the border) and is seen by every coarse unit that overlaps them.
    candidates    K native windows whose top-left corners lie on a lattice of fine units (the window always
                  lies fully inside the canvas), plus ONE gist look appended as candidate index K.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class UnitLayout:
    canvas: int
    patch: int = 14
    window: int = 112
    target_units: int = 2

    def __post_init__(self) -> None:
        if self.canvas % self.patch or self.window % self.patch:
            raise ValueError("canvas and window must be multiples of the patch size")
        if self.canvas % self.window:
            raise ValueError("canvas must be a multiple of the window so a gist token covers whole units")
        if self.canvas < self.window:
            raise ValueError("canvas smaller than the window")

    @property
    def g(self) -> int:
        return self.canvas // self.patch

    @property
    def tok(self) -> int:
        return self.window // self.patch

    @property
    def coarse_block(self) -> int:
        return self.g // self.tok

    @property
    def n_fine(self) -> int:
        return self.g * self.g

    @property
    def n_coarse(self) -> int:
        return self.tok * self.tok

    @property
    def N(self) -> int:
        return self.n_fine

    @property
    def V(self) -> int:
        return self.n_fine + self.n_coarse

    def footprint(self) -> torch.Tensor:
        """(N, V) binary: fine units a target occupies + coarse units that overlap it."""
        g, t, cb = self.g, self.target_units, self.coarse_block
        F = torch.zeros(self.N, self.V)
        for a in range(self.N):
            ar, ac = divmod(a, g)
            for r in range(ar, min(ar + t, g)):
                for c in range(ac, min(ac + t, g)):
                    F[a, r * g + c] = 1.0
                    F[a, self.n_fine + (r // cb) * self.tok + (c // cb)] = 1.0
        return F

    def candidates(self, stride_units: int = 2) -> torch.Tensor:
        """(K, 2) top-left (row, col) of native windows, in fine units, fully inside the canvas."""
        last = self.g - self.tok
        pos = sorted(set(list(range(0, last + 1, stride_units)) + [last]))
        return torch.tensor([[r, c] for r in pos for c in pos], dtype=torch.long)

    def window_units(self, topleft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """For windows with top-left `topleft` (B, 2) in fine units: the fine unit index of each of
        the tok*tok tokens (row-major) and whether it lies inside the canvas."""
        i = torch.arange(self.tok)
        rr = topleft[:, 0, None, None] + i[None, :, None]
        cc = topleft[:, 1, None, None] + i[None, None, :]
        valid = (rr >= 0) & (rr < self.g) & (cc >= 0) & (cc < self.g)
        idx = rr.clamp(0, self.g - 1) * self.g + cc.clamp(0, self.g - 1)
        return idx.reshape(topleft.shape[0], -1), valid.reshape(topleft.shape[0], -1)

    def vis_matrix(self, d_tok: torch.Tensor, d_gist: float, topleft: torch.Tensor) -> torch.Tensor:
        """(K + 1, V) believed detectability: native window k gives its fine units the measured
        position-dependent d'(token) (tok, tok); candidate K is the gist look, d_gist on every coarse unit."""
        K = topleft.shape[0]
        vis = torch.zeros(K + 1, self.V)
        idx, valid = self.window_units(topleft)
        vals = d_tok.reshape(1, -1).expand(K, -1) * valid.to(d_tok.dtype)
        vis[:K].scatter_add_(1, idx, vals)
        vis[K, self.n_fine:] = d_gist
        return vis
