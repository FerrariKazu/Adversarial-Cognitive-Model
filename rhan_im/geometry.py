"""Grid geometry and visibility maps.

A scene canvas is divided into a gh x gw grid of cells. A fixation is the centre of one
cell (so the candidate set has K = gh*gw elements and fixation k looks at cell k). The
visibility matrix vis (K, N) gives the detectability d'(k, b) with which cell b is seen
from fixation k. In the prototype it is either a parametric toy fall-off (below) or a
measured table (sensors.GaborImageWorld.calibrate; on real models, gate-1 measurement).
"""
from __future__ import annotations

import torch


def grid_centers(gh: int, gw: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(N, 2) (row, col) cell centres in cell units, row-major."""
    r = torch.arange(gh, dtype=dtype)
    c = torch.arange(gw, dtype=dtype)
    rr, cc = torch.meshgrid(r, c, indexing="ij")
    return torch.stack([rr.flatten(), cc.flatten()], dim=1)


def eccentricity(fix: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """(K, N) Euclidean distance, in cell units, between fixations and cells."""
    return torch.cdist(fix, cells)


def visibility_gaussian(ecc: torch.Tensor, d0: float, sigma: float) -> torch.Tensor:
    """d0 * exp(-ecc^2 / (2 sigma^2)). Toy fall-off used by the theory checks."""
    return d0 * torch.exp(-0.5 * (ecc / sigma) ** 2)


def visibility_hyperbolic(ecc: torch.Tensor, d0: float, e2: float) -> torch.Tensor:
    """d0 / (1 + ecc / e2). Hyperbolic fall-off (parametrisation is ours, not a fitted law)."""
    return d0 / (1.0 + ecc / e2)


def visibility_window(ecc: torch.Tensor, d0: float, radius: float) -> torch.Tensor:
    """d0 inside `radius` cells, 0 outside: a hard native-resolution window."""
    return d0 * (ecc <= radius).to(ecc.dtype)


def tiling_order(gh: int, gw: int, stride: int, lattice_only: bool = False) -> torch.Tensor:
    """Fixed coverage order: a strided lattice of fixations first (row-major), then the rest.

    This is the fair uniform-coverage baseline for a window that sees about `stride` cells
    across: it visits every part of the canvas once before it repeats anything. With
    lattice_only=True only the lattice is returned, so a caller that cycles through it keeps
    re-covering the canvas evenly (the strongest non-adaptive baseline).
    """
    n = gh * gw
    first = []
    for r in range(stride // 2, gh, stride):
        for c in range(stride // 2, gw, stride):
            first.append(r * gw + c)
    if lattice_only:
        return torch.tensor(first, dtype=torch.long)
    rest = [i for i in range(n) if i not in set(first)]
    return torch.tensor(first + rest, dtype=torch.long)
