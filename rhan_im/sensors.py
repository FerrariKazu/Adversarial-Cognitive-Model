"""Worlds that produce evidence.

ToySDTWorld         scenes drawn from the Bayes model's own generative process, with optional
                    misspecification (the true detectability differs from the believed one,
                    heavy-tailed noise). Used to test the maths (calibration, policy,
                    risk control, certificates). Evidence is independent across looks.
GaborImageWorld     a small real-pixel world: oriented Gabor targets in 1/f background noise,
                    a deterministic eccentricity-dependent blur 'retina', a template-matching
                    sensor, and *measured* detectability (calibrate()). Evidence is
                    deterministic given the pixels, so the combination mode must be "best".

Both expose  scene.measure(k) -> z (B, N, C)  and  scene.label (B,) in {0 = absent, 1 + class}.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from rhan_im.geometry import eccentricity, grid_centers
from rhan_im.sdt import zroc_slope


# ----------------------------------------------------------------------------------------
# toy SDT world
# ----------------------------------------------------------------------------------------
class ToyScene:
    def __init__(self, world: "ToySDTWorld", cell: torch.Tensor, cls: torch.Tensor, gen: Optional[torch.Generator]):
        self.world, self.cell, self.cls, self.gen = world, cell, cls, gen
        self.B = cell.shape[0]
        self.C = world.C
        self.label = torch.where(cell >= 0, cls + 1, torch.zeros_like(cls))

    def measure(self, k: torch.Tensor) -> torch.Tensor:
        w = self.world
        d_true = w.true_scale * w.vis[k]                                    # (B, V)
        mean = torch.zeros(self.B, w.V, w.C, dtype=d_true.dtype)
        idx = (self.cell >= 0).nonzero().squeeze(1)
        if w.footprint is not None:
            fp = w.footprint[self.cell.clamp_min(0)]                        # (B, V): units that see the target
        else:
            fp = torch.zeros(self.B, w.V, dtype=d_true.dtype)
            fp[idx, self.cell[idx]] = 1.0
        mean[idx, :, self.cls[idx]] = (d_true * fp)[idx]
        if w.noise_df is None:
            noise = torch.randn(self.B, w.V, w.C, generator=self.gen)
        else:
            nu = float(w.noise_df)
            gam = torch.distributions.Gamma(nu / 2.0, 0.5)
            if self.gen is None:
                chi2 = gam.sample((self.B, w.V, w.C))
            else:                                                           # Gamma has no generator argument: seed it FROM the generator so runs reproduce
                seed = int(torch.randint(0, 2 ** 62, (1,), generator=self.gen))
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    chi2 = gam.sample((self.B, w.V, w.C))
            noise = torch.randn(self.B, w.V, w.C, generator=self.gen) / (chi2 / nu).sqrt()
            noise = noise / math.sqrt(nu / (nu - 2.0))                      # unit variance
        return mean + noise


class ToySDTWorld:
    def __init__(
        self,
        vis: torch.Tensor,
        n_classes: int,
        p_null: float = 0.4,
        true_scale: float = 1.0,
        noise_df: Optional[float] = None,
        footprint: Optional[torch.Tensor] = None,
        n_cells: Optional[int] = None,
    ) -> None:
        if noise_df is not None and noise_df <= 2.0:
            raise ValueError("noise_df must exceed 2 (finite variance)")
        self.footprint = footprint
        self.vis = vis
        self.K, self.V = vis.shape
        self.N = int(n_cells) if n_cells is not None else (footprint.shape[0] if footprint is not None else self.V)
        self.C = n_classes
        self.p_null = p_null
        self.true_scale = true_scale
        self.noise_df = noise_df

    def sample(self, batch: int, gen: Optional[torch.Generator] = None) -> ToyScene:
        present = torch.rand(batch, generator=gen) >= self.p_null
        cell = torch.randint(0, self.N, (batch,), generator=gen)
        cls = torch.randint(0, self.C, (batch,), generator=gen)
        cell = torch.where(present, cell, torch.full_like(cell, -1))
        return ToyScene(self, cell, cls, gen)


# ----------------------------------------------------------------------------------------
# Gabor image world
# ----------------------------------------------------------------------------------------
def _gaussian_kernel1d(sigma: float) -> torch.Tensor:
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B, 1, H, W) with reflect padding; sigma == 0 is the identity."""
    if sigma <= 0.0:
        return img
    k = _gaussian_kernel1d(sigma)
    r = (k.numel() - 1) // 2
    x = F.pad(img, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1))


def one_over_f_noise(batch: int, size: int, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """(B, 1, S, S) zero-mean unit-variance noise with a 1/f amplitude spectrum."""
    white = torch.randn(batch, size, size, generator=gen)
    spec = torch.fft.rfft2(white)
    fy = torch.fft.fftfreq(size).abs()[:, None]
    fx = torch.fft.rfftfreq(size)[None, :]
    f = torch.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    out = torch.fft.irfft2(spec / f, s=(size, size))
    out = (out - out.mean(dim=(1, 2), keepdim=True)) / out.std(dim=(1, 2), keepdim=True)
    return out[:, None]


class GaborScene:
    def __init__(self, world: "GaborImageWorld", raw: torch.Tensor, cell: torch.Tensor, cls: torch.Tensor):
        self.world, self.raw, self.cell, self.cls = world, raw, cell, cls          # raw (B, L, N, C)
        self.B, self.C = raw.shape[0], world.C
        self.label = torch.where(cell >= 0, cls + 1, torch.zeros_like(cls))

    def measure(self, k: torch.Tensor) -> torch.Tensor:
        w = self.world
        lev = w.level[k]                                                            # (B, N)
        r = self.raw.permute(0, 2, 1, 3)                                            # (B, N, L, C)
        r = r.gather(2, lev[:, :, None, None].expand(-1, -1, 1, self.C)).squeeze(2)  # (B, N, C)
        return (r - w.mu0[lev]) / w.sd0[lev]


class GaborImageWorld:
    def __init__(
        self,
        grid: int = 8,
        cell_px: int = 8,
        n_classes: int = 4,
        amplitude: float = 1.0,
        p_null: float = 0.4,
        n_levels: int = 5,
        blur_per_level: float = 1.0,
        orthogonal_templates: bool = True,
    ) -> None:
        self.g, self.cp, self.C = grid, cell_px, n_classes
        self.S, self.N = grid * cell_px, grid * grid
        self.amp, self.p_null = amplitude, p_null
        self.L = n_levels
        self.sigmas = [blur_per_level * l for l in range(n_levels)]
        self.patterns = self._make_patterns()                                      # (C, cp, cp) peak 1
        t = self.patterns.reshape(n_classes, -1)
        t = t - t.mean(1, keepdim=True)
        t = t / t.norm(dim=1, keepdim=True)
        if orthogonal_templates:                                                   # dual basis: template j responds to pattern c only if j == c
            t = torch.linalg.solve(t @ t.T, t)
        self.templates = t                                                         # (C, cp*cp)
        cells = grid_centers(grid, grid)
        ecc = eccentricity(cells, cells)                                           # (K, N) cell units
        self.level = ecc.round().long().clamp(max=n_levels - 1)                    # (K, N) retinal level
        self.mu0 = torch.zeros(n_levels, n_classes)
        self.sd0 = torch.ones(n_levels, n_classes)
        self.d_level = torch.zeros(n_levels)
        self.calibrated = False

    # -- construction ---------------------------------------------------------------
    def _make_patterns(self) -> torch.Tensor:
        cp = self.cp
        ax = torch.arange(cp, dtype=torch.float32) - (cp - 1) / 2.0
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        pats = []
        for c in range(self.C):
            th = math.pi * c / self.C
            u = xx * math.cos(th) + yy * math.sin(th)
            env = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * (cp / 4.0) ** 2))
            pats.append(env * torch.cos(2.0 * math.pi * u / (cp / 2.0)))
        return torch.stack(pats)

    # -- scenes ---------------------------------------------------------------------
    def make_canvas(self, batch: int, gen: Optional[torch.Generator], force_present: Optional[bool] = None):
        canvas = one_over_f_noise(batch, self.S, gen)
        if force_present is None:
            present = torch.rand(batch, generator=gen) >= self.p_null
        else:
            present = torch.full((batch,), bool(force_present))
        cell = torch.randint(0, self.N, (batch,), generator=gen)
        cls = torch.randint(0, self.C, (batch,), generator=gen)
        for i in present.nonzero().flatten().tolist():
            r, c = divmod(int(cell[i]), self.g)
            canvas[i, 0, r * self.cp:(r + 1) * self.cp, c * self.cp:(c + 1) * self.cp] += self.amp * self.patterns[int(cls[i])]
        cell = torch.where(present, cell, torch.full_like(cell, -1))
        return canvas, cell, cls

    def raw_responses(self, canvas: torch.Tensor) -> torch.Tensor:
        """(B, L, N, C) template responses of every cell at every retinal blur level."""
        outs = []
        for s in self.sigmas:
            x = gaussian_blur(canvas, s)
            patches = F.unfold(x, kernel_size=self.cp, stride=self.cp)             # (B, cp*cp, N)
            outs.append(torch.einsum("bpn,cp->bnc", patches, self.templates))
        return torch.stack(outs, 1)

    def make_scene(self, batch: int, gen: Optional[torch.Generator] = None) -> GaborScene:
        canvas, cell, cls = self.make_canvas(batch, gen)
        return GaborScene(self, self.raw_responses(canvas), cell, cls)

    # -- measurement of the sensor (this is what replaces a hand-set d') -----------------
    @torch.no_grad()
    def calibrate(self, n: int = 3000, gen: Optional[torch.Generator] = None, chunk: int = 250) -> Dict[str, object]:
        """Measure the sensor: null statistics, detectability per retinal level, and how far
        the equal-variance Gaussian model is from the truth (z-ROC slope, cross-class leakage).

        d'_l = (mean z at the target cell for the target class - mean z for a null cell/class)
               / pooled standard deviation, with z standardised by the null mean/std of level l.
        """
        L, C, N = self.L, self.C, self.N
        s1, s2, cnt = torch.zeros(L, C), torch.zeros(L, C), 0
        for i0 in range(0, n, chunk):
            b = min(chunk, n - i0)
            cv, _, _ = self.make_canvas(b, gen, force_present=False)
            r = self.raw_responses(cv)                                             # (b, L, N, C)
            s1 += r.sum(dim=(0, 2))
            s2 += (r ** 2).sum(dim=(0, 2))
            cnt += b * N
        mu0 = s1 / cnt
        sd0 = (s2 / cnt - mu0 ** 2).clamp_min(1e-12).sqrt()
        self.mu0, self.sd0 = mu0.clone(), sd0.clone()
        pos_z, null_z, leak = [], [], []
        for i0 in range(0, n, chunk):
            b = min(chunk, n - i0)
            ar = torch.arange(b)
            cv, cell, cls = self.make_canvas(b, gen, force_present=True)
            r = self.raw_responses(cv)
            z = (r - self.mu0[None, :, None, :]) / self.sd0[None, :, None, :]       # (b, L, N, C)
            pos_z.append(z[ar, :, cell, cls])                                       # (b, L)
            wrong = z[ar, :, cell, :]                                               # (b, L, C)
            mask = torch.ones(b, 1, C, dtype=torch.bool)
            mask[ar, 0, cls] = False
            leak.append((wrong * mask).sum(2) / (C - 1))                            # (b, L)
            cv0, _, _ = self.make_canvas(b, gen, force_present=False)
            z0 = (self.raw_responses(cv0) - self.mu0[None, :, None, :]) / self.sd0[None, :, None, :]
            nn_ = torch.randint(0, N, (b,), generator=gen)
            cc = torch.randint(0, C, (b,), generator=gen)
            null_z.append(z0[ar, :, nn_, cc])                                       # (b, L)
        pz, nz, lk = torch.cat(pos_z, 0), torch.cat(null_z, 0), torch.cat(leak, 0)
        d = (pz.mean(0) - nz.mean(0)) / (0.5 * (pz.var(0) + nz.var(0))).sqrt()
        slopes = [zroc_slope(pz[:, l], nz[:, l])[0] for l in range(L)]
        self.d_level = d.clone()
        self.calibrated = True
        return {
            "n": n,
            "d_level": [float(x) for x in d],
            "zroc_slope": [None if s_ != s_ else float(s_) for s_ in slopes],   # None: d' too high to measure the slope
            "null_std_z": [float(x) for x in nz.std(0)],
            "pos_std_z": [float(x) for x in pz.std(0)],
            "wrong_class_leak_mean_z": [float(x) for x in lk.mean(0)],
        }

    def vis(self) -> torch.Tensor:
        """(K, N) measured detectability: d'(level(k, b))."""
        if not self.calibrated:
            raise RuntimeError("call calibrate() first")
        return self.d_level[self.level].clamp_min(0.0)
