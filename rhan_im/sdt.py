"""Signal-detection-theory helpers (equal-variance Gaussian model).

Observation model used everywhere in RHAN-IM:

    z_c | target of class c at a location seen with detectability d'  ~  N(d', 1)
    z_c | otherwise                                                    ~  N(0,  1)

Under this model the log-likelihood ratio of "target (location b, class c)"
against "no target" for one measurement is

    llr = d' * z_c - 0.5 * d'^2

which is the same quantity Najemnik & Geisler (2005) use for their ideal
searcher. d' is *measured* (psychophysics of the model), never learned by
gradient descent, so that the Bayes update stays exact under the model.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF."""
    return 0.5 * (1.0 + torch.special.erf(x / math.sqrt(2.0)))


def gaussian_llr(z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """log N(z; d, 1) - log N(z; 0, 1) = d*z - d^2/2."""
    return d * z - 0.5 * d * d


def bayes_error_equal_prior(D: torch.Tensor) -> torch.Tensor:
    """Bayes error for telling N(D,1) from N(0,1) with equal priors: Phi(-D/2)."""
    return normal_cdf(-0.5 * D)


def dprime_from_samples(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Empirical d' = (mu_pos - mu_neg) / sqrt((var_pos + var_neg) / 2)."""
    pos = pos.double().flatten()
    neg = neg.double().flatten()
    if pos.numel() < 2 or neg.numel() < 2:
        raise ValueError("need at least two samples per class")
    pooled = 0.5 * (pos.var(unbiased=True) + neg.var(unbiased=True))
    return float((pos.mean() - neg.mean()) / pooled.clamp_min(1e-12).sqrt())


def zroc_slope(pos: torch.Tensor, neg: torch.Tensor, n_points: int = 25, lo: float = 0.01, hi: float = 0.99) -> Tuple[float, float]:
    """Slope and intercept of the z-transformed ROC (NaN, NaN if it cannot be measured).

    Equal-variance Gaussian evidence gives slope 1; the slope equals sd_neg / sd_pos. A slope far from 1
    means the equal-variance SDT assumption is violated and the Bayes update built on it will be
    mis-calibrated (the risk-control layer exists for that case).

    Thresholds are placed at quantiles of BOTH samples and only points whose hit AND false-alarm rates lie
    in (lo, hi) are used, because rates near 0 or 1 are clamped and bias the slope towards 0. When the two
    distributions barely overlap (d' above roughly 4.5) fewer than 5 usable points remain and the slope is
    reported as NaN instead of a biased number."""
    pos = pos.double().flatten()
    neg = neg.double().flatten()
    qs = torch.linspace(0.02, 0.98, n_points, dtype=torch.double)
    thr = torch.cat([torch.quantile(neg, qs), torch.quantile(pos, qs)]).sort().values
    hit = (pos[None, :] > thr[:, None]).double().mean(1)
    fa = (neg[None, :] > thr[:, None]).double().mean(1)
    keep = (hit > lo) & (hit < hi) & (fa > lo) & (fa < hi)
    if int(keep.sum()) < 5:
        return float("nan"), float("nan")
    normal = torch.distributions.Normal(0.0, 1.0)
    zh, zf = normal.icdf(hit[keep]), normal.icdf(fa[keep])
    xm, ym = zf.mean(), zh.mean()
    slope = ((zf - xm) * (zh - ym)).sum() / ((zf - xm) ** 2).sum().clamp_min(1e-12)
    intercept = ym - slope * xm
    return float(slope), float(intercept)
