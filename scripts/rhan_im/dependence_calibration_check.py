#!/usr/bin/env python3
"""What does ignoring token dependence cost, and do the corrections work?  (simulation, mechanism-level)

One look of 16 tokens (a 4 x 4 block), equal detectability d per token, H_0 vs H_1 with equal priors. Five worlds:
  independent        tokens independent, no difficulty differences (the world the Bayes state assumes; control)
  equicorrelated     noise shared between the tokens of a scene, rho (Lemma 1b: exact LLR = independent sum / VIF)
  shared_difficulty  a lognormal scene-level multiplier on d, noise independent (Lemma 1c: exact LLR is a scale mixture)
  combined_mild      both at once, mild noise correlation (--rho-mild): the difficulty flag has power
  combined_strong    both at once, strong noise correlation (--rho): VIF_pos / VIF_null is close to 1, the flag is blind, and what
                     matters is how much miscalibration is left after the VIF correction alone
  (in the two combined worlds the exact LLR is the scale mixture of the VIF-corrected statistics)
Four posteriors P(H_1 | z):
  naive        the independent-token sum  sum_u (d z_u - d^2 / 2)
  vif          the naive sum divided by VIF_null (gate1.block_vif on the simulated scores; the pipeline's rule: independent if
               VIF_null <= 1.25, else corrected)
  retired      the first draft's rule: divide by max(VIF_null, VIF_pos) when that exceeds 1.25 (kept only to show why it was replaced)
  full_rule    the vif correction, then, if VIF_pos / VIF_null > 1.25, the shared-difficulty scale mixture (Lemma 1c) over the
               VIF-corrected statistics, with the spread cv ESTIMATED from the measured VIF_pos and VIF_null
               (gate1.difficulty_cv_from_vif). This is the rule the pre-registration prescribes.
  exact        the world's own likelihood with the true parameters
Reported: ECE (15 bins), mean confidence, accuracy, NLL.  Usage: python scripts/rhan_im/dependence_calibration_check.py [--n 200000]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rhan_im import gate1 as g1  # noqa: E402


def ece(p: torch.Tensor, y: torch.Tensor, bins: int = 15) -> float:
    """Expected calibration error of P(H_1) for the binary label y (confidence = max(p, 1 - p))."""
    conf = torch.maximum(p, 1 - p)
    correct = ((p > 0.5).long() == y).float()
    edges = torch.linspace(0.5, 1.0, bins + 1)
    idx = torch.bucketize(conf, edges[1:-1])
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += float(m.float().mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return out


def scores(p: torch.Tensor, y: torch.Tensor) -> dict:
    conf = torch.maximum(p, 1 - p)
    acc = ((p > 0.5).long() == y).float().mean()
    nll = -(torch.where(y == 1, p, 1 - p).clamp_min(1e-12)).log().mean()
    return {"ece": ece(p, y), "mean_confidence": float(conf.mean()), "accuracy": float(acc), "nll": float(nll)}


def lognormal_grid(cv: float, k: int = 41):
    """Quantile grid of a mean-1 lognormal with coefficient of variation cv (equal weights)."""
    s2 = math.log(1 + cv ** 2)
    u = (torch.arange(k, dtype=torch.float64) + 0.5) / k
    z = math.sqrt(2) * torch.erfinv(2 * u - 1)
    return torch.exp(math.sqrt(s2) * z - s2 / 2), torch.full((k,), 1.0 / k, dtype=torch.float64)


def mixture_llr(S: torch.Tensor, D: float, cv: float) -> torch.Tensor:
    """log E_kappa exp(kappa S - kappa^2 D / 2) for a mean-1 lognormal kappa with coefficient of variation cv (S, D already VIF-corrected)."""
    k, w = lognormal_grid(cv)
    return torch.logsumexp(w.log()[None, :] + k[None, :] * S[:, None] - (k ** 2)[None, :] * D / 2, dim=1)


KINDS = {"independent": (False, False), "equicorrelated": (True, False), "shared_difficulty": (False, True),
         "combined_mild": (True, True), "combined_strong": (True, True)}


def simulate(world: str, n: int, n_tok: int, d: float, rho: float, cv: float, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    y = (torch.rand(n, generator=gen) < 0.5).long()
    e = torch.randn(n, n_tok, generator=gen, dtype=torch.float64)
    dep, het = KINDS[world]
    if dep:
        e = math.sqrt(1 - rho) * e + math.sqrt(rho) * torch.randn(n, 1, generator=gen, dtype=torch.float64)
    kappa = torch.ones(n, dtype=torch.float64)
    if het:
        s2 = math.log(1 + cv ** 2)
        kappa = torch.exp(math.sqrt(s2) * torch.randn(n, generator=gen, dtype=torch.float64) - s2 / 2)
    z = e + (y.double() * kappa * d)[:, None]

    S, D = d * z.sum(1), n_tok * d * d                                           # footprint statistics
    naive = S - D / 2

    side = int(math.isqrt(n_tok))
    zz = z.float().reshape(n, n_tok, 1)
    lab = torch.where(y == 1, torch.zeros(n, dtype=torch.long), torch.full((n,), -1))
    vif_est = g1.block_vif(zz, y.bool()[:, None].expand(n, n_tok), lab, side, side)
    vn, vp = vif_est["vif_null"], vif_est["vif_pos"]
    thr = g1.DEFAULT_THRESHOLDS["vif_independent_max"]
    vif_used = vn if vn > thr else 1.0
    Sc, Dc = S / vif_used, D / vif_used                                          # the VIF-corrected footprint statistics
    corrected = Sc - Dc / 2
    vif_max = max(vn, vp)
    retired = naive / vif_max if vif_max > thr else naive
    cv_hat = g1.difficulty_cv_from_vif(vp, n_tok, d, vn) if vp / max(vn, 1.0) > thr else 0.0
    cv_hat = 0.0 if math.isnan(cv_hat) else cv_hat
    full = mixture_llr(Sc, Dc, cv_hat) if cv_hat > 0 else corrected

    vif_true = 1 + (n_tok - 1) * rho if dep else 1.0
    exact = mixture_llr(S / vif_true, D / vif_true, cv) if het else (S - D / 2) / vif_true
    post = lambda llr: torch.sigmoid(llr.float())
    return {"world": world, "vif_null": vn, "vif_pos": vp, "vif_used": vif_used, "cv_estimated": cv_hat, "cv_true": cv if het else 0.0,
            "naive": scores(post(naive), y), "vif_corrected": scores(post(corrected), y), "retired_max_rule": scores(post(retired), y), "full_rule": scores(post(full), y), "exact": scores(post(exact), y)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--n-tokens", type=int, default=16)
    ap.add_argument("--d", type=float, default=0.6, help="per-token detectability; 16 tokens at 0.6 give a block d' of 2.4 (error ~11 %%), where calibration is visible")
    ap.add_argument("--rho", type=float, default=0.35, help="noise correlation of the equicorrelated and combined_strong worlds")
    ap.add_argument("--rho-mild", type=float, default=0.10, help="noise correlation of the combined_mild world")
    ap.add_argument("--cv", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "docs" / "research" / "RHAN_IM_dependence_calibration.json"))
    a = ap.parse_args()
    rho_of = {"independent": 0.0, "equicorrelated": a.rho, "shared_difficulty": 0.0, "combined_mild": a.rho_mild, "combined_strong": a.rho}
    res = {"config": vars(a), "worlds": [{**simulate(w, a.n, a.n_tokens, a.d, rho_of[w], a.cv, a.seed), "rho_true": rho_of[w]} for w in KINDS]}
    for w in res["worlds"]:
        print(f"{w['world']:18s} VIF null {w['vif_null']:.2f} pos {w['vif_pos']:.2f} used {w['vif_used']:.2f} cv_hat {w['cv_estimated']:.2f} | ECE naive {w['naive']['ece']:.3f}  vif {w['vif_corrected']['ece']:.3f}  "
              f"retired {w['retired_max_rule']['ece']:.3f}  full {w['full_rule']['ece']:.3f}  exact {w['exact']['ece']:.3f} | conf naive {w['naive']['mean_confidence']:.3f} acc {w['naive']['accuracy']:.3f}")
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
