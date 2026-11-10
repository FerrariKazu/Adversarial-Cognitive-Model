"""Next-look policies. All of them are *computed* from the posterior; nothing is learned by RL.

candidates   K fixations; vis (K, V) = believed detectability d'(k, u) of evidence unit u from fixation k.

koopman_gain      closed-form one-step reduction of the ignorance mass
                      sum_b p_b [ Phi(-sqrt(D2_b)/2) - Phi(-sqrt(D2'_b(k))/2) ]
                  (the discrete analogue of Koopman's marginal-return rule for search effort,
                  with the detection function G(D2) = Phi(sqrt(D2)/2)); O(B*K*N*V) with the dense
                  (N, V) footprint (O(B*K*N) when units are cells, F = None; a sparse footprint with
                  s units per cell would make it O(B*K*N*s)).
mc_expected_max_posterior
                  exact one-step look-ahead under the model: E[max_o P(o | data, new look)],
                  estimated by posterior-predictive Monte-Carlo with common random numbers.
greedy_mode_fixation
                  look at the cell with the highest marginal posterior (the "MAP searcher" of
                  Najemnik & Geisler 2005).
random_fixation   uniform baseline (optionally without replacement).
"""
from __future__ import annotations

from typing import Optional

import torch

from rhan_im.belief import SearchPosterior
from rhan_im.sdt import bayes_error_equal_prior


def _new_D2_units(post: SearchPosterior, vis: torch.Tensor) -> torch.Tensor:
    """(B, K, V) precision of each evidence unit after one more look from each candidate."""
    d2 = vis[None] ** 2
    if post.mode == "best":
        return torch.maximum(post.D2[:, None, :], d2)
    return post.D2[:, None, :] + d2


@torch.no_grad()
def koopman_gain(post: SearchPosterior, vis: torch.Tensor) -> torch.Tensor:
    """(B, K) expected reduction of the ignorance mass from one more look at each candidate."""
    p = post.p_loc()[:, None, :]
    cur = bayes_error_equal_prior(post.hyp_precision().sqrt())[:, None, :]
    new = bayes_error_equal_prior(post.hyp_precision(_new_D2_units(post, vis)).sqrt())
    return (p * (cur - new)).sum(-1)


@torch.no_grad()
def koopman_fixation(post: SearchPosterior, vis: torch.Tensor, exclude: Optional[torch.Tensor] = None) -> torch.Tensor:
    g = koopman_gain(post, vis)
    if exclude is not None:
        g = g.masked_fill(exclude, float("-inf"))
    return g.argmax(1)


@torch.no_grad()
def mc_posterior_outcomes(
    post: SearchPosterior,
    vis: torch.Tensor,
    n_mc: int = 32,
    generator: Optional[torch.Generator] = None,
    kappa: Optional[float] = None,
) -> torch.Tensor:
    """(B, K, M, 1 + C): posterior outcome probabilities after one more look from every
    candidate, for M hypotheses drawn from the current posterior and the evidence they
    would generate. The same hypotheses and the same noise are used for every candidate
    (common random numbers), which makes differences between candidates much less noisy."""
    B, N, C, K = post.B, post.N, post.C, vis.shape[0]
    q = post.joint_log_weights().exp()                                   # (B, 1 + N*C)
    idx = torch.multinomial(q, n_mc, replacement=True, generator=generator)   # (B, M)
    oh = torch.nn.functional.one_hot(idx, 1 + N * C)[..., 1:]             # (B, M, N*C); null -> zeros
    oh = oh.reshape(B, n_mc, N, C).to(vis.dtype)                          # hypothesis (anchor, class)
    ind = oh if post.F is None else torch.einsum("bmhc,hu->bmuc", oh, post.F)   # unit-level mean indicator
    eps = torch.randn(B, n_mc, post.V, C, generator=generator, dtype=vis.dtype)
    dk = vis[None, :, None, :, None]                                       # (1, K, 1, N, 1)
    z = dk * ind[:, None] + eps[:, None]                                   # (B, K, M, N, C)
    if kappa is not None:
        z = z.clamp(-kappa, kappa)
    llr = dk * z - 0.5 * dk * dk
    if post.mode == "sum":
        u = post.u_contrib[:, None, None] + llr
    else:
        better = dk > post.d_best[:, None, None, :, None]                  # (B, K, 1, N, 1)
        u = torch.where(better, llr, post.u_contrib[:, None, None])
    if post.unit_cap is not None:
        u = u.clamp(-post.unit_cap, post.unit_cap)
    proj = u if post.F is None else torch.einsum("hu,bkmuc->bkmhc", post.F, u)
    ell = post.log_prior[:, None, None] + proj
    lc = torch.logsumexp(ell, dim=3)                                       # (B, K, M, C)
    l0 = post.log_null[:, None, None, None].expand(B, K, n_mc, 1)
    return torch.softmax(torch.cat([l0, lc], dim=-1), dim=-1)


@torch.no_grad()
def mc_expected_max_posterior(
    post: SearchPosterior,
    vis: torch.Tensor,
    n_mc: int = 32,
    generator: Optional[torch.Generator] = None,
    kappa: Optional[float] = None,
    max_elems: float = 2.0e7,
) -> torch.Tensor:
    """(B, K) expected posterior probability of the Bayes decision after one more look."""
    B, N, C, K = post.B, post.N, post.C, vis.shape[0]
    bs = max(1, int(max_elems // (K * n_mc * max(N, post.V) * C)))
    out = []
    for i0 in range(0, B, bs):
        i1 = min(B, i0 + bs)
        sub = post.slice(i0, i1)
        pr = mc_posterior_outcomes(sub, vis, n_mc, generator, kappa)
        out.append(pr.max(-1).values.mean(-1))
    return torch.cat(out, 0)


@torch.no_grad()
def evoi_fixation(
    post: SearchPosterior,
    vis: torch.Tensor,
    n_mc: int = 32,
    generator: Optional[torch.Generator] = None,
    kappa: Optional[float] = None,
    exclude: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    g = mc_expected_max_posterior(post, vis, n_mc, generator, kappa)
    if exclude is not None:
        g = g.masked_fill(exclude, float("-inf"))
    return g.argmax(1)


@torch.no_grad()
def greedy_mode_fixation(post: SearchPosterior, vis: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Fixate the cell with the highest marginal posterior (candidates are the cell centres);
    exact ties (e.g. the first look under a flat prior) are broken at random."""
    if vis.shape[0] != post.N:
        raise ValueError("greedy_mode_fixation needs one candidate per cell (K == N)")
    p = post.p_loc()
    jitter = 1e-6 * torch.rand(p.shape, generator=generator, dtype=p.dtype)
    return (p + jitter).argmax(1)


def random_fixation(
    batch: int,
    n_candidates: int,
    generator: Optional[torch.Generator] = None,
    exclude: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if exclude is None:
        return torch.randint(0, n_candidates, (batch,), generator=generator)
    w = (~exclude).to(torch.float32)
    empty = w.sum(1) == 0
    w = torch.where(empty[:, None], torch.ones_like(w), w)
    return torch.multinomial(w, 1, generator=generator).squeeze(1)
