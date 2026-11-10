"""Stopping, risk control and bounded-influence certificates.

Everything here is a statement about the *evidence channel* of rhan_im.belief.SearchPosterior
when every score is clipped to [-kappa, kappa] and every detectability is at most d_max.

max_step_swing / future_swing_bound / curtail_certificate
    Lossless early exit ("curtailment"): if the outcome-level log-margin already exceeds
    the largest change any admissible future evidence could make, the decision at the end
    of the budget is the decision now, so the remaining looks can be skipped.
patch_certificate
    If an adversary controls the evidence of at most m cells (in every look), the decision
    is provably unchanged when the margin exceeds the worst-case net shift of m cells.
min_fixations_to_reach
    No admissible evidence can raise the log-odds of any hypothesis faster than
    max_step_gain per look, so a stop threshold cannot be reached (by honest data or by an
    attacker) in fewer looks than this.
binom_cdf / ltt_fixed_sequence
    Learn-then-Test (Angelopoulos, Bates, Candes, Jordan, Lei 2021) with the exact binomial
    tail and fixed-sequence testing: picks the stopping threshold with a finite-sample
    guarantee  P( R(lambda_hat) <= alpha ) >= 1 - delta  over the calibration draw. What that
    needs (the exact binomial tail is not implied by exchangeability alone):
      * the n calibration losses are INDEPENDENT draws from the deployment distribution of
        scenes (i.i.d. at the level of the unit that is resampled: scenes cut from the same
        source image are not independent, so split and count by source image);
      * the calibration split is DISJOINT from every split used to fit anything upstream that
        the loss depends on (evidence head, standardiser, d', VIF, kappa, policy settings),
        and the threshold grid and k_min are fixed before the calibration losses are seen;
      * the losses are 0/1 and the columns are ordered from most conservative to most aggressive
        before looking at the data (fixed-sequence testing).
    The selective variant (stopped_stats_selective) controls the JOINT rate
    P(decided AND wrong), not the error rate among decided scenes P(wrong | decided); report the
    coverage next to it.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

import torch

from rhan_im.belief import SearchPosterior


# ----------------------------------------------------------------------------------------
# swing bounds
# ----------------------------------------------------------------------------------------
def max_step_swing(d_max: float, kappa: float) -> float:
    """Largest |change| of one hypothesis' log-weight from one look: |d z - d^2/2| <= kappa d + d^2/2."""
    return kappa * d_max + 0.5 * d_max * d_max


def max_step_gain(d_max: float, kappa: float) -> float:
    """Largest *increase* of a hypothesis' log-odds against H_0 from one look,
    max over d in [0, d_max] of (kappa d - d^2/2)."""
    d = min(kappa, d_max)
    return kappa * d - 0.5 * d * d


def future_swing_bound(d_max: float, kappa: float, remaining: int, mode: str = "sum") -> float:
    """Upper bound on |change| of any hypothesis' log-weight over `remaining` more looks.

    mode "sum":  each look can move a log-weight by at most g = kappa d_max + d_max^2 / 2.
    mode "best": only the best view of a cell counts, so the final contribution is a single
                 clipped view; |final - now| <= g + g regardless of the number of looks.
    """
    if remaining <= 0:
        return 0.0
    g = max_step_swing(d_max, kappa)
    return remaining * g if mode == "sum" else 2.0 * g


def min_fixations_to_reach(log_odds_needed: float, d_max: float, kappa: float) -> int:
    """Fewest looks after which any hypothesis' log-odds against H_0 can have risen by
    `log_odds_needed` (0 if nothing is needed). In mode "best" a single look is enough
    only if it is clipped-extreme; the bound is the same."""
    if log_odds_needed <= 0:
        return 0
    g = max_step_gain(d_max, kappa)
    if g <= 0:
        return 10 ** 9
    return int(math.ceil(log_odds_needed / g))


def min_fixations_to_decide(
    log_margin_now: float,
    risk_target: float,
    d_max: float,
    kappa: float,
    footprint_size: int = 1,
    rival_is_null: bool = True,
) -> int:
    """Fewest looks after which an attacker who controls EVERY unit's evidence (scores clipped to
    [-kappa, kappa]) can push an outcome to posterior risk <= risk_target.

    P(outcome) >= 1 - risk_target forces its log-weight margin over every rival to be at least
    log((1 - risk_target) / risk_target). One look can raise a hypothesis' log-weight by at most
    footprint_size * max_step_gain (all its units at the best clipped score) and lower a rival's by at
    most footprint_size * max_step_swing; H_0 never moves. log_margin_now is the current
    log(W_outcome / W_rival). With the null as the rival this is the largest of the per-rival bounds.
    """
    need = math.log((1.0 - risk_target) / risk_target) - log_margin_now
    if need <= 0:
        return 0
    g_up = max_step_gain(d_max, kappa)
    g_dn = max_step_swing(d_max, kappa)
    per_look = footprint_size * (g_up if rival_is_null else g_up + g_dn)
    if per_look <= 0:
        return 10 ** 9
    return int(math.ceil(need / per_look))


@torch.no_grad()
def curtail_certificate_simple(
    post: SearchPosterior,
    d_max: float,
    kappa: float,
    remaining: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scalar-swing version: certified iff the outcome log-margin exceeds the largest change
    any future evidence could make to the winner and to the rival (H_0 never moves).
    Looser than curtail_certificate; kept as the one-line rule and as a test reference."""
    lw = post.outcome_log_weights()                                      # (B, 1 + C)
    star = lw.argmax(1)
    swing = future_swing_bound(d_max, kappa, remaining, post.mode)
    ar = torch.arange(lw.shape[1], device=lw.device)
    required = swing * ((star != 0).to(lw.dtype)[:, None] + (ar != 0).to(lw.dtype)[None, :])
    margin = lw.gather(1, star[:, None]) - lw
    slack = (margin - required).masked_fill(ar[None, :] == star[:, None], float("inf")).min(1).values
    return slack > 0, slack


@torch.no_grad()
def curtail_certificate(
    post: SearchPosterior,
    d_cap,
    kappa: float,
    remaining: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(certified (B,) bool, slack (B,) float): lossless early exit.

    certified[i] means: whatever admissible evidence the next `remaining` looks deliver (scores
    clipped to [-kappa, kappa], unit u never seen better than d_cap[u]), the Bayes decision of
    trial i after them equals its decision now. Every unit's accumulated evidence is bounded
    to an interval, hypothesis log-weights to the footprint-sum of those intervals, outcome
    weights to the log-sum-exp of those; the winner's lowest outcome weight is compared with every
    rival's highest. slack = log(min winner weight) - log(max rival weight) for the closest rival.

    d_cap  float or (N,): the best detectability any remaining look can give each unit
           (vis.max(0).values for a candidate set `vis`).
    mode "sum":  unit evidence in  now + remaining * [-(kappa d + d^2/2),  max_d (kappa d - d^2/2)].
    mode "best": a unit already seen at d_cap cannot change at all; any other unit ends inside
                 [min(now, -(kappa d + d^2/2)), max(now, max_d(kappa d - d^2/2))].
    A unit_cap on the posterior narrows both intervals to [-unit_cap, unit_cap].
    """
    V = post.V
    dtype = post.log_cells.dtype
    dcap = torch.as_tensor(d_cap, dtype=dtype)
    dcap = dcap.expand(V).clone() if dcap.dim() == 0 else dcap
    uc = post.u_contrib
    if remaining <= 0:
        lo_u = hi_u = uc
    else:
        g_dn = kappa * dcap + 0.5 * dcap ** 2                            # (N,)
        d_star = torch.clamp(dcap, max=kappa)
        g_up = kappa * d_star - 0.5 * d_star ** 2                        # (N,)
        if post.mode == "sum":
            lo_u = uc - remaining * g_dn[None, :, None]
            hi_u = uc + remaining * g_up[None, :, None]
        else:
            frozen = (post.d_best >= dcap[None, :] - 1e-9)[:, :, None]
            lo_u = torch.where(frozen, uc, torch.minimum(uc, (-g_dn)[None, :, None].expand_as(uc)))
            hi_u = torch.where(frozen, uc, torch.maximum(uc, g_up[None, :, None].expand_as(uc)))
        if post.unit_cap is not None:
            lo_u = lo_u.clamp(min=-post.unit_cap)
            hi_u = hi_u.clamp(max=post.unit_cap)
    lo = post.log_prior + post.project(lo_u)
    hi = post.log_prior + post.project(hi_u)
    lo_o = torch.cat([post.log_null[:, None], torch.logsumexp(lo, dim=1)], dim=1)   # (B, 1 + C)
    hi_o = torch.cat([post.log_null[:, None], torch.logsumexp(hi, dim=1)], dim=1)
    star = post.outcome_log_weights().argmax(1)
    ar = torch.arange(lo_o.shape[1], device=lo_o.device)
    slack = (lo_o.gather(1, star[:, None]) - hi_o).masked_fill(ar[None, :] == star[:, None], float("inf")).min(1).values
    return slack > 0, slack


@torch.no_grad()
def patch_certificate(
    post: SearchPosterior,
    kappa: float,
    m: int,
    max_elems: float = 3.0e7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Certificate against an adversary that may rewrite the evidence of up to m units.

    Rewriting unit u (in every look, for every class) can put its accumulated evidence anywhere in
        [-kappa S1_u - D2_u / 2,  kappa S1_u - D2_u / 2]     (intersected with +-unit_cap).
    For every patch of m units (exact enumeration, C(N, m) of them) the winner's lowest outcome
    weight and every rival's highest outcome weight are computed from those intervals (the
    footprint sums them over a hypothesis' units, classes are rewritten independently, H_0 never
    moves). certified[i] holds iff the winner stays ahead for every patch; slack is the smallest
    log(min winner weight) - log(max rival weight) over patches and rivals.

    Assumes the evidence was absorbed with this `kappa` (and temper 1) and that d <= 2 kappa, so
    that clipping each look is what bounds the influence. Cost O(B * C(N, m) * N * C).
    """
    B, V, C = post.B, post.V, post.C
    m = min(int(m), V)
    if m <= 0:
        return torch.ones(B, dtype=torch.bool), torch.full((B,), float("inf"))
    lam_lo = -kappa * post.S1 - 0.5 * post.D2
    lam_hi = kappa * post.S1 - 0.5 * post.D2
    if post.unit_cap is not None:
        lam_lo = lam_lo.clamp(min=-post.unit_cap)
        lam_hi = lam_hi.clamp(max=post.unit_cap)
    combos = torch.tensor(list(itertools.combinations(range(V), m)), dtype=torch.long)
    S = combos.shape[0]
    mask = torch.zeros(S, V, dtype=torch.bool).scatter_(1, combos, True)
    star_all = post.outcome_log_weights().argmax(1)
    slack_all = torch.full((B,), float("inf"), dtype=post.log_cells.dtype)
    bs = max(1, int(max_elems // (S * V * max(C, post.N))))
    ar = torch.arange(1 + C)
    for i0 in range(0, B, bs):
        i1 = min(B, i0 + bs)
        sub = post.slice(i0, i1)
        n = i1 - i0
        uc = sub.u_contrib[:, None]                                          # (n, 1, N, C)
        lo_unit = torch.where(mask[None, :, :, None], lam_lo[i0:i1, None, :, None], uc)
        hi_unit = torch.where(mask[None, :, :, None], lam_hi[i0:i1, None, :, None], uc)
        if sub.F is not None:
            lo_unit = torch.einsum("hu,nsuc->nshc", sub.F, lo_unit)
            hi_unit = torch.einsum("hu,nsuc->nshc", sub.F, hi_unit)
        lp = sub.log_prior[:, None]
        LO = torch.logsumexp(lp + lo_unit, dim=2)                            # (n, S, C)
        HI = torch.logsumexp(lp + hi_unit, dim=2)
        l0 = sub.log_null[:, None, None].expand(n, S, 1)
        lo_o = torch.cat([l0, LO], dim=-1)                                   # (n, S, 1 + C)
        hi_o = torch.cat([l0, HI], dim=-1)
        st = star_all[i0:i1]
        lo_star = lo_o.gather(2, st[:, None, None].expand(n, S, 1))
        sl = (lo_star - hi_o).masked_fill(ar[None, None, :] == st[:, None, None], float("inf"))
        slack_all[i0:i1] = sl.min(2).values.min(1).values
    return slack_all > 0, slack_all


# ----------------------------------------------------------------------------------------
# Learn-then-Test
# ----------------------------------------------------------------------------------------
def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= k), exact, computed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    logs = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq
        for j in range(k + 1)
    ]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def ltt_fixed_sequence(losses: torch.Tensor, alpha: float, delta: float) -> Dict[str, object]:
    """Fixed-sequence Learn-then-Test with the exact binomial tail.

    losses  (n, L) in {0, 1}: loss of calibration item i if the stopping rule is run at
            threshold j. Columns MUST be ordered in advance from the most conservative
            threshold to the most aggressive one.
    Returns the p-values, the validated prefix and the chosen (most aggressive validated)
    column, or None when even the most conservative one cannot be certified.
    The guarantee P(R(lambda_hat) <= alpha) >= 1 - delta needs 0/1 losses AND independent
    calibration items from the deployment distribution, on a split disjoint from everything the
    loss was fitted on (see the module docstring); this function cannot check either.
    """
    if losses.dim() != 2:
        raise ValueError("losses must be (n, L)")
    if not bool(((losses == 0) | (losses == 1)).all()):
        raise ValueError("exact binomial LTT needs 0/1 losses")
    n, L = losses.shape
    k = losses.sum(0).round().long().tolist()
    pvals: List[float] = [binom_cdf(int(ki), n, alpha) for ki in k]
    valid: List[int] = []
    for j in range(L):
        if pvals[j] <= delta:
            valid.append(j)
        else:
            break
    return {
        "n": n,
        "empirical_risk": (losses.double().mean(0)).tolist(),
        "pvalues": pvals,
        "valid": valid,
        "chosen": valid[-1] if valid else None,
    }


def naive_plugin_choice(losses: torch.Tensor, alpha: float) -> Optional[int]:
    """The tempting alternative: the most aggressive column whose *empirical* risk is <= alpha."""
    risk = losses.double().mean(0)
    ok = (risk <= alpha).nonzero().flatten()
    return int(ok.max()) if ok.numel() else None


def _first_eligible_look(risk: torch.Tensor, lam: float, k_min: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(first, decided): the first look t >= k_min whose posterior risk is <= lam (columns of `risk` are
    0 = before any look, 1..T = after look t), and whether there is one. Without one `first` is the last
    column (the budget). k_min >= 1: a rule that answers from the prior, before its first look, is not a
    search rule, and rhan_im.agent.run_search never does it (it always takes at least one look)."""
    if k_min < 1:
        raise ValueError("k_min must be >= 1")
    B, T1 = risk.shape
    hit = risk <= lam
    hit[:, :k_min] = False
    decided = hit.any(1)
    first = torch.where(decided, hit.to(torch.float32).argmax(1), torch.full((B,), T1 - 1, device=risk.device))
    return first, decided


def stopped_stats(risk: torch.Tensor, correct: torch.Tensor, lam: float, k_min: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-trial stopping time and correctness for the rule 'stop at the first look t >= k_min whose
    posterior risk is <= lam, else at the last look (the budget)'. risk/correct are (B, T+1) traces with
    column 0 = before any look. The same rule as run_search(stop_lambda=lam, k_min=k_min), so any lam can be
    evaluated offline from one full-budget run (tests/rhan_im/test_agent.py checks the equivalence,
    including lam above the prior risk and k_min > 1)."""
    first, _ = _first_eligible_look(risk, lam, k_min)
    return first, correct[torch.arange(risk.shape[0]), first]


def stopped_stats_selective(risk: torch.Tensor, correct: torch.Tensor, lam: float, k_min: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Selective version of stopped_stats: a trial that never reaches risk <= lam (at a look t >= k_min)
    within the budget abstains (escalates) instead of being forced to answer. Returns (looks used, decided,
    correct at the stopping look). The natural loss is 1[decided and wrong] ("committed error"): a risk
    control on it bounds the JOINT rate P(decided and wrong), not P(wrong | decided); coverage is
    decided.mean() and must be reported with it."""
    first, decided = _first_eligible_look(risk, lam, k_min)
    return first, decided, correct[torch.arange(risk.shape[0]), first]
