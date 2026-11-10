"""The perception loop: look, absorb, decide whether to look again.

run_search runs every trial for the full budget K_max and records the posterior after each
look. Because the fixation policy never depends on the stopping rule, the trace of a trial
that would have stopped at look t is exactly the first t columns of its full trace, so any
stopping rule (any threshold, any certificate) can be evaluated offline from one run
(`stopping.stopped_stats`). tests/rhan_im/test_agent.py checks this equivalence against a
run that really stops, for several thresholds (including one above the prior risk) and several
minimum-look floors k_min.

The stopping rule that is implemented, offline and online alike: stop at the first look
t >= k_min (k_min >= 1) whose posterior risk is <= stop_lambda, else at the budget. A
leave-one-look-out margin (04 section 7.4) is NOT implemented.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from rhan_im.belief import SearchPosterior
from rhan_im.policy import evoi_fixation, greedy_mode_fixation, koopman_fixation, random_fixation

POLICIES = ("koopman", "evoi", "map", "random", "random_wor", "tiling", "fixed_center")


@torch.no_grad()
def run_search(
    scene,
    vis: torch.Tensor,
    policy: str,
    K_max: int,
    p_null: float,
    mode: str = "sum",
    kappa: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    n_mc: int = 32,
    order: Optional[torch.Tensor] = None,
    log_prior_cells: Optional[torch.Tensor] = None,
    believed_scale: float = 1.0,
    stop_lambda: Optional[float] = None,
    footprint: Optional[torch.Tensor] = None,
    unit_cap: Optional[float] = None,
    hook: Optional[Callable] = None,
    k_min: int = 1,
) -> Dict[str, torch.Tensor]:
    """Run `policy` for up to K_max looks. Returns traces with K_max + 1 columns (column 0 is
    the state before any look). If stop_lambda is given, a trial really stops (its posterior
    is frozen and its look counter stops) at the first look t >= k_min whose risk is <= stop_lambda
    (k_min >= 1: there is always at least one look, even when the prior risk is already below
    stop_lambda). hook(t, post, k, z) is called after the prior (t = 0, k = z = None) and after
    every look."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if k_min < 1:
        raise ValueError("k_min must be >= 1")
    B, C = scene.B, scene.C
    K, V = vis.shape
    N = footprint.shape[0] if footprint is not None else V
    belief_vis = vis * believed_scale
    post = SearchPosterior(N, C, B, p_null=p_null, log_prior_cells=log_prior_cells, mode=mode, footprint=footprint, unit_cap=unit_cap, n_units=V)
    visited = torch.zeros(B, K, dtype=torch.bool)
    active = torch.ones(B, dtype=torch.bool)
    names = ("dec", "risk", "ign", "ent", "p_null")
    rec = {n: [] for n in names}

    def record() -> None:
        idx, p = post.decision()
        rec["dec"].append(idx)
        rec["risk"].append(1.0 - p)
        rec["ign"].append(post.ignorance_mass())
        rec["ent"].append(post.entropy())
        rec["p_null"].append(post.p_null())

    record()
    if hook is not None:
        hook(0, post, None, None)
    fixes = []
    for t in range(K_max):
        if policy == "koopman":
            k = koopman_fixation(post, belief_vis)
        elif policy == "evoi":
            k = evoi_fixation(post, belief_vis, n_mc, generator, kappa)
        elif policy == "map":
            k = greedy_mode_fixation(post, belief_vis, generator)
        elif policy == "random":
            k = random_fixation(B, K, generator)
        elif policy == "random_wor":
            k = random_fixation(B, K, generator, exclude=visited)
        elif policy == "tiling":
            if order is None:
                raise ValueError("policy 'tiling' needs `order`")
            k = order[t % order.numel()].expand(B).clone()
        else:  # fixed_center: look at the middle cell every time
            k = torch.full((B,), K // 2 + (int(K ** 0.5) // 2), dtype=torch.long) if K > 1 else torch.zeros(B, dtype=torch.long)
        z = scene.measure(k)
        post.update(z, belief_vis[k], kappa=kappa, active=active)
        visited[torch.arange(B), k] = True
        fixes.append(k)
        record()
        if hook is not None:
            hook(t + 1, post, k, z)
        if stop_lambda is not None and t + 1 >= k_min:
            active = active & ~(post.risk() <= stop_lambda)          # not "> lam": a NaN risk must never stop a trial (offline: NaN <= lam is False, so it is no hit)
    out = {n: torch.stack(v, 1) for n, v in rec.items()}
    out["fix"] = torch.stack(fixes, 1)
    out["label"] = scene.label
    out["correct"] = out["dec"] == scene.label[:, None]
    out["n_fix"] = post.n_fix
    return out
