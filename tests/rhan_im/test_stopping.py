import itertools
import math

import pytest
import scipy.stats as st
import torch

from rhan_im.belief import SearchPosterior
from rhan_im.stopping import (
    binom_cdf,
    curtail_certificate,
    curtail_certificate_simple,
    future_swing_bound,
    ltt_fixed_sequence,
    max_step_gain,
    min_fixations_to_decide,
    min_fixations_to_reach,
    naive_plugin_choice,
    patch_certificate,
    stopped_stats,
    stopped_stats_selective,
)


def test_binom_cdf_matches_scipy():
    for n, k, p in [(10, 3, 0.2), (200, 7, 0.05), (1000, 60, 0.05), (50, 50, 0.3), (50, -1, 0.3), (400, 20, 0.1)]:
        assert abs(binom_cdf(k, n, p) - st.binom.cdf(k, n, p)) < 1e-10


def test_ltt_zero_losses_needs_log_delta_over_log_one_minus_alpha_items():
    alpha, delta = 0.1, 0.1
    n_need = math.ceil(math.log(delta) / math.log(1 - alpha))
    ok = ltt_fixed_sequence(torch.zeros(n_need, 1), alpha, delta)
    short = ltt_fixed_sequence(torch.zeros(n_need - 1, 1), alpha, delta)
    assert ok["chosen"] == 0 and short["chosen"] is None


def test_ltt_rejects_non_binary_losses():
    with pytest.raises(ValueError):
        ltt_fixed_sequence(torch.full((10, 2), 0.5), 0.1, 0.1)


def test_ltt_violation_rate_is_at_most_delta_while_the_naive_rule_is_not():
    g = torch.Generator().manual_seed(0)
    risks = torch.tensor([0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30])     # ordered conservative -> aggressive
    alpha, delta, n, reps = 0.10, 0.10, 150, 4000
    viol_ltt = viol_naive = 0
    for _ in range(reps):
        losses = (torch.rand(n, risks.numel(), generator=g) < risks).float()
        c = ltt_fixed_sequence(losses, alpha, delta)["chosen"]
        v = naive_plugin_choice(losses, alpha)
        viol_ltt += int(c is not None and risks[c] > alpha)
        viol_naive += int(v is not None and risks[v] > alpha)
    assert viol_ltt / reps <= delta + 0.02
    assert viol_naive / reps > 0.25            # the plug-in rule suffers from winner's curse


def test_stopped_stats_picks_the_first_crossing():
    risk = torch.tensor([[0.6, 0.4, 0.04, 0.01], [0.6, 0.5, 0.4, 0.3]])
    correct = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 0]], dtype=torch.bool)
    t, c = stopped_stats(risk, correct, 0.05)
    assert t.tolist() == [2, 3] and c.tolist() == [True, False]


def test_k_min_is_a_floor_on_the_number_of_looks_and_column_zero_is_never_a_stop():
    """A reviewer's finding: the specification promised a minimum-looks floor that the code did not have, and stopped_stats could
    stop at column 0 (the prior) although run_search always takes at least one look."""
    risk = torch.tensor([[0.01, 0.4, 0.04, 0.01], [0.6, 0.02, 0.5, 0.01], [0.6, 0.5, 0.4, 0.3]])
    correct = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=torch.bool)
    t, c = stopped_stats(risk, correct, 0.05)                                       # default k_min = 1; the prior risk 0.01 of trial 0 is not a stop
    assert t.tolist() == [2, 1, 3] and c.tolist() == [True, True, False]
    t, c = stopped_stats(risk, correct, 0.05, k_min=2)                              # trial 1 may not stop at look 1
    assert t.tolist() == [2, 3, 3] and c.tolist() == [True, True, False]
    t, decided, c = stopped_stats_selective(risk, correct, 0.05, k_min=3)
    assert t.tolist() == [3, 3, 3] and decided.tolist() == [True, True, False]
    t, decided, _ = stopped_stats_selective(risk, correct, 0.05, k_min=9)           # a floor beyond the budget: never eligible, runs to the budget
    assert t.tolist() == [3, 3, 3] and not decided.any()
    always = stopped_stats(risk, correct, 1.0)[0]                                   # a threshold that is always met stops at the first look, not before it
    assert always.tolist() == [1, 1, 1]
    for k_min in (0, -1):
        with pytest.raises(ValueError):
            stopped_stats(risk, correct, 0.05, k_min=k_min)
        with pytest.raises(ValueError):
            stopped_stats_selective(risk, correct, 0.05, k_min=k_min)


# ----------------------------------------------------------------------------- exhaustive adversaries
def line_footprint(N, half_width=1):
    i = torch.arange(N)
    return ((i[:, None] - i[None, :]).abs() <= half_width).float()


def build_state(N, C, vis, kappa, mode, seed, looks=6, strength=1.2, footprint=None, unit_cap=None):
    """A state produced by real looks that mostly support one planted hypothesis."""
    g = torch.Generator().manual_seed(seed)
    post = SearchPosterior(N, C, 1, p_null=0.3, mode=mode, footprint=footprint, unit_cap=unit_cap)
    h = int(torch.randint(1, 1 + N * C, (1,), generator=g))
    b, c = divmod(h - 1, C)
    history = []
    for _ in range(looks):
        k = int(torch.randint(0, vis.shape[0], (1,), generator=g))
        d = vis[k][None]
        z = torch.randn(1, N, C, generator=g) * 0.6
        fp = torch.zeros(N) if footprint is None else footprint[b].clone()
        if footprint is None:
            fp[b] = 1.0
        z[0, :, c] += strength * d[0] * fp
        z = z.clamp(-kappa, kappa)
        post.update(z, d, kappa=kappa)
        history.append((k, z.clone()))
    return post, history


def all_continuations(N, C, K, R, kappa, n_random=0, seed=0):
    zs = torch.tensor(list(itertools.product([-kappa, kappa], repeat=N * C)), dtype=torch.float32).view(-1, N, C)
    seqs_k = list(itertools.product(range(K), repeat=R))
    ks, zz = [], []
    g = torch.Generator().manual_seed(seed)
    for kk in seqs_k:
        if R == 1:
            for z in zs:
                ks.append(kk); zz.append(z[None])
        else:
            for i in range(len(zs)):
                for j in range(len(zs)):
                    ks.append(kk); zz.append(torch.stack([zs[i], zs[j]]))
    ks = torch.tensor(ks)
    zz = torch.stack(zz)
    if n_random:
        rk = torch.randint(0, K, (n_random, R), generator=g)
        rz = (torch.rand(n_random, R, N, C, generator=g) * 2 - 1) * kappa
        ks, zz = torch.cat([ks, rk]), torch.cat([zz, rz])
    return ks, zz


@pytest.mark.parametrize("mode", ["sum", "best"])
def test_curtail_certificate_is_sound_against_every_extreme_continuation(mode):
    N, C, K, kappa, R = 3, 2, 3, 2.0, 1
    vis = torch.tensor([[1.5, 0.8, 0.3], [0.8, 1.5, 0.8], [0.3, 0.8, 1.5]])
    d_max = float(vis.max())
    d_cap = vis.max(0).values
    ks, zz = all_continuations(N, C, K, R, kappa, n_random=3000, seed=1)
    n_cert = n_flippable_uncert = n_states = n_simple = 0
    for seed in range(60):
        post, _ = build_state(N, C, vis, kappa, mode, seed, looks=7)
        cert, slack = curtail_certificate(post, d_cap, kappa, R)
        simple, _ = curtail_certificate_simple(post, d_max, kappa, R)
        assert (not simple.item()) or cert.item(), "interval certificate must be at least as tight as the scalar rule"
        n_simple += int(simple.item())
        base = post.decision()[0].item()
        rep = post.repeat(ks.shape[0])
        for t in range(R):
            rep.update(zz[:, t], vis[ks[:, t]], kappa=kappa)
        flipped = (rep.decision()[0] != base).any().item()
        n_states += 1
        if cert.item():
            n_cert += 1
            assert not flipped, f"unsound curtailment certificate (mode={mode}, seed={seed})"
        elif flipped:
            n_flippable_uncert += 1
    assert n_cert >= 5 and n_flippable_uncert >= 5          # the test is neither vacuous nor trivially loose


def test_future_swing_in_best_mode_does_not_grow_with_the_number_of_looks():
    assert future_swing_bound(1.5, 2.0, 10, "best") == future_swing_bound(1.5, 2.0, 1, "best")
    assert future_swing_bound(1.5, 2.0, 10, "sum") == 10 * future_swing_bound(1.5, 2.0, 1, "sum")
    assert future_swing_bound(1.5, 2.0, 0, "sum") == 0.0


@pytest.mark.parametrize("mode,cap,must_certify", [("best", None, True), ("sum", 4.0, True), ("sum", None, False)])
def test_patch_certificate_is_sound_against_every_patch(mode, cap, must_certify):
    N, C, K, kappa, m, T = 9, 2, 9, 3.0, 1, 6
    F = line_footprint(N, half_width=2)                                    # a target spans 5 units
    prof = torch.tensor([3.0, 2.2, 1.6, 1.1, 0.8, 0.6, 0.5, 0.4, 0.3])
    idx = torch.arange(N)
    vis = prof[(idx[:, None] - idx[None, :]).abs()]                        # vis[k, u] = prof[|k - u|]
    n_cert = n_attackable = 0
    for seed in range(60):
        post, hist = build_state(N, C, vis, kappa, mode, seed, looks=T, strength=1.0, footprint=F, unit_cap=cap)
        cert, _ = patch_certificate(post, kappa, m)
        dec0 = post.decision()[0].item()
        z_hist = torch.stack([z[0] for _, z in hist])                      # (T, N, C)
        k_hist = torch.tensor([k for k, _ in hist])
        flipped_any = False
        # every patch of m units, every extreme rewrite of that patch's evidence in every look and class
        for cells in itertools.combinations(range(N), m):
            slots = [(t, b, c) for t in range(T) for b in cells for c in range(C)]
            bits = torch.tensor(list(itertools.product([-kappa, kappa], repeat=len(slots))), dtype=torch.float32)
            S = bits.shape[0]
            zs = z_hist[None].repeat(S, 1, 1, 1)                           # (S, T, N, C)
            for j, (t, b, c) in enumerate(slots):
                zs[:, t, b, c] = bits[:, j]
            att = SearchPosterior(N, C, S, p_null=0.3, mode=mode, footprint=F, unit_cap=cap)
            for t in range(T):
                att.update(zs[:, t], vis[k_hist[t]][None].expand(S, -1), kappa=kappa)
            if (att.decision()[0] != dec0).any():
                flipped_any = True
                break
        if cert.item():
            n_cert += 1
            assert not flipped_any, f"unsound patch certificate (mode={mode}, cap={cap}, seed={seed})"
        elif flipped_any:
            n_attackable += 1
    if must_certify:
        assert n_cert >= 3, f"certificate is vacuous here (mode={mode}, cap={cap}): {n_cert}/60"
    assert n_attackable >= 3, "test never found an attack on an uncertified state, so it proves little"


def test_min_fixations_is_tight_for_the_extremal_sequence():
    d_max, kappa, need = 1.5, 2.0, 7.0
    n = min_fixations_to_reach(need, d_max, kappa)
    g = max_step_gain(d_max, kappa)
    assert n == math.ceil(need / g) and (n - 1) * g < need <= n * g
    # the sequence z = +kappa at d = d_max realises that rate exactly (sum mode)
    post = SearchPosterior(1, 1, 1, p_null=0.5)
    start = (post.log_cells[0, 0, 0] - post.log_null[0]).item()
    for i in range(n):
        post.update(torch.full((1, 1, 1), kappa), torch.full((1, 1), d_max), kappa=kappa)
        rise = (post.log_cells[0, 0, 0] - post.log_null[0]).item() - start
        assert abs(rise - (i + 1) * g) < 1e-4


def test_global_attacker_cannot_beat_min_fixations_to_decide():
    """An attacker rewriting every unit (+kappa for one class, -kappa for the rest) is the fastest
    possible; its stopping time must not undercut the bound, and for 1 unit/1 class/1 cell it is tight."""
    kappa, d, lam = 3.0, 2.0, 0.05
    for N, C, F in [(1, 1, None), (4, 3, line_footprint(4, 1))]:
        post = SearchPosterior(N, C, 1, p_null=0.4, footprint=F)
        w = 1 if F is None else int(F.sum(1).max())
        lm = (post.outcome_log_weights()[0, 1] - post.outcome_log_weights()[0, 0]).item()
        bound = min_fixations_to_decide(lm, lam, d, kappa, footprint_size=w, rival_is_null=True)
        t_hit = None
        for t in range(1, 60):
            z = torch.full((1, N, C), -kappa)
            z[:, :, 0] = kappa
            post.update(z, torch.full((1, N), d), kappa=kappa)
            idx, p = post.decision()
            if idx.item() == 1 and 1 - p.item() <= lam:
                t_hit = t
                break
        assert t_hit is not None and t_hit >= bound, (N, C, t_hit, bound)
        if N == 1 and C == 1:
            assert t_hit == bound


def test_selective_stopping_abstains_when_the_threshold_is_never_reached():
    risk = torch.tensor([[0.6, 0.4, 0.04, 0.01], [0.6, 0.5, 0.4, 0.3]])
    correct = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 0]], dtype=torch.bool)
    t, decided, c = stopped_stats_selective(risk, correct, 0.05)
    assert t.tolist() == [2, 3] and decided.tolist() == [True, False]
