"""Brute force with the features the first brute-force test did not have: clipping, the per-unit cap, the active mask and branching.

An independent reviewer wrote this enumeration (different sizes, a non-uniform prior, wide scores so that the clip bites, an exact tie
in d across looks, a worse view arriving after a better one and a better view arriving last) and found rhan_im.belief.SearchPosterior
equal to it to 1e-15. It is kept here because the dossier said 'brute-force verified' for these features when tests/rhan_im/test_belief.py
covered footprints and both modes only.

What the enumeration establishes, and what it does not:
  * with clip and no cap the posterior IS the Bayes posterior over the 1 + N C hypotheses of the model that scores each (unit, class) score,
    clipped to [-kappa, kappa], with an N(d, 1) density under the hypothesis and an N(0, 1) density otherwise; in 'best' mode only the
    best view of each unit enters;
  * the cap is a saturation of the running accumulator, applied after every look, so the result equals the CLIPPED / CAPPED RECURSION
    (written out below) and depends on the order of the looks. It is not the posterior of an exchangeable-looks model.
"""
import math

import pytest
import torch

from rhan_im.belief import SearchPosterior

DT = torch.float64
N, V, C = 4, 6, 3
F = torch.tensor([[1, 1, 0, 0, 1, 0],
                  [0, 1, 1, 0, 1, 1],
                  [0, 0, 1, 1, 0, 1],
                  [1, 0, 0, 1, 1, 1]], dtype=DT)
P_NULL = 0.27
T = 5


def make_world(seed=12345):
    g = torch.Generator().manual_seed(seed)
    lp_cells = torch.randn(1, N, C, dtype=DT, generator=g) * 0.8                          # non-uniform prior
    looks = []
    for _ in range(T):
        z = torch.randn(V, C, dtype=DT, generator=g) * 1.8                                # wide, so that the clip bites
        d = torch.rand(V, dtype=DT, generator=g) * 2.2
        looks.append([z, d])
    looks[1][1][0] = 0.0                                                                  # a unit that is not seen at all in one look
    looks[3][1][2] = looks[0][1][2]                                                       # an exact tie in d for unit 2
    looks[4][1][3] = looks[2][1][3] * 0.5                                                 # a worse view arrives after a better one (unit 3)
    looks[4][1][5] = looks[0][1][5] + 0.4                                                 # a better view arrives last (unit 5)
    return lp_cells, looks


def lnorm(x, m):
    return -0.5 * (x - m) ** 2 - 0.5 * math.log(2 * math.pi)


def log_prior_vec(lp_cells):
    lp = lp_cells[0] - torch.logsumexp(lp_cells[0].flatten(), 0) + math.log(1 - P_NULL)
    return torch.cat([torch.tensor([math.log(P_NULL)], dtype=DT), lp.flatten()])


def brute(lp_cells, looks, mode, kappa=None, cap=None):
    """Explicit enumeration of the 1 + N C hypotheses with explicit normal densities."""
    H = 1 + N * C
    lpv = log_prior_vec(lp_cells)
    if cap is None:
        logp = lpv.clone()
        if mode == "best":
            sel = {}
            for ti, (z, d) in enumerate(looks):
                for u in range(V):
                    if float(d[u]) > sel.get(u, (None, 0.0))[1]:
                        sel[u] = (ti, float(d[u]))
            used = [(ti, u, dd) for u, (ti, dd) in sel.items()]
        else:
            used = [(ti, u, float(d[u])) for ti, (z, d) in enumerate(looks) for u in range(V)]
        for h in range(H):
            for ti, u, dd in used:
                zz = looks[ti][0][u]
                if kappa is not None:
                    zz = zz.clamp(-kappa, kappa)
                for c in range(C):
                    if h == 0:
                        m = 0.0
                    else:
                        b, cc = divmod(h - 1, C)
                        m = dd if (cc == c and F[b, u] == 1) else 0.0
                    logp[h] += lnorm(zz[c], m)
        return torch.softmax(logp, 0)
    # with the cap: the saturating accumulator, written out per (unit, class) as the specification words it
    acc = torch.zeros(V, C, dtype=DT)
    dbest = torch.zeros(V, dtype=DT)
    for z, d in looks:
        for u in range(V):
            zz = z[u]
            if kappa is not None:
                zz = zz.clamp(-kappa, kappa)
            for c in range(C):
                ll = lnorm(zz[c], float(d[u])) - lnorm(zz[c], 0.0)
                if mode == "sum":
                    acc[u, c] = acc[u, c] + ll
                elif float(d[u]) > float(dbest[u]):
                    acc[u, c] = ll
            if mode == "best" and float(d[u]) > float(dbest[u]):
                dbest[u] = d[u]
            acc[u] = acc[u].clamp(-cap, cap)
    logp = lpv.clone()
    for h in range(1, H):
        b, c = divmod(h - 1, C)
        logp[h] += sum(acc[u, c] for u in range(V) if F[b, u] == 1)
    return torch.softmax(logp, 0)


def impl(lp_cells, looks, mode, kappa=None, cap=None, active=None):
    post = SearchPosterior(N, C, 1, p_null=P_NULL, log_prior_cells=lp_cells, mode=mode, footprint=F, unit_cap=cap, n_units=V, dtype=DT)
    for ti, (z, d) in enumerate(looks):
        act = None if active is None else torch.tensor([active[ti]])
        post.update(z[None], d[None], kappa=kappa, active=act)
    return post, post.joint_log_weights().exp()[0]


@pytest.mark.parametrize("mode", ["sum", "best"])
@pytest.mark.parametrize("kappa", [None, 1.7])
@pytest.mark.parametrize("cap", [None, 2.3])
def test_posterior_equals_explicit_enumeration_with_clip_cap_and_both_modes(mode, kappa, cap):
    lp_cells, looks = make_world()
    _, got = impl(lp_cells, looks, mode, kappa, cap)
    ref = brute(lp_cells, looks, mode, kappa, cap)
    assert float((got - ref).abs().max()) < 1e-12
    assert float(got.sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("mode", ["sum", "best"])
def test_a_masked_look_is_exactly_absent(mode):
    lp_cells, looks = make_world()
    pattern = [True, False, True, True, False]
    post, got = impl(lp_cells, looks, mode, kappa=1.7, active=pattern)
    ref = brute(lp_cells, [l for l, a in zip(looks, pattern) if a], mode, 1.7, None)
    assert float((got - ref).abs().max()) < 1e-12
    assert post.n_fix.tolist() == [3]


def test_branching_equals_independent_posteriors():
    lp_cells, looks = make_world()
    g = torch.Generator().manual_seed(7)
    post = SearchPosterior(N, C, 1, p_null=P_NULL, log_prior_cells=lp_cells, mode="best", footprint=F, unit_cap=2.3, n_units=V, dtype=DT)
    for z, d in looks[:3]:
        post.update(z[None], d[None], kappa=1.7)
    rep = post.repeat(3)
    zz = torch.randn(3, V, C, dtype=DT, generator=g)
    dd = torch.rand(3, V, dtype=DT, generator=g) * 2
    rep.update(zz, dd, kappa=1.7)
    for i in range(3):
        alone = SearchPosterior(N, C, 1, p_null=P_NULL, log_prior_cells=lp_cells, mode="best", footprint=F, unit_cap=2.3, n_units=V, dtype=DT)
        for z, d in looks[:3]:
            alone.update(z[None], d[None], kappa=1.7)
        alone.update(zz[i:i + 1], dd[i:i + 1], kappa=1.7)
        assert torch.allclose(alone.log_cells, rep.log_cells[i:i + 1], atol=1e-12)


def test_the_cap_makes_the_result_depend_on_the_order_of_the_looks_and_without_it_the_sum_does_not():
    """The cap saturates the running accumulator after every look. One unit, one class, d = 1 in every look, scores +3, +3, -3 against
    the cap 2: the accumulator ends at -1.5 in that order and at +2.0 in the order -3, +3, +3; without the cap both end at +1.5."""
    z = [3.0, 3.0, -3.0]
    for cap, expect in ((None, (1.5, 1.5)), (2.0, (-1.5, 2.0))):
        ends = []
        for order in ([0, 1, 2], [2, 0, 1]):
            post = SearchPosterior(1, 1, 1, p_null=0.5, mode="sum", unit_cap=cap)
            for i in order:
                post.update(torch.full((1, 1, 1), z[i]), torch.ones(1, 1))
            ends.append(float(post.log_cells[0, 0, 0] - post.log_null[0]))
        assert ends == pytest.approx(list(expect), abs=1e-5), (cap, ends)
    lp_cells, looks = make_world()
    base = impl(lp_cells, looks, "sum", 1.7)[1]
    rev = impl(lp_cells, looks[::-1], "sum", 1.7)[1]
    assert float((base - rev).abs().max()) < 1e-12                                          # no cap: order-invariant
    cap_a, cap_b = impl(lp_cells, looks, "sum", 1.7, 2.3)[1], impl(lp_cells, looks[::-1], "sum", 1.7, 2.3)[1]
    assert float((cap_a - cap_b).abs().max()) > 1e-3                                        # cap: order-dependent


def test_best_mode_is_exact_only_for_a_degradation_chain_and_otherwise_a_deliberate_approximation():
    """Two Gaussian views of one unit, detectabilities d1 < d2, noise correlation rho. The exact log-likelihood ratio has the weights
    Sigma^-1 d. They are (0, d2), i.e. 'keep only the best view', exactly when rho = d1 / d2 (the worse view is a noisy copy of the better
    one). At any other rho the best view alone discards information: the exact information d' Sigma^-1 d is larger than d2^2 (for independent
    views it is d1^2 + d2^2, which is what 'sum' mode uses; as rho -> 1 it diverges)."""
    d = torch.tensor([0.5, 1.5], dtype=DT)
    for rho in (1 / 3, 0.0, 0.6, 0.99):
        S = torch.tensor([[1.0, rho], [rho, 1.0]], dtype=DT)
        w = torch.linalg.solve(S, d)
        info = float(d @ w)
        assert info >= float(d[1] ** 2) - 1e-12                                              # more information can never hurt
        if abs(rho - 1 / 3) < 1e-12:
            assert abs(float(w[0])) < 1e-12 and info == pytest.approx(2.25)                  # the degradation chain: 'best' is exact
        else:
            assert abs(float(w[0])) > 1e-3 and info > 2.25 + 1e-3                            # otherwise it is an approximation
    assert float(d @ torch.linalg.solve(torch.eye(2, dtype=DT), d)) == pytest.approx(0.25 + 2.25)   # rho = 0: sum mode is exact
