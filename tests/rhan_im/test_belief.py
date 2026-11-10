import math

import pytest
import torch

from rhan_im.belief import SearchPosterior

DT = torch.float64


def brute(p_null, N, C, looks, best, F=None, V=None):
    V = N if V is None else V
    """Posterior by explicit enumeration of hypotheses and Gaussian likelihoods."""
    H = 1 + N * C
    logp = torch.empty(H, dtype=DT)
    logp[0] = math.log(p_null)
    logp[1:] = math.log((1 - p_null) / (N * C))
    if best:
        used = {}
        for z, d in looks:
            for b in range(V):
                if float(d[b]) > used.get(b, (None, 0.0))[1]:
                    used[b] = (z[b], float(d[b]))
        data = [(b, zr, dd) for b, (zr, dd) in used.items()]
    else:
        data = [(b, z[b], float(d[b])) for z, d in looks for b in range(V)]
    for h in range(H):
        for b, zr, dd in data:
            for c in range(C):
                hb, hc = divmod(h - 1, C) if h > 0 else (-1, -1)
                inside = h > 0 and hc == c and (F[hb, b] == 1 if F is not None else hb == b)
                mean = dd if inside else 0.0
                logp[h] += torch.distributions.Normal(mean, 1.0).log_prob(zr[c].double())
    return torch.softmax(logp, 0)


def random_looks(N, C, T, seed, dmax=2.0):
    g = torch.Generator().manual_seed(seed)
    looks = []
    for _ in range(T):
        looks.append((torch.randn(N, C, generator=g, dtype=DT) * 1.5, torch.rand(N, generator=g, dtype=DT) * dmax))
    return looks


def line_footprint(N):
    i = torch.arange(N)
    return ((i[:, None] - i[None, :]).abs() <= 1).to(DT)


@pytest.mark.parametrize("footprint", [False, True])
@pytest.mark.parametrize("mode", ["sum", "best"])
def test_update_is_exact_bayes(mode, footprint):
    N, C, p_null = 4, 2, 0.35
    F = line_footprint(N) if footprint else None
    looks = random_looks(N, C, 4, seed=7)
    post = SearchPosterior(N, C, 1, p_null=p_null, mode=mode, footprint=F, dtype=DT)
    for z, d in looks:
        post.update(z[None], d[None])
    ref = brute(p_null, N, C, looks, best=(mode == "best"), F=F)
    got = post.joint_log_weights().exp()[0]
    assert torch.allclose(got, ref, atol=1e-9)


def test_probabilities_are_normalised_and_prior_is_right():
    post = SearchPosterior(5, 3, 4, p_null=0.4)
    assert torch.allclose(post.p_null(), torch.full((4,), 0.4))
    assert torch.allclose(post.p_cells().sum((1, 2)) + post.p_null(), torch.ones(4))
    assert torch.allclose(post.outcome_probs().sum(1), torch.ones(4))


def test_nonuniform_prior_is_normalised():
    lp = torch.randn(2, 4, 3)
    post = SearchPosterior(4, 3, 2, p_null=0.25, log_prior_cells=lp)
    assert torch.allclose(post.p_cells().sum((1, 2)), torch.full((2,), 0.75), atol=1e-6)


def test_window_level_scores_broadcast():
    N, C = 4, 3
    g = torch.Generator().manual_seed(1)
    z = torch.randn(2, 1, C, generator=g)
    d = torch.rand(2, N, generator=g)
    a = SearchPosterior(N, C, 2)
    b = SearchPosterior(N, C, 2)
    a.update(z, d)
    b.update(z.expand(2, N, C), d)
    assert torch.allclose(a.log_cells, b.log_cells)


def test_active_mask_freezes_finished_trials():
    N, C = 4, 3
    g = torch.Generator().manual_seed(2)
    post = SearchPosterior(N, C, 3)
    before = post.log_cells.clone()
    z, d = torch.randn(3, N, C, generator=g), torch.rand(3, N, generator=g) + 0.5
    active = torch.tensor([True, False, True])
    post.update(z, d, active=active)
    assert torch.equal(post.log_cells[1], before[1])
    assert not torch.equal(post.log_cells[0], before[0])
    assert post.n_fix.tolist() == [1, 0, 1]


@pytest.mark.parametrize("mode", ["sum", "best"])
def test_clipping_bounds_the_influence_of_one_look(mode):
    N, C, kappa = 6, 4, 2.5
    g = torch.Generator().manual_seed(3)
    post = SearchPosterior(N, C, 64, mode=mode)
    for _ in range(3):
        post.update(torch.randn(64, N, C, generator=g), torch.rand(64, N, generator=g) * 2, kappa=kappa)
    before = post.log_cells.clone()
    d = torch.rand(64, N, generator=g) * 2
    z = torch.randn(64, N, C, generator=g) * 1e6                     # hostile evidence
    post.update(z, d, kappa=kappa)
    bound = kappa * d + 0.5 * d * d                                  # per-look swing
    if mode == "best":
        bound = 2 * (kappa * d.max() + 0.5 * d.max() ** 2) * torch.ones_like(d)
    assert ((post.log_cells - before).abs() <= bound[..., None] + 1e-4).all()


def test_ignorance_mass_falls_as_precision_grows():
    post = SearchPosterior(6, 3, 2)
    m0 = post.ignorance_mass()
    post.D2 = post.D2 + 4.0
    m1 = post.ignorance_mass()
    assert (m1 < m0).all() and (m0 <= 1).all() and (m1 >= 0).all()


def test_repeat_branches_the_state():
    post = SearchPosterior(4, 2, 3, mode="best")
    g = torch.Generator().manual_seed(4)
    post.update(torch.randn(3, 4, 2, generator=g), torch.rand(3, 4, generator=g) + 0.2)
    rep = post.repeat(5)
    assert rep.B == 15 and torch.equal(rep.log_cells[5:10], post.log_cells[1:2].expand(5, -1, -1))


def test_best_mode_ignores_a_worse_view():
    N, C = 3, 2
    g = torch.Generator().manual_seed(5)
    post = SearchPosterior(N, C, 1, mode="best")
    z1, z2 = torch.randn(1, N, C, generator=g), torch.randn(1, N, C, generator=g)
    post.update(z1, torch.full((1, N), 1.5))
    snap = post.log_cells.clone()
    post.update(z2, torch.full((1, N), 0.7))                          # coarser view: no effect
    assert torch.equal(post.log_cells, snap)
    post.update(z2, torch.full((1, N), 2.0))                          # finer view: replaces
    assert not torch.equal(post.log_cells, snap)


def test_unit_cap_bounds_accumulated_evidence():
    post = SearchPosterior(4, 2, 8, unit_cap=3.0)
    g = torch.Generator().manual_seed(9)
    for _ in range(20):
        post.update(torch.randn(8, 4, 2, generator=g) * 3 + 1, torch.rand(8, 4, generator=g) * 2)
    assert post.u_contrib.abs().max() <= 3.0 + 1e-6


def test_binary_footprint_required():
    import pytest as _p
    with _p.raises(ValueError):
        SearchPosterior(3, 2, 1, footprint=torch.full((3, 3), 0.5))


@pytest.mark.parametrize("mode", ["sum", "best"])
def test_units_may_differ_from_cells(mode):
    """3 coarse/fine evidence units feeding 4 hypothesis cells through a binary footprint (4, 3)."""
    N, V, C, p_null = 4, 3, 2, 0.35
    F = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=DT)
    looks = random_looks(V, C, 4, seed=21)
    post = SearchPosterior(N, C, 1, p_null=p_null, mode=mode, footprint=F, n_units=V, dtype=DT)
    for z, d in looks:
        post.update(z[None], d[None])
    ref = brute(p_null, N, C, looks, best=(mode == "best"), F=F, V=V)
    assert torch.allclose(post.joint_log_weights().exp()[0], ref, atol=1e-9)


def test_units_without_footprint_are_rejected():
    with pytest.raises(ValueError):
        SearchPosterior(4, 2, 1, n_units=3)
