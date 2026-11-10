import torch

from rhan_im.belief import SearchPosterior
from rhan_im.geometry import eccentricity, grid_centers, tiling_order, visibility_window
from rhan_im.policy import (
    evoi_fixation,
    greedy_mode_fixation,
    koopman_fixation,
    koopman_gain,
    mc_expected_max_posterior,
    mc_posterior_outcomes,
    random_fixation,
)


def line_world(n=5, d0=2.0):
    cells = grid_centers(1, n)
    return visibility_window(eccentricity(cells, cells), d0, 0.5)     # each fixation sees only its own cell


def test_gain_is_nonnegative_and_zero_when_nothing_is_visible():
    post = SearchPosterior(5, 3, 4)
    assert (koopman_gain(post, line_world()) >= -1e-7).all()
    assert torch.allclose(koopman_gain(post, torch.zeros(5, 5)), torch.zeros(4, 5), atol=1e-7)


def test_best_mode_revisit_at_same_precision_has_zero_gain():
    post = SearchPosterior(5, 3, 1, mode="best")
    vis = line_world()
    post.update(torch.randn(1, 5, 3), vis[2][None])
    assert koopman_gain(post, vis)[0, 2].abs() < 1e-7
    assert koopman_gain(post, vis)[0, 0] > 0


def test_both_look_at_the_cell_that_holds_the_probability_mass():
    lp = torch.full((1, 5, 3), -8.0)
    lp[0, 2] = 0.0
    post = SearchPosterior(5, 3, 1, log_prior_cells=lp, p_null=0.2)
    vis = line_world()
    g = torch.Generator().manual_seed(0)
    assert koopman_fixation(post, vis).item() == 2
    assert evoi_fixation(post, vis, n_mc=256, generator=g).item() == 2


def test_posterior_predictive_is_a_martingale():
    """E over hypotheses ~ posterior of the next posterior = the current posterior."""
    for mode in ("sum", "best"):
        post = SearchPosterior(4, 3, 1, mode=mode)
        g = torch.Generator().manual_seed(11)
        vis = visibility_window(eccentricity(grid_centers(2, 2), grid_centers(2, 2)), 1.6, 1.1)
        post.update(torch.randn(1, 4, 3, generator=g) + 0.5, vis[1][None])
        pr = mc_posterior_outcomes(post, vis, n_mc=40_000, generator=g)     # (1, K, M, 1+C)
        mean_next = pr.mean(2)[0]                                           # (K, 1+C)
        cur = post.outcome_probs()[0]
        assert (mean_next - cur[None]).abs().max() < 0.012, mode


def test_expected_max_posterior_does_not_fall_below_the_current_max():
    post = SearchPosterior(4, 3, 3)
    vis = visibility_window(eccentricity(grid_centers(2, 2), grid_centers(2, 2)), 1.6, 1.1)
    g = torch.Generator().manual_seed(12)
    em = mc_expected_max_posterior(post, vis, n_mc=4000, generator=g)
    assert (em.min(1).values >= post.decision()[1] - 0.01).all()


def test_greedy_and_random_baselines():
    post = SearchPosterior(5, 3, 2)
    post.log_cells[0, 3] += 3.0
    vis = line_world()
    assert greedy_mode_fixation(post, vis)[0].item() == 3
    g = torch.Generator().manual_seed(0)
    visited = torch.zeros(6, 5, dtype=torch.bool)
    picks = []
    for _ in range(5):
        k = random_fixation(6, 5, g, exclude=visited)
        visited[torch.arange(6), k] = True
        picks.append(k)
    seq = torch.stack(picks, 1)
    assert all(len(set(row.tolist())) == 5 for row in seq)


def test_tiling_order_covers_everything_once():
    o = tiling_order(8, 8, 3)
    assert sorted(o.tolist()) == list(range(64)) and o[:3].tolist() == [9, 12, 15]
