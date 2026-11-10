import pytest
import torch

from rhan_im.agent import run_search
from rhan_im.geometry import eccentricity, grid_centers, tiling_order, visibility_gaussian
from rhan_im.sensors import ToySDTWorld
from rhan_im.stopping import stopped_stats, stopped_stats_selective


def toy(d0=2.0, sigma=1.2, gh=4, gw=4, C=3, **kw):
    cells = grid_centers(gh, gw)
    vis = visibility_gaussian(eccentricity(cells, cells), d0, sigma)
    return ToySDTWorld(vis, C, **kw), vis


def test_zero_visibility_gives_the_prior_mode_accuracy():
    world, vis = toy()
    vis0 = torch.zeros_like(vis)
    w0 = ToySDTWorld(vis0, 3, p_null=0.4)
    scene = w0.sample(6000, torch.Generator().manual_seed(0))
    out = run_search(scene, vis0, "random", 3, 0.4, generator=torch.Generator().manual_seed(1))
    assert abs(out["correct"][:, -1].float().mean().item() - 0.4) < 0.03        # 'absent' is the prior mode


def test_strong_evidence_gives_near_perfect_decisions():
    world, vis = toy(d0=6.0, sigma=10.0)
    scene = world.sample(2000, torch.Generator().manual_seed(2))
    out = run_search(scene, vis, "map", 1, 0.4)
    assert out["correct"][:, -1].float().mean().item() > 0.97


def test_trace_shapes():
    world, vis = toy()
    scene = world.sample(64, torch.Generator().manual_seed(3))
    out = run_search(scene, vis, "koopman", 5, 0.4)
    assert out["risk"].shape == (64, 6) and out["fix"].shape == (64, 5) and out["correct"].shape == (64, 6)


def test_offline_stopping_equals_a_run_that_really_stops():
    world, vis = toy(d0=1.6)
    lam, K = 0.05, 6
    full = run_search(world.sample(500, torch.Generator().manual_seed(5)), vis, "map", K, 0.4, generator=torch.Generator().manual_seed(9))
    stop = run_search(world.sample(500, torch.Generator().manual_seed(5)), vis, "map", K, 0.4, stop_lambda=lam, generator=torch.Generator().manual_seed(9))
    t, corr = stopped_stats(full["risk"], full["correct"], lam)
    assert torch.equal(stop["n_fix"], t)
    assert torch.equal(stop["correct"][:, -1], corr)


@pytest.mark.parametrize("lam,k_min", [(0.05, 1), (0.05, 2), (0.05, 3), (0.3, 2), (0.999, 1), (0.999, 3), (0.0, 2)])
def test_offline_equals_online_for_any_threshold_including_one_above_the_prior_risk_and_for_any_k_min(lam, k_min):
    """The offline rule (stopped_stats on a full-budget trace) and a run that really stops (run_search) are the same rule, also when the
    threshold is above the prior risk (0.6 here: offline used to stop at column 0 with 0 looks while online always took one) and with a floor."""
    world, vis = toy(d0=1.6)
    K = 6
    full = run_search(world.sample(300, torch.Generator().manual_seed(5)), vis, "map", K, 0.4, generator=torch.Generator().manual_seed(9))
    stop = run_search(world.sample(300, torch.Generator().manual_seed(5)), vis, "map", K, 0.4, stop_lambda=lam, k_min=k_min, generator=torch.Generator().manual_seed(9))
    t, corr = stopped_stats(full["risk"], full["correct"], lam, k_min=k_min)
    t_sel, decided, _ = stopped_stats_selective(full["risk"], full["correct"], lam, k_min=k_min)
    assert torch.equal(stop["n_fix"], t) and torch.equal(stop["n_fix"], t_sel)
    assert torch.equal(stop["correct"][:, -1], corr)
    assert int(t.min()) >= min(k_min, K)                                                       # never before the floor
    if lam >= 0.6:
        assert int(t.min()) >= 1 and (t == max(1, k_min)).float().mean() > 0.99               # the prior risk is already below lam: stops at the floor


def test_k_min_must_be_at_least_one():
    world, vis = toy()
    with pytest.raises(ValueError):
        run_search(world.sample(8, torch.Generator().manual_seed(1)), vis, "map", 3, 0.4, stop_lambda=0.1, k_min=0)


def test_tiling_visits_the_whole_canvas():
    world, vis = toy(gh=6, gw=6)
    scene = world.sample(4, torch.Generator().manual_seed(6))
    out = run_search(scene, vis, "tiling", 9, 0.4, order=tiling_order(6, 6, 2))
    assert len(set(out["fix"][0].tolist())) == 9
