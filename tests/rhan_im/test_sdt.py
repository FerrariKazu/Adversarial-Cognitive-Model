import torch

from rhan_im.sdt import bayes_error_equal_prior, dprime_from_samples, gaussian_llr, zroc_slope


def test_llr_is_the_log_density_ratio():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(2000, generator=g, dtype=torch.float64) * 2
    d = torch.rand(2000, generator=g, dtype=torch.float64) * 3
    n1 = torch.distributions.Normal(d, 1.0)
    n0 = torch.distributions.Normal(torch.zeros_like(d), 1.0)
    assert torch.allclose(gaussian_llr(z, d), n1.log_prob(z) - n0.log_prob(z), atol=1e-10)


def test_bayes_error_matches_the_optimal_test_by_monte_carlo():
    g = torch.Generator().manual_seed(1)
    n = 600_000
    for D in (0.5, 1.0, 2.0, 3.0):
        x0 = torch.randn(n, generator=g)
        x1 = torch.randn(n, generator=g) + D
        err = 0.5 * (x0 > D / 2).float().mean() + 0.5 * (x1 <= D / 2).float().mean()
        assert abs(err.item() - bayes_error_equal_prior(torch.tensor(D)).item()) < 2.5e-3


def test_dprime_recovers_the_true_value():
    g = torch.Generator().manual_seed(2)
    pos = torch.randn(300_000, generator=g) + 1.7
    neg = torch.randn(300_000, generator=g)
    assert abs(dprime_from_samples(pos, neg) - 1.7) < 0.02


def test_zroc_slope_is_one_for_equal_variance_and_sigma_ratio_otherwise():
    g = torch.Generator().manual_seed(3)
    neg = torch.randn(200_000, generator=g)
    s_eq, _ = zroc_slope(torch.randn(200_000, generator=g) + 1.5, neg)
    s_uneq, _ = zroc_slope(2.0 * torch.randn(200_000, generator=g) + 1.5, neg)
    assert abs(s_eq - 1.0) < 0.06
    assert abs(s_uneq - 0.5) < 0.06
