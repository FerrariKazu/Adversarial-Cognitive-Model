import torch

from rhan_im.sensors import ToySDTWorld


def _measure(seed, noise_df):
    vis = torch.full((4, 4), 2.0)
    w = ToySDTWorld(vis, n_classes=3, p_null=0.4, noise_df=noise_df)
    g = torch.Generator().manual_seed(seed)
    sc = w.sample(64, g)
    return sc.label, sc.measure(torch.zeros(64, dtype=torch.long))


def test_student_t_world_is_reproducible_given_the_generator():
    # The Gamma sampler behind the heavy-tailed noise has no generator argument; it must be seeded FROM the
    # generator, otherwise two identically seeded runs differ (this was a real reproducibility bug).
    for df in (None, 4.0):
        l1, z1 = _measure(3, df)
        torch.rand(5)                                   # disturb the global RNG between the two runs
        l2, z2 = _measure(3, df)
        assert torch.equal(l1, l2) and torch.equal(z1, z2)
    _, z3 = _measure(4, 4.0)
    assert not torch.equal(z1, z3)


def test_student_t_noise_has_unit_variance_and_heavy_tails():
    vis = torch.zeros(1, 4)
    w = ToySDTWorld(vis, n_classes=3, p_null=0.999, noise_df=4.0)
    g = torch.Generator().manual_seed(0)
    z = w.sample(20000, g).measure(torch.zeros(20000, dtype=torch.long)).flatten()
    assert abs(z.var().item() - 1.0) < 0.05
    kurt = ((z - z.mean()) ** 4).mean() / z.var() ** 2
    assert kurt.item() > 4.0                            # Gaussian is 3
