"""Unit tests for cognitive_vision_lab.backend.attacks."""
import math

import pytest
import torch
import torch.nn as nn

from cognitive_vision_lab.backend.attacks import (
    ATTACKS,
    domain_bounds,
    distance_metrics,
    fgsm,
    pgd,
    run_attack,
)
from cognitive_vision_lab.config import STL10_MEAN, STL10_STD


class _LinearNet(nn.Module):
    """Trivial classifier where gradient attacks are well-defined."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, n_classes, bias=False)

    def forward(self, x):
        return self.fc(x.flatten(1))


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = _LinearNet()
    m.eval()
    return m


@pytest.fixture
def x():
    return torch.rand(1, 3, 8, 8) * 0.5 - 0.25  # roughly in normalized domain


@pytest.fixture
def y():
    return torch.tensor([3])


class TestDomainBounds:
    def test_clamps(self):
        lo, hi = domain_bounds(STL10_MEAN, STL10_STD, "cpu")
        assert lo.shape == (1, 3, 1, 1)
        assert hi.shape == (1, 3, 1, 1)
        assert (lo < hi).all()

    def test_pixel_range_maps_to_normalized(self):
        lo, hi = domain_bounds((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), "cpu")
        # pixel 0 -> (0-0.5)/0.25 = -2 ; pixel 1 -> (1-0.5)/0.25 = +2
        assert lo[0, 0].item() == pytest.approx(-2.0)
        assert hi[0, 0].item() == pytest.approx(2.0)


class TestFgsm:
    def test_returns_same_shape(self, model, x, y):
        adv = fgsm(model, x, y, eps=0.031)
        assert adv.shape == x.shape

    def test_perturbation_within_budget(self, model, x, y):
        eps = 0.05
        adv = fgsm(model, x, y, eps=eps)
        delta = (adv - x).abs().max().item()
        assert delta <= eps + 1e-4

    def test_fools_simple_net(self, model, x, y):
        # On a linear net the one-step attack moves strongly toward the target
        with torch.no_grad():
            before = model(x).argmax(1)
        adv = fgsm(model, x, y, eps=0.3)
        with torch.no_grad():
            after = model(adv).argmax(1)
        assert after.item() != before.item()


class TestPgd:
    def test_budget_respected(self, model, x, y):
        eps = 0.062
        adv = pgd(model, x, y, eps=eps, steps=20, random_start=False)
        assert (adv - x).abs().max().item() <= eps + 1e-4

    def test_more_steps_no_worse(self, model, x, y):
        eps = 0.031
        adv10 = pgd(model, x, y, eps=eps, steps=10, random_start=False)
        adv50 = pgd(model, x, y, eps=eps, steps=50, random_start=False)
        with torch.no_grad():
            loss10 = torch.nn.functional.cross_entropy(model(adv10), y)
            loss50 = torch.nn.functional.cross_entropy(model(adv50), y)
        assert loss50 <= loss10 + 1e-4

    def test_kl_variant_runs(self, model, x, y):
        adv = pgd(model, x, y, eps=0.031, steps=5, use_kl=True)
        assert adv.shape == x.shape

    def test_targeted_flips_direction(self, model, x, y):
        # targeted=True expects y to BE the target class (minimize CE toward it)
        target = torch.tensor([(y.item() + 1) % 10])
        with torch.no_grad():
            before = model(x).argmax(1).item()
        assert before != target.item()
        adv = pgd(model, x, target, eps=0.3, steps=60, targeted=True,
                  random_start=False)
        with torch.no_grad():
            assert model(adv).argmax(1).item() == target.item()


class TestRegistry:
    def test_all_attacks_importable(self, model, x, y):
        for name in ["fgsm", "pgd", "apgd", "cw", "deepfool", "square", "fab"]:
            assert name in ATTACKS
        adv = run_attack("pgd", model, x, y, eps=0.01, steps=3)
        assert adv.shape == x.shape

    def test_apgd_runs(self, model, x, y):
        adv = run_attack("apgd", model, x, y, eps=0.031, steps=8)
        assert (adv - x).abs().max().item() <= 0.031 + 1e-4


class TestAllAttacksExecute:
    """Smoke-test every registered attack on the tiny net with small step counts.

    Guards against latent crashes (e.g. autograd graphs detached from the
    differentiated tensor) that only surface when the attack actually runs.
    """
    @pytest.mark.parametrize("name", ["fgsm", "pgd", "apgd", "cw",
                                      "deepfool", "square", "fab"])
    def test_executes_without_error(self, model, x, y, name):
        # cw and deepfool are epsilon-free (they minimize/geometrically cross
        # the boundary) — no eps kwarg for those.
        eps_free = {"cw", "deepfool"}
        kwargs = {} if name in eps_free else {"eps": 0.01}
        if name == "pgd":
            kwargs.update(steps=5)
        elif name == "apgd":
            kwargs.update(steps=5)
        elif name == "cw":
            kwargs.update(steps=5)
        elif name == "deepfool":
            kwargs.update(steps=3)
        elif name == "square":
            kwargs.update(steps=20)
        elif name == "fab":
            kwargs.update(steps=5)
        adv = run_attack(name, model, x, y, **kwargs)
        assert adv.shape == x.shape
        assert torch.isfinite(adv).all()

    def test_all_registered_are_callable(self):
        for name in ATTACKS:
            assert callable(ATTACKS[name])


class TestDistanceMetrics:
    def test_identical_images(self, x):
        m = distance_metrics(x, x.clone())
        assert m["L1 (mean px)"] == pytest.approx(0.0, abs=1e-6)
        assert m["L∞ (max px)"] == pytest.approx(0.0, abs=1e-6)
        assert m["SSIM"] == pytest.approx(1.0, abs=1e-3)
        assert m["PSNR (dB)"] == math.inf

    def test_different_images(self, x):
        m = distance_metrics(x, torch.zeros_like(x))
        assert m["L1 (mean px)"] > 0.0
        assert m["SSIM"] < 1.0
        assert math.isfinite(m["PSNR (dB)"])
