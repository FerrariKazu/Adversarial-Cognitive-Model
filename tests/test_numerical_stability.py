"""
Agent D contract test — numerical stability.

Clamped and TESTED, not hoped for (Agent D contract): extreme-magnitude
inputs produce no NaN/Inf in evidence, alpha, uncertainty, or entropy.
NaN input is the one honest exception: it propagates (garbage-in is never
silently repaired into a plausible-looking number).
"""
import pytest
import torch

from noesis_vision.uncertainty.evidential_head import (
    EVIDENCE_CLAMP_MAX,
    EVIDENCE_CLAMP_MIN,
    EvidentialHead,
)

B, D, C = 4, 384, 10


def _assert_all_finite(U):
    for name, t in (("evidence", U.evidence), ("alpha", U.alpha),
                    ("uncertainty", U.uncertainty), ("entropy", U.entropy())):
        assert torch.isfinite(t).all(), f"{name} went non-finite"


@pytest.mark.parametrize("scale", [1e3, 1e6, 1e8])
def test_extreme_magnitude_inputs_stay_finite(scale):
    torch.manual_seed(3)
    h = EvidentialHead(input_dim=D, num_classes=C)
    with torch.no_grad():
        for sign in (1.0, -1.0):
            U = h(sign * scale * torch.randn(B, D))
            _assert_all_finite(U)


def test_infinite_inputs_stay_finite():
    h = EvidentialHead(input_dim=D, num_classes=C)
    with torch.no_grad():
        U = h(torch.full((B, D), float("inf")))
        _assert_all_finite(U)                     # +inf -> clamped to the cap
        U = h(torch.full((B, D), float("-inf")))
        _assert_all_finite(U)                     # -inf -> softplus 0 -> floor


def test_clamp_bounds_enforced():
    h = EvidentialHead(input_dim=D, num_classes=C)
    with torch.no_grad():
        U_hi = h(torch.full((B, D), 1e30))
        assert bool((U_hi.evidence <= EVIDENCE_CLAMP_MAX + 1e-6).all())
        U_lo = h(torch.full((B, D), -1e30))
        assert bool((U_lo.evidence >= EVIDENCE_CLAMP_MIN - 1e-12).all())


def test_nan_propagates_honestly():
    """NaN is NOT silently repaired: it propagates so upstream bugs surface
    instead of hiding behind a plausible-looking scalar."""
    h = EvidentialHead(input_dim=D, num_classes=C)
    with torch.no_grad():
        U = h(torch.full((B, D), float("nan")))
        assert bool(torch.isnan(U.evidence).any()), (
            "NaN input must propagate — a silent repair would be a false "
            "guarantee")


def test_entropy_finite_at_clamp_extremes():
    """digamma at alpha ~ 1 + 1e-6 and at alpha ~ 1e4+1: finite."""
    h = EvidentialHead(input_dim=D, num_classes=C)
    with torch.no_grad():
        _assert_all_finite(h(torch.full((B, D), -1e30)))   # near the floor
        _assert_all_finite(h(torch.full((B, D), 1e30)))    # at the cap
