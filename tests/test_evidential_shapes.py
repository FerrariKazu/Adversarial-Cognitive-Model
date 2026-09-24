"""
Agent D contract test — evidential shapes (+ smoke: all alpha > 1).
"""
import pytest
import torch

from noesis_vision.uncertainty.evidential_head import (
    DirichletParams,
    EvidentialHead,
)

B, D, C = 4, 384, 10


def _head():
    torch.manual_seed(0)
    return EvidentialHead(input_dim=D, num_classes=C)


def test_forward_shapes_2d():
    h = _head()
    U = h(torch.randn(B, D))
    assert U.evidence.shape == (B, C)
    assert U.alpha.shape == (B, C)
    assert U.uncertainty.shape == (B,)
    assert U.uncertainty_scalar().shape == (B,)
    assert U.entropy().shape == (B,)


def test_forward_3d_tokens_pool_mean():
    """Token features (B, N, D) pool by mean inside the head — the AIS-v2
    scoring path feeds predicted token features directly (Part 1.E)."""
    h = _head()
    tokens = torch.randn(B, 16, D)
    U_tokens = h(tokens)
    U_pooled = h(tokens.mean(dim=1))
    assert U_tokens.evidence.shape == (B, C)
    assert torch.allclose(U_tokens.evidence, U_pooled.evidence)


def test_alpha_equals_evidence_plus_one():
    h = _head()
    U = h(torch.randn(B, D))
    assert torch.allclose(U.alpha, U.evidence + 1.0)


def test_smoke_all_alpha_strictly_above_one():
    """Contract smoke: forward random features -> ALL alpha > 1 (strictly —
    the two-sided clamp floors evidence positive)."""
    h = _head()
    U = h(torch.randn(B, D))
    assert bool((U.alpha > 1.0).all())


def test_uncertainty_scalar_formula():
    """C / sum(alpha_t), Part 1.A verbatim."""
    e = torch.rand(B, C) + 0.1
    U = DirichletParams(evidence=e)
    assert torch.allclose(U.uncertainty, C / (e + 1.0).sum(dim=-1))


def test_head_refuses_bad_dims():
    h = _head()
    with pytest.raises(ValueError, match="features must be"):
        h(torch.randn(B, D + 1))
    with pytest.raises(ValueError, match="input_dim/num_classes"):
        EvidentialHead(input_dim=0, num_classes=C)


def test_canonical_dirichletparams_validates():
    with pytest.raises(ValueError, match="non-negative"):
        DirichletParams(evidence=-torch.ones(B, C))
    with pytest.raises(ValueError, match=r"\(B, C\)"):
        DirichletParams(evidence=torch.ones(B, C, 1))
