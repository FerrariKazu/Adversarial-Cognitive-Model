"""
Agent E contract test — UpdateNet gradient flow and the exact update
equation z_{t+1} = z_t + Pi_t * UpdateNet(z_t, E_t).
"""
import inspect

import pytest
import torch

from noesis_vision.predictive_coding.interfaces import UpdateNet
from noesis_vision.predictive_coding.update_net import (
    DELTA_BOUND,
    ConcreteUpdateNet,
    belief_update,
)

B, DZ, N, DF = 2, 384, 16, 384


def _net():
    torch.manual_seed(0)
    return ConcreteUpdateNet(d_z=DZ)


def test_implements_agent_zero_abc():
    assert isinstance(_net(), UpdateNet)


def test_exact_equation_value_equality():
    net = _net()
    z = torch.randn(B, DZ)
    E = torch.randn(B, N, DF)
    Pi = torch.rand(B) + 0.5
    z1 = belief_update(z, E, Pi, net)
    expected = z + Pi.unsqueeze(-1) * net(z, E)
    assert torch.allclose(z1, expected)
    assert z1.shape == (B, DZ)


def test_bounded_delta():
    net = _net()
    z = torch.randn(B, DZ)
    for scale in (1.0, 1e3, 1e6):
        delta = net(z, scale * torch.randn(B, N, DF))
        assert bool((delta.abs() <= DELTA_BOUND + 1e-6).all()), (
            f"delta escaped the bound at input scale {scale} — a single "
            f"glimpse's update must stay a correction, not a rewrite")


def test_gradient_reaches_z_E_Pi_and_params():
    net = _net()
    z = torch.randn(B, DZ, requires_grad=True)
    E = torch.randn(B, N, DF, requires_grad=True)
    Pi = (torch.rand(B) + 0.5).requires_grad_(True)
    z1 = belief_update(z, E, Pi, net)
    z1.sum().backward()
    assert z.grad is not None and torch.any(z.grad != 0)
    assert E.grad is not None and torch.any(E.grad != 0), (
        "E_t must never be detached before the update (the single most "
        "repeated Gen-0 failure mode)")
    assert Pi.grad is not None
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no gradient reached UpdateNet param {name}"


def test_t0_zero_error_path_finite_and_exact():
    """At t=0 (E_0 := 0, LOCKED) the update reduces to
    z_1 = z_0 + Pi_0 * UpdateNet(z_0, 0) — the plan's own equation; it must
    be finite (no NaN from an undefined comparison) and exact."""
    net = _net()
    z0 = torch.randn(B, DZ)
    E0 = torch.zeros(B, N, DF)          # value-exact zero, no gradient path
    Pi0 = torch.rand(B) + 0.5
    z1 = belief_update(z0, E0, Pi0, net)
    assert torch.isfinite(z1).all()
    assert torch.allclose(z1, z0 + Pi0.unsqueeze(-1) * net(z0, E0))


def test_error_sensitivity_changes_output():
    """A nonzero E_t changes z_{t+1} vs the t=0 path — the update actually
    consumes the error signal (else E_t would be decorative)."""
    net = _net()
    z = torch.randn(B, DZ)
    Pi = torch.rand(B) + 0.5
    z_t0 = belief_update(z, torch.zeros(B, N, DF), Pi, net)
    z_t1 = belief_update(z, torch.randn(B, N, DF), Pi, net)
    assert not torch.allclose(z_t0, z_t1)


def test_pooling_and_dimension_mismatch_refused():
    net = _net()
    with pytest.raises(ValueError, match="must match"):
        net(torch.randn(B, DZ), torch.randn(B, N, DF + 1))
    with pytest.raises(ValueError, match=r"z must be \(B,"):
        net(torch.randn(B, DZ + 1), torch.zeros(B, N, DF))


def test_no_detach_in_forward():
    src = inspect.getsource(ConcreteUpdateNet)
    assert ".detach()" not in src, (
        "UpdateNet must never detach — hard non-detached assertion")


def test_own_optimizer_group_registration():
    """Gradient isolation (Agent A's registry): UpdateNet registers under
    its OWN group — never lumped with the backbone (Stage-2 lesson)."""
    from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry
    net = _net()
    backbone_param = torch.nn.Parameter(torch.zeros(4, 4))
    reg = OptimizerGroupRegistry()
    reg.register_backbone([backbone_param])
    reg.register("update_net", list(net.parameters()), lr_multiplier=6.67)
    assert "update_net" in reg.group_names
    group_ids = {id(p) for p in reg.group("update_net")["params"]}
    assert group_ids == {id(p) for p in net.parameters()}
    assert id(backbone_param) not in group_ids
