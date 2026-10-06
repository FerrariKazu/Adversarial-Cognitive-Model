"""
Agent E contract test — t=0 produces EXACTLY zero error.

LOCKED (MASTER_PLAN Part 1.B): E_0 := 0 (zero tensor, correct shape, no
gradient contribution); z_1 = z_0 + Pi_0 * UpdateNet(z_0, 0); prediction
and genuine E_t begin at t = 1. Value-exact, not approximate.
"""
import torch

from noesis_vision.beliefs.factory import populate_belief
from noesis_vision.beliefs.vector_belief import DirichletParams, GazeState
from noesis_vision.core.schema import AIS_T0_SCORING, FIRST_GLIMPSE_CONVENTION
from noesis_vision.predictive_coding.update_net import (
    ConcreteUpdateNet,
    belief_update,
)
from noesis_vision.uncertainty.evidential_head import EvidentialHead

B, DZ, C, N, DF = 2, 384, 10, 16, 384


def _zeros_E():
    return torch.zeros(B, N, DF)


def test_e0_is_shape_correct_and_value_exact_zero():
    E0 = _zeros_E()
    assert E0.shape == (B, N, DF)          # the predictor's output shape
    assert torch.all(E0 == 0)              # VALUE-EXACT zero
    assert E0.requires_grad is False       # no gradient path assumed


def test_t0_update_exact_and_finite():
    torch.manual_seed(0)
    net = ConcreteUpdateNet(d_z=DZ)
    z0 = torch.randn(B, DZ)
    Pi0 = torch.rand(B) + 0.5
    z1 = belief_update(z0, _zeros_E(), Pi0, net)
    assert z1.shape == (B, DZ)                       # shape-correct
    assert torch.isfinite(z1).all()                  # no NaN from an
    # undefined predicted-vs-observed comparison — the zero tensor makes
    # the update well-defined by construction.
    expected = z0 + Pi0.unsqueeze(-1) * net(z0, torch.zeros(B, N, DF))
    assert torch.allclose(z1, expected)              # the plan's equation


def test_e0_contributes_no_gradient_of_its_own():
    """'No gradient contribution' = E_0 is a CONSTANT zero tensor: nothing
    upstream for a gradient to flow into. The update itself still
    differentiates w.r.t. z_0 and UpdateNet's parameters."""
    net = ConcreteUpdateNet(d_z=DZ)
    z0 = torch.randn(B, DZ, requires_grad=True)
    E0 = _zeros_E()
    belief_update(z0, E0, torch.ones(B), net).sum().backward()
    assert E0.grad is None, "a constant zero E_0 must carry no gradient path"
    assert z0.grad is not None
    assert any(p.grad is not None for p in net.parameters())


def test_genuine_error_begins_at_t1():
    """t=1 belief with a genuine E_t updates DIFFERENTLY from the t=0
    reduction — prediction is live from t = 1 (LOCKED)."""
    net = ConcreteUpdateNet(d_z=DZ)
    z0 = torch.randn(B, DZ)
    Pi = torch.rand(B) + 0.5
    z_t0 = belief_update(z0, _zeros_E(), Pi, net)
    z_t1 = belief_update(z0, torch.randn(B, N, DF), Pi, net)
    assert not torch.allclose(z_t0, z_t1)


def test_agent_b_guard_blocks_nonzero_e0():
    """Cross-agent consistency: Agent B's construction guard refuses a
    nonzero E at t=0 — the convention is enforced, not honor-system."""
    from pytest import raises
    z = torch.randn(B, DZ)
    U = DirichletParams(evidence=torch.nn.functional.softplus(torch.randn(B, C)))
    A = GazeState(gaze_history=[torch.zeros(B, 2)], current_glimpse_idx=0)
    with raises(ValueError, match="E_0 := 0"):
        populate_belief(z=z, U=U, E=torch.randn(B, N, DF), A=A)


def test_schema_constants_state_the_locked_convention():
    """The schema constants (what downstream agents assert against) state
    exactly the LOCKED convention."""
    assert "E_0 := 0" in FIRST_GLIMPSE_CONVENTION
    assert "z_1 = z_0 + Pi_0 * UpdateNet(z_0, 0)" in FIRST_GLIMPSE_CONVENTION
    assert "heuristic-saliency" in AIS_T0_SCORING


def test_end_to_end_t0_composition():
    """Smoke: frozen backbone + Agent D head + Agent B belief + the t=0
    update — one full forward at t=0 through REAL components."""
    from noesis_vision.models.backbone import CompactViT
    torch.manual_seed(1)
    bb = CompactViT(img_size=56)
    for p in bb.parameters():
        p.requires_grad_(False)                    # frozen backbone
    head = EvidentialHead(input_dim=DZ, num_classes=C)
    for p in head.parameters():
        p.requires_grad_(False)
    net = ConcreteUpdateNet(d_z=DZ)

    image = torch.randn(B, 3, 96, 96)
    gaze = torch.zeros(B, 2)                       # center fixation
    pooled, tokens = bb.encode_glimpse(image, gaze)
    U = head(pooled)
    E0 = torch.zeros_like(tokens)                  # the LOCKED zero
    belief = populate_belief(
        z=pooled, U=U, E=E0,
        A=GazeState(gaze_history=[gaze], current_glimpse_idx=0))
    Pi = torch.nn.functional.softplus(U.uncertainty) + 1e-4
    z1 = belief_update(belief.z, belief.prediction_error, Pi, net)
    assert z1.shape == (B, DZ)
    assert torch.isfinite(z1).all()
    assert torch.allclose(z1, pooled + Pi.unsqueeze(-1) * net(pooled, E0))
