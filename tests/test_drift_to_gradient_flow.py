"""
Agent B contract test — drift_to gradient flow.

drift_to must be differentiable w.r.t. z (Agent H will eventually put it
in a loss — Part 1.G's staged L_stab). Backprop through drift_to reaches
the live belief's z; a frozen (detached) history belief contributes no
gradient. Includes the contract's smoke experiment.
"""
import torch

from noesis_vision.beliefs.drift import drift_to
from noesis_vision.beliefs.factory import populate_belief
from noesis_vision.beliefs.vector_belief import DirichletParams, GazeState

B, DZ, C, N, DF = 2, 8, 10, 5, 8


def _belief(z, t=1):
    U = DirichletParams(evidence=torch.nn.functional.softplus(torch.randn(B, C)))
    E = torch.zeros(B, N, DF) if t == 0 else torch.randn(B, N, DF)
    A = GazeState(gaze_history=[torch.zeros(B, 2)], current_glimpse_idx=t)
    return populate_belief(z=z, U=U, E=E, A=A)


def _smoke_pair():
    z_live = torch.randn(B, DZ, requires_grad=True)
    z_prev = torch.randn(B, DZ)                    # frozen history entry
    return _belief(z_live), _belief(z_prev)


def test_backprop_through_drift_reaches_z_l2():
    live, prev = _smoke_pair()
    drift_to(live, prev, distance="l2").sum().backward()
    assert live.z.grad is not None
    assert torch.all(live.z.grad != 0)
    assert prev.z.grad is None                      # frozen history: no graph


def test_backprop_through_drift_reaches_z_cosine():
    live, prev = _smoke_pair()
    drift_to(live, prev, distance="cosine").sum().backward()
    assert live.z.grad is not None
    assert torch.all(live.z.grad != 0)


def test_both_sides_differentiable_when_both_carry_graphs():
    za = torch.randn(B, DZ, requires_grad=True)
    zb = torch.randn(B, DZ, requires_grad=True)
    drift_to(_belief(za), _belief(zb)).sum().backward()
    assert za.grad is not None and zb.grad is not None


def test_gradient_scale_follows_weights():
    z1 = torch.randn(B, DZ, requires_grad=True)
    z0 = torch.randn(B, DZ)
    live, prev = _belief(z1), _belief(z0)
    d = drift_to(live, prev, weights={"z": 0.5})
    d.sum().backward()
    unweighted = drift_to(_belief(z1.detach().requires_grad_(True)),
                          _belief(z0.detach()))
    # d(0.5*drift)/dz == 0.5 * d(drift)/dz: verify on a fresh graph.
    z_check = z1.detach().requires_grad_(True)
    drift_to(_belief(z_check), _belief(z0)).sum().backward()
    assert torch.allclose(live.z.grad, 0.5 * z_check.grad)


def test_smoke_experiment_compose_drift_backprop():
    """Contract smoke: compose two dummy beliefs, compute drift_to,
    backprop, confirm gradient at z."""
    live, prev = _smoke_pair()
    drift = live.drift_to(prev)
    assert drift.shape == (B,)
    drift.mean().backward()
    assert live.z.grad is not None and live.z.grad.abs().sum() > 0
