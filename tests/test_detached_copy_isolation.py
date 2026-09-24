"""
Agent B contract test — detached_copy isolation.

The copy exists so diagnostic code can never accidentally detach the
TRAINING-path object. Mutating or using the copy must never affect the
live object's values or its autograd graph.
"""
import pytest
import torch

from noesis_vision.beliefs.factory import populate_belief
from noesis_vision.beliefs.vector_belief import DirichletParams, GazeState

B, DZ, C, N, DF = 2, 8, 10, 5, 8


def _live_belief():
    z = torch.randn(B, DZ, requires_grad=True)
    U = DirichletParams(evidence=torch.nn.functional.softplus(torch.randn(B, C)))
    E = torch.randn(B, N, DF)
    A = GazeState(gaze_history=[torch.zeros(B, 2)], current_glimpse_idx=1)
    return populate_belief(z=z, U=U, E=E, A=A)


def test_copy_is_detached():
    live = _live_belief()
    copy = live.detached_copy()
    assert not copy.z.requires_grad
    assert not copy.prediction_error.requires_grad
    assert not copy.evidence.requires_grad


def test_mutating_copy_never_touches_live_values():
    live = _live_belief()
    copy = live.detached_copy()
    z_before = live.z.detach().clone()
    copy.z.data.fill_(999.0)                        # in-place mutation of the copy
    assert torch.equal(live.z.detach(), z_before)   # live unchanged
    assert torch.all(copy.z == 999.0)


def test_copy_has_no_path_into_live_graph():
    live = _live_belief()
    copy = live.detached_copy()
    # Any autograd attempt through the copy is inert: the copy carries no
    # grad_fn, and backward on the LIVE belief is unaffected by copy use.
    d_copy = copy.z.sum()                            # a leaf without grad
    assert not d_copy.requires_grad
    live.z.backward(torch.ones_like(live.z))         # live graph intact
    assert live.z.grad is not None
    # The copy never appears in the live graph: its grad is a fresh leaf.
    assert copy.z.is_leaf and copy.z.grad is None


def test_copy_containers_are_independent():
    live = _live_belief()
    copy = live.detached_copy()
    copy.gaze_history.append(torch.ones(B, 2))       # mutate the copy's list
    assert len(live.gaze_history) == 1               # live's list untouched
    assert copy.current_glimpse_idx == live.current_glimpse_idx


def test_live_object_still_trains_after_copy_use():
    """The scenario the method exists for: a diagnostic computes things on
    the copy; the training path keeps flowing."""
    live = _live_belief()
    other = _live_belief()
    _ = live.detached_copy().as_tensor().mean().item()   # diagnostic read
    loss = live.drift_to(other).mean()
    loss.backward()                                       # training path intact
    assert live.z.grad is not None
    assert torch.all(live.z.grad != 0)
