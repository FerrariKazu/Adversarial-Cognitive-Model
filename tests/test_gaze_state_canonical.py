"""
test_gaze_state_canonical — Agent F.
================================================================================
The canonical GazeState's own contract: T-capped history that raises
LOUDLY at capacity instead of silently truncating; guarded
advance/reset; detached storage (the A_t record never carries autograd
graph); batch-(B,2) shape and [-1,+1] range validation; interop with
the belief factory boundary while the supersession integration is
pending (Agent J1 deletes the placeholder and re-imports THIS class).
"""
from __future__ import annotations

import pytest
import torch

from noesis_vision.beliefs.factory import populate_belief
from noesis_vision.gaze.gaze_state import GAZE_HISTORY_MAX_T, GazeState
from noesis_vision.uncertainty.evidential_head import DirichletParams

B = 3


def test_record_appends_and_current_reads_last():
    gs = GazeState()
    a = torch.zeros(B, 2)
    b = torch.full((B, 2), 0.5)
    out = gs.record(a).record(b)
    assert out is gs                        # chaining
    assert len(gs.gaze_history) == 2
    assert torch.equal(gs.current, b)


def test_history_capped_at_T_and_raises_at_capacity():
    gs = GazeState()
    for _ in range(GAZE_HISTORY_MAX_T):
        gs.record(torch.zeros(B, 2))
    assert len(gs.gaze_history) == GAZE_HISTORY_MAX_T == 4
    with pytest.raises(ValueError, match="capacity"):
        gs.record(torch.zeros(B, 2))         # LOUD, not silent truncation


def test_advance_and_reset_guards():
    gs = GazeState()
    assert gs.current_glimpse_idx == 0
    gs.advance()
    assert gs.current_glimpse_idx == 1
    gs.reset()
    assert gs.current_glimpse_idx == 0 and gs.gaze_history == []
    # advance() past T-1 raises; idx beyond T raises at construction.
    for _ in range(GAZE_HISTORY_MAX_T - 1):
        gs.advance()
    with pytest.raises(ValueError, match="outside"):
        gs.advance()
    with pytest.raises(ValueError):
        GazeState(current_glimpse_idx=GAZE_HISTORY_MAX_T)


def test_record_is_detached_and_cloned_storage():
    gs = GazeState()
    live = torch.zeros(B, 2, requires_grad=True)
    gs.record(live + 0.25)                  # graph-carrying input
    stored = gs.gaze_history[0]
    assert stored.requires_grad is False and stored.grad_fn is None
    stored[0, 0] = 99.0                     # in-place write through storage
    assert live[0, 0].item() == 0.0         # ...never leaks into the live value


def test_validation_shapes_ranges_and_current_at_t0():
    with pytest.raises(ValueError):
        GazeState(gaze_history=[torch.zeros(B, 3)])       # wrong last dim
    with pytest.raises(ValueError):
        GazeState(gaze_history=[torch.full((B, 2), 1.5)])  # out of range
    with pytest.raises(ValueError):
        GazeState(current_glimpse_idx=-1)
    with pytest.raises(ValueError):
        GazeState(gaze_history=[torch.zeros(2)])          # not (B, 2)
    gs = GazeState()
    with pytest.raises(IndexError):
        _ = gs.current                     # no default location invented


def test_interops_with_belief_factory_before_supersession():
    """The canonical class flows through the existing belief boundary
    (VectorBeliefState isinstance-checks a GazeState) — the placeholder
    and the canonical class are shape-compatible BY DESIGN until Agent
    J1 performs the flagged deletion/re-import."""
    gs = GazeState(gaze_history=[torch.zeros(B, 2)], current_glimpse_idx=1)
    U = DirichletParams(evidence=torch.rand(B, 8) + 0.5)
    belief = populate_belief(z=torch.randn(B, 384), U=U,
                             E=torch.randn(B, 16, 384), A=gs)
    assert belief.gaze_history is gs.gaze_history
    assert belief.current_glimpse_idx == 1
