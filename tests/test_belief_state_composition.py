"""
Agent B contract test — belief-state composition.

S=None (the core build, Part 1.D) is handled by EVERY method without
crashing: factory, as_tensor, drift_to, and every Part 1.A member. Also
locks the LOCKED E_0 := 0 boundary condition structurally (MASTER_PLAN
Part 1.B) and the Part 1.A shape validation.
"""
import pytest
import torch

from noesis_vision.beliefs.drift import drift_to
from noesis_vision.beliefs.factory import populate_belief
from noesis_vision.beliefs.vector_belief import (
    DirichletParams,
    GazeState,
    VectorBeliefState,
)

B, DZ, C, N, DF = 2, 8, 10, 5, 8


def _belief(t=0, requires_grad=False):
    z = torch.randn(B, DZ, requires_grad=requires_grad)
    U = DirichletParams(evidence=torch.nn.functional.softplus(torch.randn(B, C)))
    E = torch.zeros(B, N, DF) if t == 0 else torch.randn(B, N, DF)
    A = GazeState(gaze_history=[torch.zeros(B, 2)], current_glimpse_idx=t)
    return populate_belief(z=z, U=U, E=E, A=A)


def test_factory_composes_all_members_s_none():
    b = _belief(t=0)
    assert b.s is None                                    # the core build
    assert b.z.shape == (B, DZ)
    assert b.evidence.shape == (B, C)
    assert torch.allclose(b.alpha, b.evidence + 1.0)       # Part 1.A
    expected_u = C / (b.evidence + 1.0).sum(dim=-1)
    assert torch.allclose(b.uncertainty, expected_u)       # C / sum(alpha)
    assert b.prediction_error.shape == (B, N, DF)
    assert torch.all(b.prediction_error == 0)              # E_0 := 0 (LOCKED)
    assert b.gaze_history[0].shape == (B, 2)
    assert b.current_glimpse_idx == 0
    # as_tensor with the explicit S=None branch: the readout IS z.
    assert torch.equal(b.as_tensor(), b.z)
    # drift_to on two S=None beliefs: runs without crashing.
    d = b.drift_to(_belief(t=0))
    assert d.shape == (B,)


def test_t1_belief_carries_genuine_error():
    b = _belief(t=1)
    assert not torch.all(b.prediction_error == 0)          # E_t genuine from t=1
    assert b.current_glimpse_idx == 1


def test_e0_locked_convention_enforced():
    """t=0 with a nonzero E must raise — agents do not choose their own
    first-glimpse convention (MASTER_PLAN Part 1.B, LOCKED)."""
    z = torch.randn(B, DZ)
    U = DirichletParams(evidence=torch.nn.functional.softplus(torch.randn(B, C)))
    A = GazeState(gaze_history=[torch.zeros(B, 2)], current_glimpse_idx=0)
    with pytest.raises(ValueError, match="E_0 := 0"):
        populate_belief(z=z, U=U, E=torch.randn(B, N, DF), A=A)  # nonzero at t=0
    # Direct construction (bypassing the factory) is guarded identically.
    with pytest.raises(ValueError, match="E_0 := 0"):
        VectorBeliefState(z=z, U=U, E=torch.randn(B, N, DF), A=A)


def test_part_1a_shape_validation():
    z = torch.randn(B, DZ)
    U = DirichletParams(evidence=torch.nn.functional.softplus(torch.randn(B, C)))
    A = GazeState(gaze_history=[torch.zeros(B, 2)], current_glimpse_idx=1)
    with pytest.raises(ValueError, match=r"z must be \(B, D_z\)"):
        populate_belief(z=torch.randn(B, DZ, 1), U=U, E=torch.randn(B, N, DF), A=A)
    with pytest.raises(ValueError, match="non-negative"):
        populate_belief(z=z, U=DirichletParams(evidence=-torch.ones(B, C)),
                        E=torch.randn(B, N, DF), A=A)
    with pytest.raises(ValueError, match=r"\(B, N, D_feat\)"):
        populate_belief(z=z, U=U, E=torch.randn(B + 1, N, DF), A=A)  # batch mismatch
    # (Since the canonical-GazeState supersession, a wrong-LAST-dim entry
    # is rejected eagerly AT GazeState construction (its own guard, tested
    # in test_gaze_state_canonical); the belief-layer batch check below
    # uses a valid-shape entry with the WRONG batch.)
    bad_A = GazeState(gaze_history=[torch.zeros(B + 1, 2)],
                      current_glimpse_idx=1)
    with pytest.raises(ValueError, match=r"\(B, 2\)"):
        populate_belief(z=z, U=U, E=torch.randn(B, N, DF), A=bad_A)
    with pytest.raises(ValueError, match=r"\(B, K, D_s\) or None"):
        populate_belief(z=z, U=U, E=torch.randn(B, N, DF), A=A,
                        S=torch.randn(B, K_DUMMY := 2, 4, 4) if False else torch.randn(B, 2, 4, 4))


def test_drift_weights_and_rejections():
    b1, b0 = _belief(t=1), _belief(t=1)
    d_default = drift_to(b1, b0)
    d_weighted = drift_to(b1, b0, weights={"z": 2.0})
    assert torch.allclose(d_weighted, 2.0 * d_default)
    with pytest.raises(ValueError, match="unknown drift components"):
        drift_to(b1, b0, weights={"structure": 1.0})
    with pytest.raises(ValueError, match="distance must be one of"):
        drift_to(b1, b0, distance="manhattan")
    other = _belief(t=1)
    with pytest.raises(ValueError, match="shape mismatch"):
        drift_to(b1, _BeliefShim())  # wrong z shape


class _BeliefShim:
    """Duck-typed belief with a mismatched z shape."""

    z = torch.randn(B + 5, DZ)
    s = None
