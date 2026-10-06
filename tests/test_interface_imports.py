"""
Agent 0 contract tests — stub imports and responsible-agent errors.

Per the Agent 0 contract (MASTER_PLAN):
  - every stub file imports cleanly;
  - every ABC method raises NotImplementedError WITH A MESSAGE NAMING
    WHICH FUTURE AGENT is responsible for it;
  - smoke experiment: import every file, instantiate RHANNXAConfig with
    defaults, call every stub method once — all raise NotImplementedError
    cleanly, and none raise an unrelated error (shape mismatch, missing
    import).
"""
import inspect

import pytest
import torch

import noesis_vision  # noqa: F401  (package imports cleanly)
from noesis_vision.beliefs import interfaces as belief_ifaces
from noesis_vision.beliefs.interfaces import BeliefState
from noesis_vision.core.schema import RHANNXAConfig
from noesis_vision.predictive_coding import interfaces as pc_ifaces
from noesis_vision.predictive_coding.interfaces import GlimpseFeaturePredictor, UpdateNet

RESPONSIBLE = {
    BeliefState: "Agent B",
    GlimpseFeaturePredictor: "Agent E",
    UpdateNet: "Agent E",
}

B = 2  # batch size used across smoke calls


def _belief_stub():
    """Minimal placeholder satisfying the ABC method signatures.

    The ABC instantiates (no abstract members, per the smoke-experiment
    contract); members are called unimplemented on purpose.
    """
    return BeliefState()


def test_stub_modules_import_cleanly():
    assert belief_ifaces is not None
    assert pc_ifaces is not None
    # Property/membership sanity: the ABCs expose the Part 1.A/1.B surface.
    assert {"z", "s", "evidence", "alpha", "uncertainty", "prediction_error",
            "gaze_history", "current_glimpse_idx"} <= set(dir(BeliefState))
    assert {"as_tensor", "drift_to"} <= set(dir(BeliefState))
    assert {"predict_features", "score_candidates"} <= set(dir(GlimpseFeaturePredictor))
    assert "forward" in dir(UpdateNet)


def _assert_agent_error(exc, cls):
    responsible = RESPONSIBLE[cls]
    assert isinstance(exc, NotImplementedError)
    assert responsible in str(exc), (
        f"error must name the responsible agent ({responsible}); got: {exc}"
    )


def test_belief_state_members_name_agent_b():
    b = _belief_stub()
    for name in ("z", "s", "evidence", "alpha", "uncertainty",
                 "prediction_error", "gaze_history", "current_glimpse_idx"):
        with pytest.raises(NotImplementedError) as ei:
            getattr(b, name)
        _assert_agent_error(ei.value, BeliefState)
    with pytest.raises(NotImplementedError) as ei:
        b.as_tensor()
    _assert_agent_error(ei.value, BeliefState)
    with pytest.raises(NotImplementedError) as ei:
        b.drift_to(b)
    _assert_agent_error(ei.value, BeliefState)


def test_predictor_members_name_agent_e():
    p = GlimpseFeaturePredictor()
    gaze = torch.zeros(B, 2)
    cands = torch.zeros(B, 4, 2)
    with pytest.raises(NotImplementedError) as ei:
        p.predict_features(belief=_belief_stub(), gaze_location=gaze)
    _assert_agent_error(ei.value, GlimpseFeaturePredictor)
    with pytest.raises(NotImplementedError) as ei:
        p.score_candidates(belief=_belief_stub(), candidate_locations=cands)
    _assert_agent_error(ei.value, GlimpseFeaturePredictor)


def test_update_net_names_agent_e():
    u = UpdateNet()
    with pytest.raises(NotImplementedError) as ei:
        u.forward(torch.zeros(B, 8), torch.zeros(B, 8))
    _assert_agent_error(ei.value, UpdateNet)


def test_smoke_experiment_full_pass():
    """Contract smoke experiment: import every file, instantiate
    RHANNXAConfig with defaults, call every stub method once — every
    raised error is the expected NotImplementedError, never anything
    unrelated.
    """
    cfg = RHANNXAConfig()  # defaults instantiate cleanly
    assert cfg.schema_version

    expected = NotImplementedError
    stub_calls = 0
    b = _belief_stub()
    for name in ("z", "s", "evidence", "alpha", "uncertainty",
                 "prediction_error", "gaze_history", "current_glimpse_idx"):
        with pytest.raises(expected):
            getattr(b, name)
        stub_calls += 1
    for name in ("as_tensor", "drift_to"):
            if name == "drift_to":
                with pytest.raises(expected):
                    getattr(b, name)(b)
            else:
                with pytest.raises(expected):
                    getattr(b, name)()
            stub_calls += 1

    p = GlimpseFeaturePredictor()
    with pytest.raises(expected):
        p.predict_features(b, torch.zeros(B, 2))
    stub_calls += 1
    with pytest.raises(expected):
        p.score_candidates(b, torch.zeros(B, 4, 2))
    stub_calls += 1

    u = UpdateNet()
    with pytest.raises(expected):
        u.forward(torch.zeros(B, cfg.d_z or 8), torch.zeros(B, cfg.d_z or 8))
    stub_calls += 1

    # Every stub member was exercised and every one raised cleanly
    # (8 belief properties + as_tensor + drift_to + 2 predictor methods
    # + 1 UpdateNet forward).
    assert stub_calls == 13


def test_no_interface_returns_detached_by_construction():
    """Gradient requirement (Agent 0 contract): no stub returns anything
    at all yet — assert the contract's intent, that unimplemented
    members never SILENTLY return a detached tensor placeholder.
    """
    b = _belief_stub()
    for name in ("z", "evidence", "prediction_error"):
        with pytest.raises(NotImplementedError):
            getattr(b, name)  # raises; nothing silently returned
