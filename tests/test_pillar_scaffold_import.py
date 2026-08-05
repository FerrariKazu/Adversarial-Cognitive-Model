"""
Stage 0 — scaffold tests for Pillars 3 (SBR) and 4 (IWM).

Contract: SBR/IWM classes import and instantiate cleanly; calling their core
methods raises a clear, documented NotImplementedError (or, for
NullWorldModel, returns a safe passthrough). No import-time or
instantiation-time errors anywhere.
"""
import torch

import rhan_core  # noqa: F401  (must import cleanly)


def test_package_imports_cleanly():
    from rhan_core import RHANNext, RHANNextConfig
    from rhan_core.beliefs import BeliefState, VectorBeliefState, StructuredBeliefState
    from rhan_core.predictive_coding import base as pc_base
    from rhan_core.gaze import GazePolicy
    from rhan_core.precision import PrecisionModulator
    from rhan_core.world_model import WorldModel, NullWorldModel
    assert all(x is not None for x in (
        RHANNext, RHANNextConfig, BeliefState, VectorBeliefState,
        StructuredBeliefState, pc_base.LevelPredictor, pc_base.ErrorUnit,
        GazePolicy, PrecisionModulator, WorldModel, NullWorldModel))


def test_structured_belief_instantiates_but_raises_on_use():
    from rhan_core.beliefs import StructuredBeliefState
    sb = StructuredBeliefState(num_slots=16, slot_dim=512)
    assert sb.num_slots == 16 and sb.slot_dim == 512
    for method, args in [
        ("as_tensor", ()),
        ("uncertainty", ()),
        ("update_slots", (torch.zeros(4, 512),)),
        ("message_passing", ()),
    ]:
        try:
            getattr(sb, method)(*args)
            raise AssertionError(f"StructuredBeliefState.{method} should raise")
        except NotImplementedError as e:
            msg = str(e)
            assert "PILLAR 3" in msg or "Pillar 3" in msg, f"unclear error: {msg}"
            assert "SCAFFOLD" in msg or "scaffold" in msg, f"unclear error: {msg}"


def test_null_world_model_safe_passthrough():
    from rhan_core.beliefs.vector_belief import VectorBeliefState
    from rhan_core.world_model.null_world_model import NullWorldModel
    wm = NullWorldModel()  # instantiates cleanly, zero params
    assert sum(p.numel() for p in wm.parameters()) == 0
    s = torch.randn(4, 512)
    belief = VectorBeliefState(s)
    out = wm.simulate(belief, torch.zeros(4, 2))
    assert torch.equal(out, s), "NullWorldModel must pass its input through unchanged"


def test_shape_embedding_extractor_scaffold_raises_on_call():
    from rhan_core.predictive_coding.feature_targets import ShapeEmbeddingExtractor
    ext = ShapeEmbeddingExtractor()  # instantiate ok
    try:
        ext(torch.randn(2, 3, 48, 48))
        raise AssertionError("ShapeEmbeddingExtractor.forward should raise")
    except NotImplementedError as e:
        assert "scaffold" in str(e).lower()


def test_default_model_has_world_model_and_no_extra_pillars():
    from rhan_core.model import RHANNext
    m = RHANNext()
    assert hasattr(m, "world_model")
    assert not hasattr(m, "gaze_policy"), "AIS must be OFF by default"
    assert not hasattr(m, "hpc_stack"), "HPC must be OFF by default"
    assert not m.pillars_active
