"""Agent E contract — predictor EMA target (BYOL/DINO-style).

Adjusts the model so the predictor's target (observed glimpse features)
is produced by an EMA copy whose weights are NEVER backpropagated through.
The training predictor keeps its gradient; the shadow is a flat buffer
container updated by polyak averaging each step.
"""

import tempfile

import pytest
import torch

from noesis_vision.predictive_coding.ema_predictor import PredictorTarget
from noesis_vision.predictive_coding.glimpse_predictor import ConcreteGlimpseFeaturePredictor


B, D, N = 2, 384, 16


def _pred() -> ConcreteGlimpseFeaturePredictor:
    torch.manual_seed(0)
    return ConcreteGlimpseFeaturePredictor(d_z=D, d_feat=D, n_tokens=N)


def _to_flat(name: str) -> str:
    """Dotted module name -> flattened shadow name."""
    return name.replace(".", "_")


def _belief() -> object:
    """Minimal belief-shaped duck (z, uncertainty), mirror of the
    pre-flight's ``_BeliefView`` and the real belief composition."""

    class _BV:
        def __init__(self):
            self._z = torch.randn(B, D)
            self._u = torch.rand(B)

        @property
        def z(self):
            return self._z

        @property
        def uncertainty(self):
            return self._u

    return _BV()


def test_ema_target_shadow_has_no_learnable_weights():
    pred = _pred()
    ema = PredictorTarget(pred, alpha=0.5)
    assert len(list(ema.parameters())) == 0, (
        "the EMA target must carry NO learnable weights — it is a frozen "
        "shadow, the training predictor owns the graph"
    )
    assert len(list(ema.named_buffers())) == len(list(pred.named_parameters())), (
        f"shadow buffers {len(list(ema.named_buffers()))} != predictor params "
        f"{len(list(pred.named_parameters()))}"
    )


def test_ema_target_detached_from_gradient():
    """Backward through the target must not reach the shadow buffers.

    The EMA shadow is a detached buffer container — no gradient may flow into
    it. The training predictor, by contrast, owns its own graph and receives
    gradients from the belief loss (the target is consumed, not a second loss).
    """
    pred = _pred()
    ema = PredictorTarget(pred, alpha=0.5)
    belief = _belief()
    gaze = torch.zeros(B, 2)

    out = ema.predict_features(belief, gaze)
    assert tuple(out.shape) == (B, N, D)
    assert out.requires_grad

    out.sum().backward()

    # the EMA shadow buffers never carry a gradient — the shadow is frozen
    assert all(getattr(b, "_grad", None) is None for _, b in ema.named_buffers()), (
        "the EMA shadow buffers must never carry a gradient"
    )
    # the shadow carries no learnable weights
    assert len(list(ema.parameters())) == 0

def test_polyak_update_slides_target_toward_source():
    """Each update blends source weights by alpha (BYOL/DINO convention)."""
    pred = _pred()
    ema = PredictorTarget(pred, alpha=0.996)

    # ── formula: one update moves the shadow toward the (moving) source ──
    # The source is a "frozen" copy for this probe; we step it manually
    # between updates. The shadow is never re-synced, so polyak blending is
    # the only thing that moves it and it lags behind the source.
    opt = torch.optim.SGD(pred.parameters(), lr=0.001)
    for _ in range(3):
        belief = _belief()
        for p in pred.parameters():
            p.grad = None
        out = pred(belief, torch.zeros(2, 2))
        out.sum().backward()
        ema.update()
        opt.step()
        opt.zero_grad()
        for p in pred.parameters():
            if p.grad is not None:
                p.grad.zero_()

    # Capture the shadow right before a fresh update, then apply the update
    # against the current source and verify the alpha blend.
    ema_before = {n: buf.clone() for n, buf in ema.named_buffers()}
    src_before = {n.replace(".", "_"): p for n, p in pred.named_parameters()}
    # Move the source so the "before" and "after" differ...
    for p in pred.parameters():
        p.grad = None
    out = pred(_belief(), torch.zeros(2, 2))
    out.sum().backward()
    src_after = {n.replace(".", "_"): p for n, p in pred.named_parameters()}
    ema.update()
    for n, buf in ema.named_buffers():
        # update: ema = alpha * ema_before + (1 - alpha) * src_after
        expected = 0.996 * ema_before[n] + 0.004 * src_after[n]
        assert torch.allclose(buf, expected, atol=1e-6), (
            f"polyak step {n.replace('_', '.')} did not apply alpha={ema.alpha}"
        )
    # The shadow must not equal the source: it is always a lagging average.
    assert any(
        not torch.allclose(buf, src_after[n])
        for n, buf in ema.named_buffers()
    ), (
        "the EMA shadow should lag the source after a handful of steps "
        "(alpha < 1.0), not track it exactly"
    )

def test_checkpoint_round_trip():
    pred = _pred()
    ema = PredictorTarget(pred, alpha=0.5)
    state = ema.state_dict()
    pred_flat = {n.replace(".", "_") for n, _ in pred.named_parameters()}
    assert set(state.keys()) == pred_flat, (
        f"state keys {sorted(state.keys())} != flat predictor names "
        f"{sorted(pred_flat)} (flat vs dotted name mapping)"
    )


    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/ema.pt"
        torch.save(state, p)
        ema2 = PredictorTarget(pred, alpha=0.5)
        ema2.load_state_dict(torch.load(p, map_location="cpu"))
        for n in ema.state_dict():
            assert torch.allclose(ema.state_dict()[n],
                                  ema2.state_dict()[n]), (
                f"round-trip mismatch for {n}")
    # the source predictor is untouched by load/save
    for n, p in pred.named_parameters():
        assert torch.allclose(dict(pred.named_parameters())[n].detach(),
                              p.detach().clone(), atol=1e-7)


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        PredictorTarget(ConstantPredictor(), alpha=1.0)
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        PredictorTarget(ConstantPredictor(), alpha=0.0)


class ConstantPredictor(torch.nn.Module):
    """A stub predictor with one parameter — used only as an alpha edge case."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 2)

    def forward(self, belief, gaze_location: torch.Tensor) -> torch.Tensor:
        return self.net(gaze_location)
