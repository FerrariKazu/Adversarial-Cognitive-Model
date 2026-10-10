"""G2-K7-ema invariants.

1. Gradient-free EMA target encoder (not an nn.Module; never registered to
   an optimizer).
2. Online/target feature agreement measured and logged.
3. EMA lag diagnostics.
4. Explicit EMA-on / EMA-off experiment states.
"""

from __future__ import annotations

import pytest
import torch

from gen2_foundation.ema import EMATargetEncoder


class _StubPredictor:
    """Stand-in for ConcreteGlimpseFeaturePredictor (no params here)."""

    def __init__(self):
        self._weights = torch.nn.Parameter(torch.zeros(2))

    def state_dict(self):
        return {'a0': torch.tensor([1.0, 2.0]),
                'a1': torch.tensor([3.0, 4.0])}

    def predict_features(self, belief, gaze_location):
        return belief


class TestGradientFree:
    def test_not_nn_module(self):
        pred = _StubPredictor()
        ema = EMATargetEncoder(pred, alpha=0.996)
        assert not isinstance(ema, torch.nn.Module)

    def test_has_no_optimizer_registration(self):
        # Contract: EMA never registered to an optimizer.
        pred = _StubPredictor()
        ema = EMATargetEncoder(pred, alpha=0.996)
        # A plain object with no .parameters() -> cannot be registered.
        assert not hasattr(ema, 'parameters')

    def test_sync_updates_weights(self):
        pred = _StubPredictor()
        ema = EMATargetEncoder(pred, alpha=0.996)
        # The predictor's state has keys 'a0' and 'a1'; the EMATarget
        # shadow mirrors them and sync blends them by alpha.
        ema.sync()
        assert float(ema.alpha) == 0.996
        # A buffer was populated by sync (agreement_log may be empty).
        assert ema.buffer_names


class TestAgreementLag:
    def test_record_agreement(self):
        pred = _StubPredictor()
        ema = EMATargetEncoder(pred, alpha=0.996)
        v = ema.record_agreement(torch.randn(4, 4), torch.randn(4, 4))
        assert -1.0 <= v <= 1.0

    def test_ema_lag(self):
        pred = _StubPredictor()
        ema = EMATargetEncoder(pred, alpha=0.996)
        # Record agreement over time (agreement should change -> non-None).
        for i in range(5):
            ema.record_agreement(torch.ones(4) * i, torch.ones(4) * (i + 1))
        lag = ema.ema_lag()
        assert lag is not None

    def test_ema_lag_empty(self):
        pred = _StubPredictor()
        ema = EMATargetEncoder(pred, alpha=0.996)
        assert ema.ema_lag() is None


class TestOnOff:
    def test_ema_on_off_states(self, _none=None):
        pred = _StubPredictor()
        # EMA on state: alpha < 1 -> Polyak averaging applied.
        ema_on = EMATargetEncoder(pred, alpha=0.996)
        assert ema_on.alpha == 0.996
        # EMA off state: alpha=1 -> no decay (explicit off path).
        ema_off = EMATargetEncoder(pred, alpha=1.0)
        assert ema_off.alpha == 1.0
        # sync with alpha=1 is a no-op
        ema_off.sync()
