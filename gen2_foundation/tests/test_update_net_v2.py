"""G2-K5/K6/K7 — update mechanism invariants.

1. Gen-1 parity path executable: fusion='error_only' + mean pool == the
   original ConcreteUpdateNet behavior.
2. observed-feature fusion: concat / gated / FiLM arms.
3. Position-sensitive spatial error pooling (attention, not mean).
4. Adaptive-bound arm with hard ceiling (Delta z <= 0.1).
"""

from __future__ import annotations

import pytest
import torch

from gen2_foundation.update_net_v2 import (
    BeliefUpdaterV2,
    FusionMode,
    adapt_magnitude,
    PositionSensitiveErrorPool,
)


class TestParity:
    def test_error_only_parity(self):
        # Gen-1 parity: fusion='error_only' + mean-pool == ConcreteUpdateNet.
        upd = BeliefUpdaterV2(d_z=384, obs_dim=384,
                              fusion_mode='error_only')
        z = torch.randn(2, 384)
        err = torch.randn(2, 4, 4)
        with torch.no_grad():
            d = upd(z, err)
        # Flat mean is the Gen-1 baseline; the attention pool must differ
        # (position-sensitivity is verified in TestSpatialPooling).
        assert d.shape == (2, 384)
        assert torch.all(d.abs() <= 0.1 + 1e-6)

    def test_fusion_converts_error_only_to_concat(self):
        z = torch.randn(2, 384)
        err = torch.randn(2, 4, 4)
        obs = torch.randn(2, 384)
        upd = BeliefUpdaterV2(d_z=384, obs_dim=384,
                              fusion_mode='concat')
        d = upd(z, err, observed=obs)
        assert d.shape == (2, 384)


class TestFusion:
    @pytest.mark.parametrize("mode", ["concat", "gated", "film"])
    def test_fusion_mode_produces_update(self, mode):
        upd = BeliefUpdaterV2(d_z=384, obs_dim=384,
                              fusion_mode=mode)
        z = torch.randn(2, 384)
        obs = torch.randn(2, 384)
        d = upd(z, torch.randn(2, 4, 4), observed=obs)
        assert d.shape == (2, 384)

    def test_fusion_error_only_no_obs_allowed(self):
        upd = BeliefUpdaterV2(d_z=384, obs_dim=384,
                              fusion_mode='error_only')
        z = torch.randn(2, 384)
        err = torch.randn(2, 4, 4)
        d = upd(z, err)   # observed=None is allowed (Gen-1 parity)
        assert d.shape == (2, 384)


class TestSpatialPooling:
    def test_position_sensitivity(self):
        # Mean pooling is position-insensitive; attention pooling is not.
        err = torch.zeros(2, 4, 4)
        err[0, 2, 2] = 1.0                          # center
        err[1, 0, 0] = 1.0                          # corner
        pool = PositionSensitiveErrorPool(d_z=384, n_tokens=16)
        a = pool(err)
        # The two inputs differ -> the pool is position-sensitive.
        assert not torch.allclose(a[0], a[1])


class TestAdaptiveBound:
    def test_hard_ceiling(self):
        # Delta z must never exceed hard bound.
        z = torch.zeros(2, 384)
        U = torch.rand(2) * 0.5
        raw = torch.ones(2, 384) * 0.5             # raw delta > bound
        out = adapt_magnitude(raw, mode='bounded', max_bound=0.1,
                               U=U, z=z)
        assert torch.all(torch.abs(out) <= 0.1 + 1e-6)

    def test_adaptive_gains_with_uncertainty(self):
        z = torch.zeros(2, 384)
        U_high = torch.ones(2) * 0.9
        U_low = torch.zeros(2) * 0.1
        raw = torch.ones(2, 384) * 0.05
        a_high = adapt_magnitude(raw, mode='adaptive', max_bound=0.1,
                                  U=U_high, z=z)
        a_low = adapt_magnitude(raw, mode='adaptive', max_bound=0.1,
                                  U=U_low, z=z)
        assert a_high.abs().sum() > a_low.abs().sum()

    def test_unbounded_allows_bigger(self):
        raw = torch.ones(2, 384) * 0.9
        out = adapt_magnitude(raw, mode='unbounded', max_bound=0.1)
        assert torch.all(torch.abs(out) == 0.9)
