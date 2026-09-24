"""
test_center_bias_measurement — Agent F.
================================================================================
The measurement must catch what it exists to catch (the standing
responsiveness-guard pattern): a synthetic always-center policy is
flagged degenerate; a scattered synthetic policy is not; a constant
NON-center policy is honestly NOT flagged by the center test (it shows
zero spatial spread instead); a constant-offset detector would show
|corr|=1. Per the contract, the gate threshold itself stays
PENDING DECISION — this file tests the measurement, never a verdict.
"""
from __future__ import annotations

import pytest
import torch

from noesis_vision.gaze.center_bias import (
    CenterBiasMeter,
    gaze_location_histogram,
)


class AlwaysCenterPolicy:
    """Synthetic degenerate policy: every location is exactly (0, 0)."""

    def __call__(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, 2)


class ScatteredPolicy:
    """Synthetic healthy policy: uniform locations over the frame."""

    def __call__(self, batch: int) -> torch.Tensor:
        return (torch.rand(batch, 2) * 2.0 - 1.0) * 0.9


class AlwaysTopLeftPolicy:
    """Constant but NOT center — must NOT trip the center flag."""

    def __call__(self, batch: int) -> torch.Tensor:
        return torch.full((batch, 2), -0.8)


def test_synthetic_always_center_flagged_degenerate():
    meter = CenterBiasMeter()
    for _ in range(5):
        meter.update(AlwaysCenterPolicy()(8))
    rep = meter.report()
    assert rep.degenerate is True
    assert rep.fraction_at_center >= 0.999
    assert rep.mean_radius <= 1e-3
    assert rep.reason is not None and "always-center" in rep.reason


def test_scattered_policy_not_flagged():
    torch.manual_seed(0)
    meter = CenterBiasMeter()
    for _ in range(5):
        meter.update(ScatteredPolicy()(16))
    rep = meter.report()
    assert rep.degenerate is False
    assert rep.fraction_at_center < 0.999
    # The histogram is usable: mass across many bins, not one spike.
    hist = torch.tensor(rep.hist_x) + torch.tensor(rep.hist_y)
    occupied = (hist > 0).sum().item()
    assert occupied >= 4, f"histogram collapsed to {occupied} bins"


def test_constant_non_center_not_center_flagged():
    meter = CenterBiasMeter()
    for _ in range(3):
        meter.update(AlwaysTopLeftPolicy()(8))
    rep = meter.report()
    assert rep.degenerate is False          # not a CENTER-bias signature
    assert rep.fraction_at_center == 0.0    # but measurable: zero at center
    assert rep.mean_radius > 0.5
    # Zero spatial spread is still visible in the marginals.
    assert max(rep.hist_x) > 0.9


def test_meter_shapes_accumulation_and_detachment():
    meter = CenterBiasMeter()
    g = torch.Generator().manual_seed(3)
    locs = (torch.rand(10, 2, generator=g) * 2 - 1) * 0.9
    locs.requires_grad_(True)               # must not poison the graph
    meter.update(locs)
    meter.update(torch.zeros(4, 2))
    rep = meter.report()
    assert rep.n_locations == 14
    # fraction_at_center reflects the 4 zeros: 4/14 within radius 0.25.
    assert rep.fraction_at_center == pytest.approx(4 / 14, abs=1e-6)
    assert rep.mean_radius >= 0.0
    # Marginals are normalized histograms over the shared edges.
    assert len(rep.hist_x) == meter.hist_bins == len(rep.hist_y)
    assert abs(sum(rep.hist_x) - 1.0) < 1e-5
    assert rep.hist_edges[0] == -1.0 and rep.hist_edges[-1] == 1.0
    assert locs.grad is None                # measurement left the graph alone


def test_joint_histogram_rows_y_cols_x_and_normalization():
    torch.manual_seed(1)
    # Two corner clusters: (+0.7, -0.7) and (-0.7, +0.7).
    pts = torch.cat([torch.tensor([[0.7, -0.7]]).repeat(5, 1),
                     torch.tensor([[-0.7, 0.7]]).repeat(5, 1)])
    hist = gaze_location_histogram([pts], hist_bins=8)
    assert hist.shape == (8, 8)
    assert torch.isclose(hist.sum(), torch.tensor(1.0), atol=1e-5)
    # Bin index for coordinate v in [-1,1]: floor((v+1)/2 * bins).
    i = int((-0.7 + 1.0) / 2.0 * 8)         # = 1
    j = int((0.7 + 1.0) / 2.0 * 8)          # = 6
    # (x=+0.7, y=-0.7) -> row y=1, col x=6; (x=-0.7, y=+0.7) -> row 6, col 1.
    assert hist[i, j].item() == pytest.approx(0.5, abs=1e-6)
    assert hist[j, i].item() == pytest.approx(0.5, abs=1e-6)
    assert hist.sum() > 0 and (hist >= 0).all()


def test_meter_empty_report_raises_and_histogram_out_of_range_raises():
    meter = CenterBiasMeter()
    with pytest.raises(ValueError):
        meter.report()                       # nothing accumulated
    with pytest.raises(ValueError):
        gaze_location_histogram([])          # no locations
    with pytest.raises(ValueError):
        gaze_location_histogram([torch.full((3, 2), 5.0)])  # all outside
    with pytest.raises(ValueError):
        CenterBiasMeter(center_radius=0.0)   # invalid measurement radius
