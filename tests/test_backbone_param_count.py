"""
Agent C contract test — backbone parameter count.

Asserts the contract's 20M-25M band (excluding pillar heads) and fails
LOUDLY outside it. The split (DINOv2-shaped trunk + tied refinement) is
also asserted so a future edit cannot silently reshape the substrate.
"""
import pytest

from noesis_vision.models.backbone import (
    PARAM_COUNT_BAND,
    D_Z,
    CompactViT,
)
from noesis_vision.models.recurrent_block import WITHIN_GLIMPSE_ITERS_RANGE


def _params(m):
    return sum(p.numel() for p in m.parameters())


def test_param_count_inside_band():
    m = CompactViT(img_size=56, num_refine_iters=2)
    n = _params(m)
    lo, hi = PARAM_COUNT_BAND
    assert lo <= n <= hi, (
        f"CompactViT parameter count {n:,} ({n/1e6:.2f}M) is OUTSIDE the "
        f"contract band [{lo/1e6:.0f}M, {hi/1e6:.0f}M] — the substrate "
        f"shape changed; re-derive the budget and re-pin the band, do not "
        f"silently widen the band")


def test_band_constants_sane():
    assert PARAM_COUNT_BAND == (20_000_000, 25_000_000)


def test_trunk_refinement_split():
    """The budget's documented split: DINOv2-small-shaped trunk + ONE tied
    refinement block. Asserts the split so drift is caught piecewise."""
    m = CompactViT(img_size=56)
    trunk = sum(p.numel() for n_, p in m.named_parameters()
                if not n_.startswith("refinement"))
    refine = sum(p.numel() for n_, p in m.named_parameters()
                 if n_.startswith("refinement"))
    # Trunk: 12 blocks @ width 384 — in band by itself (the warm-startable
    # part); refinement is the new-by-design remainder (~1.8M).
    assert 20_000_000 <= trunk <= 22_500_000, (
        f"trunk {trunk:,} outside the DINOv2-small-shaped expectation "
        f"(~21.5M)")
    assert 1_000_000 <= refine <= 2_500_000, (
        f"tied refinement {refine:,} outside the one-block expectation "
        f"(~1.8M)")


def test_dz_is_384_and_schema_recordable():
    """d_z is the substrate's call (Part 1.A): D_Z = 384, matching the
    DINOv2-small width so the warm-start maps cleanly. Downstream agents
    record it in RHANNXAConfig.d_z."""
    assert D_Z == 384
    m = CompactViT(img_size=56)
    assert m.d_z == D_Z


def test_refine_iters_only_change_compute_not_params():
    """Tied weights: the LOCKED 2-3 iteration range must leave the
    parameter count identical."""
    lo, hi = WITHIN_GLIMPSE_ITERS_RANGE
    n2 = _params(CompactViT(img_size=56, num_refine_iters=lo))
    n3 = _params(CompactViT(img_size=56, num_refine_iters=hi))
    assert n2 == n3


def test_non_divisible_image_size_refused():
    with pytest.raises(ValueError, match="divisible by patch size"):
        CompactViT(img_size=48)  # 48 not divisible by 14 — the flagged
        # fovea-size deviation from the STL-10 convention is exactly this


def test_refine_iters_outside_locked_range_refused():
    with pytest.raises(ValueError, match="LOCKED range"):
        CompactViT(img_size=56, num_refine_iters=4)
    with pytest.raises(ValueError, match="LOCKED range"):
        CompactViT(img_size=56, num_refine_iters=1)
