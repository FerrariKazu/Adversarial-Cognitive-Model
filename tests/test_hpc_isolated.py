"""
Stage 2 — HPC isolation tests (project lesson #3: one mechanism at a time).

Code-level isolated on/off checks for hpc_num_levels=0 vs 1. The 5-seed
matched protocol validation (numbers, Δ vs baseline with σ_combined) remains
pending on the GPU host — see docs/rhan_next_roadmap.json stage 2.
"""
import gc

import pytest
import torch

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_B, _C, _H, _W = 2, 3, 96, 96


def _model(cfg):
    return RHANNext(config=cfg).to(_DEVICE).eval()


def test_hpc_on_adds_params_only_when_enabled():
    m_off = _model(RHANNextConfig(enable_hpc=False))
    m_on = _model(RHANNextConfig(enable_hpc=True, hpc_num_levels=1))
    n_off = sum(p.numel() for p in m_off.parameters())
    n_on = sum(p.numel() for p in m_on.parameters())
    assert n_on > n_off, "HPC on must add trainable parameters"
    assert not hasattr(m_off, "hpc_stack")
    assert hasattr(m_on, "hpc_stack")
    del m_off, m_on
    gc.collect()


def test_hpc_level_0_feature_target_is_edge_map_not_pixels():
    m = _model(RHANNextConfig(enable_hpc=True, hpc_num_levels=1))
    assert m.hpc_stack.feature_targets() == ["edge_map"]
    assert m.hpc_stack.levels[0].feature_target == "edge_map"
    del m
    gc.collect()


def test_hpc_stack_rejects_unimplemented_levels():
    """Never add two levels in the same validation cycle."""
    from rhan_core.predictive_coding.hierarchical_stack import (
        HierarchicalPredictiveStack)
    with pytest.raises(NotImplementedError, match="one level per validation"):
        HierarchicalPredictiveStack(proj_dim=512, num_levels=2, fovea_size=48)
    # Config-level guard too.
    with pytest.raises(ValueError, match="hpc_num_levels"):
        RHANNextConfig(enable_hpc=True, hpc_num_levels=2)


def test_hpc_on_off_forward_shapes_and_errors():
    torch.manual_seed(0)
    x = torch.randn(_B, _C, _H, _W, device=_DEVICE)
    m_off = _model(RHANNextConfig(enable_hpc=False))
    m_on = _model(RHANNextConfig(enable_hpc=True, hpc_num_levels=1))
    with torch.no_grad():
        out_off, traj_off = m_off(x, return_trajectory=True)
        out_on, traj_on = m_on(x, return_trajectory=True)
    assert out_off.shape == out_on.shape == (_B, 10)
    # Off: no hpc errors; On: exactly levels * steps errors collected.
    assert "hpc_errors" not in traj_off
    assert len(traj_on["hpc_errors"]) == 1 * m_on.max_steps
    # hpc loss is zero when off, positive-and-finite when on.
    l_off = m_off.get_hpc_loss(x, (out_off, traj_off))
    l_on = m_on.get_hpc_loss(x, (out_on, traj_on))
    assert float(l_off) == 0.0
    assert float(l_on) > 0.0 and torch.isfinite(l_on)
    del m_off, m_on
    gc.collect()


def test_hpc_edge_extractors_shapes():
    from rhan_core.predictive_coding.feature_targets import (
        EdgeMapExtractor, OrientationMapExtractor)
    x = torch.randn(_B, 3, 48, 48, device=_DEVICE)
    edge = EdgeMapExtractor().to(_DEVICE)
    orient = OrientationMapExtractor().to(_DEVICE)
    e = edge(x)
    o = orient(x)
    assert e.shape == (_B, 1, 48, 48)
    assert o.shape == (_B, 2, 48, 48)
    assert float(e.min()) >= 0.0 and float(e.max()) <= 1.0
    assert float(o.min()) >= -1.0 and float(o.max()) <= 1.0
