"""
Stage 0 — backward compatibility: the DEFAULT RHANNextConfig must reproduce
RHAN-v12's forward pass shape-for-shape and state-dict key-for-key.
"""
import gc
import os

import pytest
import torch

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_B, _C, _H, _W = 2, 3, 96, 96


def _make_v12():
    from model_rhan_v12 import RHANv12
    return RHANv12()


def test_default_config_state_dict_matches_v12():
    """All-pillars-off RHANNext must expose EXACTLY v12's parameter keys."""
    next_m = RHANNext()
    v12_m = _make_v12()
    next_keys = set(next_m.state_dict().keys())
    v12_keys = set(v12_m.state_dict().keys())
    assert next_keys == v12_keys, (
        f"default RHANNext state dict differs from v12: "
        f"+{sorted(next_keys - v12_keys)} -{sorted(v12_keys - next_keys)}")
    # Same parameter COUNT too.
    assert (sum(p.numel() for p in next_m.parameters())
            == sum(p.numel() for p in v12_m.parameters()))
    del next_m, v12_m
    gc.collect()


def test_default_forward_matches_v12_numerically():
    """Default RHANNext forward must be numerically identical to v12's."""
    torch.manual_seed(0)
    x = torch.randn(_B, _C, _H, _W, device=_DEVICE)
    # Construct both models under the SAME RNG state so their (identical)
    # initialization produces identical weights.
    torch.manual_seed(0)
    next_m = RHANNext().to(_DEVICE).eval()
    torch.manual_seed(0)
    v12_m = _make_v12().to(_DEVICE).eval()
    with torch.no_grad():
        out_next = next_m(x)
        out_v12 = v12_m(x)
    assert out_next.shape == (_B, 10) == out_v12.shape
    assert torch.allclose(out_next, out_v12, atol=1e-5), \
        "default RHANNext forward diverged from RHANv12"
    del next_m, v12_m
    gc.collect()


def test_forward_shapes_across_configs():
    torch.manual_seed(0)
    x = torch.randn(_B, _C, _H, _W, device=_DEVICE)
    # HPC stack module ships with Stage 1 as Pillar-1 infrastructure; its
    # gradient-reachability is asserted by test_gradient_flow.py (Stage 2).
    configs = [
        RHANNextConfig(),                            # v12-equivalent
        RHANNextConfig(enable_ais=True),             # Stage 1
        RHANNextConfig(enable_hpc=True, hpc_num_levels=1),  # Stage 2
        RHANNextConfig(enable_ais=True, enable_hpc=True, hpc_num_levels=1),
    ]
    for cfg in configs:
        m = RHANNext(config=cfg).to(_DEVICE).eval()
        with torch.no_grad():
            out = m(x)
            feat = m.get_feature_vector(x)
        assert out.shape == (_B, 10), f"{cfg}: logits {tuple(out.shape)}"
        assert feat.shape == (_B, 768), f"{cfg}: features {tuple(feat.shape)}"
        del m
        gc.collect()


def test_config_validation_gates():
    with pytest.raises(ValueError, match="enable_sbr"):
        RHANNextConfig(enable_sbr=True)
    with pytest.raises(ValueError, match="enable_iwm"):
        RHANNextConfig(enable_iwm=True)
    with pytest.raises(ValueError, match="hpc_num_levels"):
        RHANNextConfig(enable_hpc=True, hpc_num_levels=2)
    # Valid: hpc enabled with exactly one level.
    RHANNextConfig(enable_hpc=True, hpc_num_levels=1)


def test_config_roundtrip():
    cfg = RHANNextConfig(enable_ais=True, ais_halt_threshold=0.4,
                         hpc_num_levels=1, enable_hpc=True)
    cfg2 = RHANNextConfig.from_dict(cfg.to_dict())
    assert cfg2 == cfg
    with pytest.raises(ValueError, match="Unknown"):
        RHANNextConfig.from_dict({"enable_hpc": True, "not_a_field": 1})
    # from_dict must also reject scaffold pillars.
    with pytest.raises(ValueError, match="enable_sbr"):
        RHANNextConfig.from_dict({"enable_sbr": True})


def test_v12_kwargs_subset():
    cfg = RHANNextConfig(gaze_lambda=0.7, max_foraging_steps=6)
    kw = cfg.v12_kwargs()
    assert kw["gaze_lambda"] == 0.7 and kw["max_foraging_steps"] == 6
    assert kw["num_classes"] == 10 and kw["proj_dim"] == 512
    assert not any(k in kw for k in
                   ("enable_hpc", "enable_ais", "enable_sbr", "enable_iwm"))
