"""
Stage 2 — HPC disable backward-compat (non-negotiable lesson #3).

Disabling HPC must EXACTLY reproduce the validated Stage 1 config: a model
built with enable_hpc=True, hpc_num_levels=0 must behave bit-for-bit like the
AIS-v1 (halting-only variant) model with enable_hpc=False — identical state
dict, identical forward outputs (same test pattern used for the v12
backward-compat in Stage 0: same-seed construction + allclose).

This is the automated smoke-time check (health-gate criterion #3): the Stage 2
notebook runs this file's tests before Step B; a failure means HPC wiring
silently changed the AIS-v1 forward path — STOP, do not train.

The AIS-v1 baseline config (Stage 1 validated):
    enable_ais=True, ais_halt_enabled=True, ais_precision_recon_enabled=False
    (halting ON; precision-modulated recon weight OFF — the
    "AIS-v1 (halting-only variant)").
"""
import gc
import os

import pytest
import torch

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_B, _C, _H, _W = 2, 3, 96, 96

# The validated Stage 1 config (AIS-v1, halting-only variant).
AIS_V1 = dict(enable_ais=True, ais_halt_enabled=True,
              ais_precision_recon_enabled=False)


def _ais_v1(hpc_num_levels):
    """AIS-v1 config with HPC at the given level count (0 = disabled)."""
    return RHANNextConfig(enable_hpc=True, hpc_num_levels=hpc_num_levels,
                          **AIS_V1)


def _ais_v1_off():
    """The exact Stage 1 config — HPC entirely absent from the config."""
    return RHANNextConfig(enable_hpc=False, **AIS_V1)


def _model(cfg):
    return RHANNext(config=cfg).to(_DEVICE).eval()


def test_hpc_off_and_hpc_levels0_state_dicts_identical():
    """enable_hpc=True, hpc_num_levels=0 must add ZERO keys vs enable_hpc=False
    (the hpc_level1 module must not be built when no level is requested)."""
    m_off = _model(_ais_v1_off())
    m_zero = _model(_ais_v1(0))
    assert set(m_off.state_dict().keys()) == set(m_zero.state_dict().keys())
    assert (sum(p.numel() for p in m_off.parameters())
            == sum(p.numel() for p in m_zero.parameters()))
    assert not hasattr(m_off, "hpc_level1")
    assert not hasattr(m_zero, "hpc_level1")
    del m_off, m_zero
    gc.collect()


def test_hpc_off_and_hpc_levels0_forward_identical():
    """Same-seed construction (identical trees -> identical weights) must give
    bit-comparable forward outputs — the Stage-0 v12-compat pattern."""
    torch.manual_seed(0)
    x = torch.randn(_B, _C, _H, _W, device=_DEVICE)

    torch.manual_seed(0)
    m_off = _model(_ais_v1_off())
    torch.manual_seed(0)
    m_zero = _model(_ais_v1(0))

    with torch.no_grad():
        out_off = m_off(x)
        out_zero = m_zero(x)
    assert torch.allclose(out_off, out_zero, atol=1e-5), \
        "hpc_num_levels=0 changed the AIS-v1 forward outputs"
    del m_off, m_zero
    gc.collect()


def test_hpc_levels0_forward_matches_loaded_ais_v1_checkpoint():
    """(Optional, needs the Stage 1 checkpoint locally or on HF.)

    Load the ACTUAL validated AIS-v1 weights into both an HPC-off and an
    hpc_num_levels=0 model and assert identical forward outputs — the
    strongest form of the disable check (runs against the real artifact the
    Stage 2 smoke will resume from).
    """
    ckpt_name = os.environ.get("AIS_V1_CKPT",
                               "checkpoints/rhan_next_ais_v1_halting_only_best.pth")
    if not os.path.exists(ckpt_name):
        pytest.skip(f"Stage 1 checkpoint not present locally ({ckpt_name}) — "
                    "the RNG-based disable test above already covers this check")
    state = torch.load(ckpt_name, map_location="cpu", weights_only=False)
    state = state.get("model", state)

    torch.manual_seed(0)
    m_off = _model(_ais_v1_off())
    torch.manual_seed(0)
    m_zero = _model(_ais_v1(0))
    miss_off, _ = m_off.load_state_dict(state, strict=False)
    miss_zero, _ = m_zero.load_state_dict(state, strict=False)
    assert miss_off == miss_zero, "HPC-off and levels-0 must drop identical keys"

    x = torch.randn(_B, _C, _H, _W, device=_DEVICE)
    with torch.no_grad():
        out_off = m_off(x)
        out_zero = m_zero(x)
    assert torch.allclose(out_off, out_zero, atol=1e-4), \
        "hpc_num_levels=0 diverged from the loaded AIS-v1 checkpoint"
    del m_off, m_zero
    gc.collect()
