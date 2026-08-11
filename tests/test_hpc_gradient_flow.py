"""
Stage 2 — HPC gradient-flow tests (project lesson #1, applied to HPCLevel1).

The #1 historical failure mode in this project was a silently-detached loss
term: the reconstruction error was detached for two architecture generations
(v11/v12) before it was caught, making w_recon*L_recon a gradient no-op. The
reconstruction loss was fixed, the AIS gaze/precision paths were added with
tests, and NOW the HPC predictor gets the same treatment — as a HARD assertion
in this file, not a manual review step:

  1. HPCLevel1.forward's error tensor must be attached (requires_grad True,
     grad_fn present) — a regression here fails LOUD at test time instead of
     surfacing later as a degenerate Π_D pattern the way recon-mod did.
  2. Backward must reach the HPCLevel1 predictor parameters (fc + decoder).
  3. The full-model L_hpc path (get_hpc_loss -> _forage) must also flow.

See docs/rhan_next_roadmap.json stage 2, health-gate check #1.
"""
import gc

import pytest
import torch

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext
from rhan_core.predictive_coding.hpc_level1 import HPCLevel1

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_B, _C, _H, _W = 2, 3, 96, 96


def _assert_param_grads(model, name_fragments):
    """Assert every param whose name contains a fragment got a nonzero grad."""
    got = []
    for name, p in model.named_parameters():
        if any(frag in name for frag in name_fragments):
            g = p.grad
            if g is not None and g.abs().sum().item() > 0:
                got.append(name)
    assert got, (
        f"no nonzero gradient reached params matching {name_fragments}. "
        "This is the #1 historical failure mode (v11/v12 detached recon). "
        "Do NOT proceed until this is fixed.")
    return got


# ────────────────────────────────────────────────────────────────────────────
# HPCLevel1 standalone (isolated wiring module)
# ────────────────────────────────────────────────────────────────────────────

def test_hpc_level1_error_is_NOT_detached_hard_assertion():
    """CRITICAL: the HPC error tensor must stay attached to the graph.

    If this test fails, L_hpc is a gradient no-op (the v11/v12 recon bug).
    This is the Stage-1 lesson applied as a hard check, per the task spec.
    """
    torch.manual_seed(0)
    level = HPCLevel1(embed_dim=768, tap_layer="foveal_crop").to(_DEVICE)
    belief = torch.randn(_B, 512, device=_DEVICE, requires_grad=True)
    crop = torch.randn(_B, 3, 48, 48, device=_DEVICE)
    pred, err, err_map = level(belief, crop)

    assert pred.shape == (_B, 1, 48, 48)
    assert err.shape == (_B,)
    assert err_map.shape == (_B, 1, 48, 48)
    assert err.requires_grad, "HPC error is DETACHED — L_hpc is a gradient no-op"
    assert err.grad_fn is not None, "HPC error has no grad_fn — detached graph"
    assert pred.requires_grad, "prediction detached from the belief path"

    loss = err.mean()
    loss.backward()
    assert belief.grad is not None and belief.grad.abs().sum().item() > 0, \
        "top-down belief must receive gradient through the HPC error"
    del level
    gc.collect()


def test_hpc_level1_gradient_reaches_predictor_params():
    """Gradient must reach the stack's fc + decoder (the only trainable piece)."""
    torch.manual_seed(0)
    level = HPCLevel1(embed_dim=768, tap_layer="foveal_crop").to(_DEVICE)
    belief = torch.randn(_B, 512, device=_DEVICE)
    crop = torch.randn(_B, 3, 48, 48, device=_DEVICE)
    _, err, _ = level(belief, crop)
    err.mean().backward()

    got = _assert_param_grads(
        level, ["stack.levels.0.fc", "stack.levels.0.decoder"])
    assert any("fc" in g for g in got) and any("decoder" in g for g in got)
    # The extractor is non-learnable (Sobel buffers) — it must own no params.
    n_ext = sum(p.numel() for p in level.stack.levels[0].extractor.parameters())
    assert n_ext == 0, "EdgeMapExtractor must stay non-learnable"
    del level
    gc.collect()


# ────────────────────────────────────────────────────────────────────────────
# Full model path: get_hpc_loss -> _forage -> optimizer backward
# ────────────────────────────────────────────────────────────────────────────

def test_model_hpc_loss_attached_and_reaches_params():
    """End-to-end: the trainer's L_hpc path (w_hpc * L_hpc) must be a real
    gradient path into HPCLevel1 — exactly what train_rhan_next.py runs."""
    torch.manual_seed(0)
    cfg = RHANNextConfig(enable_hpc=True, hpc_num_levels=1)
    m = RHANNext(config=cfg).to(_DEVICE)
    x = torch.randn(_B, _C, _H, _W, device=_DEVICE)
    logits, traj = m(x, return_trajectory=True)
    l_hpc = m.get_hpc_loss(x, (logits, traj))
    assert l_hpc.requires_grad, "model L_hpc is detached — gradient no-op"
    assert l_hpc.grad_fn is not None
    # hpc_error_maps diagnostics must be populated too (the health gate reads
    # min/max/std from the per-epoch summary).
    assert len(traj.get("hpc_error_maps", [])) == m.max_steps
    loss = 0.10 * l_hpc
    loss.backward()
    got = _assert_param_grads(
        m, ["hpc_level1.stack.levels.0.fc", "hpc_level1.stack.levels.0.decoder"])
    assert got, "w_hpc * L_hpc must reach HPCLevel1 predictor params"
    del m
    gc.collect()


def test_hpc_on_state_dict_has_no_duplicate_keys():
    """Pin the hpc_stack alias: it is a plain-reference alias to the stack
    OWNED by hpc_level1, so the state dict must contain exactly ONE path
    (hpc_level1.stack.*) — never a second hpc_stack.* copy (PyTorch's
    remove_duplicate dedup does this; a regression here would corrupt
    checkpoint round-trips)."""
    torch.manual_seed(0)
    m = RHANNext(config=RHANNextConfig(enable_hpc=True, hpc_num_levels=1))
    keys = list(m.state_dict().keys())
    assert len(set(keys)) == len(keys), "duplicate state-dict keys present"
    assert any(k.startswith("hpc_level1.stack.") for k in keys)
    assert not any(k.startswith("hpc_stack.") for k in keys), \
        "hpc_stack alias must NOT create a second state-dict path"
    del m
    gc.collect()


def test_hpc_off_means_zero_hpc_loss():
    """hpc_num_levels=0 (or enable_hpc=False): L_hpc must be exactly 0 and no
    hpc trajectory keys collected."""
    for cfg in (RHANNextConfig(), RHANNextConfig(enable_hpc=True, hpc_num_levels=0)):
        torch.manual_seed(0)
        m = RHANNext(config=cfg).to(_DEVICE)
        x = torch.randn(_B, _C, _H, _W, device=_DEVICE)
        logits, traj = m(x, return_trajectory=True)
        assert float(m.get_hpc_loss(x, (logits, traj))) == 0.0
        assert "hpc_errors" not in traj
        del m
        gc.collect()
