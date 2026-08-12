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
import math

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


# ────────────────────────────────────────────────────────────────────────────
# Dead-head regression (2026-08-12 Stage 2 smoke).
#
# The smoke's hpc_error_mean sat frozen at its init value (0.2848) across ALL
# 10 logged epochs — including 5 main-phase epochs where the stack was
# unfrozen and w_hpc*L_hpc was in the loss — while the rest of the model
# trained fine (loss down, TrAcc up). Root cause: the Tanh output head started
# at the default kaiming init, pinning its output at the Tanh saturation
# extremes (error-map max ~1.92 ~ |−1 − 1|) where gradients vanish. The fix is
# two-part: (1) zero-init the final ConvTranspose2d so the head starts in the
# linear regime, and (2) rescale the [0, 1] Sobel target to [-1, 1] so MSE is
# computed against the Tanh head's own range. These tests pin BOTH parts.
# ────────────────────────────────────────────────────────────────────────────

def test_hpc_head_starts_in_linear_regime():
    """Dead-head fix: a fresh head must start just inside the linear Tanh
    regime (d/dz tanh = 1 - pred^2 > 0.5 everywhere, i.e. |pred| < ~0.71), never
    ±1 saturation — the state that froze the smoke's prediction error."""
    torch.manual_seed(0)
    level = HPCLevel1(embed_dim=768, tap_layer="foveal_crop").to(_DEVICE)
    belief = torch.randn(4, 512, device=_DEVICE)
    crops = torch.randn(4, 3, 48, 48, device=_DEVICE)
    pred, _, _ = level(belief, crops)
    slope = 1.0 - pred ** 2              # tanh'(z) per output pixel
    assert float(slope.min()) > 0.5, (
        "head output must stay in the Tanh linear regime (|pred| < ~0.71); "
        "a saturated head pins the prediction error at its init value — the "
        "2026-08-12 dead-head failure")
    assert float(pred.abs().max()) < 0.71
    del level
    gc.collect()


def test_hpc_target_rescaled_to_match_tanh_head():
    """Range-match fix: extract_target must return 2*t - 1 of the [0, 1]
    Sobel magnitude, so MSE is computed against the Tanh head's [-1, 1]
    range (the raw extractor contract [0, 1] is unchanged)."""
    torch.manual_seed(0)
    level = HPCLevel1(embed_dim=768, tap_layer="foveal_crop").to(_DEVICE)
    lvl = level.stack.levels[0]
    crops = torch.randn(4, 3, 48, 48, device=_DEVICE)
    raw = lvl.extractor(crops)                    # Sobel magnitude in [0, 1]
    tgt = lvl.extract_target(crops)
    assert float(raw.min()) >= 0.0 and float(raw.max()) <= 1.0
    assert float(tgt.min()) >= -1.0 - 1e-5 and float(tgt.max()) <= 1.0 + 1e-5
    assert torch.allclose(tgt, raw * lvl.TARGET_SCALE + lvl.TARGET_SHIFT)
    del level
    gc.collect()


def test_hpc_head_init_gradient_is_full_scale():
    """Dead-head discriminator: at init the output conv must receive a
    FULL-SCALE gradient. The un-fixed kaiming-inited head sat at Tanh
    saturation where d/dz tanh ~ 0, so this gradient collapsed to a tiny
    fraction of its healthy scale — the head's weights barely moved and the
    smoke's hpc_error_mean stayed frozen at its init value. Measured for the
    fixed head: ~0.4."""
    torch.manual_seed(0)
    level = HPCLevel1(embed_dim=768, tap_layer="foveal_crop").to(_DEVICE)
    belief = torch.randn(4, 512, device=_DEVICE)
    crops = torch.randn(4, 3, 48, 48, device=_DEVICE)
    _, err, _ = level(belief, crops)
    err.mean().backward()
    head = level.stack.levels[0].decoder[-2]
    gn = float(head.weight.grad.norm())
    assert gn > 0.05, (
        f"output-conv init gradient collapsed to {gn:.4f} — the head is at "
        "Tanh saturation (dead head). It must be > 0.05 for the predictor "
        "to learn; the 2026-08-12 smoke froze at its init error for this "
        "reason.")
    del level
    gc.collect()


def test_hpc_head_learns_under_optimization():
    """THE dead-head regression, measured the way the smoke does: on the REAL
    belief (correlated with the foveal crop, backbone frozen), the head must
    reduce its prediction error under a few SGD steps. The un-fixed head's
    error never moved across the smoke's 10 logged epochs (frozen at init);
    this asserts the fixed head learns (>= 10% drop over 10 steps).
    NOTE (robustness): measured drop is ~28% (B=4) / passes at B=2; if this
    ever flakes on another backend, raise steps 10->15 or relax the bar to
    5% — it is deliberately stricter than the gate's >=10%-over-15-epochs."""
    torch.manual_seed(0)
    cfg = RHANNextConfig(enable_hpc=True, hpc_num_levels=1)
    m = RHANNext(config=cfg).to(_DEVICE).eval()
    x = torch.randn(_B, _C, _H, _W, device=_DEVICE)
    # Freeze everything except the HPC predictor: the backbone produces the
    # belief, and only the head learns — exactly the mechanism the smoke
    # exercises (warmup keeps the head frozen, main phase unfreezes it).
    hpc_params = []
    for name, p in m.named_parameters():
        if "hpc_level1" in name:
            p.requires_grad = True
            hpc_params.append(p)
        else:
            p.requires_grad = False
    assert hpc_params, "expected HPCLevel1 parameters to train"
    opt = torch.optim.SGD(hpc_params, lr=0.05)

    def hpc_err():
        with torch.enable_grad():
            logits, traj = m(x, return_trajectory=True)
        return torch.stack(traj["hpc_errors"]).mean().detach()

    e0 = float(hpc_err())
    assert math.isfinite(e0), "init HPC error must be finite"
    for _ in range(10):
        opt.zero_grad()
        with torch.enable_grad():
            logits, traj = m(x, return_trajectory=True)
        torch.stack(traj["hpc_errors"]).mean().backward()
        opt.step()
    e10 = float(hpc_err())
    drop = 1.0 - e10 / e0
    assert drop >= 0.10, (
        f"HPC prediction error did not learn (10 SGD steps: {e0:.4f} -> "
        f"{e10:.4f}, {drop*100:.1f}% drop) — dead head (Tanh saturation) or "
        f"detached target. The 2026-08-12 smoke froze at its init value for "
        f"this reason.")
    del m
    gc.collect()
