"""
Stage 1 — AIS sub-mechanism ablation flags (mechanism isolation, Run A/B
pattern). The smoke health gate fired on the Pi_D ordering criterion
(car/airplane instead of car/truck), so the two new AIS sub-mechanisms must
be individually ablatable to attribute the shift:

  * ais_halt_enabled=False          -> entropy gate forced open (cont=1, v12
                                       fixed-T belief accumulation); gaze
                                       update unchanged.
  * ais_precision_recon_enabled=False -> w_recon stays FLAT (v12 recon
                                       weighting) in dynamic_trades_loss_next.

Each flag ablates exactly ONE sub-mechanism (project lesson #3). These tests
pin the ablation behavior so an isolation run provably changes only its
intended mechanism.
"""
import gc

import torch
import torch.nn.functional as F

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_B, _C, _H, _W = 2, 3, 96, 96


def _model(cfg: RHANNextConfig) -> RHANNext:
    return RHANNext(config=cfg).to(_DEVICE)


def _dummy():
    torch.manual_seed(0)
    x = torch.randn(_B, _C, _H, _W, device=_DEVICE)
    y = torch.randint(0, 10, (_B,), device=_DEVICE)
    return x, y


def _param_grad_names(model, fragments):
    got = []
    for name, p in model.named_parameters():
        if any(frag in name for frag in fragments):
            g = p.grad
            if g is not None and g.abs().sum().item() > 0:
                got.append(name)
    return got


# ────────────────────────────────────────────────────────────────────────────
# Config surface
# ────────────────────────────────────────────────────────────────────────────

def test_default_config_keeps_both_mechanisms_enabled():
    """Defaults are the smoke behavior — backward compat preserved."""
    cfg = RHANNextConfig(enable_ais=True)
    assert cfg.ais_halt_enabled is True
    assert cfg.ais_precision_recon_enabled is True
    # Round-trip survives serialization (checkpoints embed the config).
    cfg2 = RHANNextConfig.from_dict(cfg.to_dict())
    assert cfg2 == cfg


# ────────────────────────────────────────────────────────────────────────────
# Isolation A: halting disabled -> constant continuation (v12 fixed-T)
# ────────────────────────────────────────────────────────────────────────────

def test_halting_disabled_forces_constant_continuation():
    """cont == 1 for every sample/step -> v12 fixed-T belief accumulation."""
    cfg = RHANNextConfig(enable_ais=True, ais_halt_enabled=False)
    m = _model(cfg).eval()
    x, _ = _dummy()
    with torch.no_grad():
        logits, traj = m(x, return_trajectory=True)
    assert logits.shape == (_B, 10)
    conts = traj["continuations"]
    assert len(conts) == cfg.max_foraging_steps == 4
    for c in conts:
        assert torch.allclose(c, torch.ones_like(c), atol=1e-6), \
            "halting disabled must force continuation == 1.0"
    del m
    gc.collect()


def test_halting_disabled_gaze_still_trains():
    """The isolation arm must still be trainable: gradient reaches the gaze
    policy (which is NOT ablated) and the precision modulator gain."""
    cfg = RHANNextConfig(enable_ais=True, ais_halt_enabled=False)
    m = _model(cfg)
    x, y = _dummy()
    logits, traj = m(x, return_trajectory=True)
    loss = F.cross_entropy(logits, y) + m.get_reconstruction_loss(x, (logits, traj))
    loss.backward()
    assert _param_grad_names(m, ["gaze_policy.step_net"]), \
        "gaze policy must still receive gradient with halting off"
    assert _param_grad_names(m, ["precision_modulator.gain"]), \
        "precision gain must still receive gradient with halting off"
    del m
    gc.collect()


# ────────────────────────────────────────────────────────────────────────────
# Isolation B: precision-modulated recon weight disabled -> w_recon flat
# ────────────────────────────────────────────────────────────────────────────

def test_precision_recon_disabled_returns_flat_w_recon():
    """precision_recon_enabled=False must give w_recon_eff == w_recon exactly
    (the v12 reconstruction weighting), regardless of Pi_D."""
    from train_rhan_next import dynamic_trades_loss_next

    cfg = RHANNextConfig(enable_ais=True, ais_precision_recon_enabled=False)
    m = _model(cfg).eval()
    x, y = _dummy()
    weights = torch.ones(_B, device=_DEVICE)
    with torch.no_grad():
        (_, _, _, _, _, _, w_recon_eff) = dynamic_trades_loss_next(
            m, x, y, weights, x.clone(), beta_base=2.0, w_recon=0.10,
            w_hpc=0.05, precision_recon_enabled=False)
    assert torch.allclose(w_recon_eff, torch.full_like(w_recon_eff, 0.10),
                          atol=1e-6), \
        "w_recon_eff must be exactly the flat w_recon when recon modulation is off"
    del m
    gc.collect()


def test_precision_recon_enabled_modulates_weight():
    """Default (enabled): w_recon_eff = w_recon * mean(0.5 + Pi_D) — i.e. NOT
    flat, proving the consumer is active unless explicitly ablated."""
    from train_rhan_next import dynamic_trades_loss_next

    cfg = RHANNextConfig(enable_ais=True)   # ais_precision_recon_enabled=True
    m = _model(cfg).eval()
    x, y = _dummy()
    weights = torch.ones(_B, device=_DEVICE)
    with torch.no_grad():
        (_, _, _, _, _, _, w_recon_eff) = dynamic_trades_loss_next(
            m, x, y, weights, x.clone(), beta_base=2.0, w_recon=0.10,
            w_hpc=0.05, precision_recon_enabled=True)
    flat = torch.full_like(w_recon_eff, 0.10)
    assert not torch.allclose(w_recon_eff, flat, atol=1e-3), \
        "precision-modulated recon weight must differ from flat when enabled"
    del m
    gc.collect()
