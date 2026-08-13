"""Stage 2 — legacy 1-group optimizer state vs the new 2-group optimizer resume.

2026-08-13: the trainer's optimizer became TWO param groups — backbone at
phase_lr, HPC predictor at phase_lr*hpc_lr_mult (build_next_optimizer). Stage
1's AIS-v1 checkpoints carry a SINGLE-group SGD optimizer state (no HPC
module; ~322 momentum buffers over the AIS-v1 param set). Resuming one onto
the new optimizer is the checkpoint-resume hazard class this project has been
bitten by before (the destroyed rolling checkpoint, the best/rolling parity
gap): a naive load_state_dict either crashes (ValueError on group-count
mismatch) or — if the group count ever matched with a different param order —
PyTorch maps momentum buffers BY POSITION and SILENTLY misassigns them,
corrupting Stage 2 numbers quietly rather than failing loudly.

These tests pin the trainer's guard + fresh-optimizer fallback through the
REAL resume path (restore_optimizer_from_checkpoint from train_rhan_next.py —
never a re-implementation in this file):
  * a legacy 1-group state is REFUSED — no crash, no load, so the fresh
    optimizer starts every group (incl. HPC) with an EMPTY state dict;
  * after the refusal, the HPC group's momentum buffer is created fresh from
    THIS run's first gradient — never inherited from the legacy checkpoint's
    buffers (marker-filled so any inheritance is unambiguously detectable);
  * a legitimate MID-PHASE resume (cosine-decayed lrs in the saved state) is
    still allowed — the guard compares the flag-derived scheduler base_lrs,
    not the decayed current lrs (a false refusal would drop momentum and
    restart the cosine schedule on every session continuation);
  * the unguarded naive load crashes loudly (ValueError) — if PyTorch ever
    starts accepting a group-count mismatch silently, this test fails;
  * the legitimate same-shape 2-group resume round-trips momentum onto the
    SAME-named params.
"""
import gc
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1_training"))

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext
from train_rhan_next import (
    build_next_optimizer,
    optimizer_restore_compatible,
    restore_optimizer_from_checkpoint,
)

_PHASE_LR = 0.003
_T_MAX = 20


def _legacy_ais_v1_optimizer_state(seed=0, marker=7.0):
    """A checkpoint optimizer state matching Stage 1's AIS-v1 shape: ONE SGD
    group (momentum 0.9, wd 1e-4, foreach — the trainer's recipe) over a model
    with NO HPC module (enable_hpc=False), stepped a few times so the momentum
    buffers are non-zero. The buffers are then overwritten with `marker` so an
    inherited buffer is unambiguously detectable."""
    torch.manual_seed(seed)
    m = RHANNext(config=RHANNextConfig())          # AIS-v1 param set (no HPC)
    opt = torch.optim.SGD(m.parameters(), lr=_PHASE_LR, momentum=0.9,
                          weight_decay=1e-4, foreach=True)
    for _ in range(3):
        opt.zero_grad()
        for p in m.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
    state = opt.state_dict()
    assert len(state["param_groups"]) == 1
    del opt
    del m
    gc.collect()
    for v in state["state"].values():
        v["momentum_buffer"].fill_(marker)
    return state


def _new_two_group_optimizer():
    """The 2026-08-13 optimizer: two SGD groups via the trainer's builder."""
    m = RHANNext(config=RHANNextConfig(enable_hpc=True, hpc_num_levels=1))
    return m, build_next_optimizer(m, phase_lr=_PHASE_LR, hpc_lr_mult=6.67)


def _fresh_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_T_MAX, eta_min=_PHASE_LR * 0.1)


def _checkpoint(opt_state, sched_state):
    return {"optimizer": opt_state, "scheduler": sched_state}


def test_legacy_single_group_state_is_refused_not_loaded():
    """The trainer's resume path (restore_optimizer_from_checkpoint) must
    refuse a legacy 1-group state and fall back to the fresh optimizer — no
    crash, no load, no inherited momentum."""
    torch.manual_seed(0)
    legacy = _legacy_ais_v1_optimizer_state()
    assert len(legacy["param_groups"]) == 1

    _, opt = _new_two_group_optimizer()
    sched = _fresh_scheduler(opt)
    assert len(opt.param_groups) == 2

    ckpt = _checkpoint(legacy, {})
    assert restore_optimizer_from_checkpoint(opt, sched, ckpt, rank=0) is False, (
        "the resume path must refuse a legacy 1-group state — it must NOT "
        "silently load it onto the 2-group optimizer")
    # The fresh-optimizer fallback: nothing from the legacy checkpoint merged
    # in — in particular NO pre-seeded momentum for the HPC params (the
    # silent-misalignment failure).
    assert opt.state == {}, (
        "legacy momentum state leaked into the fresh optimizer: HPC params "
        "would start with a stale backbone momentum buffer")
    del opt
    gc.collect()


def test_hpc_group_momentum_starts_fresh_after_legacy_refusal():
    """After the resume path refuses the legacy state, the HPC group's
    momentum must be created fresh from THIS run's first gradient — not
    inherited from the legacy single group's buffers (which carry a
    distinctive 7.0 marker)."""
    torch.manual_seed(0)
    legacy = _legacy_ais_v1_optimizer_state(marker=7.0)
    m, opt = _new_two_group_optimizer()
    sched = _fresh_scheduler(opt)
    assert restore_optimizer_from_checkpoint(
        opt, sched, _checkpoint(legacy, {}), rank=0) is False

    # Deterministic, distinctive gradient on every param; ONE optimizer step.
    with torch.no_grad():
        for p in m.parameters():
            p.grad = torch.full_like(p, 0.5)
    opt.step()

    # SGD creates the momentum buffer on the FIRST step as a clone of the
    # first gradient (d_p = grad + wd*param ~ 0.5) — fresh-init semantics.
    # The buffer must be ~0.5, never the legacy marker 7.0.
    checked = 0
    for name, p in m.named_parameters():
        buf = opt.state[p]["momentum_buffer"]
        assert torch.allclose(buf, torch.full_like(buf, 0.5), atol=1e-2), (
            f"momentum buffer NOT freshly initialized from this run's first "
            f"gradient: {name} buf-mean={float(buf.mean()):.4f} (a legacy "
            f"buffer would read 7.0)")
        checked += 1
    assert checked > 0
    # Neither the HPC group NOR the backbone group may carry the marker.
    for p, st in opt.state.items():
        assert float(st["momentum_buffer"].abs().max()) < 7.0, (
            "a legacy momentum buffer (marker 7.0) survived into the new "
            "optimizer — silent misalignment")
    del m
    del opt
    gc.collect()


def test_naive_unguarded_load_of_legacy_state_crashes_loudly():
    """The unguarded path must FAIL LOUDLY (ValueError on group-count
    mismatch), never silently accept. If PyTorch ever starts accepting a
    1-group state into a 2-group optimizer (positionally misassigning
    momentum), this assertion fails — catching the silent corruption the
    guard exists to prevent."""
    torch.manual_seed(0)
    legacy = _legacy_ais_v1_optimizer_state()
    _, opt = _new_two_group_optimizer()
    with pytest.raises(ValueError):
        opt.load_state_dict(legacy)
    # The failed load must leave the optimizer untouched (no partial state).
    assert opt.state == {}
    del opt
    gc.collect()


def test_mid_phase_cosine_decayed_state_is_still_restorable():
    """A legitimate session-continuation resume mid-phase carries COSINE-
    DECAYED lrs in the saved optimizer state (the rolling checkpoint is saved
    every epoch), while the trainer rebuilds the optimizer at phase-start lrs.
    Comparing current lrs would falsely refuse every real mid-phase resume —
    dropping momentum and restarting the cosine schedule (the 2026-08-13
    guard regression, found in review). The guard must compare the
    FLAG-DERIVED values: the scheduler state's base_lrs."""
    torch.manual_seed(0)
    m1, opt1 = _new_two_group_optimizer()
    sched1 = _fresh_scheduler(opt1)
    for _ in range(5):                       # decay the group lrs mid-phase
        with torch.no_grad():
            for p in m1.parameters():
                p.grad = torch.randn_like(p)
        opt1.step()
        sched1.step()
    saved_opt = opt1.state_dict()
    saved_sched = sched1.state_dict()
    decayed_lrs = [g["lr"] for g in saved_opt["param_groups"]]
    assert any(abs(l - _PHASE_LR) > 1e-6 for l in decayed_lrs), \
        "fixture must actually decay the group lrs"

    m2, opt2 = _new_two_group_optimizer()
    sched2 = _fresh_scheduler(opt2)
    # Without the scheduler base_lrs source, the guard would falsely refuse:
    assert optimizer_restore_compatible(saved_opt, opt2) is False
    # With it, the legitimate resume is allowed:
    assert optimizer_restore_compatible(saved_opt, opt2, saved_sched) is True
    ckpt = _checkpoint(saved_opt, saved_sched)
    assert restore_optimizer_from_checkpoint(opt2, sched2, ckpt, rank=0) is True
    # Restored lrs must equal the SAVED (decayed) lrs — not the fresh ones.
    for g, lr in zip(opt2.param_groups, decayed_lrs):
        assert g["lr"] == pytest.approx(lr, rel=1e-6)
    del m1, m2, opt1, opt2
    gc.collect()


def test_matching_two_group_state_restores_cleanly():
    """Positive control: the legitimate same-shape 2-group resume passes the
    guard and load_state_dict maps every momentum buffer onto the SAME-named
    param (proving the guard's acceptance path is genuinely safe, and that
    the refusals above are not over-broad)."""
    torch.manual_seed(0)
    m1, opt1 = _new_two_group_optimizer()
    sched1 = _fresh_scheduler(opt1)
    with torch.no_grad():
        for p in m1.parameters():
            p.grad = torch.randn_like(p)
    opt1.step()
    saved = _checkpoint(opt1.state_dict(), sched1.state_dict())
    assert len(saved["optimizer"]["param_groups"]) == 2

    m2, opt2 = _new_two_group_optimizer()
    sched2 = _fresh_scheduler(opt2)
    assert restore_optimizer_from_checkpoint(opt2, sched2, saved, rank=0) is True

    n_checked = 0
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert tuple(p1.shape) == tuple(p2.shape)
        b1 = opt1.state[p1]["momentum_buffer"]
        b2 = opt2.state[p2]["momentum_buffer"]
        assert torch.allclose(b1, b2), (
            f"momentum buffer misassigned by position: {n2} differs from its "
            f"source {n1}")
        n_checked += 1
    assert n_checked > 0
    del m1, m2, opt1, opt2
    gc.collect()
