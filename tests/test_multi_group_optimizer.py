"""
Generation 0 — multi-group optimizer registry tests.
================================================================================

Phase 0's blocking gate is test_multi_group_optimizer.py passing. The
registry generalizes Stage 2's two-group SGD fix (backbone lr=0.003, HPC head
lr=0.003*6.67, per-group grad clip) to N named groups. Key guarantees pinned:

  1. Group structure: backbone registered first (lr_multiplier=1.0), named
     auxiliary groups after — the SAME order the pre-migration two-group
     builder produced; every param in exactly one group (a double-registered
     param would get two momentum buffers and double updates).
  2. build_optimizer applies group lr = base_lr * lr_multiplier and records
     name/clip_norm on the param group dicts (the resume guard's identity
     source).
  3. clip_grad_per_group clips EACH group to its OWN budget — never a global
     budget (the exact mechanism that fixed HPC's starvation).
  4. resume_guard refuses group-count / name / lr-ratio mismatches, accepts a
     legitimate mid-phase resume (decayed lrs + scheduler base_lrs).
  5. MIGRATION (the pre-registered test): re-wiring Stage 2's two-group setup
     through the registry produces a numerically identical optimizer — same
     group count, same lrs, same per-step |dW| for the HPC group on the same
     inputs — i.e. this is a REFACTOR being verified, not a retrain. If this
     fails, Generation 0 has a bug, full stop.
"""
import copy
import gc
import os
import sys
from pathlib import Path

# Deterministic GPU execution for the migration test's bit-exact |dW|
# comparison: CUBLAS_WORKSPACE_CONFIG must be set BEFORE torch imports
# cuBLAS. The project's documented cross-run GPU nondeterminism (grid_sample /
# attention backward) would otherwise make two identical runs differ by ~1e-4
# relative — the migration test needs the tight tolerance to mean something.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch
import torch.nn as nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1_training"))

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext
from rhan_core.optim.multi_group_optimizer import (
    OptimizerGroupRegistry,
    default_group_spec,
)
from train_rhan_next import build_next_optimizer, clip_grad_per_group

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_PHASE_LR = 0.003


# ── 1. Group structure ──────────────────────────────────────────────────────

def test_registry_group_order_and_disjointness():
    """backbone is group 0 at mult 1.0; named groups follow; disjoint params."""
    model = RHANNext(config=RHANNextConfig(enable_ais=True, enable_hpc=True,
                                           hpc_num_levels=1,
                                           enable_sbr=True)).to(_DEVICE)
    reg = default_group_spec(model)
    names = reg.group_names
    assert names[0] == "backbone", f"backbone must be first, got {names}"
    for g in reg.registered:
        assert g["lr_multiplier"] == 1.0 or g["lr_multiplier"] == 6.67, \
            f"unexpected multiplier {g['lr_multiplier']} on {g['name']}"
    # Every trainable param in exactly one group.
    seen = {}
    for g in reg.registered:
        for p in g["params"]:
            assert id(p) not in seen, \
                f"param in two groups: {p._name if hasattr(p, '_name') else p}"
            seen[id(p)] = g["name"]
    n_params = sum(len(g["params"]) for g in reg.registered)
    assert n_params == sum(1 for p in model.parameters() if p.requires_grad)


def test_registry_duplicate_registration_raises():
    reg = OptimizerGroupRegistry()
    p = nn.Parameter(torch.randn(4))
    reg.register_backbone([p])
    with pytest.raises(ValueError, match="already claimed"):
        reg.register("hpc", [p], lr_multiplier=6.67)


def test_registry_backbone_must_be_first():
    reg = OptimizerGroupRegistry()
    p = nn.Parameter(torch.randn(4))
    reg.register("hpc", [p], lr_multiplier=6.67)
    with pytest.raises(RuntimeError, match="FIRST"):
        reg.register_backbone([p])


# ── 2. build_optimizer semantics ────────────────────────────────────────────

def test_build_optimizer_lr_and_metadata():
    reg = OptimizerGroupRegistry()
    bp = nn.Parameter(torch.randn(8))
    hp = nn.Parameter(torch.randn(4))
    reg.register_backbone([bp])
    reg.register("hpc", [hp], lr_multiplier=6.67, clip_norm=1.0)
    opt = reg.build_optimizer(base_lr=_PHASE_LR)
    g0, g1 = opt.param_groups
    assert g0["lr"] == pytest.approx(_PHASE_LR)
    assert g1["lr"] == pytest.approx(_PHASE_LR * 6.67)
    assert g0["name"] == "backbone" and g1["name"] == "hpc"
    assert g1["clip_norm"] == 1.0


def test_default_group_spec_disjoint_aux_prefixes():
    """relational.* / evidence.* belong to their own groups, not 'sbr'."""
    model = RHANNext(config=RHANNextConfig(
        enable_ais=True, enable_hpc=True, hpc_num_levels=1, enable_sbr=True,
        sbr_stage="relational")).to(_DEVICE)
    reg = default_group_spec(model)
    names = reg.group_names
    assert "sbr" in names and "relational" in names and "evidence" in names
    # The sbr group must NOT own relational.* / evidence.* params.
    name_of = {id(p): n for n, p in model.named_parameters()}
    for gname in ("sbr", "relational", "evidence"):
        for p in reg.group(gname)["params"]:
            pname = name_of[id(p)]
            assert pname.startswith("structured_belief."), \
                f"{gname} group owns out-of-tree param {pname}"
            if gname == "relational":
                assert ".relational." in pname
            elif gname == "evidence":
                assert ".evidence." in pname
            else:
                assert ".relational." not in pname and ".evidence." not in pname


# ── 3. per-group clipping ───────────────────────────────────────────────────

def test_clip_grad_per_group_isolates_budgets():
    """A huge backbone grad must not dilute a small aux grad (the HPC lesson)."""
    reg = OptimizerGroupRegistry()
    bp = nn.Parameter(torch.randn(1000))
    hp = nn.Parameter(torch.randn(2))
    reg.register_backbone([bp])
    reg.register("hpc", [hp], lr_multiplier=6.67, clip_norm=0.01)
    opt = reg.build_optimizer(base_lr=_PHASE_LR)
    # Backbone grad enormous (norm ~1000), hpc grad small.
    bp.grad = torch.randn(1000) * 1000.0
    hp.grad = torch.randn(2) * 0.1
    before_h = hp.grad.clone()
    reg.clip_grad_per_group()
    # hpc clipped to ITS OWN 0.01 budget, not crushed by the backbone.
    assert hp.grad.norm() <= 0.01 * (1 + 1e-4)
    assert bp.grad.norm() <= 1.0 * (1 + 1e-4)  # backbone uses its own default budget


# ── 4. resume guard ─────────────────────────────────────────────────────────

def _registry_two_group():
    reg = OptimizerGroupRegistry()
    bp = nn.Parameter(torch.randn(8))
    hp = nn.Parameter(torch.randn(4))
    reg.register_backbone([bp])
    reg.register("hpc", [hp], lr_multiplier=6.67)
    return reg, bp, hp


def test_resume_guard_count_mismatch_refused():
    reg, _, _ = _registry_two_group()
    # Single-group state (Stage 1 AIS-v1 checkpoint shape).
    opt1 = torch.optim.SGD([nn.Parameter(torch.randn(3))], lr=0.003)
    assert reg.resume_guard(opt1.state_dict()) is False


def test_resume_guard_name_mismatch_refused():
    reg, _, _ = _registry_two_group()
    opt = reg.build_optimizer(base_lr=_PHASE_LR)
    state = copy.deepcopy(opt.state_dict())
    state["param_groups"][1]["name"] = "something_else"
    assert reg.resume_guard(state) is False


def test_resume_guard_lr_ratio_pattern():
    """A legitimate mid-phase resume (decayed lrs + scheduler base_lrs) is
    accepted; a mismatched multiplier pattern is refused."""
    reg, _, _ = _registry_two_group()
    opt = reg.build_optimizer(base_lr=_PHASE_LR)
    # Simulate a mid-phase checkpoint: lrs cosine-decayed, scheduler carries
    # the flag-derived base_lrs.
    state = copy.deepcopy(opt.state_dict())
    state["param_groups"][0]["lr"] = _PHASE_LR * 0.4
    state["param_groups"][1]["lr"] = _PHASE_LR * 6.67 * 0.4
    scheduler = {"base_lrs": [_PHASE_LR, _PHASE_LR * 6.67]}
    assert reg.resume_guard(state, saved_scheduler=scheduler) is True
    # Without the scheduler, a UNIFORMLY decayed state still satisfies the
    # lr-ratio pattern (the momentum buffers are valid for this config) —
    # but a NON-uniform decay that breaks the ratio is refused.
    assert reg.resume_guard(state) is True      # uniform decay: same ratio
    uneven = copy.deepcopy(state)
    uneven["param_groups"][1]["lr"] = _PHASE_LR * 6.67 * 0.4 * 0.5
    assert reg.resume_guard(uneven) is False
    # Different multiplier pattern -> refused (without a scheduler the saved
    # lrs are the only signal; with one, the scheduler's base_lrs are the
    # authoritative flag-derived values by design — same semantics as the
    # Stage 2 optimizer_restore_compatible guard).
    bad = copy.deepcopy(state)
    bad["param_groups"][1]["lr"] = _PHASE_LR * 3.0 * 0.4
    assert reg.resume_guard(bad) is False
    assert reg.resume_guard(bad, saved_scheduler=scheduler) is True


# ── 5. MIGRATION — the pre-registered test ─────────────────────────────────

def _two_group_legacy(model):
    """The EXACT pre-migration Stage 2 builder (from git history): backbone
    at phase_lr, 'hpc'-named params at phase_lr*hpc_lr_mult, one global clip."""
    backbone_params, hpc_params = [], []
    for name, p in model.named_parameters():
        (hpc_params if "hpc" in name else backbone_params).append(p)
    return torch.optim.SGD([
        {"params": backbone_params, "lr": _PHASE_LR},
        {"params": hpc_params, "lr": _PHASE_LR * 6.67},
    ], momentum=0.9, weight_decay=1e-4, foreach=True)


def test_hpc_migration_preserves_stage2_numbers():
    """Registry-rebuilt optimizer == pre-migration two-group optimizer:
    same group count, same lrs, same HPC-group per-step |dW| on the same
    inputs. A REFACTOR being verified, not a retrain."""
    torch.manual_seed(42)
    model = RHANNext(config=RHANNextConfig(
        enable_ais=True, enable_hpc=True, hpc_num_levels=1)).to(_DEVICE)
    model.train()

    x = torch.randn(2, 3, 96, 96, device=_DEVICE)
    y = torch.randint(0, 10, (2,), device=_DEVICE)

    def hpc_group_dw(opt):
        hpc_params = [g["params"] for g in opt.param_groups
                      if g["lr"] > _PHASE_LR][0]
        return sum(float(p.grad.detach().norm() ** 2)
                   for p in hpc_params if p.grad is not None) ** 0.5

    def run_recipe(m, opt, clip_fn):
        # Re-seed so BOTH recipes measure under the SAME dropout masks — the
        # comparison is about the optimizer mechanics (lr/momentum/clip), not
        # about RNG sampling.
        torch.manual_seed(0)
        dws = []
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits, traj = m(x, return_trajectory=True)
                l_trades = torch.nn.functional.cross_entropy(logits, y)
                l_hpc = m.get_hpc_loss(x, (logits, traj))
                loss = l_trades + 0.10 * l_hpc
            loss.backward()
            clip_fn(opt)
            dws.append(hpc_group_dw(opt))
            opt.step()
        return dws

    legacy = _two_group_legacy(model)
    dw_legacy = run_recipe(model, legacy, clip_grad_per_group)

    torch.manual_seed(42)
    model2 = RHANNext(config=RHANNextConfig(
        enable_ais=True, enable_hpc=True, hpc_num_levels=1)).to(_DEVICE)
    model2.train()
    registry_opt = build_next_optimizer(model2, phase_lr=_PHASE_LR,
                                        hpc_lr_mult=6.67)
    dw_registry = run_recipe(model2, registry_opt, clip_grad_per_group)

    for a, b in zip(dw_legacy, dw_registry):
        # Relative tolerance 0.1%: the pre-migration failure mode was a
        # ~1000x |dW| dilution (1.36e-5 vs 2.7e-3), so 0.1% catches any real
        # regression while tolerating the project's documented cross-run GPU
        # nondeterminism (~1.5 pp / kernel-order noise in grid_sample and
        # attention backward).
        rel = abs(a - b) / max(abs(a), 1e-12)
        assert rel < 1e-3, \
            f"HPC per-step |dW| diverged after registry migration: " \
            f"legacy {a:.6e} vs registry {b:.6e} (rel {rel:.2e}) — " \
            f"Generation 0 has a bug"

    # Same optimizer group layout.
    gl = [g["lr"] for g in legacy.param_groups]
    gr = [g["lr"] for g in registry_opt.param_groups]
    assert gl == gr, f"lr layout diverged: {gl} vs {gr}"

    del model, model2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()