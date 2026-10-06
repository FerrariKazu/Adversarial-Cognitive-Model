"""
Agent A contract test — per-group optimizer isolation.

Constructs the case the registry exists for: group 1's gradient norm is
100x group 2's, with a clip budget of 1.0. A GLOBAL clip suppresses
group 2's gradient to ~1% (the Stage-2 HPC starvation mechanism);
per-group clipping does not. Also the contract's smoke experiment: two
dummy param groups with different lr multipliers, 5 optimizer steps on a
toy loss — each group's effective LR must match its multiplier exactly
and clipping must be independent per group.
"""
import pytest
import torch

from noesis_vision.core.multi_group_optimizer import (
    OptimizerGroupRegistry,
    DEFAULT_CLIP_NORM,
)


def _two_groups(mult2=6.67):
    reg = OptimizerGroupRegistry()
    p1 = torch.nn.Parameter(torch.zeros(1))   # backbone (mult 1.0)
    p2 = torch.nn.Parameter(torch.zeros(1))   # auxiliary head (mult mult2)
    reg.register_backbone([p1])
    reg.register("aux", [p2], lr_multiplier=mult2)
    return reg, p1, p2


def test_per_group_clipping_isolates_small_group():
    reg, p1, p2 = _two_groups()
    p1.grad = torch.tensor([100.0])   # backbone's big TRADES-like grad
    p2.grad = torch.tensor([1.0])     # the small head's grad

    # The Gen-0 failure mode: a GLOBAL budget lets group 1 own the norm.
    torch.nn.utils.clip_grad_norm_([p1, p2], DEFAULT_CLIP_NORM)
    global_ratio = (p2.grad.norm() / p1.grad.norm()).item()
    assert global_ratio < 0.02, (
        "sanity: a global clip must crush group 2's gradient (~1%)")

    # The fix: per-group clipping restores group 2's gradient.
    p1.grad = torch.tensor([100.0])
    p2.grad = torch.tensor([1.0])
    reg.clip_grad_per_group()
    assert abs(p1.grad.norm().item() - 1.0) < 1e-5
    assert abs(p2.grad.norm().item() - 1.0) < 1e-5
    ratio = (p2.grad.norm() / p1.grad.norm()).item()
    assert ratio > 0.9, (
        f"per-group clipping must NOT suppress group 2 "
        f"(grad ratio {ratio:.3f}); the Stage-2 starvation regressed")


def test_smoke_effective_lr_matches_multiplier():
    base_lr = 0.3
    reg, p1, p2 = _two_groups(mult2=6.67)
    opt = reg.build_optimizer(base_lr, momentum=0.0, weight_decay=0.0)
    lrs = [g["lr"] for g in opt.param_groups]
    assert lrs[0] == base_lr
    assert lrs[1] == base_lr * 6.67

    # 5 optimizer steps on a toy loss path (constant grad = 1.0):
    # SGD(momentum=0): each step moves p DOWN the gradient by exactly its
    # own lr (p <- p - lr * grad), so the per-step |movement| = lr.
    deltas = {0: [], 1: []}
    for _ in range(5):
        before = [p1.item(), p2.item()]
        p1.grad = torch.tensor([1.0])
        p2.grad = torch.tensor([1.0])
        reg.clip_grad_per_group()          # independent budgets
        opt.step()
        after = [p1.item(), p2.item()]
        deltas[0].append(before[0] - after[0])
        deltas[1].append(before[1] - after[1])
    # NOTE tolerance: params are float32; at movement magnitude ~2.0 the
    # per-step measurement error is ~2e-6, so 1e-6 would false-fail. The
    # exact ratio assertion below is the precise check.
    assert all(abs(d - base_lr) < 1e-4 for d in deltas[0]), deltas[0]
    assert all(abs(d - base_lr * 6.67) < 1e-4 for d in deltas[1]), deltas[1]
    observed_ratio = sum(deltas[1]) / sum(deltas[0])
    assert abs(observed_ratio - 6.67) < 1e-4, (
        f"per-group effective-LR ratio {observed_ratio:.5f} != 6.67")


def test_double_registration_refused():
    reg, p1, _ = _two_groups()
    with pytest.raises(ValueError, match="ONE optimizer group"):
        reg.register("other", [p1], lr_multiplier=2.0)


def test_resume_guard_refuses_foreign_layouts():
    reg, p1, p2 = _two_groups()
    opt = reg.build_optimizer(0.3, momentum=0.0, weight_decay=0.0)
    good = opt.state_dict()
    assert reg.resume_guard(good) is True

    # Group-count mismatch (a pre-multiplier single-group checkpoint).
    assert reg.resume_guard({"param_groups": [{"name": "backbone",
                                               "lr": 0.3}]}) is False
    # Name mismatch with same count (positional-misassignment shape).
    swapped = {"param_groups": [{"name": "aux", "lr": 0.3},
                                {"name": "backbone", "lr": 0.3 * 6.67}]}
    assert reg.resume_guard(swapped) is False
    # lr-ratio mismatch (a different multiplier's checkpoint).
    other_mult = {"param_groups": [{"name": "backbone", "lr": 0.3},
                                   {"name": "aux", "lr": 0.9}]}
    assert reg.resume_guard(other_mult) is False
