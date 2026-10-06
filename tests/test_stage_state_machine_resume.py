"""
test_stage_state_machine_resume — Agent J1.
================================================================================
The contract's resume test, scoped to THIS phase list: kill the trainer
mid-step-2 (recurrence_only), restart, and confirm it resumes step 2 —
not step 1, not step 3 — from the correct checkpoint with optimizer
state intact. Agent A's resume_or_abort + resume_guard do the heavy
lifting; this test proves they are wired correctly for the foundation
machine (and that the machine itself never skips or repeats a phase).
Synthetic data only — no dataset needed (Agent I's marked self-test
path); HF is disabled (use_hf=False) so the test never touches the
network.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.stage_state_machine import (  # noqa: E402
    FOUNDATION_PHASES,
    advance,
    ensure_foundation_state,
    get_next_action,
)
from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig,
    FoundationModel,
    train_one_epoch,
)


def _write(roadmap: dict, path: str) -> None:
    """Seed the roadmap file so advance()'s load-persist round-trip works."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(roadmap, f)


def _cfg(tmp_path, **kw) -> FoundationConfig:
    cfg = FoundationConfig()
    cfg.epochs = 2
    cfg.use_hf = False            # NEVER network in tests
    cfg.amp = False
    cfg.force_fresh = False
    cfg.ckpt_dir = str(tmp_path / "ckpts")
    cfg.report_dir = str(tmp_path / "report")
    cfg.runs_dir = str(tmp_path / "runs")
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _fake_loaders():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(8, 3, 96, 96, generator=g)
    y = torch.randint(0, 10, (8,), generator=g)
    return {"train": [(x, y)], "val": [(x, y)]}


def _train_n_epochs(model, loaders, cfg, device, n, start_epoch=0):
    """Minimal in-test training loop mirroring run_phase's resume math."""
    from noesis_vision.core.multi_group_optimizer import (
        OptimizerGroupRegistry)
    registry = OptimizerGroupRegistry()
    groups = model.group_params()
    registry.register_backbone(groups["backbone"])
    for name in ("classifier", "evidential_head", "predictor", "update_net",
                 "precision"):
        if name in groups:
            registry.register(name, groups[name])
    optimizer = registry.build_optimizer(cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=cfg.epochs)
    rolling = os.path.join(cfg.ckpt_dir,
                           f"foundation_{model.phase}_rolling.pth")
    for epoch in range(start_epoch, start_epoch + n):
        train_one_epoch(model, loaders["train"], optimizer, registry,
                        device, None)
        scheduler.step()
        from noesis_vision.core.checkpoint import save_rolling
        save_rolling(rolling, epoch=epoch + 1, model=model,
                     optimizer=optimizer, scheduler=scheduler)
    return model, optimizer, scheduler


def _fresh_model(cfg, phase):
    torch.manual_seed(0)
    m = FoundationModel(cfg, phase)
    return m


def test_kill_mid_step2_restart_resumes_step2(tmp_path, monkeypatch):
    """The contract: kill mid-recurrence_only, restart -> the machine says
    recurrence_only/training, the rolling checkpoint resumes at the right
    epoch, and optimizer state (momentum buffers) survives."""
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(tmp_path)
    loaders = _fake_loaders()
    device = torch.device("cpu")

    # ── session 1: finish step 1, get mid-way into step 2, then "die" ─────
    roadmap_path = os.path.join(cfg.report_dir, "rm.json")
    roadmap: dict = {}
    ensure_foundation_state(roadmap)
    _write(roadmap, roadmap_path)
    roadmap = advance("backbone_only", "done", roadmap_path=roadmap_path)
    roadmap = advance("recurrence_only", "running", roadmap_path=roadmap_path)

    m1 = _fresh_model(cfg, "recurrence_only")
    m1, opt1, sch1 = _train_n_epochs(m1, loaders, cfg, device, n=1)
    rolling = os.path.join(cfg.ckpt_dir,
                           "foundation_recurrence_only_rolling.pth")
    assert os.path.exists(rolling)
    snap = torch.load(rolling, map_location="cpu", weights_only=False)
    assert snap["epoch"] == 1
    assert snap["optimizer"] is not None
    exp0 = snap["optimizer"]["state"][0]["momentum_buffer"]

    # SESSION DIES. Everything below is a fresh process simulation: new
    # model object, new optimizer, roadmap re-read from disk.
    with open(roadmap_path) as f:
        roadmap2 = json.load(f)

    # ── session 2: the machine must say STEP 2, still training ────────────
    action = get_next_action(roadmap2)
    assert action.phase == "recurrence_only", (
        f"restart resumed {action.phase!r} — must resume step 2 "
        f"(recurrence_only), not step 1 and not step 3")
    assert action.substep == "running"

    # Resume gate (Agent A): loads the rolling checkpoint (no HF).
    from noesis_vision.core.checkpoint import resume_or_abort
    state = resume_or_abort(rolling)
    assert state is not None and int(state["epoch"]) == 1

    m2 = _fresh_model(cfg, "recurrence_only")
    m2.load_state_dict(state["model"])
    registry = _build_registry(m2)
    optimizer = registry.build_optimizer(cfg.lr)
    assert registry.resume_guard(state["optimizer"]), \
        "optimizer resume guard refused a SAME-layout state"
    optimizer.load_state_dict(state["optimizer"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=cfg.epochs)
    scheduler.load_state_dict(state["scheduler"])
    # Momentum buffers survived (the 'optimizer state intact' assertion).
    got0 = optimizer.state_dict()["state"][0]["momentum_buffer"]
    assert torch.allclose(got0, exp0)

    # And training CONTINUES (epoch 2), not from scratch:
    m2, _, _ = _train_n_epochs(m2, loaders, cfg, device, n=1, start_epoch=1)
    snap2 = torch.load(rolling, map_location="cpu", weights_only=False)
    assert snap2["epoch"] == 2

    # ── finish step 2; the machine then moves to step 3 — never skips ─────
    advance("recurrence_only", "done", roadmap_path=roadmap_path)
    assert get_next_action(
        json.load(open(roadmap_path))).phase == "belief_no_f"


def _build_registry(model):
    from noesis_vision.core.multi_group_optimizer import (
        OptimizerGroupRegistry)
    registry = OptimizerGroupRegistry()
    groups = model.group_params()
    registry.register_backbone(groups["backbone"])
    for name in ("classifier", "evidential_head", "predictor", "update_net",
                 "precision"):
        if name in groups:
            registry.register(name, groups[name])
    return registry


def test_machine_never_repeats_or_skips_phases(tmp_path):
    """Walking the machine through all four phases yields exactly the J1
    phase list in order, and 'done' at the end."""
    roadmap_path = str(tmp_path / "rm.json")
    roadmap: dict = {}
    ensure_foundation_state(roadmap)
    _write(roadmap, roadmap_path)
    seen = []
    for _ in range(len(FOUNDATION_PHASES) + 1):
        action = get_next_action(json.loads(json.dumps(roadmap)))
        if action.phase is None:
            break
        assert action.phase == FOUNDATION_PHASES[len(seen)], \
            "the machine walked phases out of order"
        seen.append(action.phase)
        roadmap = advance(action.phase, "done", roadmap_path=roadmap_path)
    assert seen == FOUNDATION_PHASES
    assert get_next_action(
        json.load(open(roadmap_path))).substep == "done"


def test_resume_guard_refuses_changed_layout(tmp_path):
    """A rolling checkpoint written under a DIFFERENT phase's group layout
    must be refused, not silently restored (Agent A's identity rule)."""
    cfg = _cfg(tmp_path)
    m_belief = _fresh_model(cfg, "belief_no_f")
    registry = _build_registry(m_belief)
    optimizer = registry.build_optimizer(cfg.lr)
    from noesis_vision.core.checkpoint import save_rolling
    rolling = os.path.join(cfg.ckpt_dir,
                           "foundation_recurrence_only_rolling.pth")
    save_rolling(rolling, epoch=1, model=m_belief, optimizer=optimizer)

    m_step2 = _fresh_model(cfg, "recurrence_only")   # fewer groups
    registry2 = _build_registry(m_step2)
    state = torch.load(rolling, map_location="cpu", weights_only=False)
    assert registry2.resume_guard(state["optimizer"]) is False
