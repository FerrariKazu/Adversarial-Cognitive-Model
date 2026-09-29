"""Integration tests for the 2026-09-29 recipe correction in run_phase.

Proves, at the run_phase level (quarantined dirs, synthetic loaders, no HF):
  1. the DEFAULT config trains ADVERSARIALLY (TRADES/PGD leg executes) and
     records the recipe in the provenance manifest + embedded best config;
  2. the silent-inheritance guard ABORTS a no-rolling cold start whose best
     checkpoint is bitwise-identical to its parent's (the 2026-09-25->26
     gen1_core failure mode), and STAYS SILENT when the checkpoints differ;
  3. --force-fresh remains a legitimate audible cold start (guard passes).
"""
import json
import os
import shutil
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig,
    FoundationModel,
    build_model,
    run_phase,
)
from noesis_vision.core.checkpoint import save_best  # noqa: E402


@pytest.fixture()
def quarantine(tmp_path, monkeypatch):
    """Point every artifact root into tmp_path; synthetic loaders."""
    cfg = FoundationConfig()
    cfg.ckpt_dir = str(tmp_path / "checkpoints")
    cfg.report_dir = str(tmp_path / "report")
    cfg.runs_dir = str(tmp_path / "runs")
    cfg.use_hf = False
    cfg.hf_token = None
    cfg.epochs = 1
    cfg.batch_size = 4
    cfg.num_workers = 0
    cfg.eval_seeds = (0,)
    cfg.n_eval_samples = 8
    cfg.pgd_steps = 1
    cfg.amp = False
    cfg.smoke = False                      # real-ish path: guard ACTIVE
    for d in (cfg.ckpt_dir, cfg.report_dir, cfg.runs_dir):
        os.makedirs(d, exist_ok=True)      # the real tree pre-creates these
    return cfg, tmp_path


def _fake_loaders(cfg):
    """Synthetic loaders handed DIRECTLY to run_phase (main() owns the
    loader factory; run_phase just consumes the dict)."""
    class _L:
        def __init__(self):
            self.dataset = [0] * 8          # eval subset source

        def __iter__(self):
            yield (torch.randn(2, 3, cfg.img_size, cfg.img_size),
                   torch.randint(0, cfg.num_classes, (2,)))
    return {"train": _L(), "val": _L()}


def _run(cfg, phase, device, monkeypatch):
    monkeypatch.setattr(
        "training.train_generation1_foundation.run_clean_and_robust",
        lambda **kw: {"per_seed_csv": os.path.join(kw["out_dir"],
                                                   "per_seed.csv"),
                      "summary_csv": os.path.join(kw["out_dir"],
                                                  "summary.csv"),
                      "provenance": {}})
    monkeypatch.setattr(
        "training.train_generation1_foundation.compactness_report",
        lambda model, **kw: {"params_total": 1, "params_trainable": 1,
                             "est_macs_per_image": 1})
    return run_phase(phase, cfg, _fake_loaders(cfg), device)


def _assert_manifest_recipe(runs_dir, phase, cfg):
    # write_manifest() merges `extra` at the TOP level (manifest.update).
    with open(os.path.join(runs_dir, f"foundation_{phase}",
                           "manifest.json")) as f:
        man = json.load(f)
    recipe = man["adv_curriculum"]
    assert recipe["recipe_version"] == cfg.recipe_version
    assert recipe["w_trades"] == cfg.w_trades
    assert recipe["clean_only"] == cfg.clean_only
    assert recipe["eps_ramp"] == [0.031, 0.062, 0.094]
    return man


# ── 1. The default recipe is ADVERSARIAL, and it is recorded ────────────────
def test_default_config_trains_adversarially(quarantine, monkeypatch):
    cfg, tmp = quarantine
    assert cfg.clean_only is False          # the whole point of the fix
    device = torch.device("cpu")
    events = []
    real_toe = __import__(
        "training.train_generation1_foundation",
        fromlist=["train_one_epoch"]).train_one_epoch

    def spy_toe(model, loader, *a, **kw):
        events.append(("clean_only", kw.get("clean_only")))
        events.append(("epoch", kw.get("epoch")))
        return real_toe(model, loader, *a, **kw)

    monkeypatch.setattr(
        "training.train_generation1_foundation.train_one_epoch", spy_toe)
    result = _run(cfg, "backbone_only", device, monkeypatch)
    assert events == [("clean_only", False), ("epoch", 1)]
    _assert_manifest_recipe(str(tmp / "runs"), "backbone_only", cfg)
    # The best checkpoint embeds the recipe-carrying config (Gen-0
    # convention: eval reconstructs the exact configuration).
    best = torch.load(os.path.join(cfg.ckpt_dir,
                                   "foundation_backbone_only_best.pth"),
                      map_location="cpu", weights_only=False)
    assert best["config"]["clean_only"] is False
    assert best["config"]["w_trades"] == 0.55
    # run completed end-to-end with the recipe active
    assert result["best_val_acc"] == 0.0


def test_manifest_records_eps_ramp(quarantine, monkeypatch):
    cfg, tmp = quarantine
    _run(cfg, "recurrence_only", torch.device("cpu"), monkeypatch)
    man = _assert_manifest_recipe(str(tmp / "runs"), "recurrence_only", cfg)
    assert man["adv_curriculum"]["beta_ramp"] == [2.0, 2.0, 2.5]


# ── 2. The silent-inheritance guard ─────────────────────────────────────
def _make_best(cfg, phase, metric=0.5):
    """A real best checkpoint for `phase` (fresh-init model weights)."""
    model = build_model(cfg, phase)
    save_best(os.path.join(cfg.ckpt_dir, f"foundation_{phase}_best.pth"),
              model=model, config=cfg.to_dict(), metric_value=metric)


def _clone_parent_best_as(cfg, parent, child):
    """The failure mode: child's best file holds the parent's model payload
    BITWISE (the 2026-09-25->26 gen1_core artifact), with a valid
    code_commit so only payload equality distinguishes it."""
    from noesis_vision.core.checkpoint import (atomic_torch_save,
                                               current_code_commit)
    parent_sd = torch.load(
        os.path.join(cfg.ckpt_dir, f"foundation_{parent}_best.pth"),
        map_location="cpu", weights_only=False)["model"]
    atomic_torch_save(
        os.path.join(cfg.ckpt_dir, f"foundation_{child}_best.pth"),
        {"model": parent_sd, "config": cfg.to_dict(),
         "metric_value": 0.5, "kind": "best",
         "code_commit": current_code_commit(), "saved_at_utc": "t"})


def test_guard_aborts_bitwise_identical_inheritance(quarantine, monkeypatch):
    cfg, tmp = quarantine
    device = torch.device("cpu")
    _make_best(cfg, "ais_v2_swap")     # the DECLARED parent (DEPENDENCIES)
    _clone_parent_best_as(cfg, "ais_v2_swap", "gen1_core")
    with pytest.raises(SystemExit, match="silent-inheritance"):
        _run(cfg, "gen1_core", device, monkeypatch)


def test_guard_passes_when_checkpoints_differ(quarantine, monkeypatch):
    cfg, tmp = quarantine
    device = torch.device("cpu")
    _make_best(cfg, "ais_v2_swap")
    _make_best(cfg, "gen1_core")       # fresh init -> different weights
    result = _run(cfg, "gen1_core", device, monkeypatch)
    assert result["phase"] == "gen1_core"


def test_guard_ignores_first_phase(quarantine, monkeypatch):
    cfg, tmp = quarantine
    # backbone_only has no parent; even a weird best file must not trip it.
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save({"model": {}},
               os.path.join(cfg.ckpt_dir, "foundation_backbone_only_best.pth"))
    result = _run(cfg, "backbone_only", torch.device("cpu"), monkeypatch)
    assert result["phase"] == "backbone_only"


def test_guard_skipped_in_smoke(quarantine, monkeypatch):
    cfg, tmp = quarantine
    cfg.smoke = True
    _make_best(cfg, "ais_v2_swap")
    _clone_parent_best_as(cfg, "ais_v2_swap", "gen1_core")
    result = _run(cfg, "gen1_core", torch.device("cpu"), monkeypatch)
    assert result["phase"] == "gen1_core"   # smoke: guard off, run proceeds


# ── 3. --force-fresh remains a legitimate cold start ────────────────────────
def test_force_fresh_passes_the_guard(quarantine, monkeypatch):
    cfg, tmp = quarantine
    device = torch.device("cpu")
    _make_best(cfg, "ais_v2_swap")
    _clone_parent_best_as(cfg, "ais_v2_swap", "gen1_core")
    # force-fresh deletes best/rolling/manifest LOUDLY, then runs — the
    # guard must find nothing to compare and let the cold start proceed.
    cfg.force_fresh = True
    result = _run(cfg, "gen1_core", device, monkeypatch)
    assert result["phase"] == "gen1_core"
