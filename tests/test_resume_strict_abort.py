import json

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from training.train_gen2_foundation import (
    Gen2FoundationModel,
    build_gen2_optimizer,
    run_gen2_phase,
)


class DummyDataset(Dataset):
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.randn(3, 112, 112)
        y = torch.tensor(i % 100)
        return x, y


def _patch_loaders(monkeypatch):
    def fake_get_gen2_loaders(data_root, batch_size=48, num_workers=4, smoke=False, img_size=112):
        ds = DummyDataset(n=8)
        return (
            DataLoader(ds, batch_size=2, shuffle=False),
            DataLoader(ds, batch_size=2, shuffle=False),
            DataLoader(ds, batch_size=2, shuffle=False),
            DataLoader(ds, batch_size=2, shuffle=False),
        )

    monkeypatch.setattr(
        "training.train_gen2_foundation.get_gen2_loaders", fake_get_gen2_loaders
    )

    def fake_load_dino(self, path):
        return {"accepted": 0, "matched_trunk_keys": 0, "trunk_keys": 0, "matched_fraction": 0.0}

    import gen2_foundation.backbone as backbone_mod

    monkeypatch.setattr(backbone_mod.CompactViTGen2, "load_dino_warm_start", fake_load_dino)


def _model():
    return Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112)


def _ckpt_dirs(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    report_dir = tmp_path / "reports"
    ckpt_dir.mkdir()
    report_dir.mkdir()
    return ckpt_dir, report_dir


def _write_rolling(ckpt_dir, name, **overrides):
    model = _model()
    opt, sched = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    state = {
        "phase": "ff_112",
        "arm": "v1",
        "epoch": 5,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "best_clean_acc": 0.0,
        "best_rob_acc": 0.0,
    }
    state.update(overrides)
    torch.save(state, ckpt_dir / name)


def _run(tmp_path, ckpt_dir, report_dir, allow=False, phase_override=None, n_epochs=None):
    kwargs = dict(
        phase="ff_112",
        arm="v1",
        data_root=str(tmp_path),
        ckpt_dir=str(ckpt_dir),
        report_dir=str(report_dir),
        dino_path="checkpoints/dinov2_vits14_pretrain.pth",
        smoke=True,
        allow_fresh_optimizer=allow,
    )
    if phase_override is not None:
        kwargs["phase_override_name"] = phase_override
    if n_epochs is not None:
        kwargs["n_epochs"] = n_epochs
    return run_gen2_phase(**kwargs)


def _expect_aborts_twice(tmp_path, ckpt_dir, report_dir):
    with pytest.raises(RuntimeError):
        _run(tmp_path, ckpt_dir, report_dir)
    assert (ckpt_dir / "foundation_ff_112_rolling.pth").exists()
    # A second identical launch must abort again (no silent restart from epoch 1).
    with pytest.raises(RuntimeError):
        _run(tmp_path, ckpt_dir, report_dir)
    assert (ckpt_dir / "foundation_ff_112_rolling.pth").exists()


def test_resume_aborts_on_mismatched_optimizer(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    _write_rolling(
        ckpt_dir,
        "foundation_ff_112_rolling.pth",
        optimizer={"param_groups": [{"lr": 0.1, "params": []}], "state": {}},
    )
    _expect_aborts_twice(tmp_path, ckpt_dir, report_dir)


def test_resume_aborts_on_scheduler_present_but_load_fails(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    _write_rolling(ckpt_dir, "foundation_ff_112_rolling.pth", scheduler={"bad": True})
    _expect_aborts_twice(tmp_path, ckpt_dir, report_dir)


def test_resume_aborts_on_scheduler_missing(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    model = _model()
    opt, _ = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    state = {
        "phase": "ff_112",
        "arm": "v1",
        "epoch": 5,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
    }
    torch.save(state, ckpt_dir / "foundation_ff_112_rolling.pth")
    _expect_aborts_twice(tmp_path, ckpt_dir, report_dir)


def test_resume_aborts_on_optimizer_missing(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    model = _model()
    _, sched = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    state = {
        "phase": "ff_112",
        "arm": "v1",
        "epoch": 5,
        "model": model.state_dict(),
        "scheduler": sched.state_dict(),
    }
    torch.save(state, ckpt_dir / "foundation_ff_112_rolling.pth")
    _expect_aborts_twice(tmp_path, ckpt_dir, report_dir)
    assert not (ckpt_dir / "foundation_ff_112_rolling.pth.stale").exists()


def test_resume_aborts_on_stale_keys(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    model = _model()
    bad_model = dict(model.state_dict())
    bad_model.pop(list(bad_model.keys())[0])
    _write_rolling(ckpt_dir, "foundation_ff_112_rolling.pth", model=bad_model)
    _expect_aborts_twice(tmp_path, ckpt_dir, report_dir)
    # The abort must not consume/rename the rolling checkpoint.
    assert not (ckpt_dir / "foundation_ff_112_rolling.pth.stale").exists()


def test_resume_allows_fresh_optimizer_with_fallback_event(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    model = _model()
    opt, _ = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    state = {
        "phase": "ff_112",
        "arm": "v1",
        "epoch": 5,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
    }
    torch.save(state, ckpt_dir / "foundation_ff_112_rolling.pth")
    _run(tmp_path, ckpt_dir, report_dir, allow=True)
    log_path = report_dir / "gen2_epoch_log.jsonl"
    assert log_path.exists()
    with open(log_path) as f:
        rec = json.loads(f.readline())
    types = [ev["type"] for ev in rec["fallback_events"]]
    assert "scheduler_missing" in types


def _empty_optimizer_state():
    model = _model()
    opt, _ = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    state = opt.state_dict()
    assert state["state"] == {}
    return state


def _optimizer_state_with_history():
    model = _model()
    opt, _ = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    for group in opt.param_groups:
        for p in group["params"]:
            p.grad = torch.zeros_like(p)
    opt.step()
    opt.zero_grad(set_to_none=True)
    state = opt.state_dict()
    assert state["state"]
    return state


def test_resume_aborts_on_empty_optimizer_state(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    _write_rolling(
        ckpt_dir,
        "foundation_ff_112_rolling.pth",
        optimizer=_empty_optimizer_state(),
    )
    _expect_aborts_twice(tmp_path, ckpt_dir, report_dir)
    assert not (ckpt_dir / "foundation_ff_112_rolling.pth.stale").exists()


def test_resume_allows_empty_optimizer_state_with_fallback_event(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    _write_rolling(
        ckpt_dir,
        "foundation_ff_112_rolling.pth",
        optimizer=_empty_optimizer_state(),
    )
    _run(tmp_path, ckpt_dir, report_dir, allow=True, n_epochs=6)
    log_path = report_dir / "gen2_epoch_log.jsonl"
    assert log_path.exists()
    with open(log_path) as f:
        rec = json.loads(f.readline())
    types = [ev["type"] for ev in rec["fallback_events"]]
    assert "optimizer_state_empty" in types


def test_clean_resume_with_optimizer_state_no_fallback_events(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    _write_rolling(
        ckpt_dir,
        "foundation_ff_112_rolling.pth",
        optimizer=_optimizer_state_with_history(),
    )
    _run(tmp_path, ckpt_dir, report_dir, n_epochs=6)
    log_path = report_dir / "gen2_epoch_log.jsonl"
    assert log_path.exists()
    with open(log_path) as f:
        rec = json.loads(f.readline())
    assert rec["fallback_events"] == []


def test_resume_allows_missing_optimizer_with_fallback_event(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    model = _model()
    _, sched = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    state = {
        "phase": "ff_112",
        "arm": "v1",
        "epoch": 5,
        "model": model.state_dict(),
        "scheduler": sched.state_dict(),
    }
    torch.save(state, ckpt_dir / "foundation_ff_112_rolling.pth")
    _run(tmp_path, ckpt_dir, report_dir, allow=True)
    log_path = report_dir / "gen2_epoch_log.jsonl"
    assert log_path.exists()
    with open(log_path) as f:
        rec = json.loads(f.readline())
    types = [ev["type"] for ev in rec["fallback_events"]]
    assert "optimizer_missing" in types


def test_clean_resume_no_fallback_events(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    _write_rolling(ckpt_dir, "foundation_ff_112_rolling.pth", epoch=0)
    _run(tmp_path, ckpt_dir, report_dir)
    log_path = report_dir / "gen2_epoch_log.jsonl"
    assert log_path.exists()
    with open(log_path) as f:
        rec = json.loads(f.readline())
    assert rec["fallback_events"] == []


def test_checkpoints_record_effective_run_name(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    eff = "recurrence_112_seed17"
    _run(tmp_path, ckpt_dir, report_dir, phase_override=eff, n_epochs=1)
    for name in (f"foundation_{eff}_rolling.pth", f"foundation_{eff}_final.pth"):
        state = torch.load(ckpt_dir / name, map_location="cpu", weights_only=False)
        assert state["run_name"] == eff


def test_already_completed_branch_uses_effective_name(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir, report_dir = _ckpt_dirs(tmp_path)
    eff = "recurrence_112_seed17"
    _write_rolling(
        ckpt_dir,
        f"foundation_{eff}_rolling.pth",
        epoch=1,
        optimizer=_optimizer_state_with_history(),
    )
    # Sentinel: the arch-phase-named final must not be created or touched.
    sentinel = ckpt_dir / "foundation_ff_112_final.pth"
    sentinel.write_bytes(b"untouched")

    result = _run(
        tmp_path, ckpt_dir, report_dir, phase_override=eff, n_epochs=1
    )

    assert result["status"] == "complete"
    assert result["phase"] == eff
    assert (ckpt_dir / f"foundation_{eff}_final.pth").exists()
    assert sentinel.read_bytes() == b"untouched"
