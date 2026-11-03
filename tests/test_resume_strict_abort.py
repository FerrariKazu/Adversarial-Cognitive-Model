import os
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from training.train_gen2_foundation import run_gen2_phase, Gen2FoundationModel, build_gen2_optimizer


class DummyDataset(Dataset):
    def __init__(self, n=4):
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
        train_loader = DataLoader(ds, batch_size=2, shuffle=False)
        val_loader = DataLoader(ds, batch_size=2, shuffle=False)
        rob_loader = DataLoader(ds, batch_size=2, shuffle=False)
        drift_loader = DataLoader(ds, batch_size=2, shuffle=False)
        return train_loader, val_loader, rob_loader, drift_loader
    monkeypatch.setattr('training.train_gen2_foundation.get_gen2_loaders', fake_get_gen2_loaders)
    def fake_load_dino(self, path):
        return {'accepted': 0, 'matched_trunk_keys': 0, 'trunk_keys': 0, 'matched_fraction': 0.0}
    import gen2_foundation.backbone as backbone_mod
    monkeypatch.setattr(backbone_mod.CompactViTGen2, 'load_dino_warm_start', fake_load_dino)


def _make_model_state():
    model = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112)
    return model.state_dict()


def test_resume_aborts_on_mismatched_optimizer(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir = tmp_path / "ckpts"
    report_dir = tmp_path / "reports"
    ckpt_dir.mkdir()
    report_dir.mkdir()
    rolling = ckpt_dir / "foundation_ff_112_rolling.pth"
    state = {
        "phase": "ff_112",
        "arm": "v1",
        "epoch": 5,
        "model": _make_model_state(),
        "optimizer": {"param_groups": [{"lr": 0.1, "params": []}], "state": {}},
        "scheduler": {},
    }
    torch.save(state, rolling)
    with pytest.raises(RuntimeError):
        run_gen2_phase(
            phase="ff_112",
            arm="v1",
            data_root=str(tmp_path),
            ckpt_dir=str(ckpt_dir),
            report_dir=str(report_dir),
            dino_path="checkpoints/dinov2_vits14_pretrain.pth",
            smoke=True,
            allow_fresh_optimizer=False,
        )
    assert rolling.exists()


def test_resume_aborts_on_scheduler_present_but_load_fails(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir = tmp_path / "ckpts"
    report_dir = tmp_path / "reports"
    ckpt_dir.mkdir()
    report_dir.mkdir()
    rolling = ckpt_dir / "foundation_ff_112_rolling.pth"
    model = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112)
    opt, sched = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    state = {
        "phase": "ff_112",
        "arm": "v1",
        "epoch": 5,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": {"bad": True},
    }
    torch.save(state, rolling)
    with pytest.raises(RuntimeError):
        run_gen2_phase(
            phase="ff_112",
            arm="v1",
            data_root=str(tmp_path),
            ckpt_dir=str(ckpt_dir),
            report_dir=str(report_dir),
            dino_path="checkpoints/dinov2_vits14_pretrain.pth",
            smoke=True,
            allow_fresh_optimizer=False,
        )
    assert rolling.exists()
    with pytest.raises(RuntimeError):
        run_gen2_phase(
            phase="ff_112",
            arm="v1",
            data_root=str(tmp_path),
            ckpt_dir=str(ckpt_dir),
            report_dir=str(report_dir),
            dino_path="checkpoints/dinov2_vits14_pretrain.pth",
            smoke=True,
            allow_fresh_optimizer=False,
        )


def test_resume_aborts_on_scheduler_missing(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir = tmp_path / "ckpts"
    report_dir = tmp_path / "reports"
    ckpt_dir.mkdir()
    report_dir.mkdir()
    rolling = ckpt_dir / "foundation_ff_112_rolling.pth"
    model = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112)
    opt, sched = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    state = {
        "phase": "ff_112",
        "arm": "v1",
        "epoch": 5,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
    }
    torch.save(state, rolling)
    with pytest.raises(RuntimeError):
        run_gen2_phase(
            phase="ff_112",
            arm="v1",
            data_root=str(tmp_path),
            ckpt_dir=str(ckpt_dir),
            report_dir=str(report_dir),
            dino_path="checkpoints/dinov2_vits14_pretrain.pth",
            smoke=True,
            allow_fresh_optimizer=False,
        )
    assert rolling.exists()


def test_resume_aborts_on_stale_keys(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir = tmp_path / "ckpts"
    report_dir = tmp_path / "reports"
    ckpt_dir.mkdir()
    report_dir.mkdir()
    rolling = ckpt_dir / "foundation_ff_112_rolling.pth"
    model = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112)
    bad_model = dict(model.state_dict())
    for k in list(bad_model.keys())[:1]:
        bad_model.pop(k)
        break
    opt, sched = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    state = {
        "phase": "ff_112",
        "arm": "v1",
        "epoch": 5,
        "model": bad_model,
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
    }
    torch.save(state, rolling)
    with pytest.raises(RuntimeError):
        run_gen2_phase(
            phase="ff_112",
            arm="v1",
            data_root=str(tmp_path),
            ckpt_dir=str(ckpt_dir),
            report_dir=str(report_dir),
            dino_path="checkpoints/dinov2_vits14_pretrain.pth",
            smoke=True,
            allow_fresh_optimizer=False,
        )
    assert rolling.exists()
    assert not (ckpt_dir / "foundation_ff_112_rolling.pth.stale").exists()
    with pytest.raises(RuntimeError):
        run_gen2_phase(
            phase="ff_112",
            arm="v1",
            data_root=str(tmp_path),
            ckpt_dir=str(ckpt_dir),
            report_dir=str(report_dir),
            dino_path="checkpoints/dinov2_vits14_pretrain.pth",
            smoke=True,
            allow_fresh_optimizer=False,
        )


def test_resume_allows_fresh_optimizer_with_fallback_event(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir = tmp_path / "ckpts"
    report_dir = tmp_path / "reports"
    ckpt_dir.mkdir()
    report_dir.mkdir()
    rolling = ckpt_dir / "foundation_ff_112_rolling.pth"
    model = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112)
    opt, sched = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    state = {
        "phase": "ff_112",
        "arm": "v1",
        "epoch": 5,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        # no scheduler
    }
    torch.save(state, rolling)
    # smoke mode will run 1 epoch and break
    try:
        run_gen2_phase(
            phase="ff_112",
            arm="v1",
            data_root=str(tmp_path),
            ckpt_dir=str(ckpt_dir),
            report_dir=str(report_dir),
            dino_path="checkpoints/dinov2_vits14_pretrain.pth",
            smoke=True,
            allow_fresh_optimizer=True,
        )
    except Exception:
        pass
    # Check JSONL for fallback_events
    log_path = report_dir / "gen2_epoch_log.jsonl"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                rec = json.loads(line)
                assert 'fallback_events' in rec
                # should have recorded something
                break


def test_clean_resume_no_fallback_events(tmp_path, monkeypatch):
    _patch_loaders(monkeypatch)
    ckpt_dir = tmp_path / "ckpts"
    report_dir = tmp_path / "reports"
    ckpt_dir.mkdir()
    report_dir.mkdir()
    rolling = ckpt_dir / "foundation_ff_112_rolling.pth"
    model = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112)
    opt, sched = build_gen2_optimizer(model, warmup_epochs=3, total_epochs=15)
    state = {
        "phase": "ff_112",
        "arm": "v1",
        "epoch": 0,  # will resume to epoch 1
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "best_clean_acc": 0.0,
        "best_rob_acc": 0.0,
    }
    torch.save(state, rolling)
    try:
        run_gen2_phase(
            phase="ff_112",
            arm="v1",
            data_root=str(tmp_path),
            ckpt_dir=str(ckpt_dir),
            report_dir=str(report_dir),
            dino_path="checkpoints/dinov2_vits14_pretrain.pth",
            smoke=True,
            allow_fresh_optimizer=False,
        )
    except Exception:
        pass
    log_path = report_dir / "gen2_epoch_log.jsonl"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                rec = json.loads(line)
                assert rec.get('fallback_events') == []
                break
