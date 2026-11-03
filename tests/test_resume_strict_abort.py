import pytest
import torch

from training.train_gen2_foundation import run_gen2_phase, Gen2FoundationModel


def _make_model_state():
    model = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112)
    return model.state_dict()


def test_resume_aborts_on_mismatched_optimizer(tmp_path):
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


def test_resume_aborts_on_scheduler_missing(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    report_dir = tmp_path / "reports"
    ckpt_dir.mkdir()
    report_dir.mkdir()
    rolling = ckpt_dir / "foundation_ff_112_rolling.pth"
    from training.train_gen2_foundation import build_gen2_optimizer
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


def test_resume_aborts_on_stale_keys(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    report_dir = tmp_path / "reports"
    ckpt_dir.mkdir()
    report_dir.mkdir()
    rolling = ckpt_dir / "foundation_ff_112_rolling.pth"
    model = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112)
    bad_model = dict(model.state_dict())
    # remove a key to make stale
    for k in list(bad_model.keys())[:1]:
        bad_model.pop(k)
        break
    from training.train_gen2_foundation import build_gen2_optimizer
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
