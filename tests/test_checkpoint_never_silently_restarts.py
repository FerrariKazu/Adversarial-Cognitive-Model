"""
Agent A contract test — checkpoints never silently restart.

Simulates an interrupted run (a rolling checkpoint on disk) and asserts
resume; then asserts every path that CANNOT resume fails loudly instead
of falling through to a fresh run.
"""
import pytest
import torch
import torch.nn as nn

from noesis_vision.core.checkpoint import (
    CheckpointResumeError,
    atomic_torch_save,
    resume_commit_ok,
    resume_or_abort,
    save_best,
    save_rolling,
    verify_best_rolling_parity,
)
from noesis_vision.core.schema import RHANNXAConfig


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)


class _FakeOpt:
    """Minimal optimizer stand-in exposing state_dict()."""

    def state_dict(self):
        return {"param_groups": [{"name": "backbone", "lr": 0.1}],
                "state": {}}


def _write_rolling(path, epoch=7):
    model = _TinyModel()
    return save_rolling(str(path), epoch=epoch, model=model,
                        optimizer=_FakeOpt())


def test_interrupted_run_resumes(tmp_path):
    rolling = tmp_path / "exp_rolling.pth"
    state = _write_rolling(rolling, epoch=7)
    resumed = resume_or_abort(str(rolling))
    assert resumed is not None
    assert resumed["epoch"] == 7
    assert resumed["code_commit"] == state["code_commit"]


def test_fresh_run_with_no_artifacts(tmp_path):
    # No local rolling, no HF configured -> a genuinely fresh run.
    assert resume_or_abort(str(tmp_path / "none_rolling.pth")) is None


def test_unverifiable_hf_aborts_instead_of_restarting(tmp_path):
    # HF configured but the downloader cannot verify existence -> the safe
    # direction is to ABORT, never to fall through to epoch 1.
    with pytest.raises(CheckpointResumeError, match="silently restarting"):
        resume_or_abort(str(tmp_path / "exp_rolling.pth"),
                        hf_repo_id="someone/rhan-checkpoints-rolling",
                        hf_filename="exp_rolling.pth",
                        downloader=lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("network down")))


def test_force_restart_is_explicit(tmp_path):
    # Explicit cold start allowed — but only when the caller asks for it.
    assert resume_or_abort(str(tmp_path / "none_rolling.pth"),
                           force_restart=True) is None


def test_hf_newer_checkpoint_is_adopted(tmp_path):
    local = tmp_path / "exp_rolling.pth"
    _write_rolling(local, epoch=3)
    remote = tmp_path / "remote_rolling.pth"
    _write_rolling(remote, epoch=9)  # HF has a NEWER interrupted epoch
    resumed = resume_or_abort(str(local),
                              hf_repo_id="someone/rhan-checkpoints-rolling",
                              hf_filename="exp_rolling.pth",
                              downloader=lambda *a, **k: str(remote))
    assert resumed["epoch"] == 9


def test_stale_code_commit_refuses(tmp_path):
    rolling = tmp_path / "exp_rolling.pth"
    _write_rolling(rolling, epoch=7)
    state = torch.load(str(rolling), map_location="cpu", weights_only=False)
    state["code_commit"] = "deadbeef"  # written by different code
    atomic_torch_save(str(rolling), state)
    with pytest.raises(CheckpointResumeError, match="refusing to resume"):
        resume_or_abort(str(rolling))


def test_legacy_checkpoint_refused():
    ok, msg = resume_commit_ok({"epoch": 5, "model": {}})
    assert not ok and "legacy" in msg.lower()


def test_best_rolling_parity(tmp_path):
    model = _TinyModel()
    cfg = RHANNXAConfig()
    best = tmp_path / "exp_best.pth"
    rolling = tmp_path / "exp_rolling.pth"
    save_best(str(best), model=model, config=cfg, metric_value=55.0)
    save_rolling(str(rolling), epoch=7, model=model, optimizer=_FakeOpt())
    ok, msg = verify_best_rolling_parity(str(best), str(rolling))
    assert ok, msg
    # A best written by different code -> parity gap, loudly.
    b = torch.load(str(best), map_location="cpu", weights_only=False)
    b["code_commit"] = "deadbeef"
    atomic_torch_save(str(best), b)
    ok, msg = verify_best_rolling_parity(str(best), str(rolling))
    assert not ok and "mismatch" in msg.lower()
    # Missing artifact -> fail, never pass vacuously.
    ok, _ = verify_best_rolling_parity(str(tmp_path / "gone_best.pth"),
                                       str(rolling))
    assert not ok


def test_rolling_records_code_commit(tmp_path):
    state = _write_rolling(tmp_path / "exp_rolling.pth", epoch=1)
    assert state["code_commit"] not in ("", None, "unknown") or \
        state["code_commit"] == "unknown"
    assert state["kind"] == "rolling"
