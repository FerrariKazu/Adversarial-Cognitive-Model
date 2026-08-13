"""Resume-commit guard: a checkpoint written by DIFFERENT code must never be resumed.

2026-08-12 regression: the Stage 2 HPC smoke resumed a stale LOCAL rolling
checkpoint (old dead-head code, epoch 11) inside a re-pasted Colab runtime.
The new head fix was never instantiated; epochs 12-15 trained the old
saturated weights against the new target range and the health gate produced a
meaningless DEGENERATE verdict. The guard makes that class of error fatal:
every checkpoint records 'code_commit' (git HEAD) and resume refuses any
mismatch — including pre-guard legacy checkpoints, which by definition come
from older code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase1_training"))

from checkpoint_utils import current_code_commit, resume_commit_ok


def test_current_code_commit_is_a_sha():
    cc = current_code_commit()
    assert isinstance(cc, str), cc
    assert cc != "" and cc != "unknown", cc
    assert len(cc) >= 7, cc


def test_matching_commit_allows_resume():
    cc = current_code_commit()
    ok, msg = resume_commit_ok({"code_commit": cc}, cc)
    assert ok, msg
    assert "resumable" in msg.lower()


def test_mismatched_commit_refuses():
    ok, msg = resume_commit_ok({"code_commit": "deadbeef"}, "1111111")
    assert not ok, msg
    assert "refusing to resume" in msg.lower()


def test_legacy_checkpoint_without_commit_refuses():
    # Pre-guard checkpoints have no 'code_commit' key — by definition older
    # than this guard, so resuming them is refused, never silently accepted.
    ok, msg = resume_commit_ok({"epoch": 5, "model": {}, "config": {}}, None)
    assert not ok, msg
    assert "legacy" in msg.lower()


def test_non_dict_checkpoint_refuses():
    ok, _ = resume_commit_ok(None, "anything")
    assert not ok


def test_roundtrip_matching_explicit_current():
    # A fresh-run checkpoint saved by the trainer must pass when the same
    # code resumes it (the legitimate session-continuation path).
    cc = current_code_commit()
    saved = {"code_commit": cc, "epoch": 7}
    ok, msg = resume_commit_ok(saved, cc)
    assert ok, msg
