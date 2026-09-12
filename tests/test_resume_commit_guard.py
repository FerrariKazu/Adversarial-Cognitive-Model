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

from checkpoint_utils import (current_code_commit, resume_commit_ok,
                              training_fingerprint)


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


# ── Training fingerprint (2026-09-12) ────────────────────────────────────────
# Real, immutable repo history (do NOT rewrite these SHAs casually — they pin
# the exact incidents the fingerprint policy resolves):
#   8abb343  last commit touching training math before the fingerprint guard
#   54c3984  sbr1 gate fix            (notebooks + scripts + docs only)
#   47d30b2  Kaggle/Colab parity      (notebooks only)

def test_notebook_only_commits_share_fingerprint():
    # 54c3984 -> 47d30b2 changed only cloud_setup/, docs/, scripts/, tests/:
    # both must resolve to the same training fingerprint (8abb343).
    assert training_fingerprint("54c3984") == training_fingerprint("47d30b2")
    assert training_fingerprint("47d30b2") == "8abb343"


def test_sbr2_incident_is_now_resumable():
    # THE incident: SBR-2's rolling checkpoint was written by 54c3984; the
    # next session checked out 47d30b2 and the raw-commit guard demanded
    # deleting ~6h of real progress. Under fingerprint semantics this is a
    # legitimate resume (explicit currents — no dependence on live HEAD).
    ok, msg = resume_commit_ok({"code_commit": "54c3984", "epoch": 3},
                               current="47d30b2")
    assert ok, msg


def test_stamped_fingerprint_is_trusted():
    # New checkpoints carry 'training_fingerprint'; resume must honor the
    # stamp rather than re-resolving git (works even outside a git repo).
    ok, msg = resume_commit_ok({"code_commit": "54c3984",
                                "training_fingerprint": "8abb343"},
                               current="47d30b2")
    assert ok, msg


def test_genuine_math_change_still_refuses():
    # A checkpoint from 8abb343's training code is NOT resumable under
    # b5b8b68-era code: their training fingerprints differ (b5b8b68 predates
    # 8abb343's math changes).
    ok, msg = resume_commit_ok({"code_commit": "8abb343", "epoch": 9},
                               current="b5b8b68")
    assert not ok, msg
    assert "refusing to resume" in msg.lower()


def test_unknown_commit_refuses_safe_direction():
    # An unresolvable commit (squashed away, foreign clone) resolves to
    # itself and therefore never matches an older checkpoint's fingerprint —
    # refusal is the safe direction.
    ok, msg = resume_commit_ok({"code_commit": "8abb343"}, current="ffffff0")
    assert not ok, msg


def test_fingerprint_message_distinguishes_notebook_resume():
    ok, msg = resume_commit_ok({"code_commit": "54c3984"}, current="47d30b2")
    assert ok
    assert "notebooks only" in msg.lower()
