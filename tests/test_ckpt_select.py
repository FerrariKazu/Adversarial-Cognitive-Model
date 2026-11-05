import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloud_setup.ckpt_select import (
    CandidateError,
    can_overwrite_remote,
    files_for_active_runs,
    load_metadata,
    select_best_candidate,
    validate_candidate,
    validate_reuse_candidate,
)

COMMIT = "b8491610220a5cbea41cade12487d6683b5b7e30"
SOURCE_HASHES = {"training/train_gen2_foundation.py": "a" * 64}


def make_state(run_name="ff_112_seed17_b64", phase="ff_112", epoch=15, **overrides):
    state = {
        "run_name": run_name,
        "phase": phase,
        "epoch": epoch,
        "seed": 17,
        "commit_hash": COMMIT[:12],
        "source_hashes": dict(SOURCE_HASHES),
        "optimizer": {"state": {0: {"step": 10}}, "param_groups": [{"lr": 1e-4}]},
        "scheduler": {"last_epoch": 15},
    }
    state.update(overrides)
    return state


def test_weights_only_checkpoint_rejected():
    state = make_state(optimizer={"state": {}, "param_groups": [{"lr": 1e-4}]})
    with pytest.raises(CandidateError, match="empty"):
        validate_candidate(state, expected_run="ff_112_seed17_b64", expected_phase="ff_112")


def test_missing_internal_run_name_rejected():
    state = make_state()
    del state["run_name"]
    with pytest.raises(CandidateError, match="run_name"):
        validate_candidate(state, expected_run="ff_112_seed17_b64", expected_phase="ff_112")


def test_higher_epoch_valid_chosen_over_lower():
    low = {"metadata": make_state(epoch=4), "source": "HF_BEST", "sha256": "1" * 64}
    high = {"metadata": make_state(epoch=15), "source": "HF_ROLLING", "sha256": "2" * 64}
    selected = select_best_candidate([low, high])
    assert selected["metadata"]["epoch"] == 15
    assert selected["source"] == "HF_ROLLING"


def test_ambiguous_same_epoch_rejected():
    a = {"metadata": make_state(epoch=15), "source": "HF_BEST", "sha256": "1" * 64}
    b = {"metadata": make_state(epoch=15), "source": "HF_ROLLING", "sha256": "2" * 64}
    with pytest.raises(CandidateError, match="ambiguous"):
        select_best_candidate([a, b])


def test_partial_download_ignored(tmp_path):
    part = tmp_path / "foundation_ff_112_seed17_b64_rolling.pth.part"
    part.write_text("not a checkpoint")
    with pytest.raises(CandidateError, match="partial"):
        load_metadata(part)
    assert files_for_active_runs([part.name], ["ff_112_seed17_b64"]) == []


def test_scoped_sync_refuses_downgrade():
    assert can_overwrite_remote(4, 15) is False
    assert can_overwrite_remote(16, 15) is True
    assert can_overwrite_remote(1, None) is True


def test_sync_is_scoped_to_active_runs():
    files = [
        "foundation_ff_112_seed17_b64_rolling.pth",
        "foundation_ff_112_seed99_b64_final.pth",
        "foundation_recurrence_112_seed17_b64_final.pth",
        "foundation_ff_112_seed17_rolling.pth",
    ]
    assert files_for_active_runs(files, ["ff_112_seed17_b64"]) == [
        "foundation_ff_112_seed17_b64_rolling.pth"
    ]


def test_commit_short_prefix_matches_pinned_full_sha():
    validate_candidate(
        make_state(),
        expected_run="ff_112_seed17_b64",
        expected_phase="ff_112",
        expected_commit=COMMIT,
    )
    with pytest.raises(CandidateError, match="commit_hash mismatch"):
        validate_candidate(
            make_state(commit_hash="0" * 12),
            expected_run="ff_112_seed17_b64",
            expected_phase="ff_112",
            expected_commit=COMMIT,
        )


def test_missing_source_hashes_rejected():
    with pytest.raises(CandidateError, match="source_hashes"):
        validate_candidate(
            make_state(source_hashes={}),
            expected_run="ff_112_seed17_b64",
            expected_phase="ff_112",
        )


def test_reuse_candidate_accepts_legacy_provenance():
    legacy = {
        "phase": "consolidated_112",
        "epoch": 15,
        "optimizer": {"state": {0: {"step": 10}}, "param_groups": [{"lr": 1e-4}]},
        "scheduler": {"last_epoch": 15},
    }
    assert validate_reuse_candidate(legacy, expected_phase="consolidated_112")["epoch"] == 15


def test_load_metadata_roundtrip(tmp_path):
    path = tmp_path / "foundation_ff_112_seed17_b64_rolling.pth"
    torch.save(make_state(), path)
    state = load_metadata(path)
    assert state["run_name"] == "ff_112_seed17_b64"
    assert state["epoch"] == 15
