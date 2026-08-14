"""Stage 2 — diag-telemetry epoch-1 baseline carry across resume boundaries.

2026-08-14: the Stage 2 smoke health gate's trend check compares the FIRST
logged epoch against the LAST (rows[0] vs rows[-1] of --diag-json). When a
session resumed from the HF rolling checkpoint (epoch 14 -> 15) and appended
only its own epoch to a fresh diag file, the gate compared epoch 15 to ITSELF
("epoch 15 0.6906 -> epoch 15 0.6906, ratio 1.00") and judged the run "did
NOT decrease >= 10%" — a verdict that said nothing about the optimizer fix
under test (it had one epoch of evidence). This pins the fix: the trainer
stores the run's first-epoch summary in the rolling checkpoint
('first_epoch_diag') and prepends it to --diag-json on resume, so rows[0] is
always the true epoch-1 baseline.

These tests exercise the REAL helper (ensure_diag_baseline from
train_rhan_next.py — never a re-implementation), including the idempotency
that prevents duplicate baselines across repeated resumes.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1_training"))

from train_rhan_next import ensure_diag_baseline  # noqa: E402

_ROW1 = {"epoch": 1, "eps": 0.031, "hpc_error_mean": 0.691234,
         "beta_dyn_mean": 0.5}
_ROW15 = {"epoch": 15, "eps": 0.031, "hpc_error_mean": 0.690600,
          "beta_dyn_mean": 0.5}


def _write(tmp_path, rows):
    p = tmp_path / "diag.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def _rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()
            if l.strip()]


def test_resume_only_diag_gets_epoch1_baseline_prepended(tmp_path):
    """THE regression: a resumed session's diag starts at the resume epoch
    (only epoch 15 was logged). ensure_diag_baseline must prepend the carried
    epoch-1 summary so rows[0] == epoch 1 — the gate's trend reference."""
    p = _write(tmp_path, [_ROW15])
    assert ensure_diag_baseline(str(p), _ROW1) is True
    rows = _rows(p)
    assert [r["epoch"] for r in rows] == [1, 15]
    assert rows[0]["hpc_error_mean"] == pytest.approx(0.691234)


def test_baseline_already_first_is_idempotent(tmp_path):
    """Repeated resumes (same runtime) must not duplicate the baseline — a
    duplicated epoch-1 row would still satisfy the gate but corrupt the
    history the way the 2026-08-12 stale-telemetry pollution did."""
    p = _write(tmp_path, [_ROW1, _ROW15])
    assert ensure_diag_baseline(str(p), _ROW1) is True
    rows = _rows(p)
    assert [r["epoch"] for r in rows] == [1, 15]


def test_missing_diag_file_is_created_with_baseline(tmp_path):
    """Fresh runtime, first resume: the diag file does not exist yet (the
    previous session's Step A never finished its HF upload). The baseline
    must bootstrap the file."""
    p = tmp_path / "diag.jsonl"
    assert ensure_diag_baseline(str(p), _ROW1) is True
    assert [r["epoch"] for r in _rows(p)] == [1]


def test_none_or_non_dict_baseline_is_refused(tmp_path):
    """A checkpoint without 'first_epoch_diag' (legacy) must not fabricate a
    baseline — the gate's own >= 2 distinct-epochs guard reports the run as
    not judgeable instead."""
    p = tmp_path / "diag.jsonl"
    assert ensure_diag_baseline(str(p), None) is False
    assert ensure_diag_baseline(str(p), {"hpc_error_mean": 0.69}) is False
    assert not p.exists()
