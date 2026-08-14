"""Stage 2 — gate trend-tracking across a resume boundary (REQUIRED gate test).

2026-08-13: the smoke health gate compared rows[0] vs rows[-1] of the
--diag-json. When a session resumed from the HF rolling checkpoint (epoch
14 -> 15) and appended only its own epoch, the gate compared epoch 15 to
ITSELF ("epoch 15 0.6906 -> epoch 15 0.6906, ratio 1.00") and judged the run
"did NOT decrease >= 10%" — a verdict that said NOTHING about the optimizer
fix under test. The rule this pins:

    A trend verdict must compare the run's TRUE first logged epoch against
    the last. The epoch-1 baseline is read from the FULL diag.jsonl history
    across resumes (session 1's rows survive / the trainer carries the
    first-epoch summary in the rolling checkpoint as 'first_epoch_diag' and
    prepends it on resume). The gate must NEVER derive a trend from the
    current epoch's telemetry compared to itself: a single-epoch diag is
    reported as "NOT judgeable", not as a ratio.

These tests exercise the REAL gate (health_verdict_stage2, extracted from
the notebook sources via AST — the notebooks execute on import, so the
function is read out of the source and run in a minimal namespace) and the
REAL trainer helper (ensure_diag_baseline). Both notebooks are tested so a
future edit to one gate can never silently drift from the other.
"""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    ROOT / "cloud_setup" / "colab_notebook_noesis.py",
    ROOT / "cloud_setup" / "Kaggle_NOESIS.py",
]

# Realistic telemetry from the Stage 2 smoke family. Epoch-1 hpc_error is the
# predict-zero floor on the training distribution; the resumed session's
# epoch-15 value is whatever the head produced.
ROW1 = {"epoch": 1, "eps": 0.031, "hpc_error_mean": 0.691200,
        "hpc_error_map_min": 1e-4, "hpc_error_map_max": 1.9986,
        "hpc_error_map_std": 0.5072,
        "pi_d_per_class": {"car": 0.42, "truck": 0.38, "airplane": 0.36}}
ROW15_FLAT = {"epoch": 15, "eps": 0.031, "hpc_error_mean": 0.690600,
              "hpc_error_map_min": 1e-4, "hpc_error_map_max": 1.9986,
              "hpc_error_map_std": 0.5072,
              "pi_d_per_class": {"car": 0.40, "truck": 0.39, "airplane": 0.37}}
ROW15_LEARNED = {"epoch": 15, "eps": 0.031, "hpc_error_mean": 0.200000,
                 "hpc_error_map_min": 1e-4, "hpc_error_map_max": 1.2,
                 "hpc_error_map_std": 0.30,
                 "pi_d_per_class": {"car": 0.41, "truck": 0.40, "airplane": 0.35}}


def _extract_gate(nb_path):
    """Pull the REAL health_verdict_stage2 out of a notebook's source.

    The notebooks are executable scripts (they train on import), so they
    cannot be imported; the function is extracted by AST and run in a minimal
    namespace holding the two constants it references.
    """
    tree = ast.parse(nb_path.read_text())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and n.name == "health_verdict_stage2"), None)
    assert fn is not None, f"health_verdict_stage2 not found in {nb_path}"
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"HPC_TREND_MIN_DECREASE": 0.10, "HPC_EXPLOSION_RATIO": 10.0}
    exec(compile(mod, str(nb_path), "exec"), ns)
    return ns["health_verdict_stage2"]


def _mk_session_rows(start, end, err_fn):
    """Rows as a session would actually append them to diag.jsonl."""
    rows = []
    for ep in range(start, end + 1):
        r = dict(ROW1)
        r["epoch"] = ep
        r["hpc_error_mean"] = err_fn(ep)
        rows.append(r)
    return rows


# ── The gate, exercised on both notebooks ───────────────────────────────────
GATES = {p.name: _extract_gate(p) for p in NOTEBOOKS}


@pytest.mark.parametrize("gate", GATES.values(), ids=list(GATES))
def test_resume_only_single_epoch_is_not_judgeable_never_self_compare(gate):
    """THE regression: a resume that logged ONLY its own final epoch (session
    1's diag never reached HF before the kill) must NEVER produce a trend
    ratio. Before the fix the gate compared epoch 15 to ITSELF and printed
    'epoch 15 0.6906 -> epoch 15 0.6906, ratio 1.00' as a FAIL verdict.
    """
    v = gate([ROW15_FLAT])
    assert v["healthy"] is False
    joined = " ".join(v["reasons"])
    assert "NOT judgeable" in joined, joined
    # The self-comparison must be structurally impossible: no message may
    # reference the same epoch twice as both endpoints of a trend.
    assert "-> epoch 15" not in joined, joined
    assert v["last_epoch"] == 15


@pytest.mark.parametrize("gate", GATES.values(), ids=list(GATES))
def test_carried_epoch1_baseline_gives_true_comparison(gate):
    """With the trainer fix (ensure_diag_baseline prepends the carried
    first-epoch summary), the resumed session's diag is [epoch 1, epoch 15].
    The gate must compare the TRUE epoch-1 value (0.6912) against epoch 15 —
    the honest 1->15 comparison, not a self-comparison."""
    v = gate([ROW1, ROW15_FLAT])
    joined = " ".join(v["reasons"])
    assert "epoch 1 0.6912 -> epoch 15 0.6906" in joined, joined
    assert "NOT judgeable" not in joined, joined
    assert v["last_epoch"] == 15


@pytest.mark.parametrize("gate", GATES.values(), ids=list(GATES))
def test_full_history_across_resume_uses_epoch1_baseline(gate):
    """The FULL diag.jsonl history across resumes: session 1 logged epochs
    1-14 (file survived), session 2 appended epoch 15. min/max epoch
    selection must anchor the trend on the run's TRUE epoch 1 regardless of
    file order (rows[0] vs rows[-1] was the bug — appended history made
    rows[-1] the same epoch as a fresh session's rows[0])."""
    session1 = _mk_session_rows(1, 14, lambda ep: 0.6912 + (ep - 1) * 1e-5)
    diag = session1 + [ROW15_FLAT]
    v = gate(diag)
    joined = " ".join(v["reasons"])
    assert "epoch 1 0.6912 -> epoch 15 0.6906" in joined, joined
    # Robust to unordered rows too: epoch selection is by min/max epoch.
    v2 = gate(list(reversed(diag)))
    joined2 = " ".join(v2["reasons"])
    assert "epoch 1 0.6912 -> epoch 15 0.6906" in joined2, joined2


@pytest.mark.parametrize("gate", GATES.values(), ids=list(GATES))
def test_learning_head_across_resume_passes_trend(gate):
    """A head that actually learned across the resume boundary (epoch 1
    0.6912 -> epoch 15 0.2000, ratio 0.29) must PASS the >= 10% decrease
    check — the honest multi-epoch comparison the fix enables."""
    v = gate([ROW1, ROW15_LEARNED])
    joined = " ".join(v["reasons"])
    assert "trend OK" in joined, joined
    assert v["healthy"] is True, joined


# ── End-to-end: trainer helper + gate over the REAL resume file flow ────────
def test_end_to_end_resume_diag_flow(tmp_path):
    """Wire the two fixes together the way a real resume does: session 2
    calls ensure_diag_baseline (trainer) on a diag that holds only its own
    epoch, then the notebook's gate reads the resulting file. The verdict
    must cite the TRUE epoch-1 baseline — the self-comparison can never
    recur."""
    import sys
    sys.path.insert(0, str(ROOT / "phase1_training"))
    from train_rhan_next import ensure_diag_baseline

    diag = tmp_path / "diag.jsonl"
    with open(diag, "w") as f:
        f.write(json.dumps(ROW15_FLAT) + "\n")

    assert ensure_diag_baseline(str(diag), ROW1) is True
    rows = [json.loads(l) for l in diag.read_text().splitlines() if l.strip()]
    assert [r["epoch"] for r in rows] == [1, 15]

    for gate in GATES.values():
        v = gate(rows)
        joined = " ".join(v["reasons"])
        assert "epoch 1 0.6912 -> epoch 15 0.6906" in joined, joined
        assert "NOT judgeable" not in joined, joined
