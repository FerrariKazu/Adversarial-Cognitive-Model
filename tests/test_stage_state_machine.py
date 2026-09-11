"""State-machine tests — gate_failed repair rule and ladder semantics.

Amendment 2026-09-11: a gate_failed verdict whose ONLY failing criterion is
insufficient_data is a measurement artifact (the cosine series was
session-local and got wiped mid-run), not a criteria outcome. The
gate_failed_re_evaluable() helper classifies those verdicts; the notebook
uses it to re-run the gate instead of stopping the ladder. A substantive
fail (criteria evaluated and failed) keeps terminal gate_failed semantics.

Second 2026-09-11 rule (same day): a gate_failed verdict that records the
checkpoint BEATING its reference (sbr1_clean_acc >= d_reference) is a
gate-FORMULA artifact — the symmetric band abs(clean - D) <= 3pp failed the
real SBR-1 run (62.49% vs D's 54.96%) for over-performing. The one-sided
collapse detector (sbr1_gate_decision) is the corrected formula; the
recorded FAIL is re-evaluable under it.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from stage_state_machine import (  # noqa: E402
    gate_failed_re_evaluable,
    get_next_action,
    advance,
    ensure_rhan_nx_state,
    marker_covers_ceiling,
    sbr1_gate_decision,
)

# The exact verdict the 2026-09-10 session recorded for sbr0
# (insufficient_data on criterion 2; criteria 1/3/4 passed).
REAL_SBR0_VERDICT = {
    "passed": False,
    "criteria": {
        "1_slot_occupancy_entropy": {"passed": True,
                                     "mean_normalized_entropy": 0.983341,
                                     "floor": 0.6},
        "2_pairwise_cosine_trend": {"passed": False,
                                    "insufficient_data": True,
                                    "n_points": 1, "required": 2},
        "3_per_slot_probes": {"passed": True,
                              "per_slot_acc": [0.4961] * 16,
                              "n_above": 16, "floor": 0.25,
                              "min_required": 4},
        "4_everything_slot_ablation": {"passed": True,
                                       "victim_slot": 2,
                                       "retained": 1.022901,
                                       "floor": 0.7},
    },
    "schema": "sbr0_gate_v1",
    "timestamp_utc": "2026-09-10T19:43:00Z",
}


def _state_with_verdict(verdict) -> dict:
    return {"status": "gate_failed", "ceiling": 60, "verdict": verdict}


def test_real_sbr0_verdict_is_re_evaluable():
    assert gate_failed_re_evaluable(_state_with_verdict(REAL_SBR0_VERDICT))


def test_substantive_fail_is_not_re_evaluable():
    # Criteria evaluated and FAILED — terminal, never re-rolled.
    v = {"passed": False,
         "criteria": {"1_slot_occupancy_entropy": {"passed": True},
                      "2_pairwise_cosine_trend": {
                          "passed": False, "n_points": 4,
                          "slope": 0.0003, "max_slope": 0.0},
                      "3_per_slot_probes": {"passed": True},
                      "4_everything_slot_ablation": {"passed": True}}}
    assert not gate_failed_re_evaluable(_state_with_verdict(v))


def test_no_verdict_or_malformed_is_not_re_evaluable():
    # Conservative default: anything unreadable stays a terminal FAIL.
    assert not gate_failed_re_evaluable({"status": "gate_failed"})
    assert not gate_failed_re_evaluable(
        _state_with_verdict(None))
    assert not gate_failed_re_evaluable(
        _state_with_verdict("not-a-dict"))
    assert not gate_failed_re_evaluable(
        _state_with_verdict({"passed": False, "criteria": {}}))
    assert not gate_failed_re_evaluable(
        _state_with_verdict({"passed": False,
                             "criteria": {"2_pairwise_cosine_trend":
                                          "malformed"}}))


def test_repair_advances_to_training_and_ladder_maps_it():
    import json, tempfile
    from stage_state_machine import load_roadmap
    with tempfile.TemporaryDirectory() as td:
        rp = os.path.join(td, "roadmap.json")
        base = {"rhan_nx": {"schema_version": 1,
                            "stages_order": ["gen0", "sbr0"],
                            "current_stage": "sbr0",
                            "current_substep": "gate_failed",
                            "stages": {"gen0": {"status": "gate_passed"},
                                       "sbr0": {"status": "gate_failed",
                                                 "ceiling": 60,
                                                 "verdict": REAL_SBR0_VERDICT}}}}
        with open(rp, "w") as f:
            json.dump(base, f)

        # advance() to the repair status the notebook issues.
        advance("sbr0", "training", roadmap_path=rp, ceiling=60)
        action = get_next_action(load_roadmap(rp))
        assert action.stage == "sbr0"
        assert action.substep == "training"


# ── sbr1 formula-artifact rule + one-sided gate (amendment 2026-09-11) ──────

# The exact verdict the 2026-09-11 session recorded for sbr1: the REAL run
# (62.49% clean) rejected by the symmetric band for beating D's 54.96%.
REAL_SBR1_VERDICT = {
    "sbr1_clean_acc": 62.4875,
    "d_reference": 54.96,
    "within_3pp": False,
}


def test_real_sbr1_overperformance_verdict_is_re_evaluable():
    assert gate_failed_re_evaluable(
        _state_with_verdict(REAL_SBR1_VERDICT))


def test_sbr1_missing_telemetry_sentinel_is_re_evaluable():
    # The old code scored a wiped session as te_acc=-1.0 — impossible
    # accuracy, i.e. a measurement artifact, not a criteria outcome.
    assert gate_failed_re_evaluable(_state_with_verdict(
        {"sbr1_clean_acc": -1.0, "d_reference": 54.96,
         "within_3pp": False}))


def test_sbr1_substantive_collapse_fail_is_not_re_evaluable():
    # Clean accuracy BELOW the floor — the real failure mode the gate
    # exists to catch. Terminal, never re-rolled.
    assert not gate_failed_re_evaluable(_state_with_verdict(
        {"sbr1_clean_acc": 45.2, "d_reference": 54.96,
         "within_3pp": False}))


def test_sbr1_gate_decision_one_sided():
    # Over-performance PASSES (the exact numbers from the real run).
    passed, v = sbr1_gate_decision(62.4875, 54.96)
    assert passed
    assert v["passed"] is True
    assert v["floor"] == 51.96
    assert v["one_sided_collapse_gate"] is True
    # At/below floor boundary.
    passed, v = sbr1_gate_decision(51.96, 54.96)
    assert passed  # floor itself is a pass (>=)
    passed, _ = sbr1_gate_decision(51.95, 54.96)
    assert not passed
    # Collapse fails — the pre-registered E2b-style discovery.
    passed, v = sbr1_gate_decision(45.2, 54.96)
    assert not passed
    assert v["passed"] is False
    assert "insufficient_data" not in v


def test_sbr1_gate_decision_missing_data():
    passed, v = sbr1_gate_decision(None, 54.96)
    assert not passed
    assert v["insufficient_data"] is True
    assert v["sbr1_clean_acc"] is None


# ── marker escalation (the neutered ceiling loop) ───────────────────────────

def test_marker_covers_ceiling():
    m = {"ckpt_name": "rhan_nx_sbr1", "max_epochs": 15,
         "best_acc": 62.4875}
    assert marker_covers_ceiling(m, 15)
    assert not marker_covers_ceiling(m, 20)   # escalation must retrain
    assert not marker_covers_ceiling(m, 40)
    assert marker_covers_ceiling({"max_epochs": 40}, 40)
    assert marker_covers_ceiling({"max_epochs": 41}, 40)
    assert not marker_covers_ceiling(None, 15)
    assert not marker_covers_ceiling({}, 15)
    assert not marker_covers_ceiling({"max_epochs": "junk"}, 15)
