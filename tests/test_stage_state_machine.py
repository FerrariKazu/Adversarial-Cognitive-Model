"""State-machine tests — gate_failed repair rule and ladder semantics.

Amendment 2026-09-11: a gate_failed verdict whose ONLY failing criterion is
insufficient_data is a measurement artifact (the cosine series was
session-local and got wiped mid-run), not a criteria outcome. The
gate_failed_re_evaluable() helper classifies those verdicts; the notebook
uses it to re-run the gate instead of stopping the ladder. A substantive
fail (criteria evaluated and failed) keeps terminal gate_failed semantics.
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
