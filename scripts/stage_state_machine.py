#!/usr/bin/env python3
"""
Multi-session stage-state machine — RHAN-NX Section 4.
================================================================================

The notebook's top-level orchestration cell reads the SINGLE source of truth
for what to run next from docs/rhan_next_roadmap.json (the HF-synced copy),
so a session dying at ANY point resumes correctly by simply re-running the
notebook from the top — `get_next_action()` decides, never the developer's
memory of what ran last. This REPLACES manually-set DO_STEP_A / DO_STEP_B
boolean flags for the RHAN-NX stages.

State model (roadmap key "rhan_nx"):

    rhan_nx: {
      "schema_version": 1,
      "stages_order": ["gen0", "sbr0", "sbr1", "sbr2", "sbr3", "sbr4",
                        "ais_v2", "hpc_belief"],
      "current_stage": <stage>,
      "current_substep": <substep>,
      "stages": {
        <stage>: {
          "status": not_started | build | training | gate_pending |
                    gate_passed | gate_failed | eval_pending |
                    eval_complete | verdict_done | done,
          "depends_on": <stage or null>,
          ... (stage-specific metadata written by advance())
        }
      }
    }

substep returned by get_next_action() is derived from status:
  not_started   -> (stage, "start")     — launch this stage's first step
  build         -> (stage, "build")     — gen0: run the Generation-0 test gate
  training      -> (stage, "training")  — run/resume training (resume-safe)
  gate_pending  -> (stage, "gate")      — evaluate the pre-registered gate
  gate_passed   -> next stage's start   — or eval_pending for eval stages
  gate_failed   -> (stage, "gate_failed") — STOP; write honest failure verdict
                    (EXCEPT insufficient_data verdicts — see
                    gate_failed_re_evaluable, amendment 2026-09-11: those
                    re-enter training so the gate is re-evaluated)
  eval_pending  -> (stage, "eval")      — fresh 16-seed PGD eval on NEW ckpt
  eval_complete -> (stage, "verdict")   — write verdict + masking + advance
  verdict_done  -> (stage, "done")
  done          -> skip to the next non-done stage

Contract with the notebook: every dispatch branch MUST end by calling
advance() (which writes the roadmap immediately — never batched — so a
session dying right after a gate decision cannot lose that decision), then
the notebook's `sync_roadmap_up()` pushes it to HF.

This module is pure local-file logic (testable without HF); the HF sync
stays in the notebook's sync_roadmap_down()/sync_roadmap_up() helpers so a
fresh session always reads the HF copy before writing (never clobbering
runtime verdicts with the stale committed baseline).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple

ROADMAP_LOCAL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "docs",
                 "rhan_next_roadmap.json"))

#: Ordered ladder — the machine walks this list and returns the first stage
#: that is not done.
STAGES_ORDER = ["gen0", "sbr0", "sbr1", "sbr2", "sbr3", "sbr4",
                "ais_v2", "hpc_belief"]

#: Which stages end in a 16-seed eval (vs. gate-only stages).
EVAL_STAGES = {"sbr2", "sbr3", "sbr4", "ais_v2", "hpc_belief"}

#: Canonical statuses.
STATUSES = ("not_started", "build", "training", "gate_pending", "gate_passed",
            "gate_failed", "eval_pending", "eval_complete", "verdict_done",
            "done")

DEPENDENCIES: Dict[str, Optional[str]] = {
    "gen0": None,
    "sbr0": "gen0",
    "sbr1": "sbr0",
    "sbr2": "sbr1",
    "sbr3": "sbr2",
    "sbr4": "sbr3",
    "ais_v2": "gen0",      # independent of SBR (swap test, one-mechanism rule)
    "hpc_belief": "gen0",  # independent of SBR and of AIS-v2
}


class Action(NamedTuple):
    stage: Optional[str]      # None when everything is done
    substep: str              # start | build | training | gate | gate_failed |
                              # eval | verdict | done

    def __repr__(self) -> str:
        return f"Action(stage={self.stage!r}, substep={self.substep!r})"


# ── roadmap helpers ─────────────────────────────────────────────────────────

def load_roadmap(path: str = ROADMAP_LOCAL) -> Dict[str, Any]:
    """Read the roadmap JSON (the caller has already restored the HF copy)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"roadmap not found at {path} — run the notebook's "
            f"sync_roadmap_down() first")
    with open(path) as f:
        return json.load(f)


def ensure_rhan_nx_state(roadmap: Dict[str, Any]) -> None:
    """Create the rhan_nx key with default (not_started) state in place.

    Idempotent — a roadmap that already carries runtime verdicts is never
    reset. Called by the notebook AFTER sync_roadmap_down().
    """
    if "rhan_nx" not in roadmap or not isinstance(roadmap["rhan_nx"], dict):
        roadmap["rhan_nx"] = {
            "schema_version": 1,
            "stages_order": list(STAGES_ORDER),
            "current_stage": STAGES_ORDER[0],
            "current_substep": "not_started",
            "stages": {},
        }
    nx = roadmap["rhan_nx"]
    nx.setdefault("stages_order", list(STAGES_ORDER))
    nx.setdefault("current_stage", STAGES_ORDER[0])
    nx.setdefault("current_substep", "not_started")
    nx.setdefault("stages", {})
    for stage in STAGES_ORDER:
        st = nx["stages"].setdefault(stage, {})
        st.setdefault("status", "not_started")
        st.setdefault("depends_on", DEPENDENCIES[stage])


def _status_of(nx: Dict, stage: str) -> str:
    return str(nx["stages"].get(stage, {}).get("status", "not_started"))


# ── the machine ─────────────────────────────────────────────────────────────

def get_next_action(roadmap: Optional[Dict[str, Any]] = None) -> Action:
    """The single source of truth: what should the notebook do next?

    Walks STAGES_ORDER in dependency order and returns the first stage that
    is not done, with the substep implied by its status. All stages done ->
    Action(None, "done").
    """
    if roadmap is None:
        roadmap = load_roadmap()
    ensure_rhan_nx_state(roadmap)
    nx = roadmap["rhan_nx"]

    for stage in nx["stages_order"]:
        status = _status_of(nx, stage)
        if status == "done" or status == "verdict_done":
            continue
        if status == "not_started":
            return Action(stage, "start")
        if status == "build":
            return Action(stage, "build")
        if status == "training":
            return Action(stage, "training")
        if status == "gate_pending":
            return Action(stage, "gate")
        if status == "gate_passed":
            # Gate passed -> next step: eval for eval-stages; for gate-only
            # stages (e.g. gen0) gate_passed means the stage is complete —
            # continue to the next stage rather than returning Action(stage,
            # "done"), which the single-step dispatch has no handler for and
            # would cause it to stop prematurely.
            if stage in EVAL_STAGES:
                return Action(stage, "eval")
            continue  # treat gate_passed (non-eval) as done; find next stage
        if status == "gate_failed":
            return Action(stage, "gate_failed")     # STOP, report honestly
        if status == "eval_pending":
            return Action(stage, "eval")
        if status == "eval_complete":
            return Action(stage, "verdict")
    return Action(None, "done")


def advance(stage: str, new_status: str, roadmap_path: str = ROADMAP_LOCAL,
            write: bool = True, **metadata: Any) -> Dict[str, Any]:
    """Update a stage's status (+ metadata), then write the roadmap.

    Always syncs immediately (write=True) — never batched — so a session
    dying right after a gate decision cannot lose that decision. Returns the
    updated roadmap dict. The notebook then calls sync_roadmap_up() to push
    to HF.

    Args:
        stage: one of STAGES_ORDER.
        new_status: one of STATUSES.
        roadmap_path: where the roadmap lives.
        write: False for dry-run/no-disk usage (tests).
        metadata: extra fields recorded under stages[stage] (verdict paths,
            checkpoint names, gate scores, timestamps, ...).
    """
    if stage not in STAGES_ORDER:
        raise ValueError(f"unknown stage {stage!r} — expected one of "
                         f"{STAGES_ORDER}")
    if new_status not in STATUSES:
        raise ValueError(f"unknown status {new_status!r} — expected one of "
                         f"{STATUSES}")
    roadmap = load_roadmap(roadmap_path)
    ensure_rhan_nx_state(roadmap)
    nx = roadmap["rhan_nx"]
    st = nx["stages"][stage]
    st["status"] = new_status
    for k, v in metadata.items():
        st[k] = v
    nx["current_stage"] = stage
    nx["current_substep"] = new_status
    if write:
        with open(roadmap_path, "w") as f:
            json.dump(roadmap, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return roadmap


def reset_stage(stage: str, roadmap_path: str = ROADMAP_LOCAL) -> None:
    """Force a stage back to not_started (debug/recapture escape only)."""
    advance(stage, "not_started", roadmap_path=roadmap_path)


# ── repair rule (amendment 2026-09-11) ─────────────────────────────────────

def gate_failed_re_evaluable(stage_state: Dict[str, Any]) -> bool:
    """True when a gate_failed verdict is a measurement artifact, not a
    criteria outcome — the gate may be RE-RUN (advance to gate_pending)
    instead of stopping the ladder.

    Amendment 2026-09-11 (evidence-based; user-approved repair of the
    sbr0/gate_failed state recorded 2026-09-10T19:43:00Z): that session ran
    PRE-amendment code (84abd6b) on a wiped session-local cosine series, so
    criterion 2 recorded insufficient_data (n_points=1) and the ladder
    stopped. The amended gate (acbb143) anchors the trend at the epoch-0
    baseline and was never given the chance to evaluate that checkpoint.

    Scope is deliberately NARROW: only insufficient_data verdicts qualify.
    A verdict whose criteria were actually EVALUATED and failed (e.g. a flat
    or rising cosine trend, entropy collapse) is a substantive FAIL and
    keeps the terminal gate_failed semantics — that outcome must be written
    up honestly, never re-rolled.
    """
    v = stage_state.get("verdict")
    if not isinstance(v, dict):
        return False
    # (a) sbr0-style: criterion 2 could not be measured (lost series).
    c2 = v.get("criteria", {}).get("2_pairwise_cosine_trend")
    if isinstance(c2, dict) and c2.get("insufficient_data"):
        return True
    # (b) sbr1-style, amendment 2026-09-11: a failed verdict that records the
    # checkpoint BEATING its reference is a gate-FORMULA artifact, not a
    # criteria outcome. The original symmetric band abs(clean - D) <= 3pp
    # rejected the real SBR-1 run (62.49% clean vs D's 54.96%) for
    # over-performing the reference by +7.5pp. The one-sided collapse gate
    # (sbr1_gate_decision below) passes that checkpoint; the recorded FAIL
    # is re-evaluable.
    acc = v.get("sbr1_clean_acc")
    ref = v.get("d_reference")
    if isinstance(acc, (int, float)) and isinstance(ref, (int, float)):
        if acc >= ref:
            return True
        if acc < 0:
            # Negative accuracy is impossible — the old code's missing-telemetry
            # sentinel (-1.0). Another measurement artifact, re-evaluable.
            return True
    # (c) any verdict that could not measure anything at all.
    if v.get("insufficient_data"):
        return True
    return False


def marker_covers_ceiling(marker: Optional[Dict[str, Any]],
                          max_epochs: int) -> bool:
    """True when a training-complete marker certifies AT LEAST max_epochs.

    Amendment 2026-09-11 (sbr1 escalation fix): a marker written for a lower
    ceiling must NOT satisfy a higher one. The old unconditional check let a
    15-epoch marker answer 'already complete' to a 20-epoch request, which
    neutered the ladder's escalation loop — 'resume training to ceiling
    20/25/30/35/40' re-ran the same 15-epoch gate five times and recorded a
    terminal FAIL without training a single additional epoch.

    The caller is responsible for the ckpt_name match (the notebook's
    _stage4_read_marker_json already enforces it).
    """
    if not isinstance(marker, dict):
        return False
    try:
        return int(marker.get("max_epochs", -1)) >= int(max_epochs)
    except (TypeError, ValueError):
        return False


def sbr1_gate_decision(clean_acc: Optional[float], d_reference: float,
                       margin: float = 3.0) -> Tuple[bool, Dict[str, Any]]:
    """One-sided SBR-1 clean-accuracy gate (amendment 2026-09-11).

    Pre-registered intent (docs/rhan_next_roadmap.json -> rhan_nx.sbr1.gate):
    FAIL when SBR-1's clean accuracy COLLAPSES toward ~45% WITHOUT any
    adversarial pressure — the cheapest possible discovery of the failure
    mode E2b hit at massive compute cost. The original implementation used a
    symmetric band abs(clean - D) <= 3pp, which also fails OVER-performance:
    it rejected the real SBR-1 run (62.49% clean vs D's 54.96%) for BEATING
    the reference by +7.5pp. A strictly better model must pass a gate whose
    purpose is collapse detection.

    Decision: pass iff clean_acc >= d_reference - margin.
    Returns (passed, verdict_dict) — verdict is written to the roadmap and
    report/sbr1_gate_verdict.json verbatim, so keep it JSON-serializable.
    """
    floor = round(d_reference - margin, 2)
    if clean_acc is None:
        return False, {
            "sbr1_clean_acc": None, "d_reference": d_reference,
            "floor": floor, "one_sided_collapse_gate": True,
            "passed": False, "insufficient_data": True,
            "note": ("no durable clean-accuracy record found "
                     "(completion marker + session diag both absent)"),
        }
    passed = float(clean_acc) >= floor
    return passed, {
        "sbr1_clean_acc": float(clean_acc), "d_reference": d_reference,
        "floor": floor, "one_sided_collapse_gate": True,
        "passed": passed,
        "criterion": (f"clean_acc >= d_reference - {margin:g}pp "
                      "(collapse detector; over-performance passes)"),
    }


def report_state(roadmap: Optional[Dict[str, Any]] = None) -> str:
    """Human-readable dump of every stage's status (notebook printout)."""
    if roadmap is None:
        roadmap = load_roadmap()
    ensure_rhan_nx_state(roadmap)
    nx = roadmap["rhan_nx"]
    lines = [f"RHAN-NX state machine — current: "
             f"{nx['current_stage']}/{nx['current_substep']}"]
    for stage in nx["stages_order"]:
        st = nx["stages"].get(stage, {})
        extra = ""
        if st.get("ckpt_name"):
            extra = f" ckpt={st['ckpt_name']}"
        if st.get("verdict"):
            extra += f" verdict={st['verdict']}"
        lines.append(f"  {stage:<10} {st.get('status', 'not_started'):<14}{extra}")
    return "\n".join(lines)