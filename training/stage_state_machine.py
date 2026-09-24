"""
Generation-1 foundation state machine — Agent J1.
================================================================================

ADAPTED from scripts/stage_state_machine.py (Part 5 port table: ADAPT —
the same STAGES_ORDER / DEPENDENCIES / STATUSES JSON-state pattern, new
phase list). The orchestrating trainer (train_generation1_foundation.py)
reads get_next_action() and executes exactly one substep per invocation;
every advance() persists immediately, so a session dying at any point
resumes by re-running — the single source of truth, never the
developer's memory.

THE PHASE LIST IS EXACTLY FOUR (Agent J1 contract; Part 2 steps 5-6 are
J2's file, gated on Agent F — extending this list here is a FAILURE
CONDITION, not a shortcut):

    backbone_only    step 1 — the compact substrate, ONE fixed center
                     fixation, linear classifier head. Establishes the
                     param/FLOP baseline every later step is measured
                     against (Part 3's ablation matrix).
    recurrence_only  step 2 — + the T=4 fixed-schedule glimpse loop and
                     the tied within-glimpse refinement. Fixed/
                     independent glimpses; NO belief, NO uncertainty
                     (the plan's explicit "NO F" step).
    belief_no_f      step 3 — + the belief carrier with U_t (Agent B's
                     S=None VectorBeliefState, Agent D's EvidentialHead)
                     and the IDENTITY update: z carried forward
                     unchanged; no learned belief dynamics yet.
    belief_with_f    step 4 — + Agent E's belief dynamics (UpdateNet +
                     precision; the predict -> observe -> error ->
                     precision -> update cycle) — still fixed/heuristic
                     gaze. NO AIS-v2: Agent F is deliberately NOT wired
                     here (that is J2's step 5).

The gaze scheme for ALL phases is the PLACEHOLDER fixed schedule
(GAZE_SCHEDULE_T4) and must never be referred to as "AIS-v2" in any
log, checkpoint, or report.

Substeps per phase: not_started -> running -> trained -> eval_pending
-> eval_complete -> done. No gates at this layer: gate criteria live in
the eval harness and the J2 ladder — this machine tracks ORCHESTRATION
state only.

The roadmap JSON lives under report/ by default (runtime state, synced
to HF by the trainer exactly like Gen-0's rhan_next_roadmap.json).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, NamedTuple, Optional

#: The J1 phase list — EXACTLY the Part 2 steps 1-4. Do not extend.
FOUNDATION_PHASES = ["backbone_only", "recurrence_only",
                     "belief_no_f", "belief_with_f"]

#: Linear chain: each phase trains from scratch on the SAME recipe; the
#: comparison is matched-compute across phases (Part 3), not cumulative.
DEPENDENCIES: Dict[str, Optional[str]] = {
    "backbone_only": None,
    "recurrence_only": "backbone_only",   # reads step 1's baseline numbers
    "belief_no_f": "recurrence_only",
    "belief_with_f": "belief_no_f",
}

#: Canonical statuses (orchestration only — no gate statuses here).
STATUSES = ("not_started", "running", "trained", "eval_pending",
            "eval_complete", "done")

#: Step 4's components must show gradient (the standing rule, checked
#: explicitly — the most repeated Gen-0 failure).
GRADIENT_REQUIRED = {
    "backbone_only": ("classifier",),
    "recurrence_only": ("classifier",),
    "belief_no_f": ("evidential_head",),
    "belief_with_f": ("update_net", "precision", "predictor"),
}


class Action(NamedTuple):
    phase: Optional[str]     # None when all four phases are done
    substep: str

    def __repr__(self) -> str:
        return f"Action(phase={self.phase!r}, substep={self.substep!r})"


def load_roadmap(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"foundation roadmap not found at {path}")
    with open(path) as f:
        return json.load(f)


def ensure_foundation_state(roadmap: Dict[str, Any]) -> None:
    """Idempotently create the generation1_foundation key (never resets
    a roadmap that already carries runtime state)."""
    key = "generation1_foundation"
    if key not in roadmap or not isinstance(roadmap[key], dict):
        roadmap[key] = {
            "schema_version": 1,
            "phases_order": list(FOUNDATION_PHASES),
            "current_phase": FOUNDATION_PHASES[0],
            "current_substep": "not_started",
            "phases": {},
        }
    fnd = roadmap[key]
    fnd.setdefault("phases_order", list(FOUNDATION_PHASES))
    fnd.setdefault("current_phase", FOUNDATION_PHASES[0])
    fnd.setdefault("current_substep", "not_started")
    fnd.setdefault("phases", {})
    for phase in FOUNDATION_PHASES:
        st = fnd["phases"].setdefault(phase, {})
        st.setdefault("status", "not_started")
        st.setdefault("depends_on", DEPENDENCIES[phase])


def _status_of(fnd: Dict, phase: str) -> str:
    return str(fnd["phases"].get(phase, {}).get("status", "not_started"))


def get_next_action(roadmap: Dict[str, Any]) -> Action:
    """First not-done phase in order, with the substep its status implies."""
    ensure_foundation_state(roadmap)
    fnd = roadmap["generation1_foundation"]
    for phase in fnd["phases_order"]:
        status = _status_of(fnd, phase)
        if status == "done":
            continue
        if status == "not_started":
            return Action(phase, "start")
        return Action(phase, status)          # running/trained/eval_pending/
    return Action(None, "done")               # eval_complete


def advance(phase: str, new_status: str, roadmap_path: str, write: bool = True,
            **metadata: Any) -> Dict[str, Any]:
    """Update a phase's status (+ metadata) and persist IMMEDIATELY."""
    if phase not in FOUNDATION_PHASES:
        raise ValueError(f"unknown phase {phase!r} — expected one of "
                         f"{FOUNDATION_PHASES} (steps 5-6 belong to J2's "
                         f"file; extending this list is a failure condition)")
    if new_status not in STATUSES:
        raise ValueError(f"unknown status {new_status!r} — expected one of "
                         f"{STATUSES}")
    roadmap = load_roadmap(roadmap_path)
    ensure_foundation_state(roadmap)
    fnd = roadmap["generation1_foundation"]
    st = fnd["phases"][phase]
    st["status"] = new_status
    for k, v in metadata.items():
        st[k] = v
    fnd["current_phase"] = phase
    fnd["current_substep"] = new_status
    if write:
        os.makedirs(os.path.dirname(roadmap_path) or ".", exist_ok=True)
        with open(roadmap_path, "w") as f:
            json.dump(roadmap, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return roadmap


def report_state(roadmap: Dict[str, Any]) -> str:
    ensure_foundation_state(roadmap)
    fnd = roadmap["generation1_foundation"]
    lines = ["  generation1_foundation (J1: Part 2 steps 1-4; steps 5-6 = J2)"]
    for phase in fnd["phases_order"]:
        st = fnd["phases"][phase]
        note = st.get("best_val_acc")
        extra = f" best_val_acc={note:.4f}" if isinstance(note, float) else ""
        lines.append(f"    {phase:<16} {st.get('status', 'not_started'):<14}"
                     f"{extra}")
    return "\n".join(lines)
