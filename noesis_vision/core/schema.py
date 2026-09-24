"""
RHAN-NXA canonical configuration schema — Agent 0 output.

One schema, one source of truth. Every downstream agent imports this
config rather than reimplementing its own interpretation (MASTER_PLAN,
Agent 0 contract — the structural fix for Gen-0's interface drift).

Status labels used in field comments (see MASTER_PLAN Part 1 and
RHAN_NXA/docs/15_Status_And_Decision_System.md):

  LOCKED                  — decided; not agent discretion.
  REQUIRED                — mechanism design is required (not a claim it works).
  EXPERIMENTAL CANDIDATE  — best-reasoned default, explicitly NOT proven;
                            an experiment resolves it later.
  PENDING DECISION        — genuinely open; defaults to its safe value
                            (False/None) and is gated where re-entry order
                            matters (see GATED_FLAGS below).
  DEFERRED                — out of scope until a documented precondition exists.

This module establishes NOTHING empirically. Pure interface design —
no experiment, no claim, no number (Agent 0 contract, Scientific
Interpretation).

Source: RHAN_NXA/MASTER_PLAN.md, Parts 1.A–1.I (verbatim capture).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Optional

SCHEMA_VERSION = "0.1.0"

# LOCKED boundary conditions (MASTER_PLAN Part 1.B). Not configurable —
# exposed as constants so tests and downstream agents can assert against
# drift instead of silently diverging (the Gen-0 failure mode).
FIRST_GLIMPSE_CONVENTION = (
    "E_0 := 0 (LOCKED, MASTER_PLAN Part 1.B — the first glimpse has no "
    "predecessor belief; the update reduces to "
    "z_1 = z_0 + Pi_0 * UpdateNet(z_0, 0); prediction and genuine E_t "
    "begin at t=1)"
)
AIS_T0_SCORING = (
    "heuristic-saliency fallback (LOCKED, MASTER_PLAN Part 1.B — no "
    "U_0-conditioned prediction exists at t=0, so candidate scoring "
    "falls back to the same heuristic-saliency sampling used for "
    "candidate generation)"
)

# REJECTED outright — documented so no agent re-introduces them as a
# "fallback" without an explicit plan ruling (Part 1.B, 1.D, 1.H; Part 5).
REJECTED_OUTRIGHT = (
    "pixel-space reconstruction as E_t's target (Part 1.B)",
    "edge-map / HPC-style hand-designed feature targets (Part 1.B)",
    "legacy 16-slot Slot Attention implementation (Part 1.D — REJECTED "
    "as implemented; the concept is EXPERIMENTAL CANDIDATE, DEFERRED, "
    "2-4 slots only if revisited)",
    "external retrieval / RAG memory (Part 1.H)",
    "AIS-v1's relocated-Eq.-II gaze code (Part 5 port table)",
    "raw addition z_t + lambda*Pi*E_t — superseded by UpdateNet (Part 1.B)",
)

# Informational only (Part 5 port table): the optimizer-group config
# shape should match what the ported multi-group optimizer + registry
# expects. Agent A ports that registry and owns the final grouping —
# these names document the per-component gradient-isolation intent
# (e.g. UpdateNet's own optimizer group + |dW| pre-flight, Part 1.B),
# they do NOT decide Agent A's registry design.
OPTIMIZER_GROUP_NAMES = (
    "backbone",
    "recurrence_refinement",
    "predictor",
    "update_net",
    "precision",
    "evidential_head",
    "gaze_policy",
    "l_stab_diagnostic",
)

# Flags whose re-entry order matters. Setting the flag True without its
# documented prerequisite raises — it must never silently succeed
# (Agent 0 contract, TESTS section; non-improvisation rule).
GATED_FLAGS = {
    "enable_sbr": (
        "step6_validated_result",
        "S_t re-entry is Part 2 step 7, ONLY after step 6 has a validated "
        "result (Part 1.D; Part 4 Adjustment 2)",
    ),
    "enable_l_stab_objective": (
        "step6_validated_result",
        "L_stab is promoted from diagnostic to objective ONLY after step 6 "
        "has a validated ImageNet-100 result (Part 1.G; Part 2 step 8)",
    ),
    "adaptive_halting": (
        "step6_validated_result",
        "the core loop must be validated at fixed depth T=4 before a "
        "mechanism with a documented history of fighting other objectives "
        "is reintroduced (Part 1.C)",
    ),
    "enable_episodic_memory": (
        "temporal_experiment_evidence",
        "episodic/temporal persistence across images is DEFERRED until a "
        "specific experiment shows temporal persistence is a limiting "
        "capability (Part 1.H)",
    ),
}


@dataclass
class RHANNXAConfig:
    """Canonical RHAN-NXA configuration (Parts 1.A–1.I).

    Every field carries its decision status as a comment. Fields added
    beyond this plan require flagging back as a question — never a
    silent addition (Agent 0 contract, API/Interface Contract).
    """

    # --- Versioning -----------------------------------------------------
    # Bump on ANY interface change; tests assert round-trip stability.
    schema_version: str = SCHEMA_VERSION

    # --- Substrate identity (Part 1.A) ----------------------------------
    # num classes C: dataset/phase-level (STL-10 vs ImageNet-100 vs 1K);
    # the plan does not pin a single value — set per experiment.
    num_classes: Optional[int] = None
    # D_z = backbone embed dim: "locked once the substrate (Agent C) sets
    # it" (Part 1.A). None until Agent C exists — not guessed here.
    d_z: Optional[int] = None

    # --- Recurrence (Part 1.C — LOCKED: Hybrid Option C) ----------------
    # Backbone-only ablation arms (Part 3) disable this; the full-loop
    # default is on.
    enable_recurrence: bool = True  # LOCKED default for the full loop
    # SHARED-WEIGHT (tied) within-glimpse transformer block: iteration
    # count changes compute, NOT parameter count.
    within_glimpse_shared_weights: bool = True  # LOCKED (Universal-
    # Transformer-style tied refinement)
    # Run 2-3 times per glimpse before pooling (Part 1.C). Exact value
    # within the LOCKED range is a build/experiment choice, not a new
    # decision — enforced to the authorized range below.
    within_glimpse_iters: int = 2  # LOCKED range 2–3 (Part 1.C)
    # Across-glimpse: T fixed glimpses per image. "start at T=4" —
    # reusing the STL-10 project's own validated convention (Part 1.C).
    num_glimpses: int = 4  # LOCKED T=4 for the Gen-1 core
    # Adaptive halting: DEFERRED for the first build; gated flag.
    adaptive_halting: bool = False  # DEFERRED (Part 1.C), gated

    # --- Belief structure (Part 1.A / 1.D — LOCKED: S_t = None) ---------
    # S_t is None for the core build. REJECTED as currently implemented
    # (16-slot); if ever revisited it is Part 2 step 7, 2-4 slots only.
    enable_sbr: bool = False  # LOCKED default per Part 1.D; PENDING
    # DECISION if ever set True — see gate requirement (GATED_FLAGS).
    # Slot count if S_t is ever revisited: 2–4 ONLY (Part 1.D). None
    # while S_t is None.
    sbr_num_slots: Optional[int] = None  # 2–4 ONLY if revisited (1.D)

    # Representation-level uncertainty: the Part 1.A TENSION resolution
    # keeps class-conditioned Dirichlet for Gen-1; this is explicitly
    # PENDING DECISION, deferred past Gen-1's first integrated build.
    representation_level_uncertainty: bool = False  # PENDING DECISION
    # (Part 1.A tension; safe default False)

    # --- Uncertainty (Part 1.F — LOCKED) --------------------------------
    # One representation only: EvidentialHead's Dirichlet formulation,
    # PORTED (not reimplemented) from the STL-10 project. Feeds Pi_t,
    # AIS-v2 candidate scoring, and L_stab's drift metric. ECE is
    # evaluated (Agent I), not trained against, in Gen-1's first pass.
    enable_evidential_uncertainty: bool = True  # LOCKED (REQUIRED port)

    # --- Prediction error / belief update (Part 1.B) --------------------
    # F flag for the ablation ladder (Part 2 steps 3–4, Part 3 arms);
    # full-loop default on.
    enable_belief_update: bool = True  # LOCKED default (full loop)
    # E_t's target: latent token/patch-level features at the NEXT
    # GLIMPSE's fixation (LOCKED DEFAULT as a design). Status:
    # EXPERIMENTAL CANDIDATE — resolved by a clean, SBR-disabled rerun
    # vs a null (F = identity); do not strengthen the claim.
    error_target: str = "latent_next_glimpse"  # EXPERIMENTAL CANDIDATE
    # (design default; Part 1.B). Pixel and edge/HPC targets are
    # REJECTED — see REJECTED_OUTRIGHT.

    # --- AIS-v2 (Part 1.E — LOCKED: one mechanism, no v3) ---------------
    # Mechanism DESIGN = REQUIRED (r=0.706 is real, targeted evidence).
    # The mechanism's 16-seed accuracy/robustness contribution = PENDING
    # DECISION (Part 0 confound) — re-test cleanly; do not assume the
    # confounded +11.42 number.
    enable_ais_v2: bool = True  # REQUIRED mechanism design (Part 1.E)
    # Candidate generation heuristic: reuse whatever the validated
    # STL-10 AIS-v2 implementation used — do not redesign from scratch
    # without cause.
    candidate_generation: str = "stl10_validated_heuristic"  # LOCKED (1.E)
    # Candidate scoring: predicted uncertainty reduction via U_t's
    # Dirichlet entropy — NO separate scoring head (one uncertainty
    # representation). At t=0: see AIS_T0_SCORING above.
    candidate_scoring: str = "dirichlet_entropy_reduction"  # LOCKED (1.E)
    # K = 4–8 candidate locations (Part 1.E). Exact value within the
    # LOCKED range is an experiment choice; enforced to range below.
    num_candidates: int = 4  # LOCKED range 4–8 (Part 1.E)
    # Selection: soft (Gumbel-softmax / straight-through) during
    # training, hard argmax at inference — a phase behavior, not a
    # config flag (Part 1.E). Center-bias = FAILURE CONDITION, not a
    # passing result with an asterisk (enforced by the gate suite, not
    # this schema).

    # --- L_stab (Part 1.G — LOCKED: staged) -----------------------------
    # Phase A: diagnostic-only through the entire core build and first
    # ablation matrix. Responsiveness guard: low drift on BOTH
    # adversarial AND OOD probes = FAILED, full stop (hard, pre-
    # registered; Gate 9).
    l_stab_diagnostic_only: bool = True  # LOCKED staged phase A (1.G)
    # Phase B promotion: PENDING DECISION, hard-gated on a validated
    # step-6 result (Part 2 step 8; Part 4 Adjustment 2 analog).
    enable_l_stab_objective: bool = False  # PENDING DECISION, gated

    # --- Memory (Part 1.H) ----------------------------------------------
    # Parametric memory (theta) and within-image trajectory (h_t = B_t
    # across T steps) are inherent — no flags. Episodic persistence
    # across images: DEFERRED, gated. RAG: REJECTED outright (see
    # REJECTED_OUTRIGHT) — deliberately no field exists to enable it.
    enable_episodic_memory: bool = False  # DEFERRED (1.H), gated

    # --- V1 frontend (Part 1.I) -----------------------------------------
    # Fixed (non-learnable) Gabor frontend: EXPERIMENTAL CANDIDATE,
    # built LAST, standalone ablation required before any "integrated"
    # configuration. Default off — the core loop must validate without
    # it first.
    enable_v1_frontend: bool = False  # EXPERIMENTAL CANDIDATE, built
    # LAST (1.I); standalone ablation required

    # --- Prerequisite records (gate targets) ----------------------------
    # Set only by Agent J's confirmation of a validated step-6 result
    # (Part 2 step 6; Part 4 Adjustment 2). Free-text reference so the
    # record itself is auditable, not a bare bool.
    step6_validated_result: Optional[str] = None
    # Evidence reference from the specific temporal experiment showing
    # persistence is a limiting capability (Part 1.H).
    temporal_experiment_evidence: Optional[str] = None

    def __post_init__(self) -> None:
        # Gated flags must never silently succeed (Agent 0 contract).
        for flag_name, (prereq_name, why) in GATED_FLAGS.items():
            if getattr(self, flag_name) and not getattr(self, prereq_name):
                raise ValueError(
                    f"{flag_name}=True requires the documented prerequisite "
                    f"`{prereq_name}` to be recorded ({why}). Refusing to "
                    f"enable a gated flag without its prerequisite "
                    f"(non-improvisation rule, MASTER_PLAN Part 4)."
                )

        # LOCKED ranges — deviations are schema drift, not settings.
        if not 4 <= self.num_candidates <= 8:
            raise ValueError(
                f"num_candidates={self.num_candidates} outside the LOCKED "
                f"range 4-8 (Part 1.E)."
            )
        if self.within_glimpse_iters not in (2, 3):
            raise ValueError(
                f"within_glimpse_iters={self.within_glimpse_iters} outside "
                f"the LOCKED range 2-3 (Part 1.C)."
            )
        if self.enable_recurrence and self.num_glimpses != 4:
            raise ValueError(
                f"num_glimpses={self.num_glimpses} but T=4 is LOCKED for "
                f"the Gen-1 core (Part 1.C); changing it is a plan ruling, "
                f"not a config edit."
            )
        if self.sbr_num_slots is not None and not 2 <= self.sbr_num_slots <= 4:
            raise ValueError(
                f"sbr_num_slots={self.sbr_num_slots}: if S_t is ever "
                f"revisited it starts at 2-4 slots ONLY (Part 1.D); 16-slot "
                f"is REJECTED."
            )
        if self.enable_sbr and self.sbr_num_slots is None:
            raise ValueError(
                "enable_sbr=True requires an explicit sbr_num_slots in 2-4 "
                "(Part 1.D) — S_t re-entry is never implicit."
            )
        if self.error_target != "latent_next_glimpse":
            raise ValueError(
                f"error_target={self.error_target!r}: the only authorized "
                f"design default is 'latent_next_glimpse' (Part 1.B); "
                f"pixel/edge targets are REJECTED for Gen-1."
            )

    def to_dict(self) -> dict:
        """JSON-safe serialization (round-trip tested)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RHANNXAConfig":
        """Strict reconstruction: unknown keys are schema drift and raise."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"unknown schema keys {sorted(unknown)} — possible schema "
                f"drift; refusing silent reconciliation (bump "
                f"schema_version and reconcile explicitly instead)."
            )
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "RHANNXAConfig":
        return cls.from_dict(json.loads(s))
