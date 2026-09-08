"""
RHANNextConfig — single source of truth for the RHAN-Next model.

The DEFAULT config (all pillars False) must produce a model whose forward
pass shape-matches RHANv12's — verified by tests/test_config_backward_compat.py.
This is what lets train_rhan_next.py be a strict superset of
train_rhan_v12.py rather than a divergent codepath.

Stage gates (enforced by validate()):
  * enable_sbr and enable_iwm MUST remain False (scaffold-only pillars).
  * hpc_num_levels must be 0 or 1 in this pass (one level per validation
    cycle — see docs/rhan_next_roadmap.json).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict


@dataclass
class RHANNextConfig:
    # ── Pillar toggles ────────────────────────────────────────────────────────
    enable_hpc: bool = False      # Pillar 1 — off by default until Stage 2 lands
    hpc_num_levels: int = 0       # add levels one at a time, never jump; 0 = off,
                                  # 1 = the single implemented level (edge_map)
    enable_ais: bool = False      # Pillar 2 — off by default until Stage 1 lands
    enable_sbr: bool = False      # Pillar 3 — structured belief representation
    enable_iwm: bool = False      # Pillar 4 — MUST remain False; scaffold only

    # ── v12 hyperparameters (carried over unchanged) ─────────────────────────
    num_classes: int = 10
    embed_dim: int = 768
    proj_dim: int = 512
    num_heads: int = 12
    ff_dim: int = 3072
    num_transformer_layers: int = 8
    num_recurrent_steps: int = 2
    stem_dropout: float = 0.1
    max_foraging_steps: int = 4   # fixed T in v12; AIS may halt earlier
    fovea_size: int = 48
    metabolic_cost: float = 0.05  # retained for checkpoint compat only
    precision_tau: float = 0.1
    gaze_lambda: float = 0.5      # recon-guided gaze blend (Eq. II v12)

    # ── Pillar 3 (SBR) knobs ──────────────────────────────────────────────
    sbr_num_slots: int = 16       # number of object slots
    sbr_slot_dim: int = 512       # dimension per slot (must match proj_dim)
    sbr_slot_iters: int = 3       # slot attention refinement iterations
    sbr_num_heads: int = 4        # attention heads in slot attention

    # SBR sub-stage (RHAN-NX ladder):
    #   "legacy"            -> E2-equivalent N=1 slot wiring (the ONLY value
    #                          every pre-RHAN-NX SBR checkpoint carries); the
    #                          model must not reinterpret an old checkpoint.
    #   "gate_only"         -> SBR-0: frozen backbone, clean-only (eps=0),
    #                          slots trained against spatial stem features.
    #   "clean_classifier"  -> SBR-1: joint fine-tune, still eps=0.
    #   "adversarial_ramp"  -> SBR-2: standard 3-phase curriculum (comparable
    #                          to D/E1/E2b/E3b).
    #   "relational"        -> SBR-3: + inter-slot message passing + per-slot
    #                          shape/texture/spatial evidence heads.
    #   "uncertainty"       -> SBR-4: + hypothesis/supporting/contradictory/
    #                          uncertainty decomposition.
    sbr_stage: str = "legacy"
    # SBR-0 ONLY: freeze the entire D backbone; only the slot-attention
    # parameters are trainable (the narrow unlocked validate() path).
    freeze_backbone_for_sbr0: bool = False

    # ── Pillar 2 (AIS) knobs ─────────────────────────────────────────────────
    ais_halt_threshold: float = 0.35   # halt when belief uncertainty < this
    ais_continuation_softness: float = 8.0  # steepness of the soft gate
    ais_base_step: float = 0.20         # v12's fixed base gaze step
    ais_precision_step_range: float = 0.30  # v12's precision-scaled range

    # AIS gaze mechanism variant (RHAN-NX swap test):
    #   "halting_only"  -> AIS-v1 (Relocated Eq. II, halting-only variant) —
    #                       the ONLY value every pre-RHAN-NX checkpoint
    #                       carries (mechanistic identity documented in
    #                       rhan_core/gaze/info_gain_policy.py).
    #   "info_gain_v2"  -> AIS-v2: genuine one-step-lookahead expected
    #                       information gain (rhan_core/gaze/info_gain_policy_v2.py).
    ais_variant: str = "halting_only"

    # ── AIS sub-mechanism ablation switches (Stage 1 mechanism isolation) ────
    # Both default True = the AIS-v1 smoke behavior. Each False is ONE
    # isolated ablation of a single new sub-mechanism, everything else
    # identical (project lesson #3 — never change two knobs at once):
    #   ais_halt_enabled=False          -> entropy gate forced open (cont=1,
    #       v12 fixed-T belief accumulation); gaze update unchanged.
    #   ais_precision_recon_enabled=False -> w_recon stays FLAT (v12 recon
    #       weighting); the precision modulator no longer scales the recon
    #       loss. Trainer-side only; no eval/forward impact.
    ais_halt_enabled: bool = True
    ais_precision_recon_enabled: bool = True

    # ── Pillar 1 (HPC) knobs ─────────────────────────────────────────────────
    # w_hpc — the HPC prediction-error loss weight. Deliberately a SEPARATE
    # slot from w_recon (never reuse the recon weight's slot): ablation of the
    # whole HPC pillar is then a clean single-parameter toggle (hpc_num_levels
    # 1 -> 0, or w_hpc -> 0).
    hpc_error_weight: float = 0.10      # loss weight used by train_rhan_next.py

    # HPC predictive target (RHAN-NX swap test):
    #   "pixel"  -> predict the edge_map of the foveal crop (D's target) —
    #               the ONLY value every pre-RHAN-NX checkpoint carries.
    #   "belief" -> predict belief_t+1 from belief_t (belief-space target,
    #               motivated by E1's Lens finding that pixel reconstruction
    #               dilutes precision without transferring to robustness).
    hpc_target: str = "pixel"

    # ── v12 constructor compatibility ────────────────────────────────────────
    def v12_kwargs(self) -> Dict[str, Any]:
        """Subset of fields passed to the RHANv12 constructor unchanged."""
        names = [
            "num_classes", "embed_dim", "proj_dim", "num_heads", "ff_dim",
            "num_transformer_layers", "num_recurrent_steps", "stem_dropout",
            "max_foraging_steps", "fovea_size", "metabolic_cost",
            "precision_tau", "gaze_lambda",
        ]
        return {n: getattr(self, n) for n in names}

    def validate(self) -> None:
        """Raise ValueError on configs that break the stage discipline."""
        _SBR_STAGES = ("legacy", "gate_only", "clean_classifier",
                       "adversarial_ramp", "relational", "uncertainty")
        if self.sbr_stage not in _SBR_STAGES:
            raise ValueError(
                f"sbr_stage must be one of {_SBR_STAGES}, "
                f"got {self.sbr_stage!r}")
        if self.ais_variant not in ("halting_only", "info_gain_v2"):
            raise ValueError(
                f"ais_variant must be 'halting_only' (AIS-v1) or "
                f"'info_gain_v2' (AIS-v2), got {self.ais_variant!r}")
        if self.hpc_target not in ("pixel", "belief"):
            raise ValueError(
                f"hpc_target must be 'pixel' (edge-map target, D) or "
                f"'belief' (belief-space target, RHAN-NX swap), "
                f"got {self.hpc_target!r}")
        # RHAN-NX discipline: the ONLY narrow path that unlocks SBR-0 is the
        # frozen-backbone clean-convergence gate. "gate_only" REQUIRES the
        # frozen backbone (and vice versa) so a config can never silently
        # train the structural gate with a moving backbone (or freeze the
        # backbone outside the gate stage).
        if self.sbr_stage == "gate_only" and not self.freeze_backbone_for_sbr0:
            raise ValueError(
                "sbr_stage='gate_only' (SBR-0) REQUIRES "
                "freeze_backbone_for_sbr0=True — the structural convergence "
                "gate is the only frozen-backbone SBR path (the narrow "
                "unlocked validate() path); every other stage fine-tunes "
                "jointly.")
        if self.freeze_backbone_for_sbr0 and self.sbr_stage != "gate_only":
            raise ValueError(
                "freeze_backbone_for_sbr0=True is only meaningful under "
                "sbr_stage='gate_only' (SBR-0), got "
                f"sbr_stage={self.sbr_stage!r}")
        # SBR-3/4 build relational + evidence machinery — they must never be
        # requested for a config whose slots aren't even active.
        if self.sbr_stage in ("relational", "uncertainty") and not self.enable_sbr:
            raise ValueError(
                f"sbr_stage={self.sbr_stage!r} requires enable_sbr=True "
                "(relational evidence / uncertainty decomposition extend the "
                "slot architecture).")
        if self.enable_sbr:
            if self.sbr_num_slots < 1:
                raise ValueError(f"sbr_num_slots must be >= 1, got {self.sbr_num_slots}")
            if self.sbr_slot_dim < 1:
                raise ValueError(f"sbr_slot_dim must be >= 1, got {self.sbr_slot_dim}")
            if self.sbr_slot_iters < 1:
                raise ValueError(f"sbr_slot_iters must be >= 1, got {self.sbr_slot_iters}")
        if self.enable_iwm:
            raise ValueError(
                "enable_iwm (Pillar 4) is scaffold-only in this refactor and "
                "MUST remain False. NullWorldModel is the safe no-op default.")
        if self.hpc_num_levels < 0:
            raise ValueError(f"hpc_num_levels must be >= 0, got {self.hpc_num_levels}")
        if self.enable_hpc and self.hpc_num_levels > 1:
            raise ValueError(
                "hpc_num_levels > 1 is NOT implemented in this pass. The "
                "roadmap requires ONE level per validation cycle (never add "
                "two levels in the same cycle). Level 1 (orientation) wiring "
                "comes only after level 0 (edge map) is validated.")
        if self.max_foraging_steps < 1:
            raise ValueError("max_foraging_steps must be >= 1")

    # ── Serialization ────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RHANNextConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown RHANNextConfig fields: {sorted(unknown)}")
        cfg = cls(**{k: v for k, v in d.items() if k in known})
        cfg.validate()
        return cfg

    def __post_init__(self):
        self.validate()

    def __repr__(self) -> str:
        flags = []
        if self.enable_ais:
            flags.append("AIS")
        if self.enable_hpc:
            flags.append(f"HPC(L={self.hpc_num_levels})")
        if self.enable_sbr:
            flags.append("SBR")
        if self.enable_iwm:
            flags.append("IWM")
        return f"RHANNextConfig([{','.join(flags) or 'v12-equivalent'}])"
