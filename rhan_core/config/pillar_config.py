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
    enable_sbr: bool = False      # Pillar 3 — MUST remain False; scaffold only
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

    # ── Pillar 2 (AIS) knobs ─────────────────────────────────────────────────
    ais_halt_threshold: float = 0.35   # halt when belief uncertainty < this
    ais_continuation_softness: float = 8.0  # steepness of the soft gate
    ais_base_step: float = 0.20         # v12's fixed base gaze step
    ais_precision_step_range: float = 0.30  # v12's precision-scaled range

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
        if self.enable_sbr:
            raise ValueError(
                "enable_sbr (Pillar 3) is scaffold-only in this refactor and "
                "MUST remain False. StructuredBeliefState raises "
                "NotImplementedError on use.")
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
