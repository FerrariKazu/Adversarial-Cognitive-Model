"""Gen-2 experiment gating and configuration.

Implements:
  * `RHANGen2Config` — a frozen dataclass whose fields are the
    pre-registered Gen-2 experiment IDs. The default configuration is a
    legal control arm (no non-default mechanism active). Flipping any
    mechanism on requires its exact experiment ID.
  * `GATED_FLAGS` — every non-default mechanism keyed to its pre-registered
    experiment ID; flipping it without the ID must raise.
  * `REJECTED_OUTRIGHT` — carried verbatim from Gen-1. These mechanisms
    are prohibited from code execution and have no flag.
"""

from __future__ import annotations

import dataclasses
from typing import FrozenSet, Mapping, Optional

#: Pre-registered Gen-2 experiment IDs for the foundation slice.
#: Every non-default mechanism in the spec must be added here exactly once.
G2_EXPERIMENT_IDS = frozenset(
    {
        # K9 / K7 optimization foundation
        "G2-K9-optimization",
        "G2-K7-optimization-AdamW",
        "G2-K7-optimization-SGD",
        "G2-K7-optimization-warmup",
        # K1 fixed gist
        "G2-K1-gist-fixed",
        "G2-K1-gist-learned",
        "G2-K1-no-gist",
        "G2-K1-gist-dropout",
        # K4 precision field family
        "G2-K4-precision-const",
        "G2-K4-precision-1minusu",
        "G2-K4-precision-learned-scalar",
        "G2-K4-precision-learned-spatial",
        "G2-K4-precision-channelwise",
        "G2-K4-precision-policyconditioned",
        "G2-K4-precision-collapse-diag",
        # K5 observed-feature fusion
        "G2-K5-fusion-concat",
        "G2-K5-fusion-gated",
        "G2-K5-fusion-FILM",
        "G2-K5-observed-error-off",
        "G2-K5-observed-error-on",
        # K6 spatial error
        "G2-K6-spatial-error-on",
        "G2-K6-spatial-error-off",
        "G2-K6-spatial-error-multiscale",
        # K7 belief update
        "G2-K7-belief-bounded",
        "G2-K7-belief-adaptive",
        "G2-K7-belief-recurrent",
        "G2-K7-belief-unbounded",
        # K7 EMA
        "G2-K7-ema-off",
        "G2-K7-ema-on",
        # K8 robustness
        "G2-K8-robustness-PGD",
        "G2-K8-robustness-AutoAttack-EOT",
        "G2-K8-n_eot-1",
        "G2-K8-autoattack-point",
    }
)

#: Gen-1 mechanisms that are REJECTED OUTRIGHT. Carried verbatim from
#: `noesis_vision/core/schema.py`'s `REJECTED_OUTRIGHT`; duplicated here so
#: `gen2_foundation` is self-contained and cannot silently re-admit a
#: rejected mechanism.
REJECTED_OUTRIGHT: FrozenSet[str] = frozenset(
    {
        "pixel-space reconstruction as E_t's target (Part 1.B)",
        "edge-map / HPC-style hand-designed feature targets (Part 1.B)",
        "legacy 16-slot Slot Attention implementation (Part 1.D — REJECTED "
        "as implemented; the concept is EXPERIMENTAL CANDIDATE, DEFERRED, "
        "2-4 slots only if revisited)",
        "external retrieval / RAG memory (Part 1.H)",
        "AIS-v1's relocated-Eq.-II gaze code (Part 5 port table)",
        "raw addition z_t + lambda*Pi*E_t — superseded by UpdateNet (Part 1.B)",
    }
)

#: Gen-2 experiment IDs that are authorized NON-DEFAULT mechanisms. Setting
#: any of these True without its ID raises at config construction (Gated
#: flag G10). ID strings are frozen so downstream code can assert against
#: them.
GATED_FLAGS: Mapping[str, str] = {
    # --- K9 / K7 optimization foundation ---
    "enable_optimizer_adamw": "G2-K7-optimization-AdamW",
    "enable_optimizer_sgd": "G2-K7-optimization-SGD",
    "enable_optimizer_warmup": "G2-K7-optimization-warmup",
    # --- K1 fixed gist ---
    "gist_fixed": "G2-K1-gist-fixed",
    "gist_learned": "G2-K1-gist-learned",
    "no_gist": "G2-K1-no-gist",
    "gist_dropout": "G2-K1-gist-dropout",
    # --- K4 precision field family ---
    "precision_const": "G2-K4-precision-const",
    "precision_1minusu": "G2-K4-precision-1minusu",
    "precision_learned_scalar": "G2-K4-precision-learned-scalar",
    "precision_learned_spatial": "G2-K4-precision-learned-spatial",
    "precision_channelwise": "G2-K4-precision-channelwise",
    "precision_policyconditioned": "G2-K4-precision-policyconditioned",
    "precision_collapse_diagnostics": "G2-K4-precision-collapse-diag",
    # --- K5 observed-feature fusion ---
    "fusion_concat": "G2-K5-fusion-concat",
    "fusion_gated": "G2-K5-fusion-gated",
    "fusion_film": "G2-K5-fusion-FILM",
    "observed_error_off": "G2-K5-observed-error-off",
    "observed_error_on": "G2-K5-observed-error-on",
    # --- K6 spatial error ---
    "spatial_error_on": "G2-K6-spatial-error-on",
    "spatial_error_off": "G2-K6-spatial-error-off",
    "spatial_error_multiscale": "G2-K6-spatial-error-multiscale",
    # --- K7 belief update ---
    "belief_bounded": "G2-K7-belief-bounded",
    "belief_adaptive": "G2-K7-belief-adaptive",
    "belief_recurrent": "G2-K7-belief-recurrent",
    "belief_unbounded": "G2-K7-belief-unbounded",
    # --- K7 EMA ---
    "ema_on": "G2-K7-ema-on",
    "ema_off": "G2-K7-ema-off",
    # --- K8 robustness ---
    "robustness_pgd": "G2-K8-robustness-PGD",
    "robustness_autoattack_eot": "G2-K8-robustness-AutoAttack-EOT",
    "eot_n_eot_1": "G2-K8-n_eot-1",
    "autoattack_integration_point": "G2-K8-autoattack-point",
}


#: Pre-registered Gen-2 experiment IDs that are explicitly out of scope for
#: the foundation slice and therefore cannot be activated through this
#: package. These are NOT defaults and cannot be flipped.
G2_OUT_OF_SCOPE_IDS: FrozenSet[str] = frozenset(
    {
        # Policy zoo / K3 expansion
        "G2-K3-policy",
        "G2-K3a-information-gain",
        # Multi-scale pyramid
        "G2-K1c-gist-multi-scale",
        # Memory
        "G2-K7a-spatial-working-memory",
        "G2-K7b-spatial-memory",
        # Halting
        "G2-K8a-variable-glimpses",
        # Gen-3
        "G3-K1-world-state",
        "G3-K2-object-scene",
        "G3-K3-persistent-memory",
        "G3-K4-counterfactual",
        "G3-K5-reasoning",
        "G3-K6-self-supervised",
        "G3-K7-hierarchical-perception",
        "G3-K8-graduated-training",
        "G3-K9-robustness",
        # Deferred
        "G2-K11-a-calibration",
    }
)


@dataclasses.dataclass(frozen=True)
class RHANGen2Config:
    """Frozen Gen-2 foundation configuration.

    Every field is a pre-registered experiment ID. Defaults are FULLY
    legal control arms. Setting a non-default field requires that its
    experiment ID be present in `G2_EXPERIMENT_IDS` and in
    `GATED_FLAGS`.

    Rules enforced at construction:
      * a mechanism field may only be True if its experiment ID is present
        in `GATED_FLAGS` and `G2_EXPERIMENT_IDS`;
      * rejected-outright mechanisms (Gen-1) have NO field — they cannot be
        enabled;
      * out-of-scope IDs cannot be set;
      * one-parameter-one-group enforcement is a recipe-level invariant,
        not a config field.
    """

    # --- optimizer foundation ---
    enable_optimizer_adamw: bool = False
    enable_optimizer_sgd: bool = False
    enable_optimizer_warmup: bool = False

    # --- K1 gist ---
    # no_gist is the Gen-1 parity control (always on by default).
    no_gist: bool = True
    gist_fixed: bool = False
    gist_learned: bool = False
    gist_dropout: bool = False

    # --- K4 precision field family ---
    precision_const: bool = False
    precision_1minusu: bool = False
    precision_learned_scalar: bool = False
    precision_learned_spatial: bool = False
    precision_channelwise: bool = False
    precision_policyconditioned: bool = False
    precision_collapse_diagnostics: bool = False

    # --- K5 observed-feature fusion ---
    fusion_concat: bool = False
    fusion_gated: bool = False
    fusion_film: bool = False
    observed_error_off: bool = False
    observed_error_on: bool = False

    # --- K6 spatial error ---
    spatial_error_on: bool = False
    spatial_error_off: bool = False
    spatial_error_multiscale: bool = False

    # --- K7 belief update ---
    belief_bounded: bool = False
    belief_adaptive: bool = False
    belief_recurrent: bool = False
    belief_unbounded: bool = False

    # --- K7 EMA ---
    ema_on: bool = False
    ema_off: bool = False

    # --- K8 robustness ---
    robustness_pgd: bool = False
    robustness_autoattack_eot: bool = False
    eot_n_eot_1: bool = False
    autoattack_integration_point: bool = False

    def __post_init__(self) -> None:
        # 1) every non-default mechanism requires its exact experiment ID.
        for field_name, exp_id in GATED_FLAGS.items():
            if getattr(self, field_name) and exp_id not in G2_EXPERIMENT_IDS:
                raise ValueError(
                    f"G2-HANDOFF: mechanism '{field_name}' requires "
                    f"experiment ID '{exp_id}' (present in G2_EXPERIMENT_IDS). "
                    f"Cannot activate '{field_name}' without its registered ID. "
                    f"Refusing silent re-admission of an unregistered ID."
                )

        # 2) rejected-outright mechanisms have NO field (blocked at schema).
        for exp_id in self._active_ids():
            if exp_id in REJECTED_OUTRIGHT:
                raise ValueError(
                    f"G2-HANDOFF: experiment ID '{exp_id}' is REJECTED OUTRIGHT "
                    f"in Gen-1 and cannot be activated."
                )

        # 3) out-of-scope IDs cannot be set.
        for exp_id in self._active_ids():
            if exp_id in G2_OUT_OF_SCOPE_IDS:
                raise ValueError(
                    f"G2-HANDOFF: experiment ID '{exp_id}' is out of scope for "
                    f"the Gen-2 foundation slice (deferred). Cannot activate."
                )

    # ---- helpers ----

    def _active_ids(self) -> frozenset:
        return frozenset(
            exp_id
            for field_name, exp_id in GATED_FLAGS.items()
            if getattr(self, field_name)
        )

    def is_control_arm(self) -> bool:
        """True when the configuration is a legal control arm (the baseline).

        The Gen-1 parity arm ('no_gist', experiment ID G2-K1-no-gist) is
        the original RHAN-NXA baseline — it is legal and always available.
        Only mechanisms beyond 'no_gist' (e.g. gist_fixed, optimizer_adamw)
        are treated as non-default machine mechanisms.
        """
        active = self._active_ids()
        # no_gist (the Gen-1 parity arm) is the canonical baseline; it is
        # never a 'non-default' machine mechanism.
        return active <= frozenset(['G2-K1-no-gist'])

    def active_mechanism_ids(self) -> frozenset:
        return self._active_ids()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RHANGen2Config":
        return cls(**d)

    def __repr__(self) -> str:
        active = self._active_ids()
        return (
            f"RHANGen2Config(control={self.is_control_arm()}, "
            f"active={len(active)} mechanism(s): {sorted(active)})"
        )
