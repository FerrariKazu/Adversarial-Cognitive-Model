"""RHAN-NXA Gen-2 Foundation slice.

Reference implementation of the Gen-2 *foundation* mechanisms only (K9,
K1, K4, K5, K6, K7, K8). All mechanisms are gated behind pre-registered
experiment IDs; a default config is a legal control arm, flipping anything
requires its ID.

This package is a reference layer. It reuses Gen-1 interfaces exactly
(shared patch embedder for gist, per-group optimizer registry, the
CompoundUpdateNet gradient contract) and never modifies them. Deferred
mechanisms (policy zoo, multi-scale pyramid, memory, learning halting,
Gen-3) are deliberately NOT implemented.
"""

from __future__ import annotations

from .eot import eot_pgd_attack, build_autoattack_loss_wrapper, eot_sanity_loss
from .flags import RHANGen2Config, GATED_FLAGS, REJECTED_OUTRIGHT, G2_EXPERIMENT_IDS, G2_OUT_OF_SCOPE_IDS
from .gist import FixedGistEncoder
from .precision import PrecisionArm, PrecisionField, legacy_1_minus_u_control
from .recipe import build_adamw, AdamWGroups, PhaseRecipe
from .update_net_v2 import BeliefUpdaterV2, FusionMode, adapt_magnitude
from .ema import EMATargetEncoder

__version__ = "0.1.0"
__all__ = [
    "RHANGen2Config",
    "GATED_FLAGS",
    "REJECTED_OUTRIGHT",
    "FixedGistEncoder",
    "PrecisionField",
    "PrecisionArm",
    "legacy_1_minus_u_control",
    "BeliefUpdaterV2",
    "FusionMode",
    "adapt_magnitude",
    "EMATargetEncoder",
    "build_adamw",
    "AdamWGroups",
    "PhaseRecipe",
    "eot_pgd_attack",
]
