"""
RHANNext — the refactored, pillar-composable successor to RHAN-v12.

Design contract (enforced by tests/test_config_backward_compat.py):

  * RHANNext subclasses the FROZEN RHANv12 (phase1_training/model_rhan_v12.py
    is never modified). With the DEFAULT config (all pillars off) the state
    dict is byte-identical to RHANv12's and `forward` delegates to the exact
    v12 implementation — a v12 checkpoint loads 1:1, and the existing
    eval/pipeline keeps working unchanged.
  * New pillar components are added ONLY as new submodules behind
    RHANNextConfig toggles, so each mechanism can be isolated with an on/off
    test (project lesson #3: never add multiple mechanisms simultaneously).
  * Every new loss-bearing path (reconstruction, gaze policy, precision
    modulator, HPC stack) has an automated gradient-reachability test
    (project lesson #1).

Pillars:
  * AIS (Pillar 2, Stage 1)   — InformationGainGazePolicy + EntropyGatedHalting
                                + GlobalPrecisionModulator, gated by enable_ais.
  * HPC (Pillar 1, Stage 2)   — HierarchicalPredictiveStack (1 level), gated
                                by enable_hpc / hpc_num_levels.
  * SBR (Pillar 3) / IWM (Pillar 4) — scaffold only; NullWorldModel is always
                                wired as the safe no-op; enable_sbr/iwm
                                validate to an error.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# The frozen v12 chain inserts phase1_training on sys.path on import; we do the
# same explicitly so this package works regardless of the caller's cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_P1_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "phase1_training"))
for _p in (_P1_DIR, os.path.abspath(os.path.join(_THIS_DIR, ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_rhan_v12 import RHANv12                      # frozen backbone
from model_rhan_v10 import foveal_sample                # frozen helper

from rhan_core.beliefs.vector_belief import VectorBeliefState
from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.world_model.null_world_model import NullWorldModel


class RHANNext(RHANv12):
    """
    RHAN-Next: pillar-composable successor to RHAN-v12.

    Args:
        config: RHANNextConfig — the ONLY configuration object. Defaults to
                RHANNextConfig() (v12-equivalent). Alternative: pass config
                fields as keyword arguments (RHANNext(enable_ais=True, ...)).
    """

    def __init__(self, config: Optional[RHANNextConfig] = None, **kwargs):
        self.config = config if config is not None else RHANNextConfig(**kwargs)
        self.config.validate()
        super().__init__(**self.config.v12_kwargs())
        self._build_pillars()

    # ────────────────────────────────────────────────────────────────────────
    # Pillar construction (Stages 1-2 wire AIS / HPC here)
    # ────────────────────────────────────────────────────────────────────────
    def _build_pillars(self):
        # Pillar 4 (IWM): always present as a safe no-op (zero params/buffers,
        # so the default state dict stays identical to RHANv12's).
        self.world_model = NullWorldModel()

    @property
    def pillars_active(self) -> bool:
        """True when any implemented pillar is enabled (non-default path)."""
        return self.config.enable_ais or (
            self.config.enable_hpc and self.config.hpc_num_levels >= 1)

    # ────────────────────────────────────────────────────────────────────────
    # Forward — default config delegates EXACTLY to the frozen v12 path.
    # ────────────────────────────────────────────────────────────────────────
    def forward(self, x, return_trajectory=False):
        """v12-compatible forward. Default config: byte-for-byte v12."""
        if not self.pillars_active:
            return super().forward(x, return_trajectory=return_trajectory)
        # Stage 1+ replaces this branch with the AIS/HPC-aware loop.
        return super().forward(x, return_trajectory=return_trajectory)

    def get_feature_vector(self, x):
        """v12-compatible 768-dim feature vector (TRADES/eval compat)."""
        if not self.pillars_active:
            return super().get_feature_vector(x)
        return super().get_feature_vector(x)
