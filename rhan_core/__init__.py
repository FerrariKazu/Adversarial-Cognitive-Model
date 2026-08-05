"""
rhan_core — RHAN-Next: a properly separated package for the RHAN research
program, designed so RHAN-v12's training pipeline can adopt it without
breaking, and so four research pillars can be grown independently
without the package ever needing to be redesigned again:

    Pillar 1 — Hierarchical Predictive Coding (HPC)     [implemented, 1 level]
    Pillar 2 — Active Information-Seeking (AIS)          [implemented]
    Pillar 3 — Structured Belief Representation (SBR)    [scaffold only]
    Pillar 4 — Internal World Model (IWM)                [scaffold only]

Design rules enforced across the package (see docs/ARCHITECTURE.md):

  * Gradient-flow bugs are the #1 historical failure mode in this project
    (v11/v12 stored a detached reconstruction error for two generations).
    Every new loss term / predictor / policy in this package has an
    automated gradient-reachability test (tests/test_gradient_flow.py).
  * "Trains without crashing" is not "validated". Every stage has a
    distinct code-complete checkbox and a separate validated checkbox.
  * New mechanisms are added one at a time and isolated with on/off
    toggles (RHANNextConfig), never bundled and attributed jointly.
  * The DEFAULT config (all pillars off) reproduces RHAN-v12's forward
    pass shape-for-shape (tests/test_config_backward_compat.py).

Directory layout:

    beliefs/            BeliefState hierarchy (Vector = Pillars 1&2, Structured = P3)
    predictive_coding/  LevelPredictor / ErrorUnit ABCs, hierarchical stack (P1),
                        feature-target extractors (edge/orientation/shape)
    gaze/               GazePolicy ABC, InformationGainGazePolicy, EntropyGatedHalting (P2)
    precision/          PrecisionModulator ABC + GlobalPrecisionModulator (P2)
    world_model/        WorldModel ABC + NullWorldModel no-op (P4 scaffold)
    config/             RHANNextConfig dataclass (all pillar toggles + v12 hyperparams)
    model.py            RHANNext(nn.Module) — composes everything above
"""

__version__ = "0.1.0"

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext

__all__ = ["RHANNext", "RHANNextConfig", "__version__"]
