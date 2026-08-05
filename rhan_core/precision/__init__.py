"""
Precision subpackage — Pillar 2 (Active Information-Seeking).

`PrecisionModulator` (ABC) and `GlobalPrecisionModulator` (Stage 1 concrete)
compute a per-sample precision signal (Pi_D, unsupervised) and expose it to
other components (gaze step size, halting/recurrence depth, reconstruction
loss weight) so each consumer can be isolated and tested independently.
"""

from rhan_core.precision.base import PrecisionModulator

__all__ = ["PrecisionModulator"]
