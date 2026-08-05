"""
Gaze subpackage — Pillar 2 (Active Information-Seeking).

`InformationGainGazePolicy` (Stage 1) selects fixations that maximize an
expected reduction in belief uncertainty; `EntropyGatedHalting` (Stage 1)
stops evidence gathering once uncertainty drops below a threshold. Both are
gated behind RHANNextConfig.enable_ais and are OFF in the default
(v12-equivalent) model. Stage 0 only ships the GazePolicy ABC.
"""

from rhan_core.gaze.base import GazePolicy
from rhan_core.gaze.info_gain_policy import InformationGainGazePolicy
from rhan_core.gaze.halting import EntropyGatedHalting

__all__ = ["GazePolicy", "InformationGainGazePolicy", "EntropyGatedHalting"]
