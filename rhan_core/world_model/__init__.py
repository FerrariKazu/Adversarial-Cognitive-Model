"""
Internal world model subpackage — Pillar 4, scaffold only.

`NullWorldModel` is the default and is wired into every RHANNext instance so
every downstream call site stays functional; it returns its input unchanged
and logs a debug notice. `enable_iwm` MUST remain False until a real world
model is implemented.
"""

from rhan_core.world_model.base import WorldModel
from rhan_core.world_model.null_world_model import NullWorldModel

__all__ = ["WorldModel", "NullWorldModel"]
