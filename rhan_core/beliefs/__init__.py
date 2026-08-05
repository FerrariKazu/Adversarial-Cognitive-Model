"""
Belief representations.

Pillars 1 & 2 operate on `VectorBeliefState` (a dense (B, D) tensor — the same
spirit as v12's belief vector `s`). Pillar 3 (SBR) will introduce
`StructuredBeliefState` (object slots + relational graph) behind the same
`BeliefState` interface so no caller changes.
"""

from rhan_core.beliefs.base import BeliefState
from rhan_core.beliefs.vector_belief import VectorBeliefState
from rhan_core.beliefs.structured_belief import StructuredBeliefState

__all__ = ["BeliefState", "VectorBeliefState", "StructuredBeliefState"]
