"""
Optimizer infrastructure for the RHAN-Next pillar system.

Generation 0 (RHAN-NX): generalizes Stage 2's two-group SGD fix (backbone
lr=0.003, HPC stack lr=0.02 = 0.003 * 6.67, per-group grad clip) to N
arbitrary named groups. Every new trainable component (SBR slots, AIS-v2
candidate-evaluation head, belief-HPC predictor, relational/evidence heads)
gets its OWN optimizer group through `OptimizerGroupRegistry` — the standing
architectural rule that replaced the one-off HPC starvation fix.
"""

from rhan_core.optim.multi_group_optimizer import (
    OptimizerGroupRegistry,
    default_group_spec,
)

__all__ = ["OptimizerGroupRegistry", "default_group_spec"]