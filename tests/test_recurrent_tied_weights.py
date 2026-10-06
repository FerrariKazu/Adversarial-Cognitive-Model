"""
Agent C contract test — tied within-glimpse recurrence.

"Same parameter tensors reused, not copies" is asserted by IDENTITY, not
value equality: a forward hook on the ONE block instance must fire once
per iteration (a copy-based implementation would hold separate objects and
the single hook would fire once). Gradient accumulation onto the single
set and the LOCKED 2-3 range are asserted too.
"""
import pytest
import torch

from noesis_vision.models.backbone import CompactViT
from noesis_vision.models.recurrent_block import (
    TiedRecurrence,
    TransformerBlock,
    WITHIN_GLIMPSE_ITERS_RANGE,
)


def test_single_block_instance_is_the_one_used_every_iteration():
    """Identity check: hook on the single block fires num_iters times.

    If the recurrence ever held per-iteration COPIES, the one hooked
    instance would fire once — this test fails loudly.
    """
    m = CompactViT(img_size=56, num_refine_iters=3)
    fires = []
    handle = m.refinement.block.register_forward_hook(
        lambda mod, inp, out: fires.append(id(mod)))
    x = torch.randn(1, 18, 384)
    m.refinement(x, 3)
    handle.remove()
    assert len(fires) == 3
    assert len(set(fires)) == 1  # SAME object every iteration


def test_no_copies_structurally():
    """Exactly one TransformerBlock inside the recurrence — a ModuleList of
    copies would yield more."""
    m = CompactViT(img_size=56)
    blocks = [mm for mm in m.refinement.modules()
              if isinstance(mm, TransformerBlock)]
    assert len(blocks) == 1
    assert m.refinement.block is blocks[0]


def test_gradients_accumulate_onto_single_param_set():
    m = CompactViT(img_size=56, num_refine_iters=3)
    # Only the refinement participates: zero its grads via a targeted pass.
    x = torch.randn(2, 18, 384)
    out = m.refinement(x, 3)
    out.sum().backward()
    grads = [p.grad for p in m.refinement.block.parameters()]
    assert all(g is not None for g in grads), (
        "every parameter of the ONE tied set must receive gradient")
    assert all(torch.any(g != 0) for g in grads)


def test_iteration_count_changes_output_not_params():
    lo, hi = WITHIN_GLIMPSE_ITERS_RANGE
    m = CompactViT(img_size=56)
    x = torch.randn(1, 18, 384)
    out_lo = m.refinement(x, lo)
    out_hi = m.refinement(x, hi)
    assert not torch.allclose(out_lo, out_hi)   # compute differs...
    n1 = sum(p.numel() for p in m.refinement.parameters())
    assert n1 == sum(p.numel() for p in m.refinement.block.parameters())


def test_locked_range_enforced_at_model_side():
    m = CompactViT(img_size=56)
    x = torch.randn(1, 18, 384)
    with pytest.raises(ValueError, match="LOCKED range"):
        m.refinement(x, 1)
    with pytest.raises(ValueError, match="LOCKED range"):
        m.refinement(x, 4)


def test_tied_recurrence_is_a_distinct_type():
    """Guard against someone swapping in an untied stack silently."""
    m = CompactViT(img_size=56)
    assert isinstance(m.refinement, TiedRecurrence)
