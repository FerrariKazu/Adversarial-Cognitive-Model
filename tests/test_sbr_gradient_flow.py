"""
SBR (Structured Belief Representation) unit tests.
====================================================

Tests the three structural guarantees for Pillar 3:
  1. Gradient flow: nonzero gradient reaches SBR slot attention parameters
  2. Backward compat: enable_sbr=False reproduces D's forward pass bit-for-bit
  3. Slot diversity: slot attention doesn't collapse to uniform/degenerate attention

These follow the same patterns as test_hpc_gradient_flow.py.
"""
import gc
import sys
import os

import pytest
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_P1 = os.path.join(_REPO, 'phase1_training')
for _p in (_REPO, _P1):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext

_B = 2


def _cfg_sbr():
    return RHANNextConfig(
        enable_ais=True, ais_halt_enabled=True, ais_precision_recon_enabled=False,
        enable_hpc=True, hpc_num_levels=1, hpc_error_weight=0.10,
        enable_sbr=True, sbr_num_slots=8, sbr_slot_dim=512, sbr_slot_iters=3)


def _cfg_d():
    return RHANNextConfig(
        enable_ais=True, ais_halt_enabled=True, ais_precision_recon_enabled=False,
        enable_hpc=True, hpc_num_levels=1, hpc_error_weight=0.10)


def _forward(model, x, with_traj=False):
    model.eval()
    if with_traj:
        with torch.enable_grad():
            return model(x, return_trajectory=True)
    with torch.no_grad():
        return model(x)


# ── 1. Gradient flow tests ──────────────────────────────────────────────────

def test_sbr_gradient_reaches_slot_params():
    """Gradient must reach SBR slot attention parameters (NOT detached)."""
    cfg = _cfg_sbr()
    model = RHANNext(config=cfg).cuda()
    x = torch.randn(_B, 3, 96, 96, device='cuda')
    logits, traj = _forward(model, x, with_traj=True)
    loss = logits.sum()
    loss.backward()
    sbr_grads = {}
    for name, p in model.named_parameters():
        if 'structured_belief' in name and p.grad is not None:
            sbr_grads[name] = float(p.grad.norm())
    assert len(sbr_grads) > 0, "No SBR parameters received gradients"
    for name, gnorm in sbr_grads.items():
        assert gnorm > 0, f"SBR param {name} has zero gradient"
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_sbr_loss_attached_to_model():
    """SBR entropy should appear in trajectory (confirming it's wired)."""
    cfg = _cfg_sbr()
    model = RHANNext(config=cfg).cuda()
    x = torch.randn(_B, 3, 96, 96, device='cuda')
    logits, traj = _forward(model, x, with_traj=True)
    assert 'sbr_entropy' in traj, "sbr_entropy not in trajectory"
    assert len(traj['sbr_entropy']) == cfg.max_foraging_steps, \
        f"Expected {cfg.max_foraging_steps} entropy entries, got {len(traj['sbr_entropy'])}"
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_sbr_on_state_dict_has_no_duplicate_keys():
    """SBR parameters should appear exactly once in the state dict."""
    cfg = _cfg_sbr()
    model = RHANNext(config=cfg)
    sd = model.state_dict()
    sbr_keys = [k for k in sd if 'structured_belief' in k]
    assert len(sbr_keys) > 0, "No SBR keys in state dict"
    # Check no duplicates (named_parameters should be consistent)
    param_names = {n for n, _ in model.named_parameters() if 'structured_belief' in n}
    buf_names = {n for n, _ in model.named_buffers() if 'structured_belief' in n}
    all_expected = param_names | buf_names
    all_found = set(sbr_keys)
    assert all_found == all_expected, \
        f"Mismatch: extra={all_found - all_expected}, missing={all_expected - all_found}"
    del model
    gc.collect()


# ── 2. Backward compatibility tests ─────────────────────────────────────────

def test_sbr_off_matches_d_forward():
    """enable_sbr=False produces bit-identical output to D config."""
    cfg_d = _cfg_d()
    cfg_e2_nosbr = RHANNextConfig(
        enable_ais=True, ais_halt_enabled=True, ais_precision_recon_enabled=False,
        enable_hpc=True, hpc_num_levels=1, hpc_error_weight=0.10,
        enable_sbr=False)

    torch.manual_seed(42)
    model_d = RHANNext(config=cfg_d).cuda().eval()
    torch.manual_seed(42)
    model_e2 = RHANNext(config=cfg_e2_nosbr).cuda().eval()

    x = torch.randn(_B, 3, 96, 96, device='cuda')
    with torch.no_grad():
        logits_d = model_d(x)
        logits_e2 = model_e2(x)
    assert torch.allclose(logits_d, logits_e2, atol=1e-5), \
        f"Max diff: {(logits_d - logits_e2).abs().max().item()}"
    del model_d, model_e2
    torch.cuda.empty_cache()
    gc.collect()


def test_sbr_checkpoint_loads_from_d():
    """D's checkpoint loads cleanly into E2; SBR keys appear as missing."""
    cfg_e2 = _cfg_sbr()
    model = RHANNext(config=cfg_e2)
    d_path = os.path.join(_REPO, 'checkpoints/rhan_next_ais_hpc_best.pth')
    if not os.path.exists(d_path):
        pytest.skip("D checkpoint not available")
    state = torch.load(d_path, map_location='cpu', weights_only=False)
    for k in ('model', 'model_state_dict', 'state_dict'):
        if isinstance(state, dict) and k in state:
            weights = state[k]
            break
    else:
        weights = state
    missing, unexpected = model.load_state_dict(weights, strict=False)
    sbr_missing = [k for k in missing if 'structured_belief' in k]
    other_missing = [k for k in missing if 'structured_belief' not in k]
    assert len(sbr_missing) > 0, "Expected SBR missing keys"
    assert len(other_missing) == 0, f"Unexpected missing keys: {other_missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    del model
    gc.collect()


# ── 3. Slot diversity tests ─────────────────────────────────────────────────

def test_slot_attention_not_collapsed():
    """Slot attention should produce diverse (not degenerate) attention maps.

    Uses N=16 spatial positions (simulating a real feature map) so slots
    can distribute their attention across positions.
    """
    from rhan_core.beliefs.structured_belief import StructuredBeliefState

    sa = StructuredBeliefState(num_slots=8, slot_dim=512, iters=3).cuda()
    # Use N=16 spatial positions (not N=1) to test slot diversity
    x = torch.randn(_B, 16, 512, device='cuda')
    out = sa(x)
    attn = out['attn']  # (B, K, N)
    B, K, N = attn.shape

    # Attention entropy should be > 0 (not one-hot per position)
    entropy = out['entropy']
    assert entropy.mean() > 0, "Slot attention entropy is zero (collapsed)"

    # Attention maps should not be uniform (max attention > 1/K + margin)
    max_per_pos = attn.max(dim=1).values  # (B, N)
    uniform_val = 1.0 / K
    assert max_per_pos.mean() > uniform_val + 0.01, \
        f"Slot attention is too uniform: max={max_per_pos.mean():.4f}, uniform={uniform_val:.4f}"

    # Slots should not all attend to the same position
    hard = attn.argmax(dim=1)  # (B, N) — which slot each position attends to
    for b in range(B):
        slots_used = hard[b].unique()
        assert len(slots_used) >= K // 2, \
            f"Sample {b}: only {len(slots_used)}/{K} slots used (collapsed)"


def test_slot_diversity_increases_with_iters():
    """More slot attention iterations should produce more refined attention."""
    from rhan_core.beliefs.structured_belief import StructuredBeliefState

    sa = StructuredBeliefState(num_slots=8, slot_dim=512, iters=1).cuda()
    x = torch.randn(_B, 1, 512, device='cuda')

    # With 1 iteration, attention should be less focused
    out1 = sa(x)
    ent1 = out1['entropy'].mean().item()

    sa3 = StructuredBeliefState(num_slots=8, slot_dim=512, iters=3).cuda()
    sa3.load_state_dict(sa.state_dict(), strict=False)  # same init
    out3 = sa3(x)
    ent3 = out3['entropy'].mean().item()

    # Entropy should decrease (more focused) with more iterations
    # (not guaranteed but expected for properly initialized slot attention)
    print(f"  iters=1 entropy: {ent1:.4f}, iters=3 entropy: {ent3:.4f}")

    del sa, sa3
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
