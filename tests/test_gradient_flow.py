"""
Mandatory gradient-flow tests (project lesson #1).

Every new loss term, predictor, or policy MUST have an automated test
asserting its gradient reaches its source parameters. This file is the
regression gate for:

  * v12 reconstruction loss      -> generative_prior   (historical no-op bug)
  * AIS gaze policy step_net     -> gaze_policy        (Stage 1)
  * AIS precision modulator gain -> precision_modulator (Stage 1)
  * HPC stack predictor          -> hpc_stack          (Stage 2)

Also asserts the Stage-1 halting rule: NO step-count penalty term exists
anywhere in the loss path (the deleted halt_efficiency loss directly opposed
the project's Banach-contraction argument).
"""
import ast
import gc
import inspect
import re

import pytest
import torch
import torch.nn.functional as F

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_B, _C, _H, _W = 2, 3, 96, 96


def _model(cfg: RHANNextConfig) -> RHANNext:
    return RHANNext(config=cfg).to(_DEVICE)


def _dummy():
    torch.manual_seed(0)
    x = torch.randn(_B, _C, _H, _W, device=_DEVICE)
    y = torch.randint(0, 10, (_B,), device=_DEVICE)
    return x, y


def _assert_param_grads(model, name_fragments, must_have_grad=True):
    """Assert every parameter whose name contains a fragment got a grad."""
    got = []
    for name, p in model.named_parameters():
        if any(frag in name for frag in name_fragments):
            g = p.grad
            if g is not None and g.abs().sum().item() > 0:
                got.append(name)
    if must_have_grad:
        assert got, (
            f"no nonzero gradient reached params matching {name_fragments}. "
            "This is the #1 historical failure mode (v11/v12 detached recon). "
            "Do NOT proceed until this is fixed.")
    return got


# ────────────────────────────────────────────────────────────────────────────
# Regression: v12 reconstruction loss must reach the generative prior.
# ────────────────────────────────────────────────────────────────────────────

def test_recon_loss_reaches_generative_prior():
    """Default-config regression: recon loss is NOT a gradient no-op."""
    m = _model(RHANNextConfig())
    x, y = _dummy()
    logits, traj = m(x, return_trajectory=True)
    loss = m.get_reconstruction_loss(x, (logits, traj))
    assert loss.requires_grad
    loss.backward()
    got = _assert_param_grads(m, ["generative_prior"])
    assert got, "reconstruction loss must reach generative_prior params"
    del m
    gc.collect()


# ────────────────────────────────────────────────────────────────────────────
# Stage 1: AIS — gradient must reach the gaze policy and precision modulator.
# ────────────────────────────────────────────────────────────────────────────

def test_ais_gradient_reaches_gaze_policy_and_precision_modulator():
    cfg = RHANNextConfig(enable_ais=True, ais_halt_threshold=0.0,
                         ais_continuation_softness=8.0)
    m = _model(cfg)
    assert hasattr(m, "gaze_policy") and hasattr(m, "precision_modulator")
    x, y = _dummy()
    logits, traj = m(x, return_trajectory=True)
    assert logits.shape == (_B, 10)

    l_ce = F.cross_entropy(logits, y)
    l_recon = m.get_reconstruction_loss(x, (logits, traj))
    pi_d = traj["precisions"][-1]
    w_recon_eff = m.precision_modulator.modulate_recon_weight(0.10, pi_d)
    loss = l_ce + w_recon_eff * l_recon
    loss.backward()

    _assert_param_grads(m, ["gaze_policy.step_net"])
    _assert_param_grads(m, ["precision_modulator.gain"])
    _assert_param_grads(m, ["generative_prior"])          # recon still flows
    del m
    gc.collect()


def test_ais_policy_parameters_exist_and_init_to_v12_behavior():
    m = _model(RHANNextConfig(enable_ais=True, ais_halt_threshold=0.0))
    # step_net must own learnable params.
    n = sum(p.numel() for p in m.gaze_policy.step_net.parameters())
    assert n > 0
    # Gain must start at 1.0 (v12-equivalent behavior).
    assert abs(m.precision_modulator.gain.item() - 1.0) < 1e-6
    # step_net zero-initialized output -> scale == 1.0.
    s = torch.randn(4, 513, device=_DEVICE)
    with torch.no_grad():
        scale = 0.5 + torch.sigmoid(m.gaze_policy.step_net(s))
    assert torch.allclose(scale, torch.ones(4, 1, device=_DEVICE), atol=1e-4)
    del m
    gc.collect()


def test_ais_halting_triggers_and_is_soft():
    """EntropyGatedHalting: hard bool gate + soft differentiable gate."""
    from rhan_core.beliefs.vector_belief import VectorBeliefState
    m = _model(RHANNextConfig(enable_ais=True, ais_halt_threshold=0.5))
    u = torch.tensor([0.1, 0.9], device=_DEVICE, requires_grad=True)
    belief = VectorBeliefState(torch.randn(2, 512, device=_DEVICE),
                               uncertainty=u)
    halt = m.halt_policy.should_halt(belief, [{}])        # (B,) bool
    assert halt[0].item() is True, "u=0.1 < 0.5 must halt"
    assert halt[1].item() is False, "u=0.9 >= 0.5 must continue"
    cont = m.halt_policy.continuation(belief, [{}])       # (B,) soft
    assert cont[0].item() < 0.5 and cont[1].item() > 0.5
    # Differentiable soft gate: gradient flows through continuation.
    cont.sum().backward()
    assert belief.uncertainty().grad is not None
    del m
    gc.collect()


# ────────────────────────────────────────────────────────────────────────────
# Stage 2: HPC — gradient must reach the stack's predictor parameters.
# ────────────────────────────────────────────────────────────────────────────

def test_hpc_gradient_reaches_stack_predictor():
    cfg = RHANNextConfig(enable_hpc=True, hpc_num_levels=1)
    m = _model(cfg)
    assert hasattr(m, "hpc_stack")
    assert m.hpc_stack.feature_targets() == ["edge_map"]
    x, y = _dummy()
    logits, traj = m(x, return_trajectory=True)
    l_hpc = m.get_hpc_loss(x, (logits, traj))
    assert l_hpc.requires_grad
    loss = F.cross_entropy(logits, y) + 0.05 * l_hpc
    loss.backward()
    _assert_param_grads(m, ["hpc_stack.levels.0.fc", "hpc_stack.levels.0.decoder"])
    del m
    gc.collect()


def test_hpc_off_means_zero_hpc_loss():
    m = _model(RHANNextConfig())  # HPC off
    x, _ = _dummy()
    logits, traj = m(x, return_trajectory=True)
    l_hpc = m.get_hpc_loss(x, (logits, traj))
    assert float(l_hpc) == 0.0
    assert "hpc_errors" not in traj
    del m
    gc.collect()


# ────────────────────────────────────────────────────────────────────────────
# Stage 1 rule: NO step-count penalty anywhere in the loss path.
# ────────────────────────────────────────────────────────────────────────────

def _code_without_docstrings(source: str) -> str:
    """Strip docstrings so the scan doesn't false-positive on its own docs."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
    return ast.unparse(tree)


def test_no_step_count_penalty_in_loss_path():
    from rhan_core.gaze import halting as halting_mod
    from rhan_core.gaze import info_gain_policy as policy_mod
    import train_rhan_next as trainer_mod

    banned = [
        r"steps_used\s*/\s*max_steps",
        r"steps\s*/\s*max_steps",
        r"step_count\s*/\s*max_steps",
        r"halt_efficiency",
        r"steps_used\s*\*\s*-?1",
    ]
    for mod in (halting_mod, policy_mod):
        code = _code_without_docstrings(inspect.getsource(mod))
        for pat in banned:
            assert not re.search(pat, code), (
                f"step-count penalty pattern {pat!r} found in {mod.__name__}")

    # The trainer's loss function must not reference any step count at all.
    loss_src = _code_without_docstrings(
        inspect.getsource(trainer_mod.dynamic_trades_loss_next))
    for pat in banned:
        assert not re.search(pat, loss_src), \
            f"step-count penalty pattern {pat!r} found in the trainer loss"
    assert "steps" not in loss_src, \
        "trainer loss references 'steps' — halting must never be step-counted"
