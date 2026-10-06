"""
Belief-level HPC (D3) — gradient-flow + backward-compat tests.
================================================================================

The belief-HPC swap target (predict belief_{t+1} from belief_t instead of the
edge-map pixel target) gets the same HARD treatment every new loss path in
this project got after the detached-recon lesson:

  1. GRADIENT FLOW (hard NOT-detached assertion): the per-sample error tensor
     in the trajectory must carry a grad_fn (attached), and backward through
     the FULL model loss must reach hpc_belief.predictor.* params. The wiring
     contract (model.py): the prediction input (prev belief) is ATTACHED,
     the target (current belief) is DETACHED — bottom-up actuals never
     contribute gradients, exactly like the pixel extractor's detached
     edge-map target.
  2. DELAYED-BY-ONE WIRING: hpc_errors appears in the trajectory from step 1
     onward (predict s_{t-1} -> compare with s_t), length == max_steps - 1.
  3. BACKWARD COMPAT: hpc_target="pixel" (the ONLY value every pre-RHAN-NX
     checkpoint carries) must NOT build hpc_belief at all — state dicts stay
     compatible; the default config reproduces D's forward bit-for-bit
     (already asserted by test_hpc_disable_backward_compat for the pixel
     stack; here we assert the belief module is absent in pixel mode).
  4. OPTIMIZER GROUP: the belief predictor's params are claimed by the "hpc"
     Generation-0 group (default_group_spec) — the pre-flight |dW| target.
"""
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1_training"))

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext
from rhan_core.optim.multi_group_optimizer import default_group_spec
from rhan_core.predictive_coding.hpc_belief_level import HPCBeliefLevel

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_B = 2


def _cfg_belief():
    return RHANNextConfig(enable_ais=True, ais_variant="halting_only",
                          ais_halt_enabled=True,
                          ais_precision_recon_enabled=False,
                          enable_hpc=True, hpc_num_levels=1,
                          hpc_error_weight=0.10, hpc_target="belief")


def _cfg_pixel():
    return RHANNextConfig(enable_ais=True, ais_variant="halting_only",
                          ais_halt_enabled=True,
                          ais_precision_recon_enabled=False,
                          enable_hpc=True, hpc_num_levels=1,
                          hpc_error_weight=0.10, hpc_target="pixel")


# ── 1. Gradient flow (HARD assertions) ──────────────────────────────────────

def test_hpc_belief_error_attached_not_detached():
    """The belief-prediction error tensor in the trajectory is ATTACHED."""
    model = RHANNext(config=_cfg_belief()).to(_DEVICE).eval()
    x = torch.randn(_B, 3, 96, 96, device=_DEVICE)
    with torch.enable_grad():
        _, traj = model(x, return_trajectory=True)
    errs = traj.get("hpc_errors") or []
    assert errs, "hpc_errors missing from trajectory (belief target not wired)"
    for e in errs:
        assert e.requires_grad and e.grad_fn is not None, \
            "belief-HPC error is DETACHED — the predictor is untrainable " \
            "(the project's #1 historical failure mode, now asserted)"
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_hpc_belief_gradient_reaches_predictor():
    """Full-model backward reaches hpc_belief.predictor params (nonzero)."""
    model = RHANNext(config=_cfg_belief()).to(_DEVICE)
    x = torch.randn(_B, 3, 96, 96, device=_DEVICE)
    y = torch.randint(0, 10, (_B,), device=_DEVICE)
    model.train()
    with torch.enable_grad():
        logits, traj = model(x, return_trajectory=True)
        loss = (torch.nn.functional.cross_entropy(logits, y)
                + 0.10 * model.get_hpc_loss(x, (logits, traj)))
        loss.backward()
    grads = {n: float(p.grad.norm()) for n, p in model.named_parameters()
             if "hpc_belief" in n and p.grad is not None}
    assert grads, "no hpc_belief params received gradients"
    for n, g in grads.items():
        assert g > 0, f"hpc_belief param {n} has ZERO gradient"
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_hpc_belief_loss_alone_drives_predictor():
    """L_hpc alone (no CE) must move the predictor — the swap target's own
    gradient path, the exact thing the pixel HPC starvation killed."""
    model = RHANNext(config=_cfg_belief()).to(_DEVICE)
    model.eval()
    x = torch.randn(_B, 3, 96, 96, device=_DEVICE)
    with torch.enable_grad():
        _, traj = model(x, return_trajectory=True)
        l_hpc = model.get_hpc_loss(x, (None, traj))
        assert l_hpc.requires_grad
        l_hpc.backward()
    grads = {n: float(p.grad.norm()) for n, p in model.named_parameters()
             if "hpc_belief" in n and p.grad is not None
             and p.grad.norm() > 0}
    assert grads, "L_hpc backward did not reach hpc_belief params"
    del model
    torch.cuda.empty_cache()
    gc.collect()


# ── 2. Delayed-by-one wiring ────────────────────────────────────────────────

def test_hpc_belief_delayed_wiring_length():
    """hpc_errors has max_steps-1 entries (predict s_{t-1} vs s_t)."""
    model = RHANNext(config=_cfg_belief()).to(_DEVICE).eval()
    x = torch.randn(_B, 3, 96, 96, device=_DEVICE)
    with torch.no_grad():
        _, traj = model(x, return_trajectory=True)
    errs = traj.get("hpc_errors") or []
    assert len(errs) == model.max_steps - 1, \
        f"expected {model.max_steps-1} delayed belief errors, got {len(errs)}"
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_hpc_belief_module_step_contract():
    """HPCBeliefLevel.step: input attached, target detached, error attached."""
    level = HPCBeliefLevel(proj_dim=64).to(_DEVICE)
    prev = torch.randn(_B, 64, device=_DEVICE, requires_grad=True)
    cur = torch.randn(_B, 64, device=_DEVICE).detach()
    pred, err, err_map = level.step(prev, cur)
    assert pred.shape == (_B, 64) and err.shape == (_B,)
    assert err.requires_grad
    err.mean().backward()
    assert prev.grad is not None and prev.grad.abs().sum() > 0, \
        "belief-HPC prediction path must backprop into the input belief"
    assert level.feature_target == "belief_state"


# ── 3. Backward compat / isolation ──────────────────────────────────────────

def test_pixel_target_has_no_hpc_belief():
    """hpc_target='pixel' (D's only value) builds no belief predictor."""
    model = RHANNext(config=_cfg_pixel())
    assert not hasattr(model, "hpc_belief")
    assert "hpc_belief" not in dict(model.state_dict())
    assert hasattr(model, "hpc_level1")
    assert model.hpc_level1.feature_target == "edge_map"


def test_belief_target_has_no_pixel_stack():
    model = RHANNext(config=_cfg_belief())
    assert hasattr(model, "hpc_belief")
    assert model.hpc_stack is None
    assert "hpc_level1" not in dict(model.state_dict())


def test_belief_predictor_in_hpc_optimizer_group():
    """default_group_spec claims hpc_belief.* under the 'hpc' group (the
    pre-flight |dW| target group)."""
    model = RHANNext(config=_cfg_belief())
    reg = default_group_spec(model)
    names = [n for n, p in model.named_parameters() if "hpc_belief" in n]
    assert names, "no hpc_belief params"
    group_params = {id(p) for p in reg.group("hpc")["params"]}
    for n in names:
        p = dict(model.named_parameters())[n]
        assert id(p) in group_params, \
            f"hpc_belief param {n} not claimed by the 'hpc' optimizer group"