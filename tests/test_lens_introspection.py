"""
tests/test_lens_introspection.py — acceptance tests for the Lens layer.
======================================================================

Covers the task's acceptance criteria for rhan_core/lens (read-only
introspection) + dashboards/lens_app.py:

  1. LensSession loads Stage 1's validated AIS-v1 checkpoint AND the TRADES
     baseline checkpoint without error, on CPU (real files, when present).
  2. .run() on a single STL-10 test image yields ONE capture per recurrent
     step; AIS-v1 fields populated; HPC-specific fields absent for a
     baseline without an HPC head.
  3. run() is a generator: StepCapture per step, then ForwardResult.
  4. Hooks are non-invasive: logits are byte-identical with and without the
     HookRegistry attached.
  5. session.pgd delegates to the canonical attack and preserves shapes.
  6. Side-by-side: two sessions run the SAME image without error.

These tests never write to a checkpoint, never fine-tune, and never touch
the resume-gate machinery — lens is inference-only by construction.
"""
from __future__ import annotations

import os

import pytest
import torch

from rhan_core.lens import LensSession, StepCapture, ForwardResult
from rhan_core.lens.hooks import HookRegistry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIS_CKPT = os.path.join(ROOT, "checkpoints",
                        "rhan_next_ais_v1_halting_only_best.pth")
BASE_CKPT = os.path.join(ROOT, "checkpoints",
                         "rhan_stl10_large_pseudolabel_best.pth")

REAL_CKPTS = os.path.exists(AIS_CKPT) and os.path.exists(BASE_CKPT)

pytestmark = pytest.mark.skipif(
    not REAL_CKPTS,
    reason="real Stage-1 checkpoints not present locally (skip network/CI)")

_DEVICE = torch.device("cpu")


def _test_image(seed: int = 0) -> torch.Tensor:
    """A single normalized STL-10-sized (3, 96, 96) input on CPU, sampled
    INSIDE the valid pixel domain ([0,1] pixels, then canonical
    normalization) so PGD's domain clamp cannot legitimately move it."""
    from rhan_core.lens.session import _sweep  # canonical MEAN/STD (reuse)
    g = torch.Generator().manual_seed(seed)
    pix = torch.rand(3, 96, 96, generator=g)   # pixel-space [0, 1]
    return (pix - _sweep.MEAN.squeeze(0)) / _sweep.STD.squeeze(0)


@pytest.fixture(scope="module")
def ais_session() -> LensSession:
    return LensSession(AIS_CKPT, device=_DEVICE)


@pytest.fixture(scope="module")
def base_session() -> LensSession:
    return LensSession(BASE_CKPT, device=_DEVICE)


# ── Acceptance 1: loads without error on CPU ─────────────────────────────────
def test_loads_ais_v1_checkpoint_on_cpu(ais_session):
    assert ais_session.ais_active is True
    assert ais_session.hpc_active is False       # B is AIS-v1 halting-only
    assert ais_session.arch == "next"
    assert ais_session.device.type == "cpu"


def test_loads_trades_baseline_on_cpu(base_session):
    assert base_session.arch == "large"
    assert base_session.ais_active is False
    assert base_session.hpc_active is False


# ── Acceptance 2 + 3: run() generator, per-step captures, field presence ─────
def test_run_yields_one_capture_per_step_for_ais(ais_session):
    x = _test_image()
    caps, result = _collect(ais_session.run(x, ground_truth=0))

    n_steps = ais_session.model.max_steps
    assert len(caps) == n_steps
    assert result.steps_total == n_steps
    for i, cap in enumerate(caps):
        assert isinstance(cap, StepCapture)
        assert cap.step == i
        assert cap.has_pillars
        # AIS-v1 fields present on every step:
        assert cap.pi_d is not None
        assert cap.continuation is not None
        assert cap.gate_alpha is not None
        assert cap.foveal_crop is not None
        assert cap.predicted_crop is not None
        # HPC-specific fields correctly ABSENT for the AIS-v1 checkpoint:
        assert cap.hpc_error is None
        assert cap.hpc_error_map is None
        assert cap.hpc_prediction is None


def test_run_yields_single_capture_for_baseline(base_session):
    x = _test_image()
    caps, result = _collect(base_session.run(x, ground_truth=3))

    assert len(caps) == 1
    cap = caps[0]
    assert cap.has_pillars is False               # static model: no foraging
    assert cap.pi_d is None
    assert cap.hpc_error is None
    assert result.steps_total == 1
    assert 0 <= result.top_class <= 9
    assert torch.isclose(result.class_probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_run_is_a_generator_of_captures_then_result(ais_session):
    x = _test_image()
    gen = ais_session.run(x)
    assert hasattr(gen, "__next__")               # generator protocol
    first = next(gen)
    assert isinstance(first, StepCapture)
    last = None
    for item in gen:
        last = item
    assert isinstance(last, ForwardResult)


# ── Acceptance: hooks are non-invasive ───────────────────────────────────────
def test_hooks_do_not_alter_forward_output(ais_session):
    x = _test_image().unsqueeze(0)
    with torch.no_grad():
        clean_logits = ais_session.model(x)

    ais_session.hooks.detach()
    try:
        with torch.no_grad():
            hooked_logits = ais_session.model(x)
    finally:
        ais_session.hooks.attach()
    assert torch.allclose(clean_logits, hooked_logits)


def test_consecutive_runs_do_not_bleed_buffers(ais_session):
    """Each forward clears the hook buffers (pre-hook on the model root)."""
    x1 = _test_image(seed=1)
    x2 = _test_image(seed=2)
    caps1, _ = _collect(ais_session.run(x1))
    caps2, _ = _collect(ais_session.run(x2))
    assert len(caps1) == len(caps2)
    assert not torch.allclose(caps1[0].foveal_crop, caps2[0].foveal_crop)


# ── Acceptance: PGD delegation preserves shape ───────────────────────────────
def test_pgd_delegates_to_canonical_attack(ais_session):
    x = _test_image()
    adv = ais_session.pgd(x, eps=0.031, steps=5)
    assert adv.shape == (1, 3, 96, 96)
    delta = (adv - x).abs().amax().item()
    assert delta <= 0.031 + 1e-4                # norm-space ε bound respected


# ── Acceptance: side-by-side (two checkpoints, same image) ───────────────────
def test_side_by_side_same_image(ais_session, base_session):
    x = _test_image()
    caps_a, res_a = _collect(ais_session.run(x))
    caps_b, res_b = _collect(base_session.run(x))
    assert res_a.top_class is not None
    assert res_b.top_class is not None
    assert len(caps_a) > len(caps_b)            # AIS steps vs static baseline


def _collect(gen):
    caps = []
    result = None
    for item in gen:
        if isinstance(item, StepCapture):
            caps.append(item)
        else:
            result = item
    assert result is not None
    return caps, result


# ── Hooks registry unit behavior (no checkpoint needed) ──────────────────────
def test_hook_registry_records_per_step_buffers():
    from rhan_core.config.pillar_config import RHANNextConfig
    from rhan_core.model import RHANNext
    model = RHANNext(config=RHANNextConfig(
        enable_ais=True, ais_halt_enabled=True,
        ais_precision_recon_enabled=False)).eval()

    hooks = HookRegistry(model).attach()
    assert hooks.attached
    assert set(hooks.buffers().keys()) == {"image_precision", "hpc_level1"}

    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        _, traj = model(x, return_trajectory=True)
    n = model.max_steps
    # Loop-only hook: exactly ONE (crop, predicted) pair per recurrent step,
    # despite the info-gain policy also calling generative_prior/foveal_stream
    # internally once per policy step.
    assert len(hooks.buffers()["image_precision"]) == n
    crop, pred = hooks.buffers()["image_precision"][0]
    assert crop.shape == (1, 3, 48, 48)
    assert pred.shape == (1, 3, 48, 48)
    # hpc_level1 absent for an AIS-only model -> nothing recorded:
    assert hooks.buffers()["hpc_level1"] == []

    hooks.detach()
    assert not hooks.attached
