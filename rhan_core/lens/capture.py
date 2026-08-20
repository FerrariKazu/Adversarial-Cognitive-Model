"""
capture.py — per-forward-pass state snapshot.
=============================================

Consolidates the internal state the model already computes during a single
forward pass into one object per recurrent step (StepCapture) plus one
terminal summary (ForwardResult). Everything comes from either:

  * the model's own trajectory dict (the diagnostics the trainer already
    prints — actions, Π_D precisions, errors, gate alphas, recon errors,
    uncertainties, AIS continuations, HPC errors), or
  * the HookRegistry buffers (the loop's foveal crop + predicted crop from
    the image_precision call, and the FULL per-step HPC error map — which
    the trajectory only summarises as min/max/std).

No model code is read or modified; nothing here is a training path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

#: v12/v11 trajectory overlay mapping — pixel = (action + 1) * 48 on a 96x96
#: input (the exact mapping eval_rhan_v11's foraging-trajectory plot uses).
_GAZE_PX = 48


def _scalar(x: Any) -> Optional[float]:
    """Detach a scalar tensor (or pass a float/None through)."""
    if x is None:
        return None
    t = torch.as_tensor(x)
    if t.numel() == 0:
        return None
    return float(t.detach().cpu().reshape(-1)[0])


def _at(lst: Optional[Sequence], i: int) -> Optional[Any]:
    """trajectory-list access that never raises (missing pillar -> None)."""
    if not lst or i >= len(lst):
        return None
    return lst[i]


@dataclass
class StepCapture:
    """One snapshot of the model's internal state at recurrent step `step`.

    Pillar fields are None when the loaded checkpoint does not have that
    mechanism (e.g. hpc_error_map is None for AIS-v1 and the TRADES
    baseline; continuation is None for the static baseline). Panels in the
    dashboard are simply omitted for None fields.
    """

    step: int
    # Gaze position (model action space [-1, 1] and the 96x96 pixel overlay).
    gaze_x_norm: Optional[float] = None
    gaze_y_norm: Optional[float] = None
    gaze_x_px: Optional[float] = None
    gaze_y_px: Optional[float] = None
    # Foveal crop actually sampled at this fixation, and the generative
    # prior's reconstruction of it (recon comparison panel).
    foveal_crop: Optional[torch.Tensor] = None        # (3, 48, 48)
    predicted_crop: Optional[torch.Tensor] = None     # (3, 48, 48)
    # Sensory precision Π_D and the error magnitude at this step.
    pi_d: Optional[float] = None
    error_mag: Optional[float] = None
    # Dynamic robustness weight β_dynamic = beta_base * (0.5 + Π_D)
    # (same formula as train_rhan_v10/v11/v12 diagnostics).
    beta_dynamic: Optional[float] = None
    # Foveal/parafoveal fusion gate weight at this step.
    gate_alpha: Optional[float] = None
    # Uncertainty proxy u = 1 - Π_D.
    uncertainty: Optional[float] = None
    # AIS soft continuation weight (halt gauge): ~1 while still gathering
    # evidence, ~0 once the gate wants to stop. None when halting is off.
    continuation: Optional[float] = None
    halted: Optional[bool] = None                     # continuation < 0.5
    # Per-step reconstruction MSE (from the trajectory).
    recon_error: Optional[float] = None
    # HPC (Pillar 1): per-step prediction error, full spatial error map and
    # predicted edge map. None when hpc_num_levels == 0.
    hpc_error: Optional[float] = None
    hpc_error_map: Optional[torch.Tensor] = None      # (1, 48, 48)
    hpc_prediction: Optional[torch.Tensor] = None     # (1, 48, 48)
    # Per-step belief state (512-dim hidden vector). None for static
    # feed-forward checkpoints (TRADES baseline).
    step_belief: Optional[torch.Tensor] = None        # (512,)

    @property
    def has_pillars(self) -> bool:
        """True when this step carries foraging-loop state (vs a static
        feed-forward checkpoint, which yields a single empty capture)."""
        return self.pi_d is not None


@dataclass
class ForwardResult:
    """Terminal summary of one LensSession.run() over a single image."""

    label: str
    arch: str
    device: str
    ais_active: bool
    hpc_active: bool
    config_summary: Optional[str]
    steps_total: int
    # Effective evidence steps = Σ continuation (AIS); == steps_total when
    # halting is off (v12/baseline semantics).
    steps_effective: float
    frac_halting: Optional[float]                      # fraction of steps halted
    logits: torch.Tensor                               # (num_classes,)
    class_probs: torch.Tensor                          # (num_classes,)
    top_class: int
    top_confidence: float
    ground_truth: Optional[int] = None
    correct: Optional[bool] = None
    captures: List[StepCapture] = field(default_factory=list)

    @property
    def class_names(self) -> List[str]:
        from dataset_stl10 import STL10_CLASSES
        return STL10_CLASSES


def build_forward_result(
    model: torch.nn.Module,
    out: Tuple[torch.Tensor, Optional[dict]],
    session: Any,
    ground_truth: Optional[int] = None,
) -> ForwardResult:
    """Assemble a ForwardResult from a model forward + hook buffers.

    Args:
        model:   the loaded (eval-mode) model.
        out:     (logits, trajectory_or_None) as produced by LensSession.
        session: the LensSession — supplies hooks.buffers(), beta_base and
                 the metadata fields (label/arch/device/pillar flags).
        ground_truth: optional STL-10 label for the input image.
    """
    logits, traj = out
    logits = logits.detach()
    if logits.ndim == 2:
        logits = logits[0]                              # B=1: take the image
    probs = F.softmax(logits.float(), dim=-1)
    top_class = int(probs.argmax(dim=-1))
    top_conf = float(probs[top_class])

    steps_total = int(traj.get("steps", 1)) if traj else 1
    buffers = session.hooks.buffers()
    beta_base = session.beta_base

    # AIS continuations (None for v12 / static checkpoints).
    conts = traj.get("continuations") if traj else None
    if conts:
        steps_effective = float(sum(_scalar(c) or 1.0 for c in conts))
        frac_halting = float(sum(1.0 for c in conts if (_scalar(c) or 1.0) < 0.5)
                             ) / len(conts)
    else:
        steps_effective = float(steps_total)
        frac_halting = None

    captures: List[StepCapture] = []
    for i in range(steps_total):
        captures.append(_step_capture(
            i, traj, buffers, beta_base, len(probs)))

    return ForwardResult(
        label=session.label,
        arch=session.arch,
        device=str(session.device),
        ais_active=session.ais_active,
        hpc_active=session.hpc_active,
        config_summary=session.config_summary,
        steps_total=steps_total,
        steps_effective=steps_effective,
        frac_halting=frac_halting,
        logits=logits.cpu(),
        class_probs=probs.cpu(),
        top_class=top_class,
        top_confidence=top_conf,
        ground_truth=ground_truth,
        correct=(top_class == ground_truth) if ground_truth is not None else None,
        captures=captures,
    )


def _step_capture(
    i: int,
    traj: Optional[dict],
    buffers: Dict[str, list],
    beta_base: float,
    n_classes: int,
) -> StepCapture:
    """Build the capture for recurrent step i (index-aligned with the
    trajectory lists and the hook buffers)."""
    if traj is None:
        # Static feed-forward checkpoint (TRADES baseline): one empty capture.
        return StepCapture(step=i)

    action = _at(traj.get("actions"), i)                 # (B, 2)
    prec_pair = _buf_at(buffers, "image_precision", i)   # (crop, pred) or None
    hpc = _buf_at(buffers, "hpc_level1", i)              # (pred, err, err_map)

    pi_d = _scalar(_at(traj.get("precisions"), i))
    cont = _scalar(_at(traj.get("continuations"), i))

    gx = gy = gx_px = gy_px = None
    if action is not None:
        a = action.detach().cpu().reshape(-1)
        gx, gy = float(a[0]), float(a[1])
        gx_px, gy_px = (gx + 1.0) * _GAZE_PX, (gy + 1.0) * _GAZE_PX

    return StepCapture(
        step=i,
        gaze_x_norm=gx,
        gaze_y_norm=gy,
        gaze_x_px=gx_px,
        gaze_y_px=gy_px,
        foveal_crop=(_first(prec_pair[0]) if prec_pair is not None else None),
        predicted_crop=(_first(prec_pair[1]) if prec_pair is not None
                        else None),
        pi_d=pi_d,
        error_mag=_scalar(_at(traj.get("errors"), i)),
        beta_dynamic=(beta_base * (0.5 + pi_d)) if pi_d is not None else None,
        gate_alpha=_scalar(_at(traj.get("gate_alphas"), i)),
        uncertainty=_scalar(_at(traj.get("uncertainties"), i)),
        continuation=cont,
        halted=(cont is not None and cont < 0.5),
        recon_error=_scalar(_at(traj.get("recon_errors"), i)),
        hpc_error=(_scalar(hpc[1]) if hpc is not None else None),
        hpc_error_map=(_first(hpc[2]) if hpc is not None else None),
        hpc_prediction=(_first(hpc[0]) if hpc is not None else None),
        step_belief=_first(_at(traj.get("step_beliefs"), i)),
    )

def _buf_at(buffers: Dict[str, list], name: str, i: int) -> Optional[Any]:
    lst = buffers.get(name) or []
    return lst[i] if i < len(lst) else None


def _first(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Batch-index 0 (B=1) -> (C, H, W), detached, on CPU."""
    if t is None:
        return None
    return t.detach().cpu()[0]


# ── Belief drift computation ───────────────────────────────────────────────

def compute_belief_drift(
    clean_caps: List[StepCapture],
    adv_caps: List[StepCapture],
    metric: str = "cosine",
) -> List[Dict[str, Any]]:
    """Compute belief drift between clean and adversarial forward passes.

    For each recurrent step, computes the distance between the clean and
    adversarial belief states.  Two metrics are always computed:

      * L2 (Euclidean): ||b_clean - b_adv||_2
      * Cosine distance: 1 - cos(b_clean, b_adv)

    The `metric` argument selects which to use as the *primary* value
    returned in the ``'drift'`` key of each row.

    Args:
        clean_caps: StepCapture list from a clean-image .run().
        adv_caps:   StepCapture list from a PGD-perturbed-image .run().
        metric:     ``'cosine'`` (default) or ``'l2'``.

    Returns:
        List of dicts, one per step, with keys:
            step, drift (primary), drift_l2, drift_cosine, pi_d_clean,
            pi_d_adv, has_belief.
        ``has_belief`` is False for checkpoints without per-step belief
        state (e.g. the TRADES baseline — single-pass, no recurrence).
    """
    n_steps = max(len(clean_caps), len(adv_caps))
    rows: List[Dict[str, Any]] = []
    for t in range(n_steps):
        c = clean_caps[t] if t < len(clean_caps) else None
        a = adv_caps[t] if t < len(adv_caps) else None
        bc = c.step_belief if c is not None else None
        ba = a.step_belief if a is not None else None

        has_belief = bc is not None and ba is not None
        drift_l2: Optional[float] = None
        drift_cos: Optional[float] = None

        if has_belief:
            diff = bc.float() - ba.float()
            drift_l2 = float(diff.norm())
            cos_sim = float(F.cosine_similarity(
                bc.float().unsqueeze(0), ba.float().unsqueeze(0)))
            drift_cos = 1.0 - cos_sim

        pi_c = c.pi_d if c is not None else None
        pi_a = a.pi_d if a is not None else None

        primary = drift_cos if metric == "cosine" else drift_l2
        rows.append({
            "step": t,
            "drift": primary,
            "drift_l2": drift_l2,
            "drift_cosine": drift_cos,
            "pi_d_clean": pi_c,
            "pi_d_adv": pi_a,
            "has_belief": has_belief,
        })
    return rows


def belief_drift_summary(
    clean_caps: List[StepCapture],
    adv_caps: List[StepCapture],
) -> Dict[str, Any]:
    """Compact summary of belief drift for a single (clean, adv) pair.

    Returns:
        Dict with keys: mean_drift_cosine, max_drift_cosine,
        mean_drift_l2, max_drift_l2, steps, has_belief.
    """
    rows = compute_belief_drift(clean_caps, adv_caps)
    cos_vals = [r["drift_cosine"] for r in rows if r["drift_cosine"] is not None]
    l2_vals = [r["drift_l2"] for r in rows if r["drift_l2"] is not None]
    return {
        "mean_drift_cosine": float(np.mean(cos_vals)) if cos_vals else None,
        "max_drift_cosine": float(np.max(cos_vals)) if cos_vals else None,
        "mean_drift_l2": float(np.mean(l2_vals)) if l2_vals else None,
        "max_drift_l2": float(np.max(l2_vals)) if l2_vals else None,
        "steps": len(rows),
        "has_belief": any(r["has_belief"] for r in rows),
    }


# ── Gaze–perturbation correlation ──────────────────────────────────────────

def gaze_perturbation_correlation(
    clean_caps: List[StepCapture],
    adv_caps: List[StepCapture],
    clean_image: torch.Tensor,
    adv_image: torch.Tensor,
    patch_radius: int = 3,
) -> List[Dict[str, Any]]:
    """Correlate gaze displacement with local perturbation magnitude.

    For each recurrent step *t*, computes:

      * gaze displacement: ||g_adv(t) − g_clean(t)||₂ in pixel space
      * local perturbation: mean |δ| over a ``patch_radius``-pixel patch
        centred on the *clean* gaze position (the position the model would
        have chosen on the unperturbed image)
      * global perturbation: mean |δ| over the entire image (baseline)

    The key diagnostic is whether steps where the clean-image gaze lands in
    a high-|δ| region show *larger* displacement under attack (positive
correlation → the perturbation is drawing the gaze), or whether
displacement is uniform regardless of where the perturbation sits
(zero correlation → the gaze is mechanically iterating, not seeking).

    Args:
        clean_caps: StepCaptures from ``LensSession.run(clean_image)``.
        adv_caps:   StepCaptures from ``LensSession.run(adv_image)``.
        clean_image: (3, 96, 96) normalised clean tensor.
        adv_image:   (3, 96, 96) normalised adversarial tensor.
        patch_radius: half-size of the local patch (default 3 → 7×7 patch).

    Returns:
        List of dicts, one per step, with keys:
            step, gaze_disp_px, local_delta_mean, local_delta_max,
            global_delta_mean, displacement_excess (local − global).
    """
    H, W = clean_image.shape[-2], clean_image.shape[-1]
    delta = (adv_image.detach().cpu() - clean_image.detach().cpu())  # (3, H, W)
    delta_mag = delta.abs().mean(dim=0)                             # (H, W)
    global_mean = float(delta_mag.mean())

    n_steps = max(len(clean_caps), len(adv_caps))
    rows: List[Dict[str, Any]] = []
    for t in range(n_steps):
        c = clean_caps[t] if t < len(clean_caps) else None
        a = adv_caps[t] if t < len(adv_caps) else None
        if c is None or a is None or c.gaze_x_px is None or a.gaze_x_px is None:
            rows.append({
                "step": t, "gaze_disp_px": None, "local_delta_mean": None,
                "local_delta_max": None, "global_delta_mean": global_mean,
                "displacement_excess": None,
            })
            continue

        # Gaze displacement in pixel space.
        dx = a.gaze_x_px - c.gaze_x_px
        dy = a.gaze_y_px - c.gaze_y_px
        disp = float(np.sqrt(dx * dx + dy * dy))

        # Local perturbation patch centred on the clean gaze position.
        cx = int(np.clip(round(c.gaze_x_px), patch_radius, W - patch_radius - 1))
        cy = int(np.clip(round(c.gaze_y_px), patch_radius, H - patch_radius - 1))
        patch = delta_mag[
            cy - patch_radius : cy + patch_radius + 1,
            cx - patch_radius : cx + patch_radius + 1,
        ]
        local_mean = float(patch.mean())
        local_max = float(patch.max())

        rows.append({
            "step": t,
            "gaze_disp_px": disp,
            "local_delta_mean": local_mean,
            "local_delta_max": local_max,
            "global_delta_mean": global_mean,
            "displacement_excess": local_mean - global_mean,
        })
    return rows


def aggregate_gaze_correlation(
    all_rows: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Aggregate per-image gaze–perturbation correlation rows.

    Args:
        all_rows: list of per-image row-lists (from gaze_perturbation_correlation).

    Returns:
        Dict with:
            pearson_r: Pearson r between gaze_disp_px and local_delta_mean
                        (across all steps × images)
            spearman_rho: Spearman rank correlation (same)
            mean_displacement: mean ||gaze_disp|| across all steps/images
            mean_local_delta: mean local |δ| at clean gaze positions
            mean_global_delta: mean |δ| over entire images
            n_samples: number of (step, image) pairs
            per_step: list of dicts with per-step means across images
    """
    from scipy import stats as sp_stats

    # Flatten across images.
    disp_all: List[float] = []
    local_all: List[float] = []
    step_disp: Dict[int, List[float]] = {}
    step_local: Dict[int, List[float]] = {}

    for rows in all_rows:
        for r in rows:
            if r["gaze_disp_px"] is None or r["local_delta_mean"] is None:
                continue
            disp_all.append(r["gaze_disp_px"])
            local_all.append(r["local_delta_mean"])
            step_disp.setdefault(r["step"], []).append(r["gaze_disp_px"])
            step_local.setdefault(r["step"], []).append(r["local_delta_mean"])

    if len(disp_all) < 3:
        return {
            "pearson_r": None, "spearman_rho": None,
            "mean_displacement": None, "mean_local_delta": None,
            "mean_global_delta": None, "n_samples": len(disp_all),
            "per_step": [],
        }

    pearson_r, pearson_p = sp_stats.pearsonr(disp_all, local_all)
    spearman_rho, spearman_p = sp_stats.spearmanr(disp_all, local_all)

    # Per-step aggregation.
    per_step = []
    for t in sorted(step_disp.keys()):
        d = step_disp[t]
        l = step_local.get(t, [])
        per_step.append({
            "step": t,
            "mean_disp_px": float(np.mean(d)) if d else None,
            "std_disp_px": float(np.std(d, ddof=1)) if len(d) > 1 else None,
            "mean_local_delta": float(np.mean(l)) if l else None,
            "std_local_delta": float(np.std(l, ddof=1)) if len(l) > 1 else None,
            "n": len(d),
        })

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "mean_displacement": float(np.mean(disp_all)),
        "mean_local_delta": float(np.mean(local_all)),
        "n_samples": len(disp_all),
        "per_step": per_step,
    }
