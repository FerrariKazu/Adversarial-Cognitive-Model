"""
session.py — LensSession: load a checkpoint, wrap it with hooks, run it.
========================================================================

Loads ANY NOESIS / RHAN-Next checkpoint (A/B/C/D-matrix or otherwise) as a
read-only inference session:

  * checkpoint config auto-detection + model construction are DELEGATED to
    phase2_attacks/eval_rhan.py::_load_model — the exact logic the eval
    protocol uses (RHANNextConfig.from_dict from the checkpoint's 'config'
    key with a v12-equivalent fallback; arch registry handles 'large' via
    the frozen sweep loader). lens/ does not reimplement checkpoint parsing.
  * missing-local checkpoints are fetched via the notebooks' HuggingFace
    cache logic (phase1_training/eval_rhan_v11.py::download_checkpoint_from_hf).
  * PGD generation is DELEGATED to the canonical protocol attack
    (phase2_attacks/eval_full_epsilon_sweep.py::run_pgd_batched) in the
    Finding-17 norm-space convention.

LensSession is inference-only: it never writes a checkpoint, never touches
the resume-gate machinery, and never mutates RHANNextConfig. `.run(image)`
is a generator yielding one StepCapture per recurrent step, then the
terminal ForwardResult — the dashboard renders incrementally from it.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch

# Repo conventions: the frozen chains insert the dirs on sys.path themselves;
# we do the same so this package works regardless of the caller's cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_ROOT,
           os.path.join(_ROOT, "phase1_training"),
           os.path.join(_ROOT, "phase2_attacks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse (NOT reimplement):
#   * config-from-checkpoint + model loading   -> eval_rhan.py::_load_model
#   * canonical norm-space PGD + STL-10 bounds -> eval_full_epsilon_sweep.py
#   * HuggingFace download/cache               -> eval_rhan_v11.py
from eval_rhan import _load_model as _eval_load_model       # noqa: E402
import eval_full_epsilon_sweep as _sweep                     # noqa: E402
from eval_rhan_v11 import download_checkpoint_from_hf        # noqa: E402

from rhan_core.lens.hooks import HookRegistry                 # noqa: E402
from rhan_core.lens.capture import (                         # noqa: E402
    build_forward_result,
    ForwardResult,
    StepCapture,
)


def _matrix_arch(path: str) -> Optional[str]:
    """Arch for A/B/C/D-matrix checkpoints (registry lookup by filename).

    Reuses rhan_core/ablation/matrix.py — the same registry the dashboard's
    side-by-side selector and the eval protocol use.
    """
    try:
        from rhan_core.ablation.matrix import ABLATION_MATRIX
    except Exception:
        return None
    base = os.path.basename(path)
    for entry in ABLATION_MATRIX.values():
        ckpt = entry.get("checkpoint")
        if ckpt and os.path.basename(ckpt) == base:
            return entry.get("arch")
    return None


def _infer_arch(path: str) -> str:
    """Resolve the eval arch for a checkpoint file.

    Order: matrix registry (exact reuse) -> light state-dict key sniff for
    non-matrix files -> 'next' (RHANNext's default config is v12-identical,
    so a plain v12 checkpoint also loads 1:1 through the 'next' loader).
    Only the split-transformer TRADES baseline needs its own dispatch key
    ('ventral.'/'dorsal.' prefixes are unique to RHANLargeSTL10).
    """
    arch = _matrix_arch(path)
    if arch:
        return arch
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for k in ("model", "model_state_dict", "state_dict"):
                if k in state:
                    state = state[k]
                    break
        if isinstance(state, dict):
            keys = list(state.keys())
            if any(k.startswith("ventral.") or k.startswith("dorsal.")
                   for k in keys):
                return "large"
    except Exception:
        pass
    return "next"


class LensSession:
    """A loaded checkpoint wrapped with observation hooks.

    Args:
        checkpoint_path: path to a .pth checkpoint (downloaded from Hugging
            Face automatically when missing locally).
        arch: optional eval arch ('next' / 'large' / 'v12' / 'v11' / 'v10').
            Auto-detected from the ABLATION_MATRIX registry or state-dict
            keys when omitted.
        device: optional torch device; defaults to cuda if available.
        beta_base: β base used for the β_dynamic = beta_base * (0.5 + Π_D)
            readout (2.0 matches the Stage-2 ε=0.031 phase; 2.5 the ε=0.094
            phase — the trainer's curriculum convention).
        label: optional display name (defaults to the checkpoint filename).
    """

    def __init__(
        self,
        checkpoint_path: str,
        arch: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        beta_base: float = 2.0,
        label: Optional[str] = None,
    ):
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.beta_base = float(beta_base)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.label = label or os.path.basename(self.checkpoint_path)

        if not os.path.exists(self.checkpoint_path):
            ok = download_checkpoint_from_hf(self.checkpoint_path)
            if not ok or not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found locally and HuggingFace download "
                    f"failed: {self.checkpoint_path}")

        self.arch = arch or _infer_arch(self.checkpoint_path)
        # Reuses eval_rhan.py's exact loader: config-from-checkpoint for
        # arch 'next', frozen sweep loader otherwise. No resume-gate, no
        # checkpoint writes — inference-only.
        self.model = _eval_load_model(self.arch, self.checkpoint_path,
                                      self.device)
        self.model.eval()

        self.config = getattr(self.model, "config", None)
        self.ais_active = hasattr(self.model, "halt_policy")
        self.hpc_active = hasattr(self.model, "hpc_level1")
        self.config_summary = self._config_summary()

        self.hooks = HookRegistry(self.model).attach()

    # ── metadata ─────────────────────────────────────────────────────────────
    def _config_summary(self) -> Optional[str]:
        if self.config is None:
            return None
        try:
            cfg = self.config
            toggles = []
            if getattr(cfg, "enable_ais", False):
                toggles.append("AIS")
                toggles.append(f"halt={'on' if cfg.ais_halt_enabled else 'off'}")
                toggles.append(
                    f"recon={'on' if cfg.ais_precision_recon_enabled else 'off'}")
            if getattr(cfg, "enable_hpc", False):
                toggles.append(f"HPC(L={cfg.hpc_num_levels})")
            return ", ".join(toggles) or "v12-equivalent (all pillars off)"
        except Exception:
            return None

    def describe(self) -> Dict[str, Any]:
        """Small dict of session facts for the dashboard footer/diff view."""
        return {
            "label": self.label,
            "checkpoint": self.checkpoint_path,
            "arch": self.arch,
            "device": str(self.device),
            "config": self.config_summary,
            "ais_active": self.ais_active,
            "hpc_active": self.hpc_active,
            "beta_base": self.beta_base,
        }

    # ── PGD (delegated to the canonical protocol attack) ─────────────────────
    def pgd(self, image_tensor: torch.Tensor, eps: float, steps: int = 20,
            norm_space: bool = True) -> torch.Tensor:
        """PGD-perturb `image_tensor` with THIS session's model.

        Delegates to eval_full_epsilon_sweep.run_pgd_batched (the exact
        attack the eval protocol uses) in the Finding-17 norm-space
        convention (eps applied directly to normalized inputs). Hooks are
        detached for the attack's internal forward passes and re-attached
        afterwards, so the capture buffers are never polluted.

        Returns the adversarial image on CPU, same shape as the input.
        """
        x = image_tensor.detach().cpu()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        y = torch.zeros(x.shape[0], dtype=torch.long)
        self.hooks.detach()
        try:
            adv_cpu, _ = _sweep.run_pgd_batched(
                self.model, x, y, eps, steps, self.device,
                batch_size=x.shape[0], norm_space=norm_space)
            return adv_cpu
        finally:
            self.hooks.attach()

    # ── run: generator of one capture per recurrent step ────────────────────
    def run(
        self,
        image_tensor: torch.Tensor,
        step_by_step: bool = True,
        ground_truth: Optional[int] = None,
    ) -> Iterator[Union[StepCapture, ForwardResult]]:
        """Run one image through the model, yielding per-step state.

        Args:
            image_tensor: (3, 96, 96) or (1, 3, 96, 96) normalized STL-10
                tensor (same normalization the training pipeline uses).
            step_by_step: when True (default), yields one StepCapture per
                recurrent step, then the terminal ForwardResult. The forward
                pass itself runs once (the loop lives inside the model's
                forward, which lens/ must not edit); the generator then
                releases each step's snapshot so the dashboard can render
                incrementally instead of waiting for a full report.
            ground_truth: optional STL-10 class index for the final panel.

        Yields:
            StepCapture for each of the model's recurrent steps, then the
            ForwardResult. For a static checkpoint (no foraging loop) a
            single empty StepCapture is yielded, then the ForwardResult.
        """
        x = image_tensor.detach()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self._forward(x)

        result = build_forward_result(self.model, out, self,
                                      ground_truth=ground_truth)
        if step_by_step:
            for cap in result.captures:
                yield cap
        yield result

    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[dict]]:
        """model(x, return_trajectory=True) when supported, else model(x)."""
        import inspect
        try:
            params = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            params = {}
        if "return_trajectory" in params:
            out = self.model(x, return_trajectory=True)
            if isinstance(out, tuple) and len(out) == 2 and \
                    isinstance(out[1], dict):
                return out
            return out, None
        return self.model(x), None


# ── Batch belief-drift analysis ────────────────────────────────────────────

def run_captures(
    session: LensSession,
    image_tensor: torch.Tensor,
    ground_truth: Optional[int] = None,
) -> Tuple[ForwardResult, List["StepCapture"]]:
    """Run one image and return (ForwardResult, [StepCapture, ...])."""
    from rhan_core.lens.capture import ForwardResult as FR, StepCapture as SC
    caps: List[SC] = []
    result: Optional[FR] = None
    for item in session.run(image_tensor, step_by_step=True,
                            ground_truth=ground_truth):
        if isinstance(item, SC):
            caps.append(item)
        else:
            result = item
    assert result is not None
    return result, caps


def batch_belief_drift(
    sessions: List[LensSession],
    images: List[torch.Tensor],
    eps: float = 0.094,
    pgd_steps: int = 50,
) -> Dict[str, Any]:
    """Run clean-vs-PGD belief drift across multiple checkpoints and images.

    For each checkpoint and each image, runs the model once on the clean
    image and once on its PGD-ε-perturbed version, then computes per-step
    belief drift (cosine distance and L2) between the two.

    Args:
        sessions: list of LensSession (one per checkpoint: baseline, AIS-v1,
            HPC-only).
        images: list of (3, 96, 96) normalized STL-10 test tensors.
        eps: PGD epsilon in norm space (Finding-17 convention).
        pgd_steps: number of PGD steps for the attack.

    Returns:
        Dict with keys:
            "per_checkpoint": {label: {"rows": [...], "summary": {...}}}
            "step_labels": ["T=0", "T=1", ...] (for the table header)
            "eps": the epsilon used
            "n_images": how many images were evaluated
    """
    import numpy as np
    from rhan_core.lens.capture import compute_belief_drift, belief_drift_summary

    result: Dict[str, Any] = {
        "per_checkpoint": {},
        "eps": eps,
        "n_images": len(images),
    }

    for sess in sessions:
        all_rows: List[List[Dict]] = []  # rows per image
        for img in images:
            # Clean run
            _, clean_caps = run_captures(sess, img)
            # PGD run
            adv_img = sess.pgd(img, eps=eps, steps=pgd_steps)
            _, adv_caps = run_captures(sess, adv_img[0])
            drift_rows = compute_belief_drift(clean_caps, adv_caps)
            all_rows.append(drift_rows)

        # Average drift per step across images
        n_steps = max(len(r) for r in all_rows) if all_rows else 0
        mean_rows: List[Dict[str, Any]] = []
        for t in range(n_steps):
            cos_vals = [r[t]["drift_cosine"] for r in all_rows
                       if t < len(r) and r[t]["drift_cosine"] is not None]
            l2_vals = [r[t]["drift_l2"] for r in all_rows
                      if t < len(r) and r[t]["drift_l2"] is not None]
            pi_c = [r[t]["pi_d_clean"] for r in all_rows
                    if t < len(r) and r[t]["pi_d_clean"] is not None]
            pi_a = [r[t]["pi_d_adv"] for r in all_rows
                    if t < len(r) and r[t]["pi_d_adv"] is not None]
            mean_rows.append({
                "step": t,
                "drift_cosine_mean": float(np.mean(cos_vals)) if cos_vals else None,
                "drift_cosine_std": float(np.std(cos_vals, ddof=1)) if len(cos_vals) > 1 else None,
                "drift_l2_mean": float(np.mean(l2_vals)) if l2_vals else None,
                "drift_l2_std": float(np.std(l2_vals, ddof=1)) if len(l2_vals) > 1 else None,
                "pi_d_clean_mean": float(np.mean(pi_c)) if pi_c else None,
                "pi_d_adv_mean": float(np.mean(pi_a)) if pi_a else None,
            })

        # Aggregate summary across all images
        cos_all = [r["drift_cosine"] for rows in all_rows for r in rows
                   if r["drift_cosine"] is not None]
        l2_all = [r["drift_l2"] for rows in all_rows for r in rows
                  if r["drift_l2"] is not None]
        summary = {
            "mean_drift_cosine": float(np.mean(cos_all)) if cos_all else None,
            "max_drift_cosine": float(np.max(cos_all)) if cos_all else None,
            "mean_drift_l2": float(np.mean(l2_all)) if l2_all else None,
            "max_drift_l2": float(np.max(l2_all)) if l2_all else None,
            "has_belief": sess.hpc_active or sess.ais_active,
        }

        result["per_checkpoint"][sess.label] = {
            "rows": mean_rows,
            "summary": summary,
        }

    return result
