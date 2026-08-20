"""
hooks.py — forward-hook registry for the Lens introspection layer.
=================================================================

The ONLY place lens/ touches a live model. Registers torch forward hooks
(nn.Module.register_forward_hook / register_forward_pre_hook) on modules the
model ALREADY executes inside its foraging loop — no model code is edited,
no training or evaluation path is affected.

WHY THESE MODULES (loop-only call sites):

    image_precision    input[0], input[1] -> the foveal crop at the current
                       gaze position AND the generative prior's predicted
                       crop. Called exactly once per recurrent step in the
                       foraging loop: directly in the v12 path
                       (image_precision(x_foveal, predicted_crop, s)) and —
                       under AIS — through the GlobalPrecisionModulator's
                       precision_from_crops, which forwards to the SAME
                       module. Neither the info-gain gaze policy nor any
                       other consumer touches it, so its buffer aligns
                       index-for-index with the trajectory lists.

    hpc_level1         output -> (prediction, error, error_map). Only runs
                       when the checkpoint was trained with enable_hpc and
                       only inside the loop, so it never fires for AIS-v1 or
                       the TRADES baseline (panel simply omitted).

    (The AIS info-gain policy internally calls generative_prior /
     foveal_stream once per policy step too, so those modules are NOT
     hooked — their buffers would not align with the loop's steps.)

A forward PRE-hook on the model root clears the buffers at the start of
every forward pass, so a fresh run always starts from an empty snapshot and
consecutive .run() calls never bleed into each other.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

#: Submodule names observed per recurrent step (aligned with trajectory lists).
STEP_MODULES = ("image_precision", "hpc_level1")


class HookRegistry:
    """Attach/detach forward hooks and expose the per-pass buffers.

    Thread-safety: intended for single-threaded interactive use (Streamlit
    session, notebook). Attach once per loaded model and keep attached for
    the lifetime of the LensSession.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: List[Any] = []
        self._buffers: Dict[str, list] = {}
        self._attached = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    def attach(self) -> "HookRegistry":
        """Register the hooks (idempotent — re-attaching detaches first)."""
        if self._attached:
            return self
        self._buffers = {name: [] for name in STEP_MODULES}
        # Clear the buffers before every forward pass on the model root.
        self._handles.append(
            self.model.register_forward_pre_hook(self._reset_buffers))
        for name in STEP_MODULES:
            mod = getattr(self.model, name, None)
            if mod is None:
                continue  # checkpoint without this pillar: panel simply absent
            if name == "image_precision":
                # (actual crop, predicted crop) — the loop's exact tensors.
                self._handles.append(mod.register_forward_hook(
                    lambda m, a, o, _n=name: self._buffers[_n].append(
                        (a[0].detach(), a[1].detach()))))
            elif name == "hpc_level1":
                self._handles.append(mod.register_forward_hook(
                    lambda m, a, o, _n=name: self._buffers[_n].append(
                        tuple(t.detach() for t in o))))
        self._attached = True
        return self

    def detach(self) -> "HookRegistry":
        """Remove every registered hook and drop the buffers."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._buffers = {}
        self._attached = False
        return self

    # ── state ────────────────────────────────────────────────────────────────
    def buffers(self) -> Dict[str, list]:
        """Per-pass buffers, keyed by STEP_MODULES name.

        Each list holds one entry per recurrent step, in step order:
            image_precision -> (foveal_crop (B,3,48,48),
                                predicted_crop (B,3,48,48))
            hpc_level1      -> (prediction, error, error_map) tuple
        """
        return self._buffers

    @property
    def attached(self) -> bool:
        return self._attached

    def _reset_buffers(self, module: nn.Module, args) -> None:
        for name in STEP_MODULES:
            self._buffers[name] = []
        return None  # do not modify the forward pass in any way
