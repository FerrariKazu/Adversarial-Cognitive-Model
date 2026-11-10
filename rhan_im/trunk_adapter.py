"""Adapter between a ViT trunk (e.g. gen2_foundation.backbone.CompactViTGen2) and the Bayes state.

native_crop    integer-aligned crop of a canvas with NO resampling (zero padding outside), unlike
               noesis_vision.models.foveation.foveal_sample, which at window == frame is only a shifted,
               border-replicated copy of the frame.
WindowSensor   look(canvas, k) -> (z (B, V, C), d (B, V)): runs the trunk on the native window (or on the
               resampled whole canvas for the gist look), reads class scores from every token, standardises
               them with the measured calibration and scatters them onto the evidence units of a UnitLayout.
Any callable trunk(x) -> (cls (B, D), tokens (B, T, D)) works; tests use a transparent stand-in.
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhan_im.layout import UnitLayout
from rhan_im.window_head import EvidenceCalibration, EvidenceHead


def native_crop(canvas: torch.Tensor, topleft_px: torch.Tensor, window: int) -> torch.Tensor:
    """(B, 3, H, W) canvas, (B, 2) top-left (row, col) in pixels -> (B, 3, window, window); out-of-canvas = 0."""
    B = canvas.shape[0]
    x = F.pad(canvas, (window, window, window, window))
    a = torch.arange(window, device=canvas.device)
    rows = topleft_px[:, 0, None] + window + a[None, :]
    cols = topleft_px[:, 1, None] + window + a[None, :]
    out = x[torch.arange(B, device=canvas.device)[:, None, None], :, rows[:, :, None], cols[:, None, :]]
    return out.permute(0, 3, 1, 2).contiguous()


def ff_token_fn(backbone) -> Callable:
    """trunk(x) -> (cls, patch tokens) along EXACTLY the path the frozen ff_112 checkpoint was trained on:
    prep -> 12 pre-LN blocks -> final norm. The tied refinement block (never run by the ff path, never
    trained there) is not applied. For CompactViTGen2 at 112 px: x (B, 3, 112, 112) -> (B, 384), (B, 64, 384)."""
    def fn(x: torch.Tensor):
        h = backbone._trunk_forward(backbone._prep(x))
        return h[:, 0], h[:, backbone.num_prefix_tokens:]
    return fn


class WindowSensor(nn.Module):
    def __init__(self, trunk: Callable, head: EvidenceHead, layout: UnitLayout, calib: EvidenceCalibration, topleft: torch.Tensor) -> None:
        super().__init__()
        self.trunk, self.head, self.layout, self.calib = trunk, head, layout, calib
        self.register_buffer("topleft", topleft)                              # (K, 2) fine units
        self.K = topleft.shape[0]                                             # candidate K is the gist look

    @property
    def n_candidates(self) -> int:
        return self.K + 1

    def vis(self) -> torch.Tensor:
        return self.layout.vis_matrix(self.calib.d_tok_eff, float(self.calib.d_gist_eff), self.topleft)

    def look(self, canvas: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lay, B = self.layout, canvas.shape[0]
        C = self.head.proj.out_features
        z = torch.zeros(B, lay.V, C, dtype=canvas.dtype, device=canvas.device)
        is_gist = k == self.K
        win = ~is_gist
        if win.any():
            tl = self.topleft[k[win]]
            crop = native_crop(canvas[win], tl * lay.patch, lay.window)
            _, tokens = self.trunk(crop)                                       # (b, tok*tok, D)
            s = self.calib.standardize(self.head(tokens), 0)                   # (b, T, C)
            idx, valid = lay.window_units(tl)
            s = s * valid[..., None].to(s.dtype)
            zw = torch.zeros(int(win.sum()), lay.V, C, dtype=s.dtype, device=s.device)
            zw.scatter_add_(1, idx[..., None].expand(-1, -1, C), s)
            z[win] = zw
        if is_gist.any():
            small = F.interpolate(canvas[is_gist], size=(lay.window, lay.window), mode="area")
            _, tokens = self.trunk(small)
            z[is_gist, lay.n_fine:] = self.calib.standardize(self.head(tokens), 1)
        return z, self.vis()[k]
