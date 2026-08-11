"""Explainability methods — CAM family, gradients, occlusion.

All methods return a normalized (H, W) float32 heatmap in [0, 1].
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from cognitive_vision_lab.backend.attacks import _forward


def _find_conv(model, name_hint: Optional[str] = None) -> torch.nn.Module:
    """Pick a target conv layer: name hint → last conv → first conv."""
    root = model.module if hasattr(model, "module") else model
    if name_hint:
        for n, m in root.named_modules():
            if n == name_hint and isinstance(m, torch.nn.Conv2d):
                return m
    convs = [m for m in root.modules() if isinstance(m, torch.nn.Conv2d)]
    if convs:
        return convs[-1]
    raise ValueError("No Conv2d layer found for CAM")


def _upsample(cam: torch.Tensor, h: int, w: int) -> np.ndarray:
    cam = cam.detach().float()
    if cam.ndim == 2:
        cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cmin, cmax = cam.min(), cam.max()
    if cmax > cmin:
        cam = (cam - cmin) / (cmax - cmin)
    return cam.astype(np.float32)


def _hooks(layer: torch.nn.Module):
    store = {}

    def fwd(m, inp, out):
        store["act"] = out[0] if isinstance(out, (tuple, list)) else out

    def bwd(m, gin, gout):
        g = gout[0] if isinstance(gout, (tuple, list)) else gout
        store["grad"] = g[0] if isinstance(g, (tuple, list)) else g

    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(bwd)
    return store, h1, h2


def gradcam(model, x, class_idx=None, target_layer=None) -> np.ndarray:
    layer = _find_conv(model, target_layer)
    store, h1, h2 = _hooks(layer)
    model.zero_grad()
    out = _forward(model, x)
    c = class_idx if class_idx is not None else out.argmax(1).item()
    out[0, c].backward(retain_graph=True)
    h1.remove(); h2.remove()
    if "act" not in store or "grad" not in store:
        return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)
    act = store["act"].detach()
    grad = store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    return _upsample(cam, x.shape[2], x.shape[3])


def gradcam_pp(model, x, class_idx=None, target_layer=None) -> np.ndarray:
    layer = _find_conv(model, target_layer)
    store, h1, h2 = _hooks(layer)
    model.zero_grad()
    out = _forward(model, x)
    c = class_idx if class_idx is not None else out.argmax(1).item()
    out[0, c].backward(retain_graph=True)
    h1.remove(); h2.remove()
    act = store["act"].detach()
    grad = store["grad"].detach()
    grad2 = grad * grad
    grad3 = grad2 * grad
    alpha_num = grad2
    alpha_den = 2.0 * grad2 + (act * grad3).sum(dim=(2, 3), keepdim=True) + 1e-8
    alpha = alpha_num / alpha_den
    weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    return _upsample(cam, x.shape[2], x.shape[3])


def eigencam(model, x, class_idx=None, target_layer=None) -> np.ndarray:
    layer = _find_conv(model, target_layer)
    store, h1, h2 = _hooks(layer)
    model.zero_grad()
    out = _forward(model, x)
    c = class_idx if class_idx is not None else out.argmax(1).item()
    out[0, c].backward(retain_graph=True)
    h1.remove(); h2.remove()
    act = store["act"].detach()
    grad = store["grad"].detach()
    weight = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weight * act).sum(dim=1, keepdim=True)
    cam = cam.squeeze(1)
    ch = cam.shape[0]
    flat = cam.view(ch, -1).cpu().numpy()
    cov = flat @ flat.T
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc = eigvecs[:, -1]
        cam2d = pc @ flat
    except Exception:
        cam2d = flat.mean(axis=0)
    h, w = cam.shape[1], cam.shape[2]
    cam2d = np.maximum(cam2d, 0).reshape(h, w).astype(np.float32)
    cmin, cmax = cam2d.min(), cam2d.max()
    return ((cam2d - cmin) / (cmax - cmin + 1e-8)).astype(np.float32)


def layercam(model, x, class_idx=None, target_layer=None) -> np.ndarray:
    layer = _find_conv(model, target_layer)
    store, h1, h2 = _hooks(layer)
    model.zero_grad()
    out = _forward(model, x)
    c = class_idx if class_idx is not None else out.argmax(1).item()
    out[0, c].backward(retain_graph=True)
    h1.remove(); h2.remove()
    act = store["act"].detach()
    grad = store["grad"].detach()
    cam = (act * grad).sum(dim=1, keepdim=True)
    return _upsample(cam, x.shape[2], x.shape[3])


def scorecam(model, x, class_idx=None, target_layer=None, n_bins: int = 8) -> np.ndarray:
    layer = _find_conv(model, target_layer)
    store, h1, _ = _hooks(layer)
    out = _forward(model, x)
    c = class_idx if class_idx is not None else out.argmax(1).item()
    h1.remove()
    act = store["act"].detach()
    B, C, h, w = act.shape
    cam = torch.zeros(h, w, device=x.device)
    for i in range(C):
        m = F.interpolate(act[:, i:i+1], size=(h, w), mode="bilinear", align_corners=False)
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        if i % (max(C // n_bins, 1)) != 0:
            continue
        masked = x * (1.0 - m)
        with torch.no_grad():
            s = torch.softmax(_forward(model, masked)[0], dim=0)[c]
        cam += m.squeeze() * s.item()
    return _upsample(cam.unsqueeze(0).unsqueeze(0), x.shape[2], x.shape[3])


def integrated_gradients(model, x, class_idx=None, steps: int = 40) -> np.ndarray:
    x = x.detach().requires_grad_(True)
    baseline = torch.zeros_like(x)
    out = _forward(model, x)
    c = class_idx if class_idx is not None else out.argmax(1).item()
    grads = []
    for k in range(1, steps + 1):
        xk = baseline + (k / steps) * (x - baseline)
        xk = xk.detach().requires_grad_(True)
        with torch.enable_grad():
            outk = _forward(model, xk)
            outk[0, c].backward()
        grads.append(xk.grad.detach())
    ig = (x - baseline) * torch.stack(grads).mean(dim=0)
    return _grad_heatmap(ig)


def occlusion(model, x, class_idx=None, patch: int = 16, stride: int = 8) -> np.ndarray:
    x = x.detach()
    out = _forward(model, x)
    c = class_idx if class_idx is not None else out.argmax(1).item()
    base = out[0, c].item()
    C, H, W = x.shape[1], x.shape[2], x.shape[3]
    heat = np.zeros((H, W))
    counts = np.zeros((H, W))
    for r in range(0, H - patch + 1, stride):
        for col in range(0, W - patch + 1, stride):
            xm = x.clone()
            xm[0, :, r:r + patch, col:col + patch] = 0.0
            with torch.no_grad():
                score = _forward(model, xm)[0, c].item()
            heat[r:r + patch, col:col + patch] += (base - score)
            counts[r:r + patch, col:col + patch] += 1
    heat = np.divide(heat, np.maximum(counts, 1))
    heat = np.maximum(heat, 0)
    cmin, cmax = heat.min(), heat.max()
    return ((heat - cmin) / (cmax - cmin + 1e-8)).astype(np.float32)


def vanilla_saliency(model, x, class_idx=None) -> np.ndarray:
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        out = _forward(model, x)
        c = class_idx if class_idx is not None else out.argmax(1).item()
        out[0, c].backward()
    grad = x.grad.abs()
    return _grad_heatmap(grad)


def _grad_heatmap(grad: torch.Tensor) -> np.ndarray:
    cam = grad.abs().amax(dim=1, keepdim=True)
    return _upsample(cam, grad.shape[2], grad.shape[3])


METHODS = {
    "GradCAM": gradcam,
    "GradCAM++": gradcam_pp,
    "EigenCAM": eigencam,
    "LayerCAM": layercam,
    "ScoreCAM": scorecam,
    "Integrated Gradients": integrated_gradients,
    "Occlusion": occlusion,
    "Vanilla Saliency": vanilla_saliency,
}


def explain(method: str, model, x, class_idx=None, **kw) -> np.ndarray:
    return METHODS[method](model, x, class_idx=class_idx, **kw)
