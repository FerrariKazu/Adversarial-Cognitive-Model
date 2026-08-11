"""Adversarial attack library + perturbation distance metrics.

Convention: attacks operate on *normalized* input tensors (1,3,H,W) with a
per-channel epsilon bound (scalar or (3,) tensor) and explicit domain clamps —
identical to the Finding-17 matched evaluation convention used across the repo.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from cognitive_vision_lab.config import STL10_MEAN, STL10_STD


# ── Forward helpers ───────────────────────────────────────────────────────────
def _forward(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def domain_bounds(mean, std, device) -> tuple[torch.Tensor, torch.Tensor]:
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (-m / s), ((1.0 - m) / s)


def _eps_vec(eps, device) -> torch.Tensor:
    if torch.is_tensor(eps):
        return eps.to(device).view(1, -1, 1, 1)
    return torch.full((1, 3, 1, 1), float(eps), device=device)


def _kl_pgd_step(model, x, x0, probs_c, eps, alpha, stl_min, stl_max):
    """One TRADES-style KL PGD step (matches eval_full_epsilon_sweep)."""
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        logits = _forward(model, x)
        loss = F.kl_div(F.log_softmax(logits.float(), dim=1), probs_c, reduction="batchmean")
    grad = torch.autograd.grad(loss, x)[0]
    x = x.detach() + alpha * grad.sign()
    delta = torch.clamp(x - x0, -eps, eps)
    return torch.clamp(x0 + delta, stl_min, stl_max).detach()


# ── Attacks ───────────────────────────────────────────────────────────────────
def fgsm(model, x, y, eps=0.031, targeted=False, mean=STL10_MEAN, std=STL10_STD,
         device=None):
    device = device or x.device
    eps = _eps_vec(eps, device)
    lo, hi = domain_bounds(mean, std, device)
    xa = x.detach().clone().requires_grad_(True)
    logits = _forward(model, xa)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, xa)[0]
    sign = grad.sign() if not targeted else -grad.sign()
    xa = xa.detach() + eps * sign
    delta = torch.clamp(xa - x, -eps, eps)
    return torch.clamp(x + delta, lo, hi).detach()


def pgd(model, x, y, eps=0.031, steps=50, alpha=None, random_start=True,
        targeted=False, mean=STL10_MEAN, std=STL10_STD, device=None,
        use_kl=False, progress: Optional[Callable[[int], None]] = None):
    """Projected gradient descent (CE or KL form). alpha defaults to eps/4."""
    device = device or x.device
    eps = _eps_vec(eps, device)
    lo, hi = domain_bounds(mean, std, device)
    alpha = alpha or (eps / 4.0)
    x0 = x.detach()

    if use_kl:
        with torch.no_grad():
            logits_c = _forward(model, x0)
        probs_c = F.softmax(logits_c.float(), dim=1)

    xa = x0.clone()
    if random_start:
        # uniform(-eps, +eps) elementwise; eps may be a per-channel tensor
        xa = xa + torch.rand_like(xa) * 2 * eps - eps
    xa = torch.clamp(xa, lo, hi)

    for i in range(steps):
        if use_kl:
            xa = _kl_pgd_step(model, xa, x0, probs_c, eps, alpha, lo, hi)
        else:
            xa = xa.detach().requires_grad_(True)
            with torch.enable_grad():
                logits = _forward(model, xa)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, xa)[0]
            sign = grad.sign() if not targeted else -grad.sign()
            xa = xa.detach() + alpha * sign
            delta = torch.clamp(xa - x0, -eps, eps)
            xa = torch.clamp(x0 + delta, lo, hi).detach()
        if progress is not None:
            progress(i)
    return xa


def apgd(model, x, y, eps=0.031, steps=50, targeted=False, mean=STL10_MEAN,
         std=STL10_STD, device=None):
    """Auto-PGD: PGD with step-size decay on stalled progress."""
    device = device or x.device
    eps = _eps_vec(eps, device)
    lo, hi = domain_bounds(mean, std, device)
    alpha = eps / 4.0
    x0 = x.detach()
    xa = torch.clamp(x0 + torch.rand_like(x0) * 2 * eps - eps, lo, hi)
    best = xa.clone()
    best_loss = -1e18 if targeted else 1e18
    stall = 0
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(_forward(model, xa), y)
        grad = torch.autograd.grad(loss, xa)[0]
        sign = grad.sign() if not targeted else -grad.sign()
        xa = xa.detach() + alpha * sign
        delta = torch.clamp(xa - x0, -eps, eps)
        xa = torch.clamp(x0 + delta, lo, hi).detach()
        with torch.no_grad():
            cur = F.cross_entropy(_forward(model, xa), y).item()
        improved = cur < best_loss if not targeted else cur > best_loss
        if improved:
            best_loss = cur
            best = xa.clone()
            stall = 0
        else:
            stall += 1
        if stall >= 4:
            alpha = alpha * 0.75
            stall = 0
    return best


def cw_l2(model, x, y, steps=150, c=1.0, kappa=0.0, lr=0.01, targeted=False,
          mean=STL10_MEAN, std=STL10_STD, device=None):
    """Carlini-Wagner L2 attack via Adam in tanh space (simplified)."""
    device = device or x.device
    lo, hi = domain_bounds(mean, std, device)
    x0 = torch.clamp(x.detach(), lo, hi)
    w = torch.zeros_like(x0, requires_grad=True)
    y_target = y if targeted else None
    opt = torch.optim.Adam([w], lr=lr)
    best_adv, best_dist = None, float("inf")
    for _ in range(steps):
        opt.zero_grad()
        xa = (0.5 * (torch.tanh(w) + 1.0)) * (hi - lo) + lo
        logits = _forward(model, xa)
        dist = torch.norm((xa - x0).flatten(), 2)
        if targeted:
            loss = F.cross_entropy(logits, y_target)
            obj = loss + c * torch.relu(dist - 0.001)
        else:
            wrong = logits.clone()
            wrong[0, y] = -1e9
            obj = torch.max(wrong, dim=1).values - logits[0, y] + kappa
            obj = torch.clamp(obj, min=0.0) + c * dist
        obj.backward()
        opt.step()
        with torch.no_grad():
            d = torch.norm((xa - x0).flatten(), 2).item()
            if d < best_dist:
                best_dist = d
                best_adv = xa.detach().clone()
    return best_adv if best_adv is not None else x0.detach()


def deepfool(model, x, y=None, steps=50, overshoot=0.02, mean=STL10_MEAN,
             std=STL10_STD, device=None):
    """DeepFool: minimal L2 crossing of the linearized decision boundary."""
    device = device or x.device
    lo, hi = domain_bounds(mean, std, device)
    xa = x.detach().clone()
    total = torch.zeros_like(xa)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = _forward(model, xa)
        pred = logits.argmax(dim=1)
        per_class_grad = []
        for c in range(logits.shape[1]):
            if c == pred.item():
                continue
            xa2 = xa.detach().requires_grad_(True)
            with torch.enable_grad():
                logits2 = _forward(model, xa2)
                l2 = logits2[0, pred] - logits2[0, c]
            g2 = torch.autograd.grad(l2, xa2)[0]
            denom = torch.norm(g2.flatten(), 2)
            if denom < 1e-8:
                continue
            w = g2 / denom
            f = l2.item() / denom
            per_class_grad.append((abs(f) + 1e-8, w, f))
        if not per_class_grad:
            break
        _, w, f = min(per_class_grad, key=lambda t: t[0])
        step = (abs(f) + overshoot) * w
        xa = torch.clamp(xa + step, lo, hi).detach()
        total = total + step
        with torch.no_grad():
            if _forward(model, xa).argmax(dim=1) != pred:
                break
    return torch.clamp(x + total, lo, hi).detach()


def square(model, x, y, eps=0.031, steps=1000, p_init=0.05, mean=STL10_MEAN,
           std=STL10_STD, device=None, seed: Optional[int] = None):
    """Square Attack (query-based): random square-shaped perturbation updates."""
    rng = np.random.default_rng(seed)
    device = device or x.device
    eps = _eps_vec(eps, device).squeeze(0)
    lo, hi = domain_bounds(mean, std, device)
    x0 = x.detach()
    xa = x0.clone()
    C, H, W = x0.shape[1], x0.shape[2], x0.shape[3]

    def loss_at(t):
        with torch.no_grad():
            logits = _forward(model, t)
            return F.cross_entropy(logits, y).item()

    best_x, best_l = xa.clone(), loss_at(xa)
    for _ in range(steps):
        side = max(2, int(math.sqrt(p_init * H * W)))
        ch = int(rng.integers(0, C))
        r0, c0 = int(rng.integers(0, H - side + 1)), int(rng.integers(0, W - side + 1))
        cand = best_x.clone()
        sign = 1.0 if rng.random() < 0.5 else -1.0
        cand[0, ch, r0:r0 + side, c0:c0 + side] += sign * eps[ch] * 0.5
        delta = torch.clamp(cand - x0, -eps.view(1, 3, 1, 1), eps.view(1, 3, 1, 1))
        cand = torch.clamp(x0 + delta, lo, hi)
        l = loss_at(cand)
        if l < best_l:
            best_l, best_x = l, cand
    return best_x


def fab(model, x, y, eps=0.031, steps=50, eta=0.05, mean=STL10_MEAN,
        std=STL10_STD, device=None):
    """FAB (simplified): projection onto the boundary with adaptivity."""
    device = device or x.device
    eps = _eps_vec(eps, device)
    lo, hi = domain_bounds(mean, std, device)
    x0 = x.detach()
    xa = x0.clone()
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = _forward(model, xa)
            pred = logits.argmax(dim=1)
            loss = logits[0, pred]
        grad = torch.autograd.grad(loss, xa)[0]
        with torch.no_grad():
            d = grad / (torch.norm(grad.flatten(), 2) + 1e-8)
            margin = (logits[0, pred] - logits[0, (pred + 1) % logits.shape[1]]).item()
            xa = xa + eta * (margin + 0.1) * d
            delta = torch.clamp(xa - x0, -eps, eps)
            xa = torch.clamp(x0 + delta, lo, hi).detach()
    return xa


ATTACKS: dict[str, Callable] = {
    "fgsm": fgsm,
    "pgd": pgd,
    "apgd": apgd,
    "cw": cw_l2,
    "deepfool": deepfool,
    "square": square,
    "fab": fab,
}


def run_attack(name: str, model, x, y, **kwargs) -> torch.Tensor:
    return ATTACKS[name](model, x, y, **kwargs)


# ── Distance metrics ──────────────────────────────────────────────────────────
def _gaussian_window(window_size: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    w = g.outer(g).view(1, 1, window_size, window_size).expand(channels, 1, window_size, window_size)
    return w.contiguous()


def _ssim_torch(a: torch.Tensor, b: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    c1, c2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    ch = a.shape[1]
    win = _gaussian_window(window_size, 1.5, ch, a.device)
    mu_a = F.conv2d(a, win, padding=window_size // 2, groups=ch)
    mu_b = F.conv2d(b, win, padding=window_size // 2, groups=ch)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_aa = F.conv2d(a * a, win, padding=window_size // 2, groups=ch) - mu_aa
    sigma_bb = F.conv2d(b * b, win, padding=window_size // 2, groups=ch) - mu_bb
    sigma_ab = F.conv2d(a * b, win, padding=window_size // 2, groups=ch) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_aa + mu_bb + c1) * (sigma_aa + sigma_bb + c2)
    )
    return ssim_map.mean().item()


def distance_metrics(orig: torch.Tensor, adv: torch.Tensor,
                     mean=STL10_MEAN, std=STL10_STD) -> dict:
    """Perturbation distances in pixel space [0,1] + SSIM/PSNR + perceptual proxy."""
    device = orig.device
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    a = (orig * s + m).clamp(0, 1)
    b = (adv * s + m).clamp(0, 1)
    diff = (b - a).clamp(-1, 1)
    l1 = diff.abs().mean().item() * 255.0
    l2 = torch.norm(diff, 2).item() * 255.0
    linf = diff.abs().max().item() * 255.0
    mse = (diff**2).mean().item()
    psnr = float("inf") if mse < 1e-12 else 10.0 * math.log10(1.0 / max(mse, 1e-12))
    ssim = _ssim_torch(a, b)
    return {
        "L1 (mean px)": round(l1, 4),
        "L2 (total)": round(l2, 4),
        "L∞ (max px)": round(linf, 4),
        "SSIM": round(ssim, 4),
        "PSNR (dB)": round(psnr, 2),
        "MSE": round(mse, 6),
    }
