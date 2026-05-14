"""
FGSM — Fast Gradient Sign Method (Goodfellow et al., 2015)
===========================================================

THE MATH (plain English):
    Given a correctly classified image x with true label y, we want to find the
    smallest perturbation δ that makes the model misclassify x+δ.

    FGSM computes the gradient of the loss J(θ, x, y) with respect to the INPUT
    pixels x (not the weights θ — that's the key insight).

        x_adv = x + ε · sign(∇_x J(θ, x, y))

    Breaking this down:
    1. ∇_x J  — "In which direction should I change each pixel to INCREASE the
                 loss (make the model more wrong)?"
    2. sign() — We only care about the DIRECTION (positive or negative), not the
                magnitude. Every pixel gets nudged by exactly ±ε.
    3. ε      — The budget. How far each pixel is allowed to move.

WHAT THE GRADIENT SIGN MEANS INTUITIVELY:
    Imagine you're standing on a hillside (the loss landscape). The gradient
    tells you which direction is uphill. sign() says "just go uphill as fast
    as possible" — don't bother with gentle slopes vs steep slopes, every
    dimension takes a full step. This is the L∞ steepest-ascent direction.

COGNITIVE SCIENCE ANALOG:
    FGSM is like adding uniform static noise to a TV signal — every pixel gets
    the same magnitude of corruption (±ε). In human vision, this corresponds to
    additive luminance noise, the kind you'd see on a de-tuned CRT television.
    Humans are remarkably good at filtering this kind of noise (our visual cortex
    has built-in noise-rejection circuitry), but CNNs have no such mechanism.

DECISION BOUNDARY EXPLANATION:
    In a 3072-dimensional pixel space (32×32×3), the model's decision boundary
    is a complex hypersurface. FGSM finds a single direction that crosses this
    boundary as efficiently as possible. Because it only takes ONE step, it
    sometimes overshoots or undershoots — that's why PGD (multi-step) is stronger.
"""

import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon, device, clip_min=None, clip_max=None):
    """
    Perform a single-step FGSM attack.

    Parameters
    ----------
    model    : nn.Module   — The target classifier (must be in eval mode).
    images   : Tensor       — Clean images, shape [B, 3, 32, 32], ALREADY NORMALIZED.
    labels   : Tensor       — Ground-truth class indices, shape [B].
    epsilon  : float        — Perturbation budget in NORMALIZED pixel space.
    device   : torch.device — 'cuda' or 'cpu'.
    clip_min : Tensor       — Optional minimum clipping bounds [1, 3, 1, 1].
    clip_max : Tensor       — Optional maximum clipping bounds [1, 3, 1, 1].
    """
    images_var = images.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)

    outputs = model(images_var)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images_var.grad.data.sign()
    adv_images = images_var.data + epsilon * grad_sign

    # Optional clamping to valid pixel range
    if clip_min is not None and clip_max is not None:
        adv_images = torch.max(torch.min(adv_images, clip_max), clip_min)

    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_preds = adv_outputs.argmax(dim=1)

    return adv_images.detach(), adv_preds

    # -------------------------------------------------------------------------
    # Step 5: Get model predictions on adversarial images
    # -------------------------------------------------------------------------
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_preds = adv_outputs.argmax(dim=1)

    return adv_images.detach(), adv_preds
