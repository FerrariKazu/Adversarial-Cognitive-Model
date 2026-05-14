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


def fgsm_attack(model, images, labels, epsilon, device):
    """
    Perform a single-step FGSM attack.

    Parameters
    ----------
    model   : nn.Module   — The target classifier (must be in eval mode).
    images  : Tensor       — Clean images, shape [B, 3, 32, 32], ALREADY NORMALIZED.
    labels  : Tensor       — Ground-truth class indices, shape [B].
    epsilon : float        — Perturbation budget in NORMALIZED pixel space.
                             Because our images are normalized by CIFAR-10's
                             mean/std, ε=0.01 here does NOT mean 0.01 in [0,255].
                             It means 0.01 in the normalized space, which is
                             roughly ε * std ≈ 0.002 in raw pixel fraction.
    device  : torch.device — 'cuda' or 'cpu'.

    Returns
    -------
    adv_images : Tensor — Adversarial images, same shape as input.
    adv_preds  : Tensor — Model predictions on the adversarial images.

    IMPORTANT — Normalization caveat:
        Our attacks operate in the SAME normalized space that the model sees.
        This means the perturbation magnitude ε is measured in standard-deviation
        units, not raw pixel values. This is correct because we want the attack
        to match what the model actually processes. When we later visualize the
        images for humans (Phase 3), we will UN-normalize them back to [0, 1].
    """
    # -------------------------------------------------------------------------
    # Step 1: Enable gradient computation on the INPUT image
    # -------------------------------------------------------------------------
    # 1. WHAT: We clone the images and set requires_grad=True.
    # 2. WHY: Normally PyTorch only tracks gradients for model parameters
    #         (weights). By enabling gradients on the image tensor, we can
    #         compute ∂Loss/∂pixel — "how does each pixel affect the loss?"
    # 3. OBSERVE: After this, images_var will have a .grad attribute.
    # -------------------------------------------------------------------------
    images_var = images.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)

    # -------------------------------------------------------------------------
    # Step 2: Forward pass — compute the loss
    # -------------------------------------------------------------------------
    # 1. WHAT: Run the model and compute CrossEntropyLoss.
    # 2. WHY: We need the loss value to differentiate with respect to the input.
    #         CrossEntropy is the same loss used during training — this means
    #         we're asking "how can I make the model's training objective worse?"
    # 3. OBSERVE: loss is a single scalar value.
    # -------------------------------------------------------------------------
    outputs = model(images_var)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    # -------------------------------------------------------------------------
    # Step 3: Backward pass — compute ∇_x J
    # -------------------------------------------------------------------------
    # 1. WHAT: Backpropagate the loss to the input pixels.
    # 2. WHY: This fills images_var.grad with the gradient of the loss w.r.t.
    #         each pixel. Positive gradient = "increasing this pixel increases
    #         the loss". Negative gradient = "decreasing this pixel increases
    #         the loss".
    # 3. OBSERVE: images_var.grad will have the same shape as images [B,3,32,32].
    # -------------------------------------------------------------------------
    model.zero_grad()
    loss.backward()

    # -------------------------------------------------------------------------
    # Step 4: Create the adversarial image
    # -------------------------------------------------------------------------
    # 1. WHAT: x_adv = x + ε · sign(∇_x J)
    # 2. WHY: sign() converts the gradient into a direction-only signal (+1 or -1).
    #         Multiplying by ε ensures every pixel moves by exactly ε in the
    #         direction that maximizes loss. This is the optimal single-step
    #         perturbation under the L∞ norm.
    # 3. OBSERVE: The perturbation is exactly ε everywhere — a uniform-magnitude
    #         noise pattern, but with structure (it's not random noise).
    # -------------------------------------------------------------------------
    grad_sign = images_var.grad.data.sign()
    adv_images = images_var.data + epsilon * grad_sign

    # -------------------------------------------------------------------------
    # Step 5: Get model predictions on adversarial images
    # -------------------------------------------------------------------------
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_preds = adv_outputs.argmax(dim=1)

    return adv_images.detach(), adv_preds
