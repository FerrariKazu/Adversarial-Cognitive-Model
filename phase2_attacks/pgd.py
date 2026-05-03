"""
PGD — Projected Gradient Descent (Madry et al., 2018)
=====================================================

THE MATH (plain English):
    PGD is FGSM applied iteratively. Instead of one big step of size ε, PGD
    takes many small steps of size α, each time re-computing the gradient:

        x_0   = x + uniform_noise(-ε, ε)          ← random start
        x_t+1 = Π_{B(x,ε)} [ x_t + α · sign(∇_x J(θ, x_t, y)) ]

    Breaking this down:
    1. Random start:  We begin at a random point inside the ε-ball around x.
    2. Gradient step: Same as FGSM — take a step in the sign-of-gradient direction.
    3. Projection Π:  After each step, we project back onto the ε-ball. This
                      ensures the total perturbation never exceeds ε in L∞ norm.

WHY THE PROJECTION STEP:
    After each gradient step, the perturbed image might "drift" outside the
    allowed ε-ball. The projection step clamps the perturbation δ = x_adv - x
    to lie within [-ε, +ε] for every pixel. Think of it as a leash — the
    adversarial image can wander in any direction, but it's always yanked back
    if it strays too far from the original.

    We also clamp to valid image range because extreme perturbations could push
    pixel values to physically impossible values (below 0 or above 1 in the
    unnormalized space). In normalized space, we don't strictly enforce [0,1]
    because the normalization already maps the valid range to roughly [-2.4, 2.8].

WHY RANDOM START MAKES PGD STRONGER THAN FGSM:
    FGSM always starts from the clean image and takes one step. If the loss
    landscape has local flat regions or saddle points around x, FGSM might
    find a weak perturbation. Random start lets PGD explore different "attack
    angles" — by starting from various points in the ε-ball, PGD can find
    adversarial examples that FGSM misses. Empirically, PGD with random
    restarts finds adversarial examples ~10-30% more often than FGSM at
    the same ε budget.

COGNITIVE SCIENCE COMPARISON — PGD vs FGSM:
    If FGSM is like adding static noise to a TV signal (one uniform corruption),
    PGD is like an artist carefully retouching a photograph. Each iteration
    refines the perturbation, making it more targeted. In psychophysics terms:
    - FGSM perturbations tend to be "broadband" — they affect all spatial
      frequencies roughly equally, like Gaussian noise.
    - PGD perturbations become "tuned" — they concentrate energy in the spatial
      frequencies and orientations that the CNN is most sensitive to.
    This is why PGD-perturbed images often look MORE natural to humans than
    FGSM images at the same ε, even though they fool the model more reliably.

DECISION BOUNDARY EXPLANATION:
    FGSM takes one linear approximation of the decision boundary and jumps
    across it. PGD walks along the boundary, probing it from multiple angles,
    and finds a crossing point that is both closer to the original image and
    more reliably on the wrong side. PGD is the "strongest first-order attack"
    — if your model survives PGD, it's robust against all gradient-based attacks.
"""

import torch
import torch.nn as nn


def pgd_attack(model, images, labels, epsilon, alpha, steps, device):
    """
    Perform a multi-step PGD attack with random initialization.

    Parameters
    ----------
    model   : nn.Module    — Target classifier (eval mode).
    images  : Tensor       — Clean images [B, 3, 32, 32], normalized.
    labels  : Tensor       — True labels [B].
    epsilon : float        — Maximum L∞ perturbation budget (normalized space).
    alpha   : float        — Step size per iteration (typically ε/4 or 2.5*ε/steps).
    steps   : int          — Number of PGD iterations (20 is standard).
    device  : torch.device — 'cuda' or 'cpu'.

    Returns
    -------
    adv_images : Tensor — Adversarial images.
    adv_preds  : Tensor — Model predictions on adversarial images.
    """
    images = images.to(device)
    labels = labels.to(device)

    # -------------------------------------------------------------------------
    # Step 1: Random initialization within the ε-ball
    # -------------------------------------------------------------------------
    # 1. WHAT: Start from x + uniform(-ε, ε) instead of x itself.
    # 2. WHY: Random start prevents PGD from getting stuck in the same local
    #         region every time. It's the key difference that makes PGD strictly
    #         stronger than FGSM. Multiple random restarts can find even
    #         stronger attacks, but one restart is standard for efficiency.
    # 3. OBSERVE: adv_images starts as a noisy version of the clean input.
    # -------------------------------------------------------------------------
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = adv_images.detach()

    for i in range(steps):
        # ---------------------------------------------------------------------
        # Step 2: Compute gradient (same as FGSM's core step)
        # ---------------------------------------------------------------------
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        # ---------------------------------------------------------------------
        # Step 3: Take a small step in the sign-of-gradient direction
        # ---------------------------------------------------------------------
        # 1. WHAT: x_t+1 = x_t + α · sign(∇_x J)
        # 2. WHY: α is much smaller than ε (typically ε/4). This small step
        #         size lets PGD navigate the loss landscape carefully, finding
        #         better adversarial examples than FGSM's single large step.
        # 3. OBSERVE: Each iteration slightly refines the perturbation.
        # ---------------------------------------------------------------------
        adv_images = adv_images.detach() + alpha * adv_images.grad.sign()

        # ---------------------------------------------------------------------
        # Step 4: Project back onto the ε-ball (THE PROJECTION STEP)
        # ---------------------------------------------------------------------
        # 1. WHAT: Clamp the perturbation δ = (adv - original) to [-ε, +ε].
        # 2. WHY: Without projection, iterative steps could accumulate a total
        #         perturbation much larger than ε. The projection ensures we
        #         stay within the threat model — the attacker's "budget".
        #         Geometrically, this is projecting onto the L∞ ball centered
        #         at the original image.
        # 3. OBSERVE: After this, max(|adv_pixel - clean_pixel|) ≤ ε always.
        # ---------------------------------------------------------------------
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = (images + delta).detach()

    # -------------------------------------------------------------------------
    # Step 5: Get predictions on the final adversarial images
    # -------------------------------------------------------------------------
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_preds = adv_outputs.argmax(dim=1)

    return adv_images, adv_preds
