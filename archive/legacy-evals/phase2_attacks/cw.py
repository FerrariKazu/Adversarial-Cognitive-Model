"""
C&W — Carlini & Wagner L2 Attack (Carlini & Wagner, 2017)
=========================================================

THE MATH (plain English):
    Unlike FGSM/PGD which use the L∞ norm (max pixel change), C&W minimizes the
    L2 norm (Euclidean distance) of the perturbation:

        minimize  ‖δ‖₂  +  c · f(x + δ)

    where f(x+δ) is an objective function that equals zero when x+δ is
    misclassified, and is positive otherwise. The constant c balances between
    "make the perturbation small" and "make the attack succeed."

    This is solved via gradient descent on a change-of-variables trick:
        δ = ½(tanh(w) + 1) - x
    which automatically ensures the adversarial image stays in [0, 1].

L2 NORM vs L∞ NORM — What's the difference?
    - L∞ (FGSM/PGD): The MAXIMUM change to any single pixel.
      Think: "no pixel moves more than ε."
      Analogy: A speed limit — every car (pixel) stays under the limit.

    - L2 (C&W): The TOTAL Euclidean distance across ALL pixels.
      Think: "the overall image hasn't moved far in pixel space."
      Analogy: A fuel budget — you can drive one car fast OR many cars slowly,
      but your total fuel is capped.

    Practically, L2 attacks tend to create SMOOTHER perturbations because they
    can make large changes to a few pixels while leaving others untouched.
    L∞ attacks create UNIFORM perturbations where every pixel changes by ~ε.

PSYCHOPHYSICS CONNECTION — Just-Noticeable Difference (JND):
    In psychophysics, the Just-Noticeable Difference (JND) is the smallest
    change in a stimulus that a human can detect 50% of the time (Weber's Law).

    C&W's L2 minimization directly maps to finding perturbations near the JND
    threshold. By minimizing the total perturbation magnitude:
    - If ‖δ‖₂ < JND threshold → Humans cannot see the change at all
    - If ‖δ‖₂ ≈ JND threshold → Humans see "something is off" but can't name it
    - If ‖δ‖₂ > JND threshold → Humans clearly see the corruption

    This makes C&W the most psychophysically relevant attack for our study.
    When we present C&W-attacked images to human participants (Phase 3), we are
    directly testing: "Can CNNs be fooled by perturbations below the human JND?"

DECISION BOUNDARY EXPLANATION:
    C&W finds the CLOSEST point on the wrong side of the decision boundary,
    measured by Euclidean distance. FGSM/PGD find a point within an L∞ box.
    Geometrically, C&W draws the smallest possible sphere around the original
    image that touches the decision boundary — it finds the nearest adversarial
    example in the most natural sense of "nearest."
"""

import torch
import torch.nn as nn
import torchattacks


def cw_attack(model, images, labels, device, c=1.0, kappa=0, steps=100, lr=0.01):
    """
    Perform a C&W L2 attack using the torchattacks library.

    Parameters
    ----------
    model  : nn.Module    — Target classifier (eval mode).
    images : Tensor       — Clean images [B, 3, 32, 32], normalized.
    labels : Tensor       — True labels [B].
    device : torch.device — 'cuda' or 'cpu'.
    c      : float        — Confidence parameter. Higher c → attack tries harder
                            to misclassify, at the cost of larger perturbation.
    kappa  : float        — Confidence margin. κ=0 means "just barely misclassify."
                            κ>0 means "misclassify with high confidence."
    steps  : int          — Optimization steps for the C&W inner loop.
    lr     : float        — Learning rate for the C&W optimizer.

    Returns
    -------
    adv_images : Tensor — Adversarial images.
    adv_preds  : Tensor — Model predictions on adversarial images.

    NOTE ON IMPLEMENTATION:
        We use torchattacks.CW rather than implementing C&W from scratch because:
        1. C&W involves a complex binary search over the constant c, a tanh
           change-of-variables, and careful optimizer tuning. Getting this wrong
           produces weak attacks that don't reflect C&W's true power.
        2. torchattacks is battle-tested and matches the original paper's results.
        3. For our research question (human vs CNN perception), the attack quality
           matters more than reimplementation pedagogy.
    """
    images = images.to(device)
    labels = labels.to(device)

    # -------------------------------------------------------------------------
    # Initialize the C&W attack from torchattacks
    # -------------------------------------------------------------------------
    # 1. WHAT: Creates a CW attack object configured with our parameters.
    # 2. WHY: torchattacks handles the tanh reparameterization, binary search
    #         over c, and Adam optimizer internally. We just call it like a
    #         function.
    # 3. OBSERVE: C&W is MUCH slower than FGSM/PGD because it solves an
    #         optimization problem per batch. Expect ~10x longer runtime.
    # -------------------------------------------------------------------------
    attack = torchattacks.CW(model, c=c, kappa=kappa, steps=steps, lr=lr)

    # -------------------------------------------------------------------------
    # Generate adversarial examples
    # -------------------------------------------------------------------------
    # 1. WHAT: Runs the full C&W optimization loop.
    # 2. WHY: The attack internally does:
    #         a) Initialize w = atanh(2x - 1)
    #         b) Optimize w via Adam to minimize ‖½(tanh(w)+1) - x‖₂ + c·f(x+δ)
    #         c) Binary search c to find the smallest perturbation that succeeds
    # 3. OBSERVE: The resulting perturbations will have VARYING L2 norms per
    #         image (unlike FGSM/PGD which have uniform L∞ = ε). Some images
    #         are easier to fool (closer to the decision boundary) and will
    #         have smaller perturbations.
    # -------------------------------------------------------------------------
    adv_images = attack(images, labels)

    # -------------------------------------------------------------------------
    # Get predictions
    # -------------------------------------------------------------------------
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_preds = adv_outputs.argmax(dim=1)

    return adv_images.detach(), adv_preds
