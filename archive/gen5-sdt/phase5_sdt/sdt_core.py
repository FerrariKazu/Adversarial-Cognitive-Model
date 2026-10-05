"""
Signal Detection Theory (SDT) — Core Computations
===================================================

A COMPLETE PRIMER ON SDT FOR IMAGE CLASSIFICATION
(Read this before looking at any function — it teaches the theory from scratch.)

═══════════════════════════════════════════════════════════════════════════════
1. THE PROBLEM WITH RAW ACCURACY
═══════════════════════════════════════════════════════════════════════════════

Imagine a classifier that says "cat" for every single image. On a dataset
where 50% of images are cats, it scores 50% accuracy. But it has ZERO ability
to distinguish cats from non-cats — it's just biased toward saying "cat."

Raw accuracy confounds two things:
    (a) How SENSITIVE the system is to the signal (can it actually tell cats apart?)
    (b) How BIASED the system is (does it prefer to say "yes" or "no"?)

SDT separates them. That's why psychophysics papers use d-prime instead of
accuracy — and why your professor will take this analysis seriously.

═══════════════════════════════════════════════════════════════════════════════
2. THE FOUR OUTCOMES (FOR ONE CLASS AT A TIME)
═══════════════════════════════════════════════════════════════════════════════

For each class (e.g., "cat"), every prediction falls into one of four cells:

                        True Label
                    Cat         Not Cat
                ┌───────────┬───────────┐
    Predicted   │   HIT     │  FALSE    │
    Cat         │  (correct)│  ALARM    │
                ├───────────┼───────────┤
    Predicted   │   MISS    │  CORRECT  │
    Not Cat     │           │ REJECTION │
                └───────────┴───────────┘

    HIT:               "It IS a cat and I said cat."       (True Positive)
    FALSE ALARM (FA):  "It is NOT a cat but I said cat."   (False Positive)
    MISS:              "It IS a cat but I said something else."  (False Negative)
    CORRECT REJECTION: "It's NOT a cat and I didn't say cat."   (True Negative)

    Hit Rate (HR) = Hits / (Hits + Misses)                       [0, 1]
    False Alarm Rate (FAR) = False Alarms / (FAs + Correct Rejects)  [0, 1]

═══════════════════════════════════════════════════════════════════════════════
3. d-PRIME (d') — PERCEPTUAL SENSITIVITY
═══════════════════════════════════════════════════════════════════════════════

    d' = Z(Hit Rate) − Z(False Alarm Rate)

where Z is the inverse of the standard normal CDF (scipy.stats.norm.ppf).

Interpretation of d' values:
    d' ≈ 0.0   → No discrimination. The system is at chance. It literally
                  cannot tell signal from noise.
    d' ≈ 1.0   → Weak discrimination. Barely above chance. In psychophysics,
                  this is the conventional "threshold" — below d'=1.0 the system
                  is considered to have effectively lost the ability to detect
                  the signal.
    d' ≈ 2.0   → Moderate discrimination. Comfortable detection.
    d' ≈ 3.0   → Strong discrimination. Reliable detection.
    d' ≈ 4.65  → Near-perfect discrimination. HR≈99%, FAR≈1%. This is what a
                  well-trained model achieves on clean images.

Why is d' better than accuracy?
    • Two classifiers can have IDENTICAL accuracy but different d'. One might
      have high HR + high FAR (liberal bias), another might have moderate HR +
      very low FAR (conservative bias). Accuracy merges them; d' separates them.
    • When comparing CNN vs human under adversarial attack, d' reveals whether
      the system truly lost sensitivity or just shifted its response criterion.

═══════════════════════════════════════════════════════════════════════════════
4. BETA (β) — RESPONSE BIAS
═══════════════════════════════════════════════════════════════════════════════

    β = pdf(Z(HR)) / pdf(Z(FAR))

    β = 1.0 → Unbiased. Equal willingness to say "yes" or "no."
    β > 1.0 → Conservative. The system requires stronger evidence to say "yes."
    β < 1.0 → Liberal. The system says "yes" too easily.

Why does β matter?
    Under adversarial attack, a CNN's β might shift dramatically — it might
    become "afraid" to classify certain categories (conservative) or start
    hallucinating categories everywhere (liberal). Humans typically maintain
    a more stable β because they have metacognitive awareness of uncertainty.

═══════════════════════════════════════════════════════════════════════════════
5. HISTORICAL CONTEXT: GREEN & SWETS (1966)
═══════════════════════════════════════════════════════════════════════════════

SDT was formalized in "Signal Detection Theory and Psychophysics" by David
Green and John Swets (1966). Originally developed for radar operators in
WWII — distinguishing enemy aircraft blips from noise on a radar screen.

The insight was revolutionary: a radar operator's performance depends on
TWO independent factors:
    1. How distinct the signal is from the noise (d')
    2. How cautious the operator is (β)

This framework transferred directly to psychophysics (can you hear a faint
tone?), medical diagnosis (can you spot the tumor on the X-ray?), and now
to machine learning (can the CNN recognize this cat through adversarial noise?).

Our project uses the EXACT same framework Green & Swets defined in 1966.
The "signal" is the true class identity; the "noise" is adversarial
perturbation. We measure d' for both CNN and human to compare sensitivity.

═══════════════════════════════════════════════════════════════════════════════
6. THE LAPLACE SMOOTHING FIX
═══════════════════════════════════════════════════════════════════════════════

If Hit Rate = 1.0 or False Alarm Rate = 0.0, Z() returns ±infinity.
This happens frequently with CNN models on clean data (near-perfect scores).
The standard fix (Macmillan & Creelman, 2005) is to add 0.5 to all cells:

    Adjusted HR = (Hits + 0.5) / (Hits + Misses + 1)
    Adjusted FAR = (FA + 0.5) / (FA + CR + 1)

This "pulls" extreme rates slightly toward 0.5, preventing infinity while
introducing negligible bias for large sample sizes.

═══════════════════════════════════════════════════════════════════════════════
7. WHY THE d' THRESHOLD GAP IS MORE PUBLISHABLE THAN ACCURACY
═══════════════════════════════════════════════════════════════════════════════

Saying "CNN accuracy drops to 15% while humans stay at 85%" is descriptive.
Saying "CNN d' falls below the 1.0 detection threshold at ε=0.05, while
human d' doesn't cross that threshold until ε=0.25" is ANALYTICAL.

The d' threshold gap:
    • Quantifies sensitivity loss on a bias-free scale
    • Is directly comparable across studies and datasets
    • Maps onto established psychophysical theory (decades of literature)
    • Lets you say: "The CNN has effectively zero perceptual sensitivity at
      a noise level where humans are still comfortable" — a statement grounded
      in formal measurement theory, not just a percentage comparison.

This is what distinguishes a "class project" from a "conference submission."

References:
    Green, D. M., & Swets, J. A. (1966). Signal detection theory and
        psychophysics. Wiley.
    Macmillan, N. A., & Creelman, C. D. (2005). Detection theory: A user's
        guide (2nd ed.). Lawrence Erlbaum Associates.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# =============================================================================
#  CORE SDT FUNCTIONS
# =============================================================================

def d_prime(hits, misses, false_alarms, correct_rejections):
    """
    Compute d-prime (perceptual sensitivity index).

    PLAIN LANGUAGE:
        d' measures how well a system can TRULY distinguish a target class from
        all other classes, completely independent of any response bias.

        Think of it like this: you're at a party and trying to hear your friend
        speak (signal) over the crowd noise (noise). d' measures how clearly you
        can actually hear them — separate from whether you're the type of person
        who says "what?" a lot (conservative) or pretends to hear everything
        (liberal).

    MATH:
        1. Compute Hit Rate = Hits / (Hits + Misses)
        2. Compute FA Rate  = FAs  / (FAs + Correct Rejections)
        3. Apply Laplace smoothing (add 0.5 to each cell) to avoid log(0)
        4. d' = Z(HR) - Z(FAR)  where Z = inverse normal CDF

    Parameters
    ----------
    hits : int
        Number of correct positive identifications.
        "It IS a cat and I correctly said cat."
    misses : int
        Number of missed targets.
        "It IS a cat but I failed to identify it."
    false_alarms : int
        Number of false positive identifications.
        "It is NOT a cat but I incorrectly said cat."
    correct_rejections : int
        Number of correct negative identifications.
        "It is NOT a cat and I correctly said it's not a cat."

    Returns
    -------
    float
        d-prime value. 0 = chance, 1 = threshold, 4.65 = near-perfect.
    """
    # Laplace smoothing (Macmillan & Creelman, 2005)
    # Prevents Z(0) = -inf and Z(1) = +inf
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    fa_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)

    # d' = Z(HR) - Z(FAR)
    dprime = norm.ppf(hit_rate) - norm.ppf(fa_rate)

    return float(dprime)


def beta(hit_rate, fa_rate):
    """
    Compute response bias (β).

    PLAIN LANGUAGE:
        β tells you whether the system is trigger-happy or cautious.
        A CNN with β < 1 is "hallucinating" — it says "cat" too readily.
        A CNN with β > 1 is "suppressing" — it requires overwhelming evidence.
        β = 1 means perfectly unbiased.

    MATH:
        β = φ(Z(FAR)) / φ(Z(HR))

        where φ is the standard normal PDF and Z is the inverse CDF.
        Note: some textbooks define β as φ(Z(HR))/φ(Z(FAR)). We use the
        Macmillan & Creelman convention where β > 1 = conservative.

    Parameters
    ----------
    hit_rate : float
        Proportion of hits (0 to 1, already Laplace-smoothed).
    fa_rate : float
        Proportion of false alarms (0 to 1, already Laplace-smoothed).

    Returns
    -------
    float
        β value. 1.0 = unbiased, >1 = conservative, <1 = liberal.
    """
    # Clamp to avoid infinities at exact 0 or 1
    hr_clamped = np.clip(hit_rate, 1e-5, 1 - 1e-5)
    far_clamped = np.clip(fa_rate, 1e-5, 1 - 1e-5)

    # β = φ(Z(FAR)) / φ(Z(HR))  [Macmillan & Creelman convention]
    z_hr = norm.ppf(hr_clamped)
    z_far = norm.ppf(far_clamped)

    beta_val = np.exp(-0.5 * (z_far ** 2 - z_hr ** 2))

    return float(beta_val)


def compute_sdt_from_responses(preds, labels, target_class):
    """
    Compute full SDT metrics for a single target class.

    PLAIN LANGUAGE:
        Given a set of predictions and true labels, this function treats ONE
        class as "signal" and ALL other classes as "noise", then computes
        the four SDT cells (hit, miss, FA, CR) and derives d' and β.

    HOW THE FOUR CELLS MAP TO CLASSIFICATION:
        Signal Present = true label IS the target class
        Signal Absent  = true label is NOT the target class
        "Yes" response = model predicted the target class
        "No" response  = model predicted something else

    Parameters
    ----------
    preds : array-like
        Predicted class labels (integer indices, shape [N]).
    labels : array-like
        True class labels (integer indices, shape [N]).
    target_class : int
        The class index to treat as "signal" (0-9 for CIFAR-10).

    Returns
    -------
    dict
        {
            'hits': int,
            'misses': int,
            'false_alarms': int,
            'correct_rejections': int,
            'hit_rate': float,      (Laplace-smoothed)
            'fa_rate': float,       (Laplace-smoothed)
            'd_prime': float,
            'beta': float,
        }
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # Boolean masks
    signal_present = (labels == target_class)     # True label IS the target
    signal_absent = ~signal_present               # True label is NOT the target
    said_yes = (preds == target_class)            # Model predicted the target
    said_no = ~said_yes                           # Model predicted something else

    # Four SDT cells
    hits = int(np.sum(signal_present & said_yes))
    misses = int(np.sum(signal_present & said_no))
    false_alarms = int(np.sum(signal_absent & said_yes))
    correct_rejections = int(np.sum(signal_absent & said_no))

    # Laplace-smoothed rates
    hr = (hits + 0.5) / (hits + misses + 1)
    far = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)

    return {
        'hits': hits,
        'misses': misses,
        'false_alarms': false_alarms,
        'correct_rejections': correct_rejections,
        'hit_rate': float(hr),
        'fa_rate': float(far),
        'd_prime': d_prime(hits, misses, false_alarms, correct_rejections),
        'beta': beta(hr, far),
    }


def compute_sdt_all_classes(preds, labels, num_classes=10):
    """
    Compute SDT metrics for EVERY class, returned as a DataFrame.

    PLAIN LANGUAGE:
        Runs compute_sdt_from_responses() 10 times — once treating each CIFAR-10
        class as "signal" — and stacks the results into a table.

        The resulting DataFrame is the SDT equivalent of the per-class accuracy
        matrix from Phase 4, but it separates sensitivity from bias.

    Parameters
    ----------
    preds : array-like
        Predicted class labels (shape [N]).
    labels : array-like
        True class labels (shape [N]).
    num_classes : int
        Number of classes (default 10 for CIFAR-10).

    Returns
    -------
    pd.DataFrame
        One row per class with columns:
        [class_idx, hits, misses, false_alarms, correct_rejections,
         hit_rate, fa_rate, d_prime, beta]
    """
    rows = []
    for c in range(num_classes):
        result = compute_sdt_from_responses(preds, labels, target_class=c)
        result['class_idx'] = c
        rows.append(result)

    df = pd.DataFrame(rows)
    # Reorder columns for readability
    cols = ['class_idx', 'hits', 'misses', 'false_alarms', 'correct_rejections',
            'hit_rate', 'fa_rate', 'd_prime', 'beta']
    return df[cols]
