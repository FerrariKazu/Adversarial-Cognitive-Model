"""Scientific metrics: sensitivity (d′), calibration, robustness curves."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def accuracy_to_dprime(accuracy: float, n_classes: int = 10) -> float:
    """d′ from overall accuracy under a uniform single-label decision model.

    HR = acc, FAR = (1 - acc) / (n_classes - 1).
    """
    acc = np.clip(accuracy / 100.0, 1e-6, 1 - 1e-6)
    far = np.clip((1.0 - acc) / max(n_classes - 1, 1), 1e-6, 1 - 1e-6)
    return float(norm.ppf(acc) - norm.ppf(far))


def macro_dprime(per_class_accuracy: list[float], n_classes: int = 10) -> float:
    dprimes = [accuracy_to_dprime(a, n_classes) for a in per_class_accuracy]
    return float(np.mean(dprimes)) if dprimes else float("nan")


def entropy_from_probs(probs: np.ndarray) -> float:
    """Shannon entropy (nats) of a probability vector."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def entropy_from_accuracy(accuracy: float, n_classes: int = 10) -> float:
    """Expected entropy implied by an accuracy level (uniformity proxy)."""
    acc = np.clip(accuracy / 100.0, 0.0, 1.0)
    p_win = acc + (1 - acc) / n_classes
    p_lose = (1 - acc) / n_classes
    return float(-(p_win * np.log(p_win) + (n_classes - 1) * p_lose * np.log(p_lose)))


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> float:
    """Expected Calibration Error over confidence bins."""
    conf, corr = np.asarray(confidences), np.asarray(correct, dtype=int)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, total = 0.0, 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        acc_bin = corr[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / conf.size) * abs(acc_bin - conf_bin)
        total += mask.sum()
    return float(ece)


def find_ethresh(epsilons: list[float], dprimes: list[float],
                 threshold: float = 1.0) -> float:
    """First epsilon where d′ crosses below the threshold (default 1.0)."""
    for eps, dp in zip(epsilons, dprimes):
        if dp < threshold:
            return float(eps)
    return float(epsilons[-1] if epsilons else 0.0)


def _trapz(y, x) -> float:
    """np.trapezoid with a numpy<2.0 fallback."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def robustness_curve(accuracies: list[float], epsilons: list[float],
                     n_classes: int = 10) -> dict:
    """Bundle accuracy + d′ curve + εthresh into one structure."""
    dprimes = [accuracy_to_dprime(a, n_classes) for a in accuracies]
    return {
        "epsilons": list(epsilons),
        "accuracy": list(accuracies),
        "dprime": dprimes,
        "ethresh": find_ethresh(epsilons, dprimes),
        "clean_acc": accuracies[0] if accuracies else float("nan"),
        "auc_robustness": _trapz(dprimes, epsilons) if len(epsilons) > 1 else 0.0,
    }


def summarize(clean_acc: float, robust_at: dict[float, float],
              n_classes: int = 10) -> dict:
    """Summary stats for a model: clean, d′, robust acc by eps, εthresh."""
    epsilons = sorted(robust_at)
    accs = [clean_acc] + [robust_at[e] for e in epsilons]
    curve = robustness_curve(accs, [0.0] + epsilons, n_classes)
    return {
        "clean_acc": clean_acc,
        "robust_at": robust_at,
        "dprime_clean": accuracy_to_dprime(clean_acc, n_classes),
        "ethresh": curve["ethresh"],
        "curve": curve,
    }
