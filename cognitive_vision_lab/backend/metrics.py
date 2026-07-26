import numpy as np
from scipy.stats import norm


def accuracy_to_dprime(accuracy: float, n_classes: int = 10) -> float:
    hit_rate = accuracy / 100.0
    hit_rate = np.clip(hit_rate, 1e-6, 1 - 1e-6)
    false_alarm_rate = (1.0 - hit_rate) / (n_classes - 1)
    false_alarm_rate = np.clip(false_alarm_rate, 1e-6, 1 - 1e-6)
    return norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)


def macro_dprime(per_class_accuracy: list[float], n_classes: int = 10) -> float:
    dprimes = [accuracy_to_dprime(acc, n_classes) for acc in per_class_accuracy]
    return float(np.mean(dprimes))


def pooled_dprime(overall_accuracy: float, n_classes: int = 10) -> float:
    return accuracy_to_dprime(overall_accuracy, n_classes)


def compute_dprime_curve(
    accuracy_at_eps: list[float],
    epsilons: list[float],
    n_classes: int = 10,
):
    macro = []
    pooled = []
    for acc in accuracy_at_eps:
        macro.append(accuracy_to_dprime(acc, n_classes))
        pooled.append(accuracy_to_dprime(acc, n_classes))

    return macro, pooled


def find_ethresh(
    epsilons: list[float],
    dprimes: list[float],
    threshold: float = 1.0,
) -> float:
    for eps, dp in zip(epsilons, dprimes):
        if dp < threshold:
            return eps
    return epsilons[-1]
