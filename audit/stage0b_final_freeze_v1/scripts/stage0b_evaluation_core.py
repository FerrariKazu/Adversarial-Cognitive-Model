#!/usr/bin/env python3
"""Frozen Stage 0B metric and paired-bootstrap implementation.

The module is deterministic and operates on supplied arrays.  It does not load
checkpoints or data, so importing or unit-testing it cannot fit a temperature or
inspect a Stage 0B outcome.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stage0b_data_gate_v1" / "scripts"))
import calibration_core as cc  # noqa: E402

PRIMARY_BINS = 15
N_BOOT = 10_000
BOOTSTRAP_SEED = 20261121
ECE_INCREASE_TOLERANCE = 0.005
PROPER_SCORE_INCREASE_TOLERANCE = 0.005


def _as_pair(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = cc._validate_logits(logits)  # noqa: SLF001 - shared frozen validator
    labels = cc._validate_labels(labels, array.shape[0], array.shape[1])  # noqa: SLF001
    return array, labels


def score_logits(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> dict:
    """Compute all frozen metrics for one logit array and label vector."""
    array, labels = _as_pair(logits, labels)
    probabilities = cc.softmax(array, temperature)
    top1 = cc.topk_indices(array, 1)[:, 0]
    top5 = cc.topk_indices(array, 5)
    correct = (top1 == labels).astype(np.float64)
    top5_correct = np.any(top5 == labels[:, None], axis=1).astype(np.float64)
    confidence = probabilities.max(axis=1)
    adaptive, adaptive_counts = cc.adaptive_ece(confidence, correct, n_bins=PRIMARY_BINS)
    per_class_accuracy = {}
    for class_index in range(array.shape[1]):
        mask = labels == class_index
        if mask.any():
            per_class_accuracy[str(class_index)] = float(correct[mask].mean())
    return {
        "n": int(array.shape[0]),
        "temperature": float(temperature),
        "top1": float(correct.mean()),
        "top5": float(top5_correct.mean()),
        "macro_top1": float(np.mean(list(per_class_accuracy.values()))) if per_class_accuracy else None,
        "correct_count": int(correct.sum()),
        "nll": cc.nll(array, labels, temperature),
        "brier": cc.brier_score(probabilities, labels),
        "ece_15": cc.ece(confidence, correct, n_bins=PRIMARY_BINS),
        "adaptive_ece_15": adaptive,
        "adaptive_ece_bin_counts": adaptive_counts,
        "confidence_mean": float(confidence.mean()),
        "per_class_accuracy": per_class_accuracy,
        "reliability_15": cc.reliability_bins(confidence, correct, n_bins=PRIMARY_BINS),
        "selective": cc.coverage_risk(confidence, correct),
    }


def per_class_confidence_gap(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> dict:
    """Descriptive per-class confidence-accuracy gaps; n=5 in the primary set."""
    array, labels = _as_pair(logits, labels)
    probabilities = cc.softmax(array, temperature)
    confidence = probabilities.max(axis=1)
    top1 = cc.topk_indices(array, 1)[:, 0]
    correct = (top1 == labels).astype(np.float64)
    rows = {}
    for class_index in range(array.shape[1]):
        mask = labels == class_index
        count = int(mask.sum())
        if count:
            rows[str(class_index)] = {
                "n": count,
                "accuracy": float(correct[mask].mean()),
                "mean_confidence": float(confidence[mask].mean()),
                "confidence_accuracy_gap": float(confidence[mask].mean() - correct[mask].mean()),
                "ece_15": cc.ece(confidence[mask], correct[mask], n_bins=PRIMARY_BINS),
            }
    return rows


def paired_delta(
    raw_logits: np.ndarray,
    calibrated_logits: np.ndarray,
    raw: Mapping,
    calibrated: Mapping,
) -> dict:
    """Per-seed differences, with positive ECE reduction meaning improvement."""
    return {
        "ece_15_delta": float(raw["ece_15"] - calibrated["ece_15"]),
        "nll_delta": float(raw["nll"] - calibrated["nll"]),
        "brier_delta": float(raw["brier"] - calibrated["brier"]),
        "top1_mismatches": cc.topk_mismatches(raw_logits, calibrated_logits, 1),
        "top5_mismatches": cc.topk_mismatches(raw_logits, calibrated_logits, 5),
        "top1_agreement": cc.topk_mismatches(raw_logits, calibrated_logits, 1) == 0,
        "top5_agreement": cc.topk_mismatches(raw_logits, calibrated_logits, 5) == 0,
    }


def mean_ece_delta(raw_scores: list[dict], calibrated_scores: list[dict]) -> float:
    if len(raw_scores) != len(calibrated_scores) or not raw_scores:
        raise ValueError("matching non-empty score lists required")
    deltas = [raw["ece_15"] - calibrated["ece_15"] for raw, calibrated in zip(raw_scores, calibrated_scores)]
    return float(np.mean(deltas))


def stratified_paired_bootstrap(
    raw_logits_by_seed: list[np.ndarray],
    calibrated_logits_by_seed: list[np.ndarray],
    labels: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Class-stratified paired bootstrap of the three-seed mean ECE reduction.

    Every resample preserves five images per class.  The same image indices are
    used for raw, calibrated, and all three seeds.
    """
    if len(raw_logits_by_seed) != 3 or len(calibrated_logits_by_seed) != 3:
        raise ValueError("exactly three seeds are required")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty 1-D array")
    for raw, calibrated in zip(raw_logits_by_seed, calibrated_logits_by_seed):
        if raw.shape != calibrated.shape or raw.shape[0] != labels.size:
            raise ValueError("raw/calibrated/label shape mismatch")
    indices = cc.stratified_bootstrap_indices(labels, n_boot=n_boot, seed=seed)
    values = np.empty(n_boot, dtype=np.float64)
    for boot in range(n_boot):
        take = indices[boot]
        y = labels[take]
        deltas = []
        for raw, calibrated in zip(raw_logits_by_seed, calibrated_logits_by_seed):
            raw_score = score_logits(raw[take], y, 1.0)
            calibrated_score = score_logits(calibrated[take], y, 1.0)
            deltas.append(raw_score["ece_15"] - calibrated_score["ece_15"])
        values[boot] = float(np.mean(deltas))
    point = mean_ece_delta(
        [score_logits(raw, labels, 1.0) for raw in raw_logits_by_seed],
        [score_logits(calibrated, labels, 1.0) for calibrated in calibrated_logits_by_seed],
    )
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "point_mean_ece_reduction": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "class_stratified": True,
        "paired_across_seeds": True,
        "resamples_preserve_per_class_count": True,
    }


def acceptance_decision(
    deltas: list[dict],
    bootstrap: dict,
    *,
    all_fits_converged: bool,
    any_fit_at_boundary: bool,
) -> dict:
    """Apply only the frozen decision rule; no post-hoc thresholds."""
    ece_increases = [-float(row["ece_15_delta"]) for row in deltas]
    checks = {
        "three_seeds_present": len(deltas) == 3,
        "mean_ece_reduction_positive": float(bootstrap["point_mean_ece_reduction"]) > 0,
        "bootstrap_ci_excludes_zero": float(bootstrap["ci95_low"]) > 0,
        "no_seed_ece_increase_gt_0.005": max(ece_increases, default=float("inf")) <= ECE_INCREASE_TOLERANCE,
        "top1_identical_all_seeds": all(row["top1_agreement"] for row in deltas),
        "top5_identical_all_seeds": all(row["top5_agreement"] for row in deltas),
        "no_seed_nll_increase_gt_0.005": max((max(0.0, -float(row["nll_delta"])) for row in deltas), default=float("inf")) <= PROPER_SCORE_INCREASE_TOLERANCE,
        "no_seed_brier_increase_gt_0.005": max((max(0.0, -float(row["brier_delta"])) for row in deltas), default=float("inf")) <= PROPER_SCORE_INCREASE_TOLERANCE,
        "all_fits_converged": bool(all_fits_converged),
        "no_fit_at_boundary": not bool(any_fit_at_boundary),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "interpretation": (
            "accepted calibration improvement under the frozen rule"
            if all(checks.values())
            else "calibration improvement not established; inspect failed checks without changing thresholds"
        ),
    }
