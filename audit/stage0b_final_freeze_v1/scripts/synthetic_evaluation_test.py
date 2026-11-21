#!/usr/bin/env python3
"""Synthetic-only validation of the frozen Stage 0B evaluation core."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0b_evaluation_core as ec  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "work" / "synthetic_evaluation_test.json"


def main() -> int:
    tests = []

    def record(name, function):
        try:
            tests.append({"name": name, "passed": True, "detail": function()})
        except Exception as exc:  # noqa: BLE001
            tests.append({"name": name, "passed": False, "detail": f"{type(exc).__name__}: {exc}"})

    rng = np.random.default_rng(20261121)
    n_classes = 100
    per_class = 5
    labels = np.repeat(np.arange(n_classes), per_class).astype(np.int64)
    raw_by_seed = []
    calibrated_by_seed = []
    for seed in (17, 42, 99):
        local = np.random.default_rng(seed)
        logits = local.normal(size=(labels.size, n_classes)) * 1.6
        logits[np.arange(labels.size), labels] += 2.0
        raw_by_seed.append(logits.astype(np.float32))
        calibrated_by_seed.append((logits / 1.7).astype(np.float32))

    raw_scores = [ec.score_logits(logits, labels, 1.0) for logits in raw_by_seed]
    calibrated_scores = [ec.score_logits(logits, labels, 1.0) for logits in calibrated_by_seed]
    deltas = [
        ec.paired_delta(raw, calibrated, raw_score, calibrated_score)
        for raw, calibrated, raw_score, calibrated_score in zip(
            raw_by_seed, calibrated_by_seed, raw_scores, calibrated_scores
        )
    ]

    def test_metric_shapes():
        for row in raw_scores + calibrated_scores:
            if row["n"] != labels.size:
                raise AssertionError("wrong n")
            if row["reliability_15"].__len__() != 15:
                raise AssertionError("wrong reliability bins")
            if sum(row["adaptive_ece_bin_counts"]) != labels.size:
                raise AssertionError("adaptive bin counts do not sum to n")
            if not 0 <= row["ece_15"] <= 1:
                raise AssertionError("ECE outside [0,1]")
        return "three seeds score cleanly with 15 reliability bins"

    def test_accuracy_invariance():
        for row in deltas:
            if row["top1_mismatches"] != 0 or row["top5_mismatches"] != 0:
                raise AssertionError(f"top-k changed: {row}")
        return "Top-1 and Top-5 mismatch counts are zero for all seeds"

    def test_ece_improves_synthetic():
        values = [row["ece_15_delta"] for row in deltas]
        if not all(value > 0 for value in values):
            raise AssertionError(f"non-positive synthetic ECE delta: {values}")
        return f"per-seed ECE reductions: {[round(v, 6) for v in values]}"

    def test_bootstrap_reproducible():
        first = ec.stratified_paired_bootstrap(raw_by_seed, calibrated_by_seed, labels, n_boot=40, seed=7)
        second = ec.stratified_paired_bootstrap(raw_by_seed, calibrated_by_seed, labels, n_boot=40, seed=7)
        if first != second:
            raise AssertionError("same-seed bootstrap differs")
        if first["ci95_low"] > first["ci95_high"]:
            raise AssertionError("reversed interval")
        return f"point={first['point_mean_ece_reduction']:.6f} CI=[{first['ci95_low']:.6f},{first['ci95_high']:.6f}]"

    def test_class_stratification():
        indices = __import__("calibration_core").stratified_bootstrap_indices(labels, n_boot=25, seed=9)
        for row in indices:
            counts = np.bincount(labels[row], minlength=n_classes)
            if not np.all(counts == per_class):
                raise AssertionError("class counts changed")
        return "every synthetic resample preserves five images per class"

    def test_decision_rule():
        bootstrap = {
            "point_mean_ece_reduction": float(np.mean([row["ece_15_delta"] for row in deltas])),
            "ci95_low": min(row["ece_15_delta"] for row in deltas),
            "ci95_high": max(row["ece_15_delta"] for row in deltas),
        }
        decision = ec.acceptance_decision(deltas, bootstrap, all_fits_converged=True, any_fit_at_boundary=False)
        if not decision["accepted"]:
            raise AssertionError(f"expected acceptance: {decision}")
        blocked = ec.acceptance_decision(deltas, bootstrap, all_fits_converged=True, any_fit_at_boundary=True)
        if blocked["accepted"]:
            raise AssertionError("boundary fit was accepted")
        return "all checks pass; a boundary fit is rejected"

    def test_per_class_descriptive():
        rows = ec.per_class_confidence_gap(raw_by_seed[0], labels, 1.0)
        if len(rows) != n_classes or any(row["n"] != per_class for row in rows.values()):
            raise AssertionError("per-class rows/counts wrong")
        return "100 descriptive per-class rows with n=5 each"

    record("metric shapes and ranges", test_metric_shapes)
    record("accuracy invariance", test_accuracy_invariance)
    record("synthetic ECE improvement", test_ece_improves_synthetic)
    record("bootstrap reproducibility", test_bootstrap_reproducible)
    record("class-stratified resampling", test_class_stratification)
    record("acceptance rule including boundary rejection", test_decision_rule)
    record("per-class descriptive rows", test_per_class_descriptive)

    passed = sum(bool(row["passed"]) for row in tests)
    result = {
        "schema": "rhan-stage0b-synthetic-evaluation-test/v1",
        "status": "PASS" if passed == len(tests) else "FAIL",
        "tests_passed": passed,
        "tests_total": len(tests),
        "real_model_predictions_used": False,
        "real_temperature_fitted": False,
        "tests": tests,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['status']} tests={passed}/{len(tests)}")
    print(f"wrote {OUT}")
    if result["status"] != "PASS":
        for row in tests:
            if not row["passed"]:
                print(f"FAIL: {row['name']}: {row['detail']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
