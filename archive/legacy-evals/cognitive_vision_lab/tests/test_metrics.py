"""Unit tests for cognitive_vision_lab.backend.metrics."""
import math

import numpy as np
import pytest

from cognitive_vision_lab.backend.metrics import (
    accuracy_to_dprime,
    entropy_from_accuracy,
    entropy_from_probs,
    expected_calibration_error,
    find_ethresh,
    robustness_curve,
    summarize,
)


class TestAccuracyToDprime:
    def test_chance_level(self):
        # Chance accuracy on 10 classes -> d' ~ 0
        assert abs(accuracy_to_dprime(10.0, 10)) < 0.05

    def test_perfect(self):
        # 100% accuracy -> very large d'
        assert accuracy_to_dprime(100.0, 10) > 4.0

    def test_monotonic(self):
        a, b = accuracy_to_dprime(40.0, 10), accuracy_to_dprime(80.0, 10)
        assert b > a

    def test_clips_at_edges(self):
        assert math.isfinite(accuracy_to_dprime(0.0, 10))
        assert math.isfinite(accuracy_to_dprime(100.0, 10))


class TestEntropy:
    def test_uniform_max(self):
        p = np.ones(10) / 10.0
        assert entropy_from_probs(p) == pytest.approx(math.log(10), abs=1e-9)

    def test_certain_min(self):
        p = np.zeros(10)
        p[0] = 1.0
        assert entropy_from_probs(p) == pytest.approx(0.0, abs=1e-9)

    def test_accuracy_entropy_bounded(self):
        e = entropy_from_accuracy(50.0, 10)
        assert 0.0 <= e <= math.log(10)


class TestECE:
    def test_perfectly_calibrated(self):
        # 10 samples all at confidence 0.5, exactly 5 correct -> acc == conf
        conf = np.full(10, 0.5)
        corr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        assert expected_calibration_error(conf, corr) == pytest.approx(0.0, abs=1e-6)

    def test_terrible_calibration(self):
        conf = np.array([1.0, 1.0, 1.0, 1.0])
        corr = np.array([0, 0, 0, 0])
        assert expected_calibration_error(conf, corr) > 0.9

    def test_empty(self):
        assert math.isnan(expected_calibration_error(np.array([]), np.array([])))


class TestFindEthresh:
    def test_crossing(self):
        eps = [0.0, 0.01, 0.02, 0.03]
        dprimes = [3.0, 2.0, 0.8, 0.2]
        assert find_ethresh(eps, dprimes) == pytest.approx(0.02)

    def test_never_crosses(self):
        eps = [0.0, 0.1, 0.2]
        dprimes = [4.0, 3.5, 3.0]
        assert find_ethresh(eps, dprimes) == pytest.approx(0.2)

    def test_crosses_at_zero(self):
        eps = [0.0, 0.05]
        dprimes = [0.5, 0.1]
        assert find_ethresh(eps, dprimes) == pytest.approx(0.0)

    def test_empty(self):
        assert find_ethresh([], []) == 0.0


class TestRobustnessCurve:
    def test_shape(self):
        accs = [80.0, 60.0, 40.0]
        eps = [0.0, 0.03, 0.06]
        c = robustness_curve(accs, eps, n_classes=10)
        assert c["epsilons"] == eps
        assert c["accuracy"] == accs
        assert len(c["dprime"]) == 3
        assert c["clean_acc"] == 80.0
        assert c["auc_robustness"] > 0.0

    def test_ethresh_in_curve(self):
        accs = [80.0, 20.0, 5.0]
        c = robustness_curve(accs, [0.0, 0.03, 0.06], n_classes=10)
        assert c["ethresh"] == pytest.approx(0.03)


class TestSummarize:
    def test_bundle(self):
        s = summarize(clean_acc=80.0, robust_at={0.03: 50.0, 0.06: 20.0}, n_classes=10)
        assert s["clean_acc"] == 80.0
        assert s["robust_at"] == {0.03: 50.0, 0.06: 20.0}
        assert "curve" in s
        assert s["dprime_clean"] > 2.0
