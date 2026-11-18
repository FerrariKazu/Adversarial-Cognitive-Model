import numpy as np
import pytest

from rhan_bi.config import Config, default_config, load_config
from rhan_bi.evaluation.metrics import ece, risk_coverage, selective_at_coverage, wilson_ci
from rhan_bi.evaluation.stats import bootstrap_ci, holm, mcnemar


def test_ece_perfect_is_zero():
    probs = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]
    assert ece(probs, ["a", "b"]) < 1e-9


def test_risk_coverage_and_wilson():
    probs = [{"a": 0.9, "b": 0.1}, {"a": 0.6, "b": 0.4}, {"a": 0.2, "b": 0.8}]
    y = ["a", "a", "b"]
    curve = risk_coverage(probs, y, steps=10)
    assert len(curve) == 10
    sel = selective_at_coverage(probs, y, 2 / 3)
    assert sel["coverage"] == pytest.approx(2 / 3, abs=0.01)
    lo, hi = wilson_ci(2, 3)
    assert 0 <= lo <= hi <= 1


def test_bootstrap_and_tests():
    a = np.array([1, 0, 0, 1, 0])
    b = np.array([0, 0, 0, 1, 1])
    ci = bootstrap_ci(a, b, n_boot=200)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    mc = mcnemar([True, False, True], [True, True, False])
    assert 0 <= mc["p"] <= 1
    h = holm({"x": 0.01, "y": 0.04, "z": 0.20})
    assert h["x"]["p_holm"] <= h["y"]["p_holm"] <= h["z"]["p_holm"]


def test_config_unknown_key_and_hash_stability():
    with pytest.raises(ValueError):
        Config.from_dict({"run": {"nope": 1}})
    c1, c2 = default_config(), default_config()
    assert c1.hash() == c2.hash()


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("run:\n  n_train: 12\n  tag: test\n")
    cfg = load_config(p)
    assert cfg.run.n_train == 12 and cfg.run.tag == "test"
