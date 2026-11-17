"""Gate-2 v1.2 rules, train holdout, pure planning/sensor/metric functions, and 2-of-3 aggregation."""
import hashlib
import json
import pathlib

import pytest
import torch
import torch.nn.functional as F

from rhan_im import gate2 as g2
from rhan_im.layout import UnitLayout
from rhan_im.trunk_adapter import native_crop
from rhan_im.window_head import EvidenceCalibration, EvidenceHead

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def test_v1_2_pins_v1_1_erratum_02_and_train_holdout():
    v1_1 = _load("prereg/gate2_rules_v1_1.json")
    v1_2 = _load("prereg/gate2_rules_v1_2.json")
    hold = _load("prereg/gate2_train_holdout.json")
    assert v1_2["protocol_id"] == "rhan-im-gate2/1.2"
    assert v1_2["status"] == "FROZEN_RULES"
    assert v1_2["supersedes"]["sha256"] == hashlib.sha256((ROOT / "prereg/gate2_rules_v1_1.json").read_bytes()).hexdigest()
    assert v1_2["erratum_02"]["sha256"] == hashlib.sha256((ROOT / "docs/research/RHAN_IM_GATE2_ERRATUM_02.md").read_bytes()).hexdigest()
    assert v1_2["data_bindings"]["class_split"]["class_split_sha256"] == "289f2818f0eae4dd7b654627d4bdf197cc11d43923300c52df7b4c0bdecc1241"
    assert v1_2["data_bindings"]["class_split"]["file_sha256"] == "c2e05641bf10d4aaaa41d6dfa9dbc3571a5d3b95dc80f36dc6a4d7d86d481c25"
    assert v1_2["data_bindings"]["train_holdout"]["sha256"] == hashlib.sha256((ROOT / "prereg/gate2_train_holdout.json").read_bytes()).hexdigest()
    assert v1_2["data_bindings"]["train_holdout"]["head_training_images"] == 124_289
    assert hold["hashes"]["holdout_assignment_sha256"] == "f84663e9e0c48d9132e96d4043ce496fa53a0562134b6ec96c578fad9f6af459"
    assert {k: hold["counts"][k] for k in ("train_files", "target_holdout_images", "background_holdout_images",
                                         "holdout_images", "head_training_images")} == {
        "train_files": 126_689, "target_holdout_images": 800, "background_holdout_images": 1_600,
        "holdout_images": 2_400, "head_training_images": 124_289}
    assert v1_1["protocol_id"] == "rhan-im-gate2/1.1"
    assert v1_2["code_locking"]["status"] == "DEFERRED_TO_CODE_LOCK_STEP"
    assert set(v1_2["code_locking"]["target_files"]) == set(g2.GATE2_CODE_FILES) | {"scripts/rhan_im/gate2_aggregate.py"}


def test_train_holdout_reader_checks_assignment():
    hold = g2.read_train_holdout(ROOT / "prereg/gate2_train_holdout.json")
    assert len(hold["target_holdout"]) == 50 and len(hold["background_holdout"]) == 50
    assert hold["counts"]["target_holdout_images"] == 800 and hold["counts"]["background_holdout_images"] == 1_600
    assert hold["holdout_assignment_sha256"] == "f84663e9e0c48d9132e96d4043ce496fa53a0562134b6ec96c578fad9f6af459"


def test_build_plan_is_deterministic_balanced_and_no_reuse():
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=4)
    target = {f"t{i:02d}": [f"{j:03d}.png" for j in range(12)] for i in range(5)}
    background = {f"b{i:02d}": [f"{j:03d}.png" for j in range(20)] for i in range(5)}
    p = g2.build_plan(target, background, 30, 30, [2, 4, 6], lay, 7, "cal")
    assert int((p["label"] >= 0).sum()) == 30 and int((p["label"] < 0).sum()) == 30
    assert len(set((c, f) for c, f in zip(p["target_class"], p["target_file"]) if c is not None)) == 30
    assert len(set((c, f) for c, f in zip(p["background_class"], p["background_file"]))) == 60
    assert set(p["size_units"][p["label"] >= 0].tolist()) == {2, 4, 6}
    assert p["plan_sha256"] == g2.plan_hash(p)
    q = g2.build_plan(target, background, 30, 30, [2, 4, 6], lay, 7, "cal")
    assert q["plan_sha256"] == p["plan_sha256"]
    s = g2.slice_plan(p, 10, 20)
    assert int(s["label"].shape[0]) == 10 and s["slice"] == [10, 20]


def test_build_plan_refuses_to_reuse_an_image():
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)
    with pytest.raises(ValueError, match="not enough files"):
        g2.build_plan({"t": ["a.png"]}, {"b": ["x.png", "y.png"]}, 2, 2, [2], lay, 0, "x")


def test_look_sequence_gist_first_and_native_draws_are_seeded():
    a = g2.look_sequence(5, 9, seed=3)
    b = g2.look_sequence(5, 9, seed=3)
    assert torch.equal(a, b)
    assert torch.all(a[:, 0] == 9)
    assert bool(((a[:, 1:] >= 0) & (a[:, 1:] < 9)).all())
    c = g2.look_sequence(5, 9, seed=3, offset=2)
    assert not torch.equal(a, c)


def test_true_outcome_maps_absent_to_zero_and_class_to_plus_one():
    assert g2.true_outcome(torch.tensor([-1, 0, 3])).tolist() == [0, 1, 4]


class PatchMeanTrunk:
    def __init__(self, patch):
        self.patch = patch

    def __call__(self, x):
        tokens = F.avg_pool2d(x.mean(1, keepdim=True), self.patch).flatten(2).transpose(1, 2)
        return tokens.mean(1), tokens


def test_gate2_sensor_is_device_safe_and_normalises_inputs():
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)
    head = EvidenceHead(1, 3)
    with torch.no_grad():
        head.proj.weight.zero_()
        head.proj.bias.zero_()
        head.proj.weight[0, 0] = 1.0
    calib = EvidenceCalibration(3, lay.tok)
    calib.d_tok.fill_(2.0)
    calib.d_gist.fill_(0.5)
    sensor = g2.Gate2Sensor(PatchMeanTrunk(lay.patch), head, lay, calib, lay.candidates(2), "cpu")
    canvas = torch.zeros(2, 3, 112, 112)
    z, d = sensor.look(canvas, torch.tensor([0, sensor.n_native]))
    assert z.shape == (2, lay.V, 3) and d.shape == (2, lay.V)
    assert sensor.vis().shape == (sensor.n_candidates, lay.V)
    assert torch.isfinite(z).all() and torch.isfinite(d).all()
    assert z[0, :lay.n_fine].abs().sum() > 0
    assert z[1, lay.n_fine:].abs().sum() > 0
    assert torch.all(d[1, :lay.n_fine] == 0) and torch.all(d[1, lay.n_fine:] == 0.5)


def test_fit_sensor_from_scores_recovers_positive_d_and_reports_contract():
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)
    T, C, n = lay.tok * lay.tok, 3, 120
    gen = torch.Generator().manual_seed(0)
    fit_native = torch.randn(n, T, C, generator=gen)
    fit_gist = torch.randn(n, T, C, generator=gen)
    fit_cover_native = torch.zeros(n, T, dtype=torch.bool)
    fit_cover_gist = torch.zeros(n, T, dtype=torch.bool)
    fit_label = torch.full((n,), -1, dtype=torch.long)
    labels = torch.randint(0, C, (2 * n,), generator=gen)
    labels[n:] = -1
    d_native = torch.randn(2 * n, T, C, generator=gen)
    d_gist = torch.randn(2 * n, T, C, generator=gen)
    for i in range(n):
        d_native[i, :, labels[i]] += 2.0
        d_gist[i, :, labels[i]] += 1.0
    cover = torch.ones(2 * n, T, dtype=torch.bool)
    cover[n:] = False
    calib, report = g2.fit_sensor_from_scores(
        fit_native, fit_gist, fit_cover_native, fit_cover_gist, fit_label,
        d_native, d_gist, cover, cover, labels,
        torch.randn(n, T, C, generator=gen), torch.randn(n, T, C, generator=gen), lay,
        op_size=2, min_pos=30,
    )
    assert torch.allclose(calib.d_tok, torch.full_like(calib.d_tok, 2.0), atol=0.2)
    assert abs(float(calib.d_gist) - 1.0) < 0.2
    assert report["unit_cap"] is None and report["combination_mode"] in ("sum", "best")
    assert report["vif_gist_raw"] is None
    assert report["kappa"] > 0 and report["vif_tok_used"] >= 1.0


def test_posterior_traces_shapes_and_prior():
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)
    C, n, K = 3, 4, 2
    z = 0.1 * torch.randn(n, K, lay.V, C)
    d = torch.ones(n, K, lay.V)
    labels = torch.tensor([0, -1, 1, -1])
    eval_size = torch.tensor([2, 2, 2, 2])
    tr = g2.posterior_traces(z, d, eval_size, labels, lay, "sum", kappa=3.0, k_looks=K)
    assert tr["conf"].shape == (n, K + 1) and tr["correct"].shape == (n, K + 1)
    assert tr["probs"].shape == (n, 1 + C)
    assert torch.allclose(tr["probs"].sum(1), torch.ones(n), atol=1e-5)
    assert torch.allclose(tr["conf"][:, 0], torch.full((n,), 0.5), atol=1e-6)


def test_ece_nll_brier_and_e2_metrics():
    conf = torch.tensor([0.0, 0.0, 1.0, 1.0])
    correct = torch.tensor([False, False, True, True])
    assert g2.ece_15(conf, correct) == pytest.approx(0.0)
    probs = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    nll, brier = g2.nll_brier(probs, torch.tensor([-1, -1, 0, 0]))
    assert nll == pytest.approx(0.0) and brier == pytest.approx(0.0)
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)
    tr = g2.posterior_traces(0.1 * torch.randn(4, 2, lay.V, 3), torch.ones(4, 2, lay.V),
                             torch.tensor([2, 2, 2, 2]), torch.tensor([0, -1, 1, -1]), lay, "sum", 3.0, k_looks=2)
    e2 = g2.e2_metrics(tr, torch.tensor([0, -1, 1, -1]), torch.tensor([2, 2, 2, 2]), k_looks=2)
    assert set(e2["ece_by_size"]) == {"2"} and e2["n_scenes"] == 4
    assert e2["status"] in ("PASS", "INCONCLUSIVE", "FAIL")


def test_e5_metrics_deterministic_perfect_policy_passes():
    cal_risk = torch.zeros(40, 9)
    cal_correct = torch.ones(40, 9, dtype=torch.bool)
    test_risk = torch.zeros(60, 9)
    test_correct = torch.ones(60, 9, dtype=torch.bool)
    out = g2.e5_metrics({"risk": cal_risk, "correct": cal_correct}, {"risk": test_risk, "correct": test_correct},
                        n_cal=30, bootstrap=20, lambdas=[0.0, 0.1, 0.2, 0.5], seed=0)
    assert out["primary"] == "selective"
    assert out["selective"]["pass_validity"] and out["selective"]["pass_efficiency"]
    assert out["selective"]["mean_test_risk"] == 0.0 and out["selective"]["mean_efficiency"] > 0.25


def test_aggregate_runs_uses_the_2_of_3_rule():
    def run(seed, e2, validity, efficiency):
        return {"seed": seed, "e2": {"status": e2},
                "e5": {"selective": {"pass_validity": validity, "pass_efficiency": efficiency}}}

    runs = [run(17, "PASS", True, True), run(42, "PASS", True, True), run(99, "FAIL", False, False)]
    assert g2.aggregate_runs(runs)["status"] == "PASS"
    runs[1]["e2"]["status"] = "INCONCLUSIVE"
    out = g2.aggregate_runs(runs)
    assert out["e2_status"] == "INCONCLUSIVE" and out["status"] == "INCONCLUSIVE"
    runs[0]["e2"]["status"] = "FAIL"
    assert g2.aggregate_runs(runs)["e2_status"] == "FAIL"
    assert g2.aggregate_runs([run(1, "PASS", True, True)])["status"] == "INVALID"
