"""
Agent I contract test — the mandatory structural-consistency assertion,
exercised against EVERY report function in evaluation/, plus the ported
eval_rhan conventions (seed floor, norm-space PGD, provenance) and the
data-infra validations.

Everything runs on synthetic data — these are HARNESS tests, never
results (the eval_rhan self-test convention).
"""
import json
import os

import pandas as pd
import pytest
import torch
import torch.nn as nn

from evaluation import clean_and_robust as car
from evaluation import compactness_report as comp
from evaluation import geirhos_loader as geo
from evaluation import imagenet_c_loader as imc
from evaluation import imagenet100_loader as im1
from evaluation import shape_texture_bias as stb
from noesis_vision.core.consistency_assert import groupby_summary

N_SEEDS = list(range(41, 46))  # the 5-seed floor, exactly


# ── convention 2: the seed floor ─────────────────────────────────────────────

def test_seed_floor_enforced():
    with pytest.raises(car.ProtocolError, match=">= 5 seeds"):
        car.enforce_seed_protocol([41, 42, 43, 44])
    assert car.enforce_seed_protocol(N_SEEDS) == N_SEEDS
    # The escape hatch is explicit, never silent.
    assert car.enforce_seed_protocol([42], allow_quick=True) == [42]


def test_seed_floor_applies_to_run(tmp_path):
    tiny = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))

    def factory(seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(16, 3, 8, 8, generator=g)
        y = torch.randint(0, 4, (16,), generator=g)
        return [(x, y)]

    with pytest.raises(car.ProtocolError, match=">= 5 seeds"):
        car.run_clean_and_robust(
            tiny, factory, seeds=[41, 42], eps_list=[0.05], n_samples=8,
            ckpt_path=None, out_dir=str(tmp_path / "out"), allow_quick=False)


# ── convention 1: norm-space PGD, per-channel eps ────────────────────────────

def test_pgd_perturbation_stays_in_eps_ball():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))
    x = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 4, (4,))
    eps = 0.05
    x_adv = car.generic_pgd(model, x, y, eps=eps, steps=3)
    delta = (x_adv - x).abs()
    assert float(delta.max()) <= eps + 1e-5, (
        "per-channel eps must bound the perturbation (norm-space, "
        "elementwise)")
    assert torch.isfinite(x_adv).all()


def test_pgd_actually_attacks():
    """The attack must be able to flip predictions on a weak model —
    a no-op PGD would silently produce fake robustness numbers."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))
    x = torch.randn(32, 3, 8, 8)
    y = torch.randint(0, 4, (32,))
    with torch.no_grad():
        clean = (model(x).argmax(1) == y).float().mean()
    x_adv = car.generic_pgd(model, x, y, eps=0.2, steps=5)  # OUTSIDE no_grad:
    # PGD needs the attack graph (the harness convention — eval wraps the
    # MODEL in no_grad only for the final accuracy read).
    with torch.no_grad():
        adv = (model(x_adv).argmax(1) == y).float().mean()
    assert adv < clean


# ── the mandatory assertion: every report function ──────────────────────────

def _write_per_seed_csv(path):
    rows = []
    for seed in N_SEEDS:
        for eps, acc in ((0.0, 50.0), (0.05, 40.0)):
            rows.append({"ckpt_label": "m", "seed": seed,
                         "eps_pixel": eps, "acc_pct": acc})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_summary_table_write_asserts_then_writes(tmp_path):
    csv = _write_per_seed_csv(str(tmp_path / "epsilon_sweep_per_seed.csv"))
    df = pd.read_csv(csv)
    summary = groupby_summary(df)
    out = str(tmp_path / "summary_table.csv")
    car.write_summary_table(summary, csv, out)          # honest -> written
    assert os.path.exists(out)
    # Corrupt ONE cell -> the assertion refuses, nothing written.
    corrupt = summary.copy()
    corrupt.loc[0, "acc_pct_mean"] += 5.0
    out2 = str(tmp_path / "summary_table2.csv")
    with pytest.raises(AssertionError, match="STRUCTURAL CONSISTENCY FAILURE"):
        car.write_summary_table(corrupt, csv, out2)
    assert not os.path.exists(out2)


def test_run_end_to_end_self_test_writes_asserted_outputs(tmp_path):
    """The full harness on synthetic loaders: per-seed CSV, asserted
    summary, provenance with norm-space recorded."""
    torch.manual_seed(0)
    tiny = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4))

    def factory(seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(16, 3, 8, 8, generator=g)
        y = torch.randint(0, 4, (16,), generator=g)
        return [(x, y)]

    ckpt = tmp_path / "m_best.pth"
    torch.save({"model": tiny.state_dict()}, ckpt)
    res = car.run_clean_and_robust(
        tiny, factory, seeds=N_SEEDS, eps_list=[0.0, 0.05], n_samples=8,
        ckpt_path=str(ckpt), out_dir=str(tmp_path / "out"),
        pgd_steps=2, ckpt_label="tiny")
    prov = res["provenance"]
    assert prov["eps_space"] == "norm"                  # convention 1 recorded
    assert prov["seeds"] == N_SEEDS
    assert prov["ckpt_sha256"] and len(prov["ckpt_sha256"]) == 64
    # The written summary reproduces a fresh groupby of its own CSV.
    summary = pd.read_csv(res["summary_csv"])
    df = pd.read_csv(res["per_seed_csv"])
    ref = groupby_summary(df)
    merged = summary.merge(ref, on=["ckpt_label", "eps_pixel"])
    assert len(merged) == len(summary) > 0
    assert (merged.acc_pct_mean_x - merged.acc_pct_mean_y).abs().max() < 1e-6


# ── shape/texture report: derive-twice discipline + verified baseline ───────

def test_geirhos_baseline_is_verified_not_invented():
    """The STOP-condition artifact: the baseline travels WITH provenance,
    always."""
    assert geo.HUMAN_SHAPE_DECISION_PCT == 95.9
    p = geo.HUMAN_BASELINE_PROVENANCE
    assert "1811.12231" in p["citation"]
    assert any("arxiv.org" in u for u in p["urls"])
    assert p["verified_date"] == "2026-09-24"
    row = geo.human_baseline_row()
    assert row["shape_decision_pct"] == 95.9
    assert row["baseline_citation"] == p["citation"]


def _write_decisions_csv(path, with_keys=True, n=40):
    import random
    random.seed(0)
    rows = []
    for i in range(n):
        shape, texture = f"cat{i % 5}", f"cat{(i + 2) % 5}"
        responded = shape if i % 4 != 0 else texture
        r = {"responded_category": responded}
        if with_keys:
            r.update({"shape_category": shape, "texture_category": texture})
        rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_decision_table_with_and_without_cue_keys(tmp_path):
    csv = _write_decisions_csv(str(tmp_path / "d.csv"))
    row, n = geo.shape_texture_decision_table(csv, "m")
    assert n == 40
    assert row["shape_decision_pct"] == 75.0
    assert row["texture_decision_pct"] == 25.0
    csv2 = _write_decisions_csv(str(tmp_path / "d2.csv"), with_keys=False)
    row2, _ = geo.shape_texture_decision_table(csv2, "m")
    assert row2["shape_decision_pct"] is None           # honest absence
    assert "no invented mapping" in row2["note"]
    bad = str(tmp_path / "d3.csv")
    pd.DataFrame([{"other": 1}]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="responded_category"):
        geo.shape_texture_decision_table(bad, "m")


def test_cue_conflict_write_refuses_divergence(tmp_path):
    csv = _write_decisions_csv(str(tmp_path / "d.csv"))
    table = stb.build_cue_conflict_table(csv, "m")
    assert len(table) == 2                              # model + human rows
    out = str(tmp_path / "t.csv")
    stb.write_cue_conflict_table(table.copy(), csv, out)
    assert os.path.exists(out)
    corrupt = table.copy()
    corrupt.iloc[0, corrupt.columns.get_loc("shape_decision_pct")] = 99.9
    out2 = str(tmp_path / "t2.csv")
    with pytest.raises(AssertionError, match="diverges from its own source"):
        stb.write_cue_conflict_table(corrupt, csv, out2)
    assert not os.path.exists(out2)


def test_collect_decisions_is_a_black_box_responder():
    class Fixed(nn.Module):
        def forward(self, x):
            return torch.cat([torch.ones(x.shape[0], 1),
                              torch.zeros(x.shape[0], 3)], dim=1)
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.rand(4, 3, 8, 8), torch.zeros(4, dtype=torch.long))
    loader = DataLoader(ds, batch_size=4)
    loader.dataset.class_to_idx = {"alpha": 0, "beta": 1, "gamma": 2,
                                   "delta": 3}
    df = stb.collect_decisions(Fixed(), loader)
    assert set(df.responded_category) == {"alpha"}      # argmax of the fixed head


# ── compactness: derive-twice, assert, report ────────────────────────────────

def test_count_params_cross_check_and_anomaly():
    m = nn.Sequential(nn.Flatten(), nn.Linear(8, 4))
    m.register_buffer("frozen_mask", torch.ones(4))     # a buffer: not a param
    counts = comp.count_params(m)
    assert counts["total"] == 8 * 4 + 4                 # weights + bias
    # A state_dict key that is neither param nor buffer -> loud failure.
    sd = m.state_dict()
    sd["bogus_key"] = torch.zeros(3)
    with pytest.raises(AssertionError, match="neither parameters nor buffers"):
        orig = nn.Module.state_dict
        nn.Module.state_dict = lambda self, *a, **k: sd
        try:
            comp.count_params(m)
        finally:
            nn.Module.state_dict = orig


def test_compactness_report_self_consistent(tmp_path):
    m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten(),
                      nn.Linear(4 * 6 * 6, 10))
    rep = comp.compactness_report(m, input_size=8,
                                  out_json=str(tmp_path / "c.json"))
    assert rep["params_total"] == sum(p.numel() for p in m.parameters())
    assert rep["est_macs_per_image"] >= 4 * 6 * 6 * 10  # >= the Linear walk
    with open(tmp_path / "c.json") as f:
        assert json.load(f)["params_total"] == rep["params_total"]


# ── data-infra validations (structural, data-less) ──────────────────────────

def test_imagenet100_root_validation(tmp_path):
    root = tmp_path / "in100"
    (root / "train").mkdir(parents=True)
    for i in range(100):
        (root / "train" / f"c{i:03d}").mkdir()
    assert im1.validate_imagenet100_root(str(root), split="train") == 100
    (root / "train" / "c_extra").mkdir()                # 101 classes
    with pytest.raises(ValueError, match="exactly 100"):
        im1.validate_imagenet100_root(str(root), split="train")
    with pytest.raises(FileNotFoundError):
        im1.validate_imagenet100_root(str(tmp_path / "nope"))


def test_imagenet_c_registry_and_validation(tmp_path):
    with pytest.raises(ValueError, match="unknown corruption"):
        imc.corruption_dir_name("lava", 3)
    with pytest.raises(ValueError, match="severity"):
        imc.corruption_dir_name("fog", 6)
    assert imc.corruption_dir_name("fog", 3) == os.path.join("fog", "3")
    root = tmp_path / "in-c"
    (root / "fog" / "3").mkdir(parents=True)
    present = imc.validate_imagenet_c_root(str(root))
    assert present == 1                                 # partial, reported
    with pytest.raises(ValueError, match="missing"):
        imc.validate_imagenet_c_root(str(root), require_all=True)
    assert len(imc.IMAGENET_C_CORRUPTIONS) == 15
    assert imc.IMAGENET_C_SEVERITIES == (1, 2, 3, 4, 5)


def test_geirhos_stimuli_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        geo.validate_stimuli_root(str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no category"):
        geo.validate_stimuli_root(str(empty))
    (tmp_path / "stim" / "cat_a").mkdir(parents=True)
    assert geo.validate_stimuli_root(str(tmp_path / "stim")) == 1
