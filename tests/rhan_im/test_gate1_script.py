import importlib.util
import json
import pathlib
import types

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_script():
    spec = importlib.util.spec_from_file_location("gate1_script", ROOT / "scripts" / "rhan_im" / "gate1_measure_dprime.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tiny_args(**kw):
    base = dict(self_test=True, ckpt=None, data_root=None, device="cpu", canvas=112, window=56, stride_units=2, classes=4,
                sizes=[2, 3], alphas=[1.0], op_size=2, p_null=0.3, steps=6, batch=12, lr=2e-3, n_cal=100, n_test=140,
                n_pairs=50, seed=0, out=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_self_test_runs_end_to_end_and_is_deterministic():
    m = load_script()
    a, b = m.run(tiny_args()), m.run(tiny_args())
    for r in (a, b):
        r.pop("seconds")
    assert json.dumps(a, default=lambda o: None) == json.dumps(b, default=lambda o: None)
    for key in ("summary", "verdict_draft", "cross_look", "gist_by_overlap", "native", "gist", "within_look", "spillover", "head_training_log", "config", "trunk",
                "code_sha256", "data_listing"):
        assert key in a
    from rhan_im import gate1 as g1
    assert a["code_sha256"] == {f: g1.file_sha256(ROOT / f) for f in g1.MEASUREMENT_CODE_FILES}   # the run records the code it ran, for the aggregate to compare
    assert a["data_listing"] is None                                                               # synthetic data has no listing
    s = a["summary"]
    assert set(s) >= {"d_native_centre", "d_native_edge_ring", "d_gist", "zroc_slope_median", "null_sd_median", "pos_sd_median", "rho_null",
                      "vif_null_op", "vif_pos_op", "op_block_tokens", "d_gist_full", "spill_adjacent_op", "spill_far_op"}


def test_chunked_scene_collection_neither_drops_nor_duplicates_scenes():
    m = load_script()
    for chunk, n_pairs in ((200, 50), (64, 50), (64, 100), (37, 140), (1000, 140)):
        r = m.run(tiny_args(chunk_scenes=chunk, n_pairs=n_pairs))
        assert r["counts"]["n_cal"] == 100 and r["counts"]["n_test"] == 140 and r["counts"]["n_pairs"] == n_pairs
        assert r["counts"]["n_test_null_scenes"] + r["counts"]["n_test_target_scenes"] == 140
        assert r["cross_look"]["n_null_pairs"] > 0


def test_checkpoint_loader_reads_the_training_format_and_reproduces_the_tokens(tmp_path):
    from gen2_foundation.model import Gen2FoundationModel
    from rhan_im.trunk_adapter import ff_token_fn
    m = load_script()
    torch.manual_seed(0)
    model = Gen2FoundationModel(phase="ff_112", fovea_size=112).eval()
    path = tmp_path / "ckpt.pth"
    torch.save({"phase": "ff_112", "epoch": 3, "model": model.state_dict()}, path)
    trunk, info = m.load_trunk(str(path), 112, "cpu")
    assert info["trunk_keys_loaded"] == info["trunk_keys_needed"] and info["epoch"] == 3 and len(info["sha256"]) == 64
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        ref = model.backbone._trunk_forward(model.backbone._prep(x))[:, 2:]
        got = ff_token_fn(trunk)(x)[1]
    assert torch.equal(ref, got)


def test_checkpoint_loader_refuses_a_half_loaded_trunk(tmp_path):
    m = load_script()
    path = tmp_path / "bad.pth"
    torch.save({"model": {"backbone.cls_token": torch.zeros(1, 1, 384)}}, path)
    with pytest.raises(SystemExit):
        m.load_trunk(str(path), 112, "cpu")


def make_fake_imagefolder(root, n_classes=8, n_files=6, size=40, seed=0):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        for c in range(n_classes):
            d = root / split / f"c{c:02d}"
            d.mkdir(parents=True)
            for i in range(n_files):
                Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(d / f"img{i:03d}.png")
    return root


def test_folderbank_parts_are_disjoint_and_together_complete(tmp_path):
    m = load_script()
    root = make_fake_imagefolder(tmp_path)
    classes = sorted(p.name for p in (root / "val").iterdir())
    t, b = classes[:4], classes[4:]
    a = m.FolderBank(root, "val", t, b, seed=1, part=(0, 2))
    c = m.FolderBank(root, "val", t, b, seed=2, part=(1, 2))
    full = m.FolderBank(root, "val", t, b, seed=3)
    for cls in classes:
        sa, sc, sf = set(a.files[cls]), set(c.files[cls]), set(full.files[cls])
        assert sa and sc and not (sa & sc) and (sa | sc) == sf


def test_folderbank_refuses_a_class_too_small_to_split(tmp_path):
    m = load_script()
    root = make_fake_imagefolder(tmp_path, n_classes=4, n_files=1)
    classes = sorted(p.name for p in (root / "val").iterdir())
    with pytest.raises(SystemExit):
        m.FolderBank(root, "val", classes[:2], classes[2:], part=(1, 2))


def test_run_on_an_imagefolder_uses_disjoint_calibration_and_test_pools(tmp_path, monkeypatch):
    m = load_script()
    root = make_fake_imagefolder(tmp_path)
    calls = []
    real = m.FolderBank

    class Recorder(real):
        def __init__(self, root, split, target_classes, bg_classes, seed=0, part=None):
            calls.append((split, part))
            super().__init__(root, split, target_classes, bg_classes, seed=seed, part=part)

    monkeypatch.setattr(m, "FolderBank", Recorder)
    res = m.run(tiny_args(self_test=False, data_root=str(root), classes=2, steps=4, batch=8, n_cal=60, n_test=80, n_pairs=20))
    assert ("train", None) in calls and ("val", (0, 2)) in calls and ("val", (1, 2)) in calls
    assert ("val", None) not in calls                                           # no bank may see the whole val split
    assert "within_look" in res and set(res["within_look"]) == {"size2_alpha1.0", "size3_alpha1.0"}
    assert set(res["spillover"]) == {"size2_alpha1.0", "size3_alpha1.0"}
    s = res["summary"]
    assert "vif_null_op" in s and "vif_pos_op" in s and s["op_block_tokens"] == 4.0
    assert {"token_evidence", "vif_used", "d_block_effective", "pass_all", "unmeasured"} <= set(res["verdict_draft"])
    from rhan_im import gate1 as g1
    assert res["data_listing"] == {f"{sp}_path_listing": g1.listing_hash(root / sp) for sp in ("train", "val")}    # the listing the freeze pins
    assert res["data_listing"]["val_path_listing"]["n_files"] == 8 * 6 and res["data_listing"]["train_path_listing"]["n_files"] == 8 * 6
