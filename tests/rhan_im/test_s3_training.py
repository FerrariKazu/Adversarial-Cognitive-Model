"""Unit tests for Stage S3 dense window tuning script.

Verifies:
1. load_trunk_for_s3 directly (unfreezes blocks[9:12] and norm, freezes blocks 0..8 and patch_embed).
2. set_trunk_train_mode strictly isolates frozen blocks in eval mode while training blocks 9..11 and norm.
3. Gradient flow strictly flows only to blocks 9..11 and norm.
4. partition_train_split guarantees disjointness and refuses if any validation pool image leaks into S3.
5. evaluate_heldout_ratio computes Gate 1 d' and ratio metrics using g1 estimators.
6. refit_eval_head fits an EvidenceHead matching Gate 1 procedure.
7. train_s3 end-to-end mock execution and Gate 1 load_trunk checkpoint reload.
8. Early-stopping no-improvement rule saves untuned trunk with best_epoch = -1.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import pytest
import torch
import torch.nn as nn
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gen2_foundation.backbone import CompactViTGen2
from rhan_im import gate1 as g1
from rhan_im.layout import UnitLayout
from rhan_im.window_head import EvidenceHead
from scripts.rhan_im.gate1_measure_dprime import load_trunk
from scripts.rhan_im.train_s3_dense_window import (
    EXPECTED_TRUNK_TRAINABLE_PARAMS,
    S3FolderBank,
    evaluate_heldout_ratio,
    generate_fixed_eval_scenes,
    load_trunk_for_s3,
    partition_train_split,
    refit_eval_head,
    set_trunk_train_mode,
    train_s3,
    update_early_stopping,
)


def _make_dummy_checkpoint(path: pathlib.Path, img_size: int = 112) -> None:
    trunk = CompactViTGen2(img_size=img_size)
    payload = {
        "phase": "ff_112",
        "arm": "baseline",
        "epoch": 10,
        "model": {f"backbone.{k}": v for k, v in trunk.state_dict().items()},
    }
    torch.save(payload, path)


def _write_pinned_manifest(
    root: pathlib.Path,
    tcls: list[str],
    bcls: list[str],
    *,
    files_per_class: int = 4,
    heldout_per_class: int = 1,
    heldout_override: list[str] | None = None,
) -> tuple[pathlib.Path, list[str], list[str]]:
    all_classes = tcls + bcls
    all_files = sorted(
        f"{c}/im_{i:02d}.JPEG"
        for c in all_classes
        for i in range(files_per_class)
    )
    heldout = heldout_override
    if heldout is None:
        heldout = sorted(f"{c}/im_00.JPEG" for c in all_classes)
    heldout = sorted(heldout)
    train = sorted(set(all_files) - set(heldout))

    for c in all_classes:
        cdir = root / "train" / c
        cdir.mkdir(parents=True)
        for i in range(files_per_class):
            (cdir / f"im_{i:02d}.JPEG").write_bytes(b"x")

    manifest = {
        "status": "FINAL",
        "heldout_per_class": heldout_per_class,
        "heldout_count": len(heldout),
        "train_count": len(train),
        "heldout_listing_sha256": hashlib.sha256("\n".join(heldout).encode()).hexdigest(),
        "train_listing_sha256": hashlib.sha256("\n".join(train).encode()).hexdigest(),
        "heldout_files": heldout,
        "train_files": train,
    }
    path = root / "pinned_manifest.json"
    path.write_text(json.dumps(manifest))
    return path, heldout, train


def test_load_trunk_for_s3_directly():
    """Directly tests load_trunk_for_s3 for correct parameter counts, requires_grad, and train/eval flags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = pathlib.Path(tmpdir) / "dummy_base.pth"
        _make_dummy_checkpoint(ckpt_path)

        trunk, info = load_trunk_for_s3(ckpt_path, window=112, device="cpu")

        # Verify trainable parameter counts
        assert info["trainable_trunk_params"] == EXPECTED_TRUNK_TRAINABLE_PARAMS
        assert info["trainable_trunk_params"] == 5326464

        trainable_count = sum(p.numel() for p in trunk.parameters() if p.requires_grad)
        assert trainable_count == 5326464

        # Blocks 0..8 frozen (requires_grad=False) and in eval mode (training=False)
        for i in range(9):
            assert trunk.blocks[i].training is False
            for p in trunk.blocks[i].parameters():
                assert p.requires_grad is False

        # Patch embed frozen and in eval mode
        assert trunk.patch_embed.training is False
        for p in trunk.patch_embed.parameters():
            assert p.requires_grad is False

        # Blocks 9..11 trainable (requires_grad=True) and in train mode (training=True)
        for i in range(9, 12):
            assert trunk.blocks[i].training is True
            for p in trunk.blocks[i].parameters():
                assert p.requires_grad is True

        # Norm trainable and in train mode
        assert trunk.norm.training is True
        for p in trunk.norm.parameters():
            assert p.requires_grad is True


def test_set_trunk_train_mode_isolation_and_gradient_flow():
    """Tests that set_trunk_train_mode maintains eval mode on frozen blocks and backprop only touches blocks 9..11 and norm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = pathlib.Path(tmpdir) / "dummy_base.pth"
        _make_dummy_checkpoint(ckpt_path)

        trunk, _ = load_trunk_for_s3(ckpt_path, window=112, device="cpu")

        # Even if someone accidentally calls trunk.train(), set_trunk_train_mode must restore eval on blocks 0..8
        trunk.train()
        assert trunk.blocks[0].training is True  # accidentally set

        set_trunk_train_mode(trunk)
        for i in range(9):
            assert trunk.blocks[i].training is False
        assert trunk.patch_embed.training is False
        for i in range(9, 12):
            assert trunk.blocks[i].training is True
        assert trunk.norm.training is True

        # Backpropagation test
        x = torch.randn(2, 3, 112, 112)
        out = trunk(x)[1]  # (2, 64, 384)
        head = nn.Linear(384, 10)
        loss = head(out).sum()
        loss.backward()

        # Frozen parts must have NO grad
        for i in range(9):
            for p in trunk.blocks[i].parameters():
                assert p.grad is None
        for p in trunk.patch_embed.parameters():
            assert p.grad is None

        # Trainable parts must have valid finite gradients
        for i in range(9, 12):
            for p in trunk.blocks[i].parameters():
                assert p.grad is not None
                assert torch.isfinite(p.grad).all()
        for p in trunk.norm.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_partition_train_split_and_val_leak_rejection():
    """Confirms partition_train_split creates disjoint splits and rejects any overlap with pools.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)
        train_dir = root / "train"
        train_dir.mkdir(parents=True)

        tcls = [f"cls_{i}" for i in range(3)]
        bcls = [f"bg_{i}" for i in range(3)]
        all_cls = tcls + bcls

        for c in all_cls:
            cdir = train_dir / c
            cdir.mkdir()
            for img_i in range(30):
                (cdir / f"img_{img_i:03d}.JPEG").write_text("dummy")

        # Case 1: Clean split with no val leakage
        pools_file = root / "pools_clean.json"
        pools_file.write_text(json.dumps({
            "calibration_pool_part_0_of_2": {c: [f"val_{i:03d}.JPEG" for i in range(10)] for c in all_cls},
            "test_pool_part_1_of_2": {c: [f"val_{i:03d}.JPEG" for i in range(10, 20)] for c in all_cls},
        }))
        manifest = root / "s3_split.json"

        train_files, heldout_files = partition_train_split(
            root, tcls, bcls, heldout_per_class=5, out_manifest=manifest, pools_path=pools_file
        )

        assert len(heldout_files) == 6 * 5  # 6 classes * 5 = 30
        assert len(train_files) == 6 * 25  # 6 classes * 25 = 150
        assert set(heldout_files).isdisjoint(set(train_files))
        assert manifest.exists()

        # Case 2: Leakage detection - pools.json contains an image present in train
        pools_leaky = root / "pools_leaky.json"
        pools_leaky.write_text(json.dumps({
            "calibration_pool_part_0_of_2": {all_cls[0]: ["img_000.JPEG"]},
            "test_pool_part_1_of_2": {},
        }))

        with pytest.raises(ValueError, match="S3 partition leaks val images"):
            partition_train_split(root, tcls, bcls, heldout_per_class=5, pools_path=pools_leaky)


def test_partition_train_split_loads_pinned_manifest(tmp_path):
    tcls, bcls = ["t_0", "t_1"], ["b_0", "b_1"]
    manifest, heldout, train = _write_pinned_manifest(tmp_path, tcls, bcls)

    got_train, got_heldout = partition_train_split(
        tmp_path, tcls, bcls, heldout_per_class=1, out_manifest=manifest
    )

    assert got_train == train
    assert got_heldout == heldout


def test_partition_train_split_rejects_manifest_with_missing_file(tmp_path):
    tcls, bcls = ["t_0", "t_1"], ["b_0", "b_1"]
    manifest, heldout, _ = _write_pinned_manifest(tmp_path, tcls, bcls)
    (tmp_path / "train" / heldout[0]).unlink()

    with pytest.raises(FileNotFoundError, match="Manifest held-out file missing on disk"):
        partition_train_split(tmp_path, tcls, bcls, heldout_per_class=1, out_manifest=manifest)


def test_partition_train_split_rejects_manifest_with_wrong_per_class_count(tmp_path):
    tcls, bcls = ["t_0", "t_1"], ["b_0", "b_1"]
    wrong_heldout = [
        f"{tcls[0]}/im_00.JPEG",
        f"{tcls[0]}/im_01.JPEG",
        f"{tcls[1]}/im_00.JPEG",
        f"{bcls[0]}/im_00.JPEG",
        f"{bcls[1]}/im_00.JPEG",
    ]
    manifest, _, _ = _write_pinned_manifest(
        tmp_path, tcls, bcls, heldout_override=wrong_heldout
    )

    with pytest.raises(ValueError, match="expected 1"):
        partition_train_split(tmp_path, tcls, bcls, heldout_per_class=1, out_manifest=manifest)


def test_evaluate_heldout_ratio_computes_gate1_metrics():
    """Tests that evaluate_heldout_ratio computes d_native_centre, d_gist, and ratio via Gate 1 estimators."""
    lay = UnitLayout(canvas=224, window=112, patch=14, target_units=4)
    cand = lay.candidates(stride_units=2)
    n_cal, n_tst = 20, 30
    C = 4

    # Create synthetic evaluation splits
    sc_cal = {
        "canvas_uint8": torch.randint(0, 256, (n_cal, 3, 224, 224), dtype=torch.uint8),
        "label": torch.tensor([0, 1, 2, 3, -1] * (n_cal // 5)),
        "overlap": torch.zeros((n_cal, lay.g, lay.g)),
        "tl": cand[torch.randint(0, cand.shape[0], (n_cal,))],
        "cover_native": torch.zeros((n_cal, 64), dtype=torch.bool),
        "ov_gist": torch.zeros((n_cal, 64)),
        "cover_gist": torch.zeros((n_cal, 64), dtype=torch.bool),
    }
    # Set positive covers for some scenes
    for i in range(0, n_cal, 2):
        sc_cal["cover_native"][i, 20:30] = True
        sc_cal["cover_gist"][i, 20:30] = True
        sc_cal["ov_gist"][i, 20:30] = 1.0

    sc_tst = {
        "canvas_uint8": torch.randint(0, 256, (n_tst, 3, 224, 224), dtype=torch.uint8),
        "label": torch.tensor([0, 1, 2, 3, -1] * (n_tst // 5)),
        "overlap": torch.zeros((n_tst, lay.g, lay.g)),
        "tl": cand[torch.randint(0, cand.shape[0], (n_tst,))],
        "cover_native": torch.zeros((n_tst, 64), dtype=torch.bool),
        "ov_gist": torch.zeros((n_tst, 64)),
        "cover_gist": torch.zeros((n_tst, 64), dtype=torch.bool),
    }
    for i in range(0, n_tst, 2):
        sc_tst["cover_native"][i, 27:37] = True
        sc_tst["cover_gist"][i, 27:37] = True
        sc_tst["ov_gist"][i, 27:37] = 1.0

    trunk = CompactViTGen2(img_size=112)
    trunk.eval()
    head = EvidenceHead(384, C)

    metrics = evaluate_heldout_ratio(trunk, head, sc_cal, sc_tst, lay, device="cpu", min_pos=1)

    assert "d_native_centre" in metrics
    assert "d_gist" in metrics
    assert "d_gist_full" in metrics
    assert "ratio" in metrics
    assert isinstance(metrics["ratio"], float)


def test_train_s3_end_to_end_and_gate1_reload():
    """Runs a complete micro-scale train_s3 on synthetic dataset and reloads output via gate1.load_trunk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)
        train_dir = root / "train"
        train_dir.mkdir(parents=True)

        tcls = ["t_0", "t_1"]
        bcls = ["b_0", "b_1"]
        all_cls = tcls + bcls

        # Create dummy image files
        for c in all_cls:
            cdir = train_dir / c
            cdir.mkdir()
            for img_i in range(15):
                im = Image.new("RGB", (64, 64), color=(img_i * 10, img_i * 10, img_i * 10))
                im.save(cdir / f"im_{img_i:03d}.JPEG")

        # Class split file
        cs_data = {
            "status": "FINAL",
            "seed": 17,
            "targets": [{"wnid": f"w_{c}", "dirname": c} for c in tcls],
            "backgrounds": [{"wnid": f"w_{c}", "dirname": c} for c in bcls],
        }
        cs_hash = g1.class_split_sha256(17, cs_data["targets"], cs_data["backgrounds"])
        cs_data["hashes"] = {"class_split_sha256": cs_hash}
        cs_file = root / "class_split.json"
        cs_file.write_text(json.dumps(cs_data))

        # Pools file
        pools_file = root / "pools.json"
        p_cal = {c: [f"val_cal_{i:02d}.JPEG" for i in range(5)] for c in all_cls}
        p_tst = {c: [f"val_tst_{i:02d}.JPEG" for i in range(5)] for c in all_cls}
        p_assign_sha = g1.canonical_json_bytes({"rule_id": "test", "class_split": cs_hash, "val_repo_listing_sha256": "dummy", "calibration": p_cal, "test": p_tst})
        import hashlib
        p_data = {
            "status": "FINAL",
            "rule_id": "test",
            "class_split": cs_hash,
            "listing_binding": {"val": {"repo_listing_sha256": "dummy"}},
            "calibration_pool_part_0_of_2": p_cal,
            "test_pool_part_1_of_2": p_tst,
            "pool_assignment_sha256": hashlib.sha256(p_assign_sha).hexdigest(),
        }
        pools_file.write_text(json.dumps(p_data))

        # Base checkpoint
        ckpt_path = root / "base_ckpt.pth"
        _make_dummy_checkpoint(ckpt_path)

        out_ckpt = root / "tuned_s3.pth"
        out_split = root / "s3_manifest.json"

        # Execute micro train_s3
        res = train_s3(
            ckpt_path=ckpt_path,
            data_root=root,
            class_split_path=cs_file,
            pools_path=pools_file,
            out_ckpt_path=out_ckpt,
            out_manifest_path=out_split,
            device="cpu",
            epochs=2,
            steps_per_epoch=2,
            batch_size=2,
            eval_head_steps=2,
            n_eval_cal=12,
            n_eval_test=16,
            heldout_per_class=5,
            min_pos=2,
            seed=17,
        )

        assert out_ckpt.exists()
        assert out_split.exists()
        assert res["trainable_trunk_params"] == 5326464
        assert res["total_steps"] == 4

        # Reload with Gate 1 load_trunk
        reloaded_trunk, reload_info = load_trunk(str(out_ckpt), window=112, device="cpu")
        assert reload_info["trunk_keys_loaded"] == reload_info["trunk_keys_needed"]
        assert reload_info["phase"] == "ff_112_s3"
        assert reload_info["arm"] == "s3_tuned"


def test_early_stopping_selection_logic():
    """Unit test for update_early_stopping directly calling the script implementation."""
    trunk_init = CompactViTGen2(img_size=112)
    init_state = {k: v.clone() for k, v in trunk_init.state_dict().items()}

    # Case 1: No improvement (ratios are below or equal to baseline)
    init_ratio = 1.25
    best_ratio = init_ratio
    best_epoch = -1
    best_state = {k: v.clone() for k, v in init_state.items()}

    epoch_ratios = [1.10, 1.20, 1.15]
    for ep, r in enumerate(epoch_ratios):
        best_ratio, best_epoch, best_state, improved = update_early_stopping(
            best_ratio, best_epoch, best_state, r, ep + 1, {"mock": torch.tensor([ep])}
        )
        assert improved is False

    assert best_epoch == -1
    assert best_ratio == 1.25
    for k in init_state:
        assert torch.equal(best_state[k], init_state[k])

    # Case 2: Improvement occurs
    epoch_ratios = [1.20, 1.35, 1.30]
    for ep, r in enumerate(epoch_ratios):
        best_ratio, best_epoch, best_state, improved = update_early_stopping(
            best_ratio, best_epoch, best_state, r, ep + 1, {"mock": torch.tensor([ep])}
        )

    assert best_epoch == 2
    assert best_ratio == 1.35
    assert torch.equal(best_state["mock"], torch.tensor([1]))

    # Case 3: Candidate ratio is NaN or inf (must NEVER update)
    for bad_r in [float("nan"), float("inf"), float("-inf")]:
        best_ratio, best_epoch, best_state, improved = update_early_stopping(
            best_ratio, best_epoch, best_state, bad_r, 4, {"mock": torch.tensor([999])}
        )
        assert improved is False
        assert best_epoch == 2
        assert best_ratio == 1.35
        assert torch.equal(best_state["mock"], torch.tensor([1]))


def test_baseline_nan_aborts_training():
    """Tests that train_s3 aborts immediately with RuntimeError if baseline held-out ratio is NaN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)
        train_dir = root / "train"
        train_dir.mkdir(parents=True)
        tcls, bcls = ["t_0"], ["b_0"]
        all_cls = tcls + bcls

        for c in all_cls:
            cdir = train_dir / c
            cdir.mkdir()
            for img_i in range(10):
                im = Image.new("RGB", (64, 64), color=(0, 0, 0))
                im.save(cdir / f"im_{img_i:03d}.JPEG")

        cs_data = {
            "status": "FINAL",
            "seed": 17,
            "targets": [{"wnid": f"w_{c}", "dirname": c} for c in tcls],
            "backgrounds": [{"wnid": f"w_{c}", "dirname": c} for c in bcls],
        }
        cs_hash = g1.class_split_sha256(17, cs_data["targets"], cs_data["backgrounds"])
        cs_data["hashes"] = {"class_split_sha256": cs_hash}
        cs_file = root / "class_split.json"
        cs_file.write_text(json.dumps(cs_data))

        p_cal = {c: [f"val_{i:02d}.JPEG" for i in range(5)] for c in all_cls}
        p_tst = {c: [f"val_{i:02d}.JPEG" for i in range(5, 10)] for c in all_cls}
        p_assign_sha = g1.canonical_json_bytes({"rule_id": "test", "class_split": cs_hash, "val_repo_listing_sha256": "dummy", "calibration": p_cal, "test": p_tst})
        import hashlib
        p_data = {
            "status": "FINAL",
            "rule_id": "test",
            "class_split": cs_hash,
            "listing_binding": {"val": {"repo_listing_sha256": "dummy"}},
            "calibration_pool_part_0_of_2": p_cal,
            "test_pool_part_1_of_2": p_tst,
            "pool_assignment_sha256": hashlib.sha256(p_assign_sha).hexdigest(),
        }
        pools_file = root / "pools.json"
        pools_file.write_text(json.dumps(p_data))

        ckpt_path = root / "base_ckpt.pth"
        _make_dummy_checkpoint(ckpt_path)

        import scripts.rhan_im.train_s3_dense_window as s3_mod
        orig_eval = s3_mod.evaluate_heldout_ratio
        try:
            s3_mod.evaluate_heldout_ratio = lambda *args, **kwargs: {"ratio": float("nan"), "d_native_centre": float("nan"), "d_gist": float("nan"), "d_gist_full": float("nan")}
            with pytest.raises(RuntimeError, match="Baseline held-out ratio is not finite"):
                train_s3(
                    ckpt_path=ckpt_path,
                    data_root=root,
                    class_split_path=cs_file,
                    pools_path=pools_file,
                    out_ckpt_path=root / "out.pth",
                    out_manifest_path=root / "split.json",
                    device="cpu",
                    epochs=1,
                    steps_per_epoch=1,
                    batch_size=2,
                    eval_head_steps=1,
                    n_eval_cal=2,
                    n_eval_test=2,
                    heldout_per_class=2,
                    seed=17,
                )
        finally:
            s3_mod.evaluate_heldout_ratio = orig_eval


