from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from stage0a.checkpointing import (
    epoch_seed,
    load_checkpoint,
    restore_training_state,
    save_rolling,
    validate_resume,
)
from stage0a.config import Stage0AConfig
from stage0a.data import build_index, find_imagenet1k, split_fingerprint, stratified_holdout
from stage0a.pipeline import run_resume_drill
from stage0a.training import LinearHead
from stage0a.utils import capture_rng_state


def test_config_hash_ignores_runtime_paths(tmp_path):
    a = Stage0AConfig(backbone="vit-s14", seed=17, data_root="/a", cache_root="/b", output_root="/c")
    b = Stage0AConfig(backbone="vit-s14", seed=17, data_root="/x", cache_root="/y", output_root="/z")
    assert a.canonical_hash() == b.canonical_hash()
    assert a.to_dict(include_runtime=False)["backbone"] == "vit-s14"


def test_split_is_deterministic_and_stratified():
    labels = np.repeat(np.arange(1000), 2)
    tr1, ho1 = stratified_holdout(labels, 1, 17)
    tr2, ho2 = stratified_holdout(labels, 1, 17)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(ho1, ho2)
    assert len(tr1) == 1000 and len(ho1) == 1000
    assert len(np.intersect1d(tr1, ho1)) == 0
    assert split_fingerprint("d", tr1, ho1, 17) != split_fingerprint("d", tr1, ho1, 42)


def test_find_and_index_localization_challenge_layout(tmp_path):
    input_root = tmp_path / "input"
    dataset = input_root / "imagenet-object-localization-challenge"
    train = dataset / "ILSVRC" / "Data" / "CLS-LOC" / "train"
    val = dataset / "ILSVRC" / "Data" / "CLS-LOC" / "val"
    for i in range(1000):
        (train / f"n{i:08d}").mkdir(parents=True)
    val.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(val / "ILSVRC2012_val_00000001.JPEG")
    (dataset / "LOC_val_solution.csv").write_text(
        "ImageId,PredictionString\nILSVRC2012_val_00000001,n00000000 0 0 0 0\n", encoding="utf-8"
    )

    found = find_imagenet1k(input_root)
    assert found == dataset
    index = build_index(found, "val", expected_val_count=None)
    assert len(index) == 1
    assert index.labels.tolist() == [0]
    assert index.absolute(0).is_file()


def test_find_rejects_imagenet100(tmp_path):
    input_root = tmp_path / "input"
    train = input_root / "imagenet-100" / "train"
    val = input_root / "imagenet-100" / "val"
    for i in range(100):
        (train / f"n{i:08d}").mkdir(parents=True)
    val.mkdir(parents=True)
    assert find_imagenet1k(input_root) is None


def test_find_nested_in_datasets_folder_with_validation_and_hidden_dirs(tmp_path):
    # Tests nesting under input/datasets/<slug>/... with validation and .ipynb_checkpoints
    input_root = tmp_path / "input"
    dataset = input_root / "datasets" / "user-imagenet-slug" / "subfolder"
    train = dataset / "train"
    val = dataset / "validation"
    for i in range(1000):
        (train / f"n{i:08d}").mkdir(parents=True)
    # Add hidden folder that shouldn't disrupt class count
    (train / ".ipynb_checkpoints").mkdir(parents=True)
    for i in range(1000):
        (val / f"n{i:08d}").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(val / "n00000000" / "val_0.jpg")

    found = find_imagenet1k(input_root, max_depth=5)
    assert found == dataset
    from stage0a.data import describe_search_root
    desc = describe_search_root(input_root)
    assert "datasets" in desc
    assert "user-imagenet-slug" in desc


def test_atomic_rolling_checkpoint_roundtrip(tmp_path):
    cfg = Stage0AConfig(
        backbone="vit-s14",
        seed=17,
        data_root="/nonexistent",
        cache_root=str(tmp_path / "cache"),
        output_root=str(tmp_path / "out"),
    )
    model = LinearHead(cfg.embed_dim, cfg.num_classes)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "rolling.pt"
    state = save_rolling(
        path,
        cfg=cfg,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        epoch=2,
        batch_in_epoch=3,
        global_step=123,
        history={"events": [], "best_metric": {}},
        training_fingerprint="trainfp",
        dataset_fingerprint="datafp",
        split_fingerprint="splitfp",
        backbone_weights_sha256="backbonefp",
        hardware={"device_count": 1},
    )
    assert path.exists()
    assert (tmp_path / "rolling.pt.sha256").exists()
    assert state["epoch_seed"] == epoch_seed(17, 2)
    loaded = load_checkpoint(path)
    validate_resume(
        loaded,
        cfg=cfg,
        training_fingerprint="trainfp",
        dataset_fingerprint="datafp",
        split_fingerprint="splitfp",
        backbone_weights_sha256="backbonefp",
    )
    model2 = LinearHead(cfg.embed_dim, cfg.num_classes)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 1.0)
    scaler2 = torch.amp.GradScaler("cuda", enabled=False)
    restore_training_state(loaded, model=model2, optimizer=opt2, scheduler=sched2, scaler=scaler2)
    assert loaded["global_step"] == 123
    assert loaded["batch_in_epoch"] == 3
    assert "python" in loaded["rng"] and "torch_cpu" in loaded["rng"]


def test_resume_drill_passes(tmp_path):
    report = run_resume_drill(str(tmp_path))
    assert report["status"] == "PASS"
    assert report["phase1"]["global_step"] == 7
    assert report["phase2"]["status"] == "complete"
    assert report["resume_event_present"] is True


def test_feature_repos_separated_by_backbone():
    from stage0a.config import hf_feature_repo, HF_FEATURE_REPOS
    from stage0a.hf_sync import NoopHFStore

    repo_s = hf_feature_repo("vit-s14")
    repo_b = hf_feature_repo("vit-b14")
    assert repo_s == "FerrariKazu/rhan-stage0a-dinov2-features-vits14"
    assert repo_b == "FerrariKazu/rhan-stage0a-dinov2-features-vitb14"
    assert repo_s != repo_b

    cfg_s = Stage0AConfig(backbone="vit-s14", seed=17, data_root="/a", cache_root="/b", output_root="/c")
    cfg_b = Stage0AConfig(backbone="vit-b14", seed=17, data_root="/a", cache_root="/b", output_root="/c")
    assert cfg_s.hf_feature_repo == repo_s
    assert cfg_b.hf_feature_repo == repo_b

    import pytest
    with pytest.raises(ValueError):
        hf_feature_repo("vit-large")

    noop = NoopHFStore()
    assert "vit-s14" in noop.feature_repo("vit-s14")
    assert "vit-b14" in noop.feature_repo("vit-b14")


def test_shard_save_restore_and_hash_verification(tmp_path):
    from stage0a.features import cache_dir, restore_shard, shard_bounds
    from stage0a.utils import atomic_write_json, atomic_write_text, sha256_file

    cfg = Stage0AConfig(
        backbone="vit-s14",
        seed=17,
        data_root="/nonexistent",
        cache_root=str(tmp_path / "cache"),
        output_root=str(tmp_path / "out"),
    )
    dataset_fp = "testfp123456"
    cdir = cache_dir(cfg, "train", dataset_fp)
    cdir.mkdir(parents=True, exist_ok=True)

    n_total = 100
    n_shards = 2
    dim = cfg.embed_dim

    # Pre-allocate target memmaps
    feats_mmap = np.lib.format.open_memmap(cdir / "features.npy", mode="w+", dtype=np.float16, shape=(n_total, dim))
    labels_mmap = np.lib.format.open_memmap(cdir / "labels.npy", mode="w+", dtype=np.int64, shape=(n_total,))
    feats_mmap.flush()
    labels_mmap.flush()

    start, end = shard_bounds(n_total, 0, n_shards)
    count = end - start
    assert count == 50

    shard_feats = np.ones((count, dim), dtype=np.float16) * 3.5
    shard_labels = np.arange(count, dtype=np.int64) + 10
    shard_feats_p = cdir / "shard_00_features.npy"
    shard_labels_p = cdir / "shard_00_labels.npy"
    np.save(shard_feats_p, shard_feats)
    np.save(shard_labels_p, shard_labels)

    f_sha = sha256_file(shard_feats_p)
    l_sha = sha256_file(shard_labels_p)
    atomic_write_text(str(shard_feats_p) + ".sha256", f_sha + "\n")
    atomic_write_text(str(shard_labels_p) + ".sha256", l_sha + "\n")

    info = {
        "rank": 0,
        "start": start,
        "end": end,
        "count": count,
        "features_sha256": f_sha,
        "labels_sha256": l_sha,
    }
    shard_json_p = cdir / "shard_00.json"
    atomic_write_json(shard_json_p, info)
    atomic_write_text(str(shard_json_p) + ".sha256", sha256_file(shard_json_p) + "\n")

    # Test clean restore
    restored = restore_shard(
        cfg,
        split="train",
        dataset_fingerprint=dataset_fp,
        rank=0,
        n_shards=n_shards,
        total_count=n_total,
    )
    assert restored is not None
    assert restored.get("restored") is True

    # Verify memmap was populated
    cur_feats = np.load(cdir / "features.npy", mmap_mode="r")
    assert np.allclose(cur_feats[0:50], 3.5)
    assert np.allclose(cur_feats[50:100], 0.0)

    # Test tampering detection: corrupting the sidecar or file causes restore to fail
    atomic_write_text(str(shard_feats_p) + ".sha256", "corruptedhash\n")
    failed = restore_shard(
        cfg,
        split="train",
        dataset_fingerprint=dataset_fp,
        rank=0,
        n_shards=n_shards,
        total_count=n_total,
    )
    assert failed is None


def test_sync_provenance_artifacts(tmp_path):
    from stage0a.pipeline import _path_layout, sync_provenance_artifacts
    from stage0a.utils import atomic_write_json, atomic_write_text

    layout = _path_layout("", str(tmp_path))
    Path(layout["index_root"]).mkdir(parents=True, exist_ok=True)
    Path(layout["output_root"]).mkdir(parents=True, exist_ok=True)
    (Path(layout["output_root"]) / "split").mkdir(parents=True, exist_ok=True)

    atomic_write_json(Path(layout["index_root"]) / "index_audit.json", {"test": "index"})
    atomic_write_json(Path(layout["output_root"]) / "split" / "split.json", {"test": "split"})
    atomic_write_json(Path(layout["output_root"]) / "stage0a_audit.json", {"test": "audit"})
    atomic_write_text(Path(layout["output_root"]) / "stage0a_audit.md", "# Audit Report")

    class MockHFStore:
        best_repo = "test/best-repo"
        rolling_repo = "test/rolling-repo"
        enabled = True
        uploads = []

        def upload_verified(self, local_path, repo_id, remote_name, required=False):
            self.uploads.append((str(local_path), repo_id, remote_name))

    store = MockHFStore()
    synced = sync_provenance_artifacts(str(tmp_path), store)
    assert len(synced) == 4
    assert "index_audit.json" in synced
    assert "split.json" in synced
    assert "stage0a_audit.json" in synced
    assert "stage0a_audit.md" in synced

    remote_names = [u[2] for u in store.uploads]
    assert "stage0a/provenance/index_audit.json" in remote_names
    assert "stage0a/provenance/split.json" in remote_names
    assert "stage0a/provenance/stage0a_audit.json" in remote_names
    assert "stage0a/provenance/stage0a_audit.md" in remote_names


def test_validate_resume_legacy_config_hash(tmp_path):
    import hashlib
    from stage0a.checkpointing import validate_resume, CHECKPOINT_FORMAT_VERSION, epoch_seed

    cfg = Stage0AConfig(
        backbone="vit-s14",
        seed=17,
        data_root=str(tmp_path / "data"),
        cache_root=str(tmp_path / "cache"),
        output_root=str(tmp_path / "out"),
    )
    # Construct legacy hash where checkpoint_every_steps / eval_every_steps were included
    d = cfg.to_dict(include_runtime=False)
    d["checkpoint_every_steps"] = 500
    d["eval_every_steps"] = 500
    legacy_hash = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    saved_config = dict(cfg.to_dict(include_runtime=True))
    saved_config["checkpoint_every_steps"] = 500
    saved_config["eval_every_steps"] = 500

    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "run_id": cfg.run_id,
        "config_canonical_hash": legacy_hash,
        "config": saved_config,
        "training_fingerprint": "fake_trainfp",
        "dataset_fingerprint": "fake_datafp",
        "split_fingerprint": "fake_splitfp",
        "backbone": cfg.backbone,
        "backbone_weights_sha256": "fake_weightsfp",
        "seed": cfg.seed,
        "epoch": 1,
        "batch_in_epoch": 10,
        "global_step": 500,
        "epoch_seed": epoch_seed(cfg.seed, 1),
    }

    # Should succeed without raising ResumeValidationError on config_canonical_hash
    validate_resume(
        state,
        cfg=cfg,
        training_fingerprint="fake_trainfp",
        dataset_fingerprint="fake_datafp",
        split_fingerprint="fake_splitfp",
        backbone_weights_sha256="fake_weightsfp",
    )


def test_validate_resume_raises_on_actual_mismatch(tmp_path):
    from stage0a.checkpointing import validate_resume, ResumeValidationError, CHECKPOINT_FORMAT_VERSION, epoch_seed
    import pytest

    cfg = Stage0AConfig(
        backbone="vit-s14",
        seed=17,
        data_root=str(tmp_path / "data"),
        cache_root=str(tmp_path / "cache"),
        output_root=str(tmp_path / "out"),
    )
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "run_id": cfg.run_id,
        "config_canonical_hash": "completely_wrong_hash",
        "config": cfg.to_dict(include_runtime=True),
        "training_fingerprint": "fake_trainfp",
        "dataset_fingerprint": "fake_datafp",
        "split_fingerprint": "fake_splitfp",
        "backbone": cfg.backbone,
        "backbone_weights_sha256": "fake_weightsfp",
        "seed": cfg.seed,
        "epoch": 1,
        "batch_in_epoch": 10,
        "global_step": 500,
        "epoch_seed": epoch_seed(cfg.seed, 1),
    }

    with pytest.raises(ResumeValidationError, match="resolved experiment config hash changed"):
        validate_resume(
            state,
            cfg=cfg,
            training_fingerprint="fake_trainfp",
            dataset_fingerprint="fake_datafp",
            split_fingerprint="fake_splitfp",
            backbone_weights_sha256="fake_weightsfp",
        )
