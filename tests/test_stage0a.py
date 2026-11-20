from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from stage0a.checkpointing import (
    epoch_seed,
    load_checkpoint,
    restore_training_state,
    save_rolling,
    validate_resume,
)
from stage0a.config import Stage0AConfig
from stage0a.data import split_fingerprint, stratified_holdout
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
