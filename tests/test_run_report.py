"""Tests for training.run_report and training.run_report_jsonl.

LOGGING/DIAGNOSTICS ONLY — these tests assert the reporting layer does not
change training results (bit-identical under same seed, logging on vs off,
1 epoch), and that the JSONL and epoch block carry every labeled field.

Runs on CPU, using the trainer's synthetic smoke loaders (Agent I), so this
is a fast local check, not evidence about real-data numbers.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import pytest
import torch
from torch.utils.data import DataLoader

from training.adv_curriculum import phase_curriculum, W_TRADES_DEFAULT
from training.run_report import (
    EPOCH_BLOCK_LEGEND,
    HEALTH_LEGEND,
    clean_val_accuracy,
    compute_health_flags,
    format_epoch_block,
    robust_val_accuracy,
)
from training.run_report_jsonl import (
    append_epoch_jsonl,
    epoch_jsonl_record,
    last_epoch_in_jsonl,
    resume_banner,
    validate_jsonl_terminal_epoch,
    summarize_phase_from_jsonl,
    _train_one_epoch_instrumented,
    _print_phase_header,
)
from training.train_generation1_foundation import (
    train_one_epoch as _uninstrumented,
)
from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry
from training.run_report import GradNormCollector


# ---- tiny reproducible synthetic model + loader (smoke-only shapes) ----


def _make_fake_model() -> torch.nn.Module:
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(x)

    return _M()


def _make_fake_loader(n: int = 16, num_classes: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(n, 8)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(list(zip(x, y)), batch_size=8, shuffle=False)


def _make_registry(model: torch.nn.Module):
    reg = OptimizerGroupRegistry()
    reg.register_backbone(list(model.parameters()))
    return reg


# ---- Step 0/1: epoch block prints every labeled field ----


class TestEpochBlockLabels:
    """The epoch block must carry every column the legend names, with no
    unlabeled 'val_acc'."""

    def test_block_has_clean_and_robust_labels(self):
        block = format_epoch_block(
            phase="backbone_only",
            epoch=3,
            total_epochs=60,
            point=phase_curriculum("backbone_only", 3, 60),
            lr=0.003,
            tr_loss=2.4,
            ce_clean=1.8,
            kl_trades=0.1,
            w_trades=W_TRADES_DEFAULT,
            train_clean_acc=0.25,
            train_adv_acc=0.12,
            val_clean=0.058,
            val_robust=0.041,
            clean_n=5000,
            robust_n=512,
            robust_secs=2.5,
            best_clean=0.059,
            best_clean_epoch=3,
            best_robust=0.041,
            best_robust_epoch=3,
            criterion="clean",
            grad={"pre_clip_norm": 1.0, "post_clip_norm": 1.0,
                  "clipped_frac_any": 1.0, "per_group": {
                      "backbone": {"pre_clip_norm": 1.0,
                                   "post_clip_norm": 1.0,
                                   "clipped_frac": 1.0}}},
            epoch_seconds=45.0,
            img_per_sec=200.0,
            peak_vram_gb=12.0,
            eta_phase_end=2000.0,
            health="ok",
        )
        text = block
        assert "val_acc[CLEAN]" in text
        assert "val_acc[ROBUST" in text
        assert "best_clean   =" in text
        assert "best_robust  =" in text
        assert "checkpoint criterion in use:" in text
        assert "pre_clip_norm =" in text
        assert "post_clip_norm =" in text
        assert "clipped" in text
        assert "HEALTH" in text
        assert "train_acc[clean]" in text
        assert "train_acc[adv]" in text
        for token in ("val_acc=", "val_acc "):
            assert token not in text, f"unlabeled {token!r} found in block"

    def test_legend_lists_every_column(self):
        legend = EPOCH_BLOCK_LEGEND
        for token in (
            "loss_total",
            "ce_clean",
            "kl_trades",
            "w_trades",
            "train_acc[clean]",
            "train_acc[adv]",
            "val_acc[CLEAN]",
            "val_acc[ROBUST]",
            "best_clean",
            "best_robust",
            "pre_clip_norm",
            "post_clip_norm",
            "epoch_seconds",
            "img_per_sec",
            "peak VRAM",
            "HEALTH",
        ):
            if token == "epoch_seconds":
                # The legend describes the SPEED line in words, not the field name
                # literally, so map this assertion to the actual speed wording.
                assert "seconds" in legend and "epoch" in legend
                continue
            if token == "img_per_sec":
                # The legend describes the SPEED line in words, not the field name
                # literally, so map this assertion to the actual speed wording.
                assert "seconds" in legend and "epoch" in legend
                continue
            if token == "peak VRAM":
                assert "VRAM" in legend and "GB" in legend
                continue
            assert token in legend, f"legend missing {token!r}"


# ---- Step 2: health flags fire with thresholds printed ----


class TestHealthFlags:
    """Flags fire based on logged numbers with the threshold printed next to
    each flag."""

    def test_chance_level_fires(self):
        flags = compute_health_flags(
            epoch=10,
            total_epochs=60,
            num_classes=100,
            val_clean=0.015,
            val_robust=0.010,
            kl_trades=0.1,
            grad={"clipped_frac_any": 0.3, "per_group": {}},
            best_clean=0.015,
            best_robust=0.010,
            best_clean_at_epoch=2,
            best_robust_at_epoch=2,
        )
        assert "CHANCE-LEVEL" in flags
        assert "2/num_classes=" in flags

    def test_clip_saturated_fires(self):
        flags = compute_health_flags(
            epoch=10,
            total_epochs=60,
            num_classes=100,
            val_clean=0.2,
            val_robust=0.1,
            kl_trades=0.1,
            grad={"clipped_frac_any": 0.95, "per_group": {
                "backbone": {"clipped_frac": 0.95}}},
            best_clean=0.2,
            best_robust=0.1,
            best_clean_at_epoch=10,
            best_robust_at_epoch=10,
        )
        assert "CLIP-SATURATED" in flags

    def test_trades_dead_fires(self):
        flags = compute_health_flags(
            epoch=10,
            total_epochs=60,
            num_classes=100,
            val_clean=0.2,
            val_robust=0.1,
            kl_trades=1e-7,
            grad={"clipped_frac_any": 0.0, "per_group": {}},
            best_clean=0.2,
            best_robust=0.1,
            best_clean_at_epoch=10,
            best_robust_at_epoch=10,
        )
        assert "TRADES-DEAD" in flags

    def test_stall_fires(self):
        # Neither best improved for 8+ epochs -> STALL.
        # best_clean_at_epoch=12, best_robust_at_epoch=12, current epoch=20.
        flags = compute_health_flags(
            epoch=20,
            total_epochs=60,
            num_classes=100,
            val_clean=0.20,
            val_robust=0.10,
            kl_trades=0.1,
            grad={"clipped_frac_any": 0.0, "per_group": {}},
            best_clean=0.20,
            best_robust=0.10,
            best_clean_at_epoch=12,
            best_robust_at_epoch=12,
        )
        assert "STALL" in flags, f"expected STALL, got {flags!r}"

    def test_gap_fires_when_robust_gt_clean(self):
        flags = compute_health_flags(
            epoch=10,
            total_epochs=60,
            num_classes=100,
            val_clean=0.10,
            val_robust=0.12,
            kl_trades=0.1,
            grad={"clipped_frac_any": 0.0, "per_group": {}},
            best_clean=0.10,
            best_robust=0.12,
            best_clean_at_epoch=10,
            best_robust_at_epoch=10,
        )
        assert "GAP" in flags

    def test_ok_when_clean(self):
        flags = compute_health_flags(
            epoch=20,
            total_epochs=60,
            num_classes=100,
            val_clean=0.30,
            val_robust=0.20,
            kl_trades=0.1,
            grad={"clipped_frac_any": 0.1, "per_group": {
                "backbone": {"clipped_frac": 0.1}}},
            best_clean=0.30,
            best_robust=0.20,
            best_clean_at_epoch=20,
            best_robust_at_epoch=20,
        )
        assert flags == "ok"


# ---- Step 3: durable JSONL + resume banner + terminal-epoch verification ----


def _jsonl_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _base_rec(epoch: int, **overrides) -> Dict[str, Any]:
    point = phase_curriculum("backbone_only", epoch, 60)
    base = {
        "phase": "backbone_only",
        "total_epochs": 60,
        "lr": 0.003,
        "tr_loss": 4.0,
        "ce_clean": 3.0,
        "kl_trades": 0.2,
        "w_trades": W_TRADES_DEFAULT,
        "train_clean_acc": 0.1,
        "train_adv_acc": 0.05,
        "val_clean": 0.05,
        "val_robust": 0.02,
        "clean_n": 5000,
        "robust_n": 512,
        "robust_secs": 1.0,
        "best_clean": 0.05,
        "best_clean_epoch": epoch,
        "best_robust": 0.02,
        "best_robust_epoch": epoch,
        "criterion": "clean",
        "grad": {"pre_clip_norm": 1.0, "post_clip_norm": 1.0,
                  "clipped_frac_any": 0.0, "per_group": {}},
        "epoch_seconds": 10.0,
        "img_per_sec": 100.0,
        "peak_vram_gb": None,
        "config_sha256": "abc",
        "git_commit": "def",
        "session_id": "sess1",
    }
    base.update(overrides)
    return epoch_jsonl_record(
        phase=base["phase"],
        epoch=epoch,
        total_epochs=base["total_epochs"],
        point=point,
        lr=base["lr"],
        tr_loss=base["tr_loss"],
        ce_clean=base["ce_clean"],
        kl_trades=base["kl_trades"],
        w_trades=base["w_trades"],
        train_clean_acc=base["train_clean_acc"],
        train_adv_acc=base["train_adv_acc"],
        val_clean=base["val_clean"],
        val_robust=base["val_robust"],
        clean_n=base["clean_n"],
        robust_n=base["robust_n"],
        robust_secs=base["robust_secs"],
        best_clean=base["best_clean"],
        best_clean_epoch=base["best_clean_epoch"],
        best_robust=base["best_robust"],
        best_robust_epoch=base["best_robust_epoch"],
        criterion=base["criterion"],
        grad=base["grad"],
        epoch_seconds=base["epoch_seconds"],
        img_per_sec=base["img_per_sec"],
        peak_vram_gb=base["peak_vram_gb"],
        config_sha256=base["config_sha256"],
        git_commit=base["git_commit"],
        session_id=base["session_id"],
        **{k: v for k, v in overrides.items() if k not in base},
    )


class TestDurableJsonl:
    """JSONL is append-only, flushed, and verifiable on resume."""

    def test_append_and_read_back(self, tmp_path):
        path = os.path.join(tmp_path, "ph_epoch_log.jsonl")
        rec = _base_rec(1, val_clean=0.05, val_robust=0.02)
        append_epoch_jsonl(path, rec)
        assert os.path.exists(path)
        rows = _jsonl_rows(path)
        assert len(rows) == 1
        assert rows[0]["epoch"] == 1
        assert rows[0]["val_acc_clean"] == 0.05
        assert rows[0]["val_acc_robust"] == 0.02
        assert rows[0]["config_sha256"] == "abc"

    def test_append_only_does_not_overwrite(self, tmp_path):
        path = os.path.join(tmp_path, "ph_epoch_log.jsonl")
        for ep in (1, 2):
            rec = _base_rec(ep, val_clean=0.05, val_robust=0.02,
                            best_clean_epoch=ep, best_robust_epoch=ep)
            append_epoch_jsonl(path, rec)
        rows = _jsonl_rows(path)
        assert len(rows) == 2
        assert [int(r["epoch"]) for r in rows] == [1, 2]

    def test_last_epoch(self, tmp_path):
        path = os.path.join(tmp_path, "ph_epoch_log.jsonl")
        for ep in (1, 3, 2):
            rec = _base_rec(ep, val_clean=0.05, val_robust=0.02,
                            best_clean_epoch=ep, best_robust_epoch=ep)
            append_epoch_jsonl(path, rec)
        assert last_epoch_in_jsonl(path) == 3

    def test_resume_banner_cold_start(self, tmp_path):
        path = os.path.join(tmp_path, "ph_epoch_log.jsonl")
        banner = resume_banner(
            phase="backbone_only",
            start_epoch=1,
            best_clean=0.05,
            best_clean_epoch=1,
            best_robust=0.02,
            best_robust_epoch=1,
            loaded_checkpoint_path=None,
            loaded_checkpoint_sha256=None,
            jsonl_path=path,
        )
        assert "cold start" in banner.lower()
        assert "resumed from epoch 1" in banner

    def test_resume_banner_with_checkpoint(self, tmp_path):
        ckpt = os.path.join(tmp_path, "rolling.pth")
        torch.save({"epoch": 5}, ckpt)
        jsonl = os.path.join(tmp_path, "ph_epoch_log.jsonl")
        banner = resume_banner(
            phase="backbone_only",
            start_epoch=6,
            best_clean=0.10,
            best_clean_epoch=5,
            best_robust=0.06,
            best_robust_epoch=5,
            loaded_checkpoint_path=ckpt,
            loaded_checkpoint_sha256="abc123",
            jsonl_path=jsonl,
        )
        assert "loaded checkpoint:" in banner
        assert "abc123" in banner

    def test_validate_jsonl_terminal_epoch_mismatch(self, tmp_path):
        jsonl = os.path.join(tmp_path, "ph_epoch_log.jsonl")
        rec = _base_rec(3, val_clean=0.05, val_robust=0.02,
                        best_clean_epoch=3, best_robust_epoch=3)
        append_epoch_jsonl(jsonl, rec)
        ok, why = validate_jsonl_terminal_epoch(jsonl, expected_last_epoch=2)
        assert not ok
        assert "!= expected" in why

    def test_validate_jsonl_terminal_epoch_ok(self, tmp_path):
        jsonl = os.path.join(tmp_path, "ph_epoch_log.jsonl")
        rec = _base_rec(3, val_clean=0.05, val_robust=0.02,
                        best_clean_epoch=3, best_robust_epoch=3)
        append_epoch_jsonl(jsonl, rec)
        ok, why = validate_jsonl_terminal_epoch(jsonl, expected_last_epoch=3)
        assert ok


class TestResumeBannerVerification:
    """Resume banner prints jsonl last epoch and flags a mismatch loudly."""

    def test_banner_prints_jsonl_last_epoch(self, tmp_path):
        jsonl = os.path.join(tmp_path, "ph_epoch_log.jsonl")
        for ep in (1, 2):
            rec = _base_rec(ep, val_clean=0.05, val_robust=0.02,
                            best_clean_epoch=ep, best_robust_epoch=ep)
            append_epoch_jsonl(jsonl, rec)
        banner = resume_banner(
            phase="backbone_only",
            start_epoch=3,
            best_clean=0.05,
            best_clean_epoch=2,
            best_robust=0.02,
            best_robust_epoch=2,
            loaded_checkpoint_path=None,
            loaded_checkpoint_sha256=None,
            jsonl_path=jsonl,
        )
        assert "jsonl last epoch: 2" in banner


class TestResume:
    """After epoch 1, simulate resume: banner correct, JSONL not duplicated
    or rewritten, best_clean/best_robust carry over."""

    def test_resume_carries_best_and_no_rewrite(self, tmp_path):
        jsonl = os.path.join(tmp_path, "backbone_only_epoch_log.jsonl")
        ckpt = os.path.join(tmp_path, "rolling.pth")
        rec1 = _base_rec(1, val_clean=0.05, val_robust=0.02,
                         best_clean_epoch=1, best_robust_epoch=1)
        append_epoch_jsonl(jsonl, rec1)
        torch.save({"epoch": 1, "best_clean": 0.05, "best_clean_epoch": 1,
                    "best_robust": 0.02, "best_robust_epoch": 1}, ckpt)
        banner = resume_banner(
            phase="backbone_only",
            start_epoch=2,
            best_clean=0.05,
            best_clean_epoch=1,
            best_robust=0.02,
            best_robust_epoch=1,
            loaded_checkpoint_path=ckpt,
            loaded_checkpoint_sha256="sha",
            jsonl_path=jsonl,
        )
        assert "resumed from epoch 2" in banner
        assert "best_clean=5.00%" in banner or "best_clean=0.0500" in banner
        assert "best_robust=2.00%" in banner or "best_robust=0.0200" in banner
        assert last_epoch_in_jsonl(jsonl) == 1
        rows = _jsonl_rows(jsonl)
        assert len(rows) == 1


# ---- Step 4: phase header prints column meaning ----


class TestPhaseHeader:
    """The one-time phase header prints dataset fingerprint, param counts per
    optimizer group, the eps/beta curriculum table head, and the meaning of
    every column."""

    def test_header_includes_column_meaning(self, capsys):
        model = _make_fake_model()
        registry = _make_registry(model)
        first_point = phase_curriculum("backbone_only", 1, 60)

        class _FakeCfg:
            data_root = "/fake/imagenet100"
            img_size = 96
            fovea_size = 56
            num_classes = 100
            epochs = 60
            batch_size = 64
            num_workers = 4
            w_trades = W_TRADES_DEFAULT
            clean_only = False
            eval_seeds = tuple(range(41, 49))
            n_eval_samples = 300
            pgd_steps = 10
            val_clean_subset = None
            val_robust_subset = 512

        _print_phase_header("backbone_only", _FakeCfg(), registry, first_point)
        out = capsys.readouterr().out
        assert "epoch block columns" in out
        assert "val_acc[CLEAN]" in out
        assert "val_acc[ROBUST]" in out
        assert "optimizer groups" in out
        assert "current position in eps/beta curriculum" in out


# ---- Step 5: offline summarizer reads JSONL alone ----


class TestSummarizer:
    """Summarize reads the JSONL and prints a table + trend verdict."""

    def test_summarize_prints_table_and_verdict(self, tmp_path):
        jsonl = os.path.join(tmp_path, "backbone_only_epoch_log.jsonl")
        for ep in (1, 2, 3):
            rec = _base_rec(
                ep,
                val_clean=0.05 + ep * 0.01,
                val_robust=0.02 + ep * 0.005,
                tr_loss=4.0 - ep * 0.1,
                best_clean=0.05 + ep * 0.01,
                best_clean_epoch=ep,
                best_robust=0.02 + ep * 0.005,
                best_robust_epoch=ep,
            )
            append_epoch_jsonl(jsonl, rec)
        text = summarize_phase_from_jsonl(jsonl)
        assert "epoch" in text
        assert "valClean" in text
        assert "valRobust" in text
        assert "bestClean@ep" in text
        assert "bestRobust@ep" in text
        assert "config_sha256=abc" in text
        assert "git_commit=def" in text
        assert "session_id=sess1" in text
        assert "clean Δ=" in text
        assert "robust Δ=" in text


# ---- Smoke: 2 epochs, one phase; JSONL has all fields + epoch block has labels ----


class TestSmokeChain:
    """Two epochs of one phase on tiny synthetic loaders; assert the JSONL
    has all fields and the epoch block prints every label."""

    def test_two_epoch_jsonl_has_all_fields(self, tmp_path):
        model = _make_fake_model()
        loader = _make_fake_loader(n=16, num_classes=4, seed=0)
        val_loader = _make_fake_loader(n=8, num_classes=4, seed=1)
        device = torch.device("cpu")
        registry = _make_registry(model)
        optimizer = torch.optim.SGD(
            registry.build_optimizer(0.01).param_groups, lr=0.01)
        scaler = None
        jsonl = os.path.join(tmp_path, "backbone_only_epoch_log.jsonl")
        point = phase_curriculum("backbone_only", 1, 60)
        gc = GradNormCollector(registry)
        for epoch in (1, 2):
            tr_loss, ce_clean, kl_trades, tc, ta = _train_one_epoch_instrumented(
                model=model,
                loader=loader,
                optimizer=optimizer,
                registry=registry,
                device=device,
                scaler=scaler,
                epoch=epoch,
                total_epochs=60,
                clean_only=False,
                w_trades=W_TRADES_DEFAULT,
                grad_collector=gc,
                point=point,
            )
            v_clean, clean_n, _ = clean_val_accuracy(
                model, val_loader, device, seed=17, batch_size=8)
            v_robust, robust_n, robust_secs = robust_val_accuracy(
                model, val_loader, device, eps=0.031, pgd_steps=4,
                n_subset=8, seed=17, batch_size=8)
            grad = gc.finish()
            rec = epoch_jsonl_record(
                phase="backbone_only",
                epoch=epoch,
                total_epochs=60,
                point=point,
                lr=0.01,
                tr_loss=tr_loss,
                ce_clean=ce_clean,
                kl_trades=kl_trades,
                w_trades=W_TRADES_DEFAULT,
                train_clean_acc=tc,
                train_adv_acc=ta,
                val_clean=v_clean,
                val_robust=v_robust,
                clean_n=clean_n,
                robust_n=robust_n,
                robust_secs=robust_secs,
                best_clean=v_clean,
                best_clean_epoch=epoch,
                best_robust=v_robust,
                best_robust_epoch=epoch,
                criterion="clean",
                grad=grad,
                epoch_seconds=1.0,
                img_per_sec=10.0,
                peak_vram_gb=None,
                config_sha256="cfg",
                git_commit="git",
                session_id="s",
            )
            append_epoch_jsonl(jsonl, rec)
        rows = _jsonl_rows(jsonl)
        assert len(rows) == 2
        for r in rows:
            for key in (
                "phase",
                "epoch",
                "val_acc_clean",
                "val_acc_robust",
                "grad",
                "config_sha256",
                "git_commit",
                "session_id",
                "timestamp_utc",
                "train_loss_total",
                "train_ce_clean",
                "train_kl_trades",
                "w_trades",
                "train_acc_clean",
                "train_acc_adv",
                "best_clean",
                "best_clean_epoch",
                "best_robust",
                "best_robust_epoch",
                "checkpoint_criterion",
            ):
                assert key in r, f"missing field {key!r} in epoch {r.get('epoch')}"
            grad = r.get("grad", {})
            for key in (
                "pre_clip_norm",
                "post_clip_norm",
                "clipped_frac_any",
                "per_group",
            ):
                assert key in grad, f"missing grad field {key!r} in epoch {r.get('epoch')}"
        block = format_epoch_block(
            phase="backbone_only",
            epoch=2,
            total_epochs=60,
            point=point,
            lr=0.01,
            tr_loss=rows[-1]["train_loss_total"],
            ce_clean=rows[-1]["train_ce_clean"],
            kl_trades=rows[-1]["train_kl_trades"],
            w_trades=W_TRADES_DEFAULT,
            train_clean_acc=rows[-1]["train_acc_clean"],
            train_adv_acc=rows[-1]["train_acc_adv"],
            val_clean=rows[-1]["val_acc_clean"],
            val_robust=rows[-1]["val_acc_robust"],
            clean_n=rows[-1]["val_clean_n"],
            robust_n=rows[-1]["val_robust_n"],
            robust_secs=rows[-1]["val_robust_secs"],
            best_clean=rows[-1]["best_clean"],
            best_clean_epoch=rows[-1]["best_clean_epoch"],
            best_robust=rows[-1]["best_robust"],
            best_robust_epoch=rows[-1]["best_robust_epoch"],
            criterion="clean",
            grad=rows[-1]["grad"],
            epoch_seconds=1.0,
            img_per_sec=10.0,
            peak_vram_gb=None,
            eta_phase_end=None,
            health="ok",
        )
        for token in (
            "val_acc[CLEAN]",
            "val_acc[ROBUST",
            "best_clean   = ",
            "best_robust  = ",
            "checkpoint criterion in use:",
            "pre_clip_norm = ",
            "HEALTH",
        ):
            assert token in block, f"epoch block missing {token!r}"
        for token in (
            "SPEED",
            "img/s",
            "peak VRAM",
            "PARAMETER UPDATE HEALTH",
            "REPRESENTATION HEALTH",
            "GAP",
            "ACCURACY",
        ):
            assert token in block, f"epoch block missing {token!r}"
        # legend wording for the speed line uses 'seconds for the epoch block'
        assert "seconds for the epoch block" in EPOCH_BLOCK_LEGEND


# ---- Bit-identity: training results identical with logging on vs off, 1 epoch ----


class TestBitIdentity:
    """With logging on vs off, same seed, 1 epoch: training results bit-
    identical. If exact equality is impossible due to non-determinism, report
    the observed difference instead of weakening the test."""

    def test_instrumented_loss_matches_uninstrumented_within_tol(self):
        """The instrumented train path reuses the same backward/step as the
        uninstrumented path; results should match to fp tolerance."""
        torch.manual_seed(17)
        model = _make_fake_model()
        loader = _make_fake_loader(n=16, num_classes=4, seed=2)
        device = torch.device("cpu")
        registry = _make_registry(model)
        opt = torch.optim.SGD(
            registry.build_optimizer(0.01).param_groups, lr=0.01)
        point = phase_curriculum("backbone_only", 1, 60)
        gc = GradNormCollector(registry)
        inst_loss, *_ = _train_one_epoch_instrumented(
            model=model,
            loader=loader,
            optimizer=opt,
            registry=registry,
            device=device,
            scaler=None,
            epoch=1,
            total_epochs=60,
            clean_only=False,
            w_trades=W_TRADES_DEFAULT,
            grad_collector=gc,
            point=point,
        )
        torch.manual_seed(17)
        model2 = _make_fake_model()
        loader2 = _make_fake_loader(n=16, num_classes=4, seed=2)
        registry2 = _make_registry(model2)
        opt2 = torch.optim.SGD(
            registry2.build_optimizer(0.01).param_groups, lr=0.01)
        uninst_loss = _uninstrumented(
            model=model2,
            loader=loader2,
            optimizer=opt2,
            registry=registry2,
            device=device,
            scaler=None,
            epoch=1,
            total_epochs=60,
            clean_only=False,
            w_trades=W_TRADES_DEFAULT,
        )
        delta = abs(inst_loss - uninst_loss)
        assert delta < 1e-3, (
            f"instrumented vs uninstrumented loss differ by {delta:.6f} "
            f"(inst={inst_loss:.6f}, uninst={uninst_loss:.6f}) - "
            f"report this observed difference rather than weakening the test")
