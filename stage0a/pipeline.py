"""Stage 0A orchestration used by the Kaggle/Colab runners and CLI."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from stage0a.benchmark import hardware_inventory, run_smoke
from stage0a.checkpointing import load_checkpoint
from stage0a.config import (
    BACKBONES,
    HF_BEST_REPO,
    HF_ROLLING_REPO,
    Stage0AConfig,
    configs_for_grid,
    estimate_feature_bytes,
)
from stage0a.data import (
    audit_dataset,
    audit_storage,
    build_index,
    load_index,
    save_index,
    split_fingerprint,
    split_summary,
    stratified_holdout,
)
from stage0a.evaluation import aggregate_seed_reports
from stage0a.features import (
    FeatureCache,
    cache_name,
    cache_summary,
    finalize_cache,
    load_complete_cache,
    prepare_cache,
    restore_cache,
    restore_shard,
    upload_cache,
)
from stage0a.hf_sync import NoopHFStore, HFStore
from stage0a.training import train_run
from stage0a.utils import (
    atomic_write_json,
    atomic_write_text,
    human_bytes,
    json_load,
    resolve_hf_token,
    training_fingerprint,
    utc_now,
)


def _path_layout(data_root: str, work_root: str) -> Dict[str, str]:
    work = Path(work_root)
    return {
        "data_root": str(Path(data_root).expanduser().resolve()),
        "work_root": str(work.resolve()),
        "output_root": str((work / "stage0a").resolve()),
        "cache_root": str((work / "stage0a_cache").resolve()),
        "index_root": str((work / "stage0a" / "index").resolve()),
    }


def prepare_indices(data_root: str, work_root: str, *, expected_val_count: int = 50_000) -> Dict[str, str]:
    layout = _path_layout(data_root, work_root)
    train = build_index(data_root, "train")
    val = build_index(data_root, "val", expected_val_count=expected_val_count)
    train_dir = Path(layout["index_root"]) / "train"
    val_dir = Path(layout["index_root"]) / "val"
    save_index(train, train_dir)
    save_index(val, val_dir)
    meta = {
        "train": train.summary(),
        "val": val.summary(),
        "created_at_utc": utc_now(),
    }
    # Do not duplicate 1000 class names into the report; they live in index.json.
    meta["train"].pop("class_names", None)
    meta["val"].pop("class_names", None)
    atomic_write_json(Path(layout["index_root"]) / "index_audit.json", meta)
    return {"train_index": str(train_dir), "val_index": str(val_dir), **layout}


def _build_split(index_dir: str, index: Any, seed: int, holdout_per_class: int, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_idx, holdout_idx = stratified_holdout(index.labels, holdout_per_class, seed)
    info = split_summary(index, train_idx, holdout_idx, seed)
    info["dataset_fingerprint"] = index.fingerprint
    info["split_fingerprint"] = split_fingerprint(index.fingerprint, train_idx, holdout_idx, seed)
    np.save(out_dir / "train_indices.npy", train_idx)
    np.save(out_dir / "holdout_indices.npy", holdout_idx)
    atomic_write_json(out_dir / "split.json", info)
    info["train_indices"] = train_idx
    info["holdout_indices"] = holdout_idx
    return info


def load_split(out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    info = json_load(out_dir / "split.json")
    info["train_indices"] = np.load(out_dir / "train_indices.npy")
    info["holdout_indices"] = np.load(out_dir / "holdout_indices.npy")
    return info


def _gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _launch_extract_worker(
    cfg: Stage0AConfig,
    *,
    index_dir: str,
    split: str,
    dataset_fingerprint: str,
    rank: int,
    n_shards: int,
    gpu: int,
    out: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "stage0a.workers",
        "--mode",
        "extract",
        "--backbone",
        cfg.backbone,
        "--seed",
        str(cfg.seed),
        "--data-root",
        cfg.data_root,
        "--cache-root",
        cfg.cache_root,
        "--output-root",
        cfg.output_root,
        "--img-size",
        str(cfg.img_size),
        "--num-workers",
        str(cfg.num_workers),
        "--feature-batch-size",
        str(cfg.feature_batch_size),
        "--index-dir",
        index_dir,
        "--split",
        split,
        "--dataset-fingerprint",
        dataset_fingerprint,
        "--rank",
        str(rank),
        "--n-shards",
        str(n_shards),
        "--physical-gpu",
        str(gpu),
        "--out",
        str(out),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["STAGE0A_PHYSICAL_GPU"] = str(gpu)
    return subprocess.Popen(cmd, env=env)


def extract_split(
    cfg: Stage0AConfig,
    *,
    index_dir: str,
    split: str,
    hf_store: Any,
    reuse_remote: bool = True,
) -> FeatureCache:
    index = load_index(index_dir)
    # A verified local cache is already durable; do not re-extract merely
    # because this is a new process.  If it is corrupt, fall through and
    # rebuild rather than silently using it.
    try:
        return load_complete_cache(cfg, split, index.fingerprint)
    except Exception:
        pass
    if reuse_remote and getattr(hf_store, "enabled", False):
        cached = restore_cache(cfg, split, index.fingerprint, hf_store)
        if cached is not None:
            return cached
    n_shards = max(1, min(2, _gpu_count() or 1))
    cache = prepare_cache(cfg, index, split, n_shards=n_shards)
    out_dir = cache.directory

    missing_ranks = []
    for rank in range(n_shards):
        restored = restore_shard(
            cfg,
            split=split,
            dataset_fingerprint=index.fingerprint,
            rank=rank,
            n_shards=n_shards,
            total_count=len(index),
            hf_store=hf_store if reuse_remote else None,
        )
        if restored is None:
            missing_ranks.append(rank)

    if missing_ranks:
        procs: List[tuple[int, subprocess.Popen]] = []
        for rank in missing_ranks:
            gpu = rank % max(1, _gpu_count())
            out = out_dir / f"worker_{rank:02d}.log"
            procs.append((rank, _launch_extract_worker(
                cfg,
                index_dir=index_dir,
                split=split,
                dataset_fingerprint=index.fingerprint,
                rank=rank,
                n_shards=n_shards,
                gpu=gpu,
                out=out,
            )))
        for rank, p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(
                    f"feature extraction shard {rank} failed rc={rc}; see {out_dir}/worker_{rank:02d}.log"
                )

    finalize_cache(cfg, index, split, n_shards)
    cache = load_complete_cache(cfg, split, index.fingerprint)
    if getattr(hf_store, "enabled", False):
        upload_cache(cache, hf_store)
    return cache


def sync_provenance_artifacts(
    work_root: str,
    hf_store: Any,
    *,
    required: bool = False,
) -> Dict[str, str]:
    """Upload index_audit.json, split.json, stage0a_audit.json, stage0a_audit.md
    with SHA-256 sidecars to store.best_repo under stage0a/provenance/.
    """
    if not getattr(hf_store, "enabled", False):
        return {}
    layout = _path_layout("", work_root)
    artifacts = [
        ("index_audit.json", Path(layout["index_root"]) / "index_audit.json"),
        ("split.json", Path(layout["output_root"]) / "split" / "split.json"),
        ("stage0a_audit.json", Path(layout["output_root"]) / "stage0a_audit.json"),
        ("stage0a_audit.md", Path(layout["output_root"]) / "stage0a_audit.md"),
    ]
    synced = {}
    for name, local_path in artifacts:
        if local_path.exists():
            remote = f"stage0a/provenance/{name}"
            hf_store.upload_verified(local_path, hf_store.best_repo, remote, required=required)
            synced[name] = remote
    return synced


def _training_fingerprint() -> str:
    return training_fingerprint(Path(__file__).resolve().parents[1])


def _hardware_for_run() -> Dict[str, Any]:
    inv = hardware_inventory()
    return {
        "device_count": inv["device_count"],
        "devices": inv["devices"],
        "torch": inv["torch"],
        "cuda_version": inv["cuda_version"],
        "cpu_count": inv["cpu_count"],
        "ram_total_gb": inv["ram_total_gb"],
        "precision": "amp_float16",
    }


def run_one_training(
    cfg: Stage0AConfig,
    *,
    train_cache: FeatureCache,
    val_cache: FeatureCache,
    split_info: Dict[str, Any],
    hf_store: Any,
    force_fresh: bool = False,
    confirm_overwrite_remote: bool = False,
    max_new_steps: Optional[int] = None,
) -> Dict[str, Any]:
    backbone_sha = train_cache.meta.get("backbone_weights_sha256")
    if not backbone_sha:
        raise RuntimeError("feature cache has no backbone weight fingerprint; rebuild the cache")
    return train_run(
        cfg,
        train_cache=train_cache,
        val_cache=val_cache,
        index_dir="",
        split_info=split_info,
        training_fingerprint=_training_fingerprint(),
        backbone_weights_sha256=str(backbone_sha),
        hardware=_hardware_for_run(),
        hf_store=hf_store,
        force_fresh=force_fresh,
        confirm_overwrite_remote=confirm_overwrite_remote,
        max_new_steps=max_new_steps,
    )


def run_seed_grid(
    *,
    data_root: str,
    work_root: str,
    hf_store: Any,
    backbones: Iterable[str] = tuple(BACKBONES),
    gpus: int = 2,
    img_size: int = 224,
    num_workers: int = 4,
    feature_batch_size: int = 64,
    holdout_per_class: int = 50,
    force_fresh: bool = False,
    confirm_overwrite_remote: bool = False,
) -> Dict[str, Any]:
    layout = _path_layout(data_root, work_root)
    train_index_dir = Path(layout["index_root"]) / "train"
    val_index_dir = Path(layout["index_root"]) / "val"
    index = load_index(train_index_dir)
    split_dir = Path(layout["output_root"]) / "split"
    split_info = _build_split(train_index_dir, index, seed=17, holdout_per_class=holdout_per_class, out_dir=split_dir)
    train_cache = load_complete_cache(
        Stage0AConfig(
            backbone="vit-s14", seed=17, data_root=data_root,
            cache_root=layout["cache_root"], output_root=layout["output_root"],
            img_size=img_size, num_workers=num_workers, feature_batch_size=feature_batch_size,
        ),
        "train", index.fingerprint,
    )
    val_index = load_index(val_index_dir)
    val_cache = load_complete_cache(
        Stage0AConfig(
            backbone="vit-s14", seed=17, data_root=data_root,
            cache_root=layout["cache_root"], output_root=layout["output_root"],
            img_size=img_size, num_workers=num_workers, feature_batch_size=feature_batch_size,
        ),
        "val", val_index.fingerprint,
    )

    jobs = []
    for backbone in backbones:
        for seed in (17, 42, 99):
            cfg = Stage0AConfig(
                backbone=backbone,
                seed=seed,
                data_root=data_root,
                cache_root=layout["cache_root"],
                output_root=layout["output_root"],
                img_size=img_size,
                num_workers=num_workers,
                feature_batch_size=feature_batch_size,
            )
            jobs.append(cfg)

    results = []
    active: List[tuple[Any, subprocess.Popen, Path, int]] = []
    free_slots = list(range(max(1, min(gpus, _gpu_count() or 1))))
    n_slots = len(free_slots)

    def launch(cfg: Stage0AConfig, slot: int) -> None:
        out = Path(layout["output_root"]) / "runs" / cfg.backbone / f"seed{cfg.seed}" / "train_result.json"
        cmd = [
            sys.executable, "-m", "stage0a.cli", "train",
            "--backbone", cfg.backbone, "--seed", str(cfg.seed),
            "--data-root", data_root, "--work-root", work_root,
            "--gpus", str(n_slots),
            "--img-size", str(img_size),
            "--num-workers", str(num_workers),
            "--feature-batch-size", str(feature_batch_size),
        ]
        if isinstance(hf_store, NoopHFStore):
            cmd.append("--no-hf")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(slot)
        if force_fresh:
            cmd.append("--fresh")
        if confirm_overwrite_remote:
            cmd.append("--confirm-overwrite-remote")
        active.append((cfg, subprocess.Popen(cmd, env=env), out, slot))

    pending = list(jobs)
    while pending or active:
        while pending and free_slots:
            launch(pending.pop(0), free_slots.pop(0))
        if active:
            _drain_one(active, results, free_slots)
    return {"status": "ok", "jobs": results, "gpus_used": n_slots}


def _drain_one(
    active: List[tuple[Any, subprocess.Popen, Path, int]],
    results: List[Dict[str, Any]],
    free_slots: List[int],
) -> None:
    cfg, p, out, slot = active.pop(0)
    rc = p.wait()
    free_slots.append(slot)
    free_slots.sort()
    if rc != 0:
        raise RuntimeError(f"training run {cfg.run_id} failed rc={rc}")
    if not out.exists():
        raise RuntimeError(f"training run {cfg.run_id} produced no train_result.json")
    results.append(json_load(out))


def aggregate(
    *,
    data_root: str,
    work_root: str,
    hf_store: Any,
    target_top1: float = 0.80,
    seed_floor_top1: float = 0.79,
) -> Dict[str, Any]:
    layout = _path_layout(data_root, work_root)
    reports: Dict[str, List[Dict[str, Any]]] = {}
    for backbone in BACKBONES:
        reports[backbone] = []
        for seed in (17, 42, 99):
            cfg = Stage0AConfig(
                backbone=backbone,
                seed=seed,
                data_root=data_root,
                cache_root=layout["cache_root"],
                output_root=layout["output_root"],
            )
            run_dir = Path(layout["output_root"]) / "runs" / backbone / f"seed{seed}"
            final = run_dir / "final.pt"
            if not final.exists() and getattr(hf_store, "enabled", False):
                hf_store.download_verified(
                    hf_store.best_repo,
                    hf_store.remote_name(cfg.run_id, "final.pt"),
                    final,
                )
            state = load_checkpoint(final)
            metric = dict(state.get("metric") or {})
            metric["seed"] = seed
            metric["backbone"] = backbone
            reports[backbone].append(metric)
    out = {
        "protocol_id": "rhan-stage0a-dinov2/1",
        "created_at_utc": utc_now(),
        "results": {
            backbone: aggregate_seed_reports(
                reports[backbone],
                backbone=backbone,
                target_top1=target_top1,
                seed_floor_top1=seed_floor_top1,
            )
            for backbone in BACKBONES
        },
    }
    out["verdict"] = "PASS" if all(r["verdict"] == "PASS" for r in out["results"].values()) else "FAIL"
    atomic_write_json(Path(layout["output_root"]) / "stage0a_aggregate.json", out)
    return out


def write_audit_report(
    *,
    data_root: str,
    work_root: str,
    smoke_report: Optional[Dict[str, Any]] = None,
    resume_report: Optional[Dict[str, Any]] = None,
) -> Path:
    layout = _path_layout(data_root, work_root)
    inv = hardware_inventory()
    index_audit_p = Path(layout["index_root"]) / "index_audit.json"
    if index_audit_p.exists():
        try:
            ia = json_load(index_audit_p)
            train_info = ia.get("train", {})
            val_info = ia.get("val", {})
            ds = {
                "requested_root": data_root,
                "train_dir": str(Path(data_root) / "train"),
                "val_dir": str(Path(data_root) / "val"),
                "train_class_dirs": train_info.get("num_classes", 1000),
                "val_class_dirs": val_info.get("num_classes", 1000),
                "train_images": train_info.get("count", 1281167),
                "val_images": val_info.get("count", 50000),
                "official_val_ready": val_info.get("count") == 50000,
                "status": "ok",
            }
        except Exception:
            ds = audit_dataset(data_root)
    else:
        ds = audit_dataset(data_root)
    storage = audit_storage([layout["work_root"], data_root, "/kaggle/input", "/kaggle/working"])
    feature_estimates = {}
    artifact_estimates = {"per_run": {}, "six_runs": {}}
    total_feature_bytes = 0
    total_head_bytes = 0
    for b in BACKBONES:
        dim = int(BACKBONES[b]["embed_dim"])
        train_b = estimate_feature_bytes(b, 1_281_167, 2)
        val_b = estimate_feature_bytes(b, 50_000, 2)
        head_b = dim * 1000 * 4
        total_feature_bytes += train_b + val_b
        total_head_bytes += 3 * head_b  # 3 seeds, one head each
        feature_estimates[b] = {
            "train_fp16": human_bytes(train_b),
            "val_fp16": human_bytes(val_b),
            "train_fp32_if_used": human_bytes(estimate_feature_bytes(b, 1_281_167, 4)),
            "patch_tokens_not_cached": True,
        }
        artifact_estimates["per_run"][b] = {
            "head_parameters": human_bytes(head_b),
            "best_or_final_checkpoint": human_bytes(head_b),
            "rolling_checkpoint_with_optimizer": human_bytes(head_b * 2),
            "predictions_and_csv": "< 1 MiB per run",
        }
    artifact_estimates["six_runs"]["pooled_features"] = human_bytes(total_feature_bytes)
    artifact_estimates["six_runs"]["head_artifacts"] = human_bytes(total_head_bytes * 4)
    artifact_estimates["six_runs"]["working_peak_estimate"] = human_bytes(total_feature_bytes + total_head_bytes * 4)
    configs = [c.to_dict(include_runtime=False) for c in configs_for_grid(data_root, layout["cache_root"], layout["output_root"])]
    smoke_pass = bool((smoke_report or {}).get("both_gpus_participated"))
    resume_pass = (resume_report or {}).get("status") == "PASS"
    report = {
        "protocol_id": "rhan-stage0a-dinov2/1",
        "created_at_utc": utc_now(),
        "reference_notebooks": {
            "kaggle": "cloud_setup/Kaggle_consolidated.py",
            "colab": "cloud_setup/colab_consolidated.py",
            "kaggle_foundation": "cloud_setup/Kaggle_J1_FOUNDATION.py",
            "colab_foundation": "cloud_setup/colab_j1_foundation.py",
        },
        "hardware": inv,
        "dataset": ds,
        "storage": storage,
        "feature_estimates": feature_estimates,
        "artifact_estimates": artifact_estimates,
        "configs": configs,
        "hf": {
            "best_repo": HF_BEST_REPO,
            "rolling_repo": HF_ROLLING_REPO,
            "token_present": bool(resolve_hf_token()),
            "isolated_from_previous_namespaces": True,
        },
        "smoke_report": smoke_report,
        "resume_report": resume_report,
        "status": (
            "blocked_dataset"
            if ds.get("status") != "ok"
            else (
                "ready_for_training" if (smoke_pass and resume_pass) else "gates_pending"
            )
        ),
    }
    out_dir = Path(layout["output_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "stage0a_audit.json", report)
    md = _audit_markdown(report)
    atomic_write_text(out_dir / "stage0a_audit.md", md)
    return out_dir / "stage0a_audit.md"


def _audit_markdown(r: Dict[str, Any]) -> str:
    ds = r["dataset"]
    hw = r["hardware"]
    lines = [
        "# Stage 0A audit — frozen DINOv2-S/14 and B/14 linear probes",
        "",
        f"Protocol: `{r['protocol_id']}`  ",
        f"Created: `{r['created_at_utc']}`  ",
        f"Status: **{r['status']}**",
        "",
        "## 1. Canonical reference files",
        "",
        "| Role | Path |",
        "|---|---|",
        "| Kaggle consolidated runner | `cloud_setup/Kaggle_consolidated.py` |",
        "| Colab consolidated runner | `cloud_setup/colab_consolidated.py` |",
        "| Kaggle foundation runner | `cloud_setup/Kaggle_J1_FOUNDATION.py` |",
        "| Colab foundation runner | `cloud_setup/colab_j1_foundation.py` |",
        "",
        "Reuse: the Stage 0A launchers keep the same environment/secret bootstrap, dedicated HF repos, per-epoch/per-step sync, and automatic resume semantics. The Stage 0A checkpoint adds RNG/AMP/sampler state required by the assignment.",
        "",
        "## 2. Dataset availability",
        "",
        f"- Requested root: `{ds.get('requested_root')}`",
        f"- Status: `{ds.get('status')}`",
        f"- Train dir: `{ds.get('train_dir', 'missing')}`",
        f"- Val dir: `{ds.get('val_dir', 'missing')}`",
        f"- Train images: `{ds.get('train_images', 'unknown')}`",
        f"- Val images: `{ds.get('val_images', 'unknown')}`",
        f"- Official val ready: `{ds.get('official_val_ready', False)}`",
    ]
    if ds.get("error"):
        lines += ["", f"Error: `{ds['error']}`"]
    lines += [
        "",
        "## 3. Hardware",
        "",
        f"- Visible GPUs: `{hw.get('device_count')}`",
        f"- CUDA: `{hw.get('cuda_version')}`  torch: `{hw.get('torch')}`",
        f"- CPU count: `{hw.get('cpu_count')}`  RAM: `{hw.get('ram_total_gb')} GB`",
        f"- Kaggle runtime: `{hw.get('is_kaggle')}`",
    ]
    for d in hw.get("devices", []):
        lines.append(f"- GPU {d['index']}: {d['name']}, {d['total_memory_gb']} GB")
    lines += [
        "",
        "## 4. Storage",
        "",
        "| Path | Exists | Total | Free |",
        "|---|---|---|---|",
    ]
    for p, rec in r["storage"].items():
        lines.append(f"| `{p}` | `{rec.get('exists')}` | {rec.get('total', '')} | {rec.get('free_human', '')} |")
    lines += [
        "",
        "## 5. Feature storage estimate",
        "",
        "Only pooled CLS features are cached; full spatial patch-token maps are not persisted.",
        "",
        "| Backbone | Train fp16 | Val fp16 |",
        "|---|---|---|",
    ]
    for b, e in r["feature_estimates"].items():
        lines.append(f"| {b} | {e['train_fp16']} | {e['val_fp16']} |")
    a = r["artifact_estimates"]["six_runs"]
    lines += [
        "",
        f"Six-run durable footprint: pooled features `{a['pooled_features']}`, "
        f"head artifacts `{a['head_artifacts']}`, working peak `{a['working_peak_estimate']}`.",
        "",
        "## 6. Locked configuration",
        "",
        "| Backbone | Seed | Image size | Head steps | Batch | LR | Target |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in r["configs"]:
        lines.append(
            f"| {c['backbone']} | {c['seed']} | {c['img_size']} | {c['max_steps']} | "
            f"{c['head_batch_size']} | {c['lr']} | {c['target_top1']:.2f} |"
        )
    lines += [
        "",
        "## 7. Smoke and resume gates",
        "",
        f"- Dual-GPU smoke: `{'PASS' if (r.get('smoke_report') or {}).get('both_gpus_participated') else ('present but not dual-GPU' if r.get('smoke_report') else 'not run')}`",
        f"- Resume drill: `{('PASS' if (r.get('resume_report') or {}).get('status') == 'PASS' else 'present but failed') if r.get('resume_report') else 'not run'}`",
        "",
        "Full training must not start until the smoke report shows both GPUs participating and the resume drill passes.",
        "",
    ]
    return "\n".join(lines)


def run_resume_drill(work_root: str, backbone: str = "vit-s14", seed: int = 17) -> Dict[str, Any]:
    """Deterministic local-only checkpoint interruption/resume test.

    This is a *state-machine* test, not a substitute for a Kaggle smoke run.
    """
    from stage0a.features import FeatureCache
    from stage0a.utils import sha256_file

    out_root = Path(work_root) / "stage0a" / "stage0a_audit" / "resume_drill"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = Stage0AConfig(
        backbone=backbone,
        seed=seed,
        data_root="/nonexistent",
        cache_root=str(out_root / "cache"),
        output_root=str(out_root),
        img_size=224,
        max_steps=20,
        head_batch_size=128,
        eval_every_steps=10,
        checkpoint_every_steps=5,
        holdout_per_class=50,
        require_dual_gpu=False,
        allow_single_gpu=True,
    )
    rng = np.random.default_rng(7)
    class_names = [f"c{i}" for i in range(1000)]

    def make_cache(split: str, n: int, classes: int) -> FeatureCache:
        d = out_root / "cache" / "features" / f"drill_{split}"
        d.mkdir(parents=True, exist_ok=True)
        feats = rng.standard_normal((n, cfg.embed_dim)).astype(np.float16)
        labels = np.arange(n, dtype=np.int64) % classes
        np.save(d / "features.npy", feats)
        np.save(d / "labels.npy", labels)
        meta = {
            "complete": True,
            "format_version": 1,
            "backbone": backbone,
            "backbone_timm_name": cfg.backbone_info["timm_name"],
            "split": split,
            "img_size": 224,
            "dim": cfg.embed_dim,
            "count": n,
            "dtype": "float16",
            "representation": "pooled_cls_token",
            "dataset_fingerprint": f"drill-{split}",
            "class_names": class_names,
            "features_sha256": sha256_file(d / "features.npy"),
            "labels_sha256": sha256_file(d / "labels.npy"),
            "backbone_weights_sha256": "drill-weights",
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        return FeatureCache(directory=d, features=d / "features.npy", labels=d / "labels.npy", meta=meta)

    train_cache = make_cache("train", 2000, 1000)
    val_cache = make_cache("val", 1000, 1000)
    labels = np.load(train_cache.labels)
    train_idx, holdout_idx = stratified_holdout(labels, 1, seed)
    split_info = {
        "dataset_fingerprint": "drill-train",
        "split_fingerprint": "drill-split",
        "train_indices": train_idx,
        "holdout_indices": holdout_idx,
    }
    store = NoopHFStore()
    first = train_run(
        cfg,
        train_cache=train_cache,
        val_cache=val_cache,
        index_dir="",
        split_info=split_info,
        training_fingerprint=training_fingerprint(Path(__file__).resolve().parents[1]),
        backbone_weights_sha256="drill-weights",
        hardware={"device_count": _gpu_count(), "precision": "amp_float16", "drill": True},
        hf_store=store,
        max_new_steps=7,
    )
    if first.get("status") != "partial" or first.get("global_step") != 7:
        raise RuntimeError(f"resume drill phase 1 did not stop at step 7: {first}")
    second = train_run(
        cfg,
        train_cache=train_cache,
        val_cache=val_cache,
        index_dir="",
        split_info=split_info,
        training_fingerprint=training_fingerprint(Path(__file__).resolve().parents[1]),
        backbone_weights_sha256="drill-weights",
        hardware={"device_count": _gpu_count(), "precision": "amp_float16", "drill": True},
        hf_store=store,
    )
    if second.get("status") != "complete":
        raise RuntimeError(f"resume drill phase 2 did not complete: {second}")
    log = (Path(cfg.output_root) / "runs" / backbone / f"seed{seed}" / "metrics.jsonl").read_text()
    if '"event": "resume"' not in log:
        raise RuntimeError("resume drill did not log a resume event")
    report = {
        "status": "PASS",
        "phase1": first,
        "phase2": second,
        "resume_event_present": True,
        "note": "local state-machine drill; real dual-T4 smoke still required",
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(out_root / "resume_drill.json", report)
    return report
