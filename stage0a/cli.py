"""Command-line interface for Stage 0A.

Examples (Kaggle/Colab):
  python -m stage0a.cli prepare --data-root /kaggle/input/imagenet-1k --work-root /kaggle/working
  python -m stage0a.cli audit   --data-root /kaggle/input/imagenet-1k --work-root /kaggle/working
  python -m stage0a.cli smoke   --data-root /kaggle/input/imagenet-1k --work-root /kaggle/working
  python -m stage0a.cli extract --data-root /kaggle/input/imagenet-1k --work-root /kaggle/working
  python -m stage0a.cli train   --data-root /kaggle/input/imagenet-1k --work-root /kaggle/working --backbone vit-s14 --seed 17
  python -m stage0a.cli heads   --data-root /kaggle/input/imagenet-1k --work-root /kaggle/working
  python -m stage0a.cli aggregate --data-root /kaggle/input/imagenet-1k --work-root /kaggle/working
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from stage0a.config import BACKBONES, Stage0AConfig
from stage0a.hf_sync import HFStore
from stage0a.pipeline import (
    _hardware_for_run,
    _path_layout,
    aggregate,
    extract_split,
    load_split,
    prepare_indices,
    run_one_training,
    run_resume_drill,
    run_seed_grid,
    run_smoke,
    write_audit_report,
)
from stage0a.utils import atomic_write_json, json_load, resolve_hf_token


def _hf(no_hf: bool) -> Any:
    if no_hf:
        from stage0a.hf_sync import NoopHFStore

        return NoopHFStore()
    return HFStore(token=resolve_hf_token())


def _cfg(args: argparse.Namespace) -> Stage0AConfig:
    layout = _path_layout(args.data_root, args.work_root)
    return Stage0AConfig(
        backbone=args.backbone,
        seed=args.seed,
        data_root=args.data_root,
        cache_root=layout["cache_root"],
        output_root=layout["output_root"],
        img_size=args.img_size,
        num_workers=args.num_workers,
        feature_batch_size=args.feature_batch_size,
    )


def cmd_prepare(args: argparse.Namespace) -> int:
    out = prepare_indices(args.data_root, args.work_root)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    layout = _path_layout(args.data_root, args.work_root)
    # Prepare indices if they are absent.  This is also the dataset audit gate.
    if not (Path(layout["index_root"]) / "train" / "index.json").exists():
        prepare_indices(args.data_root, args.work_root)
    smoke_path = Path(layout["output_root"]) / "smoke" / "smoke_report.json"
    resume_path = Path(layout["output_root"]) / "stage0a_audit" / "resume_drill" / "resume_drill.json"
    smoke = json_load(smoke_path) if smoke_path.exists() else None
    resume = json_load(resume_path) if resume_path.exists() else None
    path = write_audit_report(
        data_root=args.data_root,
        work_root=args.work_root,
        smoke_report=smoke,
        resume_report=resume,
    )
    print(f"wrote {path}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    layout = _path_layout(args.data_root, args.work_root)
    index_dir = Path(layout["index_root"]) / "train"
    if not (index_dir / "index.json").exists():
        prepare_indices(args.data_root, args.work_root)
    report = run_smoke(
        index_dir=str(index_dir),
        data_root=args.data_root,
        cache_root=layout["cache_root"],
        output_root=layout["output_root"],
        img_size=args.img_size,
        batch_size=args.feature_batch_size,
        n_images=args.smoke_images,
        max_batches=args.smoke_batches,
        num_workers=args.num_workers,
        require_dual_gpu=not args.allow_single_gpu,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    layout = _path_layout(args.data_root, args.work_root)
    index_dir = Path(layout["index_root"]) / "train"
    val_dir = Path(layout["index_root"]) / "val"
    if not (index_dir / "index.json").exists():
        prepare_indices(args.data_root, args.work_root)
    store = _hf(args.no_hf)
    backbones = list(BACKBONES) if args.backbone == "all" else [args.backbone]
    for backbone in backbones:
        cfg = Stage0AConfig(
            backbone=backbone,
            seed=args.seed,
            data_root=args.data_root,
            cache_root=layout["cache_root"],
            output_root=layout["output_root"],
            img_size=args.img_size,
            num_workers=args.num_workers,
            feature_batch_size=args.feature_batch_size,
        )
        tr = extract_split(cfg, index_dir=str(index_dir), split="train", hf_store=store, reuse_remote=not args.no_reuse)
        va = extract_split(cfg, index_dir=str(val_dir), split="val", hf_store=store, reuse_remote=not args.no_reuse)
        print(json.dumps({"backbone": backbone, "train": tr.meta.get("count"), "val": va.meta.get("count")}, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    layout = _path_layout(args.data_root, args.work_root)
    cfg = _cfg(args)
    train_index_dir = Path(layout["index_root"]) / "train"
    val_index_dir = Path(layout["index_root"]) / "val"
    split_dir = Path(layout["output_root"]) / "split"
    if not (split_dir / "split.json").exists():
        raise SystemExit("split missing; run `python -m stage0a.cli prepare` first")
    split_info = load_split(split_dir)
    from stage0a.features import load_complete_cache
    from stage0a.data import load_index

    train_index = load_index(train_index_dir)
    val_index = load_index(val_index_dir)
    train_cache = load_complete_cache(cfg, "train", train_index.fingerprint)
    val_cache = load_complete_cache(cfg, "val", val_index.fingerprint)
    result = run_one_training(
        cfg,
        train_cache=train_cache,
        val_cache=val_cache,
        split_info=split_info,
        hf_store=_hf(args.no_hf),
        force_fresh=args.fresh,
        confirm_overwrite_remote=args.confirm_overwrite_remote,
        max_new_steps=args.max_new_steps,
    )
    out = Path(layout["output_root"]) / "runs" / cfg.backbone / f"seed{cfg.seed}" / "train_result.json"
    atomic_write_json(out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in ("complete", "already_complete", "partial") else 1


def cmd_heads(args: argparse.Namespace) -> int:
    result = run_seed_grid(
        data_root=args.data_root,
        work_root=args.work_root,
        hf_store=_hf(args.no_hf),
        backbones=BACKBONES if args.backbone == "all" else [args.backbone],
        gpus=args.gpus,
        img_size=args.img_size,
        num_workers=args.num_workers,
        feature_batch_size=args.feature_batch_size,
        force_fresh=args.fresh,
        confirm_overwrite_remote=args.confirm_overwrite_remote,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    out = aggregate(data_root=args.data_root, work_root=args.work_root, hf_store=_hf(args.no_hf))
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out.get("verdict") == "PASS" else 1


def cmd_resume_drill(args: argparse.Namespace) -> int:
    out = run_resume_drill(args.work_root)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out.get("status") == "PASS" else 1


def cmd_all(args: argparse.Namespace) -> int:
    # The gates are deliberately ordered: no full extraction or training
    # before the dataset/hardware/resume checks.
    prepare_indices(args.data_root, args.work_root)
    resume = run_resume_drill(args.work_root)
    write_audit_report(data_root=args.data_root, work_root=args.work_root, resume_report=resume)
    layout = _path_layout(args.data_root, args.work_root)
    smoke = run_smoke(
        index_dir=str(Path(layout["index_root"]) / "train"),
        data_root=args.data_root,
        cache_root=layout["cache_root"],
        output_root=layout["output_root"],
        img_size=args.img_size,
        batch_size=args.feature_batch_size,
        n_images=args.smoke_images,
        max_batches=args.smoke_batches,
        num_workers=args.num_workers,
        require_dual_gpu=True,
    )
    write_audit_report(data_root=args.data_root, work_root=args.work_root, smoke_report=smoke, resume_report=resume)
    cmd_extract(args)
    cmd_heads(args)
    return cmd_aggregate(args)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", required=True)
    p.add_argument("--work-root", required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--feature-batch-size", type=int, default=64)
    p.add_argument("--no-hf", action="store_true", help="local-only; tests/drills only")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("prepare")
    _add_common(q)
    q.set_defaults(func=cmd_prepare)

    q = sub.add_parser("audit")
    _add_common(q)
    q.set_defaults(func=cmd_audit)

    q = sub.add_parser("smoke")
    _add_common(q)
    q.add_argument("--smoke-images", type=int, default=512)
    q.add_argument("--smoke-batches", type=int, default=4)
    q.add_argument("--allow-single-gpu", action="store_true")
    q.set_defaults(func=cmd_smoke)

    q = sub.add_parser("extract")
    _add_common(q)
    q.add_argument("--backbone", choices=[*BACKBONES, "all"], default="all")
    q.add_argument("--seed", type=int, default=17)
    q.add_argument("--no-reuse", action="store_true")
    q.set_defaults(func=cmd_extract)

    q = sub.add_parser("train")
    _add_common(q)
    q.add_argument("--backbone", choices=list(BACKBONES), required=True)
    q.add_argument("--seed", type=int, choices=[17, 42, 99], required=True)
    q.add_argument("--fresh", action="store_true")
    q.add_argument("--confirm-overwrite-remote", action="store_true")
    q.add_argument("--max-new-steps", type=int, default=None)
    q.add_argument("--gpus", type=int, default=1)
    q.set_defaults(func=cmd_train)

    q = sub.add_parser("heads")
    _add_common(q)
    q.add_argument("--backbone", choices=[*BACKBONES, "all"], default="all")
    q.add_argument("--gpus", type=int, default=2)
    q.add_argument("--fresh", action="store_true")
    q.add_argument("--confirm-overwrite-remote", action="store_true")
    q.set_defaults(func=cmd_heads)

    q = sub.add_parser("aggregate")
    _add_common(q)
    q.set_defaults(func=cmd_aggregate)

    q = sub.add_parser("resume-drill")
    _add_common(q)
    q.set_defaults(func=cmd_resume_drill)

    q = sub.add_parser("all")
    _add_common(q)
    q.add_argument("--smoke-images", type=int, default=512)
    q.add_argument("--smoke-batches", type=int, default=4)
    q.add_argument("--gpus", type=int, default=2)
    q.add_argument("--backbone", choices=[*BACKBONES, "all"], default="all")
    q.add_argument("--seed", type=int, default=17)
    q.add_argument("--no-reuse", action="store_true")
    q.add_argument("--fresh", action="store_true")
    q.add_argument("--confirm-overwrite-remote", action="store_true")
    q.set_defaults(func=cmd_all)

    args = p.parse_args()
    try:
        rc = args.func(args)
    except Exception as e:
        print(f"STAGE0A ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
