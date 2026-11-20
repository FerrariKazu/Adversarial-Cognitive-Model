"""Subprocess entry points for sharded Stage 0A work."""
from __future__ import annotations

import argparse
import json

from stage0a.config import Stage0AConfig
from stage0a.features import extract_shard
from stage0a.utils import atomic_write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["extract"], required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--feature-batch-size", type=int, default=64)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--dataset-fingerprint", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--physical-gpu", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = Stage0AConfig(
        backbone=args.backbone,
        seed=args.seed,
        data_root=args.data_root,
        cache_root=args.cache_root,
        output_root=args.output_root,
        img_size=args.img_size,
        num_workers=args.num_workers,
        feature_batch_size=args.feature_batch_size,
        require_dual_gpu=False,
        allow_single_gpu=True,
    )
    result = extract_shard(
        cfg,
        index_dir=args.index_dir,
        split=args.split,
        dataset_fingerprint=args.dataset_fingerprint,
        rank=args.rank,
        n_shards=args.n_shards,
        physical_gpu=args.physical_gpu,
    )
    atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
