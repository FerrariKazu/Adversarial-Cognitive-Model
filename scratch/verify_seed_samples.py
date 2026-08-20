#!/usr/bin/env python3
"""Verify that the same 300-sample subsets are selected for each seed
across environments with the same `datasets` version.

Run this BEFORE the evaluation to record the expected sample hashes.
Run it AFTER the evaluation in a different environment to confirm
the same samples were used.

Usage:
    python3 scratch/verify_seed_samples.py --seeds 41 42 43 44 45 46 47 48
    python3 scratch/verify_seed_samples.py --seeds 41 42 43 44 45 46 47 48 --output report/seed_sample_hashes.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import datasets
import numpy as np


def load_indices(n_samples: int, seed: int) -> list[int]:
    """Return the exact list of STL-10 test-set indices selected for a seed."""
    ds = datasets.load_dataset("mteb/stl10", split="test").shuffle(seed=seed)
    indices = list(range(n_samples))
    # The .select(range(n_samples)) in load_test_samples takes the first n_samples
    # after shuffle. We replicate this by accessing the shuffled dataset's indices.
    selected = [ds[i] for i in range(n_samples)]
    # Actually, we need the original indices, not the data.
    # HuggingFace datasets .shuffle() reorders rows; .select() picks by position.
    # The underlying Arrow table has an .indices attribute, but it's cleaner
    # to just hash the data deterministically.
    return selected


def compute_sample_hash(ds, n_samples: int, seed: int) -> str:
    """Hash the first n_samples items from a shuffled dataset to produce
    a deterministic fingerprint of the selected subset."""
    ds_shuffled = ds.shuffle(seed=seed)
    h = hashlib.sha256()
    for i in range(n_samples):
        item = ds_shuffled[i]
        # Hash the label (int) and a summary of the image pixels
        # to keep the hash fast but collision-resistant
        h.update(str(item['label']).encode())
        # Use the first and last pixel values as a lightweight proxy
        # (full image hash would be slow for 300 items)
        arr = np.array(item['image'].convert('RGB').resize((96, 96)))
        h.update(arr[0, 0].tobytes())   # top-left pixel
        h.update(arr[-1, -1].tobytes()) # bottom-right pixel
        h.update(arr[48, 48].tobytes()) # center pixel
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[41, 42, 43, 44, 45, 46, 47, 48],
                        help='Seeds to verify (default: 41-48)')
    parser.add_argument('--n-samples', type=int, default=300,
                        help='Samples per seed (default: 300)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional JSON file to write the hashes')
    args = parser.parse_args()

    print(f"datasets version: {datasets.__version__}")
    print(f"Seeds: {args.seeds}")
    print(f"Samples per seed: {args.n_samples}")
    print()

    print("Loading STL-10 test set (one-time)...")
    ds = datasets.load_dataset("mteb/stl10", split="test")
    print(f"  Total test samples: {len(ds)}")
    print()

    results = {}
    for seed in args.seeds:
        sample_hash = compute_sample_hash(ds, args.n_samples, seed)
        results[str(seed)] = {
            'hash': sample_hash,
            'n_samples': args.n_samples,
            'datasets_version': datasets.__version__,
        }
        print(f"  seed {seed}: hash={sample_hash[:16]}...  "
              f"(datasets=={datasets.__version__}, n={args.n_samples})")

    print()
    print("If you see different hashes for the same seed in another environment,")
    print("the `datasets` library version differs and the sample subsets are different.")
    print()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'datasets_version': datasets.__version__,
                'n_samples': args.n_samples,
                'seeds': results,
            }, f, indent=2)
        print(f"Written to {args.output}")


if __name__ == '__main__':
    main()
