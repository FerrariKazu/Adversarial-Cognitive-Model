#!/usr/bin/env python3
"""
Eval TRADES Large Baseline with Same 3-Seed + Bootstrap-CI Rigor as RHAN-v11.

Runs the identical `run_statistical_significance()` sweep used for the
null-ablation so the numbers are directly comparable.

Usage:
  python3 phase1_training/eval_trades_baseline.py --num-samples 200 --steps 10
"""

import os, sys, argparse, numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan_stl10_large import RHANLargeSTL10
from dataset_stl10 import get_stl10_loaders
from eval_rhan_v11 import run_statistical_significance
from checkpoint_utils import compat_load


def main():
    parser = argparse.ArgumentParser(description="Eval TRADES Large Baseline")
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/rhan_stl10_large_pseudolabel_best.pth',
                        help='Path to TRADES Large checkpoint')
    parser.add_argument('--data-root', type=str, default='./data/stl10')
    parser.add_argument('--num-samples', type=int, default=200)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Static TRADES Large Baseline on {device}...", flush=True)

    _, test_loader = get_stl10_loaders(batch_size=32, data_root=args.data_root)

    model = RHANLargeSTL10().to(device)

    ckpt_path = args.checkpoint
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}", flush=True)
        state = compat_load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        elif isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        elif isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}", flush=True)
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found.", flush=True)

    # ── Same 3-seed + bootstrap-CI sweep as the ablation ──
    run_statistical_significance(
        model, test_loader, device,
        num_samples=args.num_samples, steps=args.steps
    )

    print(f"\n{'='*70}", flush=True)
    print(f" TRADES Large Baseline eval complete!", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == '__main__':
    main()
