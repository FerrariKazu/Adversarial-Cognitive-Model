#!/usr/bin/env python3
"""
Merge partial sweep results into final canonical JSON.

Usage:
    python3 phase2_attacks/merge_sweep_results.py \
        --sweep-json report/empirical_sweep_results_stl10.json \
        --square-json report/square_sanity_stl10.json \
        --output-json report/final_sweep_results_stl10.json

ep45 and v10_final PGD sweep results from prior run are embedded here.
"""

import argparse
import json
import sys

# PGD sweep data from prior run (verified from console output)
PRIOR_PGD = {
    "rhan_stl10_large_ep45": {
        "epsilons": [0.0, 0.002, 0.004, 0.006, 0.008, 0.016, 0.024, 0.0313],
        "accuracy": [53.8, 46.2, 38.4, 32.2, 25.2, 11.8, 2.8, 0.6],
        "macro_dprime": [1.7925, 1.4992, 1.1502, 0.9169, 0.6328, -0.0537, -0.6835, -0.9114],
        "pooled_dprime": [1.7275, 1.4612, 1.1925, 0.9751, 0.7162, 0.1080, -0.6738, -1.2880],
        "thresh_dprime_1_macro": 0.0053,
        "thresh_dprime_1_pooled": 0.0058,
    },
    "rhan_v10_final": {
        "epsilons": [0.0, 0.002, 0.004, 0.006, 0.008, 0.016, 0.024, 0.0313],
        "accuracy": [55.2, 45.6, 35.6, 30.0, 21.6, 8.0, 1.2, 0.2],
        "macro_dprime": [1.8721, 1.4967, 1.1135, 0.8948, 0.5083, -0.2906, -0.8581, -0.9796],
        "pooled_dprime": [1.7777, 1.4405, 1.0951, 0.8958, 0.5730, -0.1361, -1.0294, -1.6563],
        "thresh_dprime_1_macro": 0.0050,
        "thresh_dprime_1_pooled": 0.0050,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Merge partial sweep results")
    parser.add_argument('--sweep-json', required=True, help='JSON from Stage 1 (v11_best only)')
    parser.add_argument('--square-json', required=True, help='JSON from Stage 2 (Square Attack)')
    parser.add_argument('--output-json', default='report/final_sweep_results_stl10.json')
    args = parser.parse_args()

    final = {}

    # 1. Add prior PGD results (ep45, v10_final)
    for name, data in PRIOR_PGD.items():
        final[name] = data
        print(f"[merge] Added PGD sweep: {name}")

    # 2. Load and add v11_best PGD results
    with open(args.sweep_json) as f:
        sweep = json.load(f)
    for name, data in sweep.items():
        final[name] = data
        print(f"[merge] Added PGD sweep: {name}")

    # 3. Load and add Square Attack results
    with open(args.square_json) as f:
        square = json.load(f)
    for name, data in square.items():
        key = f"{name}_square"
        final[key] = data
        print(f"[merge] Added Square Attack: {key}")

    # 4. Write final JSON
    with open(args.output_json, 'w') as f:
        json.dump(final, f, indent=2)

    print(f"\n[merge] Final results written to {args.output_json}")
    print(f"Models in final table: {list(final.keys())}")


if __name__ == '__main__':
    main()
