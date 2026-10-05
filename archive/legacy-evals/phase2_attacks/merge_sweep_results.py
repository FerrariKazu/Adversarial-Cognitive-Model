#!/usr/bin/env python3
"""
Merge partial sweep results into final canonical JSON.

Usage:
    python3 phase2_attacks/merge_sweep_results.py \
        --prior-json report/prior_results.json \
        --sweep-json report/empirical_sweep_results_stl10.json \
        --square-json report/square_sanity_stl10.json \
        --output-json report/final_sweep_results_stl10.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Merge partial sweep results")
    parser.add_argument('--prior-json', required=True,
                        help='JSON with ep45 + v10_final PGD results (from prior run)')
    parser.add_argument('--sweep-json', required=True,
                        help='JSON from Stage 1 (v11_best only)')
    parser.add_argument('--square-json', required=True,
                        help='JSON from Stage 2 (Square Attack)')
    parser.add_argument('--output-json', default='report/final_sweep_results_stl10.json')
    args = parser.parse_args()

    final = {}

    # 1. Load prior PGD results (ep45, v10_final)
    with open(args.prior_json) as f:
        prior = json.load(f)
    for name, data in prior.items():
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

    # 5. Print copy-pasteable JSON block
    print("\n" + "=" * 70)
    print("  COPY THIS SECTION — PASTE TO CHAT")
    print("=" * 70)
    print(json.dumps(final, indent=2))
    print("=" * 70)


if __name__ == '__main__':
    main()
