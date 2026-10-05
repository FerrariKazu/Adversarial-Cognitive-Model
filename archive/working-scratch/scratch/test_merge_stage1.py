"""Self-test for scripts/merge_stage1_seed_extension.py.

Builds synthetic per-seed CSVs using the ACTUAL per-seed accuracies from the
2026-08-09 Kaggle run log (so the 5-seed aggregates must reproduce 31.07 /
22.93 / 49.40 / 53.47), appends invented-but-plausible extension seeds
46-48, runs the merge script, and asserts the merged math.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Actual per-seed acc_pct from the 2026-08-09 run log (PGD-50 legs).
MAIN = {
    # (label, eps) -> {seed: (acc_pct, macro_dprime)}
    ("rhan_next_ais_v1_halting_only", 0.0): {
        41: (50.33, 2.0008), 42: (49.00, 2.0939), 43: (44.67, 1.3856),
        44: (54.33, 1.7986), 45: (48.67, 1.8953)},
    ("rhan_next_ais_v1_halting_only", 0.094): {
        41: (28.33, 0.8497), 42: (33.67, 1.0646), 43: (29.67, 0.6463),
        44: (32.00, 1.2508), 45: (31.67, 0.9829)},
    ("trades_large_baseline", 0.0): {
        41: (51.00, 1.9081), 42: (56.33, 2.0828), 43: (50.00, 2.1296),
        44: (56.00, 1.6498), 45: (54.00, 1.5981)},
    ("trades_large_baseline", 0.094): {
        41: (20.00, -0.0160), 42: (24.00, 0.4508), 43: (18.67, 0.9342),
        44: (26.33, 0.5210), 45: (25.67, 0.2792)},
}
# Invented extension seeds (46-48) at eps=0.094 ONLY.
EXT = {
    ("rhan_next_ais_v1_halting_only", 0.094): {
        46: (30.00, 0.9000), 47: (32.00, 1.1000), 48: (31.00, 0.9500)},
    ("trades_large_baseline", 0.094): {
        46: (24.00, 0.3000), 47: (21.00, 0.1000), 48: (23.00, 0.4000)},
}

FIELDS = ['ckpt_label', 'seed', 'eps_pixel',
          'eps_norm_R', 'eps_norm_G', 'eps_norm_B',
          'acc_pct', 'macro_dprime']


def write_per_seed(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for (label, eps), seed_map in sorted(rows.items()):
            for seed in sorted(seed_map):
                acc, dp = seed_map[seed]
                w.writerow({
                    'ckpt_label': label, 'seed': seed,
                    'eps_pixel': round(eps, 4),
                    'eps_norm_R': round(eps, 4), 'eps_norm_G': round(eps, 4),
                    'eps_norm_B': round(eps, 4),
                    'acc_pct': round(acc, 2), 'macro_dprime': round(dp, 4),
                })


def main():
    tmp = tempfile.mkdtemp(prefix="merge_test_")
    main_dir = os.path.join(tmp, "main")
    ext_dir = os.path.join(tmp, "ext")
    out_dir = os.path.join(tmp, "merged")
    write_per_seed(os.path.join(main_dir, "epsilon_sweep_per_seed.csv"), MAIN)
    write_per_seed(os.path.join(ext_dir, "epsilon_sweep_per_seed.csv"), EXT)

    cmd = [sys.executable, os.path.join(REPO, "scripts",
                                        "merge_stage1_seed_extension.py"),
           "--main-dir", main_dir, "--ext-dir", ext_dir,
           "--out-dir", out_dir,
           "--baseline-label", "trades_large_baseline",
           "--pgd-steps", "50", "--n-samples", "300", "--batch-size", "64",
           "--ckpt-specs",
           "rhan_next_ais_v1_halting_only:checkpoints/fake_rolling.pth:next",
           "trades_large_baseline:checkpoints/fake.pth:large",
           "--main-seeds", "41", "42", "43", "44", "45",
           "--ext-seeds", "46", "47", "48"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        sys.exit(f"merge script failed rc={r.returncode}")

    # ── Assertions ──────────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "epsilon_sweep_results.csv")) as f:
        agg = {((row['ckpt_label']), float(row['eps_pixel'])): row
               for row in csv.DictReader(f)}
    with open(os.path.join(out_dir, "eval_provenance.json")) as f:
        prov = json.load(f)

    def expect(key, want, tol=0.011):
        got = float(agg[key]['acc_mean'])
        ok = abs(got - want) <= tol
        print(f"  {'OK ' if ok else 'FAIL'} {key} acc_mean={got} "
              f"(want {want})")
        if not ok:
            sys.exit(f"assertion failed: {key} {got} != {want}")

    # eps=0.0 rows stay 5-seed -> must reproduce the real run's numbers
    # (49.40 / 53.47), proving the 5-seed half of the merge is faithful.
    expect(("rhan_next_ais_v1_halting_only", 0.0), 49.40)
    expect(("trades_large_baseline", 0.0), 53.47)
    # eps=0.094 rows are the 8-seed MERGED means (hand-computed from the 5
    # real + 3 synthetic extension seeds).
    expect(("rhan_next_ais_v1_halting_only", 0.094), 31.0425)
    expect(("trades_large_baseline", 0.094), 22.8338)

    assert int(agg[("rhan_next_ais_v1_halting_only", 0.094)]['n_seeds']) == 8
    assert int(agg[("rhan_next_ais_v1_halting_only", 0.0)]['n_seeds']) == 5
    assert prov["seeds"] == [41, 42, 43, 44, 45, 46, 47, 48]
    assert prov["seed_extension"]["applied"] is True
    assert prov["results"][0]['ckpt_label']  # rows are dicts (strings ok)

    cvs = prov["crossover_verdicts"]
    assert len(cvs) == 1 and cvs[0]["eps"] == 0.094
    print(f"  crossover: d={cvs[0]['diff_pp']} vs 2sig="
          f"{cvs[0]['threshold_2sig']} -> {cvs[0]['verdict']}")
    assert cvs[0]["checkpoint"] == "rhan_next_ais_v1_halting_only"
    # n_seeds must be 8 in the merged crossover (baseline check).
    bl = agg[("trades_large_baseline", 0.094)]
    assert int(bl['n_seeds']) == 8

    # Per-seed merged CSV has 2 labels * (5+3) rows at 0.094 + 2*5 at 0.0.
    with open(os.path.join(out_dir, "epsilon_sweep_per_seed.csv")) as f:
        n = sum(1 for _ in csv.DictReader(f))
    assert n == 2 * 5 + 2 * 8, f"per-seed row count {n} != 26"

    print("\nALL MERGE TESTS PASSED")
    print(f"  (test dir kept for inspection: {tmp})")


if __name__ == "__main__":
    main()
