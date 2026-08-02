#!/usr/bin/env python3
"""
2-GPU Sharded 3-Seed Protocol Runner (T4x2-safe)
================================================
Splits the checkpoint list across N GPUs as INDEPENDENT single-GPU subprocesses
(CUDA_VISIBLE_DEVICES=i) and merges their per-seed CSVs into one aggregated
mean ± std table plus the epsilon-crossover significance report.

Why not DataParallel?
    Turing (sm_75) T4x2 crashes with CUDA misaligned-address in DataParallel
    backward passes. Independent processes dodge that entirely — the same
    "one model per GPU, no DDP" approach as --force-single-gpu in training.

Statistical validity:
    - Every shard runs the SAME 3-seed protocol (seeds 41/42/43, n=300/seed,
      fresh sample subset + fresh PGD init per seed). Checkpoints are sharded
      only; seeds are not.
    - Per-sample attack math is batch-size invariant (the batchmean KL factor
      cancels under grad.sign()), so raising --batch-size to 64 on a 16 GB T4
      halves wall clock without changing the protocol's properties.

Wall-clock estimates (T4-class, batch 64, eps 0.0/0.031/0.094, PGD-50, n=300):
    - 1 GPU, all 4 ckpts : ~2.8 h
    - 2 GPUs, 2 ckpts each: ~1.4 h   (Kaggle T4x2)

NOTES:
- --eps-list here defaults to [0.0, 0.031, 0.094]; eval_full_epsilon_sweep.py
  defaults to a wider pixel-space DEFAULT_EPS_LIST. ALWAYS pass --eps-list
  explicitly (the notebooks do).
- Batch size changes the exact per-sample numbers (the PGD init randn draw is
  laid out 5x64 vs 10x32), so batch-64 results are NOT comparable to earlier
  batch-32 runs. Keep the batch size fixed across platforms when merging.

Usage:
    python3 phase2_attacks/shard_2gpu.py \\
        --n-samples 300 --seeds 41 42 43 --pgd-steps 50 --batch-size 64 \\
        --eps-norm-space --eps-list 0.0 0.031 0.094 \\
        --baseline-label trades_large_baseline \\
        --ckpt-specs label:path:arch[:freeze] ... [--gpus 2] [--output-dir BASE]

Outputs (written to BASE/):
    epsilon_sweep_per_seed.csv   merged per-seed rows from all shards
    epsilon_sweep_results.csv    aggregated mean ± std (ddof=1)
    gpu0/ gpu1/ ...              each shard's own incremental CSVs
"""
import argparse
import csv
import os
import subprocess
import sys
import threading
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

EVAL = os.path.join(REPO_ROOT, "phase2_attacks", "eval_full_epsilon_sweep.py")

SEED_FIELDS = ['ckpt_label', 'seed', 'eps_pixel',
               'eps_norm_R', 'eps_norm_G', 'eps_norm_B',
               'acc_pct', 'macro_dprime']
AGG_FIELDS = ['ckpt_label', 'eps_pixel',
              'eps_norm_R', 'eps_norm_G', 'eps_norm_B',
              'acc_mean', 'acc_std', 'macro_dprime_mean', 'macro_dprime_std',
              'n_seeds']


def _mean_std(vals):
    import numpy as np
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else 0.0)


def merge_results(shard_dirs, out_dir):
    """Concatenate per-seed CSVs from each shard dir and re-aggregate mean±std.

    Returns the agg dict {(label, eps_rounded): {'accs': [...], 'dps': [...]}}
    used by the crossover report, and writes:
        out_dir/epsilon_sweep_per_seed.csv   (all seeds, all shards)
        out_dir/epsilon_sweep_results.csv     (aggregated mean ± std)
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for d in shard_dirs:
        p = os.path.join(d, 'epsilon_sweep_per_seed.csv')
        if not os.path.exists(p):
            print(f"  [MERGE] WARNING: missing {p} — shard produced no data, skipping",
                  flush=True)
            continue
        with open(p) as f:
            rows.extend(list(csv.DictReader(f)))
    if not rows:
        raise SystemExit("  [MERGE] no per-seed rows found in any shard — aborting")

    with open(os.path.join(out_dir, 'epsilon_sweep_per_seed.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SEED_FIELDS)
        w.writeheader()
        w.writerows(rows)

    agg = {}
    for r in rows:
        key = (r['ckpt_label'], round(float(r['eps_pixel']), 4))
        rec = agg.setdefault(key, {'accs': [], 'dps': []})
        rec['accs'].append(float(r['acc_pct']))
        rec['dps'].append(float(r['macro_dprime']))

    agg_rows = []
    for (label, eps), rec in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        am, astd = _mean_std(rec['accs'])
        dm, dstd = _mean_std(rec['dps'])
        # eps_norm columns come from any row of the group (identical per eps)
        src = next(r for r in rows
                   if r['ckpt_label'] == label
                   and round(float(r['eps_pixel']), 4) == eps)
        agg_rows.append({
            'ckpt_label': label, 'eps_pixel': round(eps, 4),
            'eps_norm_R': src['eps_norm_R'], 'eps_norm_G': src['eps_norm_G'],
            'eps_norm_B': src['eps_norm_B'],
            'acc_mean': round(am, 2), 'acc_std': round(astd, 2),
            'macro_dprime_mean': round(dm, 4), 'macro_dprime_std': round(dstd, 4),
            'n_seeds': len(rec['accs']),
        })
    with open(os.path.join(out_dir, 'epsilon_sweep_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        w.writeheader()
        w.writerows(agg_rows)

    # ── Final merged table ────────────────────────────────────────────────────
    print("\n" + "=" * 72, flush=True)
    print("  MERGED RESULTS TABLE - mean +- std over seeds (all shards)", flush=True)
    print("=" * 72, flush=True)
    for row in agg_rows:
        print(f"{row['ckpt_label']:<30} {row['eps_pixel']:>6.3f} "
              f"{row['acc_mean']:>7.2f}+-{row['acc_std']:<5.2f} "
              f"{row['macro_dprime_mean']:>8.4f}+-{row['macro_dprime_std']:<6.4f} "
              f"(n={row['n_seeds']})", flush=True)
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', type=int, default=2,
                        help='Number of shards/GPUs to use (auto-clamped to device count)')
    parser.add_argument('--output-dir', type=str, default='./sweep_3seed_sharded',
                        help='Base output dir; each shard writes to <dir>/gpu<i>, merged to <dir>')
    # ── pass-through arguments (same as eval_full_epsilon_sweep.py) ───────────
    parser.add_argument('--n-samples', type=int, default=300)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--pgd-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eps-list', type=float, nargs='+',
                        default=[0.0, 0.031, 0.094])
    parser.add_argument('--eps-norm-space', action='store_true')
    parser.add_argument('--baseline-label', type=str, default='trades_large_baseline')
    parser.add_argument('--ckpt-specs', type=str, nargs='+', required=True,
                        help='label:ckpt_path:arch[:freeze]  (split round-robin across GPUs)')
    args = parser.parse_args()

    # ── Clamp shard count to available GPUs ───────────────────────────────────
    try:
        import torch
        n_avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_avail = 0
    n_gpus = args.gpus if n_avail > 0 else 1
    if n_avail > 0:
        n_gpus = min(args.gpus, n_avail)
    if n_gpus != args.gpus:
        print(f"  [SHARD] NOTE: requested {args.gpus} GPUs but only {n_avail} "
              f"available — using {n_gpus}", flush=True)

    # ── Round-robin split of checkpoints across shards ────────────────────────
    shards = [[] for _ in range(n_gpus)]
    for i, spec in enumerate(args.ckpt_specs):
        shards[i % n_gpus].append(spec)
    for gi, specs in enumerate(shards):
        print(f"  [SHARD {gi}] checkpoints: {[s.split(':')[0] for s in specs]}", flush=True)

    # ── Launch one independent single-GPU eval per shard ──────────────────────
    procs = []
    t_start = time.time()
    for gi, specs in enumerate(shards):
        out_dir = os.path.join(args.output_dir, f'gpu{gi}')
        os.makedirs(out_dir, exist_ok=True)
        cmd = [sys.executable, EVAL,
               '--n-samples', str(args.n_samples),
               '--pgd-steps', str(args.pgd_steps),
               '--batch-size', str(args.batch_size),
               '--output-dir', out_dir,
               '--baseline-label', args.baseline_label]
        if args.seeds:
            cmd += ['--seeds'] + [str(s) for s in args.seeds]
        if args.eps_norm_space:
            cmd += ['--eps-norm-space']
        cmd += ['--eps-list'] + [str(e) for e in args.eps_list]
        cmd += ['--ckpt-specs'] + specs

        env = dict(os.environ)
        if n_gpus > 1:
            env['CUDA_VISIBLE_DEVICES'] = str(gi)
        print(f"\n[SHARD {gi}] launching (CUDA_VISIBLE_DEVICES="
              f"{env.get('CUDA_VISIBLE_DEVICES', 'default')})", flush=True)
        print(f"[SHARD {gi}] {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                universal_newlines=True)
        procs.append((gi, proc, out_dir))

    # Stream each shard's output with a GPU prefix (non-blocking readers)
    def _pump(gi, proc):
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[GPU{gi}] {line}", end='', flush=True)
        proc.stdout.close()

    for gi, proc, _ in procs:
        threading.Thread(target=_pump, args=(gi, proc), daemon=True).start()

    # ── Wait for all shards; partial data is still preserved per-shard ────────
    failures = []
    for gi, proc, out_dir in procs:
        rc = proc.wait()
        if rc != 0:
            failures.append((gi, rc))
            print(f"  [SHARD {gi}] FAILED rc={rc} — its partial per-seed CSV (if any) "
                  f"is at {out_dir}/epsilon_sweep_per_seed.csv", flush=True)
        else:
            print(f"  [SHARD {gi}] finished OK", flush=True)
    elapsed = time.time() - t_start
    print(f"\n  All shards done in {elapsed/60:.1f} min "
          f"({'with ' + str(len(failures)) + ' failure(s)' if failures else '— all OK'})",
          flush=True)

    # ── Merge ─────────────────────────────────────────────────────────────────
    shard_dirs = [d for _, _, d in procs]
    try:
        agg = merge_results(shard_dirs, args.output_dir)
    except SystemExit as e:
        print(e, flush=True)
        if failures:
            sys.exit(1)
        raise

    # ── Crossover significance across the merged dataset ──────────────────────
    from eval_full_epsilon_sweep import crossover_report
    labels = [s.split(':')[0] for s in args.ckpt_specs]
    print("\n  CROSSOVER SIGNIFICANCE - merged across shards "
          "(criterion: d > 2*sig_combined)", flush=True)
    for eps in args.eps_list:
        if eps == 0.0:
            continue
        for line in crossover_report(agg, float(eps), args.baseline_label, labels):
            print(line, flush=True)

    print(f"\n  Merged per-seed CSV: {os.path.join(args.output_dir, 'epsilon_sweep_per_seed.csv')}",
          flush=True)
    print(f"  Merged aggregated  : {os.path.join(args.output_dir, 'epsilon_sweep_results.csv')}",
          flush=True)
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
