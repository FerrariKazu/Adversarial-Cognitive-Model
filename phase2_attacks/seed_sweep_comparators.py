#!/usr/bin/env python3
"""
Seed a target eval sweep's per-seed resume CSV with comparator cells from an
already-completed donor sweep, so a Step-C eval only computes genuinely NEW
model cells.

WHY (Stage 4 E-variants): every E-sweep writes to a fresh output directory
(report/sweep_stage4_e{1,2,3}_*), so eval_rhan.py --resume finds no prior
cells there and re-evaluates the comparators — trades_large_baseline and
rhan_next_ais_hpc (32 cells each, ~8-9 h of T4 per model) — even though those
exact cells were already computed under the IDENTICAL protocol in the E1 sweep
(sweep_stage4_e1_d_e1_pgd100: same seeds 41-56, same 300-sample draws via
set_seed, same PGD-100 norm-space eps grid — clean cells bit-match across
sessions). This tool copies those rows into the target sweep's
epsilon_sweep_per_seed.csv (local + HF), so the next eval launch skips the
donor labels entirely and only computes the new model.

MATCHING CAVEAT (pre-registered 2026-09-02): donor rows were measured in the
E1 session while the new model's rows come from the run session — a
cross-session comparison carrying the documented ~1 pp GPU nondeterminism
floor. Accepted by design for E-variant verdicts; keep the provenance in the
verdict write-up.

Usage (run from the repo root, BEFORE the eval cell):
    python3 phase2_attacks/seed_sweep_comparators.py \
        --output-dir report/sweep_stage4_e3_d_t6_pgd100 \
        --target-subdir sweep_stage4_e3_d_t6_pgd100

Defaults:
    --donor-subdir sweep_stage4_e1_d_e1_pgd100   (the completed E1 sweep)
    --labels       trades_large_baseline,rhan_next_ais_hpc
    --repo         FerrariKazu/rhan-eval-sweep
    --no-upload    skip the HF upload (local CSV only)

Merge semantics: existing target rows are kept, except that rows for the donor
labels with the same (label, seed, eps_pixel) key are REPLACED by the donor's
canonical values. This preserves any partial new-model progress (e.g. t6 rows
from an aborted session) while making the comparator cells canonical.
"""

import os
import sys
import csv
import argparse
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_NAME = 'epsilon_sweep_per_seed.csv'
FIELDNAMES = ['ckpt_label', 'seed', 'eps_pixel',
              'eps_norm_R', 'eps_norm_G', 'eps_norm_B',
              'acc_pct', 'macro_dprime']


def _cell_key(row):
    try:
        return (row['ckpt_label'], int(row['seed']),
                round(float(row['eps_pixel']), 4))
    except (KeyError, ValueError):
        return None


def _get_hf_token():
    token = os.environ.get('HF_TOKEN')
    if token:
        return token
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            return token
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
        if token:
            return token
    except Exception:
        pass
    return None


def _hf_download_rows(repo_id, path_in_repo, hf_token):
    """Download a CSV from HF and return its rows ([] when absent)."""
    if not hf_token:
        return []
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=repo_id, repo_type='dataset',
                                     filename=path_in_repo, token=hf_token)
        with open(downloaded, newline='') as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"  [seed] no donor/current CSV on HF ({path_in_repo}): {e}",
              flush=True)
        return []


def _read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _hf_upload_csv(csv_path, repo_id, path_in_repo, hf_token):
    if not os.path.exists(csv_path) or not hf_token:
        return False
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=hf_token)
        create_repo(repo_id=repo_id, repo_type='dataset', private=False,
                    exist_ok=True, token=hf_token)
        api.upload_file(path_or_fileobj=csv_path, path_in_repo=path_in_repo,
                        repo_id=repo_id, repo_type='dataset', token=hf_token)
        print(f"  [seed] uploaded -> {repo_id}/{path_in_repo}", flush=True)
        return True
    except Exception as e:
        print(f"  [seed] WARNING: upload failed: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', required=True,
                    help='Target sweep output dir (holds epsilon_sweep_per_seed.csv)')
    ap.add_argument('--target-subdir', default=None,
                    help='HF path subdir for the target CSV (default: basename(output-dir))')
    ap.add_argument('--donor-subdir', default='sweep_stage4_e1_d_e1_pgd100',
                    help='HF/local subdir of the completed donor sweep')
    ap.add_argument('--labels', default='trades_large_baseline,rhan_next_ais_hpc',
                    help='Comma-separated comparator labels to copy from the donor')
    ap.add_argument('--repo', default='FerrariKazu/rhan-eval-sweep')
    ap.add_argument('--no-upload', action='store_true',
                    help='Skip the HF upload (local CSV only)')
    args = ap.parse_args()

    donor_labels = {s.strip() for s in args.labels.split(',') if s.strip()}
    target_subdir = args.target_subdir or os.path.basename(args.output_dir)
    target_csv = os.path.join(args.output_dir, CSV_NAME)
    donor_csv_local = os.path.join(REPO_ROOT, 'report',
                                   args.donor_subdir, CSV_NAME)
    hf_token = None if args.no_upload else _get_hf_token()

    # 1) Donor rows: local copy of the completed sweep first, then HF.
    donor_rows = _read_csv(donor_csv_local)
    source = donor_csv_local
    if donor_rows is None:
        donor_rows = _hf_download_rows(args.repo,
                                       f"{args.donor_subdir}/{CSV_NAME}",
                                       hf_token)
        source = f"HF {args.repo}/{args.donor_subdir}/{CSV_NAME}"
    if not donor_rows:
        print(f"  [seed] FATAL: donor CSV not found (local {donor_csv_local} "
              f"or HF {args.donor_subdir}/{CSV_NAME}). The eval will re-run "
              f"the comparators from scratch instead.", flush=True)
        return 1

    donor = {}
    for r in donor_rows:
        key = _cell_key(r)
        if key and r['ckpt_label'] in donor_labels:
            donor[key] = r
    found = sorted({r['ckpt_label'] for r in donor.values()})
    missing = sorted(donor_labels - set(found))
    print(f"  [seed] donor {source}: {len(donor)} comparator cell(s) "
          f"({', '.join(found) or 'NONE'})", flush=True)
    if missing:
        print(f"  [seed] WARNING: donor has no rows for label(s): "
              f"{', '.join(missing)} — those cells will still be computed.",
              flush=True)

    # 2) Current target rows (preserve any partial new-model progress).
    target_rows = _read_csv(target_csv)
    if target_rows is None:
        fetched = _hf_download_rows(args.repo, f"{target_subdir}/{CSV_NAME}",
                                    hf_token)
        target_rows = fetched
        if fetched:
            print(f"  [seed] found {len(fetched)} existing row(s) on HF "
                  f"({target_subdir}) — preserving them", flush=True)
    else:
        print(f"  [seed] found {len(target_rows)} existing row(s) locally — "
              f"preserving them", flush=True)

    # 3) Merge: keep target rows, then donor wins for comparator cells.
    merged = {}
    for r in target_rows or []:
        key = _cell_key(r)
        if key:
            merged[key] = r
    replaced = sum(1 for k in donor if k in merged)
    for k, r in donor.items():
        merged[k] = r
    merged_rows = [merged[k] for k in sorted(merged, key=lambda k: (k[0], k[1], k[2]))]

    counts = {}
    for r in merged_rows:
        counts[r['ckpt_label']] = counts.get(r['ckpt_label'], 0) + 1
    print(f"  [seed] merged CSV now has {len(merged_rows)} cell(s): "
          + ", ".join(f"{lbl}={n}" for lbl, n in sorted(counts.items())),
          flush=True)
    print(f"  [seed] comparator rows replaced: {replaced}; "
          f"non-comparator rows preserved: "
          f"{sum(n for l, n in counts.items() if l not in donor_labels)}",
          flush=True)

    _write_csv(target_csv, merged_rows)
    print(f"  [seed] wrote {target_csv}", flush=True)

    # 4) Upload so --resume in a fresh session sees the seeded rows.
    if not args.no_upload:
        _hf_upload_csv(target_csv, args.repo,
                       f"{target_subdir}/{CSV_NAME}", hf_token)
    return 0


if __name__ == '__main__':
    sys.exit(main())
