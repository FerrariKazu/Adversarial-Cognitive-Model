#!/usr/bin/env python3
"""
Find & download a mid-training checkpoint from Hugging Face revision history
=============================================================================
Rolling checkpoints are uploaded to HF under the SAME filename every epoch,
so HF keeps the full revision history. This helper recovers a specific
mid-training epoch (e.g. epoch 40 of the null-ablation v11 run) by:

  1. Listing every commit that touched the file (HfApi.list_repo_commits).
  2. Scanning a coarse grid of revisions, downloading each, reading the
     embedded 'epoch' key, and bracketing the target epoch.
  3. Refining within the bracket to the revision whose epoch is NEAREST to
     the target.
  4. Saving the winner as <out> (full torch checkpoint).

A verified pin can skip the scan entirely (--pin-revision). The null-ablation
v11 epoch-40 scan is verified: revision 82b4f6cc98d3 == epoch 41 (best_acc
48.89; the first epoch of the final eps=0.094 curriculum phase).

Usage:
    python3 phase2_attacks/find_epoch_revision.py \
        --repo FerrariKazu/rhan-checkpoints-rolling \
        --file rhan_stl10_v11_rolling.pth \
        --target-epoch 40 --out checkpoints/rhan_stl10_v11_ep41.pth \
        [--pin-revision 82b4f6cc98d3] [--coarse-step 3] [--max-downloads 12]

Exits 1 if the target epoch cannot be found within the download budget.
"""
import argparse
import os
import re
import shutil
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)


def _hf_token():
    tok = os.environ.get('HF_TOKEN')
    if tok:
        return tok
    env_path = os.path.join(REPO_ROOT, '.env')
    if os.path.exists(env_path):
        m = re.search(r'HF_TOKEN="?([^"\n]+)"?', open(env_path).read())
        if m:
            return m.group(1)
    try:
        from google.colab import userdata
        return userdata.get('HF_TOKEN')
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        pass
    return None


def list_file_revisions(api, repo, filename):
    """Return [(created_at, commit_id)] for every commit touching `filename`."""
    rows = []
    for c in api.list_repo_commits(repo, repo_type='dataset'):
        if filename in (c.title or ''):
            rows.append((c.created_at, c.commit_id))
    rows.sort(key=lambda r: r[0])
    return rows


def download_epoch(api, repo, filename, revision, token):
    """Download revision and return its embedded 'epoch' (or None)."""
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo_id=repo, filename=filename,
                            revision=revision, repo_type='dataset', token=token)
    ck = torch.load(local, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict):
        return None
    return ck.get('epoch')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo', default='FerrariKazu/rhan-checkpoints-rolling')
    p.add_argument('--file', default='rhan_stl10_v11_rolling.pth')
    p.add_argument('--target-epoch', type=int, default=40)
    p.add_argument('--out', required=True,
                   help='Where to save the recovered checkpoint')
    p.add_argument('--pin-revision', default='',
                   help='Skip the scan and use this verified revision')
    p.add_argument('--coarse-step', type=int, default=3)
    p.add_argument('--max-downloads', type=int, default=12)
    args = p.parse_args()

    token = _hf_token()
    if not token:
        print('[FIND-EPOCH] FATAL: HF_TOKEN not found', flush=True)
        sys.exit(1)
    os.environ['HF_TOKEN'] = token

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    if args.pin_revision:
        rev = args.pin_revision
        local = None
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo_id=args.repo, filename=args.file,
                                revision=rev, repo_type='dataset', token=token)
        ck = torch.load(local, map_location='cpu', weights_only=False)
        epoch = ck.get('epoch') if isinstance(ck, dict) else None
        print(f'[FIND-EPOCH] pinned {rev[:12]}: epoch={epoch} -> {args.out}',
              flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        shutil.copy2(local, args.out)
        print(f'[FIND-EPOCH] saved {args.out} '
              f'({os.path.getsize(args.out)/1e6:.1f} MB)', flush=True)
        return

    rows = list_file_revisions(api, args.repo, args.file)
    if not rows:
        print(f'[FIND-EPOCH] no revisions found for {args.file}', flush=True)
        sys.exit(1)
    print(f'[FIND-EPOCH] {len(rows)} revisions of {args.file}', flush=True)

    downloads = 0
    # ── Coarse scan: sample every coarse-step revision ─────────────
    probes = []  # (revision, epoch)
    idx = 0
    while idx < len(rows) and downloads < args.max_downloads:
        _, rev = rows[idx]
        try:
            epoch = download_epoch(api, args.repo, args.file, rev, token)
            downloads += 1
            print(f'[FIND-EPOCH] scan[{idx:3d}] {rev[:12]}: epoch={epoch}',
                  flush=True)
            if epoch is not None:
                probes.append((idx, rev, epoch))
        except Exception as e:
            print(f'[FIND-EPOCH] scan[{idx:3d}] {rev[:12]}: ERROR {e}',
                  flush=True)
        # walk backwards from the newest (most epochs usually near the end)
        idx = len(rows) - 1 if idx == 0 else idx - args.coarse_step
        if idx <= 0:
            break

    if not probes:
        print('[FIND-EPOCH] coarse scan found nothing', flush=True)
        sys.exit(1)

    # nearest probe by epoch distance
    probes.sort(key=lambda t: abs(t[2] - args.target_epoch))
    best_idx, best_rev, best_epoch = probes[0]

    # ── Refine: walk 1 revision at a time toward the target ───────
    direction = 1 if best_epoch < args.target_epoch else -1
    cursor = best_idx + direction
    while (0 <= cursor < len(rows) and downloads < args.max_downloads):
        _, rev = rows[cursor]
        try:
            epoch = download_epoch(api, args.repo, args.file, rev, token)
            downloads += 1
            print(f'[FIND-EPOCH] refine[{cursor:3d}] {rev[:12]}: epoch={epoch}',
                  flush=True)
        except Exception as e:
            print(f'[FIND-EPOCH] refine[{cursor:3d}] {rev[:12]}: ERROR {e}',
                  flush=True)
            cursor += direction
            continue
        if epoch is None:
            cursor += direction
            continue
        if abs(epoch - args.target_epoch) < abs(best_epoch - args.target_epoch):
            best_idx, best_rev, best_epoch = cursor, rev, epoch
            if epoch == args.target_epoch:
                break
        elif direction > 0 and epoch > args.target_epoch:
            break
        elif direction < 0 and epoch < args.target_epoch:
            break
        cursor += direction

    print(f'[FIND-EPOCH] nearest to epoch {args.target_epoch}: '
          f'revision {best_rev[:12]} = epoch {best_epoch} '
          f'({downloads} downloads)', flush=True)

    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo_id=args.repo, filename=args.file,
                            revision=best_rev, repo_type='dataset', token=token)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    shutil.copy2(local, args.out)
    print(f'[FIND-EPOCH] saved {args.out} '
          f'({os.path.getsize(args.out)/1e6:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
