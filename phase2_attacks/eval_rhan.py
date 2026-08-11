#!/usr/bin/env python3
"""
eval_rhan.py — FROZEN evaluation entrypoint for RHAN / RHAN-Next.
=================================================================

DO NOT MODIFY the conventions of this file (frozen in the feature/rhan-next
branch). It is a thin, non-divergent façade over the canonical matched
protocol implemented in eval_full_epsilon_sweep.py — identical conventions:

  * eps applied DIRECTLY in norm space (--eps-norm-space) with per-channel
    bound checks (the Finding-17 baseline-table convention);
  * seed-averaged protocol: n samples per seed, fresh subset + fresh PGD
    init per seed (--seeds 41 42 43 ...), PGD-50, alpha = eps/4;
  * crossover significance: (RHAN - baseline) at eps=0.094 must exceed
    2 x sqrt(std_RHAN^2 + std_baseline^2) — deliberately conservative;
  * checkpoint specs: label:path:arch[:freeze].

PROTOCOL GUARANTEES ENFORCED BY THIS ENTRYPOINT (added Stage 3.1):

  1. NORM-SPACE ONLY. `--eps-norm-space` is injected unconditionally, so the
     pixel-space default of the underlying parser can never be reached through
     this file. Every number produced here is in the Finding-17 convention.
  2. SEED FLOOR. The matched protocol requires >= 5 seeds by default; a run
     with fewer seeds ABORTS unless `--allow-quick` is passed explicitly
     (single-seed / dev sanity runs). `--allow-quick` is consumed here and is
     NOT forwarded to the underlying parser.
  3. SELF-TEST. `--self-test` runs a fast structural self-check against the
     checked-in reference `eval_rhan_selftest_ref.json` (state-dict key hash,
     parameter count, config, forward shapes) and exits. Pass
     `--regenerate-reference` to rewrite the reference file (deliberate).
  4. PROVENANCE. After every successful run, `eval_provenance.json` is
     written next to the results CSVs with the git SHA, per-checkpoint
     SHA-256, seed list, full CLI settings, timestamp, and the merged
     results table + crossover verdicts.

The ONLY extensions over eval_full_epsilon_sweep.py are:

  1. ARCH REGISTRY. arch "next" -> RHANNext (rhan_core), constructed from the
     RHANNextConfig embedded in the checkpoint's 'config' key (falling back
     to the v12-equivalent default config when absent). This is how a
     pillar-enabled RHANNext checkpoint is evaluated through the unchanged
     protocol.
  2. ABLATION-MATRIX FLAG (Stage 2). `--ablation-matrix [keys...]` builds
     `--ckpt-specs` automatically from rhan_core/ablation/matrix.py (entries
     with status VALIDATED, or PENDING once their checkpoint exists) instead
     of requiring them typed out each time — the three-way A/B/C comparison
     of the Stage 2 protocol is one flag, not a hand-built spec string. The
     flag is consumed here (never forwarded to the frozen parser) and is
     mutually exclusive with an explicit --ckpt-specs. Keys, when given,
     follow the flag directly (e.g. `--ablation-matrix C_hpc_only B_ais_only
     A_baseline`); without keys, every eligible matrix entry is used.

No other eval script may be added per stage (roadmap). Examples:

    # Stage 1 validation (AIS-v1: Relocated Equation II vs v12 baseline) —
    # 5-seed protocol; the label rhan_next_ais_v1 flows into every result
    # table row and eval_provenance.json:
    python3 phase2_attacks/eval_rhan.py --n-samples 300 --seeds 41 42 43 44 45 \
        --pgd-steps 50 --batch-size 64 --eps-norm-space --eps-list 0.0 0.094 \
        --baseline-label trades_large_baseline \
        --ckpt-specs \
          trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
          rhan_v12_baseline:checkpoints/rhan_v12_mixB_best.pth:v12 \
          rhan_next_ais_v1:checkpoints/rhan_next_ais_v1_best.pth:next

    # Dev sanity (single seed — explicit escape hatch):
    python3 phase2_attacks/eval_rhan.py --allow-quick \
        --n-samples 50 --seeds 42 --pgd-steps 10 --eps-list 0.0 0.094 \
        --ckpt-specs rhan_next_ais_v1:checkpoints/rhan_next_ais_v1_best.pth:next

    # Self-test (structural, against checked-in reference):
    python3 phase2_attacks/eval_rhan.py --self-test
"""
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS_DIR, '..'))
for _p in (_THIS_DIR, _REPO, os.path.join(_REPO, 'phase1_training')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

# The canonical protocol lives in eval_full_epsilon_sweep.py (frozen).
import eval_full_epsilon_sweep as _sweep  # noqa: E402

# Keep a reference to the ORIGINAL frozen loader before patching (the patch
# below replaces _sweep.load_model, so delegation must target this one).
_ORIGINAL_LOAD_MODEL = _sweep.load_model

MIN_PROTOCOL_SEEDS = 5
SELFTEST_REF = os.path.join(_THIS_DIR, 'eval_rhan_selftest_ref.json')
PROVENANCE_NAME = 'eval_provenance.json'


def _load_model(arch, ckpt_path, device, freeze_gaze=False):
    """Extended arch registry: adds 'next' (RHANNext); everything else
    delegates unchanged to the frozen loader."""
    if arch == "next":
        from rhan_core.config.pillar_config import RHANNextConfig
        from rhan_core.model import RHANNext

        cfg = RHANNextConfig()                     # v12-equivalent default
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device,
                               weights_only=False)
            if isinstance(state, dict):
                cfg_dict = state.get('config')
                if isinstance(cfg_dict, dict):
                    cfg = RHANNextConfig.from_dict(cfg_dict)
                    print(f"  [eval] RHANNext config from checkpoint: {cfg}",
                          flush=True)
                for k in ('model', 'model_state_dict', 'state_dict'):
                    if k in state:
                        state = state[k]
                        break
        model = RHANNext(config=cfg).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(missing)
        print(f"  Loaded {n_loaded}/{len(state)} keys "
              f"({len(missing)} missing, {len(unexpected)} unexpected)",
              flush=True)
        if missing and cfg == RHANNextConfig():
            print(f"  NOTE: {len(missing)} missing keys with the DEFAULT "
                  f"config — did this checkpoint train with pillars enabled?",
                  flush=True)
        model.eval()
        if freeze_gaze and hasattr(model, 'freeze_gaze'):
            model.freeze_gaze = True
            print("  ISOLATION TEST: foveal gaze frozen to image center (0,0)",
                  flush=True)
        return model

    return _ORIGINAL_LOAD_MODEL(arch, ckpt_path, device, freeze_gaze)


# Patch the frozen module's loader so its main() uses our registry.
_sweep.load_model = _load_model


# ── Protocol guards ───────────────────────────────────────────────────────────

def _argv_flag_values(flag, cast=None):
    """Values following `--flag v1 v2 ...` or `--flag=v1` (matching what the
    frozen parser's argparse accepts). Scans ALL occurrences and returns the
    values of the LAST one (argparse nargs='+' keeps the last occurrence).
    Returns [] when the flag is absent."""
    out = []
    for i, tok in enumerate(sys.argv):
        if tok == flag:
            vals = []
            for nxt in sys.argv[i + 1:]:
                if nxt.startswith('--'):
                    break
                vals.append(nxt)
            if vals:  # last occurrence wins (argparse semantics)
                out = vals
        elif tok.startswith(flag + '='):
            out = [tok.split('=', 1)[1]]
    if cast is not None:
        out = [cast(v) for v in out]
    return out


def _argv_scalar(flag, default=None, cast=None):
    vals = _argv_flag_values(flag, cast)
    return vals[0] if vals else default


def _seeds_from_argv():
    seeds = _argv_flag_values('--seeds', int)
    if seeds:
        return list(dict.fromkeys(seeds))
    return [_argv_scalar('--seed', 42, int)]


def _enforce_seed_protocol():
    """Require >= MIN_PROTOCOL_SEEDS seeds unless --allow-quick is present.

    Consumes `--allow-quick` (stripped from sys.argv so the frozen parser
    never sees an unknown flag). Returns the resolved seed list.
    """
    allow_quick = '--allow-quick' in sys.argv
    if allow_quick:
        sys.argv = [a for a in sys.argv if a != '--allow-quick']
        print("  [eval_rhan] --allow-quick: seed floor waived "
              "(dev/sanity run only — NOT a publishable number).", flush=True)
    seeds = _seeds_from_argv()
    if not allow_quick and len(seeds) < MIN_PROTOCOL_SEEDS:
        print("=" * 72, flush=True)
        print("  eval_rhan.py: PROTOCOL ERROR — the matched 5-seed protocol", flush=True)
        print(f"  requires >= {MIN_PROTOCOL_SEEDS} seeds, got {len(seeds)}: {seeds}.", flush=True)
        print("", flush=True)
        print(f"  Pass >= {MIN_PROTOCOL_SEEDS} seeds, e.g.:", flush=True)
        print(f"      --seeds 41 42 43 44 45", flush=True)
        print("", flush=True)
        print("  For an explicit dev/sanity run only, add --allow-quick", flush=True)
        print("  (single-seed numbers must NOT be reported as results).", flush=True)
        print("=" * 72, flush=True)
        raise SystemExit(2)
    return seeds


def _force_norm_space():
    """Inject --eps-norm-space so the pixel-space default of the underlying
    parser can never be reached through this entrypoint."""
    if '--eps-norm-space' not in sys.argv:
        sys.argv.append('--eps-norm-space')
        print("  [eval_rhan] NORM-SPACE MANDATORY: injected --eps-norm-space "
              "(pixel-space mode is disabled through this entrypoint).",
              flush=True)


def _ablation_matrix_specs():
    """Build ckpt-specs from the ablation registry for --ablation-matrix.

    Consumes `--ablation-matrix [keys...]` (stripped from sys.argv so the
    frozen parser never sees an unknown flag), then returns the spec list;
    the caller injects them as a trailing `--ckpt-specs` (last-occurrence
    wins). Mutually exclusive with an explicit --ckpt-specs (a silent
    last-wins override could mask a hand-typed spec). Returns [] when the
    flag is absent.
    """
    if not any(a == '--ablation-matrix' or a.startswith('--ablation-matrix=')
               for a in sys.argv):
        return []
    keys = _argv_flag_values('--ablation-matrix')
    # Strip the flag (+ its values, and the --flag=KEY equals form) so the
    # frozen parser never sees an unknown token.
    stripped = []
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok == '--ablation-matrix' or tok.startswith('--ablation-matrix='):
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith('--'):
                i += 1
            continue
        stripped.append(tok)
        i += 1
    sys.argv = stripped

    if '--ckpt-specs' in sys.argv:
        print("=" * 72, flush=True)
        print("  eval_rhan.py: PROTOCOL ERROR — --ablation-matrix and", flush=True)
        print("  --ckpt-specs are mutually exclusive (the registry builds", flush=True)
        print("  the specs; passing both risks a silent last-wins override).", flush=True)
        print("=" * 72, flush=True)
        raise SystemExit(2)

    from rhan_core.ablation import runner as _ablation
    specs, skipped = _ablation.eval_specs(keys=keys or None, require_present=False)
    for why in skipped:
        print(f"  [ablation-matrix] SKIPPED: {why}", flush=True)
    if not specs:
        print("=" * 72, flush=True)
        print("  eval_rhan.py: no eval-eligible ablation-matrix entries.", flush=True)
        print("  (VALIDATED always; PENDING only once its checkpoint exists.)", flush=True)
        print("=" * 72, flush=True)
        raise SystemExit(2)
    print(f"  [eval_rhan] --ablation-matrix: {len(specs)} checkpoint(s) "
          f"from the registry:", flush=True)
    for s in specs:
        print(f"    {s['label']}:{os.path.basename(s['path'])}:{s['arch']}",
              flush=True)
    return specs


# ── Provenance ────────────────────────────────────────────────────────────────

def _git_sha():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=_REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_branch():
    try:
        out = subprocess.run(['git', 'branch', '--show-current'], cwd=_REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "detached"
    except Exception:
        return "unknown"


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _ckpt_specs_from_argv():
    specs = _argv_flag_values('--ckpt-specs')
    if specs:
        parsed = []
        for s in specs:
            parts = s.split(':')
            if len(parts) < 3:
                continue
            freeze = len(parts) > 3 and parts[3].strip().lower() in (
                'freeze', 'freeze-gaze', '1', 'true')
            parsed.append({'label': parts[0], 'path': parts[1],
                           'arch': parts[2], 'freeze': freeze})
        return parsed
    return [{'label': l, 'path': p, 'arch': a, 'freeze': False}
            for l, p, a in _sweep.DEFAULT_CHECKPOINTS]


def _write_provenance():
    """Write eval_provenance.json next to the results CSVs (post-run)."""
    try:
        out_dir = _argv_scalar('--output-dir', './sweep_results')
        seeds = _seeds_from_argv()
        specs = _ckpt_specs_from_argv()

        prov = {
            "schema": "eval_rhan_provenance_v1",
            "tool": "phase2_attacks/eval_rhan.py",
            "git_sha": _git_sha(),
            "git_branch": _git_branch(),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec='seconds'),
            "eps_mode": "norm_space (forced by eval_rhan.py; Finding-17 "
                        "matched convention)",
            "seeds": seeds,
            "n_samples": _argv_scalar('--n-samples', 500, int),
            "pgd_steps": _argv_scalar('--pgd-steps', 50, int),
            "batch_size": _argv_scalar('--batch-size', 50, int),
            "baseline_label": _argv_scalar('--baseline-label',
                                           'trades_large_baseline'),
            "eps_list": _argv_flag_values('--eps-list', float)
                        or list(_sweep.DEFAULT_EPS_LIST),
            "checkpoints": [{
                "label": c['label'], "path": c['path'], "arch": c['arch'],
                "freeze": c['freeze'], "sha256": _sha256_file(c['path']),
            } for c in specs],
        }

        # Merge the aggregated results CSV produced by the frozen sweep.
        agg_csv = os.path.join(out_dir, 'epsilon_sweep_results.csv')
        rows = []
        if os.path.exists(agg_csv):
            with open(agg_csv, newline='') as f:
                rows = list(csv.DictReader(f))
        prov["results_csv"] = agg_csv
        prov["results"] = rows

        # Recompute the crossover verdicts from the aggregated rows using the
        # same conservative criterion as the frozen sweep (d > 2*sig_combined).
        prov["crossover_verdicts"] = _crossover_verdicts(rows)

        out_path = os.path.join(out_dir, PROVENANCE_NAME)
        with open(out_path, 'w') as f:
            json.dump(prov, f, indent=2, sort_keys=True)
        print(f"\n  [eval_rhan] Provenance written: {out_path}", flush=True)
    except Exception as e:  # never crash a completed eval over bookkeeping
        print(f"  [eval_rhan] WARNING: provenance write failed: {e}",
              flush=True)


def _crossover_verdicts(rows):
    """Recompute d vs 2*sqrt(rs^2 + bs^2) at every eps with >= 2 baseline seeds.

    The aggregated CSV has ONE row per (label, eps) — the seed count lives in
    the 'n_seeds' column (the frozen sweep writes per-seed lists only in
    memory / the per-seed CSV). So '>= 2 seeds' means int(n_seeds) >= 2 here.
    """
    import math
    eps_list = sorted({float(r['eps_pixel']) for r in rows})
    baseline = _argv_scalar('--baseline-label', 'trades_large_baseline')
    verdicts = []
    for eps in eps_list:
        if eps == 0.0:
            continue
        b = [r for r in rows
             if r['ckpt_label'] == baseline and float(r['eps_pixel']) == eps]
        if not b or int(b[0].get('n_seeds', 0) or 0) < 2:
            continue
        bm = float(b[0]['acc_mean'])
        bs = float(b[0]['acc_std'])
        for r in rows:
            if r['ckpt_label'] == baseline or float(r['eps_pixel']) != eps:
                continue
            rm, rs = float(r['acc_mean']), float(r['acc_std'])
            diff = rm - bm
            combined = math.sqrt(rs ** 2 + bs ** 2)
            verdict = ("CROSSOVER REAL" if diff > 2.0 * combined
                       else ("positive but NOT significant" if diff > 0
                             else "at or below baseline"))
            verdicts.append({
                "eps": eps, "checkpoint": r['ckpt_label'],
                "acc_mean": rm, "acc_std": rs,
                "baseline": baseline, "baseline_acc_mean": bm,
                "diff_pp": round(diff, 2),
                "threshold_2sig": round(2.0 * combined, 2),
                "verdict": verdict,
            })
    return verdicts


# ── Self-test ─────────────────────────────────────────────────────────────────

def _selftest_ref() -> dict:
    """Load the checked-in reference (or None if absent/malformed)."""
    if not os.path.exists(SELFTEST_REF):
        return None
    try:
        with open(SELFTEST_REF) as f:
            ref = json.load(f)
        if not isinstance(ref, dict) or 'schema' not in ref:
            return None
        return ref
    except (json.JSONDecodeError, OSError):
        return None


def _self_test(regenerate: bool = False, device=None) -> int:
    """Structural self-check against the checked-in reference JSON.

    Checks (all device-independent; pass --regenerate-reference to rewrite):
      1. default RHANNextConfig == reference config;
      2. default RHANNext state-dict key-set hash + count == reference;
      3. parameter count == reference;
      4. forward / get_feature_vector shapes on a fixed dummy input
         == reference (logits (B,10), features (B,768)).
    Returns 0 on PASS, 1 on FAIL.
    """
    from rhan_core.config.pillar_config import RHANNextConfig
    from rhan_core.model import RHANNext

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(device)
    print(f"  [self-test] device={dev}  reference={SELFTEST_REF}", flush=True)

    if not regenerate:
        ref = _selftest_ref()
        if ref is None:
            print("  [self-test] FAIL: reference file missing or malformed:\n"
                  f"    {SELFTEST_REF}\n"
                  "    Re-generate deliberately with "
                  "--self-test --regenerate-reference.", flush=True)
            return 1

    torch.manual_seed(0)
    cfg = RHANNextConfig()
    model = RHANNext(config=cfg).to(dev).eval()

    keys = sorted(model.state_dict().keys())
    key_sha = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, 3, 96, 96, device=dev)
    with torch.no_grad():
        logits = model(x)
        feats = model.get_feature_vector(x)
    checks = {
        "default_config": cfg.to_dict(),
        "state_dict_key_sha256": key_sha,
        "n_state_dict_keys": len(keys),
        "n_params": n_params,
        "forward_shapes": {"logits": list(logits.shape),
                           "features": list(feats.shape)},
    }
    del model
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    if regenerate:
        ref = {
            "schema": "eval_rhan_selftest_v1",
            "description": "Structural reference for phase2_attacks/eval_rhan.py "
                           "--self-test. Regenerate deliberately via "
                           "--self-test --regenerate-reference.",
            "generated_at_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec='seconds'),
            "generated_by_torch": torch.__version__,
            **checks,
        }
        with open(SELFTEST_REF, 'w') as f:
            json.dump(ref, f, indent=2, sort_keys=True)
        print(f"  [self-test] Reference written: {SELFTEST_REF}", flush=True)
        return 0

    failures = []
    if checks['default_config'] != ref['default_config']:
        failures.append("default_config differs from reference")
    if checks['state_dict_key_sha256'] != ref['state_dict_key_sha256']:
        failures.append("state-dict key-set hash differs from reference "
                        "(architecture drift?)")
    if checks['n_state_dict_keys'] != ref['n_state_dict_keys']:
        failures.append(f"key count {checks['n_state_dict_keys']} != "
                        f"reference {ref['n_state_dict_keys']}")
    if checks['n_params'] != ref['n_params']:
        failures.append(f"param count {checks['n_params']} != "
                        f"reference {ref['n_params']}")
    if checks['forward_shapes'] != ref['forward_shapes']:
        failures.append(f"forward shapes {checks['forward_shapes']} != "
                        f"reference {ref['forward_shapes']}")

    if failures:
        print("  [self-test] FAIL:", flush=True)
        for f in failures:
            print(f"    - {f}", flush=True)
        return 1
    print("  [self-test] PASS: config, state-dict key hash, param count, "
          "and forward shapes all match the checked-in reference.", flush=True)
    return 0


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        regen = '--regenerate-reference' in sys.argv
        sys.exit(_self_test(regenerate=regen))

    specs = _ablation_matrix_specs()
    if specs:
        sys.argv += ['--ckpt-specs'] + [
            f"{s['label']}:{s['path']}:{s['arch']}" for s in specs]

    _enforce_seed_protocol()
    _force_norm_space()
    _sweep.main()
    _write_provenance()
