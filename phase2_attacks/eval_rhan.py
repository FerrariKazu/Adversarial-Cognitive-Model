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
    init per seed (--seeds 41 42 43), PGD-50, alpha = eps/4;
  * crossover significance: (RHAN - baseline) at eps=0.094 must exceed
    2 x sqrt(std_RHAN^2 + std_baseline^2) — deliberately conservative;
  * checkpoint specs: label:path:arch[:freeze].

The ONLY extension over eval_full_epsilon_sweep.py is the arch registry:

    arch "next"  ->  RHANNext (rhan_core), constructed from the RHANNextConfig
                     embedded in the checkpoint's 'config' key (falling back
                     to the v12-equivalent default config when absent). This
                     is how a pillar-enabled RHANNext checkpoint is evaluated
                     through the unchanged protocol.

No other eval script may be added per stage (roadmap). Examples:

    # Stage 1 validation (AIS vs v12 baseline):
    python3 phase2_attacks/eval_rhan.py --n-samples 300 --seeds 41 42 43 \
        --pgd-steps 50 --batch-size 64 --eps-norm-space --eps-list 0.0 0.094 \
        --baseline-label trades_large_baseline \
        --ckpt-specs \
          trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
          rhan_v12_baseline:checkpoints/rhan_v12_mixB_best.pth:v12 \
          rhan_next_ais:checkpoints/rhan_next_ais_best.pth:next
"""
import os
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


if __name__ == '__main__':
    _sweep.main()
