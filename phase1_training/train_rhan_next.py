#!/usr/bin/env python3
"""
train_rhan_next.py — RHAN-Next training entrypoint (strict superset of
train_rhan_v12.py).
=====================================================================

The trainer is a strict superset of train_rhan_v12.py, NOT a divergent
codepath: identical data pipeline (STL-10 real + pseudo + synthetic mixes),
identical curriculum (3 phases), identical warmup freeze schedule, identical
HF rolling-checkpoint resume gates, identical diagnostics. The differences,
all gated behind RHANNextConfig toggles (OFF by default = exactly v12):

  * --enable-ais (Pillar 2, Stage 1):
        - loss gains a precision-modulated reconstruction weight
          w_recon * (0.5 + Pi_D * gain) — the GlobalPrecisionModulator's
          reconstruction-loss consumer. DEFERRED from the Stage 1 headline
          config per the 2026-08-07 isolation verdict (smoke<->isoB contrast:
          recon-mod is the confirmed driver of the smoke's Pi_D reordering);
          the validated run uses --no-ais-precision-recon (the "AIS-v1
          (halting-only variant)");
        - the gaze update and halting go through InformationGainGazePolicy /
          EntropyGatedHalting (no step-count penalty — see
          tests/test_gradient_flow.py::test_no_step_count_penalty_in_loss_path).
  * --enable-hpc / --hpc-num-levels (Pillar 1, Stage 2):
        - loss gains w_hpc * L_hpc (mean hierarchical prediction error);
        - the HPC predictor gets its OWN optimizer group at lr*hpc_lr_mult
          (default 6.67 -> 0.02 in phase 1) with per-group grad clipping
          (2026-08-13 starvation fix — see build_next_optimizer).
  * ISOLATION flags (Stage 1 mechanism isolation, Run A/B pattern — each
    ablates exactly ONE AIS sub-mechanism, everything else identical):
        --no-ais-halting          -> entropy gate forced open (cont=1,
                                     v12 fixed-T belief accumulation);
        --no-ais-precision-recon  -> w_recon stays FLAT (v12 recon weighting).

Loss (pillars on):
    L = w_trades * L_trades + w_recon_eff * L_recon + w_hpc * L_hpc
where w_recon_eff = w_recon when AIS is off OR --no-ais-precision-recon.

Checkpoints:
    *_best.pth    -> {'model': state_dict, 'config': RHANNextConfig dict, 'arch': 'rhan_next'}
    *_rolling.pth -> v12's resume dict + 'config'
The embedded config lets phase2_attacks/eval_rhan.py reconstruct the exact
pillar config without any external bookkeeping.

Frozen files: model_rhan_v12.py / eval conventions are never touched.
"""

import os
import sys

# Fail-fast on HF network stalls: huggingface_hub freezes HF_HUB_DOWNLOAD_TIMEOUT
# at import time, so set it BEFORE any huggingface_hub import (all HF imports in
# this file are function-level, so module top is safe). A stalled restore
# download now raises within ~30s per request instead of hanging the trainer
# silently (2026-08-09: the notebook's Step A hung ~2 h on a no-timeout
# hf_hub_download; the trainer's own resume-restore had the same exposure).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import gc
import json
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext
from checkpoint_utils import compat_load, current_code_commit, resume_commit_ok

# ── Reuse the frozen v12 pipeline (data, pseudo-labeling, HF, diagnostics) ──
from train_rhan_v12 import (
    load_dotenv_fallback,
    set_seed,
    STL10RawUnlabeledDataset,
    CombinedSTL10Dataset,
    BalancedBatchSampler,
    generate_pseudo_labels,
    find_optimal_dataloader_config,
    ensure_checkpoint_exists,
    sync_to_hf,
    wait_for_hf_sync,
    EpochDiagnostics,
)
from train_rhan_stl10_tdv import get_stl10_dataloaders

load_dotenv_fallback()


class RHANNextEpochDiagnostics(EpochDiagnostics):
    """EpochDiagnostics + RHANNext AIS telemetry.

    Adds the two signals Step A of the Stage 1 protocol requires, on top of
    the inherited v12 block (beta_dyn, gate alpha, recon MSE, Pi_D per class):

      1. GAZE SHIFT DISTANCE — mean ||a_t - a_{t-1}||_2 over the batch for
         every step boundary, plus the total gaze path length. This is the
         direct measure that the Eq. II v12 / info-gain gradient actually
         moves the fovea (degenerate = ~0.0).

      2. PER-SAMPLE HALTING VARIANCE — effective evidence steps per sample,
         defined as the sum of the soft continuation weights over the loop
         (max_steps steps). The HARD loop is still fixed at T = max_steps
         (hard per-sample early exit is deferred — see roadmap), so the honest
         signal that halting now varies per-sample is the DISTRIBUTION of
         effective steps (std > 0) and the fraction of samples whose
         continuation dropped below 0.5 at any step.

    `summary_dict(epoch, eps)` produces the machine-readable per-epoch
    telemetry line written by --diag-json (consumed by the notebook health
    gate that decides whether Step A is healthy enough for Step B).
    """

    def __init__(self, max_steps=4):
        super().__init__()
        self.max_steps = int(max_steps)
        self.gaze_shifts = []       # list of (B,) per step boundary, per batch
        self.effective_steps = []   # list of (B,) per batch (sum of continuations)
        self.halted_steps = []      # list of (B,) counts of steps with cont < 0.5
        # ── Stage 2 (HPC) telemetry ──────────────────────────────────────────
        # One scalar per batch (mean over steps AND samples) plus per-batch
        # error-map stats — the Step A health gate reads these to enforce the
        # downward-trend / no-explosion criteria. Per-class errors mirror the
        # pi_d_per_class block (2026-08-15 diagnosis: is the Π_D reordering
        # accompanied by a class-specific HPC error — truck worse than
        # car/airplane — or is it purely a Π_D phenomenon?).
        self.hpc_err_means = []     # float per batch (epoch-mean -> trend)
        self.hpc_emap = {'min': [], 'max': [], 'std': []}
        self.hpc_err_per_class = {c: [] for c in range(10)}

    def update(self, beta_dyn, traj_c, labels):
        super().update(beta_dyn, traj_c, labels)
        acts = traj_c.get('actions') or []
        conts = traj_c.get('continuations') or []
        if len(acts) >= 2:
            # Gaze displacement at each step boundary: ||a_t - a_{t-1}||.
            for t in range(1, len(acts)):
                shift = (acts[t] - acts[t - 1]).norm(dim=-1)    # (B,)
                self.gaze_shifts.append(shift.detach().cpu())
        if len(conts) >= 1:
            conts_cpu = torch.stack([c.detach().cpu() for c in conts], dim=0)  # (T,B)
            self.effective_steps.append(conts_cpu.sum(dim=0))   # (B,)
            self.halted_steps.append((conts_cpu < 0.5).sum(dim=0).float())  # (B,)
        # ── Stage 2 (HPC): collect per-batch error mean + map stats ──────────
        hpc_errs = traj_c.get('hpc_errors') or []
        if hpc_errs:
            # mean over steps AND samples (each err is (B,) and attached;
            # detach here — diagnostics only, the loss uses the raw tensors).
            self.hpc_err_means.append(
                float(torch.stack([e.detach().mean() for e in hpc_errs]).mean()))
            # Per ground-truth class, pooled over steps (same mask pattern as
            # the inherited precisions_per_class block).
            labels_cpu = labels.detach().cpu()
            for c in range(10):
                mask = (labels_cpu == c)
                if mask.any():
                    self.hpc_err_per_class[c].append(
                        torch.cat([e.detach().cpu()[mask] for e in hpc_errs]))
        emaps = traj_c.get('hpc_error_maps') or []
        for m in emaps:
            for k in ('min', 'max', 'std'):
                self.hpc_emap[k].append(float(m[k]))

    def _steps_used_line(self, mean_steps):
        """Override the inherited v12 line: report the hard cap + the
        soft-gated effective mean, not a bare fixed step count.

        The HARD loop is capped at self.max_steps; the entropy gate acts on the
        continuation weights, so the per-sample effective evidence steps (sum
        of continuations) is what actually varies. Presenting both together
        stops the old 'Steps used: mean=4.00 (fixed T=4)' line from reading as
        a red flag at a glance when halting is clearly active. Cosmetic only.
        """
        if self.effective_steps:
            eff = torch.cat(self.effective_steps)
            return (f"  Continuation steps: hard cap={self.max_steps}, "
                    f"soft-halted effective mean={eff.mean():.2f}")
        return super()._steps_used_line(mean_steps)

    def report(self, epoch, eps):
        super().report(epoch, eps)   # beta_dyn, gate, recon, Pi_D per class

        # ── AIS-specific: gaze shift + per-sample halting variance ──
        if self.gaze_shifts:
            shifts = torch.cat(self.gaze_shifts)               # all boundaries
            per_step = []
            nb = self.max_steps - 1
            for t in range(nb):
                sel = [s for i, s in enumerate(self.gaze_shifts)
                       if i % nb == t]
                per_step.append(torch.cat(sel).mean().item() if sel else float('nan'))
            print(f"  Gaze shift |Δa| per step (mean over batch): "
                  f"{[f'{v:.4f}' for v in per_step]}")
            print(f"  Gaze shift total path (mean over batch): {shifts.mean():.4f}")
        if self.effective_steps:
            eff = torch.cat(self.effective_steps)
            hal = torch.cat(self.halted_steps)
            print(f"  Effective evidence steps (Σ continuation): "
                  f"mean={eff.mean():.2f} std={eff.std():.2f} "
                  f"min={eff.min():.2f} max={eff.max():.2f} "
                  f"| frac with any halting: {(hal > 0).float().mean():.3f}")

        # ── Stage 2 (HPC): prediction error + error-map summary ──────────────
        # Same format as the β_dynamic / Π_D / gaze-shift block above so logs
        # stay comparable across stages. Mean should trend DOWN over training;
        # map min/max/std flag collapse (all -> 0) or explosion (std -> huge).
        if self.hpc_err_means:
            print(f"  HPC prediction error (mean): "
                  f"{np.mean(self.hpc_err_means):.4f} "
                  f"(epoch-1 baseline compare in the health gate)")
            emin = np.mean(self.hpc_emap['min'])
            emax = np.mean(self.hpc_emap['max'])
            estd = np.mean(self.hpc_emap['std'])
            print(f"  HPC error map (min/max/std): {emin:.4f} / {emax:.4f} / "
                  f"{estd:.4f} "
                  f"({'collapse' if emax < 1e-6 else 'explosion?' if estd > 5.0 else 'ok'})")
            per_cls = []
            for c in range(10):
                if self.hpc_err_per_class[c]:
                    per_cls.append((self.CLASSES[c], float(
                        torch.cat(self.hpc_err_per_class[c]).mean())))
            if per_cls:
                print("  HPC prediction error per class (mean):")
                for name, v in sorted(per_cls, key=lambda kv: -kv[1]):
                    marker = " ◄" if name in ('car', 'truck') else ""
                    print(f"    {name:<12}: {v:.4f}{marker}")

    def summary_dict(self, epoch, eps, tr_acc=None, te_acc=None):
        """Machine-readable per-epoch telemetry (written by --diag-json)."""
        d = {
            'epoch': int(epoch),
            'eps': round(float(eps), 4),
            'beta_dyn_mean': round(float(torch.cat(self.beta_dynamics).mean()), 4),
            'beta_dyn_std': round(float(torch.cat(self.beta_dynamics).std()), 4),
            'steps_hard_fixed': self.max_steps,
            # Clean accuracy (None when dry-run/rank>0 skips validation) so
            # the smoke diag lets callers WATCH clean acc next to the HPC
            # error. Intentionally NOT a health-gate criterion (pre-registered
            # checks unchanged).
            'tr_acc': round(float(tr_acc), 4) if tr_acc is not None else None,
            'te_acc': round(float(te_acc), 4) if te_acc is not None else None,
        }
        if self.gate_alphas:
            d['gate_alpha'] = round(float(np.mean(self.gate_alphas)), 4)
        if self.recon_losses:
            d['recon_mse'] = round(float(np.mean(self.recon_losses)), 4)
        if self.effective_steps:
            eff = torch.cat(self.effective_steps)
            hal = torch.cat(self.halted_steps)
            d['steps_effective_mean'] = round(float(eff.mean()), 3)
            d['steps_effective_std'] = round(float(eff.std()), 3)
            d['steps_effective_min'] = round(float(eff.min()), 3)
            d['steps_effective_max'] = round(float(eff.max()), 3)
            d['frac_halted_any'] = round(float((hal > 0).float().mean()), 4)
        if self.gaze_shifts:
            d['gaze_shift_total_mean'] = round(
                float(torch.cat(self.gaze_shifts).mean()), 5)
        # Stage 2 (HPC): the health gate's trend / explosion criteria consume
        # these keys. Always present (0.0 when HPC off) so a future gate never
        # hits a missing-key edge case.
        d['hpc_error_mean'] = round(float(np.mean(self.hpc_err_means)), 6) \
            if self.hpc_err_means else 0.0
        d['hpc_error_map_min'] = round(float(np.mean(self.hpc_emap['min'])), 6) \
            if self.hpc_emap['min'] else 0.0
        d['hpc_error_map_max'] = round(float(np.mean(self.hpc_emap['max'])), 6) \
            if self.hpc_emap['max'] else 0.0
        d['hpc_error_map_std'] = round(float(np.mean(self.hpc_emap['std'])), 6) \
            if self.hpc_emap['std'] else 0.0
        # Per-class HPC prediction error (2026-08-15 diagnosis) — mirrors
        # pi_d_per_class so the gate/dashboards can compare truck vs car vs
        # airplane directly. Present whenever hpc_errors were collected.
        d['hpc_error_per_class'] = {}
        for c in range(10):
            if self.hpc_err_per_class[c]:
                d['hpc_error_per_class'][self.CLASSES[c]] = round(
                    float(torch.cat(self.hpc_err_per_class[c]).mean()), 6)
        d['pi_d_per_class'] = {}
        for c in range(10):
            if self.precisions_per_class[c]:
                d['pi_d_per_class'][self.CLASSES[c]] = round(
                    float(torch.cat(self.precisions_per_class[c]).mean()), 4)
        # Truck-rank WATCH (gate AMENDMENT 2026-08-16): truck's rank among
        # the top-3 Π_D classes + its margin vs the #2 slot, per epoch —
        # NON-BLOCKING, logged so the 60-epoch run's watch series can be
        # assembled from --diag-json (roadmap stages['2'].watch_metrics.epochs).
        _pd = d['pi_d_per_class']
        if len(_pd) >= 2 and 'truck' in _pd:
            _ranked = sorted(_pd.items(), key=lambda kv: -kv[1])
            _truck_rank = next((i + 1 for i, (k, _) in enumerate(_ranked)
                                if k == 'truck'), None)
            if _truck_rank is not None:
                d['truck_pi_d_rank'] = int(_truck_rank)
                d['truck_pi_d_in_top3'] = bool(_truck_rank <= 3)
                d['truck_pi_d_vs_2_margin'] = round(
                    float(_pd['truck'] - _ranked[1][1]), 4)
        return d


# Components frozen during the warmup phase (v12 list + new pillar modules).
_WARMUP_FROZEN_FRAGMENTS = [
    'foveal_stream', 'precision_ctrl', 'action_init', 'parafoveal_stream',
    'foveal_gate', 'generative_prior', 'image_precision',
    'gaze_policy', 'precision_modulator', 'hpc_stack', 'hpc_level1',
]


def set_new_component_training(model, trainable):
    """Freeze/unfreeze active-inference + pillar components (warmup schedule)."""
    for name, param in model.named_parameters():
        if any(x in name for x in _WARMUP_FROZEN_FRAGMENTS):
            param.requires_grad = trainable
        else:
            param.requires_grad = True


def build_next_optimizer(model, phase_lr, hpc_lr_mult=6.67, weight_decay=1e-4):
    """Two-group SGD: backbone at phase_lr, HPC predictor at phase_lr*hpc_lr_mult.

    2026-08-13 (Stage 2 smoke #3, cold start on the fixed head): the HPC
    predictor's output conv stayed EXACTLY at its ±0.01 init draw after 15
    epochs (abs-mean 0.00516), and hpc_error_mean froze at the predict-zero
    baseline (~0.69 = mean(target^2)) for all 10 main-phase epochs. Root
    cause: optimizer starvation, not wiring — the isolated learnability test
    (lr=0.05, no w_hpc cut, no clip) proved the head learns 28%/10 steps, but
    the real loop attenuated that update ~1000x (w_hpc=0.1 loss weight x
    shared backbone lr 0.003 x the global clip_grad_norm_(...1.0) whose norm
    the 76M-param backbone's TRADES gradient dominates). Measured: raw
    last-conv grad through the real loss path ~0.004 (vs 0.54 isolated),
    per-step |dW| ~1.4e-5 on |W|~0.005 — invisible.

    Fix: the head gets its own param group at lr*hpc_lr_mult (6.67x -> 0.02
    in phase 1, matching the proven isolated recipe) and per-group grad
    clipping (clip_grad_per_group) so the backbone's norm can never dilute
    it. Loss budget (w_hpc) and the pre-registered health-gate criteria are
    UNCHANGED.
    """
    backbone_params, hpc_params = [], []
    for name, p in model.named_parameters():
        if 'hpc' in name:          # hpc_level1.stack.* (the hpc_stack alias
            hpc_params.append(p)   # never appears in named_parameters)
        else:
            backbone_params.append(p)
    return optim.SGD([
        {'params': backbone_params, 'lr': phase_lr},
        {'params': hpc_params, 'lr': phase_lr * hpc_lr_mult},
    ], momentum=0.9, weight_decay=weight_decay, foreach=True)


def clip_grad_per_group(optimizer, max_norm=1.0):
    """Per-group grad clipping (2026-08-13).

    Each param group is clipped to ITS OWN max_norm budget. This is what
    keeps the HPC head's update alive: under the old single global clip, the
    backbone's TRADES gradient owns the norm and the head's tiny grad got
    crushed to ~1e-5 updates. Groups with no gradients (e.g. the frozen HPC
    stack during warmup) are skipped.
    """
    for group in optimizer.param_groups:
        params = [p for p in group['params'] if p.grad is not None]
        if params:
            nn.utils.clip_grad_norm_(params, max_norm)


def optimizer_restore_compatible(saved_opt, optimizer, saved_scheduler=None):
    """Can a checkpoint's saved optimizer state be restored onto `optimizer`?

    The 2026-08-13 two-group optimizer (backbone lr + HPC lr*hpc_lr_mult)
    must never load a state written by a DIFFERENT configuration:
      * pre-2026-08-13 checkpoints carry ONE group (Stage 1's AIS-v1 runs had
        no HPC module) — loading them raises ValueError, and if the group
        count ever matched with a different param order, PyTorch maps
        momentum buffers BY POSITION, silently misassigning them (HPC params
        receiving backbone momentum, or vice versa — the checkpoint-resume
        bug class this project has been burned by: the destroyed rolling
        checkpoint, the best/rolling parity gap);
      * same-commit flag drift (e.g. a --hpc-lr-mult 1.0 checkpoint resumed
        with the 6.67 default) would otherwise restore the OLD head lr
        silently (2026-08-12 parse_known_args).
    Comparison source: the saved optimizer's param_groups carry the CURRENT
    (cosine-decayed) lrs of the epoch at which the checkpoint was written,
    while the trainer rebuilds a fresh optimizer at the phase-start lrs on
    resume — comparing those would falsely refuse every legitimate mid-phase
    session-continuation resume (dropping momentum + restarting the cosine
    schedule). So when `saved_scheduler` is provided, its base_lrs (the
    flag-derived phase-start lrs, invariant to decay) are compared instead;
    otherwise the saved current lrs are used as a fallback.
    Returns True only when the group count AND per-group lrs match this run's
    flags. On False the trainer falls back to a fresh optimizer with a loud
    warning instead of calling load_state_dict.
    """
    if not isinstance(saved_opt, dict):
        return False
    saved_groups = saved_opt.get('param_groups', [])
    if len(saved_groups) != len(optimizer.param_groups):
        return False
    expected = None
    if isinstance(saved_scheduler, dict):
        base = saved_scheduler.get('base_lrs')
        if (isinstance(base, (list, tuple))
                and len(base) == len(optimizer.param_groups)):
            expected = [float(x) for x in base]
    if expected is None:
        expected = [g.get('lr') for g in saved_groups]
    cur_lrs = [g.get('lr') for g in optimizer.param_groups]
    return all(
        isinstance(a, (int, float)) and isinstance(b, (int, float))
        and abs(float(a) - float(b)) < 1e-9
        for a, b in zip(expected, cur_lrs))


def restore_optimizer_from_checkpoint(optimizer, scheduler, checkpoint_data,
                                      rank=0):
    """Resume-path optimizer restore with the group-count/LR guard.

    The trainer's ONE and only way to restore optimizer/scheduler state on
    resume (2026-08-13). Refuses a saved state that does not match THIS run's
    flags (optimizer_restore_compatible) and falls back to a fresh optimizer
    with a loud warning — never a silent load_state_dict. Returns True when
    the state was restored, False when the fresh-optimizer fallback ran.
    """
    _saved_opt = checkpoint_data.get('optimizer')
    if optimizer_restore_compatible(_saved_opt, optimizer,
                                    checkpoint_data.get('scheduler')):
        optimizer.load_state_dict(_saved_opt)
        scheduler.load_state_dict(checkpoint_data['scheduler'])
        if rank == 0:
            print("Restored optimizer/scheduler state.", flush=True)
        return True
    if rank == 0:
        if (not isinstance(_saved_opt, dict) or
                len(_saved_opt.get('param_groups', [])) !=
                len(optimizer.param_groups)):
            _why = (f"group count {len(_saved_opt.get('param_groups', []))}"
                    if isinstance(_saved_opt, dict) else "no optimizer state")
        else:
            _why = "lr mismatch (different --hpc-lr-mult)"
        print(f"  WARNING: checkpoint optimizer {_why} vs this run (HPC "
              f"group). Starting a fresh optimizer for this phase.", flush=True)
    return False


def ensure_diag_baseline(diag_path, baseline_row):
    """Make `baseline_row` (the run's first-epoch telemetry summary) the first
    row of a --diag-json file, preserving any existing rows.

    Why: the health gate's trend check compares rows[0] (the epoch-1
    baseline) against the last row. A session that resumes from a rolling
    checkpoint and appends only its own epochs leaves a single-epoch diag —
    the gate then compares the final epoch to ITSELF (ratio 1.00), which
    reads as a "did not learn" verdict no matter what the model did (the
    2026-08-13 Stage 2 smoke: "epoch 15 0.6906 -> epoch 15 0.6906, ratio
    1.00"). The rolling checkpoint carries the run's first-epoch summary in
    'first_epoch_diag'; this prepends it so the trend reference survives the
    checkpoint boundary.

    Idempotent: a file that already starts with the baseline's epoch is left
    untouched (repeated resumes must not duplicate the baseline).
    Returns True when the baseline is (or already was) the first row.
    """
    if not isinstance(baseline_row, dict) or baseline_row.get('epoch') is None:
        return False
    base_epoch = int(baseline_row['epoch'])
    existing = ''
    first = None
    if os.path.exists(diag_path):
        try:
            with open(diag_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        first = json.loads(line).get('epoch')
                        break
            if first == base_epoch:
                return True                     # baseline already first
        except (OSError, ValueError):
            first = None
        try:
            with open(diag_path) as f:
                existing = f.read()
        except OSError as e:
            print(f"  WARNING: could not read {diag_path} for baseline "
                  f"prepend: {e}", flush=True)
            return False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(diag_path)), exist_ok=True)
        with open(diag_path, 'w') as f:
            f.write(json.dumps(baseline_row) + '\n')
            if existing and not existing.endswith('\n'):
                existing += '\n'
            f.write(existing)
    except OSError as e:
        print(f"  WARNING: could not prepend epoch-{base_epoch} baseline to "
              f"{diag_path}: {e}", flush=True)
        return False
    print(f"  Resume telemetry: prepended epoch-{base_epoch} baseline to "
          f"{os.path.basename(diag_path)} — the health gate's trend check "
          f"now compares epoch {base_epoch} vs the final epoch, never an "
          f"epoch against itself.", flush=True)
    return True


def dynamic_trades_loss_next(model, imgs, labels, weights, x_adv,
                             beta_base, w_recon, w_hpc,
                             precision_recon_enabled: bool = True):
    """
    The RHAN-Next loss (superset of v12's two-term loss).

        L = w_trades * L_trades + w_recon_eff * L_recon + w_hpc * L_hpc

    with w_recon_eff = w_recon * mean(0.5 + Pi_D * gain) when the precision
    modulator exists (AIS) AND precision_recon_enabled (ISOLATION B sets it
    False => w_recon_eff = w_recon flat, v12 weighting). L_hpc = 0 when HPC
    is off. No step-count penalty term exists anywhere in this function.
    """
    logits_c, traj_c = model(imgs, return_trajectory=True)
    logits_a, traj_a = model(x_adv, return_trajectory=True)

    # Per-image dynamic beta from precision (Pi_D forward pass retained).
    if len(traj_c['precisions']) > 0:
        final_precision_c = traj_c['precisions'][-1]        # (B,)
    else:
        final_precision_c = torch.full((imgs.shape[0],), 0.5, device=imgs.device)

    beta_dynamic = beta_base * (0.5 + final_precision_c)    # [beta/2, 1.5*beta]

    # TRADES robustness term (identical to v12).
    ce = nn.CrossEntropyLoss(reduction='none')
    l_ce = ce(logits_c, labels)
    l_kl = F.kl_div(
        F.log_softmax(logits_a.float(), dim=1),
        F.softmax(logits_c.float().detach(), dim=1),
        reduction='none').sum(dim=1)
    l_trades = ((l_ce + beta_dynamic * l_kl) * weights.to(l_ce.device)).mean()

    # Reconstruction loss for the generative prior (v12 fix: differentiable).
    l_recon = 0.5 * (
        model.get_reconstruction_loss(imgs, (logits_c, traj_c))
        + model.get_reconstruction_loss(x_adv, (logits_a, traj_a)))

    # HPC prediction-error loss (0.0 when HPC is off).
    l_hpc = 0.5 * (
        model.get_hpc_loss(imgs, (logits_c, traj_c))
        + model.get_hpc_loss(x_adv, (logits_a, traj_a)))

    # Precision-modulated recon weight (AIS consumer, gain-scaled).
    # ISOLATION B (--no-ais-precision-recon): keep w_recon flat (v12).
    modulator = getattr(model, 'precision_modulator', None)
    if modulator is not None and precision_recon_enabled:
        w_recon_eff = modulator.modulate_recon_weight(w_recon, final_precision_c)
    else:
        w_recon_eff = torch.tensor(w_recon, device=imgs.device)

    return (l_trades, traj_c, traj_a, beta_dynamic.detach(),
            l_recon, l_hpc, w_recon_eff)


# ────────────────────────────────────────────────────────────────────────────
# Curriculum + data prep (identical to v12)
# ────────────────────────────────────────────────────────────────────────────

CURRICULUM = [
    (1,  20, 0.031, 2.0, 4,  0.003),
    (21, 40, 0.062, 2.0, 4,  0.002),
    (41, 60, 0.094, 2.5, 4,  0.001),
]


def build_config(args) -> RHANNextConfig:
    cfg = RHANNextConfig(
        enable_ais=args.enable_ais,
        enable_hpc=args.enable_hpc,
        hpc_num_levels=args.hpc_num_levels,
        max_foraging_steps=args.max_foraging_steps,
        fovea_size=args.fovea_size,
        metabolic_cost=args.metabolic_cost,
        gaze_lambda=args.gaze_lambda,
        ais_halt_threshold=args.ais_halt_threshold,
        ais_continuation_softness=args.ais_continuation_softness,
        ais_halt_enabled=not args.no_ais_halting,
        ais_precision_recon_enabled=not args.no_ais_precision_recon,
        hpc_error_weight=args.w_hpc,
    )
    cfg.validate()
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='RHAN-Next training (superset of train_rhan_v12.py)')
    # ── v12 flags (unchanged) ────────────────────────────────────────────────
    parser.add_argument('--data-root', type=str, default='./data/stl10')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--unlabeled-batch-size', type=int, default=256)
    parser.add_argument('--accum-steps', type=int, default=32)
    parser.add_argument('--confidence-threshold', type=float, default=0.65)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--labeling-ckpt', type=str, default='')
    parser.add_argument('--target-ckpt', type=str, default='')
    parser.add_argument('--fixed-samples-per-epoch', type=int, default=0)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force-restart', action='store_true')
    parser.add_argument('--max-foraging-steps', type=int, default=4)
    parser.add_argument('--fovea-size', type=int, default=48)
    parser.add_argument('--metabolic-cost', type=float, default=0.05)
    parser.add_argument('--w-trades', type=float, default=0.55)
    parser.add_argument('--w-recon', type=float, default=0.10)
    parser.add_argument('--gaze-lambda', type=float, default=0.5)
    parser.add_argument('--synthetic-data', type=str, default='')
    parser.add_argument('--ckpt-name', type=str, default='rhan_stl10_next')
    parser.add_argument('--no-pseudo', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--freeze-gaze', action='store_true')
    parser.add_argument('--force-single-gpu', action='store_true')
    # ── RHAN-Next pillar flags ───────────────────────────────────────────────
    parser.add_argument('--enable-ais', action='store_true',
                        help='Pillar 2: AIS-v1 = relocated Eq. II v12 gaze + '
                             'entropy-gated halting (+ precision-modulated recon '
                             'weight; DEFERRED from the Stage 1 headline config '
                             'per the 2026-08-07 isolation verdict — disable with '
                             '--no-ais-precision-recon for the validated run; '
                             'replication-under-refactor control)')
    parser.add_argument('--enable-hpc', action='store_true',
                        help='Pillar 1: hierarchical predictive coding (Stage 2)')
    parser.add_argument('--hpc-num-levels', type=int, default=0,
                        help='HPC levels (0 = off, matching the config '
                             'default; 1 implemented; never jump levels)')
    parser.add_argument('--w-hpc', type=float, default=0.10,
                        help='HPC prediction-error loss weight (w_hpc — a '
                             'SEPARATE slot from w_recon; 0.0 disables the '
                             'term without touching recon weighting)')
    parser.add_argument('--hpc-lr-mult', type=float, default=6.67,
                        help='HPC predictor optimizer-group LR multiplier '
                             '(2026-08-13 starvation fix: the head learns at '
                             'lr*hpc_lr_mult = 0.02 in phase 1; the shared '
                             'backbone lr 0.003 + w_hpc=0.1 + global clip '
                             'starved it to ~1e-5 updates/step — see '
                             'build_next_optimizer. Loss budget unchanged)')
    parser.add_argument('--ais-halt-threshold', type=float, default=0.35,
                        help='EntropyGatedHalting: halt when uncertainty < this')
    parser.add_argument('--ais-continuation-softness', type=float, default=8.0)
    # ── Stage 1 mechanism-isolation flags (one ablation per run) ────────────
    parser.add_argument('--no-ais-halting', action='store_true',
                        help='ISOLATION A: force the entropy gate open (cont=1, '
                             'v12 fixed-T belief accumulation); gaze unchanged')
    parser.add_argument('--no-ais-precision-recon', action='store_true',
                        help='ISOLATION B: keep w_recon FLAT (v12 recon '
                             'weighting); precision modulator no longer scales '
                             'the reconstruction loss (trainer-side only)')
    parser.add_argument('--diag-json', type=str, default='',
                        help='Append one JSON line per epoch with machine-readable '
                             'AIS telemetry (gaze shift, effective steps, Pi_D per '
                             'class) — consumed by the notebook health gate.')
    args, _unknown = parser.parse_known_args()
    if _unknown:
        # parse_known_args() SILENTLY drops unrecognized tokens — the
        # 2026-08-12 Stage 2 smoke lost --max-epochs/--target-ckpt/
        # --diag-json this way (it ran 60 epochs from the wrong base).
        # Fail loudly so a dropped flag can never silently corrupt a run.
        print(f"\n[FATAL] train_rhan_next.py received {len(_unknown)} "
              f"unrecognized argument(s): {_unknown}", flush=True)
        print(f"  Refusing to start — parse_known_args would silently ignore "
              f"them, and a dropped flag\n  (2026-08-12: --max-epochs / "
              f"--target-ckpt / --diag-json) silently changed the run.\n  "
              f"Fix the caller or define the flag in this parser.", flush=True)
        sys.exit(2)

    # ── Environment / device ────────────────────────────────────────────────
    is_ddp = "WORLD_SIZE" in os.environ and "RANK" in os.environ
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl', init_method='env://')
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    cfg = build_config(args)

    if rank == 0:
        print(f"{'═'*60}")
        print(f"  RHAN-Next Training (superset of v12)")
        print(f"  Device: {device} | DDP: {is_ddp} (world_size={world_size})")
        print(f"  Config: {cfg}")
        if cfg.enable_ais:
            print(f"    AIS-v1 (Relocated Eq. II v12): "
                  f"halt_threshold={cfg.ais_halt_threshold}, "
                  f"softness={cfg.ais_continuation_softness}")
            if not cfg.ais_halt_enabled:
                print(f"    ISOLATION A: halting DISABLED (cont=1, "
                      f"v12 fixed-T accumulation)")
            if not cfg.ais_precision_recon_enabled:
                print(f"    AIS-v1 (halting-only variant): precision-modulated "
                      f"recon weight DISABLED (w_recon flat)")
        if cfg.enable_hpc:
            print(f"    HPC: levels={cfg.hpc_num_levels}, "
                  f"w_hpc={cfg.hpc_error_weight}, "
                  f"targets={cfg.hpc_num_levels and 'edge_map'}")
            print(f"    HPC optimizer group: lr x {args.hpc_lr_mult} "
                  f"(head lr {CURRICULUM[0][-1] * args.hpc_lr_mult:.4f} "
                  f"in phase 1) + per-group grad clip "
                  f"(2026-08-13 starvation fix)")
        print(f"  Loss weights: trades={args.w_trades}, recon={args.w_recon}"
              + (f", hpc={args.w_hpc}" if cfg.enable_hpc else ""))
        print(f"  Max epochs: {args.max_epochs}")
        print(f"{'═'*60}", flush=True)

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.abspath(os.path.join(script_dir, '..', 'checkpoints'))
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ── 1. Pseudo-labels (unless --no-pseudo) — identical to v12 ────────────
    pseudo_indices = pseudo_lbls = None
    if not args.no_pseudo:
        unlabeled_dataset = STL10RawUnlabeledDataset(args.data_root)
        if rank == 0:
            from model_rhan_stl10_pretrained import RHANUnifiedSTL10
            labeling_model = RHANUnifiedSTL10().to(device, memory_format=torch.channels_last)
            best_labeling_ckpt = args.labeling_ckpt or os.path.join(
                ckpt_dir, 'rhan_stl10_pseudolabel_best.pth')
            best_labeling_ckpt = ensure_checkpoint_exists(best_labeling_ckpt)
            if os.path.exists(best_labeling_ckpt):
                from checkpoint_utils import compat_load
                labeling_model.load_state_dict(
                    compat_load(best_labeling_ckpt, map_location=device))
            else:
                print("Error: labeling checkpoint not found!", flush=True)
                sys.exit(1)
            num_workers = min(4, os.cpu_count() or 2)
            unlabeled_loader = torch.utils.data.DataLoader(
                unlabeled_dataset, batch_size=args.unlabeled_batch_size,
                shuffle=False, num_workers=num_workers, pin_memory=True)
            pseudo_indices, pseudo_lbls, _ = generate_pseudo_labels(
                labeling_model, unlabeled_loader, device, args.confidence_threshold)
            del labeling_model
            torch.cuda.empty_cache()
            gc.collect()
        if is_ddp:
            import torch.distributed as dist
            dist.barrier()
        if len(pseudo_indices) == 0:
            if rank == 0:
                print("Error: No pseudo-labels generated. Exiting.", flush=True)
            sys.exit(1)
    else:
        if rank == 0:
            print("--no-pseudo active: real (+ synthetic only).", flush=True)

    # ── 2/3. Data prep — identical to v12 ───────────────────────────────────
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    norm_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))])
    trainset_raw = torchvision.datasets.STL10(args.data_root, split='train', download=True)
    num_real = len(trainset_raw)
    real_imgs = torch.zeros(num_real, 3, 96, 96, dtype=torch.float32)
    for i in range(num_real):
        real_imgs[i] = norm_transform(trainset_raw[i][0])
    real_labels = torch.tensor([trainset_raw[i][1] for i in range(num_real)])

    synth_imgs = synth_labels = None
    if args.synthetic_data and os.path.exists(args.synthetic_data):
        print(f"Loading synthetic data from {args.synthetic_data}...", flush=True)
        synth_dict = torch.load(args.synthetic_data, map_location='cpu')
        synth_imgs = synth_dict['imgs'].to(torch.uint8).contiguous()
        synth_labels = synth_dict['labels']
        print(f"  Loaded {synth_imgs.size(0)} synthetic images", flush=True)

    train_transform = T.Compose([
        T.RandomCrop(96, padding=12), T.RandomHorizontalFlip()])
    unlabeled_dataset = None
    if not args.no_pseudo:
        unlabeled_dataset = STL10RawUnlabeledDataset(args.data_root)
    combined_dataset = CombinedSTL10Dataset(
        real_imgs, real_labels, unlabeled_dataset, pseudo_indices, pseudo_lbls,
        synthetic_imgs=synth_imgs, synthetic_labels=synth_labels,
        transform=train_transform)
    del unlabeled_dataset
    gc.collect()

    real_indices = list(range(len(real_imgs)))
    pseudo_indices_list = list(range(len(real_imgs), len(combined_dataset)))
    # Real-only config (--no-pseudo without synthetic data): the shared v12
    # BalancedBatchSampler requires a non-empty second index pool; mirror the
    # real indices into it (dataset weights stay 1.0 since idx < n_real).
    if not pseudo_indices_list:
        pseudo_indices_list = list(real_indices)
    if is_ddp:
        import random
        random.Random(args.seed + rank).shuffle(real_indices)
        random.Random(args.seed + rank).shuffle(pseudo_indices_list)
        real_indices = real_indices[rank::world_size]
        pseudo_indices_list = pseudo_indices_list[rank::world_size]

    sampler = BalancedBatchSampler(
        real_indices, pseudo_indices_list,
        batch_size=args.batch_size // world_size if is_ddp else args.batch_size)
    optimal_config = find_optimal_dataloader_config(combined_dataset, sampler, is_ddp, rank)
    loader_kwargs = {"pin_memory": True}
    if optimal_config["num_workers"] > 0:
        loader_kwargs.update(num_workers=optimal_config["num_workers"],
                             persistent_workers=True,
                             prefetch_factor=3)
    trainloader = DataLoader(combined_dataset, batch_sampler=sampler, **loader_kwargs)

    _, testloader, stl_min, stl_max = get_stl10_dataloaders(
        args.data_root, batch_size=64)
    stl_min, stl_max = stl_min.to(device), stl_max.to(device)

    # ── 4. Model — RHANNext ─────────────────────────────────────────────────
    model = RHANNext(config=cfg).to(device, memory_format=torch.channels_last)

    # ── 5. Base checkpoint (strict=False — new pillar modules initialize) ───
    best_target_ckpt = args.target_ckpt or os.path.join(
        ckpt_dir, 'rhan_stl10_large_pseudolabel_best.pth')
    best_target_ckpt = ensure_checkpoint_exists(best_target_ckpt)
    if os.path.exists(best_target_ckpt):
        from checkpoint_utils import compat_load
        ckpt = compat_load(best_target_ckpt, map_location=device)
        for k in ('model_state_dict', 'model', 'state_dict'):
            if isinstance(ckpt, dict) and k in ckpt:
                ckpt = ckpt[k]
                break
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if rank == 0:
            print(f"Loaded base checkpoint: {best_target_ckpt}", flush=True)
            print(f"  Missing (new pillar modules): {len(missing)}", flush=True)
            print(f"  Unexpected keys: {len(unexpected)}", flush=True)
    elif rank == 0:
        print(f"Warning: base checkpoint not found — random init.", flush=True)

    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        print(f"RHANNext instantiated: {total:,} params ({cfg})", flush=True)

    if args.compile and rank == 0:
        print("Compiling model with torch.compile()...", flush=True)
    if args.compile:
        model = torch.compile(model, mode="default")

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, broadcast_buffers=False)
    elif torch.cuda.device_count() > 1 and not args.force_single_gpu:
        if rank == 0:
            print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)", flush=True)
        model = nn.DataParallel(model)

    raw_model = model.module if hasattr(model, 'module') else model
    if args.freeze_gaze:
        raw_model.freeze_gaze = True
        if rank == 0:
            print("  ISOLATION TEST: gaze frozen to center (0,0)", flush=True)

    # ── 6. Curriculum / resume / optimizer — identical to v12 ───────────────
    scaler = GradScaler('cuda')
    best_acc = 0.0
    start_epoch = 1
    checkpoint_data = None
    optimizer = None
    scheduler = None
    current_phase_start = None

    best_path = os.path.join(ckpt_dir, f'{args.ckpt_name}_best.pth')
    rolling_path = os.path.join(ckpt_dir, f'{args.ckpt_name}_rolling.pth')

    # Code identity of THIS run, recorded in every checkpoint and enforced at
    # resume: a checkpoint written by different code must never be resumed
    # (the 2026-08-12 stale-resume bug that invalidated the Stage 2 smoke).
    code_commit = current_code_commit()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from google.colab import userdata
            hf_token = userdata.get('HF_TOKEN')
        except Exception:
            pass
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass

    # Mandatory HF resume gate (same semantics as v12 — never silently
    # restart; once a rolling checkpoint exists on HF we restore or abort).
    local_epoch = -1
    if not args.force_restart:
        if os.path.exists(rolling_path):
            try:
                from checkpoint_utils import compat_load
                local_epoch = compat_load(rolling_path, map_location='cpu').get('epoch', -1)
            except Exception:
                local_epoch = -1

        hf_rolling_exists = hf_listing_ok = False
        if rank == 0:
            try:
                from huggingface_hub import HfApi
                rolling_filename = f"{args.ckpt_name}_rolling.pth"
                hf_files = HfApi(token=hf_token).list_repo_files(
                    repo_id='FerrariKazu/rhan-checkpoints-rolling',
                    repo_type='dataset')
                hf_rolling_exists = rolling_filename in hf_files
                hf_listing_ok = True
            except Exception as e:
                print(f"Hugging Face repo listing failed: {e}", flush=True)

        if rank == 0 and hf_rolling_exists:
            print("Hugging Face has a rolling checkpoint — resume is MANDATORY.", flush=True)
            last_err = None
            for attempt in range(1, 4):
                try:
                    from huggingface_hub import hf_hub_download
                    from checkpoint_utils import compat_load
                    temp = hf_hub_download(
                        repo_id='FerrariKazu/rhan-checkpoints-rolling',
                        filename=rolling_filename, repo_type='dataset',
                        token=hf_token)
                    remote_epoch = compat_load(temp, map_location='cpu').get('epoch', -1)
                    if remote_epoch >= local_epoch:
                        os.makedirs(os.path.dirname(rolling_path), exist_ok=True)
                        shutil.copy(temp, rolling_path)
                        local_epoch = remote_epoch
                        print(f"  Synchronized HF checkpoint (Epoch {remote_epoch})", flush=True)
                    break
                except Exception as e:
                    last_err = e
                    print(f"  HF resume attempt {attempt}/3 failed: {e}", flush=True)
                    if attempt < 3:
                        time.sleep(15 * attempt)
            if not os.path.exists(rolling_path):
                print(f"\n[FATAL] {rolling_filename} exists on HF but could not be "
                      f"restored ({last_err}). Aborting instead of a silent restart.",
                      flush=True)
                sys.exit(1)
        elif rank == 0 and not hf_listing_ok and not os.path.exists(rolling_path):
            print(f"\n[FATAL] Could not verify HF rolling checkpoint — no local copy "
                  f"exists. Aborting rather than silently restarting.", flush=True)
            sys.exit(1)

        if os.path.exists(rolling_path):
            from checkpoint_utils import compat_load
            checkpoint_data = compat_load(rolling_path, map_location=device)
            # Code-identity resume guard (2026-08-12 stale-resume bug): a
            # checkpoint written by DIFFERENT code must never be resumed — the
            # invalid Stage 2 smoke resumed old dead-head weights at epoch 12
            # and produced a meaningless health-gate verdict. Legacy
            # checkpoints (no recorded commit) are refused too: they are by
            # definition older than this guard. POLICY NOTE: this applies to
            # the rolling RESUME path only — base-checkpoint loads
            # (--target-ckpt) and eval loads are unaffected, so the legacy
            # AIS-v1 artifacts still work as Stage 2's base.
            _commit_ok, _commit_why = resume_commit_ok(checkpoint_data, code_commit)
            if not _commit_ok:
                print(f"\n[FATAL] Refusing to resume {rolling_path}:", flush=True)
                print(f"  {_commit_why}", flush=True)
                print("  Resuming across a code change silently invalidates the "
                      "run. Delete the stale\n  rolling/best artifacts (locally "
                      "and on HF) and re-run for a genuine cold start.",
                      flush=True)
                sys.exit(2)
            raw_model.load_state_dict(checkpoint_data['model'])
            best_acc = checkpoint_data.get('best_acc', 0.0)
            start_epoch = checkpoint_data['epoch'] + 1
            if rank == 0:
                print(f"Resuming from Epoch {start_epoch} "
                      f"(best val {best_acc:.2f}%)", flush=True)
                # ALSO restore the *_best.pth from HF (best-effort). A fresh
                # session has no local best artifact, and without this, if no
                # epoch beats the restored best_acc the best is never written
                # locally and never re-synced — Step C's eval then has nothing
                # to point at (the 2026-08-08 Kaggle Step C runtime error
                # this fixes).
                if not os.path.exists(best_path):
                    try:
                        from huggingface_hub import hf_hub_download
                        temp_best = hf_hub_download(
                            repo_id='FerrariKazu/rhan-checkpoints',
                            filename=f"{args.ckpt_name}_best.pth",
                            repo_type='dataset', token=hf_token)
                        compat_load(temp_best, map_location='cpu')  # loadable?
                        os.makedirs(os.path.dirname(best_path), exist_ok=True)
                        shutil.copy(temp_best, best_path)
                        print(f"  Restored best checkpoint from HF "
                              f"({os.path.basename(best_path)})", flush=True)
                    except Exception as e:
                        print(f"  WARNING: no *_best.pth on HF to restore "
                              f"({e}) — finalize will fall back to the "
                              f"final-epoch model if nothing improves.",
                              flush=True)
    elif rank == 0:
        print("--force-restart: starting from Epoch 1.", flush=True)

    # ── Telemetry baseline carried across resume boundaries (2026-08-14) ────
    # The health gate's trend check compares the FIRST logged epoch of the RUN
    # against the last. The rolling checkpoint stores the run's first-epoch
    # summary ('first_epoch_diag'); a resumed session prepends it to
    # --diag-json so rows[0] is always the true epoch-1 baseline — never a
    # session-local epoch that makes the gate compare a number to itself (the
    # 2026-08-13 Stage 2 verdict "epoch 15 0.6906 -> epoch 15 0.6906, ratio
    # 1.00" was exactly that self-comparison, not evidence about the head).
    first_epoch_diag = (checkpoint_data.get('first_epoch_diag')
                        if isinstance(checkpoint_data, dict) else None)

    # Fresh-start telemetry: a cold start (start_epoch == 1) must NOT append to
    # a diag jsonl left behind by a previous session/code — the health gate's
    # epoch-1 reference would be polluted (the 2026-08-12 gate compared the old
    # code's 0.2848 against the new code's 1.0144 within one history). Skipped
    # in dry-run (a validation invocation must never delete telemetry).
    if (not args.dry_run and start_epoch == 1 and args.diag_json
            and os.path.exists(args.diag_json)):
        try:
            os.remove(args.diag_json)
            if rank == 0:
                print(f"  Fresh start: cleared stale {args.diag_json}", flush=True)
        except OSError as e:
            if rank == 0:
                print(f"  WARNING: could not clear stale {args.diag_json}: {e}",
                      flush=True)

    # Resume telemetry continuity: prepend the carried epoch-1 baseline so a
    # session that resumes and logs only its own epochs still yields a diag
    # whose rows[0] is the run's first logged epoch (the gate needs >= 2
    # DISTINCT epochs; a single-epoch diag degenerates the trend check into a
    # self-comparison — the 2026-08-13 verdict). Skipped in dry-run.
    if (not args.dry_run and start_epoch > 1 and args.diag_json
            and first_epoch_diag):
        ensure_diag_baseline(args.diag_json, first_epoch_diag)

    # ── 7. Training loop ────────────────────────────────────────────────────
    WARMUP_EPOCHS = 5
    diagnostics = RHANNextEpochDiagnostics(max_steps=cfg.max_foraging_steps)

    last_epoch = start_epoch - 1  # honest label for the finalize fallback
    for epoch in range(start_epoch, args.max_epochs + 1):
        last_epoch = epoch
        t0 = time.time()
        diagnostics.reset()

        for p_start, p_end, eps, beta, steps, lr in CURRICULUM:
            if p_start <= epoch <= p_end:
                phase_params = (eps, beta, steps)
                phase_lr = lr
                if current_phase_start != p_start:
                    current_phase_start = p_start
                    optimizer = build_next_optimizer(
                        raw_model, phase_lr, args.hpc_lr_mult)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=p_end - p_start + 1,
                        eta_min=phase_lr * 0.1)
                    if (epoch == start_epoch and checkpoint_data is not None
                            and 'optimizer' in checkpoint_data):
                        # Guarded resume (restore_optimizer_from_checkpoint):
                        # pre-2026-08-13 checkpoints carry a single-group
                        # optimizer state (no HPC group) — loading one into
                        # the two-group optimizer would ValueError or silently
                        # misalign groups; and a same-commit flag change (e.g.
                        # --hpc-lr-mult) would silently restore the OLD head
                        # lr. On any mismatch we fall back to a fresh
                        # optimizer with a loud warning.
                        restore_optimizer_from_checkpoint(
                            optimizer, scheduler, checkpoint_data, rank)
                    if rank == 0:
                        print(f"\n--- Epoch {epoch}: phase {p_start}-{p_end} "
                              f"(lr={phase_lr}) ---", flush=True)
                break
        eps, beta, steps = phase_params

        if epoch <= WARMUP_EPOCHS:
            if rank == 0:
                print("Warmup: freezing active-inference + pillar components, "
                      "training generative prior.", flush=True)
            set_new_component_training(raw_model, False)
            for name, param in raw_model.named_parameters():
                if 'generative_prior' in name:
                    param.requires_grad = True
        else:
            if rank == 0:
                print("Main Phase: training all components", flush=True)
            set_new_component_training(raw_model, True)

        model.train()
        total_loss = n_total = correct = 0
        total_batch_size = args.batch_size * world_size if is_ddp else args.batch_size
        num_batches = (min(len(trainloader), 600) if args.fixed_samples_per_epoch <= 0
                       else max(1, args.fixed_samples_per_epoch // total_batch_size))
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (imgs, lbls, weights) in enumerate(trainloader):
            if batch_idx >= num_batches:
                break
            is_accum = ((batch_idx + 1) % args.accum_steps != 0
                        and (batch_idx + 1) < num_batches)
            if is_ddp and is_accum:
                sync_ctx = model.no_sync()
            else:
                from contextlib import nullcontext
                sync_ctx = nullcontext()
            with sync_ctx:
                imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)

                if epoch <= WARMUP_EPOCHS:
                    with autocast('cuda'):
                        logits, traj_c = model(imgs, return_trajectory=True)
                        l_trades = nn.CrossEntropyLoss()(logits, lbls)
                        l_recon = raw_model.get_reconstruction_loss(imgs, (logits, traj_c))
                        l_hpc = raw_model.get_hpc_loss(imgs, (logits, traj_c))
                        loss = (l_trades + args.w_recon * l_recon
                                + args.w_hpc * l_hpc) / args.accum_steps
                        beta_dyn = (beta * (0.5 + traj_c['precisions'][-1])
                                    if len(traj_c['precisions']) > 0
                                    else torch.full((imgs.shape[0],), beta, device=device))
                    scaler.scale(loss).backward()
                    diagnostics.update(beta_dyn, traj_c, lbls)
                else:
                    # ── PGD adversarial examples (identical to v12) ─────────
                    raw_model.eval()
                    with torch.no_grad():
                        with autocast('cuda'):
                            probs_c = F.softmax(raw_model(imgs).float(), dim=1)
                    x_adv = torch.clamp(
                        imgs.clone().detach() + 0.001 * torch.randn_like(imgs),
                        stl_min, stl_max)
                    for _ in range(steps):
                        x_adv.requires_grad_(True)
                        with torch.enable_grad():
                            with autocast('cuda'):
                                logits_a_pgd = raw_model(x_adv)
                                loss_adv = F.kl_div(
                                    F.log_softmax(logits_a_pgd.float(), dim=1),
                                    probs_c, reduction='batchmean')
                        grad = torch.autograd.grad(loss_adv, x_adv)[0]
                        x_adv = x_adv.detach() + (eps / steps) * grad.sign()
                        x_adv = torch.clamp(
                            imgs + torch.clamp(x_adv - imgs, -eps, eps),
                            stl_min, stl_max).detach()
                    model.train()

                    # ── RHAN-Next loss (superset of v12) ────────────────────
                    with autocast('cuda'):
                        (l_trades, traj_c, traj_a, beta_dyn, l_recon, l_hpc,
                         w_recon_eff) = dynamic_trades_loss_next(
                            raw_model, imgs, lbls, weights, x_adv, beta,
                            args.w_recon, args.w_hpc,
                            precision_recon_enabled=cfg.ais_precision_recon_enabled)
                        loss = (args.w_trades * l_trades
                                + w_recon_eff * l_recon
                                + args.w_hpc * l_hpc) / args.accum_steps
                    scaler.scale(loss).backward()
                    diagnostics.update(beta_dyn, traj_c, lbls)

            if (batch_idx + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                clip_grad_per_group(optimizer, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            B = imgs.size(0)
            total_loss += l_trades.item() * B
            with torch.no_grad():
                with autocast('cuda'):
                    logits_c_acc = model(imgs)
            correct += logits_c_acc.argmax(1).eq(lbls).sum().item()
            n_total += B

            if rank == 0 and batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{num_batches} | "
                      f"Loss: {l_trades.item():.4f} | β_dyn: {beta_dyn.mean():.3f} "
                      f"| Steps: {traj_c['steps']}", flush=True)
            if args.dry_run and rank == 0:
                print("Dry-run: 1 training step OK.", flush=True)
                break

        if num_batches % args.accum_steps != 0:
            scaler.unscale_(optimizer)
            clip_grad_per_group(optimizer, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # ── Validation (clean test) ─────────────────────────────────────────
        val_acc = 0.0
        if rank == 0 and not args.dry_run:
            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for v_imgs, v_lbls in testloader:
                    v_imgs, v_lbls = v_imgs.to(device), v_lbls.to(device)
                    with autocast('cuda'):
                        logits = model(v_imgs)
                    val_correct += logits.argmax(1).eq(v_lbls).sum().item()
                    val_total += v_lbls.size(0)
            val_acc = 100.0 * val_correct / val_total

        if is_ddp:
            import torch.distributed as dist
            va = torch.tensor([val_acc], device=device)
            dist.broadcast(va, src=0)
            val_acc = va.item()

        marker = ''
        if val_acc > best_acc:
            best_acc = val_acc
            marker = ' ★'
            if rank == 0:
                torch.save({'model': raw_model.state_dict(),
                            'config': cfg.to_dict(),
                            'arch': 'rhan_next',
                            'code_commit': code_commit}, best_path)
                sync_to_hf(best_path)

        if rank == 0:
            t_epoch = time.time() - t0
            total_images = n_total * world_size if is_ddp else n_total
            ips = total_images / t_epoch if t_epoch > 0 else 0
            eph = 3600.0 / t_epoch if t_epoch > 0 else 0
            print(f"Epoch {epoch:03d}/{args.max_epochs:03d} (ε={eps:.3f}) | "
                  f"Loss:{total_loss/max(n_total,1):.3f} | "
                  f"TrAcc:{100.*correct/max(n_total,1):.1f}% TeAcc:{val_acc:.1f}% | "
                  f"Throughput:{ips:.2f} img/sec ({eph:.2f} epochs/hour) | "
                  f"{t_epoch:.0f}s{marker}", flush=True)
            diagnostics.report(epoch, eps)
            if args.diag_json and rank == 0:
                try:
                    d = diagnostics.summary_dict(
                        epoch, eps,
                        tr_acc=100.0 * correct / max(n_total, 1),
                        te_acc=val_acc)
                    with open(args.diag_json, 'a') as f:
                        f.write(json.dumps(d) + '\n')
                    # Capture the run's first logged epoch ONCE (cold start:
                    # epoch 1; a resume with no carried baseline: the first
                    # epoch of this session). Never overwrite a carried
                    # baseline — the gate's trend reference must stay the
                    # run's epoch 1 across every subsequent session.
                    if first_epoch_diag is None:
                        first_epoch_diag = d
                except OSError as e:
                    print(f"  WARNING: could not write --diag-json: {e}",
                          flush=True)

            torch.save({'epoch': epoch,
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc': best_acc,
                        'config': cfg.to_dict(),
                        'arch': 'rhan_next',
                        'first_epoch_diag': first_epoch_diag,
                        'code_commit': code_commit}, rolling_path)
            sync_to_hf(rolling_path)
            gc.collect()
            torch.cuda.empty_cache()

        if args.dry_run:
            break
        if is_ddp:
            import torch.distributed as dist
            dist.barrier()

    if rank == 0:
        print("\nFinalizing Hugging Face sync...", flush=True)
        best_fallback_used = False
        if not os.path.exists(best_path):
            # Last-resort: nothing improved this session AND no *_best.pth on
            # HF to restore. Write the final-epoch model as the best artifact
            # so Step C's eval always has a checkpoint, then sync it to HF.
            best_fallback_used = True
            print(f"  WARNING: {os.path.basename(best_path)} does not exist — "
                  f"no epoch beat the restored best ({best_acc:.2f}%) and no "
                  f"HF best was available. Writing the FINAL-EPOCH model "
                  f"(epoch {last_epoch}) as the best artifact.", flush=True)
            torch.save({'model': raw_model.state_dict(),
                        'config': cfg.to_dict(),
                        'arch': 'rhan_next',
                        'code_commit': code_commit}, best_path)
        sync_to_hf(best_path)
        wait_for_hf_sync()
        print(f"{'═'*60}")
        if best_fallback_used:
            # The artifact is NOT the peak-val model — say so explicitly so no
            # reader or downstream parser mistakes it for the 54.05% weights.
            print(f"  Training complete. Peak best val: {best_acc:.2f}% — but "
                  f"those weights are NOT available (lost to a session wipe).")
            print(f"  {os.path.basename(best_path)} = FINAL-EPOCH model "
                  f"(epoch {last_epoch}), written as the eval artifact.")
        else:
            print(f"  Training complete. Best: {best_acc:.2f}% -> {best_path}")
        print(f"  Config: {cfg}")
        print(f"{'═'*60}")


if __name__ == '__main__':
    main()
