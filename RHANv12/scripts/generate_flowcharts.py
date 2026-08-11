#!/usr/bin/env python3
"""
generate_flowcharts.py
======================
Pure-matplotlib schematic figures for the report:

  evolution_timeline.png      -- v1 → v12 version history
  bio_mapping.png             -- human visual system ↔ RHAN module mapping
  training_pipeline.png       -- data → PGD → loss → checkpoint → HF
  loss_pipeline.png           -- the two-term v12 loss, with sub-terms
  inference_pipeline.png      -- inference stages
  active_inference_cycle.png  -- Observe→Predict→Error→Precision→Update loop
  comp_graph.png              -- forward / feedback / gradient-flow graph
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _report_common as C

import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _arrow(ax, x1, y1, x2, y2, color, lw=1.8, style='-|>', ls='-'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=15, lw=lw, color=color,
                        linestyle=ls, zorder=2)
    ax.add_patch(a)


def _box(ax, x, y, title, sub='', w=2.6, h=0.9, fc='#ffffff', ec=None,
         fs=10, sfs=8, bold=True):
    ec = ec or C.PALETTE['ink']
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.3,
                       zorder=3)
    ax.add_patch(b)
    ty = y + (0.12 if sub else 0.0)
    ax.text(x, ty, title, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=C.PALETTE['ink'],
            zorder=4)
    if sub:
        ax.text(x, y - 0.17, sub, ha='center', va='center', fontsize=sfs,
                color=C.PALETTE['gray'], zorder=4)


def _figure(w, h, name, title=None):
    plt = C.set_style()
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    return fig, ax


def evolution_timeline():
    fig, ax = _figure(12, 14, 'evolution_timeline.png')
    P = C.PALETTE
    stages = [
        ('v1–v2', 'Recurrent top-down feedback\n+ adversarial curriculum',
         'First evidence recurrence beats feedforward; fine-tuning harms', P['indigo']),
        ('v3', 'Joint scratch training,\nventral/dorsal split',
         'εthresh=0.090 — key methodological win', P['indigo']),
        ('v4', 'Multi-scale gated feedback,\nactive CLIP',
         'Ongoing CLIP loss regresses geometry', P['gray']),
        ('v5', 'Frequency separation\n(M-pathway gates)',
         'wL>wH emerges; εthresh=0.103', P['teal']),
        ('v6', 'Dynamic gating + ACT',
         'Regressed — architecture was not the bottleneck', P['gray']),
        ('v5-TRADES', 'TRADES objective',
         'εthresh=0.1113 — principled loss beats arch tweaks', P['teal']),
        ('Hardened', 'Class-hardened TRADES + margin',
         'εthresh=0.1246 but car/truck still collapse', P['teal']),
        ('curriculum', '3-phase epsilon curriculum',
         'εthresh=0.1850 — CIFAR-10 ceiling reached', P['teal']),
        ('UNIFIED', '96×96 STL-10, from scratch',
         'Phase-0 pseudo-label pretraining; 72.9% clean', P['indigo']),
        ('Large', '55.6M params + 41.6K pseudo-labels',
         '+11.5 pp clean, +1.3 pp AA robust (Finding 14)', P['indigo']),
        ('v10', 'Active inference: precision + foraging + halt',
         'Pre-registered FEP hypothesis; εthresh flat (Finding 16)', P['gray']),
        ('v11', 'Parafoveal + gate + generative prior,\nimage-space error',
         'Multi-resolution fix; 3 losses still sabotage (Finding 17)', P['gray']),
        ('v12', 'LOCKED-IN: fixed T=4, recon-guided gaze,\n2-term loss',
         'Deletes the 3 opposing losses; real recon gradients; 75.4M p', P['rust']),
    ]
    y = 13.2
    x0, x1 = 3.0, 8.9
    for i, (v, what, why, col) in enumerate(stages):
        _box(ax, x0, y, v, what, w=2.9, h=0.82, fc='#fbfbfe', ec=col, fs=9.5,
             sfs=7.8)
        ax.text(x1, y, why, ha='left', va='center', fontsize=8.2,
                color=C.PALETTE['ink'])
        if i < len(stages) - 1:
            _arrow(ax, x0, y - 0.41, x0, y - 0.82 - 0.41, col)
        y -= 1.02
    fig.savefig(os.path.join(C.FIG_DIR, 'evolution_timeline.png'))
    print('[flowcharts] evolution_timeline.png', flush=True)


def bio_mapping():
    fig, ax = _figure(13, 7.2, 'bio_mapping.png',
                      'Human visual system ↔ RHAN-v12 module mapping')
    P = C.PALETTE
    left = ['Retina (fovea + periphery)', 'LGN', 'V1 (frequency channels)',
            'V2/V4 (form & pattern)', 'IT cortex (object identity)',
            'Working memory (belief)', 'Motor planning (gaze saccades)',
            'Perceptual inference loop']
    right = ['foveal_sample + Foveal/ParafovealStream',
             'WideSEConvStemLarge (shortcut fusion)',
             'Ventral/Dorsal split transformer',
             'RecurrentFeedbackLarge + predictive coder',
             'CLS readout + classifier',
             'belief state $s_t \\in \\mathbb{R}^{512}$, precision-weighted',
             'action_init + Eq. II gaze update (recon-guided)',
             'T=4 active-inference loop (locked-in)']
    y = 6.5
    for l, r in zip(left, right):
        _box(ax, 2.6, y, l, w=3.3, h=0.62, fc='#eef6f6', ec=P['teal'], fs=8.6)
        _box(ax, 10.0, y, r, w=4.4, h=0.62, fc='#f3f0fb', ec=P['indigo'],
             fs=8.6)
        _arrow(ax, 4.25, y, 7.8 - 0.0, y, P['gray'], lw=1.2)
        if y > 1.4:
            _arrow(ax, 2.6, y - 0.31, 2.6, y - 0.62 - 0.31, P['teal'], lw=1.0)
            _arrow(ax, 10.0, y - 0.31, 10.0, y - 0.62 - 0.31, P['indigo'],
                   lw=1.0)
        y -= 0.85
    fig.savefig(os.path.join(C.FIG_DIR, 'bio_mapping.png'))
    print('[flowcharts] bio_mapping.png', flush=True)


def training_pipeline():
    fig, ax = _figure(13, 4.6, 'training_pipeline.png',
                      'RHAN-v12 training pipeline')
    P = C.PALETTE
    _box(ax, 1.0, 3.2, '5K labeled', 'STL-10 train', w=1.9, fc='#eef6f6')
    _box(ax, 3.1, 3.2, '100K unlabeled', 'conf≥0.65 → ~46K pseudo',
         w=2.0, fc='#eef6f6')
    _box(ax, 5.2, 3.2, '115K synthetic', 'optional (Step 0b)', w=1.9,
         fc='#fbf6ee')
    _box(ax, 3.1, 1.9, 'CombinedSTL10Dataset', 'BalancedBatchSampler\n'
         'weights 1.0 / 0.5 / 0.5', w=3.6, fc='#f2f4f8', fs=9)
    _box(ax, 7.6, 1.9, 'PGD attack', 'ε·step, KL vs clean softmax\n'
         'steps from curriculum', w=2.6, fc='#fdf3f0', fs=9)
    _box(ax, 10.3, 1.9, '2-term loss', 'w$_t$·L$_t$ + w$_r$·L$_r$',
         w=2.0, fc='#f3f0fb', fs=9)
    _box(ax, 10.3, 0.5, 'SGD + CosineAnnealing', 'warmup 5 ep, phases',
         w=2.0, fc='#f2f4f8', fs=9)
    _box(ax, 7.6, 0.5, 'checkpoints', 'rolling / best\nHF async sync',
         w=2.6, fc='#f2f4f8', fs=9)
    _box(ax, 4.5, 0.5, 'curriculum', '1–20: ε=.031, β=2\n21–40: .062, 2\n'
         '41–60: .094, 2.5', w=3.0, fc='#fbf6ee', fs=8.6)
    _arrow(ax, 1.95, 3.2, 2.1, 3.2, P['teal'])
    _arrow(ax, 4.1, 3.2, 4.6, 3.2, P['teal'])
    _arrow(ax, 6.15, 3.2, 6.3, 3.2, P['amber'])
    _arrow(ax, 3.1, 2.75, 3.1, 2.45, P['ink'])
    _arrow(ax, 4.9, 1.9, 6.3, 1.9, P['ink'])
    _arrow(ax, 8.9, 1.9, 9.3, 1.9, P['rust'])
    _arrow(ax, 10.3, 1.35, 10.3, 1.05, P['ink'])
    _arrow(ax, 9.3, 0.5, 8.9, 0.5, P['ink'])
    _arrow(ax, 6.3, 0.5, 6.0, 0.5, P['amber'])
    fig.savefig(os.path.join(C.FIG_DIR, 'training_pipeline.png'))
    print('[flowcharts] training_pipeline.png', flush=True)


def loss_pipeline():
    fig, ax = _figure(13, 4.8, 'loss_pipeline.png',
                      'v12 loss — exactly two terms (auxiliary losses deleted)')
    P = C.PALETTE
    _box(ax, 1.6, 3.4, 'x (clean)', w=1.6, fc='#eef6f6', fs=9)
    _box(ax, 1.6, 2.2, 'x$_{adv}$ (PGD)', w=1.6, fc='#fdf3f0', fs=9)
    _box(ax, 3.6, 3.4, 'CE(clean, y)', w=2.0, fc='#f3f0fb', fs=9)
    _box(ax, 3.6, 2.2, 'KL(adv || clean)', w=2.0, fc='#f3f0fb', fs=9)
    _box(ax, 6.2, 2.8, 'β$_{dyn}$ = β(0.5+Π$_D$)', w=2.3, fc='#fbf6ee', fs=9)
    _box(ax, 8.8, 2.8, 'L$_{trades}$ = (CE + β·KL)·w', w=2.8, fc='#f3f0fb', fs=9)
    _box(ax, 3.6, 0.7, 'L$_{recon}$ = mean MSE(x$_t$, x̂$_t$)', w=3.6,
         fc='#f3f0fb', fs=9)
    _box(ax, 8.8, 0.7, 'w$_t$·L$_{trades}$ + w$_r$·L$_{recon}$', w=2.9,
         fc='#fbfbfe', ec=P['rust'], fs=9.5)
    _box(ax, 11.9, 2.8, 'backward', 'reaches generative\nprior + gaze path',
         w=1.7, fc='#eef6f6', fs=8.6)
    _arrow(ax, 2.4, 3.4, 2.6, 3.4, P['teal'])
    _arrow(ax, 2.4, 2.2, 2.6, 2.2, P['rust'])
    _arrow(ax, 4.6, 3.4, 5.1, 3.0, P['ink'])
    _arrow(ax, 4.6, 2.2, 5.1, 2.6, P['ink'])
    _arrow(ax, 7.35, 2.8, 7.3, 2.8, P['amber'])
    _arrow(ax, 7.35, 2.8, 7.3, 2.8, P['amber'])
    _arrow(ax, 5.6, 0.7, 7.4, 0.7, P['ink'])
    _arrow(ax, 7.45, 0.7, 7.35, 0.7, P['ink'])
    _arrow(ax, 8.8, 1.9, 8.8, 1.25, P['ink'])
    _arrow(ax, 10.25, 2.8, 11.05, 2.8, P['rust'])
    _arrow(ax, 10.25, 0.7, 11.05, 0.7, P['rust'])
    ax.text(6.2, 3.9, 'Π$_D$ from ImageSpacePrecision forward pass '
                      '(unsupervised, Eq. III)', ha='center', fontsize=8.4,
            color=P['gray'])
    ax.text(3.6, 4.2, 'real w=1.0 · pseudo/synth w=0.5', ha='center',
            fontsize=8.4, color=P['gray'])
    fig.savefig(os.path.join(C.FIG_DIR, 'loss_pipeline.png'))
    print('[flowcharts] loss_pipeline.png', flush=True)


def inference_pipeline():
    fig, ax = _figure(13, 4.4, 'inference_pipeline.png',
                      'RHAN-v12 inference (locked-in forward pass)')
    P = C.PALETTE
    steps = ['peripheral pass', 'belief init $s_0$', 'forage t=0..3',
             'accumulate belief', 'readout']
    xs = [1.2, 3.4, 6.0, 8.6, 11.2]
    for x, s in zip(xs, steps):
        _box(ax, x, 2.6, s, w=2.0, fc='#f2f4f8', fs=9)
    _box(ax, 6.0, 0.9, 'foveal sample → encode → gate → predict → '
         'Π$_D$ → update s → gaze', w=7.4, fc='#fbfbfe', ec=P['indigo'],
         fs=8.6)
    _arrow(ax, 2.2, 2.6, 2.4, 2.6, P['ink'])
    _arrow(ax, 4.4, 2.6, 5.0, 2.6, P['ink'])
    _arrow(ax, 7.0, 2.6, 7.6, 2.6, P['ink'])
    _arrow(ax, 9.6, 2.6, 10.2, 2.6, P['ink'])
    _arrow(ax, 6.0, 2.05, 6.0, 1.45, P['indigo'])
    _arrow(ax, 6.0, 0.9, 6.0, 0.35, P['gray'], ls='--')
    ax.text(6.0, 0.15, 'T=4 unconditional; final crop is classification',
            ha='center', fontsize=8.2, color=P['gray'])
    ax.text(6.0, 3.4, 'no halt network · gaze frozen only in the '
                      'run-B isolation variant', ha='center', fontsize=8.4,
            color=P['gray'])
    fig.savefig(os.path.join(C.FIG_DIR, 'inference_pipeline.png'))
    print('[flowcharts] inference_pipeline.png', flush=True)


def active_inference_cycle():
    fig, ax = _figure(9.5, 8.5, 'active_inference_cycle.png',
                      'The active-inference cycle (Eq. II & III)')
    P = C.PALETTE
    stages = [
        ('Observe', 'foveal_sample at $a_t$', '#eef6f6', P['teal']),
        ('Predict', 'prior $\\hat x = G(s_t)$', '#f3f0fb', P['indigo']),
        ('Error', '$\\|x_t - \\hat x_t\\|^2$ (pixel)', '#fdf3f0', P['rust']),
        ('Precision', '$\\Pi_D$: $\\tau \\dot\\Pi = e^2 - \\Pi$', '#fbf6ee',
         P['amber']),
        ('Belief', '$s \\leftarrow (1-\\Pi)s + \\Pi c$', '#eef6f6', P['teal']),
        ('Gaze', '$a \\leftarrow a + \\eta\\,\\text{norm}(\\lambda\\nabla R'
         '+(1{-}\\lambda)\\nabla F)$', '#fbf6ee', P['amber']),
    ]
    n = len(stages)
    cx, cy, R = 4.75, 4.2, 3.0
    import math
    for i, (title, sub, fc, ec) in enumerate(stages):
        ang = math.pi / 2 - 2 * math.pi * i / n
        x, y = cx + R * math.cos(ang), cy + R * math.sin(ang)
        _box(ax, x, y, title, sub, w=2.7, h=1.15, fc=fc, ec=ec, fs=10.5,
             sfs=8.2)
        nx = cx + R * math.cos(ang - 2 * math.pi / n)
        ny = cy + R * math.sin(ang - 2 * math.pi / n)
        _arrow(ax, x, y, nx, ny, P['gray'], lw=1.6)
    ax.text(cx, cy, 'repeat\nT=4', ha='center', va='center', fontsize=12,
            fontweight='bold', color=P['ink'])
    fig.savefig(os.path.join(C.FIG_DIR, 'active_inference_cycle.png'))
    print('[flowcharts] active_inference_cycle.png', flush=True)


def comp_graph():
    fig, ax = _figure(13, 5.4, 'comp_graph.png',
                      'Computational graph: forward, feedback and gradient flow')
    P = C.PALETTE
    _box(ax, 1.2, 3.6, 'x', w=1.2, fc='#f2f4f8', fs=10)
    _box(ax, 3.0, 4.3, 'stem', w=1.5, fc='#eef6f6', fs=9)
    _box(ax, 4.8, 4.3, 'tokeniser', w=1.5, fc='#eef6f6', fs=9)
    _box(ax, 6.6, 4.6, 'ventral|dorsal', w=1.9, fc='#f3f0fb', fs=9)
    _box(ax, 6.6, 3.4, 'feedback', '×2', w=1.9, fc='#f3f0fb', fs=9)
    _box(ax, 8.8, 4.0, 'peripheral_proj', w=1.9, fc='#eef6f6', fs=8.6)
    _box(ax, 8.8, 2.6, 'active loop', 'foveal·gate·prior\nΠ$_D$·belief·gaze',
         w=2.3, fc='#fbfbfe', ec=P['indigo'], fs=8.6)
    _box(ax, 11.7, 4.0, 'classifier', w=1.5, fc='#f2f4f8', fs=9)
    _box(ax, 11.7, 2.6, 'L$_{trades}$', w=1.5, fc='#f3f0fb', fs=9)
    _box(ax, 11.7, 1.2, 'L$_{recon}$', w=1.5, fc='#f3f0fb', fs=9)
    _box(ax, 11.7, 0.0, 'backward', w=1.5, fc='#fdf3f0', fs=9)
    _arrow(ax, 1.8, 3.6, 2.25, 4.1, P['ink'])
    _arrow(ax, 3.75, 4.3, 4.05, 4.3, P['teal'])
    _arrow(ax, 5.55, 4.3, 5.65, 4.4, P['indigo'])
    _arrow(ax, 7.55, 4.6, 7.55, 3.95, P['indigo'])
    _arrow(ax, 7.55, 3.4, 7.55, 2.6, P['gray'], ls='--')
    ax.text(7.75, 3.0, 'gaze-grad path\n(autograd, detached)', fontsize=7.2,
            color=P['gray'])
    _arrow(ax, 8.6, 4.0, 8.8, 4.0, P['teal'])
    _arrow(ax, 8.8, 3.4, 8.8, 3.15, P['ink'])
    _arrow(ax, 9.95, 2.6, 11.0, 3.8, P['teal'])
    _arrow(ax, 11.7, 3.45, 11.7, 3.15, P['ink'])
    _arrow(ax, 11.7, 2.05, 11.7, 1.75, P['indigo'])
    _arrow(ax, 11.7, 0.65, 11.7, 0.55, P['rust'], lw=2.2)
    ax.text(9.2, 0.4, 'gradients reach: stem·transformers·foveal·prior·gate\n'
            'recon term reaches generative_prior (v12 fix)',
            fontsize=8.2, color=P['gray'], ha='center')
    fig.savefig(os.path.join(C.FIG_DIR, 'comp_graph.png'))
    print('[flowcharts] comp_graph.png', flush=True)


def tensor_flow():
    """Per-stage tensor dimensions through the v12 forward pass."""
    fig, ax = _figure(13, 5.6, 'tensor_flow.png',
                      'Tensor flow through RHAN-v12 (batch B, STL-10 96×96)')
    P = C.PALETTE
    rows = [
        ('x', '(B, 3, 96, 96)'),
        ('stem', '(B, 768, 12, 12)'),
        ('tokeniser', '(B, 145, 768)'),
        ('ventral | dorsal', '(B, 145, 384) × 2'),
        ('feedback (×2)', '(B, 145, 768)'),
        ('CLS → peripheral_proj', '(B, 768) → (B, 512)'),
        ('foveal loop t=0..3', '(B,3,48,48) → (B,512) → Π_D → (B,512)'),
        ('gaze update', '(B,2) + R(x,a): (B,48,48)'),
        ('belief_unproj', '(B,512) → (B,768)'),
        ('classifier', '(B,10)'),
    ]
    y = 5.2
    for name, shape in rows:
        _box(ax, 4.2, y, name, shape, w=4.6, h=0.44, fc='#fbfbfe',
             ec=P['indigo'], fs=8.8, sfs=8.0, bold=False)
        if y > 0.9:
            _arrow(ax, 4.2, y - 0.22, 4.2, y - 0.44 - 0.22, P['gray'], lw=1.0)
        y -= 0.48
    ax.text(8.6, 4.4, 'peripheral pass runs the full encoder once;\n'
            'the foveal loop re-encodes 48\u00b2 crops T=4 times;\n'
            'gaze gradients re-run foveal+prior 3×', fontsize=9,
            color=P['gray'], va='center')
    ax.text(8.6, 2.6, 'total: 45.6 GFLOPs / image\n(measured, hooks)',
            fontsize=10, color=P['teal'], va='center', fontweight='bold')
    fig.savefig(os.path.join(C.FIG_DIR, 'tensor_flow.png'))
    print('[flowcharts] tensor_flow.png', flush=True)


def _esc(s):
    """Escape underscores outside $...$ math segments for LaTeX."""
    out = []
    in_math = False
    for ch in s:
        if ch == '$':
            in_math = not in_math
        if ch == '_' and not in_math:
            out.append(r'\_')
        else:
            out.append(ch)
    return ''.join(out)


def data_tables():
    """Config + results tables derived from the recorded JSON artifacts."""
    import json

    # ---- curriculum / hyperparameters ----
    cur = [(1, 20, 0.031, 2.0, 'PGD-4', 3e-3, 'warmup 1-5'),
           (21, 40, 0.062, 2.0, 'PGD-4', 2e-3, ''),
           (41, 60, 0.094, 2.5, 'PGD-4', 1e-3, '')]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\begin{tabular}{r r r r r r l}", r"\toprule",
             r"\textbf{Phase} & \textbf{Epochs} & $\varepsilon$ (norm) & "
             r"$\beta$ & \textbf{PGD steps} & \textbf{LR} & \textbf{Note} \\\\",
             r"\midrule"]
    for i, (e1, e2, eps, b, steps, lr, note) in enumerate(cur, 1):
        lines.append(f"{i} & {e1}--{e2} & {eps} & {b} & {steps} & {lr} & "
                     f"{note} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                 r"\caption{RHAN-v12 curriculum (60-epoch primary schedule; "
                 r"Step 0b probes use 10 epochs; synthetic mixes rerun the same "
                 r"schedule with `--synthetic-data`).}",
                 r"\label{tab:curriculum}", r"\end{table}"])
    with open(os.path.join(C.TAB_DIR, 'curriculum.tex'), 'w') as f:
        f.write('\n'.join(lines))

    hyper = [('optimizer', 'SGD, momentum 0.9, weight_decay 1e-4, foreach'),
             ('scheduler', 'CosineAnnealingLR per phase, eta_min = lr/10'),
             ('batch', '8 (T4) / 16+ (A100); accum 32 $\rightarrow$ effective 256'),
             ('mixed precision', 'torch.amp autocast + GradScaler'),
             ('grad clip', 'clip_grad_norm_ 1.0'),
             ('warmup', 'epochs 1--5: clean CE + recon only; new heads frozen'),
             ('w_trades / w_recon', '0.55 / 0.10'),
             ('gaze_lambda', '0.5 (CLI-tunable, Eq. II v12)'),
             ('pseudo-labels', 'conf $\geq$ 0.65 on 100K unlabeled (~46K kept)'),
             ('loss weighting', 'real 1.0, pseudo/synthetic 0.5'),
             ('augmentation', 'RandomCrop(96, pad 12) + RandomHorizontalFlip'),
             ('checkpoints', 'rolling + best; async HF sync after each epoch')]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\begin{tabular}{l l}", r"\toprule",
             r"\textbf{Setting} & \textbf{Value} \\\\", r"\midrule"]
    for k, v in hyper:
        lines.append(f"{_esc(k)} & {_esc(v)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                 r"\caption{Training hyperparameters (from train\_rhan\_v12.py).}",
                 r"\label{tab:hyper}", r"\end{table}"])
    with open(os.path.join(C.TAB_DIR, 'hyperparams.tex'), 'w') as f:
        f.write('\n'.join(lines))

    # ---- isolation results table (from report JSON) ----
    with open(os.path.join(C.REPO_ROOT, 'report',
                           'isolation_sweep_results.json')) as f:
        iso = json.load(f)
    rows = [('run_a_norecon', 'Run A --- prior OFF (norecon)'),
            ('trades_large_baseline_in_run_a', 'TRADES Large baseline'),
            ('run_b_fixedgaze', 'Run B --- gaze frozen')]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\begin{tabular}{l r r r r}", r"\toprule",
             r"\textbf{Model} & $\varepsilon$=0.0 & 0.031 & 0.062 & 0.094 \\\\",
             r"\midrule"]
    for key, lab in rows:
        pts = iso['sweeps'][key]['points']
        accs = [p['acc_pct'] for p in pts]
        lines.append(f"{lab} & {accs[0]:.1f} & {accs[1]:.1f} & "
                     f"{accs[2]:.1f} & {accs[3]:.1f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                 r"\caption{Matched norm-space isolation sweeps (PGD-50, "
                 r"n=500, seed 42; Finding-17 convention; source: "
                 r"report/isolation\_sweep\_results.json).}",
                 r"\label{tab:isolation}", r"\end{table}"])
    with open(os.path.join(C.TAB_DIR, 'isolation.tex'), 'w') as f:
        f.write('\n'.join(lines))

    # ---- Finding-17 table ----
    rows17 = [('Loss-ablated v11 (ep54)', 'Real + 115K synth',
               51.7, 46.5, 38.8, 31.8),
              ('Null ablation (v11)', 'Real only', 47.9, 45.9, 42.6, 39.3),
              ('TRADES Large baseline', 'Real only', 52.8, 48.0, 40.3, 33.7)]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\begin{tabular}{l l r r r r}", r"\toprule",
             r"\textbf{Model} & \textbf{Data} & $\varepsilon$=0.0 & 0.031 "
             r"& 0.062 & 0.094 \\\\", r"\midrule"]
    for lab, data, *accs in rows17:
        lines.append(f"{lab} & {data} & {accs[0]:.1f} & {accs[1]:.1f} & "
                     f"{accs[2]:.1f} & {accs[3]:.1f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                 r"\caption{Finding-17: the architecture-only advantage and "
                 r"the synthetic-data penalty (accuracy \%, Finding-17 "
                 r"norm-space grid).}",
                 r"\label{tab:finding17}", r"\end{table}"])
    with open(os.path.join(C.TAB_DIR, 'finding17.tex'), 'w') as f:
        f.write('\n'.join(lines))

    # ---- Step 0a diagnostics summary ----
    with open(os.path.join(C.REPO_ROOT, 'report',
                           'step0_diagnostics.json')) as f:
        s0 = json.load(f)
    a = s0['step0a']
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\begin{tabular}{l r r}", r"\toprule",
             r"\textbf{Point} & $\varepsilon$=0.0 & 0.094 \\\\",
             r"\midrule"]
    lines.append(f"null-ablation ep41 (3-seed) & "
                 f"{a['null_ablation_ep41_eps0.000']['acc_mean']:.2f}"
                 f"$\\pm${a['null_ablation_ep41_eps0.000']['acc_std']:.2f} & "
                 f"{a['null_ablation_ep41_eps0.094']['acc_mean']:.2f}"
                 f"$\\pm${a['null_ablation_ep41_eps0.094']['acc_std']:.2f} "
                 f"\\\\")
    lines.append(f"known epoch-60 (same legacy run) & ~48.7 & "
                 f"{a['known_epoch60_eps0.094'].split(' ')[0]} "
                 f"$\\pm$ {a['known_epoch60_eps0.094'].split(' ')[2]} "
                 f"\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                 r"\caption{Step 0a provenance-scoped numbers. Both rows are "
                 r"LEGACY v11 (recon gradient was a silent no-op); they are "
                 r"NOT evidence for v12's own trajectory. Source: "
                 r"report/step0\_diagnostics.json.}",
                 r"\label{tab:step0a}", r"\end{table}"])
    with open(os.path.join(C.TAB_DIR, 'step0a.tex'), 'w') as f:
        f.write('\n'.join(lines))
    print('[flowcharts] wrote curriculum.tex, hyperparams.tex, isolation.tex, '
          'finding17.tex, step0a.tex', flush=True)


def main():
    evolution_timeline()
    bio_mapping()
    training_pipeline()
    loss_pipeline()
    inference_pipeline()
    active_inference_cycle()
    comp_graph()
    tensor_flow()
    data_tables()


if __name__ == '__main__':
    main()
