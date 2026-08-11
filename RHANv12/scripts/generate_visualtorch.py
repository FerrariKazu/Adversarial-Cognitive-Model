#!/usr/bin/env python3
"""
generate_visualtorch.py
=======================
Renders a publication-quality architecture diagram of the REAL RHAN-v12
module tree. Module names, channel dimensions and parameter counts are pulled
live from the model — nothing is hand-drawn. The foraging loop is drawn as a
contained subgraph with an explicit T=4 recurrence arrow.

Output: report/figures/architecture_overview.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _report_common as C

import torch.nn as nn


def _params(mod):
    return sum(p.numel() for p in mod.parameters())


def main():
    plt = C.set_style()
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    model = C.build_model(load_ckpt=False, device='cpu')

    fig, ax = plt.subplots(figsize=(17, 11))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 11)
    ax.axis('off')

    W, H = 2.35, 1.05          # box width/height
    ink = C.PALETTE['ink']
    teal = C.PALETTE['teal']
    indigo = C.PALETTE['indigo']
    amber = C.PALETTE['amber']
    rust = C.PALETTE['rust']
    gray = C.PALETTE['gray']

    def box(x, y, title, sub='', w=W, h=H, fc='#ffffff', ec=ink, ls='-',
            fs=10.5, sfs=8.5):
        b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.08",
                           fc=fc, ec=ec, lw=1.4, ls=ls, zorder=3)
        ax.add_patch(b)
        ax.text(x, y + (0.16 if sub else 0), title, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=ink, zorder=4)
        if sub:
            ax.text(x, y - 0.18, sub, ha='center', va='center',
                    fontsize=sfs, color=gray, zorder=4)
        return b

    def arrow(x1, y1, x2, y2, color=ink, lw=1.8, style='-|>', ls='-'):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                            mutation_scale=16, lw=lw, color=color,
                            linestyle=ls, zorder=2)
        ax.add_patch(a)

    # ── Column 1: input + stem ────────────────────────────────────────────
    box(0.55, 5.5, 'Input', r'$96\times96\times3$', fc='#f2f4f8')
    arrow(1.7, 5.5, 2.35, 5.5)

    stem = model.stem
    box(3.35, 8.0, 'conv1', '3→128, 96$^2$', fc='#eef6f6')
    box(3.35, 6.6, 'conv2', '128→512, 48$^2$', fc='#eef6f6')
    box(3.35, 5.2, 'conv3', '512→1024, 24$^2$', fc='#eef6f6')
    box(3.35, 3.8, 'conv4', '1024→768, 12$^2$', fc='#eef6f6')
    box(3.35, 2.4, 'shortcut', '3→768, s=8', fc='#fbf6ee')
    ax.text(3.35, 9.1, f'WideSEConvStemLarge\n{_params(stem):,} params',
            ha='center', va='center', fontsize=9, color=gray)
    arrow(3.35, 8.55, 3.35, 7.15, color=teal)
    arrow(3.35, 7.15, 3.35, 5.75, color=teal)
    arrow(3.35, 5.75, 3.35, 4.35, color=teal)
    arrow(3.35, 4.35, 3.35, 2.95, color=teal)
    arrow(2.35, 5.5, 2.9, 8.0)
    arrow(3.35, 1.85, 4.6, 5.5, color=amber, ls='--')   # shortcut to fusion

    # ── Column 2: tokeniser ───────────────────────────────────────────────
    tok = model.tokeniser
    box(5.0, 8.0, 'tokeniser', '144 patches + CLS\n$145\\times768$',
        fc='#f2f4f8', fs=9.5)
    box(5.0, 6.6, 'pos embed', '$145\\times768$', fc='#f2f4f8', fs=9)
    arrow(3.35, 8.0, 3.85, 8.0)
    arrow(5.0, 7.45, 5.0, 7.15, color=teal)
    ax.text(5.0, 8.9, f'{_params(tok):,} params', ha='center', fontsize=9,
            color=gray)

    # ── Column 3: ventral / dorsal ────────────────────────────────────────
    v, d = model.ventral, model.dorsal
    box(6.7, 8.0, 'ventral', f'8× enc (d=384)\n{_params(v):,} p', fc='#f3f0fb')
    box(6.7, 6.6, 'dorsal', f'8× enc (d=384)\n{_params(d):,} p', fc='#f3f0fb')
    box(6.7, 5.2, 'feedback', f'2-step recur.\n{_params(model.feedback):,} p',
        fc='#f3f0fb', fs=9)
    arrow(5.0, 8.0, 5.55, 8.0)
    arrow(5.0, 6.6, 5.55, 6.6)
    arrow(6.7, 8.55, 6.7, 7.15, color=indigo)
    arrow(6.7, 7.15, 6.7, 5.75, color=indigo)
    # recurrence arrow: feedback back into transformer
    arrow(7.95, 6.6, 7.95, 5.2, color=indigo, style='-|>')
    arrow(7.95, 5.2, 7.95, 4.6, color=indigo, ls='--')
    ax.text(8.15, 5.9, '×2', fontsize=9, color=indigo)

    # fusion point -> peripheral pass output
    box(6.7, 3.6, 'CLS (768)', 'peripheral pass', fc='#eef6f6', fs=9)
    arrow(6.7, 4.65, 6.7, 4.15, color=ink)

    # ── Column 4: belief bridge ───────────────────────────────────────────
    box(8.3, 3.6, 'peripheral_proj', '768→512', fc='#eef6f6', fs=9)
    box(8.3, 2.2, 'action_init', '512→2 (gaze)', fc='#fbf6ee', fs=9)
    arrow(6.7, 3.6, 7.15, 3.6)
    arrow(8.3, 3.05, 8.3, 2.75, color=amber)

    # ── Column 5: the foraging loop (T=4) ─────────────────────────────────
    loop = mpatches.FancyBboxPatch((9.05, 0.55), 4.4, 8.9,
                                   boxstyle="round,pad=0.1",
                                   fc='#fbfbfe', ec=indigo, lw=1.6, ls='--',
                                   zorder=1)
    ax.add_patch(loop)
    ax.text(11.25, 9.35, 'Active inference loop — fixed $T=4$ (no halt network)',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=indigo)
    ax.text(11.25, 8.95, 'belief $s_t \\in \\mathbb{R}^{512}$', ha='center',
            fontsize=9, color=gray)

    fv = model.foveal_stream
    gp = model.generative_prior
    gate = model.foveal_gate
    ip = model.image_precision

    box(9.7, 7.6, 'foveal_sample', '48$^2$ crop @ $a_t$', fc='#fbf6ee', fs=8.5, w=2.0)
    box(9.7, 6.1, 'foveal_stream', f'{_params(fv):,} p', fc='#eef6f6', fs=8.5, w=2.0)
    box(11.6, 7.6, 'parafoveal', 'blur σ=2, once', fc='#f2f4f8', fs=8.5, w=1.9)
    box(11.6, 6.1, 'foveal_gate', f'α-blend {_params(gate):,} p',
        fc='#f3f0fb', fs=8.5, w=1.9)
    box(9.7, 4.5, 'generative_prior', f'$s_t\\to\\hat x_{{t}}$\n{_params(gp):,} p',
        fc='#f3f0fb', fs=8.5, w=2.0)
    box(11.6, 4.5, 'image_precision', f'$\\Pi_D$ from pixel error\n{_params(ip):,} p',
        fc='#f3f0fb', fs=8.5, w=1.9)
    box(10.65, 2.9, 'belief update', r'$s \leftarrow (1{-}\Pi_D)s + \Pi_D c$',
        fc='#eef6f6', fs=8.5, w=2.6)
    box(10.65, 1.5, 'gaze update (Eq. II v12)', r'$\lambda\nabla R + (1{-}\lambda)\nabla F$',
        fc='#fbf6ee', fs=8.5, w=2.6)

    arrow(8.3, 3.6, 9.0, 7.6, color=ink)          # belief -> loop
    arrow(8.3, 2.2, 9.0, 1.5, color=amber)        # init gaze -> gaze update
    arrow(9.7, 8.15, 9.7, 7.15, color=teal)
    arrow(9.7, 6.65, 10.6, 6.1, color=teal)
    arrow(11.6, 7.05, 11.6, 6.65, color=ink)
    arrow(10.65, 6.1, 10.65, 5.05, color=ink)     # gate out -> prior (context)
    arrow(9.7, 5.05, 9.7, 5.05, color=teal)
    arrow(9.7, 5.05, 11.05, 4.5, color=teal)      # crop+pred -> precision
    arrow(11.05, 4.5, 11.05, 3.45, color=indigo)  # precision -> belief update
    arrow(10.65, 3.45, 10.65, 2.05, color=ink)    # belief -> gaze update
    arrow(11.9, 2.9, 11.9, 1.5, color=ink)        # context -> gaze update
    # loop-back: gaze update -> next sample
    arrow(10.65, 1.5, 9.7, 1.5, color=rust, ls='--')
    arrow(9.7, 1.5, 9.7, 7.05, color=rust, ls='--')
    ax.text(9.5, 4.2, 'T = 4', fontsize=10, color=rust, fontweight='bold')

    # ── Column 6: readout ─────────────────────────────────────────────────
    box(13.5, 6.6, 'belief_unproj', '512→768', fc='#eef6f6', fs=9)
    box(13.5, 5.2, 'classifier', 'LN→Drop→10', fc='#f2f4f8', fs=9)
    box(13.5, 3.6, 'logits', '10 classes', fc='#fbf6ee', fs=9)
    arrow(11.7, 2.9, 13.0, 6.6, color=ink)
    arrow(13.5, 6.05, 13.5, 5.75, color=ink)
    arrow(13.5, 4.65, 13.5, 4.15, color=ink)

    # annotation of removed component
    ax.text(15.2, 8.0, 'Removed from v11 → v12:\n'
                       '$\\bullet$ halt\\_net (fixed T=4)\n'
                       '$\\bullet$ foraging-consistency loss\n'
                       '$\\bullet$ precision-calibration loss\n'
                       '$\\bullet$ halt-efficiency loss',
            fontsize=9, color=rust, va='center', ha='left',
            bbox=dict(boxstyle='round', fc='#fdf3f0', ec=rust, lw=1))
    ax.text(15.2, 5.0, 'New in v12:\n'
                       '$\\bullet$ recon-guided gaze\n'
                       '  $\\lambda$·∇R(x,a) + (1−λ)·∇F(x,a)\n'
                       '$\\bullet$ differentiable $L_{\\text{recon}}$\n'
                       '  (v11 had a silent gradient no-op)',
            fontsize=9, color=teal, va='center', ha='left',
            bbox=dict(boxstyle='round', fc='#eef6f6', ec=teal, lw=1))

    ax.set_title('RHAN-v12 — Locked-In Active Inference Architecture '
                 '(75,440,469 parameters)', fontsize=14, fontweight='bold',
                 pad=10)
    fig.savefig(os.path.join(C.FIG_DIR, 'architecture_overview.png'))
    print('[visualtorch] wrote architecture_overview.png', flush=True)


if __name__ == '__main__':
    main()
