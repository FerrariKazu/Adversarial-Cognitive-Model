#!/usr/bin/env python3
"""
generate_math_figures.py
========================
Schematic + analytic figures for the mathematical sections:

  frequency_gating.png   -- M/P pathway frequency split with learnable gates,
                            plus a real image's low/high-pass decomposition.
  precision_dynamics.png -- closed-form solutions of the precision ODE
                            tau*dPi/dt = e^2 - Pi, and the dynamic-beta map.
  banach_contraction.png -- geometric error attenuation gamma^t per foraging
                            step, and the v10 halt-loss conflict.
  gaze_update.png        -- Eq. II v12: two gradient signals -> blended step.
  trades_objective.png   -- 2-D toy contour of the TRADES objective.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _report_common as C

import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


def frequency_gating():
    plt = C.set_style()
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # real image low/high split
    imgs, _ = C.load_test_images(n=1, seed=9)
    img = C.denorm(imgs[0:1])[0].permute(1, 2, 0).numpy()
    import scipy.ndimage as ndi
    low = ndi.gaussian_filter(img, sigma=3.0)
    high = np.clip(img - low, 0, 1)
    ax2.imshow(img)
    ax2.set_title('original', fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])

    inset_low = ax2.inset_axes([0.03, -0.42, 0.45, 0.4])
    inset_low.imshow(low)
    inset_low.set_title('low-pass wL≈0.70', fontsize=8)
    inset_low.set_xticks([])
    inset_low.set_yticks([])
    inset_high = ax2.inset_axes([0.52, -0.42, 0.45, 0.4])
    inset_high.imshow(high)
    inset_high.set_title('high-pass wH≈0.54', fontsize=8)
    inset_high.set_xticks([])
    inset_high.set_yticks([])

    # schematic: pathway split
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    P = C.PALETTE

    def box(x, y, t, fc, ec, w=2.0, fs=8.6):
        b = FancyBboxPatch((x - w / 2, y - 0.42), w, 0.84,
                           boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.3)
        ax1.add_patch(b)
        ax1.text(x, y, t, ha='center', va='center', fontsize=fs,
                 color=C.PALETTE['ink'])

    from matplotlib.patches import FancyArrowPatch

    def arrow(x1, y1, x2, y2, c):
        ax1.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                      mutation_scale=14, lw=1.6, color=c))

    box(2.2, 8.6, 'stem (768)', '#eef6f6', P['teal'])
    box(1.2, 6.4, 'low freq (M)', '#f3f0fb', P['indigo'])
    box(3.2, 6.4, 'high freq (P)', '#f3f0fb', P['indigo'])
    box(1.2, 4.4, 'ventral (d=384)', '#f3f0fb', P['indigo'])
    box(3.2, 4.4, 'dorsal (d=384)', '#f3f0fb', P['indigo'])
    box(2.2, 2.2, 'fused 768', '#eef6f6', P['teal'])
    box(2.2, 0.6, 'sigmoid gates:\nwL, wH (learned)', '#fbf6ee', P['amber'],
        w=3.4)
    arrow(2.2, 8.2, 1.2, 6.8, P['teal'])
    arrow(2.2, 8.2, 3.2, 6.8, P['indigo'])
    arrow(1.2, 6.0, 1.2, 4.8, P['indigo'])
    arrow(3.2, 6.0, 3.2, 4.8, P['indigo'])
    arrow(1.2, 4.0, 2.2, 2.6, P['teal'])
    arrow(3.2, 4.0, 2.2, 2.6, P['teal'])
    ax1.set_title('Frequency-separated ventral/dorsal streams\n'
                  'learnable M-pathway gates (v5 origin, v12 inherited)',
                  fontsize=10.5)

    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, 'frequency_gating.png'))
    print('[math] frequency_gating.png', flush=True)


def precision_dynamics():
    plt = C.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    tau = 0.1
    t = np.linspace(0, 0.5, 300)
    P = C.PALETTE
    for e2, col in zip([0.1, 0.2, 0.4, 0.8], [P['teal'], P['indigo'],
                                              P['amber'], P['rust']]):
        pi0 = 0.5
        pi = e2 + (pi0 - e2) * np.exp(-t / tau)
        axes[0].plot(t, pi, lw=2, label=f'$e^2$={e2}')
    axes[0].axhline(0.8, color=P['gray'], ls=':', lw=1)
    axes[0].axhline(0.2, color=P['gray'], ls=':', lw=1)
    axes[0].set_xlabel('time (foraging steps)')
    axes[0].set_ylabel('Π_D')
    axes[0].set_title(r'Precision ODE: $\tau\dot\Pi = e^2 - \Pi$, '
                      r'$\Pi \in [0.2,0.8]$', fontsize=11)
    axes[0].legend(fontsize=8)

    e2 = np.linspace(0, 0.8, 200)
    beta0 = 2.0
    pi_star = np.clip(e2, 0.2, 0.8)
    beta_eff = beta0 * (0.5 + pi_star)
    axes[1].plot(e2, beta_eff, lw=2, color=P['teal'])
    axes[1].axhline(beta0, color=P['gray'], ls=':', lw=1)
    axes[1].fill_between(e2, beta0 * 0.5, beta0 * 1.5, color=P['teal'],
                         alpha=0.08)
    axes[1].set_xlabel('prediction error $e^2$')
    axes[1].set_ylabel(r'$\beta_{dyn} = \beta_0(0.5+\Pi_D)$')
    axes[1].set_title('Dynamic TRADES β modulation', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, 'precision_dynamics.png'))
    print('[math] precision_dynamics.png', flush=True)


def banach_contraction():
    plt = C.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    P = C.PALETTE
    t = np.arange(0, 12)
    for g, col in zip([0.3, 0.5, 0.7, 0.9], [P['teal'], P['indigo'],
                                             P['amber'], P['rust']]):
        axes[0].plot(t, g ** t, 'o-', lw=1.8, ms=4,
                     label=f'γ={g}')
    axes[0].axvspan(0, 4, color=P['teal'], alpha=0.08)
    axes[0].text(2, 0.72, 'T=4 (v12 fixed)', ha='center', fontsize=8.6,
                 color=P['teal'])
    axes[0].set_xlabel('foraging step t')
    axes[0].set_ylabel(r'$\delta_t / \delta_0 = \gamma^t$')
    axes[0].set_title('Banach contraction: each step multiplies the '
                      'perturbation by γ<1', fontsize=11)
    axes[0].legend(fontsize=8)

    # halt-loss conflict illustration
    x = np.arange(0, 8)
    info_gain = 1.0 - 0.75 ** x
    axes[1].plot(x, info_gain, 'o-', color=P['indigo'], lw=1.8, label=
                 'attenuation benefit of t steps')
    axes[1].axhline(0.25, color=P['rust'], ls='--', lw=1.4, label=
                    'halt loss: minimize steps')
    axes[1].fill_between(x, 0, info_gain, color=P['indigo'], alpha=0.08)
    axes[1].set_xlabel('foraging steps')
    axes[1].set_ylabel('relative benefit')
    axes[1].set_title('v10 halt-efficiency loss fought the contraction\n'
                      '(penalized using more steps) — deleted in v12',
                      fontsize=10.5)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, 'banach_contraction.png'))
    print('[math] banach_contraction.png', flush=True)


def gaze_update():
    plt = C.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    P = C.PALETTE
    # Panel A: schematic of the two gradient signals
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    from matplotlib.patches import Circle, FancyArrowPatch
    ax.add_patch(Circle((5, 5), 2.4, fill=False, ec=P['ink'], lw=1.4))
    ax.add_patch(Circle((5, 5), 0.15, fc=P['rust'], ec=P['rust']))
    ax.text(5, 5.35, 'gaze $a$', ha='center', fontsize=9)
    ax.text(1.2, 8.4, '∇$_a$R(x,a)\npixel-space error', fontsize=8.6,
            color=P['teal'])
    ax.text(7.6, 8.4, '∇$_a$F(x,a)\nfeature-space error', fontsize=8.6,
            color=P['indigo'])
    ax.add_patch(FancyArrowPatch((2.2, 7.3), (4.3, 5.8), arrowstyle='-|>',
                                 mutation_scale=14, lw=2, color=P['teal']))
    ax.add_patch(FancyArrowPatch((7.8, 7.3), (5.7, 5.8), arrowstyle='-|>',
                                 mutation_scale=14, lw=2, color=P['indigo']))
    ax.add_patch(FancyArrowPatch((5, 5), (6.6, 4.1), arrowstyle='-|>',
                                 mutation_scale=16, lw=2.4, color=P['rust']))
    ax.text(6.9, 4.0, '$g = \\lambda\\nabla R + (1{-}\\lambda)\\nabla F$',
            fontsize=8.6, color=P['rust'])
    ax.set_title('Eq. II v12 — reconstruction-guided gaze\n'
                 '(single scalar objective, λ=0.5)', fontsize=10.5)

    # Panel B: real gaze trajectory + error map from a forward pass
    ax2 = axes[1]
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = C.build_model(load_ckpt=True, device=device)
    model.eval()
    from phase1_training.model_rhan_v10 import foveal_sample
    imgs, _ = C.load_test_images(n=1, seed=11)
    with torch.no_grad():
        _, traj = model(imgs.to(device), return_trajectory=True)
        s = model.peripheral_proj(model._peripheral_pass(imgs.to(device)))
        pred = model.generative_prior(s)
        a_final = traj['actions'][-1]
        crop = foveal_sample(imgs.to(device), a_final, fovea_size=48)
    disp = C.denorm(imgs).squeeze(0).permute(1, 2, 0).numpy()
    ax2.imshow(disp)
    acts = traj['actions']
    gx = [acts[j][0, 1].item() for j in range(4)]
    gy = [acts[j][0, 0].item() for j in range(4)]
    px = (np.array(gx) + 1) / 2 * 96
    py = (np.array(gy) + 1) / 2 * 96
    ax2.plot(px, py, 'o-', color=P['amber'], lw=1.8, ms=5)
    R = model.compute_recon_error_map(crop, pred)[0].cpu().numpy()
    ax2.imshow(R, cmap='magma', alpha=0.35)
    ax2.set_title('Real trajectory: gaze follows reconstruction error',
                  fontsize=10.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, 'gaze_update.png'))
    print('[math] gaze_update.png', flush=True)


def trades_objective():
    plt = C.set_style()
    from scipy.stats import multivariate_normal
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    P = C.PALETTE
    x = np.linspace(-3.5, 3.5, 160)
    X, Y = np.meshgrid(x, x)
    grid = np.dstack([X, Y])
    c0 = multivariate_normal([-1.2, -1.2], [[1.0, 0.3], [0.3, 1.0]])
    c1 = multivariate_normal([1.4, 1.0], [[1.2, -0.2], [-0.2, 0.9]])
    p0 = c0.pdf(grid)
    p1 = c1.pdf(grid)

    for ax, (name, lam) in zip(axes, [
            ('CE only (β=0)', 0.0),
            ('TRADES β=2', 2.0),
            ('TRADES β=6 (high)', 6.0)]):
        # toy TRADES surrogate: -log p_y + beta * D_KL(softmax-ish)
        # use |p0-p1| overlap as a proxy for boundary pressure
        ce = -np.log(p0 + 1e-9)
        overlap = np.maximum(p0 - p1, 0)
        L = ce + lam * overlap
        cs = ax.contourf(X, Y, L, levels=24, cmap='viridis_r')
        ax.contour(X, Y, p0 - p1, levels=[0], colors=[P['rust']], lw=1.8)
        ax.set_title(f'{name}\n(large β widens the robust margin)',
                     fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Toy illustration of the TRADES objective: '
                 'CE(y, f(x)) + β·KL(f(x), f(x_adv))', fontsize=12,
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(C.FIG_DIR, 'trades_objective.png'))
    print('[math] trades_objective.png', flush=True)


def main():
    frequency_gating()
    precision_dynamics()
    banach_contraction()
    gaze_update()
    trades_objective()


if __name__ == '__main__':
    main()
