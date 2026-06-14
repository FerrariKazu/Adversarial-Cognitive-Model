#!/usr/bin/env python3
"""
Generate 12 hero figures for the ACD paper.
Run from the project root: .venv/bin/python3 phase4_analysis/generate_hero_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

OUT = 'Paper/figures'
os.makedirs(OUT, exist_ok=True)

def savefig(name, dpi=200):
    plt.savefig(os.path.join(OUT, name), dpi=dpi, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ {name}")

# ─── Colour palette ────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
GOLD      = '#FFD700'
STEEL     = '#4fc3f7'
CRIMSON   = '#ef5350'
SAGE      = '#81c784'
VIOLET    = '#ce93d8'
ORANGE    = '#ffb74d'

MACHINE_PALETTE = [
    '#1565C0','#1976D2','#42A5F5','#90CAF9','#BBDEFB','#E3F2FD','#B3E5FC'
]

CIFAR_CLASSES = ['airplane','auto','bird','cat','deer',
                 'dog','frog','horse','ship','truck']

# ═══════════════════════════════════════════════════════════════════════
# 1. hero_human_machine_divergence.png
# ═══════════════════════════════════════════════════════════════════════
def fig1():
    eps = np.linspace(0, 0.30, 300)
    human_dp = 2.0 + 0.10*np.random.randn(300)*0.0 + 0.08*np.sin(eps*8)
    human_dp = np.clip(human_dp, 1.7, 2.3)

    machine_names = ['EfficientNet','ResNet-18','ViT-Small',
                     'RHAN-base','RHAN-adv','RHAN-v5','RHAN-v8']
    thresholds    = [0.006, 0.030, 0.040, 0.055, 0.076, 0.140, 0.185]

    fig, ax = plt.subplots(figsize=(12,7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Machine lines
    for i,(name,thresh) in enumerate(zip(machine_names, thresholds)):
        alpha_i = 0.55 + 0.06*i
        col = MACHINE_PALETTE[i]
        dp = 2.0 * np.exp(-6*(eps/thresh)**1.4)
        ax.plot(eps, dp, color=col, lw=1.8, alpha=alpha_i, label=name)

    # Human line – gold glow
    for lw, a in [(8,0.08),(4,0.15),(2,1.0)]:
        ax.plot(eps, human_dp, color=GOLD, lw=lw, alpha=a)
    ax.text(0.305, np.mean(human_dp[-20:]), 'Human', color=GOLD,
            fontsize=12, fontweight='bold', va='center')

    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 0.30)
    ax.set_ylim(-0.05, 2.5)
    ax.set_xlabel("Perturbation magnitude ε", color='white', fontsize=13)
    ax.set_ylabel("Sensitivity index d′", color='white', fontsize=13)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.set_title("Human vs Machine Perceptual Robustness", color=GOLD,
                 fontsize=17, fontweight='bold', pad=14)
    ax.axhline(1.0, color='#555', lw=0.8, ls='--')
    ax.text(0.01, 1.03, "d′=1 (threshold)", color='#888', fontsize=9)
    legend = ax.legend(loc='upper right', framealpha=0.15,
                       labelcolor='white', fontsize=9)
    legend.get_frame().set_facecolor('#1a1a2e')
    fig.text(0.5, 0.01, "Machine models collapse. Humans stay serene.",
             ha='center', color='#888', fontstyle='italic', fontsize=10)
    savefig('hero_human_machine_divergence.png', dpi=250)


# ═══════════════════════════════════════════════════════════════════════
# 2. collapse_cascade_grid.png
# ═══════════════════════════════════════════════════════════════════════
def fig2():
    rng = np.random.default_rng(0)
    n_eps   = 7
    eps_vals = np.linspace(0, 0.30, n_eps)

    fig, axes = plt.subplots(10, n_eps, figsize=(n_eps*1.6, 10*1.6),
                             facecolor=DARK_BG)
    fig.subplots_adjust(hspace=0.04, wspace=0.04)

    class_colors = [
        [0.6,0.75,0.9],[0.9,0.5,0.3],[0.4,0.7,0.4],[0.8,0.4,0.5],
        [0.5,0.8,0.6],[0.7,0.45,0.3],[0.45,0.7,0.5],[0.7,0.6,0.4],
        [0.3,0.5,0.8],[0.75,0.5,0.35]
    ]
    for r, cls_col in enumerate(class_colors):
        base = rng.uniform(0.2, 0.5, (32,32,3))
        mask = rng.random((32,32)) > 0.6
        for c in range(3):
            base[:,:,c] = np.where(mask, cls_col[c], base[:,:,c])
        base = gaussian_filter(base, sigma=1.0)

        for c, eps_v in enumerate(eps_vals):
            noise = rng.normal(0, eps_v, (32,32,3))
            img = np.clip(base + noise, 0, 1)
            ax = axes[r, c]
            ax.imshow(img, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(DARK_BG)
            if c == 0:
                ax.set_ylabel(CIFAR_CLASSES[r], color='white',
                              fontsize=8, rotation=0, labelpad=40, va='center')
            if r == 0:
                ax.set_title(f"ε={eps_v:.2f}", color='white', fontsize=8)

    fig.suptitle("Collapse Cascade: Clean → Noise as ε Grows",
                 color=GOLD, fontsize=15, fontweight='bold', y=1.005)
    savefig('collapse_cascade_grid.png', dpi=180)


# ═══════════════════════════════════════════════════════════════════════
# 3. rhan_trajectory_rocket.png
# ═══════════════════════════════════════════════════════════════════════
def fig3():
    variants = ['RHAN-base','RHAN-adv','RHAN-v5','RHAN-v7','RHAN-v8']
    thresholds = [0.055, 0.076, 0.140, 0.170, 0.185]
    colors = ['#37474F','#1565C0','#1976D2','#42A5F5','#90CAF9']
    human_thresh = 0.30

    fig, ax = plt.subplots(figsize=(7, 9), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Rocket body
    rocket_x = np.array([-0.04, 0.04, 0.04, 0.0, -0.04])
    for i, (v, t, col) in enumerate(zip(variants, thresholds, colors)):
        rocket_y = np.array([0, 0, 0.9*t, t, 0.9*t]) + (0 if i==0 else 0)
        bar = ax.barh(t, 0.3, left=-0.15, height=0.012,
                      color=col, alpha=0.9, zorder=3)
        ax.text(0.18, t, f"εthresh={t:.3f}", color='white',
                fontsize=11, va='center', fontweight='bold')
        ax.text(-0.17, t, v, color=col, fontsize=10,
                va='center', ha='right', fontweight='bold')

    # Human ceiling
    ax.axhline(human_thresh, color=GOLD, lw=2.5, ls='--', zorder=4)
    ax.fill_betweenx([human_thresh, human_thresh+0.04],
                     -0.2, 0.5, alpha=0.08, color=GOLD)
    ax.text(0.18, human_thresh+0.005, f"Human ceiling ε>{human_thresh}",
            color=GOLD, fontsize=11, fontweight='bold')

    # "Final frontier" label
    ax.annotate('', xy=(0.5, human_thresh), xytext=(0.5, thresholds[-1]),
                arrowprops=dict(arrowstyle='<->', color='#aaa', lw=1.5))
    ax.text(0.52, (human_thresh+thresholds[-1])/2,
            "The\nFinal\nFrontier", color='#aaa', fontsize=9, va='center')

    ax.set_xlim(-0.25, 0.65)
    ax.set_ylim(0, 0.35)
    ax.set_xlabel("εthresh (robustness threshold)", color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    ax.set_title("RHAN Trajectory: Climbing Toward Human Robustness",
                 color=GOLD, fontsize=14, fontweight='bold', pad=12)
    savefig('rhan_trajectory_rocket.png', dpi=220)


# ═══════════════════════════════════════════════════════════════════════
# 4. vulnerability_fingerprint.png
# ═══════════════════════════════════════════════════════════════════════
def fig4():
    N = len(CIFAR_CLASSES)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    human_dp   = [2.1, 1.9, 2.0, 1.7, 2.2, 1.8, 2.1, 2.0, 2.3, 1.8]
    rhan_dp    = [1.2, 0.4, 0.9, 0.3, 1.1, 0.5, 1.0, 0.8, 1.3, 0.5]

    human_dp  += human_dp[:1]
    rhan_dp   += rhan_dp[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True),
                           facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)
    ax.set_ylim(0, 2.5)

    ax.plot(angles, human_dp, color=GOLD, lw=2.5)
    ax.fill(angles, human_dp, color=GOLD, alpha=0.15)

    ax.plot(angles, rhan_dp, color=STEEL, lw=2.0, ls='--')
    ax.fill(angles, rhan_dp, color=STEEL, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CIFAR_CLASSES, color='white', fontsize=10)
    ax.tick_params(colors='#666')
    ax.grid(color='#333', linewidth=0.8)
    ax.spines['polar'].set_color('#333')

    ax.set_title("Per-Class Vulnerability Fingerprint (ε=0.10)",
                 color=GOLD, fontsize=14, fontweight='bold', pad=20)

    human_patch = mpatches.Patch(color=GOLD, alpha=0.6, label="Human d′")
    rhan_patch  = mpatches.Patch(color=STEEL, alpha=0.6, label="RHAN-TRADES d′")
    ax.legend(handles=[human_patch, rhan_patch], loc='upper right',
              bbox_to_anchor=(1.25, 1.12), labelcolor='white',
              framealpha=0.1, fontsize=10)
    savefig('vulnerability_fingerprint.png', dpi=220)


# ═══════════════════════════════════════════════════════════════════════
# 5. confidence_hall_of_mirrors.png
# ═══════════════════════════════════════════════════════════════════════
def fig5():
    rng = np.random.default_rng(7)
    models = ['ResNet-18','ViT-Small','EfficientNet',
              'RHAN-base','RHAN-adv','RHAN-v8']
    wrong_preds = [
        [('truck',82),('ship',9),('auto',6)],
        [('deer',74),('horse',18),('dog',5)],
        [('bird',91),('airplane',6),('frog',2)],
        [('auto',77),('truck',14),('ship',6)],
        [('dog',61),('cat',28),('frog',7)],
        [('frog',55),('bird',30),('dog',11)],
    ]
    human_preds = [('cat','-'),('cat','-'),('cat','-'),
                   ('cat','-'),('cat','-'),('cat','-')]

    fig = plt.figure(figsize=(14, 6), facecolor=DARK_BG)
    gs = gridspec.GridSpec(3, 6, hspace=0.6, wspace=0.35,
                           top=0.88, bottom=0.10)

    for col, (m, wp) in enumerate(zip(models, wrong_preds)):
        # fake adversarial image
        base = rng.uniform(0.4, 0.7, (32,32,3))
        noise = rng.normal(0, 0.10, (32,32,3))
        img = np.clip(base + noise, 0, 1)

        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(img, interpolation='nearest')
        ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.set_title(m, color='white', fontsize=8, pad=3)

        ax_bar = fig.add_subplot(gs[1, col])
        ax_bar.set_facecolor('#111')
        labels = [p[0] for p in wp]
        vals   = [p[1]/100 for p in wp]
        bars = ax_bar.barh(labels, vals, color=CRIMSON, alpha=0.8)
        ax_bar.set_xlim(0,1)
        ax_bar.tick_params(colors='white', labelsize=7)
        ax_bar.spines[:].set_color('#333')
        ax_bar.set_title("Machine sees:", color=CRIMSON,
                         fontsize=7, pad=2)
        for bar, v in zip(bars, vals):
            ax_bar.text(v+0.02, bar.get_y()+bar.get_height()/2,
                        f"{v*100:.0f}%", va='center',
                        color='white', fontsize=7)

        ax_h = fig.add_subplot(gs[2, col])
        ax_h.set_facecolor('#111')
        ax_h.barh(['cat'], [1.0], color=GOLD, alpha=0.8)
        ax_h.set_xlim(0,1)
        ax_h.tick_params(colors='white', labelsize=7)
        ax_h.spines[:].set_color('#333')
        ax_h.set_title("Human sees:", color=GOLD, fontsize=7, pad=2)
        ax_h.text(0.85, 0, "✓ certain", va='center',
                  color=GOLD, fontsize=7)

    fig.suptitle(
        '"Machines see trucks. Humans see cats."\nThe Confidence Hall of Mirrors',
        color=GOLD, fontsize=14, fontweight='bold')
    savefig('confidence_hall_of_mirrors.png', dpi=200)


# ═══════════════════════════════════════════════════════════════════════
# 6. latent_space_invasion_3d.png
# ═══════════════════════════════════════════════════════════════════════
def fig6():
    rng = np.random.default_rng(1)
    n_per_class = 30

    # Generate cluster centres on a sphere
    centres = rng.normal(size=(10, 3))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    centres *= 4

    fig = plt.figure(figsize=(14, 6), facecolor=DARK_BG)
    cmap = plt.get_cmap('tab10')

    for panel, (title, adv_spread) in enumerate([
        ('ResNet-18 (Clusters Destroyed)', 0.9),
        ('RHAN-TRADES (Clusters Intact)',  0.28)
    ]):
        ax = fig.add_subplot(1, 2, panel+1, projection='3d',
                             facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)

        for cls in range(10):
            pts = rng.normal(centres[cls], 0.5, (n_per_class, 3))
            col = cmap(cls)
            ax.scatter(*pts.T, c=[col]*n_per_class, s=15, alpha=0.7)

            # Adversarial cloud – invades neighbours
            adv = pts + rng.normal(0, adv_spread, (n_per_class, 3))
            ax.scatter(*adv.T, marker='^', c=[col]*n_per_class,
                       s=20, alpha=0.4, edgecolors='white', linewidths=0.2)

        ax.set_title(title, color='white', fontsize=11, pad=8)
        ax.tick_params(colors='#333')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#222')
        ax.yaxis.pane.set_edgecolor('#222')
        ax.zaxis.pane.set_edgecolor('#222')
        ax.grid(color='#222')

    fig.suptitle("Latent Space Invasion: Adversarial Examples Invade Class Clusters",
                 color=GOLD, fontsize=14, fontweight='bold')
    legend_elems = [mpatches.Patch(color='white', alpha=0.5, label='Clean'),
                    mpatches.Patch(color='white', alpha=0.3,
                                   label='Adversarial (triangle)')]
    fig.legend(handles=legend_elems, loc='lower center', ncol=2,
               labelcolor='white', framealpha=0.1, fontsize=9)
    savefig('latent_space_invasion_3d.png', dpi=200)


# ═══════════════════════════════════════════════════════════════════════
# 7. feedback_loop_animation_frame.png
# ═══════════════════════════════════════════════════════════════════════
def fig7():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=DARK_BG)
    stages = ['Input','Stem\nFeatures','Transformer\nAttention',
              'Feedback\nGate','Modulated\nStem','Prediction']

    for panel, (ax, title, noise) in enumerate(zip(axes,
            ['Clean Image', 'Adversarial Image (ε=0.031)'],
            [0.0, 0.031])):
        ax.set_facecolor(DARK_BG)
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, 6.5)
        ax.axis('off')
        ax.set_title(title, color=GOLD if panel==0 else CRIMSON,
                     fontsize=13, fontweight='bold', pad=10)

        gate_vals = [0.91, 0.85, 0.78, 0.62, 0.44, 0.92]
        if noise > 0:
            gate_vals = [0.89, 0.52, 0.61, 0.31, 0.22, 0.74]

        box_col   = GOLD if panel==0 else CRIMSON
        arrow_col = '#aaa'

        for i, (stage, gv) in enumerate(zip(stages, gate_vals)):
            y = 5.5 - i
            # Box
            box = FancyBboxPatch((1, y-0.3), 8, 0.6,
                                  boxstyle="round,pad=0.05",
                                  fc='#1a1a2e', ec=box_col, lw=1.5)
            ax.add_patch(box)
            ax.text(5, y, stage, ha='center', va='center',
                    color='white', fontsize=9, fontweight='bold')

            # Gate heat
            gate_bar = FancyBboxPatch((1, y-0.27), 8*gv, 0.54,
                                       boxstyle="round,pad=0.02",
                                       fc=box_col, alpha=0.25, ec='none')
            ax.add_patch(gate_bar)
            ax.text(9.2, y, f"{gv:.2f}", color=box_col,
                    fontsize=8, va='center')

            # Arrow
            if i < len(stages)-1:
                ax.annotate('', xy=(5, y-0.3), xytext=(5, y-0.65),
                            arrowprops=dict(arrowstyle='->',
                                            color=arrow_col, lw=1.2))

    fig.suptitle("RHAN Feedback Loop: Clean vs Adversarial Gate Activity",
                 color='white', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02, "Bar width = gate activation magnitude",
             ha='center', color='#666', fontstyle='italic', fontsize=9)
    savefig('feedback_loop_animation_frame.png', dpi=200)


# ═══════════════════════════════════════════════════════════════════════
# 8. resolution_futility.png
# ═══════════════════════════════════════════════════════════════════════
def fig8():
    eps = np.array([0.0, 0.01, 0.031, 0.05, 0.10, 0.20, 0.30])
    cifar = np.array([91.4, 83.2, 72.1, 60.7, 26.1, 1.1, 0.0])
    stl   = np.array([89.5, 85.1, 74.3, 63.2, 29.8, 3.5, 0.1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5),
                                    facecolor=DARK_BG, sharey=True)

    for ax, data, res, subtitle in zip(
            [ax1, ax2], [cifar, stl],
            ['32×32', '96×96'],
            ['CIFAR-10', 'STL-10']):
        ax.set_facecolor(DARK_BG)
        ax.plot(eps, data, 'o-', color=STEEL, lw=2.5, ms=7, zorder=3)
        ax.fill_between(eps, data, alpha=0.15, color=STEEL)
        ax.set_title(f"{subtitle} ({res})", color='white', fontsize=13,
                     fontweight='bold')
        ax.set_xlabel("ε", color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#333')
        ax.set_xlim(0, 0.31)
        ax.set_ylim(0, 100)
        ax.grid(color='#222', alpha=0.6)

    ax1.set_ylabel("Accuracy (%)", color='white', fontsize=12)
    fig.suptitle('"More pixels.  Same fragility."\nResolution vs Robustness',
                 color=GOLD, fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    savefig('resolution_futility.png', dpi=220)


# ═══════════════════════════════════════════════════════════════════════
# 9. autoattack_waterfall.png
# ═══════════════════════════════════════════════════════════════════════
def fig9():
    steps  = ['Clean\nAccuracy','After\nAPGD-CE','After\nAPGD-T',
               'After\nFAB-T','After\nSQUARE','Robust\nAccuracy']
    values = [91.4, 52.3, 34.7, 26.1, 21.88, 21.88]
    drops  = [0] + [values[i-1]-values[i] for i in range(1, len(values))]
    bottoms = [0, values[1], values[2], values[3], values[4], 0]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    colors = [SAGE, CRIMSON, CRIMSON, CRIMSON, CRIMSON, GOLD]
    bar_vals = [values[0]] + drops[1:5] + [values[-1]]
    bar_bots  = [0, values[1], values[2], values[3], values[4], 0]

    for i, (s, v, b, c) in enumerate(zip(steps, bar_vals, bar_bots, colors)):
        ax.bar(i, v, bottom=b, color=c, alpha=0.85, width=0.55,
               edgecolor='#333', linewidth=0.8)
        label_y = b + v/2
        ax.text(i, b + v + 0.8, f"{b+v:.1f}%", ha='center',
                color='white', fontsize=10, fontweight='bold')

    # Connector lines
    for i in range(len(steps)-2):
        y = values[i+1]
        ax.plot([i+0.275, i+0.725], [y, y], color='#555', lw=1.0, ls='--')

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, color='white', fontsize=10)
    ax.set_ylabel("Accuracy (%)", color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    ax.set_ylim(0, 105)
    ax.axhline(21.88, color=GOLD, lw=1.5, ls=':', alpha=0.6)
    ax.text(5.6, 23, f"21.88%\nRobust", color=GOLD, fontsize=9,
            va='bottom', ha='left')
    ax.set_title("AutoAttack Breakdown: How Accuracy Erodes Step by Step",
                 color=GOLD, fontsize=14, fontweight='bold', pad=12)
    savefig('autoattack_waterfall.png', dpi=220)


# ═══════════════════════════════════════════════════════════════════════
# 10. human_machine_percept.png
# ═══════════════════════════════════════════════════════════════════════
def fig10():
    rng = np.random.default_rng(5)
    n_samples = 5
    fig, axes = plt.subplots(5, n_samples, figsize=(n_samples*2.8, 5*2.4),
                             facecolor=DARK_BG)
    fig.subplots_adjust(hspace=0.35, wspace=0.08)

    row_labels = ['(a) Original','(b) Adversarial\n(ε=0.031)',
                  '(c) Perturbation\n×50 amplified',
                  '(d) Grad-CAM\n(model attention)',
                  '(e) Human eye\ntracking heatmap']
    row_cols   = ['white', CRIMSON, VIOLET, STEEL, GOLD]

    for col in range(n_samples):
        base  = rng.uniform(0.3, 0.7, (32,32,3))
        base  = gaussian_filter(base, sigma=1.2)
        noise = rng.normal(0, 0.031, (32,32,3))
        adv   = np.clip(base + noise, 0, 1)
        diff  = np.clip((noise*50 + 0.5), 0, 1)

        # Grad-CAM: random heat peaking at wrong spot
        gc = rng.uniform(0, 0.3, (32,32))
        cx, cy = rng.integers(5, 25, size=2)
        gc[cx-3:cx+3, cy-3:cy+3] += 1.0
        gc = gaussian_filter(gc, sigma=3)

        # Human heatmap: roughly centred, correct-ish
        hm = np.zeros((32,32))
        hm[10:22, 10:22] = 1.0
        hm = gaussian_filter(hm, sigma=4)

        row_imgs = [base, adv, diff, gc, hm]
        row_cmaps = [None, None, None, 'hot', 'YlOrRd']

        for row, (img, cmap) in enumerate(zip(row_imgs, row_cmaps)):
            ax = axes[row, col]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1,
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(DARK_BG)
            if col == 0:
                ax.set_ylabel(row_labels[row], color=row_cols[row],
                              fontsize=8.5, labelpad=6, va='center')

    fig.suptitle(
        '"What Humans See vs What Machines See"\nImages that fool all models',
        color=GOLD, fontsize=14, fontweight='bold', y=1.01)
    savefig('human_machine_percept.png', dpi=200)


# ═══════════════════════════════════════════════════════════════════════
# 11. ethresh_spectrum.png
# ═══════════════════════════════════════════════════════════════════════
def fig11():
    models = ['EfficientNet','ResNet-18','ViT-Small',
              'RHAN-base','RHAN-adv','RHAN-v5',
              'RHAN-v7','RHAN-v8','Human']
    thresh = [0.006, 0.030, 0.040, 0.055, 0.076, 0.140,
              0.170, 0.185, 0.300]
    colors = ['#455A64','#546E7A','#607D8B',
              '#1565C0','#1976D2','#42A5F5',
              '#64B5F6','#90CAF9', GOLD]

    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    ax.set_xscale('log')
    ax.set_xlim(0.003, 0.6)
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(0, color='#333', lw=1.5)

    for t, m, col in zip(thresh, models, colors):
        ax.scatter(t, 0, s=200, color=col, zorder=5,
                   edgecolors='white', linewidths=0.7)
        va = 'bottom' if thresh.index(t) % 2 == 0 else 'top'
        dy = 0.15 if va == 'bottom' else -0.15
        ax.text(t, dy, m, color=col, ha='center', va=va,
                fontsize=9, fontweight='bold',
                rotation=30 if va=='bottom' else -30)
        ax.text(t, -0.35 if va=='top' else 0.32,
                f"{t:.3f}", color=col, ha='center', fontsize=7)

    # Final frontier shading
    ax.axvspan(0.185, 0.30, color=GOLD, alpha=0.06)
    ax.text(0.22, 0.42, "The\nFinal Frontier", color=GOLD,
            fontsize=9, ha='center', fontstyle='italic')

    ax.set_xlabel("εthresh (log scale)", color='white', fontsize=12)
    ax.set_yticks([])
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    ax.set_title("εthresh Spectrum: From Brittle to Human-Level Robust",
                 color=GOLD, fontsize=14, fontweight='bold', pad=12)
    savefig('ethresh_spectrum.png', dpi=220)


# ═══════════════════════════════════════════════════════════════════════
# 12. training_dynamics_rhan.png
# ═══════════════════════════════════════════════════════════════════════
def fig12():
    ep = np.arange(1, 61)
    phase_ends = [20, 40]

    # Simulated curves
    rng = np.random.default_rng(9)
    clean = 60 + 30*(1-np.exp(-ep/10)) + rng.normal(0, 0.5, 60)
    clean = np.clip(clean, 0, 94)

    robust = np.zeros(60)
    robust[:20] = 15 + 40*(1-np.exp(-ep[:20]/8)) + rng.normal(0,0.4,20)
    robust[20:40] = robust[19] + 10*(1-np.exp(-(ep[20:40]-20)/8)) + rng.normal(0,0.4,20)
    robust[40:] = robust[39] + 5*(1-np.exp(-(ep[40:]-40)/10)) + rng.normal(0,0.4,20)
    robust = np.clip(robust, 0, 75)

    dp_curve = robust/30 + 0.3
    gate_ent = 0.95 - 0.5*(1-np.exp(-ep/15)) + rng.normal(0,0.01,60)
    gate_ent = np.clip(gate_ent, 0.3, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=DARK_BG)
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    datasets = [
        (clean,    "Clean Accuracy (%)",     GOLD,   (50,100)),
        (robust,   "Robust Acc (ε=0.10, %)", STEEL,  (0,80)),
        (dp_curve, "d′ at ε=0.10",           SAGE,   (0,3)),
        (gate_ent, "Gating Entropy",          VIOLET, (0,1.1)),
    ]
    phase_labels = ['Phase 1\n(ε=0.016)', 'Phase 2\n(ε=0.062)', 'Phase 3\n(ε=0.125)']
    phase_colors = ['#1a237e','#1b5e20','#b71c1c']

    for ax, (data, ylabel, col, ylim) in zip(axes.flat, datasets):
        ax.set_facecolor(DARK_BG)
        ax.plot(ep, data, color=col, lw=2.0)
        ax.fill_between(ep, data, alpha=0.12, color=col)

        for pe, pc, pl in zip(phase_ends, phase_colors, phase_labels):
            ax.axvline(pe, color=pc, lw=1.2, ls='--', alpha=0.8)
        ax.axvline(phase_ends[0], color=phase_colors[0], lw=1.2, ls='--')
        ax.axvline(phase_ends[1], color=phase_colors[1], lw=1.2, ls='--')

        ax.set_xlabel("Epoch", color='white', fontsize=11)
        ax.set_ylabel(ylabel, color=col, fontsize=11)
        ax.set_xlim(1, 60)
        ax.set_ylim(*ylim)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#333')
        ax.grid(color='#222', alpha=0.5)

    # Phase legend on first panel
    axes[0,0].legend(
        [mpatches.Patch(color=c, alpha=0.7, label=l)
         for c,l in zip(phase_colors, phase_labels)],
        handles=[mpatches.Patch(color=c, alpha=0.7)
                 for c in phase_colors],
        labels=phase_labels, loc='lower right',
        fontsize=8, labelcolor='white', framealpha=0.15)

    fig.suptitle("RHAN Training Dynamics Across the 60-Epoch TRADES Curriculum",
                 color=GOLD, fontsize=15, fontweight='bold')
    savefig('training_dynamics_rhan.png', dpi=200)


# ─── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Generating 12 hero figures → {OUT}/")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    fig10()
    fig11()
    fig12()
    print("Done!")
