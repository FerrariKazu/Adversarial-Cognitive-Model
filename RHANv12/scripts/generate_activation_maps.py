#!/usr/bin/env python3
"""
generate_activation_maps.py
===========================
Figures derived from real RHAN-v12 forward passes and gradients:

  stem_activations.png     -- conv1 channel activations on real images.
  gradcam_saliency.png     -- Grad-CAM on the last stem stage + input
                              saliency, for the true class.
  recon_prior.png          -- foveal crop @ final gaze, GenerativePrior
                              reconstruction, and the R(x,a) error map.
  parameter_distribution.png -- params per top-level module (live counts).
  flop_distribution.png      -- hook-counted FLOP share per module.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _report_common as C


def stem_activations(model, device):
    plt = C.set_style()
    imgs, _ = C.load_test_images(n=2, seed=5)
    imgs = imgs.to(device)
    acts = []
    with torch.no_grad():
        for i in range(2):
            a = model.stem.conv1(imgs[i:i + 1])          # (1,128,96,96)
            acts.append(a[0].cpu())

    fig, axes = plt.subplots(2, 10, figsize=(13, 3.2))
    for r in range(2):
        for c in range(10):
            ax = axes[r][c]
            chan = acts[r][c * 8 + 3]
            chan = (chan - chan.min()) / (chan.max() - chan.min() + 1e-8)
            ax.imshow(chan, cmap='inferno')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0 and c == 0:
                ax.set_title('conv1 channels', fontsize=9)
    fig.suptitle('RHAN-v12 stem conv1 activations (16 of 128 channels) on '
                 'real STL-10 test images', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(C.FIG_DIR, 'stem_activations.png'))
    print('[activations] stem_activations.png', flush=True)


def gradcam_saliency(model, device):
    plt = C.set_style()
    imgs, labels = C.load_test_images(n=4, seed=6)
    imgs = imgs.to(device)
    labels = labels.to(device)

    # hook the deepest stem stage (conv4 Sequential -> 12x12x768).
    # NOTE: do NOT detach — the feature map must stay on the graph so
    # torch.autograd.grad(logits, fm) can flow gradients back to it.
    conv4_out = {}
    def _hook(_m, _i, o):
        conv4_out['o'] = o
    h = model.stem.conv4.register_forward_hook(_hook)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(4):
        x = imgs[i:i + 1].requires_grad_(True)
        logits = model(x)
        target = labels[i].item()
        fm = conv4_out['o']                            # (1,768,12,12)
        # single backward: Grad-CAM grads + input saliency grads together
        g_fm, g_x = torch.autograd.grad(logits[0, target], (fm, x))
        w = g_fm.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * fm).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(0), size=(96, 96),
                            mode='bilinear', align_corners=False)[0, 0]

        sal = g_x.abs().amax(dim=1)[0]                 # (96,96)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        model.zero_grad(set_to_none=True)

        axes[i][0].imshow(C.denorm(x).squeeze(0).permute(1, 2, 0))
        axes[i][0].set_title(C.STL10_CLASSES[target], fontsize=9)
        axes[i][1].imshow(cam.detach().cpu(), cmap='jet', alpha=1.0)
        axes[i][2].imshow(sal.detach().cpu(), cmap='hot')
        axes[i][3].imshow(C.denorm(x).squeeze(0).permute(1, 2, 0))
        axes[i][3].imshow(cam.detach().cpu(), cmap='jet', alpha=0.45)
        for j in range(4):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
        if i == 0:
            axes[0][0].set_title('input · true class', fontsize=9)
            axes[0][1].set_title('Grad-CAM (stem conv4)', fontsize=9)
            axes[0][2].set_title('input saliency |∇L|', fontsize=9)
            axes[0][3].set_title('Grad-CAM overlay', fontsize=9)
    h.remove()
    fig.suptitle('Explainability: Grad-CAM and input saliency for the true '
                 'class (real RHAN-v12 gradients)', fontsize=12,
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(C.FIG_DIR, 'gradcam_saliency.png'))
    print('[activations] gradcam_saliency.png', flush=True)


def recon_prior(model, device):
    plt = C.set_style()
    from phase1_training.model_rhan_v10 import foveal_sample
    imgs, _ = C.load_test_images(n=4, seed=7)
    imgs = imgs.to(device)
    with torch.no_grad():
        _, traj = model(imgs, return_trajectory=True)
        s = model.peripheral_proj(model._peripheral_pass(imgs))
        pred = model.generative_prior(s)              # (B,3,48,48)
        a_final = traj['actions'][-1]
        crop = foveal_sample(imgs, a_final, fovea_size=48)
    acts = traj['actions']

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(4):
        axes[i][0].imshow(C.denorm(imgs[i:i + 1]).squeeze(0).permute(1, 2, 0))
        # overlay gaze trajectory
        gx = np.array([acts[j][i, 1].item() for j in range(4)])
        gy = np.array([acts[j][i, 0].item() for j in range(4)])
        px = (gx + 1) / 2 * 96
        py = (gy + 1) / 2 * 96
        axes[i][0].plot(px, py, 'o-', color=C.PALETTE['amber'], ms=4, lw=1.4)
        axes[i][0].set_title('input + gaze path', fontsize=9)
        axes[i][1].imshow(C.denorm(crop[i:i + 1]).squeeze(0).permute(1, 2, 0))
        axes[i][1].set_title('foveal crop @ a*', fontsize=9)
        axes[i][2].imshow(C.denorm(pred[i:i + 1]).squeeze(0).permute(1, 2, 0))
        axes[i][2].set_title('GenerativePrior prediction', fontsize=9)
        R = model.compute_recon_error_map(crop, pred)[i].cpu().numpy()
        axes[i][3].imshow(R, cmap='magma')
        axes[i][3].set_title('error map R(x,a)', fontsize=9)
        for j in range(4):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
    fig.suptitle('Reconstruction-guided gaze: foveal evidence, generative '
                 'prediction and the R(x,a) error map (real forward passes)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(C.FIG_DIR, 'recon_prior.png'))
    print('[activations] recon_prior.png', flush=True)


def parameter_distribution():
    plt = C.set_style()
    import json
    with open(os.path.join(C.GEN_DIR, 'model_stats.json')) as f:
        stats = json.load(f)
    top = stats['top_level']
    names = list(top.keys())
    vals = [top[n]['numel'] / 1e6 for n in names]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cols = [C.PALETTE['teal'] if n in ('stem', 'ventral', 'dorsal') else
            C.PALETTE['indigo'] if n in ('feedback', 'foveal_stream',
                                         'generative_prior') else
            C.PALETTE['amber'] for n in names]
    bars = ax.barh(names, vals, color=cols, edgecolor=C.PALETTE['ink'],
                   linewidth=0.6)
    ax.invert_yaxis()
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2,
                f'{v:.1f}M', va='center', fontsize=8.4)
    ax.set_xlabel('Parameters (millions)')
    total_m = stats['total_params'] / 1e6
    ax.set_title('RHAN-v12 parameter distribution by top-level module '
                 f'(total {total_m:.1f}M)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(vals) * 1.22)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, 'parameter_distribution.png'))
    print('[activations] parameter_distribution.png', flush=True)


def flop_distribution():
    plt = C.set_style()
    import json
    with open(os.path.join(C.GEN_DIR, 'complexity.json')) as f:
        cx = json.load(f)
    share = cx['top_level_flop_share']
    names = list(share.keys())
    vals = [share[n] / 1e9 for n in names]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cols = [C.PALETTE['teal'] if n in ('stem', 'ventral', 'dorsal') else
            C.PALETTE['indigo'] if n in ('feedback', 'foveal_stream',
                                         'generative_prior') else
            C.PALETTE['amber'] for n in names]
    bars = ax.barh(names, vals, color=cols, edgecolor=C.PALETTE['ink'],
                   linewidth=0.6)
    ax.invert_yaxis()
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.12, b.get_y() + b.get_height() / 2,
                f'{v:.1f}G', va='center', fontsize=8.4)
    ax.set_xlabel('FLOPs per image (billions, hook-counted)')
    ax.set_title('RHAN-v12 FLOP share by top-level module (45.6 GFLOPs/img, '
                 'T=4 + feedback + gaze gradient included)', fontsize=12,
                 fontweight='bold')
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, 'flop_distribution.png'))
    print('[activations] flop_distribution.png', flush=True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = C.build_model(load_ckpt=True, device=device)
    model.eval()
    stem_activations(model, device)
    gradcam_saliency(model, device)
    recon_prior(model, device)
    parameter_distribution()
    flop_distribution()


if __name__ == '__main__':
    main()
