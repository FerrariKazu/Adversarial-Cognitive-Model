#!/usr/bin/env python3
"""
generate_attention_diagrams.py
==============================
Figures derived from REAL forward passes of the loaded RHAN-v12:

  attention_maps.png      -- CLS→patch self-attention, ventral & dorsal,
                             layers {0, 7}, on real STL-10 test images.
  recurrent_refinement.png-- belief-loop trajectories (Π_D, error, gate α,
                             gaze, recon MSE) across the fixed T=4 steps,
                             clean vs. small-perturbation inputs.
  epsilon_curves.png      -- accuracy-vs-ε_norm from the recorded isolation
                             sweeps (report/*.json) + the 3-seed Step 0a
                             point with error bar.
  tsne_embeddings.png     -- t-SNE of 768-d belief readouts (class-colored).
  confusion_matrix.png    -- clean + PGD-20(ε=0.031) confusion on test data.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _report_common as C


# ── helpers ────────────────────────────────────────────────────────────────────

def _layerwise_attention(model, tokens, stream):
    """Replicate TransformerEncoder forward layer-by-layer to capture maps.

    The real `_run_transformer` runs inside torch.utils.checkpoint, which
    deserializes modules and defeats instance-level monkeypatching, and the
    encoder requests need_weights=False. Instead we run the SAME first-pass
    input through the same layer modules directly, calling self_attn with
    need_weights=True; because the weights object and the first-pass input
    are identical, the returned maps ARE the model's maps for this input.
    norm_first=True layout is replicated exactly.
    """
    maps = {}
    x = tokens
    for li, layer in enumerate(getattr(model, stream).layers):
        n1 = layer.norm1(x)
        out, w = layer.self_attn(n1, n1, n1, need_weights=True,
                                 average_attn_weights=True)
        x = x + layer.dropout1(out)
        n2 = layer.norm2(x)
        x = x + layer._ff_block(n2)
        maps[f'{stream}.l{li}'] = w.detach().float()
    return maps


def _pgd(model, x, eps=0.094, steps=10, alpha_frac=0.25, device='cuda'):
    """Compact KL-vs-clean-softmax PGD for the perturbation comparisons."""
    stl_min = (0.0 - torch.tensor(C.MEAN, device=device)) / torch.tensor(
        C.STD, device=device)
    stl_max = (1.0 - torch.tensor(C.MEAN, device=device)) / torch.tensor(
        C.STD, device=device)
    stl_min = stl_min.view(1, 3, 1, 1)
    stl_max = stl_max.view(1, 3, 1, 1)
    with torch.no_grad():
        logits_c = model(x)
        if isinstance(logits_c, tuple):
            logits_c = logits_c[0]
        probs_c = F.softmax(logits_c.float(), dim=1)
    x_adv = x.clone() + 0.001 * torch.randn_like(x)
    x_adv = x_adv.clamp(stl_min, stl_max)
    alpha = eps * alpha_frac
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            logits_a = model(x_adv)
            if isinstance(logits_a, tuple):
                logits_a = logits_a[0]
            loss = F.kl_div(F.log_softmax(logits_a.float(), dim=1),
                            probs_c, reduction='batchmean')
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv.detach() + alpha * g.sign())
        delta = (x_adv - x).clamp(-eps, eps)
        x_adv = (x + delta).clamp(stl_min, stl_max).detach()
    return x_adv


# ── figures ────────────────────────────────────────────────────────────────────

def attention_maps(model, device):
    plt = C.set_style()
    imgs, _ = C.load_test_images(n=4, seed=1)
    imgs = imgs.to(device)
    # first-pass token embeddings (identical to the model's own first call)
    with torch.no_grad():
        stem_features = model.stem(imgs)
        tokens = model.tokeniser(stem_features)
    store = {}
    with torch.no_grad():
        store.update(_layerwise_attention(model, tokens[:, :, :384], 'ventral'))
        store.update(_layerwise_attention(model, tokens[:, :, 384:], 'dorsal'))

    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    keys = [('ventral.l0', 'Ventral · layer 0'),
            ('ventral.l7', 'Ventral · layer 7'),
            ('dorsal.l0', 'Dorsal · layer 0'),
            ('dorsal.l7', 'Dorsal · layer 7')]
    for row, (key, title) in enumerate(keys):
        att = store[key]                          # (B, 145, 145)
        for col in range(4):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(C.STL10_CLASSES[int(_[col])], fontsize=9)
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
            # CLS attention over spatial patches (heads already averaged)
            a = att[col, 0, 1:].reshape(12, 12).cpu().numpy()
            ax.imshow(a, cmap='viridis', vmin=0, vmax=a.max())
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('RHAN-v12 self-attention: CLS → spatial patches (real '
                 'STL-10 test images)', fontsize=12, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(C.FIG_DIR, 'attention_maps.png'))
    print('[attention] attention_maps.png', flush=True)


def recurrent_refinement(model, device):
    plt = C.set_style()
    imgs, _ = C.load_test_images(n=6, seed=2)
    imgs = imgs.to(device)
    with torch.no_grad():
        _, traj_c = model(imgs, return_trajectory=True)
    adv = _pgd(model, imgs, eps=0.094, steps=8)
    with torch.no_grad():
        _, traj_a = model(adv, return_trajectory=True)

    def series(traj, key):
        return np.array([t.float().mean().item() for t in traj[key]])

    t = np.arange(1, 5)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    pairs = [('precisions', 'Π_D (sensory precision)'),
             ('errors', 'prediction error magnitude'),
             ('gate_alphas', 'gate α (foveal weight)'),
             ('recon_errors', 'recon MSE (pixel)'),
             ('actions', 'gaze |a| (norm)')]
    for i, (key, lab) in enumerate(pairs):
        ax = axes.flat[i]
        c = series(traj_c, key)
        a = series(traj_a, key)
        ax.plot(t, c, 'o-', color=C.PALETTE['teal'], label='clean',
                lw=1.8, ms=5)
        ax.plot(t, a, 's--', color=C.PALETTE['rust'], label='ε=0.094 PGD',
                lw=1.8, ms=5)
        ax.set_xlabel('foraging step')
        ax.set_ylabel(lab)
        ax.set_xticks(t)
        if i == 0:
            ax.legend(fontsize=8)
    # final panel: belief trajectory of one sample (softmax entropy across steps
    # is not exposed; plot per-step accumulated belief norm as a proxy)
    ax = axes.flat[5]
    norms = []
    for step in range(4):
        b = traj_c['actions'][step]
        norms.append(b[0].norm().item())
    ax.plot(t, norms, 'o-', color=C.PALETTE['indigo'], lw=1.8, ms=5,
            label='gaze path length')
    ax.set_xlabel('foraging step')
    ax.set_ylabel('|gaze| of sample 0')
    ax.set_xticks(t)
    ax.legend(fontsize=8)
    fig.suptitle('Belief-loop dynamics across the fixed T=4 (mean over 6 '
                 'test images; clean vs. PGD-8 ε=0.094)', fontsize=12,
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(C.FIG_DIR, 'recurrent_refinement.png'))
    print('[attention] recurrent_refinement.png', flush=True)


def epsilon_curves():
    plt = C.set_style()
    import json
    with open(os.path.join(C.REPO_ROOT, 'report',
                           'isolation_sweep_results.json')) as f:
        iso = json.load(f)

    def curve(key):
        pts = iso['sweeps'][key]['points']
        xs = [p['eps_norm'] for p in pts]
        ys = [p['acc_pct'] for p in pts]
        return xs, ys

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    styles = [
        ('run_a_norecon', 'Run A — prior OFF (norecon)', C.PALETTE['teal'], 'o-'),
        ('trades_large_baseline_in_run_a', 'TRADES Large baseline',
         C.PALETTE['gray'], 's--'),
        ('run_b_fixedgaze', 'Run B — gaze frozen', C.PALETTE['amber'], 'd-'),
    ]
    for key, lab, col, st in styles:
        xs, ys = curve(key)
        ax.plot(xs, ys, st, color=col, label=lab, lw=2, ms=6)

    # Finding-17 real-only null ablation
    ax.plot([0.0, 0.031, 0.062, 0.094], [47.9, 45.9, 42.6, 39.3],
            '^--', color=C.PALETTE['indigo'], lw=1.6, ms=6,
            label='Finding-17 null ablation (real only)')
    # 3-seed Step 0a point (epoch-41 mid-training v11)
    ax.errorbar([0.094], [27.78], yerr=[2.67], fmt='*', ms=13,
                color=C.PALETTE['rust'], capsize=4, lw=1.6,
                label='Step 0a: ep41 3-seed (27.78±2.67)')
    ax.errorbar([0.0], [44.78], yerr=[4.60], fmt='*', ms=13,
                color=C.PALETTE['rust'], capsize=4, lw=1.6)
    ax.axhline(50.0, color=C.PALETTE['grid'], lw=0.8, ls=':')
    ax.set_xlabel('ε_norm (applied directly in normalized space)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(15, 65)
    ax.legend(fontsize=8.4, loc='lower left')
    ax.set_title('Matched norm-space sweeps (Finding-17 convention) — '
                 'PGD-50, n=500, seed=42', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, 'epsilon_curves.png'))
    print('[attention] epsilon_curves.png', flush=True)


def tsne_embeddings(model, device):
    plt = C.set_style()
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    n = 600
    imgs, labels = C.load_test_images(n=n, seed=3)
    feats = []
    with torch.no_grad():
        for i in range(0, n, 24):
            xb = imgs[i:i + 24].to(device)
            f = model.get_feature_vector(xb)
            feats.append(f.cpu())
    X = torch.cat(feats).numpy()
    y = labels.numpy()
    Xp = PCA(n_components=50, random_state=0).fit_transform(X)
    Xt = TSNE(n_components=2, perplexity=30, init='pca', max_iter=1500,
              random_state=0).fit_transform(Xp)
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    import matplotlib.cm as cm
    cmap = plt.get_cmap('tab10', 10)
    for c in range(10):
        m = y == c
        ax.scatter(Xt[m, 0], Xt[m, 1], s=9, alpha=0.75,
                   color=cmap(c), label=C.STL10_CLASSES[c])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7.6, markerscale=2, ncol=2, loc='upper left')
    ax.set_title('t-SNE of 768-d belief readouts (600 STL-10 test images)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, 'tsne_embeddings.png'))
    print('[attention] tsne_embeddings.png', flush=True)


def confusion_matrices(model, device):
    plt = C.set_style()
    n_clean = 400
    imgs, labels = C.load_test_images(n=n_clean, seed=4)
    imgs = imgs.to(device)
    labels = labels.to(device)

    def evaluate(x):
        preds = []
        with torch.no_grad():
            for i in range(0, x.size(0), 24):
                logits = model(x[i:i + 24])
                if isinstance(logits, tuple):
                    logits = logits[0]
                preds.append(logits.argmax(1).cpu())
        return torch.cat(preds).numpy()

    y = labels.cpu().numpy()
    pred_clean = evaluate(imgs)
    adv = _pgd(model, imgs[:120], eps=0.031, steps=20)
    pred_adv = evaluate(adv)
    y_adv = y[:120]

    def confmat(y_true, y_pred):
        M = np.zeros((10, 10))
        for t, p in zip(y_true, y_pred):
            M[t, p] += 1
        row_sum = M.sum(1, keepdims=True)
        row_sum[row_sum == 0] = 1
        return M / row_sum

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, (M, title) in zip(axes, [
            (confmat(y, pred_clean), 'Clean (400 images)'),
            (confmat(y_adv, pred_adv), 'PGD-20 ε=0.031 (120 images)')]):
        im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(10), C.STL10_CLASSES, rotation=90, fontsize=7.2)
        ax.set_yticks(range(10), C.STL10_CLASSES, fontsize=7.2)
        ax.set_xlabel('predicted')
        ax.set_ylabel('true')
        ax.set_title(title, fontsize=10.5)
        for i in range(10):
            for j in range(10):
                if M[i, j] > 0.04:
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                            fontsize=6.4,
                            color='white' if M[i, j] > 0.5 else 'black')
    fig.suptitle('RHAN-v12 confusion structure on STL-10 test '
                 '(real forward passes)', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(C.FIG_DIR, 'confusion_matrix.png'))
    print('[attention] confusion_matrix.png', flush=True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = C.build_model(load_ckpt=True, device=device)
    model.eval()
    attention_maps(model, device)
    recurrent_refinement(model, device)
    epsilon_curves()
    tsne_embeddings(model, device)
    confusion_matrices(model, device)


if __name__ == '__main__':
    main()
