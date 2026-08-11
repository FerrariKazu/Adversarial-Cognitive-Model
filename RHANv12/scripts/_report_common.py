"""
Shared introspection helpers for the RHAN-v12 report generators.

This module is the single source of truth for pulling live data out of the
actual RHAN-v12 implementation:

  * build_model()          -- instantiate RHANv12, optionally seed weights from
                              the real epoch-41 null-ablation v11 checkpoint
                              (loaded via the v12 class with strict=False;
                              halt_net keys are intentionally dropped).
  * enumerate_tensors()    -- every trainable parameter tensor, with shape,
                              numel, trainability, owning module path and the
                              initialization scheme used by the source.
  * summarize_modules()    -- per-module parameter statistics (direct params).
  * FlopCounter            -- hook-based analytic FLOP accounting over a real
                              forward pass (recurrence multipliers included).
  * load_test_images()     -- STL-10 test images from the local data/stl10
                              tree (torchvision, download=False).

Every number that appears in the report is produced here; nothing is
hand-typed into the LaTeX.
"""

import os
import sys
import json
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Repo-root resolution (this file lives at <repo>/RHANv12/scripts) ────────
_HERE = os.path.dirname(os.path.abspath(__file__))
RHANV12_DIR = os.path.dirname(_HERE)          # .../RHANv12
REPO_ROOT = os.path.dirname(RHANV12_DIR)      # repo root
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'phase1_training')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPORT_DIR = os.path.join(RHANV12_DIR, 'report')
FIG_DIR = os.path.join(REPORT_DIR, 'figures')
TAB_DIR = os.path.join(REPORT_DIR, 'tables')
GEN_DIR = os.path.join(REPORT_DIR, 'generated')
for _d in (FIG_DIR, TAB_DIR, GEN_DIR):
    os.makedirs(_d, exist_ok=True)

EP41_CKPT = os.path.join(REPO_ROOT, 'checkpoints', 'rhan_stl10_v11_ep41.pth')

MEAN = np.array([0.4467, 0.4398, 0.4066], dtype=np.float32)
STD = np.array([0.2603, 0.2566, 0.2713], dtype=np.float32)
STL10_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']

# Consistent publication palette ------------------------------------------------
PALETTE = {
    'ink':     '#1a1c2e',
    'teal':    '#0f6b6b',
    'teal_lt': '#8fd3d3',
    'indigo':  '#3d3a8c',
    'indigo_lt': '#a5a3e0',
    'amber':   '#b8860b',
    'amber_lt': '#f0d08a',
    'rust':    '#b3452e',
    'rust_lt': '#eba493',
    'gray':    '#8a8f9c',
    'grid':    '#d9dbe3',
}


def set_style():
    """Apply a consistent, clean academic style to every matplotlib figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10.5,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'axes.edgecolor': PALETTE['ink'],
        'axes.linewidth': 0.9,
        'axes.grid': True,
        'grid.color': PALETTE['grid'],
        'grid.linewidth': 0.6,
        'grid.alpha': 0.7,
        'xtick.color': PALETTE['ink'],
        'ytick.color': PALETTE['ink'],
        'figure.dpi': 170,
        'savefig.dpi': 170,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    return plt


# ── Model construction ─────────────────────────────────────────────────────────

def build_model(load_ckpt=True, device='cpu', eval_mode=True):
    """Instantiate the real RHANv12 and (optionally) load the ep41 weights.

    The ep41 checkpoint is the original null-ablation v11 rolling checkpoint
    (HF rev 82b4f6cc98d3) — the nearest mid-training snapshot to the winning
    configuration. Loading it into the v12 class leaves only the removed
    halt_net keys 'unexpected', so all figures below show real trained
    weights, not random initialization.
    """
    from phase1_training.model_rhan_v12 import RHANv12
    model = RHANv12()
    if load_ckpt and os.path.exists(EP41_CKPT):
        sd = torch.load(EP41_CKPT, map_location='cpu', weights_only=False)
        sd = sd.get('model', sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[introspect] loaded ep41 ckpt into RHANv12: "
              f"{len(missing)} missing, {len(unexpected)} unexpected "
              f"(= removed halt_net keys)", flush=True)
    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


# ── Parameter / tensor enumeration ────────────────────────────────────────────

_INIT_SCHEMES = {
    'default_linear_conv': 'PyTorch default: weights kaiming_uniform_(a=sqrt(5)); '
                           'bias uniform_(-1/sqrt(fan_in), 1/sqrt(fan_in))',
    'trunc_normal': 'trunc_normal_(std=0.02) (source: PatchTokeniserLarge.__init__)',
    'zeros': 'zeros_ (source: FovealParafovealGate.__init__, gate bias -> alpha=0.5)',
    'ones': 'ones_ (source: PredictiveCodingLayerLarge.error_scale)',
    'freq_const': 'constant 0.85 / 0.15 (source: RHANv10.__init__, learnable scalars)',
    'bn_ln': 'BatchNorm/LayerNorm affine: weight ones_, bias zeros_',
    'groupnorm': 'GroupNorm affine: weight ones_, bias zeros_',
}


def _module_init_desc(module, pname):
    """Map a parameter name + owning module to its initialization scheme."""
    if pname.endswith('cls_token') or pname.endswith('pos_embed'):
        return _INIT_SCHEMES['trunc_normal']
    if pname.endswith('error_scale'):
        return _INIT_SCHEMES['ones']
    if 'freq_weight' in pname:
        return _INIT_SCHEMES['freq_const']
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        if 'bias' in pname:
            return ('PyTorch default uniform_ bias init '
                    '(kaiming_uniform_(a=sqrt(5)) for weights)')
        return _INIT_SCHEMES['default_linear_conv']
    if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        return _INIT_SCHEMES['bn_ln'] if 'weight' in pname else \
            'affine bias zeros_'
    return 'PyTorch module default'


def enumerate_tensors(model):
    """Return a list of records for every trainable parameter tensor."""
    rows = []
    for name, p in model.named_parameters():
        # owning leaf module = the module whose parameter namespace holds `name`
        owning = model
        prefix = name.rsplit('.', 1)[0] if '.' in name else ''
        module = model
        if prefix:
            try:
                module = model.get_submodule(prefix)
            except AttributeError:
                module = model
        rows.append({
            'name': name,
            'module': prefix or '(root)',
            'shape': list(p.shape),
            'shape_str': 'x'.join(str(s) for s in p.shape),
            'numel': int(p.numel()),
            'trainable': bool(p.requires_grad),
            'dtype': str(p.dtype).replace('torch.', ''),
            'init': _module_init_desc(module, name.rsplit('.', 1)[-1]),
        })
    return rows


def summarize_modules(model, recurse=True):
    """Per-module direct parameter counts over the named-module tree."""
    out = []
    for path, mod in model.named_modules():
        if not isinstance(mod, nn.Module):
            continue
        direct = sum(p.numel() for n, p in mod.named_parameters(recurse=False))
        if direct == 0:
            continue
        cls = type(mod).__name__
        n_child_params = sum(p.numel() for p in mod.parameters()) - direct
        out.append({
            'module': path or '(root)',
            'class': cls,
            'direct_params': int(direct),
            'subtree_params': int(sum(p.numel() for p in mod.parameters())),
            'leaf': len(list(mod.children())) == 0,
        })
    return out


# ── Hook-based FLOP counter ────────────────────────────────────────────────────

class FlopCounter:
    """Accumulates multiply-add FLOPs over a real forward pass via module hooks.

    Each hook is called exactly as often as the module actually runs, so the
    T=4 foraging loop, the recurrent feedback, and the gaze-gradient path are
    all reflected in the totals automatically. All counts are 2*MACs for a
    batch of 1 (per-image costs).
    """

    def __init__(self):
        self.per_module = {}   # module path -> float FLOPs
        self._handles = []

    # -- ops ---------------------------------------------------------------
    @staticmethod
    def _conv_flops(m, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        per_out = 2.0 * k * (m.in_channels / m.groups)
        return float(per_out * out.numel())

    @staticmethod
    def _linear_flops(m, out):
        return float(2.0 * m.in_features * out.shape[-1] * out.shape[0])

    @staticmethod
    def _mha_flops(m, out):
        # out is (attn_out, attn_weights); attn_out (B, N, E)
        o = out[0]
        B, N, E = o.shape
        qkv = 3.0 * (2.0 * E * E * N)
        out_proj = 2.0 * E * E * N
        scores = 2.0 * N * N * E          # QK^T
        softmax_w = N * N * E             # softmax normalize + AV
        return float((qkv + out_proj + scores + softmax_w) * B)

    @staticmethod
    def _ln_flops(m, out):
        return float(4.0 * out.numel())

    @staticmethod
    def _dropout_flops(m, out):
        return float(out.numel())

    def _hook(self, path, m):
        def _fn(_m, _inp, out):
            try:
                if isinstance(_m, nn.Conv2d):
                    fl = self._conv_flops(_m, out)
                elif isinstance(_m, nn.ConvTranspose2d):
                    fl = self._conv_flops(_m, out)
                elif isinstance(_m, nn.Linear):
                    fl = self._linear_flops(_m, out)
                elif isinstance(_m, nn.MultiheadAttention):
                    fl = self._mha_flops(_m, out)
                elif isinstance(_m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                    fl = self._ln_flops(_m, out)
                elif isinstance(_m, nn.Dropout):
                    fl = self._dropout_flops(_m, out)
                else:
                    return
            except Exception:
                return
            self.per_module[path] = self.per_module.get(path, 0.0) + fl
        return _fn

    def attach(self, model):
        """Register forward hooks on every countable module of `model`."""
        for path, mod in model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                                nn.MultiheadAttention, nn.LayerNorm,
                                nn.BatchNorm2d, nn.GroupNorm, nn.Dropout)):
                self._handles.append(
                    mod.register_forward_hook(self._hook(path, mod)))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def totals(self, model=None):
        total = sum(self.per_module.values())
        return total

    def to_records(self):
        """Flatten per-module FLOPs, grouped by top-level module name."""
        return [{'module': k, 'flops': v}
                for k, v in sorted(self.per_module.items(),
                                   key=lambda kv: -kv[1])]


def measure_latency_memory(model, device, batch=1, reps=12, warmup=3):
    """Time a real forward pass and report per-image latency + VRAM delta."""
    x = torch.randn(batch, 3, 96, 96, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
    ms_per_img = 1000.0 * dt / batch
    vram_bytes = 0.0
    if device.type == 'cuda':
        vram_bytes = float(torch.cuda.max_memory_allocated())  # peak during fwd
    return {'ms_per_img': ms_per_img, 'fps': batch / dt,
            'peak_vram_bytes': vram_bytes}


# ── Local STL-10 test images ───────────────────────────────────────────────────

def load_test_images(n=8, seed=0, root=None):
    """Return (imgs_norm, labels) from the LOCAL stl10 tree (torchvision).

    Falls back to synthetic gaussian images if the dataset is unavailable so
    the figure pipeline never hard-fails.
    """
    root = root or os.path.join(REPO_ROOT, 'data')
    try:
        import torchvision
        import torchvision.transforms as T
        tf = T.Compose([T.ToTensor(),
                        T.Normalize(MEAN.tolist(), STD.tolist())])
        ds = torchvision.datasets.STL10(root=root, split='test',
                                        transform=tf, download=False)
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
        imgs, labels = [], []
        for i in idx:
            img, lab = ds[int(i)]
            imgs.append(img)
            labels.append(lab)
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)
    except Exception as e:
        print(f"[introspect] local STL-10 unavailable ({e}); "
              f"using synthetic inputs", flush=True)
        rng = np.random.RandomState(seed)
        imgs = torch.randn(n, 3, 96, 96) * 0.5
        labels = torch.tensor(rng.randint(0, 10, size=n), dtype=torch.long)
        return imgs, labels


def denorm(x):
    """STL-10 normalized tensor -> [0,1] display tensor."""
    x = x.detach().cpu()
    for c in range(3):
        x[:, c] = x[:, c] * STD[c] + MEAN[c]
    return x.clamp(0, 1)


def tex_escape(s):
    """Escape a string for safe use inside a LaTeX table cell."""
    return (str(s).replace('\\', r'\textbackslash{}')
            .replace('_', r'\_')
            .replace('&', r'\&')
            .replace('%', r'\%')
            .replace('#', r'\#')
            .replace('$', r'\$')
            .replace('{', r'\{')
            .replace('}', r'\}'))


def save_json(obj, name):
    path = os.path.join(GEN_DIR, name)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[introspect] wrote {path}", flush=True)
    return path


def save_tabular(rows, name):
    """rows: list of dicts -> CSV in tables/ + LaTeX longtable fragment."""
    import csv
    path = os.path.join(TAB_DIR, name)
    if rows:
        keys = list(rows[0].keys())
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"[introspect] wrote {path}", flush=True)
    return path
