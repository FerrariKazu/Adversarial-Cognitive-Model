"""Representation drift: embedding extraction + PCA/t-SNE/UMAP trajectories."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from cognitive_vision_lab.backend.attacks import _forward, pgd


def extract_embedding(model, x: torch.Tensor, layer_key: str = "penultimate") -> np.ndarray:
    """Return a 1-D embedding vector from the model (pre-logits by default)."""
    hook_out: dict = {}

    def hook_fn(m, inp, out):
        hook_out["v"] = out

    root = model.module if hasattr(model, "module") else model
    target = None
    if layer_key == "penultimate":
        # Last Linear/Conv layer before the output head.
        lin_layers = [m for m in root.modules() if isinstance(m, torch.nn.Linear)]
        target = lin_layers[-2] if len(lin_layers) >= 2 else (lin_layers[-1] if lin_layers else None)
        if target is None:
            convs = [m for m in root.modules() if isinstance(m, torch.nn.Conv2d)]
            target = convs[-1] if convs else None
    else:
        for n, m in root.named_modules():
            if n == layer_key:
                target = m
                break
    if target is None:
        raise ValueError(f"No layer found for key {layer_key}")

    h = target.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _forward(model, x)
    finally:
        h.remove()
    v = hook_out["v"]
    if isinstance(v, (tuple, list)):
        v = v[0]
    v = v.detach().float()
    if v.ndim == 4:  # conv feature map -> global average pool
        v = v.mean(dim=(2, 3))
    return v.flatten().cpu().numpy()


def drift_trajectory(model, x, y, eps: float = 0.031, layer_key: str = "penultimate",
                     steps: int = 40) -> dict:
    """Clean → adversarial → recovered embeddings for one image."""
    x_adv = pgd(model, x, y, eps=eps, steps=steps, use_kl=False)
    e_clean = extract_embedding(model, x, layer_key)
    e_adv = extract_embedding(model, x_adv, layer_key)
    e_rec = extract_embedding(model, x, layer_key)  # identity anchor (recovery = clean rep)
    return {
        "clean": e_clean,
        "adversarial": e_adv,
        "recovered": e_rec,
        "eps": eps,
    }


def reduce(points: np.ndarray, method: str = "PCA", n_dims: int = 2,
           perplexity: int = 20, seed: int = 42) -> np.ndarray:
    """Reduce a (N, D) matrix to (N, n_dims) using PCA / t-SNE / UMAP."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[0] == 1 or pts.shape[1] == 1:
        return np.zeros((pts.shape[0], n_dims))
    method = method.upper()
    if method == "PCA":
        from sklearn.decomposition import PCA

        return PCA(n_components=n_dims).fit_transform(pts)
    if method == "TSNE":
        from sklearn.manifold import TSNE

        return TSNE(n_components=n_dims, perplexity=min(perplexity, max(pts.shape[0] - 1, 1)),
                    random_state=seed).fit_transform(pts)
    if method == "UMAP":
        try:
            import umap

            return umap.UMAP(n_components=n_dims, random_state=seed).fit_transform(pts)
        except ImportError:
            from sklearn.manifold import TSNE

            return TSNE(n_components=n_dims, perplexity=min(perplexity, max(pts.shape[0] - 1, 1)),
                        random_state=seed).fit_transform(pts)
    raise ValueError(f"Unknown reduction: {method}")


def multi_image_drift(model, images: list[torch.Tensor], labels, eps: float,
                      layer_key: str = "penultimate") -> dict:
    """Embeddings for a small batch across clean/adv states (for PCA scatter)."""
    from cognitive_vision_lab.backend.attacks import pgd

    emb = {"clean": [], "adv": []}
    for i, x in enumerate(images):
        e0 = extract_embedding(model, x, layer_key)
        xa = pgd(model, x, labels[i].unsqueeze(0), eps=eps, steps=30)
        e1 = extract_embedding(model, xa, layer_key)
        emb["clean"].append(e0)
        emb["adv"].append(e1)
    return emb
