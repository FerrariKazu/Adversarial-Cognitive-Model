"""07 — GradCAM & Saliency: side-by-side explainability with controls."""
import numpy as np
import streamlit as st
import torch

from cognitive_vision_lab.backend.explainability import METHODS, explain
from cognitive_vision_lab.backend.models import class_names, list_models, load_model
from cognitive_vision_lab.components.layout import footer, hero, section
from cognitive_vision_lab.utils.io import denormalize, pil_to_tensor, procedural_sample


@st.cache_resource(show_spinner="Loading model…")
def _load(model_id: str, device: str = "cpu"):
    return load_model(model_id, device=device)


def _conv_layers(model) -> list[str]:
    root = model.module if hasattr(model, "module") else model
    return [n for n, m in root.named_modules() if isinstance(m, torch.nn.Conv2d)]


def _overlay(img: np.ndarray, heat: np.ndarray, opacity: float) -> np.ndarray:
    import matplotlib.cm as cm

    colored = cm.magma(heat)[..., :3]
    return (img * (1 - opacity) + colored * opacity)


def render() -> None:
    hero("GradCAM & Explainability", "Compare attribution methods side by side.")

    models = [m for m in list_models() if m["source"] != "ckpt" or m["available"]]
    names = [m["name"] for m in models]
    sel = st.selectbox("Model", names, index=0)
    entry = next(m for m in models if m["name"] == sel)
    device = st.radio("Device", ["cpu", "cuda"], horizontal=True)

    if not entry["available"] and entry["source"] == "checkpoint":
        st.warning("Checkpoint not present on this host.")
        footer()
        return

    handle = _load(entry["id"], device=device)
    mean, std = ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)) if handle.is_stl10 else (
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    size = 96 if handle.is_stl10 else 224

    image = procedural_sample("dog", size=size)
    x = pil_to_tensor(image, mean, std, size).to(next(handle.model.parameters()).device)
    with torch.no_grad():
        probs = handle.predict(x).squeeze(0)

    labels = class_names(handle)
    top3 = probs.argsort(descending=True)[:3].tolist()

    c1, c2, c3 = st.columns(3)
    methods = c1.multiselect("Methods", list(METHODS.keys()),
                             default=["GradCAM", "GradCAM++", "Integrated Gradients"])
    opacity = c3.slider("Heatmap opacity", 0.0, 1.0, 0.55, 0.05)
    cls_opt = {labels[i] if i < len(labels) else str(i): i for i in top3}
    cls_name = c2.selectbox("Class", list(cls_opt.keys()))
    class_idx = cls_opt[cls_name]
    layers = _conv_layers(handle.model)
    target_layer = c2.selectbox("Target conv layer (CAM)", ["auto (last conv)"] + layers)

    img_np = denormalize(x.cpu().float(), mean, std).clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()

    cols = st.columns(len(methods) + 1)
    cols[0].image((img_np * 255).astype(np.uint8), caption=f"Input · class={cls_name}", width=240)
    for col, method in zip(cols[1:], methods):
        try:
            with st.spinner(f"Computing {method}…"):
                tl = None if target_layer.startswith("auto") else target_layer
                heat = explain(method, handle.model, x, class_idx=class_idx, target_layer=tl)
            overlay = _overlay(img_np, heat, opacity)
            col.image((np.clip(overlay, 0, 1) * 255).astype(np.uint8), caption=method, width=240)
        except Exception as e:  # noqa: BLE001
            col.error(f"{method}: {e}")

    section("Educational context")
    with st.expander(" CAM mathematics", expanded=False):
        from cognitive_vision_lab.components.equations import render_equation

        render_equation("gradcam")
        st.caption("GradCAM++ reweights gradients by second-order terms; EigenCAM takes the "
                   "first principal component of the activation-weighted channels; LayerCAM "
                   "uses element-wise grad × activation; ScoreCAM masks the input per channel "
                   "and weights by softmax gain.")
    footer()


if st.runtime.exists():
    render()
