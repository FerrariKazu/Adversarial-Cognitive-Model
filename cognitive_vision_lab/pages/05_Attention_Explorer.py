"""05 — Attention Explorer: attention maps across layers and recurrent iterations."""
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from cognitive_vision_lab.backend.models import list_models, load_model
from cognitive_vision_lab.components.layout import footer, hero, section
from cognitive_vision_lab.utils.io import denormalize, pil_to_tensor, procedural_sample


@st.cache_resource(show_spinner="Loading model…")
def _load(model_id: str, device: str = "cpu"):
    return load_model(model_id, device=device)


def _attention_layers(model) -> list[str]:
    names = []
    for n, m in model.named_modules():
        if "attn" in n.lower() or "attention" in n.lower():
            names.append(n)
    return names


def _extract_map(model, x, layer_name: str, size: int = 96) -> np.ndarray:
    store: dict = {}

    def hook(m, inp, out):
        v = out[0] if isinstance(out, (tuple, list)) else out
        store["v"] = v.detach()

    root = model.module if hasattr(model, "module") else model
    target = dict(root.named_modules()).get(layer_name)
    if target is None:
        return None
    h = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        h.remove()
    v = store.get("v")
    if v is None or not torch.is_tensor(v) or v.ndim < 3:
        return None
    v = v.float()
    if v.ndim == 4 and v.shape[1] > 1 and v.shape[1] < 16:
        v = v.mean(dim=1)
    m = v.squeeze().mean(dim=0)
    if m.ndim != 2:
        m = m[0]
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(size, size),
                         mode="bilinear", align_corners=False).squeeze().cpu().numpy()


def _feature_attention(model, x, size: int = 96) -> np.ndarray:
    """CNN fallback: channel-mean activation heatmap as a 'feature attention' proxy."""
    act = {}

    def hook(m, inp, out):
        v = out[0] if isinstance(out, (tuple, list)) else out
        act["v"] = v.detach()

    root = model.module if hasattr(model, "module") else model
    convs = [m for m in root.modules() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        return np.zeros((size, size))
    h = convs[-1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        h.remove()
    v = act["v"].float().abs().mean(dim=1, keepdim=True)
    m = F.interpolate(v, size=(size, size), mode="bilinear", align_corners=False).squeeze()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m.cpu().numpy()


def render() -> None:
    hero("Attention Explorer", "Where does the model look? Attention maps across layers "
                               "and recurrent iterations.")

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

    image = procedural_sample("airplane", size=size)
    x = pil_to_tensor(image, mean, std, size).to(next(handle.model.parameters()).device)

    layers = _attention_layers(handle.model)
    st.markdown("### Controls")
    c1, c2, c3 = st.columns(3)
    if layers:
        layer = c1.selectbox("Attention layer", layers)
        iteration = c2.slider("Iteration", 0, 4, 0,
                              help="For recurrent models: foraging iteration (approx).")
        opacity = c3.slider("Heatmap opacity", 0.0, 1.0, 0.55, 0.05)
        mode = "attention"
    else:
        c1.caption("No explicit attention modules detected — using **feature attention** "
                   "(channel-mean activation of the last conv layer) as a proxy.")
        layer = None
        iteration = c2.slider("Iteration", 0, 4, 0)
        opacity = c3.slider("Heatmap opacity", 0.0, 1.0, 0.55, 0.05)
        mode = "feature"

    if mode == "attention":
        heat = _extract_map(handle.model, x, layer, size=size)
    else:
        heat = _feature_attention(handle.model, x, size=size)
    if heat is None:
        heat = np.zeros((size, size))

    img_np = denormalize(x.cpu().float(), mean, std).clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    import matplotlib.cm as cm

    col1, col2, col3 = st.columns(3)
    col1.image((img_np * 255).astype(np.uint8), caption="Input", width=300)
    col2.image(heat, caption=f"{'Attention' if mode == 'attention' else 'Feature'} map",
               width=300, clamp=True, output_format="PNG")
    overlay = (img_np * (1 - opacity) + cm.magma(heat)[..., :3] * opacity)
    col3.image((np.clip(overlay, 0, 1) * 255).astype(np.uint8),
               caption="Overlay", width=300)

    section("Educational context")
    with st.expander("📐 Attention mathematics", expanded=False):
        from cognitive_vision_lab.components.equations import render_equation

        render_equation("attention")
        if mode == "feature":
            st.caption("Feature attention proxy: A = mean over channels of |activation|, "
                       "normalized and upsampled.")
    footer()


if st.runtime.exists():
    render()
