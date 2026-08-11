"""03 — Interactive Inference: single-image analysis with full diagnostics."""
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from cognitive_vision_lab.backend.models import (
    class_names,
    list_models,
    load_model,
    profile_model,
)
from cognitive_vision_lab.components.equations import render_equation
from cognitive_vision_lab.components.layout import footer, hero, section
from cognitive_vision_lab.utils.io import procedural_sample, stl10_demo_sample
from cognitive_vision_lab.utils.math_helpers import EQUATIONS


@st.cache_resource(show_spinner="Loading model…")
def _load(model_id: str, device: str = "cpu"):
    return load_model(model_id, device=device)


def _rhan_diagnostics(model, x):
    """RHAN-specific trajectory diagnostics if the model supports them."""
    out = {}
    if hasattr(model, "freeze_gaze"):
        out["freeze_gaze"] = bool(getattr(model, "freeze_gaze", False))
    try:
        with st.spinner("Running recurrent forward pass…"):
            _, traj = model(x, return_trajectory=True)
        steps = traj.get("steps", 0)
        out["recurrent_steps"] = int(steps) if steps else None
        out["actions"] = traj.get("actions")
        out["errors"] = traj.get("errors")
        out["precisions"] = traj.get("precisions")
    except Exception:
        pass
    return out


def render() -> None:
    hero("Interactive Inference", "Upload an image and inspect how a model perceives it.")

    models = [m for m in list_models() if m["source"] != "ckpt" or m["available"]]
    names = [m["name"] for m in models]
    sel = st.selectbox("Model", names, index=names.index("ResNet-18") if "ResNet-18" in names else 0)
    entry = next(m for m in models if m["name"] == sel)
    device = st.radio("Device", ["cpu", "cuda"], horizontal=True,
                      help="cuda requires a GPU on this host.")

    c1, c2 = st.columns([1, 1])
    with c1:
        src = st.radio("Image source", ["Demo (STL-10 style)", "Upload"], horizontal=True)
        if src == "Upload":
            up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp"])
            if up is None:
                st.info("Upload an image to continue.")
                footer()
                return
            from PIL import Image

            image = Image.open(up).convert("RGB")
        else:
            image = stl10_demo_sample(3)
        st.image(image, caption="Input", width=260)

    if not entry["available"] and entry["source"] == "checkpoint":
        st.warning(f"Checkpoint `{entry.get('checkpoint')}` is not present on this host. "
                   "Run training or download it from HuggingFace to enable live inference.")
        footer()
        return

    with st.spinner("Loading model…"):
        handle = _load(entry["id"], device=device)
    tfm = handle.transform

    st.markdown("")
    x = tfm(image).unsqueeze(0).to(next(handle.model.parameters()).device)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = handle.predict(x).squeeze(0)
    latency = (time.perf_counter() - t0) * 1000.0

    labels = class_names(handle)
    probs_np = probs.cpu().numpy()
    top5 = np.argsort(probs_np)[-5:][::-1]
    entropy = float(-(probs_np * np.log(probs_np + 1e-12)).sum())

    with c2:
        st.metric("Prediction", labels[top5[0]] if top5[0] < len(labels) else str(top5[0]),
                  delta=f"{probs_np[top5[0]]*100:.1f}% confidence")
        m1, m2, m3 = st.columns(3)
        m1.metric("Entropy (nats)", f"{entropy:.3f}")
        m2.metric("Latency", f"{latency:.1f} ms")
        m3.metric("Top-5 hits", "5/5" if top5[0] < len(labels) else "n/a")
        df = pd.DataFrame({"class": [labels[i] if i < len(labels) else str(i) for i in top5],
                           "probability": [probs_np[i] for i in top5]})
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("class:N", sort="-x"),
            color=alt.value("#2563EB"),
        ).properties(height=220, title="Top-5 classes")
        st.altair_chart(chart, width="stretch")

    with st.expander("📐 Softmax & entropy", expanded=False):
        render_equation("softmax")
        render_equation("cross_entropy")
        st.markdown(f"**Entropy** — $H = -\\sum_i p_i \\log p_i$")

    section("RHAN dynamics", "For recurrent models: foraging steps, precision, error signals.")
    diag = _rhan_diagnostics(handle.model, x)
    if diag.get("recurrent_steps") is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Recurrent steps used", diag["recurrent_steps"])
        if diag.get("precisions"):
            c2.metric("Mean dynamic precision Π_D", f"{torch.stack([p.float().mean() for p in diag['precisions']]).mean().item():.3f}")
        if diag.get("errors"):
            errs = [e.float().mean().item() for e in diag["errors"]]
            c3.metric("Final prediction error", f"{errs[-1]:.4f}" if errs else "n/a")
        if diag.get("actions"):
            st.caption("Gaze trajectory (foveal sampling path) recorded — see Attention Explorer.")
    else:
        st.caption("This model does not expose recurrent trajectory diagnostics "
                   "(feed-forward or unavailable trajectory API).")

    if handle.profile:
        st.caption("Profile: " + ", ".join(f"{k}={v}" for k, v in handle.profile.items()))

    with st.expander("⚙ Model internals", expanded=False):
        st.json({
            "model_id": entry["id"], "name": sel, "family": entry["family"],
            "dataset": entry["dataset"], "stl10": entry["stl10"],
            "device": str(next(handle.model.parameters()).device),
        })
    footer()


if st.runtime.exists():
    render()
