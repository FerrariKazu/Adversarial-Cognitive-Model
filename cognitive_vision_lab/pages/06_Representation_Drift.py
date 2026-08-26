"""06 — Representation Drift: embedding trajectories under attack."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from cognitive_vision_lab.backend.benchmark import load_drift_stats
from cognitive_vision_lab.backend.embeddings import multi_image_drift, reduce
from cognitive_vision_lab.backend.models import list_models, load_model
from cognitive_vision_lab.components.layout import footer, hero, section
from cognitive_vision_lab.utils.io import pil_to_tensor, procedural_sample


@st.cache_resource(show_spinner="Loading model…")
def _load(model_id: str, device: str = "cpu"):
    return load_model(model_id, device=device)


def _animated_3state(points: list[dict]) -> go.Figure:
    """points: [{label, x, y}...] grouped by state: clean/adv/recovered."""
    fig = go.Figure()
    states = ["clean", "adversarial", "recovered"]
    colors = {"clean": "#16A34A", "adversarial": "#DC2626", "recovered": "#2563EB"}
    for state in states:
        sub = [p for p in points if p["state"] == state]
        if not sub:
            continue
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in sub], y=[p["y"] for p in sub],
            mode="markers+text", name=state,
            marker=dict(size=12, color=colors[state], opacity=0.85),
            text=[p["label"] for p in sub], textposition="top center",
            textfont=dict(size=8),
        ))
    # transition arrows clean -> adv -> recovered
    for i in range(0, len(points) - 2, 3):
        c, a = points[i], points[i + 1]
        fig.add_annotation(x=a["x"], y=a["y"], ax=c["x"], ay=c["y"],
                           axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowsize=1, arrowwidth=1.4, arrowcolor="#94A3B8")
    fig.update_layout(
        template="none", height=520,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        title="Embedding drift: clean → adversarial → recovered",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def render() -> None:
    hero("Representation Drift", "Watch model representations move under attack — PCA, "
                                 "t-SNE, and UMAP trajectories.")

    models = [m for m in list_models() if m["source"] != "ckpt" or m["available"]]
    names = [m["name"] for m in models]
    sel = st.selectbox("Model", names, index=0)
    entry = next(m for m in models if m["name"] == sel)
    device = st.radio("Device", ["cpu", "cuda"], horizontal=True)

    live = entry["available"] or entry["source"] != "checkpoint"
    if not live:
        st.warning("Checkpoint not present on this host.")
        footer()
        return

    method = st.radio("Reduction", ["PCA", "t-SNE", "UMAP"], horizontal=True)
    eps = st.slider("ε attack strength", 0.01, 0.094, 0.031, 0.001, format="%.3f")

    with st.spinner("Computing embeddings…"):
        handle = _load(entry["id"], device=device)
        mean, std = ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)) if handle.is_stl10 else (
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        size = 96 if handle.is_stl10 else 224
        labels = ["a", "b", "c", "d", "e"]
        images = [pil_to_tensor(procedural_sample(l, size=size), mean, std, size)
                  for l in labels]
        y = torch.arange(5, dtype=torch.long)
        xs = torch.cat(images, dim=0).to(next(handle.model.parameters()).device)

        points = []
        try:
            emb = multi_image_drift(handle.model, list(xs), y, eps=eps)
            clean_pts = reduce(np.array(emb["clean"]), method, n_dims=2)
            adv_pts = reduce(np.array(emb["adv"]), method, n_dims=2)
            for i, lbl in enumerate(labels):
                points.append({"label": f"{lbl}+", "x": clean_pts[i, 0], "y": clean_pts[i, 1], "state": "clean"})
                points.append({"label": f"{lbl}·", "x": adv_pts[i, 0], "y": adv_pts[i, 1], "state": "adversarial"})
                points.append({"label": f"{lbl}◦", "x": clean_pts[i, 0], "y": clean_pts[i, 1], "state": "recovered"})
        except Exception as e:  # noqa: BLE001
            st.error(f"Embedding computation failed: {e}")
            st.stop()

    st.plotly_chart(_animated_3state(points), width="stretch")

    st.caption("Green = clean embedding · Red = under attack · Blue = recovered (clean anchor). "
               "Arrows show the perturbation trajectory per sample.")

    section("Accuracy-level drift (real SDT data)", "What happens to *performance* as ε grows?")
    drift = load_drift_stats()
    if drift is not None and not drift.empty:
        st.dataframe(drift.head(200), width="stretch", hide_index=True)
    else:
        st.caption("No precomputed drift stats file found — the accuracy curves on the "
                   "Human vs AI page carry the same story at the metric level.")

    with st.expander(" Dimensionality reduction mathematics", expanded=False):
        from cognitive_vision_lab.components.equations import render_equation

        render_equation("pca")
        render_equation("tsne")
        render_equation("umap")
    footer()


if st.runtime.exists():
    render()
