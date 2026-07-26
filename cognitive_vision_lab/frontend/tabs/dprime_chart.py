import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from cognitive_vision_lab.config import SWEEP_PATH, CHECKPOINTS_DIR, CHECKPOINTS_TIER2_DIR
from cognitive_vision_lab.backend.model_registry import list_available_models


def _display_name(ckpt_name: str) -> str:
    name = ckpt_name.replace(".pth", "").replace(":Zone.Identifier", "")
    name = name.replace("_", " ").title()
    name = name.replace("Stl", "STL").replace("Rhan", "RHAN")
    return name


def _sweep_key(ckpt_name: str) -> str:
    key = ckpt_name.replace(".pth", "").replace(":Zone.Identifier", "")
    for suffix in ["_best", "_final", "_rolling", "_checkpoint"]:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def render():
    if not SWEEP_PATH.exists():
        st.warning(f"Sweep results not found at {SWEEP_PATH}")
        return

    with open(SWEEP_PATH) as f:
        data = json.load(f)

    all_ckpts = []
    for d in [CHECKPOINTS_DIR, CHECKPOINTS_TIER2_DIR]:
        if d.exists():
            for fpath in d.iterdir():
                if fpath.name.endswith(".pth") and ":Zone.Identifier" not in fpath.name:
                    all_ckpts.append(fpath.name)
    all_ckpts = sorted(set(all_ckpts))

    data_map = {}
    for ckpt in all_ckpts:
        sk = _sweep_key(ckpt)
        if sk in data:
            data_map[ckpt] = data[sk]
        else:
            for dk in data:
                if dk in ckpt or ckpt.startswith(dk):
                    data_map[ckpt] = data[dk]
                    break

    col_ctrl, col_chart = st.columns([1, 3])

    with col_ctrl:
        st.subheader("Model Selection")
        selected = []
        for ckpt in all_ckpts:
            has_data = ckpt in data_map
            disp = _display_name(ckpt)
            if has_data:
                checked = st.checkbox(disp, value=ckpt in [
                    "rhan_stl10_large_ep45_best.pth",
                    "rhan_v10_final.pth",
                    "rhan_stl10_v11_best.pth",
                ], key=f"dp_{ckpt}")
            else:
                checked = st.checkbox(f"{disp} ⚠ no data", value=False, key=f"dp_{ckpt}", disabled=True)
            if checked:
                selected.append(ckpt)

        st.markdown("---")
        st.info(f"Showing {len(data_map)} of {len(all_ckpts)} checkpoints with sweep data. "
                f"Run **Benchmark** tab to generate data for the rest.")

        metric = st.radio(
            "d' variant",
            ["macro_dprime", "pooled_dprime"],
            format_func=lambda x: x.replace("_", " ").title(),
            horizontal=True,
        )

        st.markdown("---")
        st.markdown(
            "Ref line at **d′ = 1.0** — perceptual collapse threshold. "
            "Humans maintain d′ > 1.0 up to ε ≈ **0.30**."
        )

    with col_chart:
        fig = go.Figure()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                   "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        plotted = 0
        for ckpt in selected:
            if ckpt not in data_map:
                continue
            mod = data_map[ckpt]
            eps = mod["epsilons"]
            dp = mod[metric]
            thresh_key = "thresh_dprime_1_macro" if "macro" in metric else "thresh_dprime_1_pooled"
            thresh = mod.get(thresh_key)
            color = colors[plotted % len(colors)]
            disp = _display_name(ckpt)

            fig.add_trace(go.Scatter(
                x=eps, y=dp, mode="lines+markers",
                name=disp,
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color),
                hovertemplate="<b>%{text}</b><br>ε=%{x}<br>d′=%{y:.3f}<extra></extra>",
                text=[disp] * len(eps),
            ))

            if thresh is not None and thresh in eps:
                idx = eps.index(thresh)
                fig.add_trace(go.Scatter(
                    x=[thresh], y=[dp[idx]],
                    mode="markers",
                    marker=dict(symbol="star", size=16, color=color, line=dict(width=1.5, color="black")),
                    showlegend=False,
                    hovertemplate="<b>εthresh</b>: %{x}<br>d′=%{y:.3f}<extra></extra>",
                ))
            plotted += 1

        if plotted == 0:
            st.info("No models with sweep data selected. Select models with data or run Benchmark tab first.")
            return

        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.6,
                      annotation_text="d′ = 1.0 (Perceptual Collapse)",
                      annotation_position="top left")
        fig.add_hrect(y0=1.0, y1=2.5, x0=0.03, x1=0.32, line_width=0,
                      fillcolor="green", opacity=0.04,
                      annotation_text="Human vision (d′ > 1.0 up to ε=0.30)",
                      annotation_position="top right")

        fig.update_layout(
            title="d′ (Sensitivity Index) vs. Perturbation ε",
            xaxis_title="Perturbation ε (pixel-space)",
            yaxis_title="d′ (Sensitivity)",
            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
            height=500, hovermode="x unified", template="plotly_white",
        )
        fig.update_xaxes(tickformat=".4f")
        fig.update_yaxes(tickformat=".2f")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw Data"):
        rows = []
        for ckpt in selected:
            if ckpt not in data_map:
                continue
            mod = data_map[ckpt]
            disp = _display_name(ckpt)
            for i, eps in enumerate(mod["epsilons"]):
                rows.append({
                    "Model": disp,
                    "ε": eps,
                    "Accuracy %": mod["accuracy"][i],
                    "Macro d'": round(mod["macro_dprime"][i], 4),
                    "Pooled d'": round(mod["pooled_dprime"][i], 4),
                })
        st.dataframe(rows, use_container_width=True)
