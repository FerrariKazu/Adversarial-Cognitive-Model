"""Reusable Plotly chart builders (consistent academic styling)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

PALETTE = ["#2563EB", "#0EA5E9", "#16A34A", "#D97706", "#DC2626",
           "#7C3AED", "#DB2777", "#0891B2"]

_LAYOUT = dict(
    template="none",
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#334155"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=50, r=20, t=50, b=50),
    hoverlabel=dict(bgcolor="white"),
)


def line_fig(x, series: dict[str, list[float]], xlabel: str = "", ylabel: str = "",
             title: str = "") -> go.Figure:
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name,
                                 line=dict(width=2.5, color=PALETTE[i % len(PALETTE)])))
    fig.update_layout(**_LAYOUT, title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    return fig


def dprime_curve_fig(models: list[dict], xlabel: str = "ε (L∞, normalized)",
                     title: str = "Sensitivity d′ vs ε") -> go.Figure:
    """models: [{name, epsilons, dprime}]."""
    fig = go.Figure()
    for i, m in enumerate(models):
        fig.add_trace(go.Scatter(
            x=m["epsilons"], y=m["dprime"], mode="lines+markers", name=m["name"],
            line=dict(width=2.5, color=PALETTE[i % len(PALETTE)]),
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#94A3B8",
                  annotation_text="d′ = 1.0 (collapse threshold)",
                  annotation_position="top right")
    fig.update_layout(**_LAYOUT, title=title, xaxis_title=xlabel,
                      yaxis_title="macro d′")
    return fig


def accuracy_curve_fig(models: list[dict], xlabel: str = "ε (L∞, normalized)",
                       title: str = "Accuracy vs ε") -> go.Figure:
    fig = go.Figure()
    for i, m in enumerate(models):
        fig.add_trace(go.Scatter(x=m["epsilons"], y=m["accuracy"],
                                 mode="lines+markers", name=m["name"],
                                 line=dict(width=2.5, color=PALETTE[i % len(PALETTE)])))
    fig.update_layout(**_LAYOUT, title=title, xaxis_title=xlabel,
                      yaxis_title="Accuracy (%)")
    return fig


def radar_fig(categories: list[str], values: dict[str, list[float]],
              title: str = "") -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(values.items()):
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=categories + categories[:1], fill="toself",
            name=name, line=dict(color=PALETTE[i % len(PALETTE)]),
            opacity=0.55,
        ))
    fig.update_layout(**_LAYOUT, title=title, polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, gridcolor="#E2E8F0"),
    ))
    return fig


def parallel_fig(df: pd.DataFrame, dims: list[str], color: str = "εthresh",
                 title: str = "Parallel coordinates") -> go.Figure:
    def _clean(col):
        return pd.to_numeric(col, errors="coerce")

    line_color = _clean(df[color]) if color in df else df.iloc[:, 0]
    fig = go.Figure(go.Parcoords(
        line=dict(color=line_color, colorscale="Blues", showscale=True),
        dimensions=[
            dict(
                label=c,
                values=_clean(df[c]),
                range=[
                    float(_clean(df[c]).min(skipna=True)),
                    float(_clean(df[c]).max(skipna=True)),
                ],
            )
            for c in dims
        ],
    ))
    fig.update_layout(**_LAYOUT, title=title, height=500)
    return fig


def bar_fig(x, y, names: list[str], ylabel: str, title: str = "",
            color: str = "#2563EB") -> go.Figure:
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=color, text=[f"{v:.1f}" for v in y],
                           textposition="outside"))
    fig.update_layout(**_LAYOUT, title=title, xaxis_title="", yaxis_title=ylabel)
    return fig


def scatter_fig(x, y, names=None, xlabel: str = "", ylabel: str = "",
                title: str = "") -> go.Figure:
    text = names if names is not None and len(names) else None
    fig = px.scatter(x=x, y=y, text=text,
                     color_discrete_sequence=PALETTE, template="none")
    fig.update_traces(marker=dict(size=11), textposition="top center")
    fig.update_layout(**_LAYOUT, title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    return fig


def robust_scatter_fig(x, y, names: list[str], xlabel: str = "", ylabel: str = "",
                       title: str = "", highlight: list[str] | None = None,
                       annotations: list[tuple[float, float, str]] | None = None,
                       height: int | None = None) -> go.Figure:
    """Clean-vs-robust style scatter with hover, highlighted points, and callouts.

    Args:
        highlight: model names (or case-insensitive substrings) drawn as red
            diamonds — e.g. ["RHAN-Large"] to always single out the project's model.
        annotations: [(x, y, text)] callouts placed on the figure.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    names = [str(n) for n in names]
    hl = [h.lower() for h in (highlight or [])]

    def _is_hl(name: str) -> bool:
        low = name.lower()
        return any(h in low for h in hl)

    base_idx = [i for i, n in enumerate(names) if not _is_hl(n)]
    hl_idx = [i for i, n in enumerate(names) if _is_hl(n)]

    fig = go.Figure()
    if base_idx:
        fig.add_trace(go.Scatter(
            x=[xs[i] for i in base_idx], y=[ys[i] for i in base_idx],
            mode="markers+text", text=[names[i] for i in base_idx],
            textposition="top center", textfont=dict(size=10, color="#64748B"),
            marker=dict(size=11, color="#94A3B8", line=dict(color="white", width=1.2)),
            customdata=[names[i] for i in base_idx],
            hovertemplate="%{customdata}<br>%{x:.2f} · %{y:.3f}<extra></extra>",
            name="", showlegend=False,
        ))
    if hl_idx:
        fig.add_trace(go.Scatter(
            x=[xs[i] for i in hl_idx], y=[ys[i] for i in hl_idx],
            mode="markers+text", text=[names[i] for i in hl_idx],
            textposition="top center", textfont=dict(size=11, color="#DC2626"),
            marker=dict(size=17, color="#DC2626", symbol="diamond",
                        line=dict(color="white", width=2)),
            customdata=[names[i] for i in hl_idx],
            hovertemplate="%{customdata}<br>%{x:.2f} · %{y:.3f}<extra></extra>",
            name="highlighted",
        ))
    for ax, ay, txt in (annotations or []):
        fig.add_annotation(x=ax, y=ay, text=txt, showarrow=True, arrowhead=2,
                           arrowcolor="#DC2626", font=dict(color="#DC2626", size=12),
                           bgcolor="rgba(255,255,255,0.9)", ax=-30, ay=-55)
    fig.update_layout(**_LAYOUT, title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    if height:
        fig.update_layout(height=height)
    return fig


def flow_fig(pos: dict, labels: dict, edges: list[tuple[str, str]],
             title: str = "RHAN processing flow") -> go.Figure:
    """Top-down flow diagram from a {key: (x, y)} layout."""
    xs = [pos[k][0] for k in pos]
    ys = [pos[k][1] for k in pos]
    fig = go.Figure()
    for src, dst in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color="#94A3B8", width=1.5),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=26, color="#2563EB", line=dict(color="white", width=1.5)),
        text=[labels[k] for k in pos], textposition="middle center",
        textfont=dict(size=9, color="white"),
        customdata=[k for k in pos], hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT, title=title, height=560,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def sankey_fig(labels: list[str], sources: list[int], targets: list[int],
               values: list[float], title: str = "") -> go.Figure:
    fig = go.Figure(go.Sankey(
        node=dict(pad=14, thickness=18, line=dict(color="white", width=0.5),
                  label=labels, color="#2563EB"),
        link=dict(source=sources, target=targets, value=values, color="rgba(37,99,235,0.35)"),
    ))
    fig.update_layout(**_LAYOUT, title=title, height=520)
    return fig
