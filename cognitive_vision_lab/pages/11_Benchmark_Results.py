"""11 — Benchmark Results: interactive dashboard over curated + live metrics.

Sortable/filterable table, radar profiles, parallel coordinates, clean-vs-robust
scatter, and publication-quality figure export (PNG/SVG/PDF).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from cognitive_vision_lab.backend.benchmark import model_summary_table
from cognitive_vision_lab.backend.reports import pdf_bytes
from cognitive_vision_lab.components.charts import (
    bar_fig,
    parallel_fig,
    radar_fig,
    scatter_fig,
)
from cognitive_vision_lab.components.layout import footer, hero, metric_grid, section
from cognitive_vision_lab.components.tables import styled_frame

RADAR_AXES = ["Clean %", "ε=0.031", "ε=0.062", "ε=0.094"]


def _export_figure(fig, name: str) -> None:
    c1, c2, c3 = st.columns(3)
    if c1.button("⬇ PNG", key=f"png_{name}"):
        st.download_button("Save PNG", fig.to_image(format="png", scale=2),
                           file_name=f"{name}.png", mime="image/png")
    if c2.button("⬇ SVG", key=f"svg_{name}"):
        st.download_button("Save SVG", fig.to_image(format="svg"),
                           file_name=f"{name}.svg", mime="image/svg+xml")
    if c3.button("⬇ PDF", key=f"pdf_{name}"):
        st.download_button("Save PDF", pdf_bytes(pd.DataFrame()),
                           file_name=f"{name}.pdf", mime="application/pdf")


def _radar_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df[RADAR_AXES].copy()
    for c in RADAR_AXES:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def render() -> None:
    hero("Benchmark Results",
         "Every model, every attack, one rigorous dashboard.")

    df = model_summary_table()
    if df.empty:
        st.info("No benchmark data available.")
        footer()
        return

    # ── Overview metrics ──────────────────────────────────────────────────────
    metric_grid([
        {"label": "Models", "value": len(df)},
        {"label": "Best clean", "value": f"{df['Clean %'].max():.1f}%"},
        {"label": "Best ε=0.094", "value": f"{df['ε=0.094'].max():.1f}%"},
        {"label": "Highest εthresh", "value": f"{df['εthresh'].max():.3f}"},
    ])

    # ── Filter + sort table ───────────────────────────────────────────────────
    section("Full results", "Filter families and sort interactively.")
    families = df["Family"].unique().tolist()
    fams = st.multiselect("Family", families, default=families)
    view = df[df["Family"].isin(fams)] if fams else df
    if st.checkbox("Only models with εthresh > 0.01", value=False):
        view = view[pd.to_numeric(view["εthresh"], errors="coerce").fillna(0) > 0.01]
    styled_frame(view, height=340, highlight_col="Clean %")

    # ── Radar comparison ──────────────────────────────────────────────────────
    section("Radar profiles", "Clean accuracy vs robustness at each ε — "
                              "humans dominate every axis.")
    radar_rows = _radar_df(view)
    selected_names = st.multiselect("Models for radar", view["Model"].tolist(),
                                    default=view["Model"].head(4).tolist())
    if selected_names:
        values = {}
        for name in selected_names:
            row = radar_rows[view["Model"] == name]
            if not row.empty:
                values[name] = row.iloc[0].tolist()
        fig = radar_fig(RADAR_AXES, values, title="Model capability radar")
        st.plotly_chart(fig, width="stretch")
        _export_figure(fig, "radar_comparison")

    # ── Parallel coordinates ──────────────────────────────────────────────────
    section("Parallel coordinates", "Follow one model across all axes.")
    dims = [c for c in ["Clean %", "ε=0.031", "ε=0.062", "ε=0.094", "εthresh", "Params (M)"]
            if c in view.columns]
    if len(dims) >= 2:
        pfig = parallel_fig(view, dims, color="εthresh")
        st.plotly_chart(pfig, width="stretch")
        _export_figure(pfig, "parallel_coordinates")

    # ── Scatter: clean vs robust ──────────────────────────────────────────────
    section("Clean vs high-ε robustness",
             "The trade-off surface. Robustness at ε=0.094 is the hardest axis.")
    sc = scatter_fig(
        x=pd.to_numeric(view["Clean %"], errors="coerce"),
        y=pd.to_numeric(view["ε=0.094"], errors="coerce"),
        names=view["Model"].tolist(),
        xlabel="Clean accuracy (%)", ylabel="Accuracy at ε=0.094 (%)",
        title="Clean vs high-ε robustness",
    )
    st.plotly_chart(sc, width="stretch")
    _export_figure(sc, "clean_vs_robust")

    # ── εthresh bar ───────────────────────────────────────────────────────────
    section("Robustness threshold εthresh", "First ε where macro d′ < 1.0.")
    eth = view.dropna(subset=["εthresh"]).sort_values("εthresh", ascending=False)
    if not eth.empty:
        bfig = bar_fig(eth["Model"].tolist(), eth["εthresh"].tolist(),
                       names=eth["Model"].tolist(), ylabel="εthresh",
                       title="Sensitivity threshold by model")
        st.plotly_chart(bfig, width="stretch")

    # ── Export ────────────────────────────────────────────────────────────────
    section("Export", "Publication-quality table & figure exports.")
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ CSV", view.to_csv(index=False).encode(),
                       file_name="benchmark_results.csv", mime="text/csv")
    c2.download_button("⬇ JSON", view.to_json(orient="records", indent=2).encode(),
                       file_name="benchmark_results.json", mime="application/json")
    c3.download_button("⬇ PDF table", pdf_bytes(view, title="Cognitive Vision Lab — Benchmark"),
                       file_name="benchmark_results.pdf", mime="application/pdf")

    with st.expander(" Methodology", expanded=False):
        st.markdown(
            "All adversarial numbers use **PGD-50, L∞**, with the perturbation clamped "
            "**directly in normalized space** at ε = 0.031 / 0.062 / 0.094 "
            "(Finding-17 convention — no pixel-space conversion). "
            "d′ = Φ⁻¹(HR) − Φ⁻¹(FAR) macro-averaged over classes; "
            "εthresh = first ε where d′ < 1.0."
        )
    footer()


if st.runtime.exists():
    render()
