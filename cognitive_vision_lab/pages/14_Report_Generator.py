"""14 — Report Generator: research-ready exports in every format."""
from __future__ import annotations

import streamlit as st

from cognitive_vision_lab.backend.benchmark import model_summary_table
from cognitive_vision_lab.backend.reports import (
    latex_table,
    markdown_report,
    pdf_bytes,
    to_csv_bytes,
    to_json_bytes,
    to_markdown_bytes,
)
from cognitive_vision_lab.components.layout import footer, hero, section


def _selection() -> str:
    return st.radio("Report scope", ["All models", "Selected models"],
                    horizontal=True, key="rep_scope")


def render() -> None:
    hero("Report Generator",
         "One click from raw numbers to publication-ready tables.")

    df = model_summary_table()
    if df.empty:
        st.info("No benchmark data to report.")
        footer()
        return

    scope = _selection()
    if scope == "Selected models":
        picks = st.multiselect("Models", df["Model"].tolist(),
                               default=df["Model"].head(5).tolist())
        df = df[df["Model"].isin(picks)] if picks else df

    title = st.text_input("Report title", "Cognitive Vision Lab — Model Benchmark")
    author = st.text_input("Author / affiliation", "RHAN Project")

    section("Preview", "Markdown preview of the generated report.")
    md = markdown_report(df, title=title)
    st.markdown(md)

    # ── Export bundle ─────────────────────────────────────────────────────────
    section("Download bundle")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.download_button("📄 Markdown", to_markdown_bytes(md),
                       file_name="report.md", mime="text/markdown")
    c2.download_button("📐 LaTeX", latex_table(df).encode(),
                       file_name="report_table.tex", mime="application/x-tex")
    c3.download_button("📊 CSV", to_csv_bytes(df),
                       file_name="report.csv", mime="text/csv")
    c4.download_button("🧾 JSON", to_json_bytes(df),
                       file_name="report.json", mime="application/json")
    c5.download_button("📕 PDF", pdf_bytes(df, title=f"{title} — {author}"),
                       file_name="report.pdf", mime="application/pdf")

    with st.expander("📐 Underlying equations", expanded=False):
        from cognitive_vision_lab.components.equations import render_equation

        render_equation("dprime")
        render_equation("pgd")
        st.markdown(
            "**εthresh** = first ε where macro d′ < 1.0. "
            "Attack convention: PGD-50, L∞, perturbation clamped directly in "
            "normalized space (Finding-17)."
        )

    footer()


if st.runtime.exists():
    render()
