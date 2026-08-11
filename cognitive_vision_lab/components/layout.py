"""Page scaffolding — hero headers, sections, pills, metric grids, footers."""
from __future__ import annotations

import streamlit as st

from cognitive_vision_lab.config import APP_SUBTITLE, APP_VERSION, GITHUB_URL, PRIMARY_COLOR
from cognitive_vision_lab.utils.theme import inject_css


def hero(title: str, subtitle: str = "", icon: str = "🧠") -> None:
    """Professional hero header for the top of every page."""
    inject_css()
    st.markdown(
        f"""
        <div class="cvl-hero">
          <h1>{icon} {title}</h1>
          <p>{subtitle or APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)
    st.markdown("")


def pill(text: str, kind: str = "info") -> None:
    st.markdown(
        f'<span class="cvl-pill cvl-pill-{kind}">{text}</span>',
        unsafe_allow_html=True,
    )


def metric_grid(items: list[dict]) -> None:
    """items: [{label, value, note}] — rendered as a responsive card row."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="cvl-card">
                  <div class="cvl-card-title">{item.get('label', '')}</div>
                  <div class="cvl-card-value">{item.get('value', '—')}</div>
                  <div class="cvl-card-note">{item.get('note', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def footer() -> None:
    st.markdown("---")
    st.caption(
        f"Cognitive Vision Lab v{APP_VERSION} · {APP_SUBTITLE} · "
        f"[GitHub]({GITHUB_URL})"
    )


def info_banner(text: str, kind: str = "info") -> None:
    fn = {"info": st.info, "warning": st.warning, "success": st.success, "error": st.error}
    fn.get(kind, st.info)(text)


def page_tabs(names: list[str]) -> list:
    return st.tabs(names)


def equation_html(latex: str) -> str:
    return f'<div class="cvl-equation">$${latex}$$</div>'
