"""Page scaffolding — minimal headers, sections, footers."""
from __future__ import annotations

import streamlit as st

from cognitive_vision_lab.config import APP_SUBTITLE, APP_VERSION, GITHUB_URL
from cognitive_vision_lab.utils.theme import inject_css


def hero(title: str, subtitle: str = "", icon: str = "") -> None:
    """Minimal page header — just text, no card/gradient."""
    inject_css()
    label = f"{icon} {title}" if icon else title
    st.markdown(f"# {label}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("")


def section(title: str, caption: str = "") -> None:
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)


def pill(text: str, kind: str = "info") -> None:
    """Inline tag — plain text with a dash prefix."""
    st.markdown(f"`{text}`")


def metric_grid(items: list[dict]) -> None:
    """items: [{label, value, note}] — rendered as st.metric columns."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(
                label=item.get("label", ""),
                value=item.get("value", "—"),
                help=item.get("note") or None,
            )


def footer() -> None:
    st.markdown("---")
    st.caption(f"v{APP_VERSION} · [GitHub]({GITHUB_URL})")


def info_banner(text: str, kind: str = "info") -> None:
    fn = {"info": st.info, "warning": st.warning, "success": st.success, "error": st.error}
    fn.get(kind, st.info)(text)


def page_tabs(names: list[str]) -> list:
    return st.tabs(names)


def equation_html(latex: str) -> str:
    return f"$${latex}$$"
