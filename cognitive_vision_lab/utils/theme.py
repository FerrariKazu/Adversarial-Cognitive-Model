"""Theme helpers: CSS injection and dark/light mode toggle."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from cognitive_vision_lab.config import ASSETS_DIR, PRIMARY_COLOR

_DARK_CSS = """
[data-testid="stAppViewContainer"] { background-color: #0B1220; }
[data-testid="stHeader"] { background-color: transparent; }
"""


def inject_css() -> None:
    """Inject the professional stylesheet once."""
    css_path = ASSETS_DIR / "style.css"
    css = ""
    if css_path.exists():
        css = css_path.read_text()
    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def theme_toggle() -> None:
    """Dark/light toggle stored in session state; reruns the script on change."""
    dark = st.session_state.get("cvl_dark_mode", False)
    new_dark = st.sidebar.toggle(
        "Dark mode",
        value=dark,
        key="cvl_dark_toggle",
        help="Switch between light and dark themes.",
    )
    if new_dark != dark:
        st.session_state["cvl_dark_mode"] = new_dark
        st.rerun()
    if new_dark:
        st.markdown(f"<style>{_DARK_CSS}</style>", unsafe_allow_html=True)


def brand_header() -> None:
    """Sidebar brand block."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:0.4rem 0 0.8rem;">
              <div style="font-size:1.15rem;font-weight:700;color:{PRIMARY_COLOR};">🧠 Cognitive Vision Lab</div>
              <div style="font-size:0.78rem;color:#64748b;">Interactive Benchmarking Platform<br>for Human-Like AI Vision</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
