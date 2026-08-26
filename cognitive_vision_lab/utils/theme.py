"""Theme helpers: CSS injection."""
from __future__ import annotations

import streamlit as st

from cognitive_vision_lab.config import ASSETS_DIR


def inject_css() -> None:
    """Inject the lab stylesheet once."""
    css_path = ASSETS_DIR / "style.css"
    css = ""
    if css_path.exists():
        css = css_path.read_text()
    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def brand_header() -> None:
    """Sidebar brand — plain text."""
    with st.sidebar:
        st.markdown("**Cognitive Vision Lab**")

