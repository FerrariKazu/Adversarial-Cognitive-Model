"""Educational math blocks rendered as styled, expandable sections."""
from __future__ import annotations

import streamlit as st

from cognitive_vision_lab.utils.math_helpers import EQUATIONS


def equation_block(title: str, latex: str, note: str = "") -> None:
    """Render one equation in a styled container inside an expander."""
    with st.expander(f"📐 {title}", expanded=False):
        st.markdown(
            f"""
            <div class="cvl-equation">$${latex}$$</div>
            """,
            unsafe_allow_html=True,
        )
        if note:
            st.caption(note)


def equations_sidebar(keys: list[str]) -> None:
    """Show a curated set of equations in an expandable section."""
    with st.expander("📐 Mathematical foundations", expanded=False):
        for k in keys:
            if k in EQUATIONS:
                title, latex = EQUATIONS[k]
                st.markdown(
                    f"<div class='cvl-equation'><b>{title}</b><br>$${latex}$$</div>",
                    unsafe_allow_html=True,
                )


def render_equation(key: str) -> None:
    """Render a single named equation from the registry."""
    if key in EQUATIONS:
        title, latex = EQUATIONS[key]
        equation_block(title, latex)
