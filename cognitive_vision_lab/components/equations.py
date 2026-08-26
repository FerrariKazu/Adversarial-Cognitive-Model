"""Educational math blocks rendered as styled, expandable sections."""
from __future__ import annotations

import streamlit as st

from cognitive_vision_lab.utils.math_helpers import EQUATIONS


def equation_block(title: str, latex: str, note: str = "") -> None:
    """Render one equation inside an expander."""
    with st.expander(title, expanded=False):
        st.latex(latex)
        if note:
            st.caption(note)


def equations_sidebar(keys: list[str]) -> None:
    """Show a curated set of equations in an expandable section."""
    with st.expander("Equations", expanded=False):
        for k in keys:
            if k in EQUATIONS:
                title, latex = EQUATIONS[k]
                st.markdown(f"**{title}**")
                st.latex(latex)


def render_equation(key: str) -> None:
    """Render a single named equation from the registry."""
    if key in EQUATIONS:
        title, latex = EQUATIONS[key]
        equation_block(title, latex)
