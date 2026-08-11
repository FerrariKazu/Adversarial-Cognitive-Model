"""Styled dataframe helpers."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def styled_frame(df: pd.DataFrame, height: int = 380, highlight_col: str | None = None) -> None:
    """Render a DataFrame with light column styling."""
    if df.empty:
        st.info("No data available.")
        return
    cols = list(df.columns)
    if highlight_col and highlight_col in df:
        styled = df.style.format(precision=2).background_gradient(
            subset=[highlight_col], cmap="Blues"
        )
        st.dataframe(styled, width="stretch", height=height, hide_index=True)
    else:
        st.dataframe(df.style.format(precision=2), width="stretch",
                     height=height, hide_index=True)


def sortable_table(df: pd.DataFrame, key: str = "tbl") -> pd.DataFrame:
    """Expose sort/filter controls and return the filtered frame."""
    cols = st.multiselect("Columns", df.columns.tolist(), default=df.columns.tolist(),
                          key=f"{key}_cols")
    sort_col = st.selectbox("Sort by", df.columns.tolist(), key=f"{key}_sort")
    ascending = st.toggle("Ascending", value=False, key=f"{key}_asc")
    if not cols:
        return df
    out = df[cols]
    if sort_col in out:
        out = out.sort_values(sort_col, ascending=ascending)
    return out
