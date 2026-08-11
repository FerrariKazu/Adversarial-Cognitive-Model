"""Unit tests for cognitive_vision_lab.backend.reports."""
import json

import pandas as pd
import pytest

from cognitive_vision_lab.backend.reports import (
    latex_table,
    markdown_report,
    pdf_bytes,
    to_csv_bytes,
    to_json_bytes,
    to_markdown_bytes,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "Model": ["RHAN-Large", "ResNet-18", "Human"],
        "Clean %": [52.6, 83.1, 84.0],
        "ε=0.094": [33.7, 0.0, 66.0],
    })


class TestMarkdown:
    def test_contains_sections(self, df):
        md = markdown_report(df)
        assert "# Benchmark Report" in md
        assert "## Summary" in md
        assert "## Results" in md
        assert "RHAN-Large" in md

    def test_empty(self):
        md = markdown_report(pd.DataFrame())
        assert "No models selected" in md

    def test_highlights_top_clean(self, df):
        md = markdown_report(df)
        assert "Human" in md.split("Highest clean accuracy")[1][:60]


class TestLatex:
    def test_table_syntax(self, df):
        tex = latex_table(df)
        assert "\\begin{table}" in tex
        assert "\\toprule" in tex and "\\bottomrule" in tex
        assert "ResNet" in tex

    def test_escapes_underscores(self, df):
        tex = latex_table(pd.DataFrame({"A": ["foo_bar"]}))
        assert "foo\\_bar" in tex


class TestByteExports:
    def test_csv(self, df):
        raw = to_csv_bytes(df).decode()
        assert "Model,Clean %" in raw

    def test_json_roundtrip(self, df):
        records = json.loads(to_json_bytes(df))
        assert len(records) == 3
        assert records[0]["Model"] == "RHAN-Large"

    def test_markdown_bytes(self):
        assert to_markdown_bytes("hello").decode() == "hello"

    def test_pdf_starts_with_pdf_magic(self, df):
        data = pdf_bytes(df)
        assert data[:4] == b"%PDF"
