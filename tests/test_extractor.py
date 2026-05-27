"""Tests pour src/patterns/extractor.py"""

import numpy as np
import pandas as pd
import pytest

from src.patterns.extractor import (
    extract_prefix, extract_suffix, extract_first_token, extract_last_token,
    extract_domain, enrich_dataframe, enrich_dataframe_multi, TRANSFORMATIONS,
)


# extract_prefix

class TestExtractPrefix:
    def test_basic(self):
        assert extract_prefix("90012", 3) == "900"

    def test_n_equals_length(self):
        assert extract_prefix("90012", 5) == "90012"

    def test_n_greater_than_length(self):
        assert extract_prefix("AB", 3) == "AB"

    def test_none_returns_nan(self):
        assert pd.isna(extract_prefix(None, 3))

    def test_nan_returns_nan(self):
        assert pd.isna(extract_prefix(float("nan"), 3))

    def test_numeric_input_coerced_to_str(self):
        assert extract_prefix(12345, 2) == "12"


# extract_suffix

class TestExtractSuffix:
    def test_basic(self):
        assert extract_suffix("gmail.com", 3) == "com"

    def test_n_greater_than_length(self):
        assert extract_suffix("IL", 3) == "IL"

    def test_none_returns_nan(self):
        assert pd.isna(extract_suffix(None, 2))


# extract_first_token

class TestExtractFirstToken:
    def test_space_separator(self):
        assert extract_first_token("John Smith") == "John"

    def test_comma_separator(self):
        assert extract_first_token("Aarhus, Pam J.") == "Aarhus"

    def test_dot_separator(self):
        assert extract_first_token("Pam.J.Smith") == "Pam"

    def test_strips_whitespace(self):
        assert extract_first_token("  Alice  ") == "Alice"

    def test_empty_string(self):
        assert extract_first_token("") == ""

    def test_none_returns_nan(self):
        assert pd.isna(extract_first_token(None))

    def test_single_word(self):
        assert extract_first_token("Madonna") == "Madonna"


# extract_last_token

class TestExtractLastToken:
    def test_space_separator(self):
        assert extract_last_token("John Smith") == "Smith"

    def test_comma_dot_separator(self):
        assert extract_last_token("Aarhus, Pam J.") == "J"

    def test_single_word(self):
        assert extract_last_token("Chicago") == "Chicago"

    def test_none_returns_nan(self):
        assert pd.isna(extract_last_token(None))


# extract_domain

class TestExtractDomain:
    def test_simple_domain(self):
        assert extract_domain("john@gmail.com") == "gmail.com"

    def test_subdomain(self):
        assert extract_domain("alice@us.example.com") == "us.example.com"

    def test_lowercases(self):
        assert extract_domain("JOHN@GMAIL.COM") == "gmail.com"

    def test_no_at_returns_nan(self):
        assert pd.isna(extract_domain("pas_un_email"))

    def test_none_returns_nan(self):
        assert pd.isna(extract_domain(None))


# TRANSFORMATIONS dict

class TestTransformations:
    def test_contains_all_expected_keys(self):
        expected = {
            "raw", "prefix_1", "prefix_2", "prefix_3", "prefix_4", "prefix_5",
            "suffix_2", "suffix_3", "suffix_4",
            "first_token", "last_token", "domain",
            "uppercase", "lowercase",
        }
        assert set(TRANSFORMATIONS.keys()) == expected

    def test_raw_preserves_value(self):
        assert TRANSFORMATIONS["raw"]("hello") == "hello"

    def test_raw_returns_nan_for_null(self):
        assert pd.isna(TRANSFORMATIONS["raw"](None))

    def test_uppercase(self):
        assert TRANSFORMATIONS["uppercase"]("hello") == "HELLO"

    def test_lowercase(self):
        assert TRANSFORMATIONS["lowercase"]("HELLO") == "hello"

    def test_all_transforms_callable(self):
        for name, fn in TRANSFORMATIONS.items():
            assert callable(fn), f"{name} n'est pas callable"


# enrich_dataframe

class TestEnrichDataframe:
    def test_adds_derived_columns(self):
        df = pd.DataFrame({"zip": ["90012", "10001"], "city": ["LA", "NY"]})
        df_e = enrich_dataframe(df, "zip", ["prefix_3", "prefix_2"])
        assert "zip__prefix_3" in df_e.columns
        assert "zip__prefix_2" in df_e.columns

    def test_preserves_original_columns(self):
        df = pd.DataFrame({"zip": ["90012"], "city": ["LA"]})
        df_e = enrich_dataframe(df, "zip", ["prefix_3"])
        assert "zip" in df_e.columns
        assert "city" in df_e.columns

    def test_values_correct(self):
        df = pd.DataFrame({"zip": ["90012", "90013", "10001"]})
        df_e = enrich_dataframe(df, "zip", ["prefix_3"])
        assert df_e["zip__prefix_3"].tolist() == ["900", "900", "100"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"zip": ["90012"]})
        cols_before = list(df.columns)
        enrich_dataframe(df, "zip", ["prefix_3"])
        assert list(df.columns) == cols_before

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="introuvable"):
            enrich_dataframe(df, "b", ["raw"])

    def test_unknown_transform_warns_and_skips(self, capsys):
        df = pd.DataFrame({"zip": ["90012"]})
        df_e = enrich_dataframe(df, "zip", ["unknown_xform", "prefix_2"])
        captured = capsys.readouterr()
        assert "unknown_xform" in captured.out
        assert "zip__unknown_xform" not in df_e.columns
        assert "zip__prefix_2" in df_e.columns


# enrich_dataframe_multi

class TestEnrichDataframeMulti:
    def test_multiple_columns(self):
        df = pd.DataFrame({
            "zip": ["90012", "10001"],
            "name": ["John Smith", "Susan Miller"],
        })
        df_e = enrich_dataframe_multi(df, {
            "zip": ["prefix_3"],
            "name": ["first_token", "last_token"],
        })
        assert df_e["zip__prefix_3"].tolist() == ["900", "100"]
        assert df_e["name__first_token"].tolist() == ["John", "Susan"]
        assert df_e["name__last_token"].tolist() == ["Smith", "Miller"]

    def test_unknown_column_warns_and_skips(self, capsys):
        df = pd.DataFrame({"a": [1]})
        df_e = enrich_dataframe_multi(df, {"missing": ["raw"], "a": ["raw"]})
        captured = capsys.readouterr()
        assert "missing" in captured.out
        assert "a__raw" in df_e.columns

    def test_empty_config_returns_copy(self):
        df = pd.DataFrame({"a": [1, 2]})
        df_e = enrich_dataframe_multi(df, {})
        assert list(df_e.columns) == ["a"]
        assert df_e is not df  # copie, pas la même réf


# Null safety

class TestNullSafety:
    def test_nan_input_propagates(self):
        """Les valeurs nulles produisent des nan, pas des chaînes vides ou '0'."""
        df = pd.DataFrame({"x": ["abc", None, np.nan, "def"]})
        df_e = enrich_dataframe(df, "x", ["prefix_2", "uppercase"])
        assert df_e["x__prefix_2"].iloc[0] == "ab"
        assert pd.isna(df_e["x__prefix_2"].iloc[1])
        assert pd.isna(df_e["x__prefix_2"].iloc[2])
        assert df_e["x__prefix_2"].iloc[3] == "de"
        assert pd.isna(df_e["x__uppercase"].iloc[1])
