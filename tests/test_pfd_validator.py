"""Tests pour src/patterns/pfd_validator.py"""

import pandas as pd
import pytest

from src.patterns.extractor import enrich_dataframe
from src.patterns.pfd_validator import compute_support_confidence


# Cas de base

class TestComputeSupportConfidence:
    def test_perfect_dependency(self, df_zip_city):
        df_e = enrich_dataframe(df_zip_city, "ZIP", ["prefix_3"])
        res = compute_support_confidence(df_e, "ZIP__prefix_3", "CITY")
        assert res["confidence"] == 1.0
        assert res["support"] == 12

    def test_partial_dependency(self, df_name_gender):
        df_e = enrich_dataframe(df_name_gender, "name", ["first_token"])
        res = compute_support_confidence(df_e, "name__first_token", "gender")
        # Alex apparaît M et F (1 contre 1 -> mode arbitraire mais 1/2 cohérent)
        # John: 2/2, Mary: 2/2, Bob: 2/2, Alex: 1/2 -> 7/8 = 0.875
        assert res["confidence"] == pytest.approx(0.875, abs=0.001)
        assert res["support"] == 8

    def test_returns_lhs_rhs(self):
        df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
        res = compute_support_confidence(df, "a", "b")
        assert res["lhs"] == "a"
        assert res["rhs"] == "b"

    def test_empty_dataframe_zero_support(self):
        df = pd.DataFrame({"a": [], "b": []})
        res = compute_support_confidence(df, "a", "b")
        assert res["support"] == 0
        assert res["confidence"] == 0


# Champs detailed=False vs detailed=True

class TestDetailed:
    def test_default_no_violations_no_groups(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
        res = compute_support_confidence(df, "a", "b")
        assert "violations" not in res
        assert "groups" not in res

    def test_detailed_adds_fields(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
        res = compute_support_confidence(df, "a", "b", detailed=True)
        assert "violations" in res
        assert "groups" in res
        assert isinstance(res["violations"], list)
        assert isinstance(res["groups"], list)

    def test_violations_captures_inconsistencies(self):
        df = pd.DataFrame({
            "lhs": ["A", "A", "A"],
            "rhs": ["x", "x", "y"],
        })
        res = compute_support_confidence(df, "lhs", "rhs", detailed=True)
        # mode=x (2 occurrences), 1 violation (y)
        assert len(res["violations"]) == 1
        v = res["violations"][0]
        assert v["lhs"] == "A"
        assert v["rhs_found"] == "y"
        assert v["rhs_expected"] == "x"

    def test_groups_details_structure(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [1, 1, 2, 2]})
        res = compute_support_confidence(df, "a", "b", detailed=True)
        assert len(res["groups"]) == 2
        for g in res["groups"]:
            assert {"lhs_value", "size", "dominant_rhs", "confidence", "violations"} <= set(g)


# Gestion des nulls

class TestNullHandling:
    def test_null_rhs_excluded(self, df_with_nulls):
        """
        df_with_nulls :
          code:  ["A1", "A2", None, "B1", "B2", "B3"]
          label: ["alpha", "alpha", "alpha", "beta", None, "beta"]

        Calcul lhs=label, rhs=code :
        - groupby(label) avec dropna=True (défaut) exclut la ligne label=None (B2)
        - group "alpha" : rows 0,1,2 → code=[A1, A2, None] → dropna → [A1, A2]
            size=2, modes={A1,A2} (tie → 1er), consistent=1
        - group "beta"  : rows 3,5 → code=[B1, B3]
            size=2, modes={B1,B3} (tie → 1er), consistent=1
        Total : support = 4, consistent = 2 → confidence = 0.5
        """
        res = compute_support_confidence(df_with_nulls, "label", "code")
        assert res["support"] == 4
        assert res["confidence"] == 0.5

    def test_all_null_rhs_zero_support(self):
        df = pd.DataFrame({
            "a": [1, 1, 2],
            "b": [None, None, None],
        })
        res = compute_support_confidence(df, "a", "b")
        assert res["support"] == 0
        assert res["confidence"] == 0


# Rounding

class TestRounding:
    def test_confidence_rounded_to_4_decimals(self):
        df = pd.DataFrame({
            "a": ["x"] * 7,
            "b": ["1", "1", "1", "1", "1", "1", "2"],
        })
        res = compute_support_confidence(df, "a", "b")
        # 6/7 = 0.857142...
        assert res["confidence"] == 0.8571


# Préservation de l'API publique

class TestPublicApi:
    def test_returns_dict_with_required_keys(self):
        df = pd.DataFrame({"a": [1], "b": [1]})
        res = compute_support_confidence(df, "a", "b")
        assert set(res.keys()) == {"lhs", "rhs", "support", "confidence"}
