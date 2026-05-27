"""Tests pour src/agent/semantic_profiler.py (avec FakeLLMProvider)."""

import json

import pandas as pd

from src.agent.semantic_profiler import (
    _normalize_col, _build_col_mapping, analyze_column_sample,
    semantic_profile, suggest_candidate_pairs,
)


# Normalisation des noms de colonnes

class TestNormalizeCol:
    def test_lowercase(self):
        assert _normalize_col("ZIP") == "zip"

    def test_spaces_to_underscores(self):
        assert _normalize_col("Full Name") == "full_name"

    def test_strips_whitespace(self):
        assert _normalize_col("  city  ") == "city"

    def test_mixed(self):
        assert _normalize_col("ADDRESS 2") == "address_2"


class TestBuildColMapping:
    def test_basic_mapping(self):
        df = pd.DataFrame(columns=["ZIP", "Full Name"])
        mapping = _build_col_mapping(df)
        assert mapping == {"zip": "ZIP", "full_name": "Full Name"}

    def test_collision_warns(self, recwarn):
        df = pd.DataFrame(columns=["ZIP", "zip"])
        _build_col_mapping(df)
        assert any("collision" in str(w.message) for w in recwarn.list)


# Analyse d'une colonne

class TestAnalyzeColumnSample:
    def test_contains_samples(self):
        df = pd.DataFrame({"x": ["a", "b", "c", "d", "e"]})
        out = analyze_column_sample(df, "x", sample_size=3)
        assert "Samples" in out
        assert "a" in out and "b" in out and "c" in out

    def test_reports_unique_and_nulls(self):
        df = pd.DataFrame({"x": ["a", "a", None, "b"]})
        out = analyze_column_sample(df, "x")
        assert "Unique values: 2" in out
        assert "Nulls: 1" in out

    def test_uses_display_name(self):
        df = pd.DataFrame({"ZIP": ["1", "2"]})
        out = analyze_column_sample(df, "ZIP", display_name="zip")
        assert "Column: zip" in out


# semantic_profile (avec FakeLLM)

class TestSemanticProfile:
    def test_parses_llm_response(self, fake_llm):
        response = json.dumps({
            "column_types": {"zip": "code_postal", "city": "nom_ville"},
            "transformation_recommendations": {
                "zip": ["prefix_3", "raw"],
                "city": ["raw"],
            },
            "promising_rhs_targets": ["city"],
            "reasoning": "test",
        })
        provider = fake_llm(response)
        df = pd.DataFrame({"ZIP": ["90012"], "CITY": ["LA"]})

        result = semantic_profile(df, llm_provider=provider)

        # Les noms normalisés doivent être remappés vers les noms originaux
        assert "ZIP" in result["column_types"]
        assert "CITY" in result["column_types"]
        assert result["transformation_recommendations"]["ZIP"] == ["prefix_3", "raw"]
        assert "CITY" in result["promising_rhs_targets"]
        assert result["llm_provider"] == "fake"

    def test_strips_markdown_fence(self, fake_llm):
        response = "```json\n" + json.dumps({
            "column_types": {"a": "type"},
            "transformation_recommendations": {"a": ["raw"]},
            "promising_rhs_targets": [],
            "reasoning": "",
        }) + "\n```"
        provider = fake_llm(response)
        df = pd.DataFrame({"a": [1]})
        result = semantic_profile(df, llm_provider=provider)
        assert result["column_types"]["a"] == "type"

    def test_includes_provider_metadata(self, fake_llm):
        response = json.dumps({
            "column_types": {},
            "transformation_recommendations": {},
            "promising_rhs_targets": [],
            "reasoning": "",
        })
        provider = fake_llm(response)
        df = pd.DataFrame({"a": [1]})
        result = semantic_profile(df, llm_provider=provider)
        assert result["llm_provider"] == "fake"
        assert result["llm_model"] == "fake-1"


# suggest_candidate_pairs (avec FakeLLM)

class TestSuggestCandidatePairs:
    def test_parses_valid_pairs(self, fake_llm):
        response = json.dumps([
            {"lhs_col": "zip", "transform": "prefix_3", "rhs": "city"},
        ])
        provider = fake_llm(response)
        df = pd.DataFrame({"ZIP": ["90012"], "CITY": ["LA"]})

        pairs = suggest_candidate_pairs(df, llm_provider=provider)
        assert len(pairs) == 1
        assert pairs[0] == {"lhs_col": "ZIP", "transform": "prefix_3", "rhs": "CITY"}

    def test_rejects_unknown_columns(self, fake_llm, capsys):
        response = json.dumps([
            {"lhs_col": "ghost", "transform": "raw", "rhs": "city"},
            {"lhs_col": "zip", "transform": "prefix_3", "rhs": "city"},
        ])
        provider = fake_llm(response)
        df = pd.DataFrame({"ZIP": ["90012"], "CITY": ["LA"]})

        pairs = suggest_candidate_pairs(df, llm_provider=provider)
        assert len(pairs) == 1
        captured = capsys.readouterr()
        assert "ignor" in captured.out.lower()

    def test_rejects_unknown_transform(self, fake_llm):
        response = json.dumps([
            {"lhs_col": "zip", "transform": "fake_xform", "rhs": "city"},
        ])
        provider = fake_llm(response)
        df = pd.DataFrame({"ZIP": ["90012"], "CITY": ["LA"]})

        pairs = suggest_candidate_pairs(df, llm_provider=provider)
        assert pairs == []

    def test_rejects_self_dependency(self, fake_llm):
        response = json.dumps([
            {"lhs_col": "zip", "transform": "raw", "rhs": "zip"},
        ])
        provider = fake_llm(response)
        df = pd.DataFrame({"ZIP": ["90012"], "CITY": ["LA"]})

        pairs = suggest_candidate_pairs(df, llm_provider=provider)
        assert pairs == []

    def test_rejects_domain_on_non_email(self, fake_llm):
        """La transformation 'domain' n'a de sens que sur des emails."""
        response = json.dumps([
            {"lhs_col": "zip", "transform": "domain", "rhs": "city"},
        ])
        provider = fake_llm(response)
        df = pd.DataFrame({"ZIP": ["90012"], "CITY": ["LA"]})

        pairs = suggest_candidate_pairs(df, llm_provider=provider)
        assert pairs == []

    def test_accepts_domain_on_emails(self, fake_llm):
        response = json.dumps([
            {"lhs_col": "email", "transform": "domain", "rhs": "city"},
        ])
        provider = fake_llm(response)
        df = pd.DataFrame({
            "email": ["a@gmail.com", "b@yahoo.com"],
            "city": ["LA", "NY"],
        })

        pairs = suggest_candidate_pairs(df, llm_provider=provider)
        assert len(pairs) == 1
        assert pairs[0]["transform"] == "domain"

    def test_empty_response_returns_empty(self, fake_llm):
        provider = fake_llm("")
        df = pd.DataFrame({"a": [1]})
        pairs = suggest_candidate_pairs(df, llm_provider=provider)
        assert pairs == []

    def test_malformed_json_returns_empty(self, fake_llm):
        provider = fake_llm("[not valid json,]")
        df = pd.DataFrame({"a": [1]})
        pairs = suggest_candidate_pairs(df, llm_provider=provider)
        assert pairs == []

    def test_extracts_array_from_surrounding_text(self, fake_llm):
        response = (
            'Voici ma réponse :\n'
            '[{"lhs_col": "zip", "transform": "prefix_3", "rhs": "city"}]\n'
            'Fin.'
        )
        provider = fake_llm(response)
        df = pd.DataFrame({"ZIP": ["90012"], "CITY": ["LA"]})
        pairs = suggest_candidate_pairs(df, llm_provider=provider)
        assert len(pairs) == 1
