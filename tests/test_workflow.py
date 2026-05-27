"""Tests pour src/agent/workflow.py (avec FakeLLMProvider)."""

import json

import pandas as pd

from src.agent.workflow import (
    workflow_classical, workflow_agent_v1, workflow_agent_v2,
)


# workflow_classical (sans LLM)

class TestWorkflowClassical:
    def test_returns_expected_structure(self, df_zip_city):
        result = workflow_classical(df_zip_city, min_support=3, min_confidence=0.9)
        assert "discovered_pfds" in result
        assert "execution_time_seconds" in result
        assert "total_candidates_tested" in result
        assert "metadata" in result
        assert result["metadata"]["approach"] == "classical"

    def test_finds_zip_to_city(self, df_zip_city):
        result = workflow_classical(df_zip_city, min_support=3, min_confidence=0.9)
        lhs_rhs = {(p["lhs"], p["rhs"]) for p in result["discovered_pfds"]}
        assert ("ZIP__prefix_3", "CITY") in lhs_rhs

    def test_metadata_records_params(self, df_zip_city):
        result = workflow_classical(df_zip_city, min_support=5, min_confidence=0.7)
        assert result["metadata"]["min_support"] == 5
        assert result["metadata"]["min_confidence"] == 0.7


# workflow_agent_v1

class TestWorkflowAgentV1:
    def test_calls_llm_and_uses_config(self, fake_llm, df_zip_city):
        response = json.dumps({
            "column_types": {"zip": "code_postal", "city": "ville", "state": "etat"},
            "transformation_recommendations": {
                "zip": ["prefix_3"],
                "city": ["raw"],
                "state": ["raw"],
            },
            "promising_rhs_targets": ["city"],
            "reasoning": "",
        })
        provider = fake_llm(response)

        result = workflow_agent_v1(
            df_zip_city, min_support=3, min_confidence=0.9, llm_provider=provider
        )

        assert result["metadata"]["approach"] == "agent_v1"
        assert result["metadata"]["llm_provider"] == "fake"
        # Le LLM a été appelé une fois
        assert len(provider.calls) == 1
        # La PFD attendue est découverte
        lhs_rhs = {(p["lhs"], p["rhs"]) for p in result["discovered_pfds"]}
        assert ("ZIP__prefix_3", "CITY") in lhs_rhs

    def test_fewer_candidates_than_classical(self, fake_llm, df_zip_city):
        """L'optimisation LLM doit réduire le nombre de candidats testés."""
        response = json.dumps({
            "column_types": {"zip": "code", "city": "ville", "state": "etat"},
            "transformation_recommendations": {
                "zip": ["prefix_3"],  # 1 transform au lieu de 14
                "city": ["raw"],
                "state": ["raw"],
            },
            "promising_rhs_targets": ["city"],
            "reasoning": "",
        })
        provider = fake_llm(response)

        v1 = workflow_agent_v1(
            df_zip_city, min_support=3, min_confidence=0.9, llm_provider=provider
        )
        classical = workflow_classical(
            df_zip_city, min_support=3, min_confidence=0.9
        )
        assert v1["total_candidates_tested"] < classical["total_candidates_tested"]

    def test_missing_column_in_response_falls_back_to_raw(self, fake_llm, df_zip_city):
        # Le LLM oublie 'state' dans sa réponse
        response = json.dumps({
            "column_types": {"zip": "code", "city": "ville"},
            "transformation_recommendations": {
                "zip": ["prefix_3"],
                "city": ["raw"],
            },
            "promising_rhs_targets": [],
            "reasoning": "",
        })
        provider = fake_llm(response)
        # Ne doit pas crasher
        result = workflow_agent_v1(
            df_zip_city, min_support=3, min_confidence=0.9, llm_provider=provider
        )
        assert "discovered_pfds" in result


# workflow_agent_v2

class TestWorkflowAgentV2:
    def test_validates_proposed_pairs(self, fake_llm, df_zip_city):
        response = json.dumps([
            {"lhs_col": "zip", "transform": "prefix_3", "rhs": "city"},
        ])
        provider = fake_llm(response)

        result = workflow_agent_v2(
            df_zip_city, min_support=3, min_confidence=0.9, llm_provider=provider
        )

        assert result["metadata"]["approach"] == "agent_v2"
        assert result["total_candidates_tested"] == 1
        assert len(result["discovered_pfds"]) == 1
        pfd = result["discovered_pfds"][0]
        assert pfd["lhs"] == "ZIP__prefix_3"
        assert pfd["rhs"] == "CITY"
        assert pfd["confidence"] == 1.0

    def test_includes_candidates_proposed(self, fake_llm, df_zip_city):
        response = json.dumps([
            {"lhs_col": "zip", "transform": "prefix_3", "rhs": "city"},
        ])
        provider = fake_llm(response)
        result = workflow_agent_v2(
            df_zip_city, min_support=3, min_confidence=0.9, llm_provider=provider
        )
        assert "candidates_proposed" in result
        assert len(result["candidates_proposed"]) == 1

    def test_empty_proposals_returns_empty(self, fake_llm, df_zip_city):
        provider = fake_llm("[]")
        result = workflow_agent_v2(
            df_zip_city, min_support=3, min_confidence=0.9, llm_provider=provider
        )
        assert result["discovered_pfds"] == []
        assert result["total_candidates_tested"] == 0

    def test_filters_below_thresholds(self, fake_llm):
        """Si la paire validée n'atteint pas le seuil, elle n'est pas retournée."""
        df = pd.DataFrame({
            "code":  ["A", "A", "B"],
            "label": ["x", "y", "z"],  # pas de PFD réelle
        })
        response = json.dumps([
            {"lhs_col": "code", "transform": "raw", "rhs": "label"},
        ])
        provider = fake_llm(response)
        result = workflow_agent_v2(df, min_support=10, min_confidence=0.99,
                                   llm_provider=provider)
        assert result["discovered_pfds"] == []
