"""Tests pour src/experiments/runner.py"""

import json
import os

import pandas as pd
import pytest

from src.experiments.runner import (
    normalize_workflow_result, run_one, run_experiments,
)


# normalize_workflow_result

class TestNormalizeWorkflowResult:
    def test_extracts_minimal_fields(self):
        raw = {
            "discovered_pfds": [
                {"lhs": "a", "rhs": "b", "support": 10, "confidence": 0.9,
                 "extra_field": "ignored"},
            ],
            "execution_time_seconds": 1.5,
            "total_candidates_tested": 42,
        }
        out = normalize_workflow_result(raw)
        assert len(out["discovered_pfds"]) == 1
        pfd = out["discovered_pfds"][0]
        assert set(pfd.keys()) == {"lhs", "rhs", "support", "confidence"}
        assert pfd["lhs"] == "a"

    def test_preserves_top_level_fields(self):
        raw = {
            "discovered_pfds": [],
            "execution_time_seconds": 2.0,
            "total_candidates_tested": 5,
            "candidates_proposed": [{"x": 1}],
        }
        out = normalize_workflow_result(raw)
        assert out["execution_time_seconds"] == 2.0
        assert out["total_candidates_tested"] == 5
        assert out["candidates_proposed"] == [{"x": 1}]

    def test_missing_fields_default(self):
        out = normalize_workflow_result({})
        assert out["discovered_pfds"] == []
        assert out["execution_time_seconds"] == 0
        assert out["total_candidates_tested"] == 0
        assert out["candidates_proposed"] is None


# run_one (avec workflow classical, sans LLM)

class TestRunOne:
    def test_classical_workflow(self, df_zip_city):
        m = run_one(df_zip_city, "classical", min_support=3, min_confidence=0.9)
        assert m["nb_pfds"] >= 1
        assert m["candidates_tested"] > 0
        assert "discovered_pfds" in m
        assert m["candidates_proposed"] is None

    def test_unknown_workflow_raises(self, df_zip_city):
        with pytest.raises(ValueError, match="Workflow inconnu"):
            run_one(df_zip_city, "ghost_workflow",
                    min_support=1, min_confidence=0.5)


# run_experiments (end-to-end, classical seulement)

class TestRunExperiments:
    def test_writes_results_files(self, tmp_path, df_zip_city):
        # On crée un dataset temporaire
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df_zip_city.to_csv(data_dir / "mini.csv", index=False)

        results_dir = tmp_path / "results"

        run_experiments(
            datasets=["mini.csv"],
            workflows=["classical"],
            providers=[],
            n_runs=1,
            min_support=3,
            min_confidence=0.9,
            data_dir=str(data_dir),
            results_dir=str(results_dir),
        )

        # Un fichier JSON par combinaison
        json_files = list(results_dir.glob("*.json"))
        assert len(json_files) == 1

        with open(json_files[0]) as f:
            content = json.load(f)
        assert content["dataset"] == "mini.csv"
        assert content["workflow"] == "classical"
        assert content["provider"] == "none"
        assert "aggregated" in content
        assert "runs" in content

        # Et le tableau comparatif
        assert (results_dir / "comparison_table.csv").exists()

    def test_skips_missing_dataset(self, tmp_path, capsys):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        results_dir = tmp_path / "results"

        run_experiments(
            datasets=["ghost.csv"],
            workflows=["classical"],
            providers=[],
            n_runs=1,
            data_dir=str(data_dir),
            results_dir=str(results_dir),
        )
        captured = capsys.readouterr()
        assert "SKIP" in captured.out
        assert "ghost.csv" in captured.out

    def test_multiple_runs_aggregated(self, tmp_path, df_zip_city):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df_zip_city.to_csv(data_dir / "mini.csv", index=False)

        results_dir = tmp_path / "results"

        run_experiments(
            datasets=["mini.csv"],
            workflows=["classical"],
            providers=[],
            n_runs=3,
            min_support=3,
            min_confidence=0.9,
            data_dir=str(data_dir),
            results_dir=str(results_dir),
        )

        json_files = list(results_dir.glob("*.json"))
        with open(json_files[0]) as f:
            content = json.load(f)
        assert content["aggregated"]["nb_runs"] == 3
        assert len(content["runs"]) == 3
