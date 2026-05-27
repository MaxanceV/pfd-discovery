"""Tests pour src/experiments/metrics.py"""

import pandas as pd

from src.experiments.metrics import (
    extract_basic_metrics, aggregate_runs, build_comparison_table,
)


# extract_basic_metrics

class TestExtractBasicMetrics:
    def test_basic(self):
        result = {
            "discovered_pfds": [
                {"lhs": "a", "rhs": "b", "support": 10, "confidence": 0.9},
                {"lhs": "c", "rhs": "d", "support": 20, "confidence": 1.0},
            ],
            "execution_time_seconds": 2.5,
            "total_candidates_tested": 100,
        }
        m = extract_basic_metrics(result)
        assert m["nb_pfds"] == 2
        assert m["execution_time"] == 2.5
        assert m["candidates_tested"] == 100
        assert m["mean_support"] == 15
        assert m["mean_confidence"] == 0.95
        assert m["validation_rate"] == 0.02

    def test_empty_pfds(self):
        result = {
            "discovered_pfds": [],
            "execution_time_seconds": 1.0,
            "total_candidates_tested": 50,
        }
        m = extract_basic_metrics(result)
        assert m["nb_pfds"] == 0
        assert m["mean_support"] == 0.0
        assert m["mean_confidence"] == 0.0
        assert m["validation_rate"] == 0.0

    def test_zero_candidates_validation_rate_none(self):
        result = {
            "discovered_pfds": [],
            "execution_time_seconds": 0,
            "total_candidates_tested": 0,
        }
        m = extract_basic_metrics(result)
        assert m["validation_rate"] is None

    def test_missing_keys_default_to_zero(self):
        m = extract_basic_metrics({})
        assert m["nb_pfds"] == 0
        assert m["execution_time"] == 0.0
        assert m["candidates_tested"] == 0


# aggregate_runs

class TestAggregateRuns:
    def test_empty_returns_empty_dict(self):
        assert aggregate_runs([]) == {}

    def test_single_run_flat(self):
        runs = [{
            "nb_pfds": 5,
            "candidates_tested": 100,
            "mean_support": 12.3,
            "mean_confidence": 0.87,
            "execution_time": 2.5,
            "validation_rate": 0.05,
        }]
        agg = aggregate_runs(runs)
        assert agg["nb_runs"] == 1
        assert agg["nb_pfds"] == 5
        assert agg["mean_support"] == 12.3
        assert agg["mean_confidence"] == 0.87
        assert "execution_time_stdev" not in agg

    def test_multiple_runs_aggregates_mean(self):
        runs = [
            {"nb_pfds": 4, "candidates_tested": 100, "mean_support": 10,
             "mean_confidence": 0.8, "execution_time": 1.0, "validation_rate": 0.04},
            {"nb_pfds": 6, "candidates_tested": 100, "mean_support": 14,
             "mean_confidence": 0.9, "execution_time": 3.0, "validation_rate": 0.06},
        ]
        agg = aggregate_runs(runs)
        assert agg["nb_runs"] == 2
        assert agg["nb_pfds"] == 5  # moyenne
        assert agg["mean_support"] == 12
        assert agg["mean_confidence"] == 0.85
        assert agg["execution_time"] == 2.0
        assert "execution_time_stdev" in agg


# build_comparison_table

class TestBuildComparisonTable:
    def test_basic_table(self):
        results = {
            ("t1.csv", "classical", "none"): {
                "nb_runs": 1, "nb_pfds": 3, "candidates_tested": 50,
                "mean_support": 12.0, "mean_confidence": 0.91,
                "execution_time": 1.5, "validation_rate": 0.06,
            },
            ("t2.csv", "agent_v1", "claude"): {
                "nb_runs": 1, "nb_pfds": 2, "candidates_tested": 20,
                "mean_support": 15.0, "mean_confidence": 0.95,
                "execution_time": 4.2, "validation_rate": 0.10,
            },
        }
        df = build_comparison_table(results)
        assert len(df) == 2
        assert "dataset" in df.columns
        assert "workflow" in df.columns
        assert "llm" in df.columns
        assert "nb_pfds" in df.columns

    def test_empty_returns_empty_dataframe(self):
        df = build_comparison_table({})
        assert df.empty

    def test_sorted_by_dataset_workflow_llm(self):
        results = {
            ("t2.csv", "classical", "none"): {"nb_pfds": 1},
            ("t1.csv", "agent_v1", "claude"): {"nb_pfds": 2},
            ("t1.csv", "classical", "none"): {"nb_pfds": 3},
        }
        df = build_comparison_table(results)
        assert df.iloc[0]["dataset"] == "t1.csv"
        assert df.iloc[0]["workflow"] == "agent_v1"
        assert df.iloc[-1]["dataset"] == "t2.csv"

    def test_column_order(self):
        results = {
            ("a", "b", "c"): {
                "nb_runs": 1, "nb_pfds": 1, "candidates_tested": 1,
                "mean_support": 1, "mean_confidence": 1,
                "execution_time": 1, "validation_rate": 1,
            },
        }
        df = build_comparison_table(results)
        # dataset, workflow, llm doivent être en tête
        assert list(df.columns)[:3] == ["dataset", "workflow", "llm"]
