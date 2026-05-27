"""Tests pour src/patterns/pfd_discovery.py"""

import pandas as pd

from src.patterns.pfd_discovery import discover_pfds


# Cas nominal

class TestDiscoverPfds:
    def test_finds_expected_pfd(self, df_zip_city):
        pfds, stats = discover_pfds(df_zip_city, min_support=3, min_confidence=0.9)
        lhs_rhs = {(p["lhs"], p["rhs"]) for p in pfds}
        assert ("ZIP__prefix_3", "CITY") in lhs_rhs

    def test_returns_stats(self, df_zip_city):
        pfds, stats = discover_pfds(df_zip_city, min_support=3, min_confidence=0.9)
        assert "execution_time_seconds" in stats
        assert "total_candidates_tested" in stats
        assert stats["total_candidates_tested"] > 0
        assert stats["execution_time_seconds"] >= 0

    def test_each_pfd_has_required_keys(self, df_zip_city):
        pfds, _ = discover_pfds(df_zip_city, min_support=3, min_confidence=0.9)
        for p in pfds:
            assert {"lhs", "rhs", "support", "confidence"} <= set(p.keys())


# Filtres min_support / min_confidence

class TestFiltering:
    def test_min_support_filters(self, df_zip_city):
        # Avec support=100, aucun pattern n'a assez de tuples (12 lignes)
        pfds, _ = discover_pfds(df_zip_city, min_support=100, min_confidence=0.5)
        assert pfds == []

    def test_min_confidence_filters(self, df_name_gender):
        # Avec conf=0.99, aucune PFD partielle ne passe
        pfds, _ = discover_pfds(df_name_gender, min_support=2, min_confidence=0.99)
        # first_token(name) -> gender avec Alex = 0.875, donc rejeté
        assert not any(
            p["lhs"] == "name__first_token" and p["rhs"] == "gender"
            for p in pfds
        )


# Config personnalisée

class TestConfig:
    def test_config_restricts_search(self, df_zip_city):
        config = {"ZIP": ["prefix_3"], "CITY": ["raw"], "STATE": ["raw"]}
        pfds, stats = discover_pfds(
            df_zip_city, min_support=3, min_confidence=0.9, config=config
        )
        full_pfds, full_stats = discover_pfds(
            df_zip_city, min_support=3, min_confidence=0.9
        )
        # Config restreinte => moins de candidats testés
        assert stats["total_candidates_tested"] < full_stats["total_candidates_tested"]

    def test_config_missing_column_warns(self, df_zip_city, capsys):
        config = {"GHOST_COL": ["raw"]}
        discover_pfds(df_zip_city, min_support=3, min_confidence=0.9, config=config)
        captured = capsys.readouterr()
        assert "GHOST_COL" in captured.out


# Filtre auto-dépendance (fix #2)

class TestSelfDependencyFilter:
    def test_self_dep_filtered(self):
        df = pd.DataFrame({"name": ["John", "Mary", "Bob"]})
        pfds, _ = discover_pfds(df, min_support=1, min_confidence=0.99)
        # name__raw -> name doit être filtré
        assert not any(
            p["lhs"] == "name__raw" and p["rhs"] == "name" for p in pfds
        )

    def test_shared_prefix_columns_not_filtered_wrongly(self):
        """Régression du bug startswith : 'name_full__raw' ne doit PAS être
        filtré sous prétexte qu'il commence par 'name'."""
        df = pd.DataFrame({
            "name":      ["John", "John", "Mary"],
            "name_full": ["John Smith", "John Doe", "Mary Jane"],
        })
        pfds, stats = discover_pfds(df, min_support=1, min_confidence=0.5)
        # On doit pouvoir tester name_full__raw -> name (différentes colonnes)
        # Le candidat doit avoir été évalué (pas filtré par le bug startswith)
        lhs_rhs = {(p["lhs"], p["rhs"]) for p in pfds}
        # name_full unique pour chaque ligne -> name_full__raw -> name : conf = 1.0
        assert ("name_full__raw", "name") in lhs_rhs


# Empty / edge cases

class TestEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        pfds, stats = discover_pfds(df, min_support=1, min_confidence=0.5)
        assert pfds == []
        assert stats["total_candidates_tested"] >= 0

    def test_single_column_no_pfds(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        pfds, _ = discover_pfds(df, min_support=1, min_confidence=0.5)
        # Toutes les paires lhs -> rhs avec lhs.split('__')[0] == rhs sont filtrées
        assert pfds == []
