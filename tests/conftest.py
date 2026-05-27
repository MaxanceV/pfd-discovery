"""
Fixtures partagées pour la suite pytest.

- Ajoute la racine du projet au sys.path pour que `from src.xxx import yyy` marche.
- Fournit des DataFrames synthétiques reproductibles.
- Fournit un FakeLLMProvider pour tester les workflows agentiques sans appel réseau.
"""

import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.llm_provider import LLMProvider  # noqa: E402


# DataFrames synthétiques

@pytest.fixture
def df_zip_city():
    """
    PFD attendue : ZIP__prefix_3 -> CITY (confidence = 1.0)
    Les 3 premiers chiffres du ZIP déterminent parfaitement la ville.
    """
    return pd.DataFrame({
        "ZIP":  ["90001", "90002", "90003", "10001", "10002", "10003",
                 "60601", "60602", "60603", "33101", "33102", "33103"],
        "CITY": ["Los Angeles"] * 3 + ["New York"] * 3
              + ["Chicago"] * 3 + ["Miami"] * 3,
        "STATE": ["CA"] * 3 + ["NY"] * 3 + ["IL"] * 3 + ["FL"] * 3,
    })


@pytest.fixture
def df_name_gender():
    """
    PFD partielle : first_token(name) -> gender.
    'Alex' apparaît avec les deux genres -> confidence < 1.0
    """
    return pd.DataFrame({
        "name":   ["John Smith", "John Doe", "Mary Jane", "Mary Sue",
                   "Bob Marley", "Bob Dylan", "Alex Kim", "Alex Ross"],
        "gender": ["M", "M", "F", "F", "M", "M", "M", "F"],
    })


@pytest.fixture
def df_with_nulls():
    """DataFrame avec valeurs manquantes des deux côtés."""
    return pd.DataFrame({
        "code":  ["A1", "A2", None, "B1", "B2", "B3"],
        "label": ["alpha", "alpha", "alpha", "beta", None, "beta"],
    })


@pytest.fixture
def df_slide7():
    """
    Dataset du cours, slide 7 :
        A | B | C | D
        1 | 1 | 1 | 1
        1 | 2 | 2 | 1
        2 | 1 | 1 | 2
        2 | 2 | 2 | 2
    FDs attendues : A <-> D, B <-> C
    """
    return pd.DataFrame({
        "A": [1, 1, 2, 2],
        "B": [1, 2, 1, 2],
        "C": [1, 2, 1, 2],
        "D": [1, 1, 2, 2],
    })


# Fake LLM provider

class FakeLLMProvider(LLMProvider):
    """
    Provider de test qui renvoie une réponse fixée à l'avance.
    Permet de tester les workflows sans appel API réel.
    """

    def __init__(self, response: str = "", model_name: str = "fake-1"):
        self.model_name = model_name
        self.provider_name = "fake"
        self.response = response
        self.calls = []

    def call(self, prompt: str, max_tokens: int = 2000) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return self.response

    def validate_credentials(self) -> bool:
        return True


@pytest.fixture
def fake_llm():
    """Factory pour créer un FakeLLMProvider avec une réponse donnée."""
    def _make(response: str = ""):
        return FakeLLMProvider(response=response)
    return _make
