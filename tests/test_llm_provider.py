"""
Tests pour src/agent/llm_provider.py.

Aucun appel réseau n'est fait — on teste uniquement la factory, la détection,
et le comportement quand aucun provider n'est configuré.
"""

import pytest

from src.agent.llm_provider import LLMFactory, get_default_provider


# Factory : routing par nom

class TestFactoryCreate:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="inconnu"):
            LLMFactory.create("provider_inexistant")

    def test_case_insensitive(self, monkeypatch):
        # Sans clé API ni package, on attend ValueError ou ImportError
        # mais PAS "inconnu" -> le nom est bien reconnu
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(Exception) as exc_info:
            LLMFactory.create("CLAUDE")
        assert "inconnu" not in str(exc_info.value)

    def test_unknown_lists_available(self):
        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create("fake_xyz")
        msg = str(exc_info.value)
        # Le message doit lister les providers connus
        assert "claude" in msg


# list_detected_providers

class TestListDetected:
    def test_returns_list(self, monkeypatch):
        # On nettoie l'env pour ne rien détecter
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                    "GROQ_API_KEY", "HUGGINGFACE_API_KEY", "MISTRAL_API_KEY",
                    "OLLAMA_BASE_URL"]:
            monkeypatch.delenv(key, raising=False)
        assert LLMFactory.list_detected_providers() == []

    def test_detects_single_key(self, monkeypatch):
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                    "GROQ_API_KEY", "HUGGINGFACE_API_KEY", "MISTRAL_API_KEY",
                    "OLLAMA_BASE_URL"]:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "fake_key")
        assert LLMFactory.list_detected_providers() == ["groq"]


# get_default_provider

class TestGetDefaultProvider:
    def test_raises_when_no_provider(self, monkeypatch):
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                    "GROQ_API_KEY", "HUGGINGFACE_API_KEY", "MISTRAL_API_KEY",
                    "OLLAMA_BASE_URL"]:
            monkeypatch.delenv(key, raising=False)
        with pytest.raises(RuntimeError, match="Aucun LLM provider"):
            get_default_provider()


# API_KEY_TO_PROVIDER mapping

class TestKeyMapping:
    def test_mapping_covers_documented_providers(self):
        expected_keys = {
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
            "GROQ_API_KEY", "HUGGINGFACE_API_KEY", "MISTRAL_API_KEY",
            "OLLAMA_BASE_URL",
        }
        assert set(LLMFactory.API_KEY_TO_PROVIDER.keys()) == expected_keys

    def test_each_entry_has_name_and_class(self):
        for key, (name, cls) in LLMFactory.API_KEY_TO_PROVIDER.items():
            assert isinstance(name, str) and name == name.lower()
            assert callable(cls)


# FakeLLMProvider (vient de conftest)

class TestFakeProvider:
    def test_fake_provider_records_calls(self, fake_llm):
        provider = fake_llm("hello world")
        result = provider.call("prompt 1")
        assert result == "hello world"
        assert len(provider.calls) == 1
        assert provider.calls[0]["prompt"] == "prompt 1"

    def test_fake_provider_validates(self, fake_llm):
        assert fake_llm().validate_credentials() is True

    def test_fake_provider_has_required_attrs(self, fake_llm):
        provider = fake_llm()
        assert provider.provider_name == "fake"
        assert provider.model_name == "fake-1"
