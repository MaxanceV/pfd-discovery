"""
Abstraction générique pour les LLM providers.
Support automatique de multiples LLMs via une convention de nommage simple.

Convention de nommage des clés API :
  Toutes les clés API doivent suivre le format : MODEL_NAME_API_KEY
  
  Exemples :
    GOOGLE_API_KEY → Provider Gemini
    GROQ_API_KEY → Provider Groq
    HUGGINGFACE_API_KEY → Provider HuggingFace
    MISTRAL_API_KEY → Provider Mistral
    ANTHROPIC_API_KEY → Provider Claude
    OPENAI_API_KEY → Provider Open AI

Auto-découverte :
  - LLMFactory scanne automatiquement le .env pour les clés *_API_KEY
  - Crée les providers correspondants
  - Format les noms pour affichage lisible ("Open AI" au lieu de "openai")

Architecture :
  - LLMProvider : interface abstraite
  - Implementations : ClaudeProvider, OpenAIProvider, GeminiProvider, GroqProvider, etc.
  - LLMFactory : factory avec auto-découverte
  - format_provider_name() : formate les noms pour l'affichage
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(ABC):
    """Interface abstraite pour tous les LLM providers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider_name = self.__class__.__name__.replace("Provider", "")
    
    @abstractmethod
    def call(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Appelle le LLM avec le prompt donné.
        Retourne la réponse brute du modèle.
        """
        pass
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Vérifie que les credentials sont configurés correctement."""
        pass


# ─── Provider Claude (Anthropic) ──────────────────────────────────────────
class ClaudeProvider(LLMProvider):
    """Provider pour Claude (Anthropic)."""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model_name)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ ANTHROPIC_API_KEY non trouvée dans .env")
        
        from anthropic import Anthropic
        self.client = Anthropic(api_key=self.api_key)
    
    def call(self, prompt: str, max_tokens: int = 2000) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def validate_credentials(self) -> bool:
        return bool(self.api_key)


# ─── Provider OpenAI ─────────────────────────────────────────────────────
class OpenAIProvider(LLMProvider):
    """Provider pour OpenAI (GPT-4, GPT-3.5-turbo)."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(model_name)
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY non trouvée dans .env")
        
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
    
    def call(self, prompt: str, max_tokens: int = 2000) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def validate_credentials(self) -> bool:
        return bool(self.api_key)


# ─── Provider Gemini ────────────────────────────────────────────────
class GeminiProvider(LLMProvider):
    """Provider pour Google Gemini (Méthode forte : API REST directe)."""
    
    # On utilise 1.5-flash : ultra-rapide, gratuit (15 requêtes/min), et toujours en ligne.
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        super().__init__(model_name)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ GOOGLE_API_KEY non trouvée dans .env")
    
    def call(self, prompt: str, max_tokens: int = 2000) -> str:
        import urllib.request
        import json
        
        # URL directe vers l'API moderne de Google (contourne les bugs du SDK)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        # Format attendu par l'API Gemini
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.1
            }
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                # Extraction du texte depuis la réponse complexe de Gemini
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            # S'il y a une erreur HTTP (ex: 429), on essaie de lire le vrai message de Google
            if hasattr(e, 'read'):
                error_body = e.read().decode('utf-8')
                raise RuntimeError(f"Erreur API Gemini ({e.code}) : {error_body}")
            raise RuntimeError(f"Erreur de connexion Gemini : {str(e)}")
            
    def validate_credentials(self) -> bool:
        return bool(self.api_key)


# ─── Provider Groq ───────────────────────────────────────────────────────
class GroqProvider(LLMProvider):
    """Provider pour Groq (modèles rapides)."""
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        super().__init__(model_name)
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY non trouvée dans .env")
        
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("⚠️  groq nécessaire. Installe : pip install groq")
    
    def call(self, prompt: str, max_tokens: int = 2000) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def validate_credentials(self) -> bool:
        return bool(self.api_key)


# ─── Provider HuggingFace ────────────────────────────────────────────────
class HuggingFaceProvider(LLMProvider):
    """Provider pour HuggingFace Inference API (via librairie officielle)."""
    
    # Qwen 2.5 est puissant, rapide, et surtout NON restreint (pas de contrat à signer)
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        super().__init__(model_name)
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ HUGGINGFACE_API_KEY non trouvée dans .env")
        
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_key)
        except ImportError:
            raise ImportError("⚠️  huggingface_hub nécessaire. Installe : pip install huggingface_hub")
    
    def call(self, prompt: str, max_tokens: int = 2000) -> str:
        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Erreur API HuggingFace : {str(e)}")
    
    def validate_credentials(self) -> bool:
        return bool(self.api_key)


# ─── Provider Mistral ────────────────────────────────────────────────────
class MistralProvider(LLMProvider):
    """Provider pour Mistral AI."""
    
    def __init__(self, model_name: str = "mistral-large-latest"):
        super().__init__(model_name)
        self.api_key = os.getenv("MISTRAL_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ MISTRAL_API_KEY non trouvée dans .env")
        
        try:
            from mistralai.client import Mistral
            self.client = Mistral(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(f"⚠️  mistralai nécessaire. Installe : pip install mistralai (Erreur: {e})")
    
    def call(self, prompt: str, max_tokens: int = 2000) -> str:
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Erreur Mistral : {str(e)}")
    
    def validate_credentials(self) -> bool:
        return bool(self.api_key)


# ─── Provider Ollama (Local) ─────────────────────────────────────────────
class OllamaProvider(LLMProvider):
    """Provider pour Ollama (modèles locaux gratuits)."""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("⚠️  requests nécessaire pour Ollama. Installe : pip install requests")
    
    def call(self, prompt: str, max_tokens: int = 2000) -> str:
        import requests
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "num_predict": max_tokens
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Erreur Ollama : {response.text}")
        
        return response.json()["response"]
    
    def validate_credentials(self) -> bool:
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False


# ─── Factory LLM avec auto-découverte ────────────────────────────────────
class LLMFactory:
    """
    Factory pour créer les LLM providers avec auto-découverte.
    
    Scanne automatiquement le .env pour les clés *_API_KEY et 
    les mappe aux providers correspondants.
    """
    
    # Mapping entre les clés API et leurs providers
    # Format : "CLEFS_API_KEY": ("provider_name", ProviderClass)
    API_KEY_TO_PROVIDER = {
        "ANTHROPIC_API_KEY": ("claude", ClaudeProvider),
        "OPENAI_API_KEY": ("openai", OpenAIProvider),
        "GOOGLE_API_KEY": ("gemini", GeminiProvider),
        "GROQ_API_KEY": ("groq", GroqProvider),
        "HUGGINGFACE_API_KEY": ("huggingface", HuggingFaceProvider),
        "MISTRAL_API_KEY": ("mistral", MistralProvider),
    }
    
    @staticmethod
    def get_available_providers() -> Dict[str, str]:
        """
        Retourne les providers disponibles et leurs status.
        Scanne le .env pour les clés API disponibles.
        
        Returns:
            dict { provider_name: "status_message" }
        """
        status = {}
        
        for api_key, (provider_name, ProviderClass) in LLMFactory.API_KEY_TO_PROVIDER.items():
            try:
                if os.getenv(api_key):
                    try:
                        provider = ProviderClass()
                        is_available = provider.validate_credentials()
                        status[provider_name] = "✅ Disponible" if is_available else "❌ Credentials manquantes"
                    except ImportError as ie:
                        status[provider_name] = f"❌ {str(ie)[:60]}"
                    except Exception as e:
                        status[provider_name] = f"❌ {str(e)[:60]}"
                else:
                    status[provider_name] = "❌ Clé API manquante"
            except Exception as e:
                status[provider_name] = f"❌ {str(e)[:50]}"
        
        return status
    
    @staticmethod
    def create(provider_name: str, model_name: Optional[str] = None) -> LLMProvider:
        """
        Crée un provider LLM.
        
        Args:
            provider_name : nom du provider (ex: "claude", "openai", "gemini", "groq", etc.)
            model_name    : optionnel, utilise un défaut si non fourni
        
        Returns:
            Instance du LLMProvider
        
        Raises:
            ValueError si le provider est inconnu
        """
        provider_name = provider_name.lower()
        
        # Trouver dans le mapping
        for api_key, (name, ProviderClass) in LLMFactory.API_KEY_TO_PROVIDER.items():
            if name == provider_name:
                if model_name:
                    return ProviderClass(model_name)
                else:
                    return ProviderClass()
        
        # Provider non trouvé
        available_names = [name for name, _ in LLMFactory.API_KEY_TO_PROVIDER.values()]
        raise ValueError(f"❌ Provider '{provider_name}' inconnu. Disponibles : {', '.join(available_names)}")
    
    @staticmethod
    def list_providers():
        """Affiche les providers disponibles et leur status."""
        print("\n🤖 LLM Providers disponibles :\n")
        status = LLMFactory.get_available_providers()
        
        # Formater avec noms plus lisibles
        for provider_name, msg in status.items():
            display_name = format_provider_name(provider_name)
            print(f"  {display_name:<20} : {msg}")
    
    @staticmethod
    def list_detected_providers() -> List[str]:
        """
        Retourne la liste des providers détectés dans le .env.
        
        Returns:
            list de noms de providers (ex: ["gemini", "groq", "huggingface", "mistral"])
        """
        detected = []
        
        for api_key, (provider_name, _) in LLMFactory.API_KEY_TO_PROVIDER.items():
            if os.getenv(api_key):
                detected.append(provider_name)
        
        return detected


def format_provider_name(provider_name: str) -> str:
    """
    Formate le nom d'un provider pour l'affichage en "Nom Présenté".
    
    Exemples :
      "claude" → "Claude"
      "openai" → "Open AI"
      "gemini" → "Gemini"
      "groq" → "Groq"
      "huggingface" → "Hugging Face"
      "mistral" → "Mistral"
    
    Args:
        provider_name : nom du provider en minuscules
    
    Returns:
        Nom formaté avec majuscule et espaces appropriés
    """
    # Cas spéciaux
    special_cases = {
        "claude": "Claude",
        "openai": "Open AI",
        "gemini": "Gemini",
        "groq": "Groq",
        "huggingface": "Hugging Face",
        "mistral": "Mistral",
        "ollama": "Ollama"
    }
    
    if provider_name in special_cases:
        return special_cases[provider_name]
    
    # Cas général : capitaliser la première lettre
    return provider_name.capitalize()


def get_default_provider() -> LLMProvider:
    """
    Retourne le premier provider disponible détecté dans le .env.
    
    Ordre de priorité : Claude → OpenAI → Gemini → Groq → HuggingFace → Mistral
    
    Raises:
        RuntimeError si aucun provider n'est configuré
    """
    priority_order = ["claude", "openai", "gemini", "groq", "huggingface", "mistral", "ollama"]
    
    for provider_name in priority_order:
        try:
            provider = LLMFactory.create(provider_name)
            if provider.validate_credentials():
                display_name = format_provider_name(provider_name)
                print(f"✅ Utilisation du provider par défaut : {display_name}")
                return provider
        except:
            continue
    
    raise RuntimeError(
        "❌ Aucun LLM provider disponible. Configure au moins une clé API dans .env :\n"
        "   ANTHROPIC_API_KEY (Claude)\n"
        "   OPENAI_API_KEY (Open AI)\n"
        "   GOOGLE_API_KEY (Gemini)\n"
        "   GROQ_API_KEY (Groq)\n"
        "   HUGGINGFACE_API_KEY (Hugging Face)\n"
        "   MISTRAL_API_KEY (Mistral)\n"
        "   Ou démarre Ollama localement"
    )
