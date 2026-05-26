# Support Multi-LLM pour la Découverte de PFDs

L'implémentation supporte plusieurs LLMs de manière générique et automatique.

## Convention de nommage

Toutes les clés API suivent le format `MODEL_NAME_API_KEY` :

| Clé API dans `.env` | Provider | Exemple |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude | `ANTHROPIC_API_KEY=sk-ant-...` |
| `OPENAI_API_KEY` | Open AI | `OPENAI_API_KEY=sk-...` |
| `GOOGLE_API_KEY` | Gemini | `GOOGLE_API_KEY=AIza...` |
| `GROQ_API_KEY` | Groq | `GROQ_API_KEY=gsk_...` |
| `HUGGINGFACE_API_KEY` | Hugging Face | `HUGGINGFACE_API_KEY=hf_...` |
| `MISTRAL_API_KEY` | Mistral | `MISTRAL_API_KEY=...` |

Pour ajouter un nouveau modèle : ajoute une clé `NEWMODEL_API_KEY` dans `.env` et enregistre le provider dans `LLMFactory.API_KEY_TO_PROVIDER`.

---

## Auto-découverte

Le système scanne automatiquement le `.env` pour les clés `*_API_KEY` et instancie les providers correspondants.

```python
from src.agent.llm_provider import LLMFactory

LLMFactory.list_providers()
# Output :
#   Groq             : disponible
#   Hugging Face     : disponible
#   Mistral          : disponible
```

---

## Configuration du `.env`

```bash
GOOGLE_API_KEY=Exemple
GROQ_API_KEY=Exemple
HUGGINGFACE_API_KEY=Exemple
MISTRAL_API_KEY=Exemple
```

Ne pas committer le `.env` (il est dans `.gitignore`).

---

## Utilisation

### Exemple 1 : Gemini (par défaut)

```python
import pandas as pd
from src.agent.workflow import workflow_agent_v1

df = pd.read_csv("data/pfd_validation/t2.csv")
results = workflow_agent_v1(df)
print(f"Modèle utilisé : {results['metadata']['llm_provider']}")
```

### Exemple 2 : Groq explicitement

```python
from src.agent.llm_provider import LLMFactory
from src.agent.workflow import workflow_agent_v1

provider = LLMFactory.create("groq")
results = workflow_agent_v1(df, llm_provider=provider)
```

### Exemple 3 : Hugging Face

```python
provider = LLMFactory.create("huggingface")
results = workflow_agent_v1(df, llm_provider=provider)
```

---

## Comparaison automatique des modèles

```python
from src.agent.llm_comparison import full_comparison

results = full_comparison(df)
```

Résultat :

```
Provider        Profiling        Sélection        Qualité
---------------------------------------------------------
GROQ            OK 0.89s         OK 0.65s          0.86%
HUGGING FACE    OK 2.15s         OK 1.87s          0.82%
MISTRAL         OK 1.35s         OK 0.98s          0.89%
```

---

## Formatage des noms pour les rapports

```python
from src.agent.llm_provider import format_provider_name

format_provider_name("openai")       # "Open AI"
format_provider_name("huggingface")  # "Hugging Face"
format_provider_name("mistral")      # "Mistral"
format_provider_name("groq")         # "Groq"
```

---

## Fichier de test

Voir : [tests/test_llm_comparison.py](tests/test_llm_comparison.py)

```bash
python tests/test_llm_comparison.py
```

---

## Ajout d'un nouveau LLM

1. **Créer une classe Provider** dans `src/agent/llm_provider.py` :

```python
class NewModelProvider(LLMProvider):
    """Provider pour NewModel."""

    def __init__(self, model_name: str = "default-model"):
        super().__init__(model_name)
        self.api_key = os.getenv("NEWMODEL_API_KEY")

        if not self.api_key:
            raise ValueError("NEWMODEL_API_KEY non trouvée dans .env")

        from newmodel_sdk import Client
        self.client = Client(api_key=self.api_key)

    def call(self, prompt: str, max_tokens: int = 2000) -> str:
        response = self.client.generate(prompt, max_tokens=max_tokens)
        return response.text

    def validate_credentials(self) -> bool:
        return bool(self.api_key)
```

2. **Ajouter au mapping** dans `LLMFactory` :

```python
API_KEY_TO_PROVIDER = {
    # ... providers existants ...
    "NEWMODEL_API_KEY": ("newmodel", NewModelProvider),
}
```

3. **Utiliser** :

```python
provider = LLMFactory.create("newmodel")
results = workflow_agent_v1(df, llm_provider=provider)
```

---

## Fichiers clés

```
src/agent/
├── llm_provider.py              # Abstraction générique + auto-découverte
├── llm_comparison.py            # Comparaison des modèles
├── semantic_profiler.py         # Support multi-LLM
├── candidate_selector.py        # Support multi-LLM
└── workflow.py                  # Support multi-LLM

tests/
└── test_llm_comparison.py       # Exemples d'utilisation
```

---

## Cas d'usage

### Comparer tous les modèles

```python
from src.agent.llm_comparison import full_comparison

results = full_comparison(df)
```

### Utiliser un modèle spécifique

```python
provider = LLMFactory.create("groq")
results = workflow_agent_v1(df, llm_provider=provider)
```

### Lister les modèles disponibles

```python
from src.agent.llm_provider import LLMFactory

available = LLMFactory.list_detected_providers()
print(f"Modèles détectés : {available}")
```

### Formater les noms pour un rapport

```python
from src.agent.llm_provider import format_provider_name

names = ["openai", "huggingface", "groq"]
formatted = [format_provider_name(n) for n in names]
print(formatted)  # ["Open AI", "Hugging Face", "Groq"]
```
