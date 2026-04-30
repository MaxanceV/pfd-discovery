# 🤖 Support Multi-LLM pour la Découverte de PFDs

Ton implémentation supporte maintenant **tous les LLMs** de manière générique et automatique ! 

## 🎯 Convention de nommage (IMPORTANT ⭐)

**Toutes les clés API doivent suivre le format :**

```
MODEL_NAME_API_KEY
```

### Exemples

| Clé API dans `.env` | Provider | Format | Exemple |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Claude | `ANTHROPIC_API_KEY=sk-ant-...` | ✅ Implémenté |
| `OPENAI_API_KEY` | Open AI | `OPENAI_API_KEY=sk-...` | ✅ Implémenté |
| `GOOGLE_API_KEY` | Gemini | `GOOGLE_API_KEY=AIza...` | ✅ Implémenté |
| `GROQ_API_KEY` | Groq | `GROQ_API_KEY=gsk_...` | ✅ Implémenté |
| `HUGGINGFACE_API_KEY` | Hugging Face | `HUGGINGFACE_API_KEY=hf_...` | ✅ Implémenté |
| `MISTRAL_API_KEY` | Mistral | `MISTRAL_API_KEY=...` | ✅ Implémenté |

**Ça marche aussi pour de nouveaux modèles** : juste ajoute une clé `NEWMODEL_API_KEY` au `.env` et le code fonctionnera automatiquement !

---

## 📦 Auto-découverte générique

Le système scanne **automatiquement** le `.env` pour trouver les clés `*_API_KEY` et crée les providers correspondants.

```python
# Auto-découverte : détecte toutes les clés API configurées
from src.agent.llm_provider import LLMFactory

LLMFactory.list_providers()
# Output :
# 🤖 LLM Providers disponibles :
#   Gemini           : ✅ Disponible
#   Groq             : ✅ Disponible
#   Hugging Face     : ✅ Disponible
#   Mistral          : ✅ Disponible
```

---

## 🔧 Configuration du `.env`

Tu peux configurer **autant de clés API que tu veux** en suivant la convention :

```bash
# Requis : au moins une clé API
GOOGLE_API_KEY=AIzaSyAkyVOvTXMi-pXVheK-dZR4vk7cr9SsgXs
GROQ_API_KEY=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
HUGGINGFACE_API_KEY=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
MISTRAL_API_KEY=YRgKRtVo9MY4zF0GqWf5GrpQQfBCURno
```

**Attention :** Ne pas commit le `.env` (il est dans `.gitignore` ✅)

---

## 🚀 Utilisation simple

### Exemple 1 : Utiliser Gemini (par défaut)

```python
import pandas as pd
from src.agent.workflow import workflow_agent_v1

df = pd.read_csv("data/pfd_validation/t2.csv")

# Utilise automatiquement Gemini (détecté du .env)
results = workflow_agent_v1(df)

print(f"Modèle utilisé : {results['metadata']['llm_provider']}")
```

### Exemple 2 : Utiliser Groq explicitement

```python
from src.agent.llm_provider import LLMFactory
from src.agent.workflow import workflow_agent_v1

provider = LLMFactory.create("groq")
results = workflow_agent_v1(df, llm_provider=provider)
```

### Exemple 3 : Utiliser Hugging Face

```python
provider = LLMFactory.create("huggingface")
results = workflow_agent_v1(df, llm_provider=provider)
```

---

## 📊 Comparaison automatique des modèles

```python
from src.agent.llm_comparison import full_comparison, export_results

# Compare TOUS les modèles détectés automatiquement
results = full_comparison(df)

# Exporte les résultats
export_results(results, output_file="results/llm_comparison.json")
```

**Résultat :**

```
Provider        Profiling        Sélection        Qualité
───────────────────────────────────────────────────────
GEMINI          ✅ 1.50s         ✅ 1.23s          0.88%
GROQ            ✅ 0.89s         ✅ 0.65s          0.86%
HUGGING FACE    ✅ 2.15s         ✅ 1.87s          0.82%
MISTRAL         ✅ 1.35s         ✅ 0.98s          0.89%
```

---

## 🔍 Formatage des noms pour les rapports

La fonction `format_provider_name()` formate automatiquement les noms :

```python
from src.agent.llm_provider import format_provider_name

format_provider_name("openai")       # → "Open AI"
format_provider_name("huggingface")  # → "Hugging Face"
format_provider_name("mistral")      # → "Mistral"
format_provider_name("groq")         # → "Groq"
```

**Utilise cette fonction pour tes tableaux d'analyse et rapports !**

---

## 🧪 Fichier de test

Vois : [tests/test_llm_comparison.py](tests/test_llm_comparison.py)

Contient des exemples prêts à exécuter :

```bash
python tests/test_llm_comparison.py
```

---

## ⚙️ Ajout d'un nouveau LLM (extensibilité)

**Pour ajouter un nouveau modèle (ex: Anthropic Claude Batch, LLaMA via API, etc.) :**

1. **Créer une classe Provider** dans `src/agent/llm_provider.py` :

```python
class NewModelProvider(LLMProvider):
    """Provider pour NewModel."""
    
    def __init__(self, model_name: str = "default-model"):
        super().__init__(model_name)
        self.api_key = os.getenv("NEWMODEL_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ NEWMODEL_API_KEY non trouvée dans .env")
        
        # Initialiser le client
        from newmodel_sdk import Client
        self.client = Client(api_key=self.api_key)
    
    def call(self, prompt: str, max_tokens: int = 2000) -> str:
        response = self.client.generate(prompt, max_tokens=max_tokens)
        return response.text
    
    def validate_credentials(self) -> bool:
        return bool(self.api_key)
```

2. **Ajouter au mapping** :

```python
# Dans LLMFactory
API_KEY_TO_PROVIDER = {
    # ... providers existants ...
    "NEWMODEL_API_KEY": ("newmodel", NewModelProvider),
}
```

3. **Utiliser immédiatement** :

```python
provider = LLMFactory.create("newmodel")
results = workflow_agent_v1(df, llm_provider=provider)
```

**C'est tout !** Le reste fonctionne automatiquement. ✨

---

## 📈 Prochaines étapes

1. ✅ **Configure le `.env`** en suivant la convention `MODEL_NAME_API_KEY`

2. ✅ **Teste les providers** :
   ```bash
   python tests/test_llm_comparison.py
   ```

3. ✅ **Lance des comparaisons** pour analyser les différences :
   ```python
   from src.agent.llm_comparison import full_comparison
   
   results = full_comparison(df)  # Teste TOUS les providers du .env
   ```

4. ✅ **Documente les résultats** dans ton rapport final en formatant les noms avec `format_provider_name()`

---

## 🔗 Fichiers clés

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

## 💡 Cas d'usage

### Comparaison rapide de tous les modèles
```python
from src.agent.llm_comparison import full_comparison

results = full_comparison(df)
```

### Utiliser un modèle spécifique
```python
provider = LLMFactory.create("groq")
results = workflow_agent_v1(df, llm_provider=provider)
```

### Déterminer les modèles disponibles
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

---

## ✨ Avantages

✅ **Générique** : ajoute nouveaux LLMs sans changer le code  
✅ **Auto-détection** : scanne `.env` automatiquement  
✅ **Formatage** : noms lisibles pour les rapports  
✅ **Comparaison** : teste tous les modèles en parallèle  
✅ **Extensible** : 5 minutes pour ajouter un nouveau LLM  

Ton projet n'est plus limité à un LLM ! 🚀
