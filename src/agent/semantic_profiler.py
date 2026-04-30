"""
Profiler sémantique utilisant un LLM pour analyser et suggérer les meilleures
transformations de patterns pour chaque colonne.

Rôle : 
  1. Analyser les types sémantiques des colonnes (email, ZIP, nom, etc.)
  2. Suggérer les transformations pertinentes pour chaque colonne
  3. Proposer les colonnes cibles les plus prometteuses
  
Sortie : configuration optimisée de transformations à appliquer.

Support multi-LLM : Claude, OpenAI, Gemini, Ollama
"""

import pandas as pd
import json
from src.agent.llm_provider import LLMFactory, LLMProvider, get_default_provider


def analyze_column_sample(df: pd.DataFrame, col: str, sample_size: int = 5) -> str:
    """
    Crée un résumé d'une colonne pour l'analyse LLM.
    Montre des exemples, le type, la cardinalité, etc.
    """
    sample_values = df[col].dropna().unique()[:sample_size]
    sample_str = ", ".join(str(v) for v in sample_values)
    
    unique_count = df[col].nunique()
    null_count = df[col].isna().sum()
    
    return f"""
Column: {col}
  Samples: [{sample_str}]
  Unique values: {unique_count}
  Nulls: {null_count}
  Type: {df[col].dtype}
"""


def semantic_profile(df: pd.DataFrame, llm_provider: LLMProvider = None) -> dict:
    """
    Analyse les colonnes d'un DataFrame avec un LLM et suggère les 
    meilleures transformations pour chaque colonne.
    
    Args:
        df           : DataFrame à analyser
        llm_provider : LLMProvider à utiliser (défaut : premier disponible)
    
    Returns:
        dict avec structure :
        {
          "column_types": { col_name: "semantic_type" },
          "transformation_recommendations": { col_name: ["transform1", "transform2", ...] },
          "promising_rhs_targets": [list of target columns],
          "reasoning": "explanation from the LLM",
          "llm_provider": "name of the provider used"
        }
    """
    
    # Utiliser le provider par défaut si non fourni
    if llm_provider is None:
        llm_provider = get_default_provider()
    
    # Construire le contexte du DataFrame
    columns_summary = "\n".join(
        analyze_column_sample(df, col) for col in df.columns
    )
    
    # Transformations disponibles
    available_transforms = [
        "raw", "prefix_1", "prefix_2", "prefix_3", "prefix_4", "prefix_5",
        "suffix_2", "suffix_3", "suffix_4",
        "first_token", "last_token", "domain",
        "uppercase", "lowercase"
    ]
    
    prompt = f"""Tu es un expert en qualité des données spécialisé dans les Pattern Functional Dependencies (PFDs).

Voici un échantillon de dataset :
{columns_summary}

Tâche : Analyse ces colonnes et :
1. Identifie le TYPE SÉMANTIQUE de chaque colonne (ex: email, code_postal, nom, identifiant, etc.)
2. Pour CHAQUE colonne, suggère les meilleures TRANSFORMATIONS parmi :
   {', '.join(available_transforms)}
3. Identifie les colonnes cibles (RHS) les plus PROMETTEUSES pour des PFDs

Considérations :
- Pour les codes postaux → prefix_3, prefix_5, raw
- Pour les noms → first_token, last_token, raw
- Pour les emails → domain, raw
- Pour les identifiants → raw uniquement (pas d'autres patterns utiles)
- Pour les states/pays → raw, uppercase
- Pour les numéros de téléphone → prefix_3, prefix_4, suffix_2

Réponds en JSON strict (pas de markdown, pas d'explication avant) :
{{
  "column_types": {{"column_name": "semantic_type", ...}},
  "transformation_recommendations": {{"column_name": ["transform1", "transform2", ...], ...}},
  "promising_rhs_targets": ["col1", "col2", ...],
  "reasoning": "brief explanation"
}}
"""
    
    # Appel au LLM provider
    response_text = llm_provider.call(prompt, max_tokens=2000)
    
    # Nettoyer la réponse si elle contient du markdown
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()
    elif response_text.startswith("```"):
        response_text = response_text.replace("```", "").strip()
    
    # Parser le JSON
    result = json.loads(response_text)
    result["llm_provider"] = llm_provider.provider_name
    result["llm_model"] = llm_provider.model_name
    
    return result


def get_optimized_config(df: pd.DataFrame, llm_provider: LLMProvider = None) -> dict:
    """
    Wrapper qui retourne directement la configuration optimisée
    pour enrich_dataframe_multi().
    
    Args:
        df           : DataFrame à analyser
        llm_provider : LLMProvider optionnel
    
    Returns:
        dict { column_name: [list of transformations] }
    """
    profile = semantic_profile(df, llm_provider=llm_provider)
    
    # Extraire les recommandations
    config = profile.get("transformation_recommendations", {})
    
    # Valider que chaque colonne du DF est dans la config
    for col in df.columns:
        if col not in config:
            config[col] = ["raw"]
    
    return config


def get_profile_summary(df: pd.DataFrame, llm_provider: LLMProvider = None) -> dict:
    """
    Retourne le profil complet (types + recommendations + targets).
    Utile pour le debugging et l'inspection.
    
    Args:
        df           : DataFrame à analyser
        llm_provider : LLMProvider optionnel
    """
    return semantic_profile(df, llm_provider=llm_provider)
