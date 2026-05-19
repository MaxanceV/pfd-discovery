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
from src.patterns.extractor import TRANSFORMATIONS


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
    
    # Transformations disponibles — source unique : extractor.TRANSFORMATIONS
    available_transforms = list(TRANSFORMATIONS.keys())
    
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


def suggest_candidate_pairs(df: pd.DataFrame, llm_provider: LLMProvider = None) -> list:
    """
    Demande au LLM de proposer directement des paires de dépendances candidates.

    Utilisé par workflow_agent_v2 (Guided Search) : le LLM oriente la recherche
    AVANT tout calcul en proposant des paires (colonne source, transformation,
    colonne cible) sémantiquement plausibles. L'algorithme ne valide ensuite
    que ces paires — aucune exploration brute force.

    Args:
        df           : DataFrame à analyser
        llm_provider : LLMProvider à utiliser (défaut : premier disponible)

    Returns:
        Liste de dicts validés, chacun de la forme :
        {"lhs_col": "ZIP", "transform": "prefix_3", "rhs": "CITY"}
        Les paires dont les colonnes ou la transformation sont inconnues sont ignorées.
    """
    if llm_provider is None:
        llm_provider = get_default_provider()

    columns_summary = "\n".join(
        analyze_column_sample(df, col) for col in df.columns
    )
    available_transforms = list(TRANSFORMATIONS.keys())

    prompt = f"""Tu es un expert en qualité des données spécialisé dans les Pattern Functional Dependencies (PFDs).

Voici un dataset avec les colonnes suivantes :
{columns_summary}

Transformations disponibles : {', '.join(available_transforms)}

Tâche : Propose des dépendances fonctionnelles de pattern plausibles pour ce dataset.
Pour chaque dépendance, indique la colonne source, la transformation à appliquer, et la colonne cible.

Règles :
- Propose uniquement des dépendances sémantiquement sensées
- La colonne source et la colonne cible doivent être différentes
- Évite les dépendances triviales (identifiant → autre colonne) ou sans sens métier
- Propose entre 5 et 15 candidats

Exemples de bonnes dépendances :
- {{"lhs_col": "ZIP", "transform": "prefix_3", "rhs": "CITY"}}  → les 3 premiers chiffres du code postal déterminent la ville
- {{"lhs_col": "email", "transform": "domain", "rhs": "organization"}}  → le domaine email détermine l'organisation
- {{"lhs_col": "full_name", "transform": "first_token", "rhs": "gender"}}  → le prénom est corrélé au genre

IMPORTANT : réponds UNIQUEMENT avec le tableau JSON.
Aucun texte avant, aucun texte après, aucune explication.
Commence ta réponse par [ et termine par ]
Format compact, sans retours à la ligne ni indentation :
[{{"lhs_col": "col1", "transform": "t1", "rhs": "col2"}},{{"lhs_col": "col3", "transform": "t2", "rhs": "col4"}}]"""

    response_text = llm_provider.call(prompt, max_tokens=2500)

    # Extraire uniquement le tableau JSON (ignore tout texte avant/après)
    start = response_text.find("[")
    end   = response_text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        print(f"  [Warning] Aucun tableau JSON trouvé ({llm_provider.provider_name}) :")
        print(f"  {response_text[:300]}")
        return []
    try:
        raw_pairs = json.loads(response_text[start:end + 1])
    except json.JSONDecodeError:
        print(f"  [Warning] Réponse LLM non parseable ({llm_provider.provider_name}) :")
        print(f"  {response_text[start:start + 300]}")
        return []

    # Valider que chaque paire référence des colonnes et transformations existantes
    valid_cols = set(df.columns)
    valid_transforms = set(TRANSFORMATIONS.keys())
    validated = []
    for p in raw_pairs:
        if not isinstance(p, dict):
            continue
        lhs_col   = p.get("lhs_col", "")
        transform = p.get("transform", "")
        rhs       = p.get("rhs", "")
        if lhs_col in valid_cols and transform in valid_transforms and rhs in valid_cols and lhs_col != rhs:
            validated.append({"lhs_col": lhs_col, "transform": transform, "rhs": rhs})

    rejected = len(raw_pairs) - len(validated)
    if rejected > 0:
        print(f"  [Warning] {rejected} paires ignorées (colonnes ou transformation inconnues)")

    return validated
