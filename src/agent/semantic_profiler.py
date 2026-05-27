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
import warnings
from src.agent.llm_provider import LLMProvider, get_default_provider
from src.patterns.extractor import TRANSFORMATIONS


# ---
# Normalisation des noms de colonnes
# ---

def _normalize_col(col: str) -> str:
    """
    Normalise un nom de colonne pour le LLM : minuscules, espaces → underscores.
    Exemples : "Full Name" → "full_name", "ADDRESS_2" → "address_2"

    strip() avant replace() : sinon "  city  " devient "__city__" car strip()
    ne retire pas les underscores.
    """
    return col.strip().lower().replace(" ", "_")


def _build_col_mapping(df: pd.DataFrame) -> dict:
    """
    Crée un mapping { nom_normalisé → nom_original }.

    Le LLM reçoit les noms normalisés dans le prompt et répond avec eux.
    Ce mapping permet de retrouver les noms exacts du DataFrame après
    parsing de la réponse, avant validation.

    Exemple : {"full_name": "Full Name", "address_2": "ADDRESS_2"}
    """
    mapping = {_normalize_col(col): col for col in df.columns}
    if len(mapping) < len(df.columns):
        warnings.warn(
            "Deux colonnes ont le même nom normalisé — risque de collision "
            "dans la correspondance LLM → DataFrame."
        )
    return mapping


def analyze_column_sample(df: pd.DataFrame, col: str,
                           sample_size: int = 5,
                           display_name: str = None) -> str:
    """
    Crée un résumé d'une colonne pour l'analyse LLM.
    Montre des exemples, le type, la cardinalité, etc.

    Args:
        display_name : nom à afficher dans le prompt (normalisé).
                       Si None, utilise le nom original de la colonne.
    """
    if display_name is None:
        display_name = col

    sample_values = df[col].dropna().unique()[:sample_size]
    sample_str = ", ".join(str(v) for v in sample_values)

    unique_count = df[col].nunique()
    null_count = df[col].isna().sum()

    return f"""
Column: {display_name}
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
    
    if llm_provider is None:
        llm_provider = get_default_provider()

    # Normalisation : le LLM reçoit des noms sans espaces ni majuscules
    col_mapping = _build_col_mapping(df)
    reverse_mapping = {original: normalized for normalized, original in col_mapping.items()}

    columns_summary = "\n".join(
        analyze_column_sample(df, col, display_name=reverse_mapping[col])
        for col in df.columns
    )

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

IMPORTANT : utilise EXACTEMENT les noms de colonnes tels qu'ils apparaissent
dans le résumé ci-dessus (pas de majuscules, pas d'espaces).

Réponds en JSON strict (pas de markdown, pas d'explication avant) :
{{
  "column_types": {{"column_name": "semantic_type", ...}},
  "transformation_recommendations": {{"column_name": ["transform1", "transform2", ...], ...}},
  "promising_rhs_targets": ["col1", "col2", ...],
  "reasoning": "brief explanation"
}}
"""

    response_text = llm_provider.call(prompt, max_tokens=2000)

    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()
    elif response_text.startswith("```"):
        response_text = response_text.replace("```", "").strip()

    result = json.loads(response_text)

    # Remapper les noms normalisés → noms originaux du DataFrame
    result["column_types"] = {
        col_mapping.get(k, k): v
        for k, v in result.get("column_types", {}).items()
    }
    result["transformation_recommendations"] = {
        col_mapping.get(k, k): v
        for k, v in result.get("transformation_recommendations", {}).items()
    }
    result["promising_rhs_targets"] = [
        col_mapping.get(t, t) for t in result.get("promising_rhs_targets", [])
    ]

    result["llm_provider"] = llm_provider.provider_name
    result["llm_model"]    = llm_provider.model_name

    return result



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

    col_mapping = _build_col_mapping(df)
    reverse_mapping = {original: normalized for normalized, original in col_mapping.items()}

    columns_summary = "\n".join(
        analyze_column_sample(df, col, display_name=reverse_mapping[col])
        for col in df.columns
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
- IMPORTANT : utilise EXACTEMENT les noms de colonnes tels qu'ils apparaissent
  dans le résumé ci-dessus

Format de réponse — UNIQUEMENT le tableau JSON, rien d'autre :
[{{"lhs_col": "nom_colonne_source", "transform": "nom_transformation", "rhs": "nom_colonne_cible"}}]

Commence ta réponse par [ et termine par ]"""

    response_text = llm_provider.call(prompt, max_tokens=2500)

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

    # Remapper noms normalisés → noms originaux, puis valider
    valid_cols      = set(df.columns)
    valid_transforms = set(TRANSFORMATIONS.keys())
    validated = []

    for p in raw_pairs:
        if not isinstance(p, dict):
            continue
        # Remapping : nom normalisé proposé par le LLM → nom original du DataFrame
        lhs_col   = col_mapping.get(p.get("lhs_col", ""), p.get("lhs_col", ""))
        rhs       = col_mapping.get(p.get("rhs", ""),     p.get("rhs", ""))
        transform = p.get("transform", "")

        if lhs_col in valid_cols and transform in valid_transforms and rhs in valid_cols and lhs_col != rhs:
            # domain n'a de sens que si la colonne contient des emails
            if transform == "domain" and not df[lhs_col].dropna().astype(str).str.contains('@').any():
                continue
            validated.append({"lhs_col": lhs_col, "transform": transform, "rhs": rhs})

    rejected = len(raw_pairs) - len(validated)
    if rejected > 0:
        print(f"  [Warning] {rejected} paires ignorées (colonnes ou transformation inconnues)")

    return validated
