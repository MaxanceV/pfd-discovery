"""
Sélecteur de candidats utilisant un LLM pour filtrer et prioriser les
Pattern Functional Dependencies candidates.

Rôle :
  1. Recevoir une liste de PFD candidates (même de faible qualité)
  2. Utiliser un LLM pour évaluer leur pertinence métier
  3. Classer/filtrer les meilleures candidates
  4. Optimiser la recherche en réduisant le nombre de tests à effectuer

Stratégie :
  - Montre au LLM des candidates avec leur support/confidence
  - Demande une évaluation sémantique et logique métier
  - Retourne un ranking + filtre pour éliminer les aberrations

Support multi-LLM : Claude, OpenAI, Gemini, Ollama
"""

import pandas as pd
import json
from src.agent.llm_provider import LLMFactory, LLMProvider, get_default_provider


def format_candidate_for_llm(candidate_dict: dict) -> str:
    """
    Formate une PFD candidate pour présentation au LLM.
    Input : résultat de compute_support_confidence()
    """
    lhs = candidate_dict.get("lhs", "unknown")
    rhs = candidate_dict.get("rhs", "unknown")
    support = candidate_dict.get("support", 0)
    confidence = candidate_dict.get("confidence", 0)
    violations = len(candidate_dict.get("violations", []))
    
    return f"{lhs} → {rhs} [support={support}, confidence={confidence:.2%}, violations={violations}]"


def select_best_candidates(candidates_list: list, 
                          df_metadata: dict = None,
                          top_k: int = 10,
                          min_confidence: float = 0.80,
                          llm_provider: LLMProvider = None) -> dict:
    """
    Filtre et classe les PFD candidates avec un LLM.
    
    Args:
        candidates_list   : liste de dicts { lhs, rhs, support, confidence, violations, groups }
        df_metadata       : info contextuelle (types sémantiques, etc.) - optionnel
        top_k             : nombre de top candidates à retourner
        min_confidence    : threshold de confidence minimum
        llm_provider      : LLMProvider optionnel
    
    Returns:
        dict avec :
        {
          "selected_candidates": [top_k candidates triés],
          "rejected_candidates": [candidates rejetées + raison],
          "reasoning": "explication du LLM",
          "confidence_score": float (score de confiance global),
          "llm_provider": "name of the provider used"
        }
    """
    
    # Utiliser le provider par défaut si non fourni
    if llm_provider is None:
        llm_provider = get_default_provider()
    
    if not candidates_list:
        return {
            "selected_candidates": [],
            "rejected_candidates": [],
            "reasoning": "Aucune candidate fournie",
            "confidence_score": 0.0,
            "llm_provider": llm_provider.provider_name
        }
    
    if len(candidates_list) > 100:
        candidates_list = rank_and_filter(candidates_list, min_confidence=0.85)[:100]

    # Formater les candidates (le reste du code ne change pas...)
    formatted = "\n".join(
        f"  {i+1}. {format_candidate_for_llm(c)}"
        for i, c in enumerate(candidates_list)
    )
    
    # Contexte métier optionnel
    metadata_str = ""
    if df_metadata:
        metadata_str = f"""
Context métier fourni :
{json.dumps(df_metadata, indent=2, ensure_ascii=False)}

"""
    
    prompt = f"""Tu es un expert en qualité des données et Pattern Functional Dependencies (PFDs).

Tu dois évaluer et classer les PFD candidates suivantes selon leur pertinence MÉTIER et LOGIQUE :

{formatted}

{metadata_str}

Critères d'évaluation :
1. **Support** : nombre de tuples concernés (plus = mieux)
2. **Confidence** : proportion de tuples cohérents (90%+ = bon signal)
3. **Logique métier** : la dépendance a-t-elle du sens ? 
   - prefix(ZIP) → city = ✅ bon sens métier
   - first_token(name) → gender = ⚠️ mauvais sens métier
   - random_col → random_col2 = ❌ pas de sens
4. **Violations** : peu de violations = meilleur signal

Tâche :
1. Évalue chaque candidate (accepte/rejette)
2. Classe les acceptées par pertinence (meilleures d'abord)
3. Retourne TOP {top_k} candidates pertinentes

Réponds en JSON strict (pas de markdown) :
{{
  "selected_candidates": [
    {{
      "lhs": "column__pattern",
      "rhs": "target_column", 
      "score": 0.95,
      "reason": "brief explanation why this is good"
    }},
    ...
  ],
  "rejected_candidates": [
    {{
      "lhs": "column__pattern",
      "rhs": "target_column",
      "reason": "why rejected"
    }},
    ...
  ],
  "reasoning": "overall strategy and main findings"
}}
"""
    
    # Appel au LLM provider
    response_text = llm_provider.call(prompt, max_tokens=3000)
    
    # Nettoyer markdown si présent
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "").replace("```", "").strip()
    elif response_text.startswith("```"):
        response_text = response_text.replace("```", "").strip()
    
    # Parser JSON
    result = json.loads(response_text)
    
    # Calculer un score de confiance global
    selected = result.get("selected_candidates", [])
    confidence_score = sum(c.get("score", 0) for c in selected) / len(selected) if selected else 0.0
    
    return {
        "selected_candidates": selected[:top_k],
        "rejected_candidates": result.get("rejected_candidates", []),
        "reasoning": result.get("reasoning", ""),
        "confidence_score": round(confidence_score, 3),
        "llm_provider": llm_provider.provider_name,
        "llm_model": llm_provider.model_name
    }


def rank_and_filter(candidates_list: list,
                   min_confidence: float = 0.80,
                   min_support: int = 10) -> list:
    """
    Filtre rapidement les candidates sans LLM selon des heuristiques simples.
    Utile comme première passe avant le LLM.
    
    Returns:
        liste filtrée et triée par (confidence DESC, support DESC)
    """
    filtered = [
        c for c in candidates_list
        if c.get("confidence", 0) >= min_confidence and c.get("support", 0) >= min_support
    ]
    
    # Trier par confidence puis support
    sorted_candidates = sorted(
        filtered,
        key=lambda c: (c.get("confidence", 0), c.get("support", 0)),
        reverse=True
    )
    
    return sorted_candidates


def get_top_candidates_for_testing(candidates_list: list,
                                   top_k: int = 5,
                                   use_llm: bool = True,
                                   df_metadata: dict = None,
                                   llm_provider: LLMProvider = None) -> list:
    """
    Interface principale : retourne les top K candidates à tester en priorité.
    
    Args:
        candidates_list : sortie de discover_pfds()
        top_k           : nombre de candidates à retourner
        use_llm         : si False, utilise juste le tri heuristique
        df_metadata     : contexte métier optionnel pour l'LLM
        llm_provider    : LLMProvider optionnel
    
    Returns:
        liste ordonnée des meilleures candidates { lhs, rhs, ... }
    """
    
    if not use_llm:
        # Juste tri heuristique
        ranked = rank_and_filter(candidates_list)
        return ranked[:top_k]
    
    # Avec LLM
    evaluation = select_best_candidates(
        candidates_list,
        df_metadata=df_metadata,
        top_k=top_k,
        llm_provider=llm_provider
    )
    
    # Retourner les selected candidates avec leurs infos originales
    selected = evaluation["selected_candidates"]
    
    # Mapper back aux infos complètes
    result = []
    for sel in selected:
        # Chercher la candidate originale
        for orig in candidates_list:
            if orig["lhs"] == sel["lhs"] and orig["rhs"] == sel["rhs"]:
                result.append({
                    **orig,
                    "llm_score": sel["score"],
                    "llm_reason": sel["reason"]
                })
                break
    
    return result
