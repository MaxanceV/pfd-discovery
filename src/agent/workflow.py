"""
Workflows de découverte de Pattern Functional Dependencies (PFDs).

Trois approches :
  1. **Classique** : algorithmes TANE/FASTFD sans intervention LLM
  2. **Agentique v1** : LLM analyse les colonnes → transformations optimisées → découverte classique
  3. **Agentique v2** : v1 + LLM filtre et priorise les candidates → découverte ciblée
"""

import pandas as pd
import time
import json
from typing import Tuple, List, Dict, Optional

# Imports des modules existants
from src.patterns.pfd_discovery import discover_pfds
from src.patterns.extractor import enrich_dataframe_multi, TRANSFORMATIONS

# Imports des modules agentiques
from src.agent.llm_provider import LLMProvider, get_default_provider
from src.agent.semantic_profiler import get_optimized_config, get_profile_summary, suggest_candidate_pairs


# ─── Workflow 1 : Approche classique ─────────────────────────────────────
def workflow_classical(df: pd.DataFrame, 
                      min_support: int = 10, 
                      min_confidence: float = 0.85) -> Dict:
    print("\n" + "="*60)
    print("🔵 WORKFLOW 1 : APPROCHE CLASSIQUE (Brute-force)")
    print("="*60)
    
    start_time = time.time()
    
    print(f"📊 Dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"🔧 Transformations : {len(TRANSFORMATIONS)} par colonne")
    
    # Découverte classique sans config (force brute)
    discovered_pfds, stats = discover_pfds(df, min_support=min_support, min_confidence=min_confidence)
    
    end_time = time.time()
    
    result = {
        "discovered_pfds": discovered_pfds,
        "execution_time_seconds": round(end_time - start_time, 2),
        "total_candidates_tested": stats["total_candidates_tested"],
        "metadata": {
            "approach": "classical",
            "min_support": min_support,
            "min_confidence": min_confidence
        }
    }
    
    print(f"\n✅ Résultats classiques :")
    print(f"   Découvertes : {len(discovered_pfds)} PFDs")
    print(f"   Temps : {result['execution_time_seconds']}s")
    print(f"   Candidates testées : {result['total_candidates_tested']}")
    
    return result


# ─── Workflow 2 : Agentique v1 ───────────────────────────────────────────
def workflow_agent_v1(df: pd.DataFrame,
                      min_support: int = 10,
                      min_confidence: float = 0.85,
                      llm_provider: Optional[LLMProvider] = None) -> Dict:
    
    if llm_provider is None:
        llm_provider = get_default_provider()
    
    print("\n" + "="*60)
    print(f"🟠 WORKFLOW 2 : AGENTIQUE v1 (LLM: {llm_provider.provider_name.upper()})")
    print("="*60)
    
    start_time = time.time()
    
    print(f"📊 Dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    print(f"🤖 LLM ({llm_provider.provider_name}) analyse les colonnes et suggère les transformations...")
    # Un seul appel API — on extrait la config depuis le même profil
    profile = get_profile_summary(df, llm_provider=llm_provider)

    print(f"📋 Profil sémantique :")
    print(f"   Types identifiés : {list(profile.get('column_types', {}).keys())}")

    # Extraire la config directement depuis le profil déjà obtenu (pas de second appel API)
    config = profile.get("transformation_recommendations", {})
    for col in df.columns:
        if col not in config:
            config[col] = ["raw"]
    
    selected_transforms_count = sum(len(v) for v in config.values())
    all_transforms_count = len(TRANSFORMATIONS) * len(df.columns)
    
    print(f"🔧 Transformations sélectionnées : {selected_transforms_count}/{all_transforms_count} ({100*selected_transforms_count//all_transforms_count}%)")
    
    print(f"📈 Application du filtre LLM à la découverte...")
    # 🔥 L'UNIQUE appel à discover_pfds, AVEC la config
    discovered_pfds, stats = discover_pfds(
        df, 
        min_support=min_support, 
        min_confidence=min_confidence, 
        config=config
    )
    
    end_time = time.time()
    
    result = {
        "discovered_pfds": discovered_pfds,
        "execution_time_seconds": round(end_time - start_time, 2),
        "total_candidates_tested": stats["total_candidates_tested"],
        "llm_optimization": {
            "column_types": profile.get("column_types", {}),
            "recommendations": profile.get("transformation_recommendations", {}),
            "promising_targets": profile.get("promising_rhs_targets", [])
        },
        "metadata": {
            "approach": "agent_v1",
            "llm_provider": llm_provider.provider_name,
            "llm_model": llm_provider.model_name,
            "min_support": min_support,
            "min_confidence": min_confidence,
            "optimization_ratio": f"{100*selected_transforms_count//all_transforms_count}%"
        }
    }
    
    print(f"\n✅ Résultats agentique v1 ({llm_provider.provider_name}) :")
    print(f"   Découvertes : {len(discovered_pfds)} PFDs")
    print(f"   Temps : {result['execution_time_seconds']}s")
    print(f"   Candidates testées : {result['total_candidates_tested']}")
    
    return result


# ─── Workflow 3 : Agentique v2 (Guided Search) ───────────────────────────
def workflow_agent_v2(df: pd.DataFrame,
                      min_support: int = 10,
                      min_confidence: float = 0.85,
                      llm_provider: Optional[LLMProvider] = None) -> Dict:
    """
    Guided Search : le LLM propose des paires candidates AVANT tout calcul,
    l'algorithme valide uniquement celles-là.

    Séparation des rôles :
      - LLM  : boussole sémantique — propose les règles plausibles
      - Algo : validation — calcule support et confidence sur ces règles uniquement
    """
    if llm_provider is None:
        llm_provider = get_default_provider()

    print("\n" + "="*60)
    print(f"🟢 WORKFLOW 2 : GUIDED SEARCH (LLM: {llm_provider.provider_name.upper()})")
    print("="*60)

    start_time = time.time()

    print(f"📊 Dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"🤖 LLM ({llm_provider.provider_name}) propose des dépendances candidates...")

    # Étape 1 — Le LLM propose des paires (lhs_col, transform, rhs) sans aucun calcul
    candidate_pairs = suggest_candidate_pairs(df, llm_provider=llm_provider)

    print(f"   {len(candidate_pairs)} paires proposées par le LLM")

    if not candidate_pairs:
        print("   Aucune paire proposée — arrêt.")
        return {
            "discovered_pfds": [],
            "execution_time_seconds": round(time.time() - start_time, 2),
            "total_candidates_tested": 0,
            "metadata": {
                "approach": "agent_v2",
                "llm_provider": llm_provider.provider_name,
                "llm_model": llm_provider.model_name,
            }
        }

    # Étape 2 — L'algorithme valide uniquement les paires proposées
    print(f"📈 Validation algorithmique des {len(candidate_pairs)} paires...")

    from src.patterns.extractor import enrich_dataframe
    from src.patterns.pfd_validator import compute_support_confidence

    discovered_pfds = []
    for pair in candidate_pairs:
        lhs_col   = pair["lhs_col"]
        transform = pair["transform"]
        rhs       = pair["rhs"]
        lhs_enriched = f"{lhs_col}__{transform}"

        df_enriched = enrich_dataframe(df, lhs_col, [transform])
        metrics = compute_support_confidence(df_enriched, lhs_enriched, rhs)

        if metrics["support"] >= min_support and metrics["confidence"] >= min_confidence:
            discovered_pfds.append({
                "lhs":        lhs_enriched,
                "rhs":        rhs,
                "support":    metrics["support"],
                "confidence": metrics["confidence"],
            })

    end_time = time.time()

    result = {
        "discovered_pfds": discovered_pfds,
        "execution_time_seconds": round(end_time - start_time, 2),
        "total_candidates_tested": len(candidate_pairs),
        "metadata": {
            "approach": "agent_v2",
            "llm_provider": llm_provider.provider_name,
            "llm_model": llm_provider.model_name,
            "min_support": min_support,
            "min_confidence": min_confidence,
        }
    }

    print(f"\n✅ Résultats guided search ({llm_provider.provider_name}) :")
    print(f"   Paires proposées par le LLM : {len(candidate_pairs)}")
    print(f"   PFDs validées : {len(discovered_pfds)}")
    print(f"   Temps : {result['execution_time_seconds']}s")

    return result