"""
Comparaison des différents LLM providers sur la découverte de PFDs.

Teste les mêmes workflows avec différents modèles et compare :
  - Qualité des résultats
  - Temps d'exécution
  - Pertinence des patterns découverts
  - Capacité à identifier les patterns métier pertinents
"""

import pandas as pd
import time
from typing import Dict, List, Any
from src.agent.llm_provider import LLMFactory, LLMProvider
from src.agent.semantic_profiler import get_profile_summary, get_optimized_config
from src.agent.candidate_selector import select_best_candidates
from src.patterns.pfd_discovery import discover_pfds
from src.patterns.extractor import enrich_dataframe_multi


def _run_with_llm(provider: LLMProvider, func, *args, **kwargs) -> Dict[str, Any]:
    """Exécute func(*args, **kwargs) en mesurant le temps et capturant les erreurs."""
    start = time.time()
    try:
        result = func(*args, **kwargs)
        return {
            "provider": provider.provider_name,
            "model": provider.model_name,
            "execution_time": round(time.time() - start, 2),
            "result": result,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "provider": provider.provider_name,
            "model": provider.model_name,
            "execution_time": round(time.time() - start, 2),
            "result": None,
            "success": False,
            "error": str(e),
        }


def profile_with_llm(df: pd.DataFrame,
                     llm_provider: LLMProvider) -> Dict[str, Any]:
    """
    Profile une DataFrame avec un LLM spécifique.

    Returns:
        {
          "provider": "provider_name",
          "model": "model_name",
          "execution_time": float,
          "profile": {...},
          "success": bool,
          "error": str or None
        }
    """
    run = _run_with_llm(llm_provider, get_profile_summary, df, llm_provider=llm_provider)
    return {**run, "profile": run["result"]}


def select_candidates_with_llm(candidates: List[Dict],
                                llm_provider: LLMProvider,
                                df_metadata: Dict = None,
                                top_k: int = 10) -> Dict[str, Any]:
    """
    Sélectionne les meilleures candidates avec un LLM spécifique.

    Returns:
        {
          "provider": "provider_name",
          "model": "model_name",
          "execution_time": float,
          "selected_count": int,
          "confidence_score": float,
          "result": {...},
          "success": bool,
          "error": str or None
        }
    """
    run = _run_with_llm(
        llm_provider, select_best_candidates,
        candidates, df_metadata=df_metadata, top_k=top_k, llm_provider=llm_provider
    )
    result = run["result"] or {}
    return {
        **run,
        "selected_count": len(result.get("selected_candidates", [])),
        "confidence_score": result.get("confidence_score", 0),
    }


def compare_llm_profiles(df: pd.DataFrame, 
                        providers_list: List[str] = None) -> Dict[str, Any]:
    """
    Teste le profiling sémantique avec plusieurs LLMs et compare.
    
    Args:
        df               : DataFrame à analyser
        providers_list   : liste de providers à tester (ex: ["claude", "openai", "gemini"])
                          Si None, teste tous les providers disponibles
    
    Returns:
        {
          "dataset": "dataset_info",
          "profiles": {
            "provider_name": {
              "success": bool,
              "execution_time": float,
              "column_types": {...},
              "reasoning": str,
              "error": str or None
            }
          },
          "comparison": {
            "fastest": "provider_name",
            "slowest": "provider_name",
            "successful": int,
            "failed": int
          }
        }
    """
    
    if providers_list is None:
        providers_list = ["claude", "openai", "gemini"]
    
    print(f"\n{'='*70}")
    print(f"🤖 COMPARAISON DES PROFILS SÉMANTIQUES (LLM Profiling)")
    print(f"{'='*70}")
    print(f"Dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"Providers à tester : {', '.join(providers_list).upper()}\n")
    
    results = {}
    times = {}
    
    for provider_name in providers_list:
        print(f"  ⏱️  Profiling avec {provider_name.upper()}...", end=" ", flush=True)
        
        try:
            provider = LLMFactory.create(provider_name)
            result = profile_with_llm(df, provider)
            results[provider_name] = result
            times[provider_name] = result["execution_time"]
            
            if result["success"]:
                print(f"✅ {result['execution_time']}s")
            else:
                print(f"❌ Erreur : {result['error']}")
        except Exception as e:
            results[provider_name] = {
                "provider": provider_name,
                "success": False,
                "error": str(e),
                "execution_time": 0
            }
            print(f"❌ {str(e)}")
    
    # Analyse comparative
    successful = sum(1 for r in results.values() if r.get("success"))
    failed = sum(1 for r in results.values() if not r.get("success"))
    fastest = min((name for name, t in times.items() if times[name] > 0), 
                  key=lambda x: times[x], default=None)
    slowest = max((name for name, t in times.items() if times[name] > 0), 
                  key=lambda x: times[x], default=None)
    
    return {
        "dataset": {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns)
        },
        "profiles": results,
        "comparison": {
            "successful": successful,
            "failed": failed,
            "fastest": fastest,
            "slowest": slowest,
            "times": times
        }
    }


def compare_llm_candidate_selection(candidates: List[Dict],
                                   df_metadata: Dict = None,
                                   providers_list: List[str] = None,
                                   top_k: int = 10) -> Dict[str, Any]:
    """
    Teste la sélection de candidates avec plusieurs LLMs et compare.
    
    Args:
        candidates      : liste de candidates à évaluer
        df_metadata     : contexte métier optionnel
        providers_list  : providers à tester
        top_k           : nombre de top candidates
    
    Returns:
        {
          "candidates_count": int,
          "selections": { ... },
          "comparison": { ... }
        }
    """
    
    if providers_list is None:
        providers_list = ["claude", "openai", "gemini"]
    
    print(f"\n{'='*70}")
    print(f"🤖 COMPARAISON DE LA SÉLECTION DE CANDIDATES (LLM Ranking)")
    print(f"{'='*70}")
    print(f"Candidates à évaluer : {len(candidates)}")
    print(f"Providers à tester : {', '.join(providers_list).upper()}\n")
    
    results = {}
    times = {}
    selected_counts = {}
    
    for provider_name in providers_list:
        print(f"  ⏱️  Évaluation avec {provider_name.upper()}...", end=" ", flush=True)
        
        try:
            provider = LLMFactory.create(provider_name)
            result = select_candidates_with_llm(
                candidates,
                llm_provider=provider,
                df_metadata=df_metadata,
                top_k=top_k
            )
            results[provider_name] = result
            times[provider_name] = result["execution_time"]
            selected_counts[provider_name] = result["selected_count"]
            
            if result["success"]:
                print(f"✅ {result['execution_time']}s ({result['selected_count']} sélectionnées)")
            else:
                print(f"❌ Erreur : {result['error']}")
        except Exception as e:
            results[provider_name] = {
                "provider": provider_name,
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "selected_count": 0
            }
            print(f"❌ {str(e)}")
    
    # Analyse comparative
    successful = sum(1 for r in results.values() if r.get("success"))
    failed = sum(1 for r in results.values() if not r.get("success"))
    fastest = min((name for name, t in times.items() if times[name] > 0), 
                  key=lambda x: times[x], default=None)
    slowest = max((name for name, t in times.items() if times[name] > 0), 
                  key=lambda x: times[x], default=None)
    
    return {
        "candidates_count": len(candidates),
        "selections": results,
        "comparison": {
            "successful": successful,
            "failed": failed,
            "fastest": fastest,
            "slowest": slowest,
            "times": times,
            "selected_counts": selected_counts
        }
    }


def full_comparison(df: pd.DataFrame, 
                   providers_list: List[str] = None,
                   min_support: int = 10,
                   min_confidence: float = 0.85) -> Dict[str, Any]:
    """
    Comparaison complète : profiling + découverte + sélection avec tous les providers.
    
    Returns:
        {
          "profiling_results": {...},
          "discovery_results": {...},
          "selection_results": {...},
          "overall_comparison": {...}
        }
    """
    
    if providers_list is None:
        providers_list = ["claude", "openai", "gemini"]
    
    print(f"\n{'='*70}")
    print(f"🚀 COMPARAISON COMPLÈTE DES LLM PROVIDERS")
    print(f"{'='*70}")
    
    # 1. Profiling sémantique
    profiling = compare_llm_profiles(df, providers_list)
    
    # 2. Découverte classique (benchmark)
    print(f"\n{'='*70}")
    print(f"📊 DÉCOUVERTE CLASSIQUE (Benchmark - pas de LLM)")
    print(f"{'='*70}")
    print("  ⏱️  Exécution de la découverte classique...", end=" ", flush=True)
    
    start = time.time()
    discovered_pfds, stats = discover_pfds(df, min_support=min_support, min_confidence=min_confidence)
    classic_time = time.time() - start
    
    print(f"✅ {classic_time:.2f}s ({len(discovered_pfds)} PFDs trouvées)")
    
    # 3. Sélection de candidates
    selection = compare_llm_candidate_selection(
        discovered_pfds,
        providers_list=providers_list,
        top_k=10
    )
    
    # Résumé comparatif
    print(f"\n{'='*70}")
    print(f"📊 RÉSUMÉ COMPARATIF")
    print(f"{'='*70}\n")
    
    print(f"{'Provider':<15} {'Profiling':<15} {'Sélection':<15} {'Qualité':<15}")
    print("-" * 60)
    
    for provider_name in providers_list:
        prof_time = profiling["profiles"].get(provider_name, {}).get("execution_time", "-")
        prof_success = "✅" if profiling["profiles"].get(provider_name, {}).get("success") else "❌"
        
        sel_info = selection["selections"].get(provider_name, {})
        sel_time = sel_info.get("execution_time", "-")
        sel_success = "✅" if sel_info.get("success") else "❌"
        sel_count = sel_info.get("selected_count", 0)
        
        quality = f"{sel_info.get('confidence_score', 0):.2%}" if sel_success else "N/A"
        
        prof_display = f"{prof_success} {prof_time}s" if prof_success == "✅" else f"{prof_success}"
        sel_display = f"{sel_success} {sel_time}s" if sel_success == "✅" else f"{sel_success}"
        
        print(f"{provider_name.upper():<15} {prof_display:<15} {sel_display:<15} {quality:<15}")
    
    print(f"\nBenchmark (classique, sans LLM) : {classic_time:.2f}s\n")
    
    return {
        "profiling_results": profiling,
        "discovery_results": {
            "pfds_found": len(discovered_pfds),
            "execution_time": round(classic_time, 2)
        },
        "selection_results": selection,
        "providers_tested": providers_list
    }


