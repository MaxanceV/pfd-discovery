"""
Démonstration et exemple d'utilisation du système multi-LLM.

Montre comment :
  1. Utiliser différents LLM providers de manière dynamique
  2. Comparer les résultats entre modèles
  3. Exécuter les workflows avec des LLMs spécifiques
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent (racine du projet) au chemin Python
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.agent.llm_provider import LLMFactory
from src.agent.llm_comparison import full_comparison, export_results
from src.agent.workflow import workflow_classical, workflow_agent_v1


def main_demo():
    """Exemple d'utilisation complète."""
    
    # ───────────────────────────────────────────────────────────────────
    # 1. Charger un dataset
    # ───────────────────────────────────────────────────────────────────
    print("\n📂 Chargement du dataset...")
    
    dataset_path = "data/pfd_validation/t2.csv"  # Entreprises (ZIP, city, phone)
    
    if not os.path.exists(dataset_path):
        print(f"⚠️  {dataset_path} non trouvé. Datasets disponibles :")
        print("   - data/pfd_validation/t1.csv (Employés)")
        print("   - data/pfd_validation/t2.csv (Entreprises)")
        print("   - data/pfd_validation/t3.csv (Licences)")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    # ───────────────────────────────────────────────────────────────────
    # 2. Récupérer dynamiquement les providers disponibles
    # ───────────────────────────────────────────────────────────────────
    print("\n🤖 Providers disponibles :")
    LLMFactory.list_providers()
    
    detected_providers = LLMFactory.list_detected_providers()
    
    if not detected_providers:
        print("\n❌ Aucun provider n'est configuré dans le fichier .env.")
        print("👉 Ajoute au moins une clé (ex: GROQ_API_KEY, MISTRAL_API_KEY...) pour continuer.")
        return

    # ───────────────────────────────────────────────────────────────────
    # 3. Exemple 1 : Workflow Classique vs Agentique (1er provider dispo)
    # ───────────────────────────────────────────────────────────────────
    first_provider_name = detected_providers[0]
    
    print("\n" + "="*70)
    print(f"📊 EXEMPLE 1 : Comparaison Classique vs Agentique ({first_provider_name.upper()})")
    print("="*70)
    
    try:
        provider_1 = LLMFactory.create(first_provider_name)
        
        # Workflow classique
        print("\n🔵 Workflow Classique (pas de LLM)...")
        results_classical = workflow_classical(df, min_support=10, min_confidence=0.85)
        
        # Workflow agentique v1 avec le provider détecté
        print(f"\n🟠 Workflow Agentique v1 (avec {first_provider_name.upper()})...")
        results_v1 = workflow_agent_v1(df, min_support=10, min_confidence=0.85, 
                                        llm_provider=provider_1)
        
        # Afficher la comparaison
        print(f"\n📈 Comparaison :")
        print(f"   Classique  : {len(results_classical['discovered_pfds'])} PFDs en {results_classical['execution_time_seconds']}s")
        print(f"   Agent v1   : {len(results_v1['discovered_pfds'])} PFDs en {results_v1['execution_time_seconds']}s")
        
    except Exception as e:
        print(f"❌ Erreur avec {first_provider_name.upper()} : {e}")
    
    # ───────────────────────────────────────────────────────────────────
    # 4. Exemple 2 : Utiliser un second provider (si disponible)
    # ───────────────────────────────────────────────────────────────────
    if len(detected_providers) > 1:
        second_provider_name = detected_providers[1]
        
        print("\n" + "="*70)
        print(f"📊 EXEMPLE 2 : Tester un autre provider ({second_provider_name.upper()})")
        print("="*70)
        
        try:
            provider_2 = LLMFactory.create(second_provider_name)
            
            print(f"\n🟠 Workflow Agentique v1 (avec {second_provider_name.upper()})...")
            results_v2 = workflow_agent_v1(df, min_support=10, min_confidence=0.85,
                                               llm_provider=provider_2)
            
            print(f"\n   Résultat : {len(results_v2['discovered_pfds'])} PFDs en {results_v2['execution_time_seconds']}s")
            
        except Exception as e:
            print(f"❌ Erreur avec {second_provider_name.upper()} : {e}")
    else:
        print("\n" + "="*70)
        print("📊 EXEMPLE 2 : Ignoré (Un seul provider configuré dans le .env)")
        print("="*70)
    
    # ───────────────────────────────────────────────────────────────────
    # 5. Exemple 3 : Comparaison complète de tous les modèles
    # ───────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("📊 EXEMPLE 3 : Comparaison complète des modèles")
    print("="*70)
    
    try:
        print(f"🔍 Lancement des tests sur les providers configurés : {', '.join(detected_providers).upper()}")
        
        results = full_comparison(df, providers_list=detected_providers)
        
        # Exporter les résultats (fusion automatique avec l'historique)
        export_results(results, output_file="results/llm_comparison.json")
        
    except Exception as e:
        print(f"❌ Erreur comparaison : {e}")


if __name__ == "__main__":
    main_demo()