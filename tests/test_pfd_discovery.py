import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
import time
from src.patterns.pfd_discovery import discover_pfds

def test_full_discovery():
    # 1. Configuration des chemins et dossiers
    data_path = "data/pfd_validation/t2.csv"
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("=" * 60)
    print(f"LANCEMENT DE LA DÉCOUVERTE SUR : {data_path}")
    print("=" * 60)

    # 2. Chargement des données
    if not os.path.exists(data_path):
        print(f"❌ Erreur : Le fichier {data_path} est introuvable.")
        return

    df = pd.read_csv(data_path)
    
    # 3. Exécution de la découverte (Approche Classique)
    # On récupère la liste des PFDs et le dictionnaire de stats
    found_pfds, stats = discover_pfds(df, min_support=10, min_confidence=0.85)

    # 4. Affichage des règles (en premier)
    print("\n--- LISTE DES PFDs DÉCOUVERTES ---")
    if found_pfds:
        # Tri par confiance décroissante
        found_pfds.sort(key=lambda x: x['confidence'], reverse=True)
        
        for pfd in found_pfds:
            print(f"Rule: {pfd['lhs']} -> {pfd['rhs']} | Conf: {pfd['confidence']} | Supp: {pfd['support']}")
    else:
        print("Aucune PFD trouvée.")

    # 5. Affichage des STATS (en dernier pour visibilité immédiate)
    print("\n" + "=" * 60)
    print("📈 MÉTRIQUES DE PERFORMANCE (BASELINE CLASSIQUE)")
    print(f"  - Temps d'exécution : {stats['execution_time_seconds']} secondes")
    print(f"  - Candidats explorés : {stats['total_candidates_tested']}")
    print("=" * 60)

    # 6. Sauvegarde JSON complète (Data + Metadata)
    output_file = os.path.join(results_dir, "discovery_results.json")
    
    output_data = {
        "metadata": {
            "dataset": data_path,
            "params": {"min_support": 10, "min_confidence": 0.85},
            "performance": stats
        },
        "results_count": len(found_pfds),
        "pfds": found_pfds
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Rapport détaillé généré : {output_file}")

if __name__ == "__main__":
    test_full_discovery()