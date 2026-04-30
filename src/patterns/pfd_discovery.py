import pandas as pd
from src.patterns.extractor import TRANSFORMATIONS, enrich_dataframe_multi
from src.patterns.pfd_validator import compute_support_confidence
import time

# 🔥 Ajout du paramètre config=None
def discover_pfds(df: pd.DataFrame, 
                  min_support: int = 10, 
                  min_confidence: float = 0.85,
                  config: dict = None):
    """
    Parcourt les transformations possibles pour découvrir des PFDs.
    Si une config est fournie, limite la recherche aux transformations spécifiées.
    """
    start_time = time.time()
    candidates_count = 0
    original_cols = df.columns.tolist()
    
    # 1. Préparation de l'enrichissement
    # 🔥 Si pas de config fournie (Classique), on teste TOUT
    if config is None:
        config = {col: list(TRANSFORMATIONS.keys()) for col in original_cols}
        
    df_enriched = enrich_dataframe_multi(df, config)
    
    pattern_cols = [c for c in df_enriched.columns if "__" in c]
    discovered_pfds = []

    print(f"🔍 Recherche sur {len(pattern_cols)} patterns et {len(original_cols)} cibles...")

    # 2. Boucle de découverte
    for lhs in pattern_cols:
        for rhs in original_cols:
            # On évite de tester une colonne contre elle-même (ex: name__raw -> name)
            if lhs.startswith(rhs):
                continue
            candidates_count += 1
            # Validation
            res = compute_support_confidence(df_enriched, lhs, rhs)
            
            # 3. Filtrage selon les seuils
            if res['support'] >= min_support and res['confidence'] >= min_confidence:
                discovered_pfds.append(res)
                print(f"Trouvé : {lhs} -> {rhs} (conf: {res['confidence']})")

    end_time = time.time()
    
    execution_stats = {
        "execution_time_seconds": round(end_time - start_time, 2),
        "total_candidates_tested": candidates_count
    }
    return discovered_pfds, execution_stats