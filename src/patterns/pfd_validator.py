"""
Validation d'une PFD approximative.
Métriques du cours (slides 7-8 du Approximate_PFDs.pdf) :
  support(X) = nb tuples satisfaisant le pattern sur X
  conf(X → Y) = proportion de tuples cohérents sur Y dans chaque groupe
"""

import pandas as pd


def compute_support_confidence(df: pd.DataFrame, 
                                lhs_col: str, 
                                rhs_col: str) -> dict:
    """
    Calcule support et confidence pour la règle lhs_col → rhs_col.
    
    Returns:
        dict avec 'support', 'confidence', 'violations', 'groups'
    """
    # Grouper par lhs
    groups = df.groupby(lhs_col)[rhs_col]
    
    support = 0
    consistent_count = 0
    violations = []
    group_details = []
    
    for lhs_val, group in groups:
        # On ignore les groupes vides ou composés uniquement de NaN
        group = group.dropna()
        if group.empty:
            continue
            
        size = int(len(group)) # Conversion explicite
        support += size
        
        modes = group.mode()
        if modes.empty:
            continue
            
        mode_val = modes[0]
        # Conversion du résultat de .sum() en int Python standard
        mode_count = int((group == mode_val).sum())
        
        consistent_count += mode_count
        violation_count = int(size - mode_count)
        
        group_details.append({
            "lhs_value": str(lhs_val),
            "size": size,
            "dominant_rhs": str(mode_val),
            "confidence": float(round(mode_count / size, 4)),
            "violations": violation_count
        })
        
        if violation_count > 0:
            for idx, val in group.items():
                if val != mode_val:
                    violations.append({
                        "index": int(idx), # Crucial : l'index est souvent un int64
                        "lhs": str(lhs_val),
                        "rhs_found": str(val),
                        "rhs_expected": str(mode_val)
                    })
    
    confidence = consistent_count / support if support > 0 else 0
    
    return {
        "lhs": lhs_col,
        "rhs": rhs_col,
        "support": support,
        "confidence": round(confidence, 4),
        "violations": violations,
        "groups": group_details
    }