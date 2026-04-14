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
        size = len(group)
        support += size
        
        # Mode = valeur la plus fréquente dans le groupe
        mode_val = group.mode()[0]
        mode_count = (group == mode_val).sum()
        
        consistent_count += mode_count
        violation_count = size - mode_count
        
        group_details.append({
            "lhs_value": lhs_val,
            "size": size,
            "dominant_rhs": mode_val,
            "confidence": mode_count / size,
            "violations": violation_count
        })
        
        if violation_count > 0:
            for idx, val in group.items():
                if val != mode_val:
                    violations.append({
                        "index": idx,
                        "lhs": lhs_val,
                        "rhs_found": val,
                        "rhs_expected": mode_val
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