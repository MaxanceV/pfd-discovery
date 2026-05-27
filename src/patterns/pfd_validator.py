"""
Validation d'une PFD approximative.
Métriques du cours (slides 7-8 du Approximate_PFDs.pdf) :
  support(X) = nb tuples satisfaisant le pattern sur X
  conf(X → Y) = proportion de tuples cohérents sur Y dans chaque groupe
"""

import pandas as pd


def compute_support_confidence(df: pd.DataFrame,
                                lhs_col: str,
                                rhs_col: str,
                                detailed: bool = False) -> dict:
    """
    Calcule support et confidence pour la règle lhs_col → rhs_col.

    Args:
        detailed : si True, ajoute 'violations' (liste détaillée des tuples en
                   désaccord) et 'groups' (détail par groupe LHS). Coûteux —
                   à activer uniquement pour le debug, pas dans le hot path
                   de la découverte brute-force.

    Returns:
        dict avec 'lhs', 'rhs', 'support', 'confidence' (toujours)
        + 'violations', 'groups' si detailed=True
    """
    # Choix RHS : on exclut les valeurs nulles du RHS avant de calculer
    # support et confidence. La dépendance est donc mesurée uniquement sur
    # les tuples où les deux côtés sont connus (tuples complets).
    # Conséquence connue : si le RHS a beaucoup de nulls, la confidence est
    # artificiellement gonflée car le dénominateur est réduit. Ce biais est
    # accepté — c'est la pratique courante dans les outils de data quality
    # ("quand on a les deux valeurs, sont-elles cohérentes ?").
    groups = df.groupby(lhs_col)[rhs_col]

    support = 0
    consistent_count = 0
    violations = [] if detailed else None
    group_details = [] if detailed else None

    for lhs_val, group in groups:
        group = group.dropna()
        if group.empty:
            continue

        size = int(len(group))
        support += size

        modes = group.mode()
        if modes.empty:
            continue

        mode_val = modes[0]
        mode_count = int((group == mode_val).sum())
        consistent_count += mode_count

        if not detailed:
            continue

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
                        "index": int(idx),
                        "lhs": str(lhs_val),
                        "rhs_found": str(val),
                        "rhs_expected": str(mode_val)
                    })

    confidence = consistent_count / support if support > 0 else 0

    result = {
        "lhs": lhs_col,
        "rhs": rhs_col,
        "support": support,
        "confidence": round(confidence, 4),
    }
    if detailed:
        result["violations"] = violations
        result["groups"] = group_details
    return result
