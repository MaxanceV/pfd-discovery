"""
Métriques d'évaluation pour la découverte de PFDs.

Fonctions exposées :
    extract_basic_metrics(result)    — métriques techniques d'un workflow
    aggregate_runs(runs)             — agrège plusieurs runs (moyenne, écart-type)
    build_comparison_table(results)  — tableau comparatif (dataset × workflow × llm)
"""

import statistics
from typing import Any, Dict, List, Tuple

import pandas as pd


# ---
# Métriques techniques d'un workflow
# ---

def extract_basic_metrics(workflow_result: Dict) -> Dict[str, Any]:
    """
    Extrait les métriques techniques du dict retourné par un workflow.

    Clés attendues dans workflow_result :
        discovered_pfds         — liste de PFDs (chacune avec 'support' et 'confidence')
        execution_time_seconds  — durée d'exécution
        total_candidates_tested — nombre de candidats explorés

    Retourne :
        nb_pfds, execution_time, candidates_tested, mean_support, mean_confidence
    """
    pfds = workflow_result.get("discovered_pfds", [])
    supports    = [p.get("support", 0)    for p in pfds]
    confidences = [p.get("confidence", 0) for p in pfds]

    n_tested = int(workflow_result.get("total_candidates_tested", 0) or 0)
    validation_rate = round(len(pfds) / n_tested, 3) if n_tested > 0 else None

    return {
        "nb_pfds":           len(pfds),
        "execution_time":    float(workflow_result.get("execution_time_seconds", 0) or 0),
        "candidates_tested": n_tested,
        "mean_support":      sum(supports)     / len(supports)     if supports     else 0.0,
        "mean_confidence":   sum(confidences)  / len(confidences)  if confidences  else 0.0,
        "validation_rate":   validation_rate,
    }


# ---
# Agrégation sur plusieurs runs
# ---

def aggregate_runs(runs: List[Dict]) -> Dict[str, Any]:
    """
    Agrège plusieurs runs d'une même configuration.

    - Si n_runs == 1 : retourne les métriques à plat, sans suffixes.
    - Si n_runs > 1  : calcule moyenne + écart-type pour les métriques qui
                       varient (qualité), et moyenne simple pour les métriques
                       déterministes (volume, temps).

    Métriques retenues dans le tableau final :
        nb_pfds, candidates_tested    — volume de la découverte
        mean_support, mean_confidence — qualité statistique des PFDs
        execution_time                — coût computationnel
        execution_time_stdev          — variance du temps si n_runs > 1
    """
    if not runs:
        return {}

    flat_keys = ["nb_pfds", "candidates_tested", "mean_support", "mean_confidence", "execution_time", "validation_rate"]

    # Cas n_runs == 1 : valeurs à plat, pas de suffixes
    if len(runs) == 1:
        r = runs[0]
        return {
            "nb_runs":           1,
            "nb_pfds":           r.get("nb_pfds"),
            "candidates_tested": r.get("candidates_tested"),
            "mean_support":      round(r.get("mean_support", 0), 2),
            "mean_confidence":   round(r.get("mean_confidence", 0), 3),
            "execution_time":    round(r.get("execution_time", 0), 2),
            "validation_rate":   r.get("validation_rate"),
        }

    # Cas n_runs > 1 : moyenne pour tout + écart-type pour execution_time
    agg = {"nb_runs": len(runs)}

    for key in flat_keys:
        values = [r[key] for r in runs if r.get(key) is not None]
        agg[key] = round(statistics.mean(values), 3) if values else None

    et_values = [r["execution_time"] for r in runs if r.get("execution_time") is not None]
    if len(et_values) > 1:
        agg["execution_time_stdev"] = round(statistics.stdev(et_values), 3)

    return agg


# ---
# Tableau comparatif
# ---

def build_comparison_table(aggregated_results: Dict[Tuple[str, str, str], Dict]) -> pd.DataFrame:
    """
    Construit le tableau comparatif principal : une ligne par (dataset, workflow, llm).

    Colonnes dans l'ordre :
        dataset, workflow, llm, nb_runs,
        nb_pfds, candidates_tested,
        mean_support, mean_confidence,
        execution_time,
        recall_meaningful, precision_approx
        (+ recall_meaningful_stdev, precision_approx_stdev si n_runs > 1)

    Args:
        aggregated_results : { (dataset, workflow, llm) -> dict agrégé }
                             typiquement produit par aggregate_runs()

    Returns:
        DataFrame trié par dataset / workflow / llm, exportable en CSV
    """
    rows = []
    for (dataset, workflow, llm), agg in aggregated_results.items():
        row = {"dataset": dataset, "workflow": workflow, "llm": llm}
        row.update(agg)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["dataset", "workflow", "llm"]).reset_index(drop=True)

    base_cols = [
        "dataset", "workflow", "llm", "nb_runs",
        "nb_pfds", "candidates_tested",
        "mean_support", "mean_confidence",
        "execution_time", "validation_rate",
    ]
    extra_cols = [c for c in ["execution_time_stdev"] if c in df.columns]
    ordered = [c for c in base_cols + extra_cols if c in df.columns]
    return df[ordered]
