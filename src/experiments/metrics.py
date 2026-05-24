"""
Métriques d'évaluation pour la découverte de PFDs.

Ce module est entièrement générique : il ne connaît aucun dataset, aucun nom
de colonne, aucun fichier CSV.

Fonctions exposées :
    extract_basic_metrics(result)    — métriques techniques d'un workflow
    aggregate_runs(runs)             — agrège plusieurs runs (moyenne, écart-type)
    build_comparison_table(results)  — tableau comparatif (dataset × workflow × llm)

Fonctions pour le notebook de légitimité (utilisées en post-traitement) :
    load_ground_truth(path)                  — charge le fichier JSON de vérité terrain
    build_validated_gt(df, hypotheses, ...)  — valide empiriquement les hypothèses GT
    classify_pfd_heuristic(pfd, df)          — classe une PFD comme meaningful/spurious
    compute_recall(discovered, gt)           — rappel vs vérité terrain
    compute_precision_approx(discovered, df) — précision approximative par heuristique
"""

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.patterns.extractor import enrich_dataframe
from src.patterns.pfd_validator import compute_support_confidence


# ─────────────────────────────────────────────────────────────────────────────
# Chargement de la vérité terrain
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(path: str) -> Dict[str, List[Dict]]:
    """
    Charge le fichier data/ground_truth.json et retourne un dict
    { nom_fichier_csv -> [liste d'hypothèses] }.

    Chaque hypothèse est un dict avec les clés :
        lhs_col, transform, rhs_col, classification, justification

    La clé "_comment" est ignorée automatiquement.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Supprimer la clé de documentation si présente
    raw.pop("_comment", None)
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Validation empirique de la vérité terrain
# ─────────────────────────────────────────────────────────────────────────────

def build_validated_gt(df, hypotheses, gt_min_confidence=0.85, gt_min_support=50):
    """
    Vérifie empiriquement chaque hypothèse de ground_truth.json sur le DataFrame.

    Une hypothèse est validée si sa confidence et son support mesurés sur les
    données réelles dépassent les seuils gt_min_*. Seules les hypothèses validées
    servent au calcul du rappel.

    Returns:
        (validated, log)
          validated : liste de dicts avec 'lhs' au format "col__transform"
          log       : toutes les hypothèses + leur résultat empirique
    """
    effective_min_support = min(gt_min_support, 10) if len(df) < 100 else gt_min_support

    validated = []
    log = []

    for hyp in hypotheses:
        lhs_col      = hyp["lhs_col"]
        transform    = hyp["transform"]
        rhs_col      = hyp["rhs_col"]
        classif      = hyp.get("classification", "meaningful")
        lhs_enriched = lhs_col + "__" + transform

        if lhs_col not in df.columns or rhs_col not in df.columns:
            log.append({**hyp, "lhs": lhs_enriched, "rhs": rhs_col,
                        "confidence": None, "support": None,
                        "kept": False, "error": "colonne absente du dataset"})
            continue

        try:
            df_e    = enrich_dataframe(df, lhs_col, [transform])
            metrics = compute_support_confidence(df_e, lhs_enriched, rhs_col)
            conf    = metrics["confidence"]
            supp    = metrics["support"]
        except Exception as exc:
            log.append({**hyp, "lhs": lhs_enriched, "rhs": rhs_col,
                        "confidence": None, "support": None,
                        "kept": False, "error": str(exc)})
            continue

        kept = (conf >= gt_min_confidence and supp >= effective_min_support)

        log.append({**hyp, "lhs": lhs_enriched, "rhs": rhs_col,
                    "confidence": conf, "support": supp, "kept": kept, "error": ""})

        if kept:
            validated.append({
                "lhs":            lhs_enriched,
                "rhs":            rhs_col,
                "classification": classif,
                "justification":  hyp.get("justification", ""),
                "confidence":     conf,
                "support":        supp,
            })

    return validated, log


# ─────────────────────────────────────────────────────────────────────────────
# Classification heuristique meaningful / spurious
# ─────────────────────────────────────────────────────────────────────────────
#
# Sert à estimer la précision quand on n'a pas de vérité terrain pour une PFD :
# on regarde si la règle a l'air triviale ou non.
#
# Trois critères de trivialité (dans l'ordre) :
#   1. Transformation purement orthographique (uppercase/lowercase) → spurious
#   2. LHS quasi-unique en mode raw → c'est un identifiant, pas une règle → spurious
#   3. Redondance lexicale entre nom de colonne LHS et RHS → spurious
# Sinon → meaningful (à confirmer manuellement si besoin)

def classify_pfd_heuristic(pfd: Dict, df: pd.DataFrame) -> Tuple[str, str]:
    """
    Classifie une PFD découverte comme 'meaningful' ou 'spurious'.

    Args:
        pfd : dict avec au moins les clés 'lhs' (ex: "ZIP__prefix_3") et 'rhs'
        df  : DataFrame source, utilisé pour mesurer la cardinalité du LHS

    Returns:
        (classification, justification)  — classification ∈ {"meaningful", "spurious"}
    """
    lhs = pfd.get("lhs", "")
    rhs = pfd.get("rhs", "")

    # Le LHS peut être "colonne__transformation" ou juste "colonne"
    if "__" in lhs:
        lhs_col, transform = lhs.split("__", 1)
    else:
        lhs_col, transform = lhs, "raw"

    # Critère 1 : transformation orthographique, aucun contenu sémantique
    if transform in ("uppercase", "lowercase"):
        return "spurious", "Transformation purement orthographique"

    # Critère 2 : colonne quasi-unique = identifiant, pas une règle métier
    if transform == "raw" and lhs_col in df.columns:
        unique_ratio = df[lhs_col].nunique() / max(len(df), 1)
        if unique_ratio > 0.9:
            return "spurious", f"LHS quasi-unique ({unique_ratio:.0%}) — probablement un identifiant"

    # Critère 3 : redondance lexicale (ex: "Department" → "Department Name")
    def normalize(s: str) -> str:
        return s.lower().replace("_", "").replace(" ", "")

    nl, nr = normalize(lhs_col), normalize(rhs)
    if nl and nr and nl != nr and (nl in nr or nr in nl):
        return "spurious", f"Redondance lexicale entre '{lhs_col}' et '{rhs}'"

    return "meaningful", "Aucun critère de trivialité détecté"


# ─────────────────────────────────────────────────────────────────────────────
# Métriques techniques d'un workflow
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Rappel par rapport à la vérité terrain
# ─────────────────────────────────────────────────────────────────────────────

def _pfd_matches(pfd: Dict, expected_lhs: str, expected_rhs: str) -> bool:
    """Vérifie si une PFD découverte correspond à une hypothèse attendue."""
    return pfd.get("lhs") == expected_lhs and pfd.get("rhs") == expected_rhs


def compute_recall(discovered: List[Dict], validated_gt: List[Dict]) -> Dict[str, Any]:
    """
    Calcule le rappel du workflow par rapport à la vérité terrain validée.

    Deux rappels sont calculés :
        recall             — fraction de toutes les PFDs attendues retrouvées
        recall_meaningful  — fraction des PFDs non triviales retrouvées
                             (métrique principale : révèle la valeur métier)

    Args:
        discovered   : PFDs retournées par le workflow
                       (chacune avec 'lhs', 'rhs', 'confidence', 'support')
        validated_gt : hypothèses de ground_truth.json ayant passé les seuils
                       (chacune avec 'lhs', 'rhs', 'classification')

    Returns:
        dict avec recall, recall_meaningful, et le détail par hypothèse
    """
    found = 0
    found_meaningful = 0
    total_meaningful = sum(1 for gt in validated_gt if gt.get("classification") == "meaningful")
    details = []

    for gt in validated_gt:
        match = next(
            (p for p in discovered if _pfd_matches(p, gt["lhs"], gt["rhs"])),
            None
        )
        is_found = match is not None

        if is_found:
            found += 1
            if gt.get("classification") == "meaningful":
                found_meaningful += 1

        details.append({
            "expected_lhs":      gt["lhs"],
            "expected_rhs":      gt["rhs"],
            "classification":    gt.get("classification", "unknown"),
            "found":             is_found,
            "actual_confidence": match.get("confidence") if match else None,
            "actual_support":    match.get("support")    if match else None,
        })

    return {
        "total_expected":    len(validated_gt),
        "found":             found,
        "recall":            found / len(validated_gt) if validated_gt else 0.0,
        "total_meaningful":  total_meaningful,
        "found_meaningful":  found_meaningful,
        "recall_meaningful": found_meaningful / total_meaningful if total_meaningful else 0.0,
        "details":           details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Précision approximative
# ─────────────────────────────────────────────────────────────────────────────

def compute_precision_approx(discovered: List[Dict], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Estime la précision : proportion des PFDs trouvées classées 'meaningful'
    par l'heuristique (sans LLM, sans vérité terrain).

    Approximative par nature — à compléter manuellement ou via LLM si besoin.

    Returns:
        meaningful_count, spurious_count, precision_approx,
        per_pfd_classification (détail de chaque PFD)
    """
    if not discovered:
        return {
            "meaningful_count": 0,
            "spurious_count": 0,
            "precision_approx": 0.0,
            "per_pfd_classification": [],
        }

    per_pfd = []
    meaningful = 0
    spurious = 0

    for pfd in discovered:
        cls, reason = classify_pfd_heuristic(pfd, df)
        per_pfd.append({
            "lhs":            pfd.get("lhs"),
            "rhs":            pfd.get("rhs"),
            "classification": cls,
            "reason":         reason,
            "confidence":     pfd.get("confidence"),
            "support":        pfd.get("support"),
        })
        if cls == "meaningful":
            meaningful += 1
        else:
            spurious += 1

    return {
        "meaningful_count":       meaningful,
        "spurious_count":         spurious,
        "precision_approx":       meaningful / len(discovered),
        "per_pfd_classification": per_pfd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agrégation sur plusieurs runs
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Tableau comparatif (pour le rapport)
# ─────────────────────────────────────────────────────────────────────────────

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
