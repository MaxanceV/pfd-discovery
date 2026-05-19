"""
Orchestrateur des experiences de decouverte de PFDs.

Generique : aucun nom de dataset ni de colonne en dur ici.
Toute la connaissance specifique aux donnees vient de data/ground_truth.json.

Pipeline pour chaque (dataset x workflow x provider) :
  1. Charger le CSV
  2. Valider empiriquement les hypotheses de ground_truth.json
  3. Executer le workflow (classical, agent_v1, agent_v2)
  4. Calculer rappel + precision approximative vs verite terrain
  5. Sauvegarder les resultats

Usage :
    python src/experiments/runner.py
    python src/experiments/runner.py --datasets t2.csv --workflows classical
    python src/experiments/runner.py --workflows agent_v1 --providers claude mistral
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.agent.llm_provider import LLMFactory
from src.agent.workflow import workflow_classical, workflow_agent_v1, workflow_agent_v2
from src.experiments.metrics import (
    load_ground_truth,
    extract_basic_metrics,
    compute_recall,
    compute_precision_approx,
    aggregate_runs,
    build_comparison_table,
)
from src.patterns.pfd_validator import compute_support_confidence
from src.patterns.extractor import enrich_dataframe


# -----------------------------------------------------------------------------
# Configuration globale
# -----------------------------------------------------------------------------

DEFAULTS = {
    "data_dir":          "data/pfd_validation",
    "ground_truth_path": "data/ground_truth.json",
    "results_dir":       "results",
    "min_support":       10,
    "min_confidence":    0.85,
    "gt_min_confidence": 0.85,
    "gt_min_support":    50,   # abaisse a 10 automatiquement si dataset < 100 lignes
}


# -----------------------------------------------------------------------------
# Validation empirique de la verite terrain
# -----------------------------------------------------------------------------

def build_validated_gt(df, hypotheses, gt_min_confidence, gt_min_support):
    """
    Verifie empiriquement chaque hypothese de ground_truth.json sur le DataFrame.

    Une hypothese est validee si sa confidence et son support mesures sur les
    donnees reelles depassent les seuils gt_min_*. Seules les hypotheses validees
    servent au calcul du rappel.

    Returns:
        (validated, log)
          validated : liste de dicts avec 'lhs' au format "col__transform"
          log       : toutes les hypotheses + leur resultat empirique
    """
    effective_min_support = min(gt_min_support, 10) if len(df) < 100 else gt_min_support

    validated = []
    log = []

    for hyp in hypotheses:
        lhs_col   = hyp["lhs_col"]
        transform = hyp["transform"]
        rhs_col   = hyp["rhs_col"]
        classif   = hyp.get("classification", "meaningful")
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
                "lhs":           lhs_enriched,
                "rhs":           rhs_col,
                "classification":classif,
                "justification": hyp.get("justification", ""),
                "confidence":    conf,
                "support":       supp,
            })

    return validated, log


# -----------------------------------------------------------------------------
# Normalisation du format de sortie des workflows
# -----------------------------------------------------------------------------

def normalize_workflow_result(raw):
    """
    Ramene la sortie de n'importe quel workflow au format uniforme attendu
    par extract_basic_metrics et compute_recall.

    Necessaire car workflow_agent_v2 retourne des candidats selectionnes par
    le LLM qui ont 'score' au lieu de 'confidence', et pas forcement 'support'.
    """
    pfds = raw.get("discovered_pfds", [])
    normalized = []
    for p in pfds:
        normalized.append({
            "lhs":        p.get("lhs", ""),
            "rhs":        p.get("rhs", ""),
            "support":    p.get("support",    p.get("llm_score", 0)),
            "confidence": p.get("confidence", p.get("score",     0)),
        })
    return {
        "discovered_pfds":         normalized,
        "execution_time_seconds":  raw.get("execution_time_seconds", 0),
        "total_candidates_tested": raw.get("total_candidates_tested", 0),
    }


# -----------------------------------------------------------------------------
# Execution d'un run unique
# -----------------------------------------------------------------------------

def run_one(df, workflow_name, validated_gt, min_support, min_confidence, llm_provider=None):
    """
    Execute un workflow sur un DataFrame et retourne toutes les metriques.
    """
    if workflow_name == "classical":
        raw = workflow_classical(df, min_support=min_support, min_confidence=min_confidence)
    elif workflow_name == "agent_v1":
        raw = workflow_agent_v1(df, min_support=min_support, min_confidence=min_confidence,
                                llm_provider=llm_provider)
    elif workflow_name == "agent_v2":
        raw = workflow_agent_v2(df, min_support=min_support, min_confidence=min_confidence,
                                llm_provider=llm_provider)
    else:
        raise ValueError("Workflow inconnu : " + workflow_name +
                         ". Valeurs valides : classical, agent_v1, agent_v2")

    result   = normalize_workflow_result(raw)
    basic    = extract_basic_metrics(result)
    recall_m = compute_recall(result["discovered_pfds"], validated_gt)
    prec_m   = compute_precision_approx(result["discovered_pfds"], df)

    return {
        **basic,
        "recall_meaningful": recall_m["recall_meaningful"],
        "recall_details":    recall_m["details"],   # détail par hypothèse, pour les JSON individuels
        "precision_approx":  prec_m["precision_approx"],
        "discovered_pfds":   result["discovered_pfds"],
    }


# -----------------------------------------------------------------------------
# Boucle principale
# -----------------------------------------------------------------------------

def run_experiments(datasets, workflows, providers, n_runs=1,
                    min_support=DEFAULTS["min_support"],
                    min_confidence=DEFAULTS["min_confidence"],
                    data_dir=DEFAULTS["data_dir"],
                    ground_truth_path=DEFAULTS["ground_truth_path"],
                    results_dir=DEFAULTS["results_dir"]):
    """
    Lance toutes les combinaisons (dataset x workflow x provider).

    Pour chaque combinaison :
      - Execute n_runs runs (pour mesurer la variance des workflows agentiques)
      - Agregee et sauvegarde les resultats
      - Construit le tableau comparatif final en CSV
    """
    os.makedirs(results_dir, exist_ok=True)

    # La verite terrain est chargee une seule fois — generique, pas de noms de colonnes ici
    all_gt = load_ground_truth(ground_truth_path)

    aggregated_results = {}

    for dataset_name in datasets:
        csv_path = os.path.join(data_dir, dataset_name)
        if not os.path.exists(csv_path):
            print("[SKIP] " + dataset_name + " introuvable dans " + data_dir)
            continue

        df = pd.read_csv(csv_path)
        print("\n" + "="*60)
        print("Dataset : " + dataset_name +
              "  (" + str(df.shape[0]) + " lignes x " + str(df.shape[1]) + " cols)")
        print("="*60)

        # Valider empiriquement les hypotheses pour CE dataset
        hypotheses   = all_gt.get(dataset_name, [])
        validated_gt, gt_log = build_validated_gt(
            df, hypotheses,
            gt_min_confidence=DEFAULTS["gt_min_confidence"],
            gt_min_support=DEFAULTS["gt_min_support"],
        )

        print("Verite terrain : " + str(len(hypotheses)) + " hypotheses -> " +
              str(len(validated_gt)) + " validees empiriquement")
        for h in gt_log:
            status   = "OK" if h["kept"] else "KO"
            conf_str = ("conf=" + str(round(h["confidence"], 3))
                        if h["confidence"] is not None else h["error"])
            print("  [" + status + "] " + h["lhs"] + " -> " + h["rhs"] + "  " + conf_str)

        # Sauvegarder le log de validation (utile pour le rapport)
        with open(os.path.join(results_dir, dataset_name + "_gt_validation.json"),
                  "w", encoding="utf-8") as f:
            json.dump(gt_log, f, indent=2, ensure_ascii=False)

        for workflow_name in workflows:
            # Pour "classical", pas de provider LLM
            provider_list = [None] if workflow_name == "classical" else [
                LLMFactory.create(p) for p in providers
            ]

            for provider in provider_list:
                provider_name = provider.provider_name if provider else "none"
                print("\n--- " + dataset_name + " | " + workflow_name + " | " +
                      provider_name + " (" + str(n_runs) + " run(s)) ---")

                runs = []
                for run_idx in range(n_runs):
                    try:
                        m = run_one(df=df, workflow_name=workflow_name,
                                    validated_gt=validated_gt,
                                    min_support=min_support,
                                    min_confidence=min_confidence,
                                    llm_provider=provider)
                        runs.append(m)
                        print("  Run " + str(run_idx+1) + ": " +
                              str(m["nb_pfds"]) + " PFDs | " +
                              "recall=" + str(round(m["recall_meaningful"], 2)) + " | " +
                              "precision~" + str(round(m["precision_approx"], 2)) + " | " +
                              str(round(m["execution_time"], 1)) + "s")
                    except Exception as exc:
                        print("  Run " + str(run_idx+1) + ": ERREUR — " + str(exc))

                if not runs:
                    continue

                agg = aggregate_runs(runs)
                aggregated_results[(dataset_name, workflow_name, provider_name)] = agg

                detail = {
                    "dataset":            dataset_name,
                    "workflow":           workflow_name,
                    "provider":           provider_name,
                    "params":             {"min_support": min_support,
                                          "min_confidence": min_confidence},
                    "gt_validated_count": len(validated_gt),
                    "aggregated":         agg,
                    "runs":               runs,
                }
                fname = dataset_name + "_" + workflow_name + "_" + provider_name + ".json"
                with open(os.path.join(results_dir, fname), "w", encoding="utf-8") as f:
                    json.dump(detail, f, indent=2, ensure_ascii=False, default=str)

    # Tableau comparatif final
    if aggregated_results:
        table = build_comparison_table(aggregated_results)
        table_path = os.path.join(results_dir, "comparison_table.csv")
        table.to_csv(table_path, index=False)
        print("\n" + "="*60)
        print("Tableau comparatif : " + table_path)
        print(table.to_string(index=False))


# -----------------------------------------------------------------------------
# Point d'entree CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Runner d'experiences PFD — generique, pilote par ground_truth.json"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["t1.csv", "t2.csv", "t3.csv", "US_Phone_Code.csv"],
        help="Fichiers CSV (cherches dans data_dir)"
    )
    parser.add_argument(
        "--workflows", nargs="+", default=["classical"],
        choices=["classical", "agent_v1", "agent_v2"],
    )
    parser.add_argument("--providers",       nargs="+", default=[])
    parser.add_argument("--runs",            type=int,   default=1)
    parser.add_argument("--min-support",     type=int,   default=DEFAULTS["min_support"])
    parser.add_argument("--min-confidence",  type=float, default=DEFAULTS["min_confidence"])
    parser.add_argument("--data-dir",        default=DEFAULTS["data_dir"])
    parser.add_argument("--ground-truth",    default=DEFAULTS["ground_truth_path"])
    parser.add_argument("--results-dir",     default=DEFAULTS["results_dir"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Auto-detection des providers si workflows agentiques sans provider explicite
    if any(w != "classical" for w in args.workflows) and not args.providers:
        args.providers = LLMFactory.list_detected_providers()
        if args.providers:
            print("Providers detectes automatiquement : " + str(args.providers))
        else:
            print("Aucun provider LLM detecte — seul le workflow classique sera execute.")
            args.workflows = ["classical"]

    run_experiments(
        datasets=args.datasets,
        workflows=args.workflows,
        providers=args.providers,
        n_runs=args.runs,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        data_dir=args.data_dir,
        ground_truth_path=args.ground_truth,
        results_dir=args.results_dir,
    )
