"""
Orchestrateur des experiences de decouverte de PFDs.

Generique : aucun nom de dataset ni de colonne en dur ici.

Pipeline pour chaque (dataset x workflow x provider) :
  1. Charger le CSV
  2. Executer le workflow (classical, agent_v1, agent_v2)
  3. Calculer les metriques brutes (support, confidence, temps, nb_pfds)
  4. Sauvegarder les resultats

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
    extract_basic_metrics,
    aggregate_runs,
    build_comparison_table,
)


# -----------------------------------------------------------------------------
# Configuration globale
# -----------------------------------------------------------------------------

DEFAULTS = {
    "data_dir":       "data/pfd_validation",
    "results_dir":    "results",
    "min_support":    10,
    "min_confidence": 0.85,
}


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
        "candidates_proposed":     raw.get("candidates_proposed"),
    }


# -----------------------------------------------------------------------------
# Execution d'un run unique
# -----------------------------------------------------------------------------

def run_one(df, workflow_name, min_support, min_confidence, llm_provider=None):
    """
    Execute un workflow sur un DataFrame et retourne les metriques brutes.
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

    result = normalize_workflow_result(raw)
    basic  = extract_basic_metrics(result)

    return {
        **basic,
        "discovered_pfds":     result["discovered_pfds"],
        "candidates_proposed": result.get("candidates_proposed"),
    }


# -----------------------------------------------------------------------------
# Boucle principale
# -----------------------------------------------------------------------------

def run_experiments(datasets, workflows, providers, n_runs=1,
                    min_support=DEFAULTS["min_support"],
                    min_confidence=DEFAULTS["min_confidence"],
                    data_dir=DEFAULTS["data_dir"],
                    results_dir=DEFAULTS["results_dir"]):
    """
    Lance toutes les combinaisons (dataset x workflow x provider).

    Pour chaque combinaison :
      - Execute n_runs runs (pour mesurer la variance des workflows agentiques)
      - Agregee et sauvegarde les resultats
      - Construit le tableau comparatif final en CSV
    """
    os.makedirs(results_dir, exist_ok=True)

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
                                    min_support=min_support,
                                    min_confidence=min_confidence,
                                    llm_provider=provider)
                        runs.append(m)
                        print("  Run " + str(run_idx+1) + ": " +
                              str(m["nb_pfds"]) + " PFDs | " +
                              str(round(m["execution_time"], 1)) + "s")
                    except Exception as exc:
                        print("  Run " + str(run_idx+1) + ": ERREUR — " + str(exc))

                if not runs:
                    continue

                agg = aggregate_runs(runs)
                aggregated_results[(dataset_name, workflow_name, provider_name)] = agg

                detail = {
                    "dataset":   dataset_name,
                    "workflow":  workflow_name,
                    "provider":  provider_name,
                    "params":    {"min_support": min_support,
                                 "min_confidence": min_confidence},
                    "aggregated": agg,
                    "runs":       runs,
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
        description="Runner d'experiences PFD — generique"
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
    parser.add_argument("--providers",      nargs="+", default=[])
    parser.add_argument("--runs",           type=int,   default=1)
    parser.add_argument("--min-support",    type=int,   default=DEFAULTS["min_support"])
    parser.add_argument("--min-confidence", type=float, default=DEFAULTS["min_confidence"])
    parser.add_argument("--data-dir",       default=DEFAULTS["data_dir"])
    parser.add_argument("--results-dir",    default=DEFAULTS["results_dir"])
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
        results_dir=args.results_dir,
    )
