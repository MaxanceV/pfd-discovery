# PFD-Discovery — Découverte de Pattern Functional Dependencies

> Master Qualité des Données — Université Paris Dauphine
> Cours de Khalid Belhajjame
> **Maxance Villame · Ferdinand Martin Lavigne · Baptiste Matrat · Marie Probert · Justine Rault**

---

## Présentation du projet

Ce projet implémente et compare deux approches pour la découverte de **Pattern Functional Dependencies (PFDs) approximatives** :

1. **Approche classique** : découverte brute-force par extraction de patterns et validation algorithmique
2. **Approche agentique** : utilisation d'un LLM pour guider la recherche

Une PFD est une généralisation des dépendances fonctionnelles classiques. Au lieu de comparer des valeurs entières, elle compare des **patterns** extraits de ces valeurs.

**Exemple :** `prefix(zip, 3) → state` — les 3 premiers chiffres du code postal déterminent l'état.

---

## Prérequis

- Python 3.10 ou supérieur
- Git
- VSCode (recommandé)
- Git Bash (Windows) — recommandé plutôt que PowerShell

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/MaxanceV/pfd-discovery.git
cd pfd-discovery
```

### 2. Créer l'environnement virtuel

**Windows (Git Bash) :**
```bash
python -m venv venv
source venv/Scripts/activate
```

**Mac / Linux :**
```bash
python -m venv venv
source venv/bin/activate
```

Tu dois voir `(venv)` apparaître au début de ta ligne de commande.

> **Problème PowerShell ?** Lance cette commande une seule fois en admin :
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API

Crée un fichier `.env` à la racine du projet. **Ce fichier ne doit JAMAIS être commité sur Git.**

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

---

## Structure du projet

```
pfd-discovery/
│
├── data/
│   ├── CHE/                        # Données ChEMBL (protéines, variants...)
│   │   ├── mechanism_refs.csv
│   │   ├── metabolism_refs.csv
│   │   ├── protein_classification.csv
│   │   ├── research_companies.csv
│   │   └── variant_sequences.csv
│   │
│   ├── DGOV/                       # Données gouvernementales US brutes
│   │   ├── 570-1.csv               # Employés (nom, genre, département)
│   │   ├── 6339-1.csv              # Crimes par ville
│   │   ├── 6397-1.csv              # Statistiques démographiques
│   │   ├── 10492-1.csv             # Licences (zip, city, state)
│   │   └── 10642-1.csv             # Entreprises (ZIP, city, phone)
│   │
│   └── pfd_validation/             # Datasets nettoyés pour les expériences
│       ├── t1.csv                  # Employés — first_token(name) → gender
│       ├── t2.csv                  # Entreprises — prefix(ZIP,3) → city
│       ├── t3.csv                  # Licences — prefix(Zip,3) → city
│       └── US_Phone_Code.csv       # State → Code téléphone
│
├── src/
│   ├── patterns/                   # Découverte PFD
│   │   ├── extractor.py            # Extraction de patterns (prefix, token, domain...)
│   │   ├── pfd_validator.py        # Calcul support et confidence d'une PFD
│   │   └── pfd_discovery.py        # Algorithme classique de découverte PFD (brute-force)
│   │
│   ├── agent/                      # Couche LLM agentique
│   │   ├── llm_provider.py         # Abstraction multi-providers (Claude, Mistral, Groq...)
│   │   ├── semantic_profiler.py    # LLM analyse le schéma et les types d'attributs
│   │   └── workflow.py             # Les 3 workflows (classique, agentique v1, v2)
│   │
│   └── experiments/
│       ├── runner.py               # Lance les expériences sur les datasets
│       └── metrics.py              # Calcule et compare les métriques
│
├── tests/                          # Tests unitaires et d'intégration
│   ├── test_extractor.py           # Tests des transformations de patterns
│   ├── test_pdf_validator.py       # Tests du calcul support/confidence
│   ├── test_pfd_discovery.py       # Tests de la découverte classique
│   └── test_agent.py               # Tests des workflows agentiques
│
├── notebooks/
│   └── exploration.ipynb           # Exploration et visualisation des résultats
│
├── PatternFD-miniprojet/
│   └── Approximate_PFDs.pdf        # Slides du cours (référence)
│
├── results/                        # Résultats JSON générés automatiquement
├── .env                            # Clés API — NE PAS COMMITER
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Lancer les tests

```bash
python tests/test_extractor.py
python tests/test_pdf_validator.py
python tests/test_pfd_discovery.py
python tests/test_agent.py
```

---

## Datasets disponibles

| Fichier | Contenu | PFDs attendues |
|---|---|---|
| `pfd_validation/t1.csv` | Employés (nom, genre, département) | `first_token(name) → gender` |
| `pfd_validation/t2.csv` | Entreprises (ZIP, ville, téléphone) | `prefix(ZIP,3) → city` |
| `pfd_validation/t3.csv` | Licences (zip, city, state) | `prefix(Zip,3) → city` |
| `pfd_validation/US_Phone_Code.csv` | États US et codes téléphone | `State → Code` |

---

## Points importants

- Ne jamais commiter le fichier `.env`
- Commiter régulièrement avec des messages clairs
- Utiliser les mêmes datasets pour toutes les méthodes (comparaison équitable)
- Seuils par défaut suggérés : `support >= 10`, `confidence >= 0.85`

## Lancer les expériences

### Commandes

| Commande | Description |
|---|---|
| `python -m src.experiments.runner` | Workflow classique sur tous les datasets |
| `python -m src.experiments.runner --datasets t1.csv` | Workflow classique sur un dataset |
| `python -m src.experiments.runner --workflows classical agent_v1 agent_v2` | Tous les workflows, providers auto-détectés |
| `python -m src.experiments.runner --workflows agent_v1 agent_v2 --providers claude mistral` | Workflows agentiques avec providers spécifiques |
| `python -m src.experiments.runner --workflows agent_v1 --runs 3` | Plusieurs runs pour mesurer la variance |
| `python -m src.experiments.runner --min-support 20 --min-confidence 0.9` | Seuils personnalisés |

Les résultats sont sauvegardés dans `results/` : un fichier JSON par combinaison (dataset × workflow × provider) et un tableau comparatif `comparison_table.csv`.

---

## Arguments du runner

Référence complète de tous les arguments acceptés par `src/experiments/runner.py`.

| Argument | Type | Défaut | Description |
|---|---|---|---|
| `--datasets` | liste | `t1.csv t2.csv t3.csv US_Phone_Code.csv` | Un ou plusieurs fichiers CSV à traiter (cherchés dans `--data-dir`) |
| `--workflows` | liste | `classical` | Workflows à exécuter. Valeurs possibles : `classical`, `agent_v1`, `agent_v2` |
| `--providers` | liste | *(auto-détecté)* | Providers LLM à utiliser pour les workflows agentiques. Valeurs possibles : `groq`, `mistral`, `huggingface`. Si absent, les providers disponibles sont détectés via les clés API du `.env` |
| `--runs` | entier | `1` | Nombre de répétitions par combinaison (dataset × workflow × provider), utile pour mesurer la variance des workflows agentiques |
| `--min-support` | entier | `10` | Seuil minimum de support pour retenir une PFD (nombre de lignes couvertes) |
| `--min-confidence` | flottant | `0.85` | Seuil minimum de confiance pour retenir une PFD (entre 0 et 1) |
| `--data-dir` | chemin | `data/pfd_validation` | Dossier racine où sont cherchés les fichiers CSV |
| `--results-dir` | chemin | `results` | Dossier de sortie pour les fichiers JSON et le tableau comparatif |

### Exemples

```bash
# Workflow classique sur tous les datasets (défaut)
python -m src.experiments.runner

# Un seul dataset, un seul workflow
python -m src.experiments.runner --datasets t2.csv --workflows classical

# Tous les workflows avec providers explicites
python -m src.experiments.runner --workflows classical agent_v1 agent_v2 --providers groq mistral

# 3 répétitions, seuils personnalisés
python -m src.experiments.runner --workflows agent_v2 --runs 3 --min-support 20 --min-confidence 0.9

# Dataset custom dans un autre dossier
python -m src.experiments.runner --datasets mon_fichier.csv --data-dir data/CHE --results-dir results/CHE
```