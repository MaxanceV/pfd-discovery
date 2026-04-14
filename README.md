# PFD-Discovery — Découverte de Pattern Functional Dependencies

> Master Qualité des Données — Université Paris Dauphine
> Cours de Khalid Belhajjame
> **Maxance Villame · Ferdinand Martin Lavigne · Baptiste Matrat · Marie Probert · Justine Rault**

---

## Présentation du projet

Ce projet implémente et compare deux approches pour la découverte de **Pattern Functional Dependencies (PFDs) approximatives** :

1. **Approche classique** : algorithmes TANE et FASTFD vus en cours
2. **Approche agentique** : utilisation d'un LLM pour guider la recherche

Une PFD est une généralisation des dépendances fonctionnelles classiques. Au lieu de comparer des valeurs entières, elle compare des **patterns** extraits de ces valeurs.

**Exemple :** `prefix(zip, 3) → city` — les 3 premiers chiffres du code postal déterminent la ville.

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
│   ├── core/                       # Algorithmes du cours
│   │   ├── partition.py            # Partitions d'équivalence (brique de base TANE)
│   │   ├── tane.py                 # Algorithme TANE complet avec pruning
│   │   └── fastfd.py               # Algorithme FASTFD : difference sets + hitting sets
│   │
│   ├── patterns/                   # Extension PFD
│   │   ├── extractor.py            # Extraction de patterns (prefix, token, domain...)
│   │   ├── pfd_validator.py        # Calcul support et confidence d'une PFD
│   │   └── pfd_discovery.py        # Algorithme classique de découverte PFD
│   │
│   ├── agent/                      # Couche LLM agentique
│   │   ├── semantic_profiler.py    # LLM analyse le schéma et les types d'attributs
│   │   ├── candidate_selector.py   # LLM filtre et priorise les candidats
│   │   └── workflow.py             # Les 3 workflows du cours
│   │
│   └── experiments/
│       ├── runner.py               # Lance les expériences sur les datasets
│       └── metrics.py              # Calcule et compare les métriques
│
├── tests/
│   └── test_core.py                # Tests sur le dataset exemple du cours (slide 7)
│
├── results/                        # Résultats JSON générés automatiquement
├── notebooks/                      # Exploration et visualisation
├── .env                            # Clés API — NE PAS COMMITER
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Lancer les tests

```bash
python tests/test_core.py
```

Résultat attendu (dataset exemple du cours, slide 7) :

```
=== TANE ===
  ['A'] → D
  ['D'] → A
  ['B'] → C
  ['C'] → B

=== FASTFD ===
  ['A'] → D
  ['B'] → C
  ['C'] → B
  ['D'] → A
```

Les deux algorithmes doivent retourner exactement les mêmes FDs.

---

## Workflow Git — travailler à 5

### Règle principale : chacun travaille sur sa propre branche, jamais directement sur `main`

```bash
# Créer sa branche
git checkout -b feature/ma-partie

# Travailler puis commiter régulièrement
git add src/mon_fichier.py
git commit -m "feat: description de ce que j'ai fait"
git push origin feature/ma-partie
```

Quand c'est prêt → ouvrir une **Pull Request** sur GitHub vers `main`. Un autre coéquipier relit et merge.

### Répartition des branches

| Personne | Branche | Fichiers |
|---|---|---|
| Maxance Villame | `feature/pattern-extractor` | `src/patterns/extractor.py` |
| Ferdinand Martin Lavigne | `feature/validator` | `src/patterns/pfd_validator.py` + `pfd_discovery.py` |
| Baptiste Matrat | `feature/agent` | `src/agent/` |
| Marie Probert | `feature/experiments` | `src/experiments/` |
| Justine Rault | `feature/report` | Rapport + intégration + README |

### Mettre à jour sa branche avec les derniers changements de `main`

```bash
git checkout main
git pull origin main
git checkout feature/ma-partie
git merge main
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

## Dépendances

| Package | Usage |
|---|---|
| `pandas` | Manipulation des DataFrames |
| `numpy` | Calculs numériques |
| `anthropic` | API Claude (LLM agentique) |
| `openai` | API OpenAI (comparaison LLMs) |
| `google-generativeai` | API Gemini (comparaison LLMs) |
| `python-dotenv` | Chargement des clés API depuis `.env` |
| `pytest` | Tests unitaires |
| `tqdm` | Barres de progression |
| `jupyter` | Notebooks d'exploration |

---

## Points importants

- Ne jamais commiter le fichier `.env`
- Toujours travailler sur une branche, jamais directement sur `main`
- Commiter régulièrement avec des messages clairs
- Utiliser les mêmes datasets pour toutes les méthodes (comparaison équitable)
- Seuils par défaut suggérés : `support >= 10`, `confidence >= 0.85`
- En cas de conflit Git : ne pas forcer le merge, consulter l'équipe