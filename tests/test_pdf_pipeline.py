"""
Test du pipeline complet :
  1. Charger un dataset
  2. Enrichir avec des patterns
  3. Lancer TANE/FASTFD sur les colonnes derivees
  4. Interpreter les resultats comme des PFDs
"""

import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.patterns.extractor import enrich_dataframe, enrich_dataframe_multi
from src.core.tane import tane
from src.core.fastfd import fastfd


def run_pfd_discovery(df_enriched, original_cols, algo="fastfd"):
    """
    Lance la decouverte de FDs sur le DataFrame enrichi.
    Filtre pour ne garder que les regles interessantes :
      - LHS = colonne derivee (contient '__')
      - RHS = colonne originale
    """
    if algo == "tane":
        fds = tane(df_enriched)
    else:
        fds = fastfd(df_enriched)

    pfds = []
    for lhs, rhs in fds:
        lhs_list = list(lhs)
        # Garder uniquement : LHS contient un pattern, RHS est une colonne originale
        has_pattern = any("__" in col for col in lhs_list)
        rhs_is_original = rhs in original_cols
        if has_pattern and rhs_is_original:
            pfds.append((lhs_list, rhs))

    return pfds


# ════════════════════════════════════════════════════════════════════════════
# TEST 1 : t2.csv — prefix(ZIP,3) → city
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1 : t2.csv — ZIP → City")
print("=" * 60)

df = pd.read_csv("data/pfd_validation/t2.csv").head(200)
print(f"Colonnes disponibles : {list(df.columns)}")

# Enrichir ZIP avec des prefixes
df_e = enrich_dataframe(df, "ZIP", ["prefix_3", "prefix_2", "prefix_5"])
print(f"Colonnes apres enrichissement : {list(df_e.columns)}")
print()

# On garde seulement les colonnes utiles pour l'algo
cols_to_test = ["ZIP__prefix_3", "ZIP__prefix_2", "CITY", "STATE"]
cols_to_test = [c for c in cols_to_test if c in df_e.columns]
df_small = df_e[cols_to_test].dropna()

print(f"Sous-ensemble : {df_small.shape[0]} lignes x {df_small.shape[1]} colonnes")
print()

pfds = run_pfd_discovery(df_small, original_cols=["CITY", "STATE"])
print("PFDs decouvertes :")
if pfds:
    for lhs, rhs in sorted(pfds):
        print(f"  {lhs} → {rhs}")
else:
    print("  Aucune PFD trouvee")


# ════════════════════════════════════════════════════════════════════════════
# TEST 2 : t1.csv — first_token(name) → gender
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 2 : t1.csv — first_token(Full Name) → Gender")
print("=" * 60)

df = pd.read_csv("data/pfd_validation/t1.csv").head(200)

# Enrichir le nom avec first_token
df_e = enrich_dataframe(df, "Full Name", ["first_token"])

# Garder seulement les colonnes utiles
cols_to_test = ["Full Name__first_token", "Gender"]
df_small = df_e[cols_to_test].dropna()

print(f"Sous-ensemble : {df_small.shape[0]} lignes x {df_small.shape[1]} colonnes")
print()

pfds = run_pfd_discovery(df_small, original_cols=["Gender"])
print("PFDs decouvertes :")
if pfds:
    for lhs, rhs in sorted(pfds):
        print(f"  {lhs} → {rhs}")
else:
    print("  Aucune PFD exacte trouvee (normal : une PFD approx peut quand meme exister)")


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 : US_Phone_Code.csv — State → Code
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 3 : US_Phone_Code.csv — State → Code")
print("=" * 60)

df = pd.read_csv("data/pfd_validation/US_Phone_Code.csv").dropna()
print(f"Colonnes : {list(df.columns)}")
print(f"Lignes : {len(df)}")
print()

# Pas besoin de pattern ici, FD directe
fds = fastfd(df)
print("FDs decouvertes :")
for lhs, rhs in sorted(fds):
    print(f"  {list(lhs)} → {rhs}")