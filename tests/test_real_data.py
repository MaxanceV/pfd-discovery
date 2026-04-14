import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.tane import tane
from src.core.fastfd import fastfd

# ── Choisir le dataset ──────────────────────────────────────────────────────

# Option 1 : t1.csv — Employés (name, gender, department...)
#df = pd.read_csv("data/pfd_validation/t1.csv")

# Option 2 : t2.csv — Entreprises (ZIP, city, phone...)
#df = pd.read_csv("data/pfd_validation/t2.csv")

# Option 3 : t3.csv — Licences (zip, city, state...)
df = pd.read_csv("data/pfd_validation/t3.csv")

# Option 4 : US_Phone_Code.csv — State → Code
# df = pd.read_csv("data/pfd_validation/US_Phone_Code.csv")

# ── Infos sur le dataset ────────────────────────────────────────────────────
print(f"Dataset : {df.shape[0]} lignes x {df.shape[1]} colonnes")
print(f"Colonnes : {list(df.columns)}")
print()

# ── Attention : TANE et FASTFD sont lents sur les gros datasets ─────────────
# On limite aux colonnes qui nous interessent
# Pour t1.csv : tester name/gender/department
cols = list(df.columns)[:4]   # prendre les 4 premieres colonnes
df_small = df[cols].dropna().head(1000) # max 100 lignes pour commencer

print(f"Sous-ensemble teste : {df_small.shape[0]} lignes x {df_small.shape[1]} colonnes")
print(f"Colonnes : {list(df_small.columns)}")
print()

# ── TANE ────────────────────────────────────────────────────────────────────
print("=== TANE ===")
fds_tane = tane(df_small)
if fds_tane:
    for lhs, rhs in sorted(fds_tane):
        print(f"  {list(lhs)} → {rhs}")
else:
    print("  Aucune FD trouvee")

print()

# ── FASTFD ──────────────────────────────────────────────────────────────────
print("=== FASTFD ===")
fds_fastfd = fastfd(df_small)
if fds_fastfd:
    for lhs, rhs in sorted(fds_fastfd):
        print(f"  {list(lhs)} → {rhs}")
else:
    print("  Aucune FD trouvee")