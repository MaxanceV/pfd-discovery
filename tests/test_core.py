import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.tane import tane
from src.core.fastfd import fastfd

# Exemple exact du cours (slide 7 et 17-23)
df = pd.DataFrame({
    "A": ["a1", "a2", "a1"],
    "B": ["b1", "b1", "b2"],
    "C": ["c1", "c1", "c2"],
    "D": ["d1", "d2", "d1"]
})

print("=== TANE ===")
fds_tane = tane(df)
for lhs, rhs in sorted(fds_tane):
    print(f"  {list(lhs)} → {rhs}")

print("\n=== FASTFD ===")
fds_fastfd = fastfd(df)
for lhs, rhs in sorted(fds_fastfd):
    print(f"  {list(lhs)} → {rhs}")

# Résultat attendu (slides 20-23) :
# A → D,  D → A,  B → C,  C → B