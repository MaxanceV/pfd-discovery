"""
Algorithme FASTFD -- perspective par paires de tuples.
Slides 30-37 du cours Constraint_Discovery.

Idee generale :
  1. Calculer les difference sets entre toutes les paires (ti, tj)
     diff(ti,tj) = {A | ti[A] != tj[A]}
  2. Pour chaque attribut cible A :
     a. Garder les diff sets qui contiennent A -> retirer A -> D^A_I
     b. Trouver les minimal hitting sets de D^A_I
     c. Chaque hitting set = un LHS d'une FD minimale X -> A

Avantage vs TANE : adapte aux tables larges (beaucoup d'attributs,
peu de tuples). Complexite dominee par O(n^2) paires de tuples.
"""

import pandas as pd


def compute_difference_sets(df: pd.DataFrame) -> list:
    """
    Calcule tous les difference sets entre paires de tuples.
    diff(ti, tj) = frozenset des colonnes ou les valeurs different.
    Slide 31 du cours.

    Complexite : O(n^2 x m) avec n=nb tuples, m=nb colonnes.
    """
    cols = list(df.columns)
    rows = df.values.tolist()
    diff_sets = []

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            diff = frozenset(
                cols[k] for k in range(len(cols))
                if rows[i][k] != rows[j][k]
            )
            if diff:
                diff_sets.append(diff)

    return diff_sets


def build_da(diff_sets: list, target: str) -> list:
    """
    Construit D^A_I pour l'attribut cible A.
    Slides 32-33 du cours.

    Etape 1 : garder uniquement les diff sets qui contiennent A
              (seules les paires qui different sur A sont pertinentes)
    Etape 2 : retirer A de ces ensembles
              (on cherche ce qui distingue les tuples, pas A lui-meme)

    Exemple (slide 33) :
      diff_sets = [{A,D}, {B,C}, {A,B,C,D}]
      build_da(..., "A") -> [{D}, {B,C,D}]
    """
    da = []
    for ds in diff_sets:
        if target in ds:
            reduced = ds - frozenset([target])
            if reduced:
                da.append(reduced)
    return list(set(da))


def find_minimal_hitting_sets(sets: list) -> list:
    """
    Trouve tous les minimal hitting sets de 'sets'.
    Un hitting set X verifie : pour tout S dans sets, X inter S != vide.
    Exploration DFS avec elagage (slides 35-37).

    Exemple (slide 33) :
      D^A_I = [{D}, {B,C,D}]
      Hitting sets minimaux : [{D}]
      -> FD : D -> A
    """
    if not sets:
        return [frozenset()]

    # Trier par taille croissante -> on coupe plus tot
    sets_sorted = sorted(sets, key=len)
    results = []

    def dfs(remaining: list, current: frozenset):
        # Plus de contraintes -> hitting set trouve
        if not remaining:
            # Verifier minimalite : aucun resultat existant
            # ne doit etre un sous-ensemble de current
            for r in results:
                if r <= current:
                    return
            results.append(current)
            return

        # Prendre le premier ensemble non encore couvert
        first = remaining[0]

        for attr in sorted(first):
            new_current = current | frozenset([attr])

            # Pruning : deja domine par un resultat existant
            if any(r <= new_current for r in results):
                continue

            # Eliminer les sets maintenant couverts par attr
            new_remaining = [s for s in remaining if attr not in s]
            dfs(new_remaining, new_current)

    dfs(sets_sorted, frozenset())
    return results


def fastfd(df: pd.DataFrame) -> list:
    """
    Decouverte de toutes les FDs minimales de df avec FASTFD.
    Slides 30-37 du cours Constraint_Discovery.

    Returns:
        Liste de tuples (lhs, rhs) identique au format de tane().

    Exemple de sortie sur le dataset du cours (slide 7) :
        [("A",), "D"]
        [("D",), "A"]
        [("B",), "C"]
        [("C",), "B"]
    """
    attributes = list(df.columns)
    diff_sets = compute_difference_sets(df)
    found_fds = []

    for target in attributes:
        da = build_da(diff_sets, target)

        if not da:
            # Aucune paire ne differe sur target -> pas de FD interessante
            continue

        hitting_sets = find_minimal_hitting_sets(da)

        for hs in hitting_sets:
            if hs:
                lhs = tuple(sorted(hs))
                found_fds.append((lhs, target))

    return found_fds