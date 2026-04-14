"""
Algorithme TANE -- decouverte de FDs minimales.
Slides 11-28 du cours Constraint_Discovery.

Idee generale :
  1. Explorer le treillis d'attributs niveau par niveau (BFS)
     Niveau 1 : {A}, {B}, {C}...
     Niveau 2 : {A,B}, {A,C}, {B,C}...
     etc.
  2. Pour chaque ensemble X au niveau k, tester X moins {A} -> A
     pour tout A dans X
  3. Deux regles d'elagage (pruning) :
     - Pruning 1 (slide 15) : si X -> A trouvee, tout superset de X
       pour le meme RHS A est non-minimal -> ignore
     - Pruning 2 (slides 16-23) : si X -> A trouvee, ne plus
       generer de candidats dont le LHS contient a la fois X et A
"""

import pandas as pd
from itertools import combinations
from .partition import compute_partition, refine_partitions, check_fd_holds


def tane(df: pd.DataFrame, max_lhs_size: int = None) -> list:
    """
    Decouverte de toutes les FDs minimales de df avec TANE.

    Args:
        df           : le DataFrame a analyser
        max_lhs_size : taille max du LHS (None = pas de limite)

    Returns:
        Liste de tuples (lhs, rhs) ou
          lhs = tuple trie d'attributs
          rhs = string (un seul attribut)

    Exemple de sortie sur le dataset du cours (slide 7) :
        [("A",), "D"]
        [("D",), "A"]
        [("B",), "C"]
        [("C",), "B"]
    """
    attributes = list(df.columns)

    # Etape 1 : calculer les partitions de niveau 1
    # Un seul scan du DataFrame, jamais relu apres.
    partitions = {}
    for attr in attributes:
        key = frozenset([attr])
        partitions[key] = compute_partition(df, [attr])

    found_fds = []

    # minimal_lhs[A] = liste des LHS minimaux deja trouves pour RHS=A
    # Sert au Pruning 1 : si un LHS connu est sous-ensemble du candidat,
    # le candidat est non-minimal -> on le saute
    minimal_lhs = {attr: [] for attr in attributes}

    # dominated[X] = True si X ne doit plus etre explore
    # Sert au Pruning 2 : evite de generer des supersets inutiles
    dominated = set()

    # BFS niveau par niveau
    # current_level = liste des frozensets d'attributs du niveau courant
    current_level = [frozenset([attr]) for attr in attributes]
    level_num = 1

    while current_level:

        if max_lhs_size and level_num > max_lhs_size:
            break

        next_level_candidates = []

        for X in current_level:

            if X in dominated:
                continue

            # Tester X moins {A} -> A pour chaque A dans X
            for A in list(X):
                lhs = X - frozenset([A])

                # Pruning 1 : verifier qu'aucun sous-ensemble de lhs
                # n'est deja un LHS minimal pour A.
                # Si oui, ce candidat est non-minimal -> on saute.
                already_minimal = any(
                    known <= lhs for known in minimal_lhs[A]
                )
                if already_minimal:
                    continue

                if not lhs:
                    # Niveau 1 : lhs vide = tester si A est constant
                    pi_lhs = [frozenset(range(len(df)))]
                else:
                    if lhs not in partitions:
                        continue
                    pi_lhs = partitions[lhs]

                pi_x = partitions[X]

                # Test de la FD via les partitions (slide 13)
                if check_fd_holds(pi_lhs, pi_x):
                    lhs_tuple = tuple(sorted(lhs)) if lhs else ()
                    found_fds.append((lhs_tuple, A))

                    # Enregistrer ce LHS comme minimal pour A
                    minimal_lhs[A].append(lhs)

                    # Pruning 2 : marquer X comme domine pour eviter
                    # de generer des supersets inutiles vers le niveau suivant
                    dominated.add(X)

        # Generer les candidats du niveau suivant
        # On fusionne deux ensembles qui different d'un seul element
        seen = set()
        current_set = set(current_level)

        for i, X1 in enumerate(current_level):
            for X2 in current_level[i+1:]:
                union = X1 | X2
                if len(union) != level_num + 1:
                    continue
                if union in seen:
                    continue

                # Pruning 2 : ne pas explorer les supersets domines
                if union in dominated:
                    continue

                # Verifier que tous les sous-ensembles de taille level_num
                # sont dans le niveau courant (condition de completude)
                subs = [frozenset(c) for c in combinations(union, level_num)]
                if not all(s in current_set for s in subs):
                    continue

                seen.add(union)

                # Calculer la partition par raffinement
                # (sans relire le DataFrame !)
                if union not in partitions:
                    partitions[union] = refine_partitions(
                        partitions[X1], partitions[X2]
                    )

                next_level_candidates.append(union)

        current_level = next_level_candidates
        level_num += 1

    return found_fds