"""
Partitions d'equivalence -- brique fondamentale de TANE.

Rappels du cours (Constraint_Discovery slides 13-14) :
  - pi_X  = partition de la relation induite par l'attribut set X
  - X -> A  tient  ssi  |pi_X| == |pi_{X union {A}}|
  - Raffinement (slide 26) : pi_{X union Y} = intersections des classes
    -> permet de NE PAS relire le DataFrame pour les niveaux > 1
"""

import pandas as pd


def compute_partition(df: pd.DataFrame, attributes: list) -> list:
    """
    Calcule la partition induite par 'attributes' sur df.
    Retourne une liste de frozensets d'index de tuples.
    On ne garde que les classes de taille > 1 (les singletons
    ne peuvent jamais creer de violation de FD).

    Exemple :
      df avec colonne A = [a1, a2, a1]
      compute_partition(df, ["A"]) -> [{0,2}]  (singleton {1} ignore)
    """
    groups = df.groupby(attributes, sort=False).groups
    return [frozenset(idx) for idx in groups.values() if len(idx) > 1]


def refine_partitions(pi_x: list, pi_y: list) -> list:
    """
    Calcule pi_{X union Y} a partir de pi_X et pi_Y par intersection.
    Slide 26 du cours.

    Formule :
      pi_{X union Y} = { C_X inter C_Y | C_X in pi_X, C_Y in pi_Y,
                         |C_X inter C_Y| > 1 }

    Pourquoi c'est efficace : on ne retouche JAMAIS le DataFrame
    une fois les partitions de niveau 1 calculees.
    """
    result = []
    for cx in pi_x:
        for cy in pi_y:
            inter = cx & cy
            if len(inter) > 1:
                result.append(inter)
    return result


def check_fd_holds(pi_x: list, pi_xa: list) -> bool:
    """
    X -> A tient ssi |pi_X| == |pi_{X union A}|
    Slide 13 du cours.

    Intuition : si ajouter A ne raffine pas la partition de X,
    c'est que A est deja entierement determine par X.
    """
    return len(pi_x) == len(pi_xa)


def error_rate(pi_x: list, pi_xa: list, n_tuples: int) -> float:
    """
    Taux d'erreur d'une FD approximative.
    Utilise pour les FDs approx (slides 48-50 du cours).

    error = (|pi_XA| - |pi_X|) / n_tuples
    Plus c'est proche de 0, plus la FD est presque vraie.
    """
    if n_tuples == 0:
        return 0.0
    return (len(pi_xa) - len(pi_x)) / n_tuples