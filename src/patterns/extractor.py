"""
Extraction de patterns pour les PFDs.
Slides 39-42 du cours Approximate_PFDs.

Pipeline :
  1. Extraire des patterns depuis les valeurs brutes
  2. Enrichir le DataFrame avec les colonnes derivees
  3. Ces colonnes servent ensuite de LHS dans la decouverte de PFDs

Transformations disponibles :
  - prefix(value, n)     : n premiers caracteres
  - suffix(value, n)     : n derniers caracteres
  - first_token(value)   : premier mot (separateur : espace, virgule, point)
  - last_token(value)    : dernier mot
  - domain(email)        : domaine d'une adresse email
  - uppercase / lowercase : normalisation de casse
"""

import pandas as pd
import re


def extract_prefix(value, n: int) -> str:
    """
    Retourne les n premiers caracteres de value.
    Exemple : extract_prefix("90012", 3) -> "900"
    Utilise pour les codes postaux, codes pays, prefixes de codes...
    """
    if pd.isna(value):
        return ""
    return str(value)[:n]


def extract_suffix(value, n: int) -> str:
    """
    Retourne les n derniers caracteres de value.
    Exemple : extract_suffix("john@gmail.com", 3) -> "com"
    Utilise pour les extensions, fins de codes...
    """
    if pd.isna(value):
        return ""
    return str(value)[-n:]


def extract_first_token(value) -> str:
    """
    Retourne le premier mot de value.
    Separateurs : espace, virgule, point.
    Exemple : extract_first_token("John Smith") -> "John"
    Exemple : extract_first_token("Aarhus, Pam J.") -> "Aarhus"
    Utilise pour extraire le prenom ou le nom depuis un nom complet.
    """
    if pd.isna(value):
        return ""
    tokens = re.split(r'[\s,\.]+', str(value).strip())
    tokens = [t for t in tokens if t]
    return tokens[0] if tokens else ""


def extract_last_token(value) -> str:
    """
    Retourne le dernier mot de value.
    Separateurs : espace, virgule, point.
    Exemple : extract_last_token("John Smith") -> "Smith"
    Utilise pour extraire le nom de famille.
    """
    if pd.isna(value):
        return ""
    tokens = re.split(r'[\s,\.]+', str(value).strip())
    tokens = [t for t in tokens if t]
    return tokens[-1] if tokens else ""


def extract_domain(email) -> str:
    """
    Retourne le domaine d'une adresse email.
    Exemple : extract_domain("john@gmail.com") -> "gmail.com"
    Exemple : extract_domain("alice@us.example.com") -> "us.example.com"
    Retourne "" si la valeur n'est pas une adresse email valide.
    """
    if pd.isna(email):
        return ""
    s = str(email)
    if '@' not in s:
        return ""
    return s.split('@')[-1].lower().strip()


# ── Catalogue de toutes les transformations disponibles ────────────────────
# Cle    : nom de la transformation (utilise pour nommer les colonnes derivees)
# Valeur : lambda appliquee sur chaque valeur de la colonne
TRANSFORMATIONS = {
    "raw":        lambda v: str(v) if pd.notna(v) else "",
    "prefix_1":   lambda v: extract_prefix(v, 1),
    "prefix_2":   lambda v: extract_prefix(v, 2),
    "prefix_3":   lambda v: extract_prefix(v, 3),
    "prefix_4":   lambda v: extract_prefix(v, 4),
    "prefix_5":   lambda v: extract_prefix(v, 5),
    "suffix_2":   lambda v: extract_suffix(v, 2),
    "suffix_3":   lambda v: extract_suffix(v, 3),
    "suffix_4":   lambda v: extract_suffix(v, 4),
    "first_token": lambda v: extract_first_token(v),
    "last_token":  lambda v: extract_last_token(v),
    "domain":      lambda v: extract_domain(v),
    "uppercase":   lambda v: str(v).upper() if pd.notna(v) else "",
    "lowercase":   lambda v: str(v).lower() if pd.notna(v) else "",
}


def enrich_dataframe(df: pd.DataFrame,
                     column: str,
                     transforms: list) -> pd.DataFrame:
    """
    Ajoute au DataFrame des colonnes derivees par transformation.
    La colonne originale est conservee.

    Args:
        df         : DataFrame source
        column     : nom de la colonne a transformer
        transforms : liste de noms de transformations (cles de TRANSFORMATIONS)

    Returns:
        Nouveau DataFrame avec les colonnes derivees ajoutees.
        Nom des nouvelles colonnes : "{column}__{transform}"

    Exemple :
        enrich_dataframe(df, "zip", ["prefix_3", "prefix_5"])
        -> ajoute les colonnes "zip__prefix_3" et "zip__prefix_5"
    """
    if column not in df.columns:
        raise ValueError(f"Colonne '{column}' introuvable dans le DataFrame.")

    df_enriched = df.copy()

    for t in transforms:
        if t not in TRANSFORMATIONS:
            print(f"  [Warning] Transformation '{t}' inconnue, ignoree.")
            continue
        new_col = f"{column}__{t}"
        df_enriched[new_col] = df[column].apply(TRANSFORMATIONS[t])

    return df_enriched


def enrich_dataframe_multi(df: pd.DataFrame,
                           columns_transforms: dict) -> pd.DataFrame:
    """
    Enrichit le DataFrame sur plusieurs colonnes en une seule fois.

    Args:
        df                   : DataFrame source
        columns_transforms   : dict { nom_colonne: [liste de transforms] }

    Exemple :
        enrich_dataframe_multi(df, {
            "zip":   ["prefix_3", "prefix_5"],
            "name":  ["first_token", "last_token"],
            "email": ["domain"]
        })
    """
    df_enriched = df.copy()
    for column, transforms in columns_transforms.items():
        if column not in df.columns:
            print(f"  [Warning] Colonne '{column}' introuvable, ignoree.")
            continue
        for t in transforms:
            if t not in TRANSFORMATIONS:
                print(f"  [Warning] Transformation '{t}' inconnue, ignoree.")
                continue
            new_col = f"{column}__{t}"
            df_enriched[new_col] = df[column].apply(TRANSFORMATIONS[t])
    return df_enriched