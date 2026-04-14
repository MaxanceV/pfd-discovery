"""
Tests pour src/patterns/extractor.py
Valide chaque transformation sur des valeurs unitaires
puis sur les vrais datasets du projet.
"""

import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.patterns.extractor import (
    extract_prefix, extract_suffix, extract_first_token,
    extract_last_token, extract_domain,
    enrich_dataframe, enrich_dataframe_multi, TRANSFORMATIONS
)

# ── Tests unitaires ─────────────────────────────────────────────────────────

def test_prefix():
    assert extract_prefix("90012", 3) == "900"
    assert extract_prefix("90012", 5) == "90012"
    assert extract_prefix("AB",    3) == "AB"    # plus court que n
    assert extract_prefix(None,    3) == ""
    assert extract_prefix(float('nan'), 3) == ""
    print("OK  extract_prefix")

def test_suffix():
    assert extract_suffix("gmail.com", 3) == "com"
    assert extract_suffix("IL",        3) == "IL"   # plus court que n
    assert extract_suffix(None,        2) == ""
    print("OK  extract_suffix")

def test_first_token():
    assert extract_first_token("John Smith")       == "John"
    assert extract_first_token("Aarhus, Pam J.")   == "Aarhus"
    assert extract_first_token("  Alice  ")        == "Alice"
    assert extract_first_token("")                 == ""
    assert extract_first_token(None)               == ""
    print("OK  extract_first_token")

def test_last_token():
    assert extract_last_token("John Smith")       == "Smith"
    assert extract_last_token("Aarhus, Pam J.")   == "J"
    assert extract_last_token("Chicago")          == "Chicago"
    assert extract_last_token(None)               == ""
    print("OK  extract_last_token")

def test_domain():
    assert extract_domain("john@gmail.com")        == "gmail.com"
    assert extract_domain("alice@us.example.com")  == "us.example.com"
    assert extract_domain("pas_un_email")          == ""
    assert extract_domain(None)                    == ""
    assert extract_domain("JOHN@GMAIL.COM")        == "gmail.com"  # lowercase
    print("OK  extract_domain")

def test_enrich_dataframe():
    df = pd.DataFrame({
        "zip":  ["90012", "90013", "10001"],
        "city": ["Los Angeles", "Los Angeles", "New York"]
    })
    df_e = enrich_dataframe(df, "zip", ["prefix_3", "prefix_2"])

    assert "zip__prefix_3" in df_e.columns
    assert "zip__prefix_2" in df_e.columns
    assert df_e["zip__prefix_3"].tolist() == ["900", "900", "100"]
    assert df_e["zip__prefix_2"].tolist() == ["90",  "90",  "10"]
    assert "zip"  in df_e.columns  # colonne originale conservee
    assert "city" in df_e.columns  # autres colonnes conservees
    print("OK  enrich_dataframe")

def test_enrich_dataframe_multi():
    df = pd.DataFrame({
        "zip":  ["90012", "10001"],
        "name": ["John Smith", "Susan Miller"]
    })
    df_e = enrich_dataframe_multi(df, {
        "zip":  ["prefix_3"],
        "name": ["first_token", "last_token"]
    })
    assert "zip__prefix_3"    in df_e.columns
    assert "name__first_token" in df_e.columns
    assert "name__last_token"  in df_e.columns
    assert df_e["name__first_token"].tolist() == ["John",  "Susan"]
    assert df_e["name__last_token"].tolist()  == ["Smith", "Miller"]
    print("OK  enrich_dataframe_multi")

# ── Tests sur les vrais datasets ────────────────────────────────────────────

def test_on_t1():
    """t1.csv : Employes — verifier first_token(Full Name) -> genre potentiel"""
    path = "data/pfd_validation/t1.csv"
    if not os.path.exists(path):
        print("SKIP test_on_t1 (fichier introuvable)")
        return

    df = pd.read_csv(path).head(50)
    df_e = enrich_dataframe(df, "Full Name", ["first_token", "last_token"])

    assert "Full Name__first_token" in df_e.columns
    assert "Full Name__last_token"  in df_e.columns
    assert df_e["Full Name__first_token"].iloc[0] != ""

    print(f"OK  test_on_t1 — exemple first_token: "
          f"'{df['Full Name'].iloc[0]}' -> "
          f"'{df_e['Full Name__first_token'].iloc[0]}'")

def test_on_t2():
    """t2.csv : Entreprises — verifier prefix(ZIP,3) -> city"""
    path = "data/pfd_validation/t2.csv"
    if not os.path.exists(path):
        print("SKIP test_on_t2 (fichier introuvable)")
        return

    df = pd.read_csv(path).head(50)

    # Trouver la colonne ZIP (peut s'appeler ZIP, ZIP Code, Zip...)
    zip_col = next((c for c in df.columns if "zip" in c.lower()), None)
    if not zip_col:
        print("SKIP test_on_t2 (colonne ZIP introuvable)")
        return

    df_e = enrich_dataframe(df, zip_col, ["prefix_3", "prefix_5"])
    assert f"{zip_col}__prefix_3" in df_e.columns

    print(f"OK  test_on_t2 — exemple prefix_3: "
          f"'{df[zip_col].iloc[0]}' -> "
          f"'{df_e[f'{zip_col}__prefix_3'].iloc[0]}'")

def test_on_t3():
    """t3.csv : Licences — verifier prefix(Zip,3) -> city"""
    path = "data/pfd_validation/t3.csv"
    if not os.path.exists(path):
        print("SKIP test_on_t3 (fichier introuvable)")
        return

    df = pd.read_csv(path).head(50)
    zip_col = next((c for c in df.columns if "zip" in c.lower()), None)
    if not zip_col:
        print("SKIP test_on_t3 (colonne ZIP introuvable)")
        return

    df_e = enrich_dataframe(df, zip_col, ["prefix_3"])
    assert f"{zip_col}__prefix_3" in df_e.columns

    print(f"OK  test_on_t3 — exemple prefix_3: "
          f"'{df[zip_col].iloc[0]}' -> "
          f"'{df_e[f'{zip_col}__prefix_3'].iloc[0]}'")

def test_on_us_phone():
    """US_Phone_Code.csv : State -> Code — verifier raw et prefix"""
    path = "data/pfd_validation/US_Phone_Code.csv"
    if not os.path.exists(path):
        print("SKIP test_on_us_phone (fichier introuvable)")
        return

    df = pd.read_csv(path)
    df_e = enrich_dataframe(df, "State", ["uppercase", "lowercase"])

    assert "State__uppercase" in df_e.columns
    print(f"OK  test_on_us_phone — {len(df)} etats charges")

# ── Lancement ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("TESTS UNITAIRES")
    print("=" * 50)
    test_prefix()
    test_suffix()
    test_first_token()
    test_last_token()
    test_domain()
    test_enrich_dataframe()
    test_enrich_dataframe_multi()

    print()
    print("=" * 50)
    print("TESTS SUR LES VRAIS DATASETS")
    print("=" * 50)
    test_on_t1()
    test_on_t2()
    test_on_t3()
    test_on_us_phone()

    print()
    print("Tous les tests sont passes.")