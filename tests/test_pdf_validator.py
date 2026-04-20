import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.patterns.extractor import enrich_dataframe
from src.patterns.pfd_validator import compute_support_confidence

def run_real_data_test():
    # Chemins des fichiers (ajuste si nécessaire selon ton arborescence)
    data_dir = "data/pfd_validation/"
    
    # --- TEST 1 : t2.csv (ZIP -> City) ---
    print("=" * 60)
    print("TEST SUR DONNÉES RÉELLES : t2.csv (ZIP -> City)")
    print("=" * 60)
    
    t2_path = os.path.join(data_dir, "t2.csv")
    if os.path.exists(t2_path):
        df_t2 = pd.read_csv(t2_path)
        
        # 1. Enrichissement : Extraction du préfixe de 3 caractères sur le ZIP
        df_t2_e = enrich_dataframe(df_t2, column="ZIP", transforms=["prefix_3"])
        
        # 2. Validation de la PFD : ZIP__prefix_3 -> City
        res_t2 = compute_support_confidence(df_t2_e, "ZIP__prefix_3", "CITY")
        
        print(f"PFD testée : {res_t2['lhs']} -> {res_t2['rhs']}")
        print(f"Support    : {res_t2['support']}")
        print(f"Confiance  : {res_t2['confidence']}")
        print(f"Violations : {len(res_t2['violations'])}")
        
        # Vérification par rapport aux attentes du projet (conf >= 0.85)
        if res_t2['confidence'] >= 0.85:
            print(" PFD Valide selon les seuils du projet.")
        else:
            print(" Confiance trop faible pour valider la PFD.")
    else:
        print(f"Fichier {t2_path} introuvable.")

    # --- TEST 2 : t1.csv (Full Name -> Gender) ---
    print("\n" + "=" * 60)
    print("TEST SUR DONNÉES RÉELLES : t1.csv (Name -> Gender)")
    print("=" * 60)
    
    t1_path = os.path.join(data_dir, "t1.csv")
    if os.path.exists(t1_path):
        df_t1 = pd.read_csv(t1_path)
        
        # 1. Enrichissement : Extraction du premier mot (prénom)
        df_t1_e = enrich_dataframe(df_t1, column="Full Name", transforms=["first_token"])
        
        # 2. Validation de la PFD : Full Name__first_token -> Gender
        res_t1 = compute_support_confidence(df_t1_e, "Full Name__first_token", "Gender")
        
        print(f"PFD testée : {res_t1['lhs']} -> {res_t1['rhs']}")
        print(f"Support    : {res_t1['support']}")
        print(f"Confiance  : {res_t1['confidence']}")
        print(f"Violations : {len(res_t1['violations'])}")
        
        if res_t1['violations']:
            print(f"Exemple de violation : Index {res_t1['violations'][0]['index']}")
            print(f"  LHS: {res_t1['violations'][0]['lhs']} | Trouvé: {res_t1['violations'][0]['rhs_found']} | Attendu: {res_t1['violations'][0]['rhs_expected']}")

if __name__ == "__main__":
    run_real_data_test()