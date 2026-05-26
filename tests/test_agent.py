import os
import sys
from google import genai
import pandas as pd
from dotenv import load_dotenv

# 1. Configuration
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Erreur : GOOGLE_API_KEY non trouvée dans le fichier .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

# 2. Création du dataset "Slide 7" (Exemple classique du cours)
# Dans ce dataset, on a A <-> D et B <-> C
data = {
    'A': [1, 1, 2, 2],
    'B': [1, 2, 1, 2],
    'C': [1, 2, 1, 2],
    'D': [1, 1, 2, 2]
}
df = pd.DataFrame(data)

# 3. Prompt pour l'IA (Simule src/agent/semantic_profiler.py)
prompt = f"""
Tu es un expert en Qualité des Données spécialisé dans les Pattern Functional Dependencies (PFD).
Voici un échantillon de données (format CSV) :
{df.to_csv(index=False)}

Tâche :
1. Analyse les relations entre les colonnes A, B, C et D.
2. Identifie les dépendances fonctionnelles (FD) évidentes.
3. Peux-tu prédire si une colonne détermine parfaitement une autre ?

Réponds de manière concise sous la forme : 'X -> Y'.
"""

# 4. Exécution du test
print("Test de l'agentique (Gemini)")
try:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    print("\nAnalyse :")
    print(response.text)

    if "A -> D" in response.text and "B -> C" in response.text:
        print("\nOK : dépendances identiques à l'algo classique.")
    else:
        print("\nRésultats différents de l'algo classique.")

except Exception as e:
    print(f"Erreur lors de l'appel API : {e}")