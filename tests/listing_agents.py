import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Modèles disponibles pour votre clé :")
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(f"- {m.name}")
