# llm_extraction.py (Version corrigée pour mistralai v1.0+)
import os
import json
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

# Configuration Client (Nouvelle syntaxe v1.0)
API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY) if API_KEY else None

def extract_job_info(raw_text):
    """
    Envoie le texte brut à Mistral et retourne un dictionnaire structuré.
    """
    if not client:
        return {"error": "Clé API Mistral manquante dans le fichier .env"}

    system_prompt = """
    Tu es un assistant expert en recrutement. Ta tâche est d'extraire les informations d'une offre d'emploi textuelle.
    Réponds UNIQUEMENT au format JSON strict valide, sans balises markdown (pas de ```json ... ```), juste l'objet JSON brut.
    
    Les champs requis sont :
    - title (titre du poste, chaine)
    - company_name (nom de l'entreprise, chaine)
    - city (ville, chaine ou "Non spécifié")
    - region (région, chaine ou null)
    - contract_type (CDI, CDD, Stage, Alternance, Freelance)
    - salary_min (nombre entier ou null)
    - salary_max (nombre entier ou null)
    - description (résumé succinct de 3 phrases max, chaine)
    """

    user_prompt = f"Voici l'offre à analyser :\n\n{raw_text}"

    try:
        # Appel API (Nouvelle syntaxe)
        chat_response = client.chat.complete(
            model="mistral-large-latest", # ou "open-mixtral-8x7b"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"} # Force le mode JSON
        )

        response_content = chat_response.choices[0].message.content
        
        # Nettoyage de sécurité au cas où le modèle ajoute quand même du markdown
        clean_json = response_content.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        
        data = json.loads(clean_json.strip())
        return data

    except json.JSONDecodeError:
        return {"error": "Le LLM n'a pas renvoyé un JSON valide."}
    except Exception as e:
        return {"error": f"Erreur API Mistral : {str(e)}"}