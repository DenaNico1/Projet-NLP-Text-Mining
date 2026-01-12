# # llm_extraction.py (VERSION FINALE CORRIGÉE)
# """
# Extraction LLM avec fallback
# CORRECTIONS :
# - timeout → timeout_ms (millisecondes)
# - Debug chargement clé API
# - Fallback automatique
# """

# import os
# import json
# import time
# from mistralai import Mistral
# from dotenv import load_dotenv

# # Charger .env
# load_dotenv()

# # ============================================
# # CONFIGURATION CLIENT
# # ============================================

# API_KEY = os.getenv("MISTRAL_API_KEY")

# # Debug chargement clé
# print(f"🔑 Chargement clé API Mistral...")
# if API_KEY:
#     print(f"   ✅ Clé trouvée : {API_KEY[:15]}...{API_KEY[-5:]}")
# else:
#     print(f"   ❌ Clé manquante dans .env")

# # Initialiser client
# if API_KEY:
#     try:
#         # CORRECTION : timeout_ms (millisecondes)
#         client = Mistral(api_key=API_KEY, timeout_ms=90000)  # 90 secondes = 90000 ms
#         print(f"   ✅ Client Mistral initialisé")
#     except Exception as e:
#         print(f"   ❌ Erreur init client : {e}")
#         client = None
# else:
#     client = None


# # ============================================
# # EXTRACTION LLM
# # ============================================


# def extract_job_info(raw_text, timeout_fallback=True):
#     """
#     Extraction avec fallback automatique
#     """

#     if not client:
#         print("⚠️ Client Mistral non disponible → Fallback extraction basique")
#         return {
#             "warning": "API Mistral non disponible, extraction basique utilisée",
#             **extract_fallback(raw_text),
#         }

#     system_prompt = """
#     Tu es un assistant expert en recrutement. Ta tâche est d'extraire les informations d'une offre d'emploi textuelle.
#     Réponds UNIQUEMENT au format JSON strict valide, sans balises markdown (pas de ```json ... ```), juste l'objet JSON brut.

#     Champs :
#      - title (titre du poste, chaine)
#      - company_name (nom de l'entreprise, chaine)
#      - city (ville, chaine ou "Non spécifié")
#      - region (région, chaine ou null)
#      - contract_type (CDI, CDD, Stage, Alternance, Freelance)
#      - salary_min (nombre entier ou null)
#      - salary_max (nombre entier ou null)
#      - description (description complète de l'offre en prenant compte tout ce qui est texte concernant l'offre, chaine)

#     """

#     user_prompt = f"Offre :\n\n{raw_text[:3700]}"

#     # 2 tentatives max
#     for attempt in range(2):
#         try:
#             print(f" Tentative {attempt + 1}/2 Mistral...")

#             # Appel API (timeout géré par client)
#             chat_response = client.chat.complete(
#                 model="open-mistral-7b",  # Gratuit, rapide
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 response_format={"type": "json_object"},
#             )

#             response_content = chat_response.choices[0].message.content

#             # Nettoyage
#             clean_json = response_content.strip()
#             if clean_json.startswith("```json"):
#                 clean_json = clean_json[7:]
#             if clean_json.endswith("```"):
#                 clean_json = clean_json[:-3]

#             data = json.loads(clean_json.strip())

#             print(f"   ✅ Extraction réussie")
#             return data

#         except json.JSONDecodeError:
#             print(f"   ❌ JSON invalide")
#             return {
#                 "warning": "JSON invalide, extraction basique utilisée",
#                 **extract_fallback(raw_text),
#             }

#         except Exception as e:
#             error_msg = str(e)
#             print(f"   ⚠️ Erreur : {error_msg[:80]}")

#             # Si dernière tentative
#             if attempt == 1:
#                 if timeout_fallback:
#                     print(f"   → Fallback extraction basique")
#                     return {
#                         "warning": f"Erreur Mistral, extraction basique utilisée",
#                         **extract_fallback(raw_text),
#                     }
#                 else:
#                     return {"error": f"Erreur Mistral : {error_msg}"}

#             # Attendre avant retry
#             time.sleep(3)

#     return {"error": "Échec extraction"}


# # ============================================
# # FALLBACK EXTRACTION BASIQUE
# # ============================================


# def extract_fallback(raw_text):
#     """
#     Extraction basique sans LLM (regex/parsing)
#     """

#     import re

#     data = {
#         "title": "Non spécifié",
#         "company_name": "Non spécifié",
#         "city": "Non spécifié",
#         "region": None,
#         "contract_type": "Non spécifié",
#         "salary_min": None,
#         "salary_max": None,
#         "description": raw_text[:500],
#     }

#     text_lower = raw_text.lower()
#     lines = raw_text.split("\n")

#     # === TITRE (première ligne significative) ===
#     for line in lines[:10]:
#         line = line.strip()
#         if 15 < len(line) < 150:
#             keywords = [
#                 "data",
#                 "engineer",
#                 "scientist",
#                 "analyst",
#                 "stage",
#                 "développeur",
#                 "ingénieur",
#                 "manager",
#                 "chef",
#             ]
#             if any(kw in line.lower() for kw in keywords):
#                 data["title"] = line
#                 break

#     # === ENTREPRISE (ligne après titre, ou "entreprise :") ===
#     for i, line in enumerate(lines[:15]):
#         if "entreprise" in line.lower() and ":" in line:
#             # Format "Entreprise : NomEntreprise"
#             parts = line.split(":", 1)
#             if len(parts) > 1:
#                 data["company_name"] = parts[1].strip()
#                 break
#         elif i > 0 and data["title"] != "Non spécifié":
#             # Ligne suivant le titre
#             line_clean = line.strip()
#             if 3 < len(line_clean) < 50 and not any(
#                 c.isdigit() for c in line_clean[:10]
#             ):
#                 data["company_name"] = line_clean
#                 break

#     # === CONTRAT ===
#     if "cdi" in text_lower:
#         data["contract_type"] = "CDI"
#     elif "cdd" in text_lower:
#         data["contract_type"] = "CDD"
#     elif "stage" in text_lower:
#         data["contract_type"] = "Stage"
#     elif "alternance" in text_lower:
#         data["contract_type"] = "Alternance"
#     elif "freelance" in text_lower or "indépendant" in text_lower:
#         data["contract_type"] = "Freelance"

#     # === VILLE ===
#     villes_fr = [
#         "Paris",
#         "Lyon",
#         "Marseille",
#         "Toulouse",
#         "Bordeaux",
#         "Lille",
#         "Nice",
#         "Nantes",
#         "Strasbourg",
#         "Montpellier",
#         "Rennes",
#         "Grenoble",
#         "Dijon",
#         "Angers",
#         "Reims",
#     ]

#     for ville in villes_fr:
#         if ville.lower() in text_lower:
#             data["city"] = ville
#             break

#     # === SALAIRE ===
#     # Format : "50K - 70K€" ou "50-70K€"
#     salary_patterns = [
#         r"(\d{2,3})\s*[kK]\s*-\s*(\d{2,3})\s*[kK]",
#         r"(\d{2,3})\s*[kK]€\s*-\s*(\d{2,3})\s*[kK]€",
#         r"(\d{2,3})\.?(\d{3})?\s*€\s*-\s*(\d{2,3})\.?(\d{3})?\s*€",
#     ]

#     for pattern in salary_patterns:
#         match = re.search(pattern, raw_text)
#         if match:
#             try:
#                 data["salary_min"] = int(match.group(1)) * 1000
#                 data["salary_max"] = int(match.group(2)) * 1000
#                 break
#             except:
#                 pass

#     return data


# # ============================================
# # TEST
# # ============================================

# if __name__ == "__main__":

#     test_text = """
#     Stage Data Engineer - 6 mois - Market Risks - F/H

#     Natixis

#     Localisation : Paris, Île-de-France
#     Type de contrat : Stage

#     Nous recherchons un stagiaire Data Engineer pour rejoindre notre équipe.

#     Missions principales :
#     - Développement pipelines de données
#     - Analyse et traitement données financières
#     - Collaboration équipes Data Science

#     Profil recherché :
#     - Formation Bac+5 en Data Science/Engineering
#     - Python, SQL, Spark
#     - Connaissance marchés financiers appréciée
#     """

#     print("=" * 60)
#     print("TEST EXTRACTION")
#     print("=" * 60)

#     result = extract_job_info(test_text, timeout_fallback=True)

#     print("\n Résultat :")
#     print(json.dumps(result, indent=2, ensure_ascii=False))

# -------------------------------------------------

# llm_extraction.py (VERSION DESCRIPTION COMPLÈTE)
"""
Extraction LLM avec fallback
CORRECTIONS :
- Augmentation limite texte pour description complète
- Gestion intelligente des tokens
- Fallback si texte trop long
"""

import os
import json
import time
from mistralai import Mistral
from dotenv import load_dotenv

# Charger .env
load_dotenv()

# ============================================
# CONFIGURATION CLIENT
# ============================================

API_KEY = os.getenv("MISTRAL_API_KEY")

# Debug chargement clé
print(f"🔑 Chargement clé API Mistral...")
if API_KEY:
    print(f"   ✅ Clé trouvée : {API_KEY[:15]}...{API_KEY[-5:]}")
else:
    print(f"   ❌ Clé manquante dans .env")

# Initialiser client
if API_KEY:
    try:
        client = Mistral(
            api_key=API_KEY, timeout_ms=120000
        )  # 120 secondes pour textes longs
        print(f"   ✅ Client Mistral initialisé")
    except Exception as e:
        print(f"   ❌ Erreur init client : {e}")
        client = None
else:
    client = None


# ============================================
# EXTRACTION LLM
# ============================================


def extract_job_info(raw_text, timeout_fallback=True, max_chars=12000):
    """
    Extraction avec gestion texte long

    Args:
        raw_text: Texte brut de l'offre
        timeout_fallback: Si True, utilise extraction basique en cas d'erreur
        max_chars: Nombre max de caractères à envoyer au LLM (défaut: 12000 ≈ 3000 tokens)
    """

    if not client:
        print("⚠️ Client Mistral non disponible → Fallback extraction basique")
        return {
            "warning": "API Mistral non disponible, extraction basique utilisée",
            **extract_fallback(raw_text),
        }

    # Tronquer intelligemment si nécessaire
    text_to_send = raw_text
    is_truncated = False

    if len(raw_text) > max_chars:
        print(
            f"   ⚠️ Texte long ({len(raw_text)} chars) → Troncature à {max_chars} chars"
        )
        text_to_send = raw_text[:max_chars]
        is_truncated = True

    system_prompt = """
    Tu es un assistant expert en recrutement. Ta tâche est d'extraire les informations d'une offre d'emploi textuelle.
    Réponds UNIQUEMENT au format JSON strict valide, sans balises markdown (pas de ```json ... ```), juste l'objet JSON brut.
    
    Champs à extraire :
     - title : titre du poste (string)
     - company_name : nom de l'entreprise (string)
     - city : ville (string ou "Non spécifié")
     - region : région française (string ou null)
     - contract_type : type de contrat (CDI, CDD, Stage, Alternance, Freelance)
     - salary_min : salaire minimum annuel en euros (integer ou null)
     - salary_max : salaire maximum annuel en euros (integer ou null)
     - description : description COMPLÈTE en TEXTE BRUT (pas de JSON, pas de structure)
         * Copie INTÉGRALEMENT tout le texte de l'offre
         * Garde le formatage texte (sauts de ligne, listes)
         * Ne structure PAS en JSON
         * Ne résume PAS
         * Prends TOUT : missions, compétences, formation, avantages, contexte, technologies
         * Format : texte naturel comme dans l'offre originale

    IMPORTANT pour la description : 
    - C'est un champ texte simple, PAS un objet JSON
    - Copie tout le contenu tel quel
    - Conserve les sauts de ligne avec \n
    - Ne transforme PAS en structure JSON
    """

    user_prompt = f"Offre d'emploi à analyser :\n\n{text_to_send}"

    # 2 tentatives max
    for attempt in range(2):
        try:
            print(f"🔄 Tentative {attempt + 1}/2 Mistral...")

            # Appel API
            chat_response = client.chat.complete(
                model="open-mistral-7b",  # Ou "mistral-small-latest" pour meilleure qualité
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=2000,  # Augmenté pour descriptions longues
            )

            response_content = chat_response.choices[0].message.content

            # Nettoyage
            clean_json = response_content.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]

            data = json.loads(clean_json.strip())

            # Ajouter info si troncature
            if is_truncated:
                data["_metadata"] = {
                    "truncated": True,
                    "original_length": len(raw_text),
                    "processed_length": len(text_to_send),
                }

            print(
                f"   ✅ Extraction réussie (description: {len(data.get('description', ''))} chars)"
            )
            return data

        except json.JSONDecodeError as e:
            print(f"   ❌ JSON invalide: {e}")
            return {
                "warning": "JSON invalide, extraction basique utilisée",
                **extract_fallback(raw_text),
            }

        except Exception as e:
            error_msg = str(e)
            print(f"   ⚠️ Erreur : {error_msg[:100]}")

            # Si dernière tentative
            if attempt == 1:
                if timeout_fallback:
                    print(f"   → Fallback extraction basique")
                    return {
                        "warning": f"Erreur Mistral, extraction basique utilisée",
                        **extract_fallback(raw_text),
                    }
                else:
                    return {"error": f"Erreur Mistral : {error_msg}"}

            # Attendre avant retry
            time.sleep(3)

    return {"error": "Échec extraction"}


# ============================================
# FALLBACK EXTRACTION BASIQUE
# ============================================


def extract_fallback(raw_text):
    """
    Extraction basique sans LLM (regex/parsing)
    Prend TOUTE la description
    """

    import re

    data = {
        "title": "Non spécifié",
        "company_name": "Non spécifié",
        "city": "Non spécifié",
        "region": None,
        "contract_type": "Non spécifié",
        "salary_min": None,
        "salary_max": None,
        "description": raw_text,  # DESCRIPTION COMPLÈTE
    }

    text_lower = raw_text.lower()
    lines = raw_text.split("\n")

    # === TITRE (première ligne significative) ===
    for line in lines[:10]:
        line = line.strip()
        if 15 < len(line) < 150:
            keywords = [
                "data",
                "engineer",
                "scientist",
                "analyst",
                "stage",
                "développeur",
                "ingénieur",
                "manager",
                "chef",
            ]
            if any(kw in line.lower() for kw in keywords):
                data["title"] = line
                break

    # === ENTREPRISE ===
    for i, line in enumerate(lines[:15]):
        if "entreprise" in line.lower() and ":" in line:
            parts = line.split(":", 1)
            if len(parts) > 1:
                data["company_name"] = parts[1].strip()
                break
        elif i > 0 and data["title"] != "Non spécifié":
            line_clean = line.strip()
            if 3 < len(line_clean) < 50 and not any(
                c.isdigit() for c in line_clean[:10]
            ):
                data["company_name"] = line_clean
                break

    # === CONTRAT ===
    if "cdi" in text_lower:
        data["contract_type"] = "CDI"
    elif "cdd" in text_lower:
        data["contract_type"] = "CDD"
    elif "stage" in text_lower:
        data["contract_type"] = "Stage"
    elif "alternance" in text_lower:
        data["contract_type"] = "Alternance"
    elif "freelance" in text_lower or "indépendant" in text_lower:
        data["contract_type"] = "Freelance"

    # === VILLE ===
    villes_fr = [
        "Paris",
        "Lyon",
        "Marseille",
        "Toulouse",
        "Bordeaux",
        "Lille",
        "Nice",
        "Nantes",
        "Strasbourg",
        "Montpellier",
        "Rennes",
        "Grenoble",
        "Dijon",
        "Angers",
        "Reims",
    ]

    for ville in villes_fr:
        if ville.lower() in text_lower:
            data["city"] = ville
            break

    # === SALAIRE ===
    salary_patterns = [
        r"(\d{2,3})\s*[kK]\s*-\s*(\d{2,3})\s*[kK]",
        r"(\d{2,3})\s*[kK]€\s*-\s*(\d{2,3})\s*[kK]€",
        r"(\d{2,3})\.?(\d{3})?\s*€\s*-\s*(\d{2,3})\.?(\d{3})?\s*€",
    ]

    for pattern in salary_patterns:
        match = re.search(pattern, raw_text)
        if match:
            try:
                data["salary_min"] = int(match.group(1)) * 1000
                data["salary_max"] = int(match.group(2)) * 1000
                break
            except:
                pass

    return data


# ============================================
# FONCTION UTILITAIRE
# ============================================


def estimate_tokens(text):
    """
    Estime le nombre de tokens (approximatif: 1 token ≈ 4 caractères)
    """
    return len(text) // 4


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    test_text = """
    Stage Data Engineer - 6 mois - Market Risks - F/H
    
    Natixis
    
    Localisation : Paris, Île-de-France
    Type de contrat : Stage
    
    Description complète du poste :
    
    Nous recherchons un stagiaire Data Engineer pour rejoindre notre équipe Market Risks.
    
    Missions principales :
    - Développement de pipelines de données en Python et Spark
    - Analyse et traitement de données financières volumineuses
    - Collaboration avec les équipes Data Science et Risk Management
    - Mise en place de tableaux de bord de suivi
    - Documentation technique des processus
    
    Profil recherché :
    - Formation Bac+5 en Data Science, Engineering, ou équivalent
    - Compétences techniques : Python, SQL, Spark, Git
    - Connaissance des marchés financiers appréciée
    - Anglais professionnel
    - Rigueur et esprit d'équipe
    
    Compétences techniques détaillées :
    - Python (pandas, numpy, scikit-learn)
    - SQL avancé (PostgreSQL, MySQL)
    - Big Data (Apache Spark, Hadoop)
    - Cloud (AWS ou Azure)
    - DevOps (Docker, CI/CD)
    - Visualisation (Tableau, PowerBI)
    
    Avantages :
    - Rémunération attractive
    - Tickets restaurant
    - Télétravail partiel possible
    - Locaux modernes en plein cœur de Paris
    - Accès à la formation continue
    """

    print("=" * 60)
    print("TEST EXTRACTION AVEC DESCRIPTION COMPLÈTE")
    print("=" * 60)
    print(f"Longueur texte input : {len(test_text)} caractères")
    print(f"Tokens estimés : ~{estimate_tokens(test_text)} tokens")
    print()

    result = extract_job_info(test_text, timeout_fallback=True, max_chars=15000)

    print("\n📊 Résultat :")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print(
        f"\n📏 Longueur description extraite : {len(result.get('description', ''))} caractères"
    )
