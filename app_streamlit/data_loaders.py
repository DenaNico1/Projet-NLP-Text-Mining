# """
# DATA_LOADERS.PY - VERSION POSTGRESQL EMBEDDINGS
# Charge embeddings depuis PostgreSQL au lieu de fichier local
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import json
# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# from nlp_pipeline_wrapper import process_single_offre


# from config import MODELS_DIR, RESULTS_DIR
# from config_db import get_db_connection


# # ============================================
# # CHARGEMENT EMBEDDINGS DEPUIS POSTGRESQL
# # ============================================
# @st.cache_resource
# @st.cache_data(ttl=3600)  # Cache 1h
# def load_offres_with_embeddings():
#     """
#     Charge offres + embeddings depuis PostgreSQL

#     OPTIMISATION:
#     - 1 seule requête (JOIN)
#     - Cache Streamlit (pas de rechargement à chaque matching)
#     - Retourne DataFrame + embeddings numpy array
#     """
#     conn = get_db_connection()

#     # Requête combinée (performance)
#     query = """
#         SELECT
#             o.*,
#             e.embedding
#         FROM v_offres_nlp_complete o
#         LEFT JOIN offres_embeddings e ON o.offre_id = e.offre_id
#         ORDER BY o.offre_id
#     """

#     df = pd.read_sql(query, conn)
#     conn.close()

#     # Extraire embeddings dans array numpy
#     embeddings_list = []
#     missing_embeddings = []

#     for idx, row in df.iterrows():
#         embedding_val = row["embedding"]
#         if embedding_val is not None:
#             try:
#                 emb_array = np.array(embedding_val, dtype=np.float32)
#                 if emb_array.shape[0] > 0:
#                     embeddings_list.append(emb_array)
#                 else:
#                     embeddings_list.append(None)
#                     missing_embeddings.append((idx, row["offre_id"]))
#             except Exception as e:
#                 embeddings_list.append(None)
#                 missing_embeddings.append((idx, row["offre_id"]))
#         else:
#             embeddings_list.append(None)
#             missing_embeddings.append((idx, row["offre_id"]))

#     # Convertir en numpy array (gère None)
#     embeddings_array = np.array(
#         [emb if emb is not None else np.zeros(384) for emb in embeddings_list],
#         dtype=np.float32,
#     )

#     # Warning si embeddings manquants
#     if missing_embeddings:
#         st.sidebar.warning(f"⚠️ {len(missing_embeddings)} offres sans embedding")

#     # Supprimer colonne embedding du DataFrame (déjà dans array)
#     df = df.drop(columns=["embedding"], errors="ignore")

#     return df, embeddings_array


# # ============================================
# # CHARGEMENT AUTRES COMPOSANTS
# # ============================================


# @st.cache_resource
# def load_matching_models():
#     """Charge modèles ML"""
#     with open(MODELS_DIR / "matching_model.pkl", "rb") as f:
#         system = pickle.load(f)

#     rf_model = system["rf_model"]
#     tfidf_vec = system["tfidf_vectorizer"]

#     # Modèle embeddings
#     emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

#     return rf_model, tfidf_vec, emb_model


# @st.cache_data
# def load_cv_base():
#     """Charge CV fictifs"""
#     with open(RESULTS_DIR / "cv_base_fictifs.json", "r", encoding="utf-8") as f:
#         return json.load(f)


# @st.cache_data
# def load_metrics():
#     """Charge métriques modèle"""
#     try:
#         with open(RESULTS_DIR / "matching_metrics.json", "r") as f:
#             return json.load(f)
#     except:
#         return {}


# @st.cache_data(ttl=300)
# def load_profils_data():
#     """
#     Charge données pour page Profils

#     Returns:
#         tuple: (df_offres, profils_stats)
#     """
#     from config_db import load_offres_with_nlp

#     df = load_offres_with_nlp()

#     # Calculer stats profils
#     df_class = df[df["status"] == "classified"]

#     profils_stats = {
#         "distribution": df_class["profil_assigned"].value_counts().to_dict(),
#         "total_classified": len(df_class),
#         "total_unclassified": len(df[df["status"] == "unclassified"]),
#         "classification_rate": len(df_class) / len(df) * 100,
#     }

#     return df, profils_stats


# @st.cache_data(ttl=300)
# def load_topics_lda():
#     """Charge topics LDA"""
#     try:
#         with open(RESULTS_DIR / "topics_lda.json", "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception:
#         st.warning("⚠️ topics_lda.json non trouvé")
#         return None


# @st.cache_data(ttl=300)
# def load_topics_data():
#     """
#     Charge données pour page Topics

#     Returns:
#         tuple: (df_offres, topics_data)
#     """
#     from config_db import load_offres_with_nlp
#     import json
#     from pathlib import Path

#     df = load_offres_with_nlp()

#     topics = load_topics_lda()
#     return df, topics


# # ============================================
# # FONCTION PRINCIPALE (COMPATIBLE ANCIEN CODE)
# # ============================================


# def load_matching_data():
#     """
#     Charge toutes données matching

#     NOUVEAUTÉ: Embeddings depuis PostgreSQL
#     COMPATIBILITÉ: Même signature que avant
#     """
#     # Charger offres + embeddings (PostgreSQL)
#     df, embeddings = load_offres_with_embeddings()

#     # Charger modèles
#     rf_model, tfidf_vec, emb_model = load_matching_models()

#     # Charger CV base
#     cv_base = load_cv_base()

#     # Charger métriques
#     metrics = load_metrics()

#     # Afficher statut
#     st.sidebar.success(
#         f"✅ {len(df)} offres + {len(embeddings)} embeddings (PostgreSQL)"
#     )

#     return df, embeddings, rf_model, tfidf_vec, emb_model, cv_base, metrics


# def ajouter_offre_avec_embedding(offre_data, emb_model=None):
#     """
#     Ajoute offre + calcule + stocke embedding automatiquement

#     VERSION DEBUG : Affiche état connexion à chaque étape
#     """

#     conn = None

#     try:
#         # ============================================
#         # DEBUG : CONNEXION
#         # ============================================

#         print(" DEBUG: Création connexion...")
#         conn = get_db_connection()
#         print(f" DEBUG: Connexion créée, closed={conn.closed}")

#         # VÉRIFICATION : Connexion vivante ?
#         if conn.closed:
#             raise Exception("Connexion fermée immédiatement après création !")

#         # Test rapide connexion
#         cur = conn.cursor()
#         cur.execute("SELECT 1")
#         cur.fetchone()
#         print(" DEBUG: Test SELECT 1 OK")

#         # ============================================
#         # 1. MAPPER DIMENSIONS
#         # ============================================

#         print(" DEBUG: Mapping source...")
#         source_name = offre_data.get("source", "Import IA")
#         if "indeed" in source_name.lower():
#             source_id = 1
#         elif "france" in source_name.lower() or "travail" in source_name.lower():
#             source_id = 2
#         else:
#             source_id = 3

#         # Localisation
#         print(" DEBUG: Mapping localisation...")
#         city = offre_data.get("city", "Non spécifié")
#         cur.execute(
#             """
#             SELECT localisation_id FROM dim_localisation
#             WHERE LOWER(city) = LOWER(%s)
#             LIMIT 1
#         """,
#             (city,),
#         )

#         loc_result = cur.fetchone()
#         if loc_result:
#             localisation_id = loc_result[0]
#             print(f" DEBUG: Localisation trouvée ID={localisation_id}")
#         else:
#             print(f" DEBUG: Création nouvelle localisation {city}")
#             cur.execute(
#                 """
#                 INSERT INTO dim_localisation (city, region)
#                 VALUES (%s, 'Non spécifié')
#                 RETURNING localisation_id
#             """,
#                 (city,),
#             )
#             localisation_id = cur.fetchone()[0]
#             print(f" DEBUG: Localisation créée ID={localisation_id}")

#         # Entreprise
#         print(" DEBUG: Mapping entreprise...")
#         company = offre_data.get("company_name", "Non spécifié")
#         cur.execute(
#             """
#             SELECT entreprise_id FROM dim_entreprise
#             WHERE LOWER(company_name) = LOWER(%s)
#             LIMIT 1
#         """,
#             (company,),
#         )

#         ent_result = cur.fetchone()
#         if ent_result:
#             entreprise_id = ent_result[0]
#             print(f" DEBUG: Entreprise trouvée ID={entreprise_id}")
#         else:
#             print(f" DEBUG: Création nouvelle entreprise {company}")
#             cur.execute(
#                 """
#                 INSERT INTO dim_entreprise (company_name)
#                 VALUES (%s)
#                 RETURNING entreprise_id
#             """,
#                 (company,),
#             )
#             entreprise_id = cur.fetchone()[0]
#             print(f" DEBUG: Entreprise créée ID={entreprise_id}")

#         # Contrat
#         contract_mapping = {
#             "CDI": 1,
#             "CDD": 2,
#             "Stage": 3,
#             "Alternance": 4,
#             "Freelance": 5,
#         }
#         contract_type = offre_data.get("contract_type", "Non spécifié")
#         contrat_id = contract_mapping.get(contract_type, 6)
#         temps_id = 1

#         print(
#             f" DEBUG: IDs mappés - source={source_id}, loc={localisation_id}, "
#             f"ent={entreprise_id}, contrat={contrat_id}"
#         )

#         # ============================================
#         # 2. INSÉRER OFFRE
#         # ============================================

#         print(" DEBUG: Insertion fact_offres...")
#         job_id_source = offre_data.get("external_job_id", "")
#         if not job_id_source and "job_info" in offre_data:
#             job_id_source = offre_data["job_info"].get("job_id", "")

#         # Fallback : Générer ID unique
#         if not job_id_source:
#             import uuid

#             job_id_source = f"manual_{uuid.uuid4().hex[:12]}"
#             print(f" DEBUG: Pas d'external_job_id, ID généré: {job_id_source}")

#         # CONVERTIR description (liste → string)
#         description = offre_data.get("description", "")
#         if isinstance(description, list):
#             description = " ".join(description)
#             print(
#                 f" DEBUG: Description convertie liste→string ({len(description)} chars)"
#             )

#         print(f" DEBUG: job_id_source={job_id_source}")
#         print(f" DEBUG: title={offre_data['title'][:50]}...")

#         print(" DEBUG: Insertion fact_offres...")
#         cur.execute(
#             """
#             INSERT INTO fact_offres (
#                 source_id,
#                 localisation_id,
#                 entreprise_id,
#                 contrat_id,
#                 temps_id,
#                 job_id_source,
#                 title,
#                 description,
#                 url,
#                 salary_min,
#                 salary_max,
#                 created_at,
#                 scraped_at
#             )
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
#             RETURNING offre_id
#         """,
#             (
#                 source_id,
#                 localisation_id,
#                 entreprise_id,
#                 contrat_id,
#                 temps_id,
#                 job_id_source,
#                 offre_data["title"],
#                 description,
#                 offre_data.get("url", ""),
#                 offre_data.get("salary_min"),
#                 offre_data.get("salary_max"),
#             ),
#         )

#         offre_id = cur.fetchone()[0]
#         print(f" DEBUG: Offre insérée ID={offre_id}")

#         # ============================================
#         # 3. EXTERNAL JOB ID
#         # ============================================

#         if offre_data.get("external_source") and offre_data.get("external_job_id"):
#             print(f" DEBUG: Insertion external_job_id...")
#             cur.execute(
#                 """
#                 INSERT INTO fact_offres_external (
#                     offre_id, external_source, external_job_id
#                 )
#                 VALUES (%s, %s, %s)
#             """,
#                 (
#                     offre_id,
#                     offre_data["external_source"],
#                     offre_data["external_job_id"],
#                 ),
#             )
#             print(" DEBUG: External ID inséré")

#         # ============================================
#         # 4. EMBEDDING
#         # ============================================

#         print(" DEBUG: Calcul embedding...")
#         if emb_model is None:
#             from sentence_transformers import SentenceTransformer

#             emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

#         text = f"{offre_data['title']} {offre_data.get('description', '')[:500]}"
#         embedding = emb_model.encode(text).tolist()
#         print(f" DEBUG: Embedding calculé (dim={len(embedding)})")

#         print(" DEBUG: Insertion embedding...")
#         cur.execute(
#             """
#             INSERT INTO offres_embeddings (offre_id, embedding)
#             VALUES (%s, %s)
#         """,
#             (offre_id, embedding),
#         )
#         print(" DEBUG: Embedding inséré")

#         # Pipeline NLP
#         try:
#             nlp_results = process_single_offre(
#                 offre_id=offre_id,
#                 title=offre_data["title"],
#                 description=description,
#                 conn=conn,
#             )
#             print(
#                 f" NLP: {nlp_results['profil']}, {nlp_results['num_competences']} compétences"
#             )
#         except Exception as e:
#             print(f" NLP erreur (non bloquante): {e}")

#         # ============================================
#         # 5. COMMIT
#         # ============================================

#         print(" DEBUG: Commit transaction...")
#         conn.commit()
#         print(" DEBUG: Commit OK")

#         return offre_id

#     except Exception as e:
#         print(f" DEBUG: ERREUR - {type(e).__name__}: {str(e)}")
#         print(f" DEBUG: Connexion closed={conn.closed if conn else 'None'}")

#         # Rollback si connexion vivante
#         if conn and not conn.closed:
#             try:
#                 print(" DEBUG: Rollback...")
#                 conn.rollback()
#                 print(" DEBUG: Rollback OK")
#             except Exception as rb_err:
#                 print(f" DEBUG: Erreur rollback - {rb_err}")

#         raise Exception(f"Erreur insertion offre : {str(e)}")

#     finally:
#         print(" DEBUG: Finally - nettoyage connexion...")
#         if conn:
#             try:
#                 if not conn.closed:
#                     print(" DEBUG: Fermeture connexion...")
#                     conn.close()
#                     print(" DEBUG: Connexion fermée")
#                 else:
#                     print(" DEBUG: Connexion déjà fermée")
#             except Exception as close_err:
#                 print(f" DEBUG: Erreur fermeture - {close_err}")


"""
DATA_LOADERS.PY - VERSION POSTGRESQL EMBEDDINGS
Version optimisée performance (colonnes conservées)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sentence_transformers import SentenceTransformer

from config import MODELS_DIR, RESULTS_DIR
from config_db import get_db_connection
from nlp_pipeline_wrapper import process_single_offre


# ============================================
# CHARGEMENT OFFRES + EMBEDDINGS (OPTIMISÉ)
# ============================================


@st.cache_data(ttl=3600, show_spinner="Chargement des offres...")
def load_offres_with_embeddings():
    """
    Charge offres + embeddings depuis PostgreSQL
    OPTIMISATIONS :
    - cache unique
    - pas de iterrows
    - conversion numpy vectorisée
    """
    conn = get_db_connection()

    query = """
        SELECT 
            o.*,
            e.embedding
        FROM v_offres_nlp_complete o
        LEFT JOIN offres_embeddings e ON o.offre_id = e.offre_id
        ORDER BY o.offre_id
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # === Conversion embeddings (SANS iterrows) ===
    embedding_series = df["embedding"]

    dim = 384
    embeddings_array = np.zeros((len(df), dim), dtype=np.float32)

    missing_mask = embedding_series.isna()

    # Convertir uniquement les embeddings existants
    valid_embeddings = embedding_series[~missing_mask].values
    if len(valid_embeddings) > 0:
        embeddings_array[~missing_mask] = np.vstack(
            [np.asarray(e, dtype=np.float32) for e in valid_embeddings]
        )

    if missing_mask.any():
        st.sidebar.warning(f"⚠️ {missing_mask.sum()} offres sans embedding")

    # Supprimer colonne embedding du DataFrame
    df = df.drop(columns=["embedding"], errors="ignore")

    return df, embeddings_array


# ============================================
# MODÈLES ML (CHARGÉS UNE SEULE FOIS)
# ============================================


@st.cache_resource(show_spinner="Chargement des modèles ML...")
def load_matching_models():
    with open(MODELS_DIR / "matching_model.pkl", "rb") as f:
        system = pickle.load(f)

    rf_model = system["rf_model"]
    tfidf_vec = system["tfidf_vectorizer"]

    emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    return rf_model, tfidf_vec, emb_model


# ============================================
# DONNÉES ANNEXES (CACHE DATA)
# ============================================


@st.cache_data
def load_cv_base():
    with open(RESULTS_DIR / "cv_base_fictifs.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    try:
        with open(RESULTS_DIR / "matching_metrics.json", "r") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_profils_data():
    from config_db import load_offres_with_nlp

    df = load_offres_with_nlp()

    df_class = df[df["status"] == "classified"]

    profils_stats = {
        "distribution": df_class["profil_assigned"].value_counts().to_dict(),
        "total_classified": len(df_class),
        "total_unclassified": len(df[df["status"] == "unclassified"]),
        "classification_rate": len(df_class) / len(df) * 100 if len(df) else 0,
    }

    return df, profils_stats


@st.cache_data(ttl=300)
def load_topics_lda():
    try:
        with open(RESULTS_DIR / "topics_lda.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        st.warning("⚠️ topics_lda.json non trouvé")
        return None


@st.cache_data(ttl=300)
def load_topics_data():
    from config_db import load_offres_with_nlp

    df = load_offres_with_nlp()
    topics = load_topics_lda()
    return df, topics


# ============================================
# CHARGEMENT GLOBAL MATCHING
# ============================================


def load_matching_data():
    """
    Charge toutes les données nécessaires au matching
    """
    df, embeddings = load_offres_with_embeddings()
    rf_model, tfidf_vec, emb_model = load_matching_models()
    cv_base = load_cv_base()
    metrics = load_metrics()

    st.sidebar.success(
        f"✅ {len(df)} offres chargées ({embeddings.shape[0]} embeddings)"
    )

    return df, embeddings, rf_model, tfidf_vec, emb_model, cv_base, metrics


# ============================================
# INSERTION OFFRE + EMBEDDING (OPTIMISÉ)
# ============================================


def ajouter_offre_avec_embedding(offre_data, emb_model):
    """
    Ajoute une offre + embedding + NLP
    Version nettoyée (sans prints bloquants)
    """

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # === Mapping source ===
        source_name = offre_data.get("source", "Import IA").lower()
        source_id = (
            1 if "indeed" in source_name else 2 if "france" in source_name else 3
        )

        # === Localisation ===
        city = offre_data.get("city", "Non spécifié")
        cur.execute(
            "SELECT localisation_id FROM dim_localisation WHERE LOWER(city)=LOWER(%s) LIMIT 1",
            (city,),
        )
        row = cur.fetchone()
        if row:
            localisation_id = row[0]
        else:
            cur.execute(
                "INSERT INTO dim_localisation (city, region) VALUES (%s, 'Non spécifié') RETURNING localisation_id",
                (city,),
            )
            localisation_id = cur.fetchone()[0]

        # === Entreprise ===
        company = offre_data.get("company_name", "Non spécifié")
        cur.execute(
            "SELECT entreprise_id FROM dim_entreprise WHERE LOWER(company_name)=LOWER(%s) LIMIT 1",
            (company,),
        )
        row = cur.fetchone()
        if row:
            entreprise_id = row[0]
        else:
            cur.execute(
                "INSERT INTO dim_entreprise (company_name) VALUES (%s) RETURNING entreprise_id",
                (company,),
            )
            entreprise_id = cur.fetchone()[0]

        contract_mapping = {
            "CDI": 1,
            "CDD": 2,
            "Stage": 3,
            "Alternance": 4,
            "Freelance": 5,
        }
        contrat_id = contract_mapping.get(offre_data.get("contract_type"), 6)

        description = offre_data.get("description", "")
        if isinstance(description, list):
            description = " ".join(description)

        # === Insertion offre ===
        cur.execute(
            """
            INSERT INTO fact_offres (
                source_id, localisation_id, entreprise_id, contrat_id, temps_id,
                job_id_source, title, description, url,
                salary_min, salary_max, created_at, scraped_at
            )
            VALUES (%s,%s,%s,%s,1,%s,%s,%s,%s,%s,%s,NOW(),NOW())
            RETURNING offre_id
            """,
            (
                source_id,
                localisation_id,
                entreprise_id,
                contrat_id,
                offre_data.get("external_job_id", ""),
                offre_data["title"],
                description,
                offre_data.get("url", ""),
                offre_data.get("salary_min"),
                offre_data.get("salary_max"),
            ),
        )

        offre_id = cur.fetchone()[0]

        # === Embedding ===
        text = f"{offre_data['title']} {description[:500]}"
        embedding = emb_model.encode(text).astype(np.float32).tolist()

        cur.execute(
            "INSERT INTO offres_embeddings (offre_id, embedding) VALUES (%s,%s)",
            (offre_id, embedding),
        )

        # === NLP ===
        process_single_offre(
            offre_id=offre_id,
            title=offre_data["title"],
            description=description,
            conn=conn,
        )

        conn.commit()
        return offre_id

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()
