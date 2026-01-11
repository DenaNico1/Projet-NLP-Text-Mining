"""
DATA_LOADERS.PY - VERSION POSTGRESQL EMBEDDINGS
Charge embeddings depuis PostgreSQL au lieu de fichier local
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Import conditionnel de sentence_transformers (optionnel pour pages basiques)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from config import MODELS_DIR, RESULTS_DIR
from config_db import DB_CONFIG
import psycopg2

# ============================================
# CHARGEMENT EMBEDDINGS DEPUIS POSTGRESQL
# ============================================

@st.cache_data(ttl=3600)  # Cache 1h
def load_offres_with_embeddings():
    """
    Charge offres + embeddings depuis PostgreSQL
    
    OPTIMISATION:
    - 1 seule requête (JOIN)
    - Cache Streamlit (pas de rechargement à chaque matching)
    - Retourne DataFrame + embeddings numpy array
    
    NOTE: Crée sa propre connexion temporaire (évite conflit avec cache)
    """
    # Créer connexion temporaire (pas de cache_resource)
    conn = psycopg2.connect(**DB_CONFIG)
    
    try:
        # Requête combinée (performance)
        query = """
            SELECT 
                o.*,
                e.embedding
            FROM v_offres_nlp_complete o
            LEFT JOIN offres_embeddings e ON o.offre_id = e.offre_id
            ORDER BY o.offre_id
        """
        
        df = pd.read_sql(query, conn)
    finally:
        # Fermer connexion temporaire
        conn.close()
    
    # Extraire embeddings dans array numpy
    embeddings_list = []
    missing_embeddings = []
    
    for idx, row in df.iterrows():
        embedding_val = row['embedding']
        if embedding_val is not None:
            try:
                emb_array = np.array(embedding_val, dtype=np.float32)
                if emb_array.shape[0] > 0:
                    embeddings_list.append(emb_array)
                else:
                    embeddings_list.append(None)
                    missing_embeddings.append((idx, row['offre_id']))
            except Exception:
                embeddings_list.append(None)
                missing_embeddings.append((idx, row['offre_id']))
        else:
            embeddings_list.append(None)
            missing_embeddings.append((idx, row['offre_id']))
    
    # Convertir en numpy array (gère None)
    embeddings_array = np.array([emb if emb is not None else np.zeros(384) 
                                   for emb in embeddings_list], dtype=np.float32)
    
    # Warning si embeddings manquants
    if missing_embeddings:
        st.sidebar.warning(f"⚠️ {len(missing_embeddings)} offres sans embedding")
    
    # Supprimer colonne embedding du DataFrame (déjà dans array)
    df = df.drop(columns=['embedding'], errors='ignore')
    
    return df, embeddings_array

# ============================================
# CHARGEMENT AUTRES COMPOSANTS
# ============================================

@st.cache_resource
def load_matching_models():
    """Charge modèles ML"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        st.error("❌ sentence-transformers non disponible. Installez: pip install sentence-transformers")
        return None, None, None
    
    with open(MODELS_DIR / 'matching_model.pkl', 'rb') as f:
        system = pickle.load(f)
    
    rf_model = system['rf_model']
    tfidf_vec = system['tfidf_vectorizer']
    
    # Modèle embeddings
    emb_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    return rf_model, tfidf_vec, emb_model

@st.cache_data
def load_cv_base():
    """Charge CV fictifs"""
    with open(RESULTS_DIR / 'cv_base_fictifs.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_metrics():
    """Charge métriques modèle"""
    try:
        with open(RESULTS_DIR / 'matching_metrics.json', 'r') as f:
            return json.load(f)
    except:
        return {}

# ============================================
# FONCTION PRINCIPALE (COMPATIBLE ANCIEN CODE)
# ============================================

def load_matching_data():
    """
    Charge toutes données matching
    
    NOUVEAUTÉ: Embeddings depuis PostgreSQL
    COMPATIBILITÉ: Même signature que avant
    """
    # Charger offres + embeddings (PostgreSQL)
    df, embeddings = load_offres_with_embeddings()
    
    # Charger modèles
    rf_model, tfidf_vec, emb_model = load_matching_models()
    
    # Charger CV base
    cv_base = load_cv_base()
    
    # Charger métriques
    metrics = load_metrics()
    
    # Afficher statut
    st.sidebar.success(f"✅ {len(df)} offres + {len(embeddings)} embeddings (PostgreSQL)")
    
    return df, embeddings, rf_model, tfidf_vec, emb_model, cv_base, metrics

# ============================================
# FONCTION AJOUT OFFRE AVEC EMBEDDING AUTO
# ============================================

def ajouter_offre_avec_embedding(offre_data, emb_model=None):
    """
    Ajoute offre + calcule + stocke embedding automatiquement
    
    Usage dans fonctionnalité ajout d'offres de l'app
    
    Args:
        offre_data (dict): Données offre (title, description, etc.)
        emb_model: Modèle embeddings (si None, charge automatiquement)
    
    Returns:
        int: offre_id de l'offre ajoutée
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # 1. Insérer offre (retourne ID)
        cur.execute("""
            INSERT INTO fact_offres (
                source_id, localisation_id, entreprise_id, 
                contrat_id, temps_id, title, description, 
                url, salary_min, salary_max
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING offre_id
        """, (
            offre_data.get('source_id', 1),
            offre_data.get('localisation_id'),
            offre_data.get('entreprise_id'),
            offre_data.get('contrat_id', 1),
            offre_data.get('temps_id', 1),
            offre_data['title'],
            offre_data.get('description', ''),
            offre_data.get('url', ''),
            offre_data.get('salary_min'),
            offre_data.get('salary_max')
        ))
        
        offre_id = cur.fetchone()[0]
        
        # 2. Calculer embedding
        if emb_model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers non disponible")
            emb_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        text = f"{offre_data['title']} {offre_data.get('description', '')[:500]}"
        embedding = emb_model.encode(text).tolist()
        
        # 3. Stocker embedding
        cur.execute("""
            INSERT INTO offres_embeddings (offre_id, embedding)
            VALUES (%s, %s)
        """, (offre_id, embedding))
        
        conn.commit()
        
        # Invalider cache Streamlit
        load_offres_with_embeddings.clear()
        
        return offre_id
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# ============================================
# CHARGEMENT DONNÉES (SESSION STATE) ET MÉTRIQUES POUR INSIGHTS
# ============================================
def get_data():
    """
    Charge offres depuis PostgreSQL avec session state
    1 seul chargement par session - Instantané ensuite
    Returns:
        pd.DataFrame: Offres complètes (38 colonnes)
    """
    from config_db import load_offres_with_nlp
    if 'df_offres' not in st.session_state:
        with st.spinner("🔄 Chargement offres PostgreSQL..."):
            st.session_state.df_offres = load_offres_with_nlp()
    return st.session_state.df_offres

@st.cache_data
def load_clustering_metrics():
    """Charge métriques clustering"""
    try:
        with open(RESULTS_DIR / 'clustering_metrics.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        st.warning("⚠️ clustering_metrics.json non trouvé")
        return {}

@st.cache_data
def load_classification_quality():
    """Charge qualité classification"""
    try:
        with open(RESULTS_DIR / 'classification_quality.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        st.warning("⚠️ classification_quality.json non trouvé")
        return {}

@st.cache_data
def load_profils_distribution():
    """Charge distribution profils"""
    try:
        with open(RESULTS_DIR / 'profils_distribution.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        st.warning("⚠️ profils_distribution.json non trouvé")
        return {}

@st.cache_data
def load_topics_lda():
    """Charge topics LDA"""
    try:
        with open(RESULTS_DIR / 'topics_lda.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        st.warning("⚠️ topics_lda.json non trouvé")
        return None

def load_insights_data():
    """
    Charge données page Insights
    Retourne: (df, clustering_metrics, quality_metrics)
    """
    df = get_data()
    clustering = load_clustering_metrics()
    quality = load_classification_quality()
    return df, clustering, quality

def load_profils_data():
    """
    Charge données page Profils
    Retourne: (df, profils_stats)
    """
    df = get_data()
    profils_stats = load_profils_distribution()
    return df, profils_stats

def load_topics_data():
    """
    Charge données page Topics
    Retourne: (df, topics)
    """
    df = get_data()
    topics = load_topics_lda()
    return df, topics