"""
Configuration Base de Données PostgreSQL (Supabase)
Projet NLP Text Mining - Master SISE

Connexion centralisée à l'entrepôt de données cloud
Modèle en étoile : 5 dimensions + 2 faits
"""

import os
import psycopg2
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

# Charger variables d'environnement
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-1-eu-north-1.pooler.supabase.com'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.znkulobexqmrshfkgynv'),
    'password': os.getenv('DB_PASSWORD', '')
}

# ============================================
# CONNEXION
# ============================================

@st.cache_resource
def get_db_connection():
    """
    Crée et cache une connexion PostgreSQL
    Utilisé par Streamlit pour éviter reconnexions multiples
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"❌ Erreur connexion PostgreSQL: {e}")
        st.info("Vérifiez votre fichier .env et credentials Supabase")
        return None

# ============================================
# CHARGEMENT DONNÉES (VUE COMPLÈTE)
# ============================================

@st.cache_data(ttl=300)  # Cache 5 minutes
def load_offres_complete():
    """
    Charge toutes les offres via la vue v_offres_complete
    (sans résultats NLP)
    
    Returns:
        pd.DataFrame: Offres avec toutes dimensions jointes
    """
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
    SELECT 
        o.offre_id,
        o.job_id_source,
        s.source_name as source,
        o.title,
        e.company_name,
        l.city,
        l.department,
        l.region,
        l.latitude,
        l.longitude,
        c.contract_type,
        c.experience_level,
        o.salary_min,
        o.salary_max,
        t.date_posted,
        o.description,
        o.url,
        o.scraped_at
    FROM fact_offres o
    LEFT JOIN dim_source s ON o.source_id = s.source_id
    LEFT JOIN dim_localisation l ON o.localisation_id = l.localisation_id
    LEFT JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
    LEFT JOIN dim_contrat c ON o.contrat_id = c.contrat_id
    LEFT JOIN dim_temps t ON o.temps_id = t.temps_id
    """
    
    try:
        df = pd.read_sql(query, conn)
        
        # Calculer salary_annual (moyenne min/max) pour compatibilité
        df['salary_annual'] = df[['salary_min', 'salary_max']].mean(axis=1)
        
        return df
    except Exception as e:
        st.error(f"Erreur chargement offres: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache 5 minutes
def load_offres_with_nlp():
    """
    ⭐ FONCTION PRINCIPALE APRÈS MIGRATION NLP COMPLÈTE
    
    Charge toutes les offres avec résultats NLP (profils, topics, clusters)
    Via la vue v_offres_nlp_complete
    
    Cette fonction REMPLACE l'ancien pickle data_with_profiles.pkl
    100% COMPATIBLE - Toutes colonnes pickle incluses
    
    Returns:
        pd.DataFrame: Offres complètes avec résultats NLP (38 colonnes)
            Colonnes BASE:
            - offre_id, title, company_name, city, region, etc.
            - salary_min, salary_max, salary_annual
            - latitude, longitude, contract_type, etc.
            
            Colonnes NLP PRINCIPALES:
            - status, profil_assigned, score_classification
            - competences_found, topic_id, cluster_id
            
            Colonnes NLP DÉTAILLÉES:
            - num_tokens, num_competences
            - profil_score, profil_confidence, profil_second, profil_second_score
            - score_title, score_description, score_competences
            - cascade_pass
            
            Colonnes PREPROCESSING:
            - description_clean, text_for_sklearn, tokens
            
            Colonnes SUPPLÉMENTAIRES:
            - duration, salary_text, source_name
    """
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    # Utiliser directement la vue v_offres_nlp_complete qui contient TOUTES les colonnes
    query = "SELECT * FROM v_offres_nlp_complete"
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Erreur chargement offres avec NLP: {e}")
        return pd.DataFrame()

# ============================================
# CHARGEMENT DONNÉES AVEC COMPÉTENCES
# ============================================

@st.cache_data(ttl=300)
def load_offres_with_competences():
    """
    Charge offres avec leurs compétences
    
    Returns:
        pd.DataFrame: Offres avec colonne 'competences_found' (liste)
    """
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
    SELECT 
        o.offre_id,
        o.title,
        e.company_name,
        l.city,
        l.region,
        s.source_name as source,
        ARRAY_AGG(DISTINCT c.skill_label) FILTER (WHERE c.skill_label IS NOT NULL) as competences_found
    FROM fact_offres o
    LEFT JOIN dim_source s ON o.source_id = s.source_id
    LEFT JOIN dim_localisation l ON o.localisation_id = l.localisation_id
    LEFT JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
    LEFT JOIN fact_competences c ON o.offre_id = c.offre_id
    GROUP BY o.offre_id, o.title, e.company_name, l.city, l.region, s.source_name
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Erreur chargement compétences: {e}")
        return pd.DataFrame()

# ============================================
# REQUÊTES SPÉCIFIQUES
# ============================================

def load_offres_by_region(region):
    """Charge offres d'une région spécifique"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
    SELECT o.*, l.region, e.company_name
    FROM fact_offres o
    JOIN dim_localisation l ON o.localisation_id = l.localisation_id
    JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
    WHERE l.region = %s
    """
    
    try:
        df = pd.read_sql(query, conn, params=(region,))
        return df
    except Exception as e:
        st.error(f"Erreur chargement région: {e}")
        return pd.DataFrame()

def load_stats_regions():
    """Charge statistiques par région (vue pré-calculée)"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = "SELECT * FROM v_stats_region"
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Erreur stats régions: {e}")
        return pd.DataFrame()

def get_top_competences(limit=20):
    """Top N compétences les plus demandées"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
    SELECT 
        skill_label,
        COUNT(*) as nb_offres
    FROM fact_competences
    GROUP BY skill_label
    ORDER BY nb_offres DESC
    LIMIT %s
    """
    
    try:
        df = pd.read_sql(query, conn, params=(limit,))
        return df
    except Exception as e:
        st.error(f"Erreur top compétences: {e}")
        return pd.DataFrame()

def get_top_entreprises(limit=20):
    """Top N entreprises qui recrutent le plus"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
    SELECT 
        e.company_name,
        COUNT(o.offre_id) as nb_offres
    FROM fact_offres o
    JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
    GROUP BY e.company_name
    ORDER BY nb_offres DESC
    LIMIT %s
    """
    
    try:
        df = pd.read_sql(query, conn, params=(limit,))
        return df
    except Exception as e:
        st.error(f"Erreur top entreprises: {e}")
        return pd.DataFrame()

# ============================================
# AJOUT NOUVELLE OFFRE (pour future implémentation)
# ============================================

def add_offre(offre_data):
    """
    Ajoute une nouvelle offre dans l'entrepôt
    
    Args:
        offre_data (dict): Données de l'offre
            {
                'job_id_source': str,
                'title': str,
                'company_name': str,
                'city': str,
                'region': str,
                'contract_type': str,
                'description': str,
                'url': str,
                'source': str,
                ...
            }
    
    Returns:
        int: ID de l'offre créée, ou None si erreur
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    cursor = conn.cursor()
    
    try:
        # 1. Récupérer ou créer source_id
        cursor.execute(
            "SELECT source_id FROM dim_source WHERE source_name = %s",
            (offre_data.get('source', 'Manuel'),)
        )
        result = cursor.fetchone()
        if result:
            source_id = result[0]
        else:
            cursor.execute(
                "INSERT INTO dim_source (source_name) VALUES (%s) RETURNING source_id",
                (offre_data.get('source', 'Manuel'),)
            )
            source_id = cursor.fetchone()[0]
        
        # 2. Récupérer ou créer entreprise_id
        cursor.execute(
            "SELECT entreprise_id FROM dim_entreprise WHERE company_name = %s",
            (offre_data.get('company_name', 'Non spécifié'),)
        )
        result = cursor.fetchone()
        if result:
            entreprise_id = result[0]
        else:
            cursor.execute(
                "INSERT INTO dim_entreprise (company_name) VALUES (%s) RETURNING entreprise_id",
                (offre_data.get('company_name', 'Non spécifié'),)
            )
            entreprise_id = cursor.fetchone()[0]
        
        # 3. Insérer offre (simplifié - à compléter avec autres dimensions)
        cursor.execute("""
            INSERT INTO fact_offres (
                source_id, entreprise_id, job_id_source, title, description, url, scraped_at
            ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
            RETURNING offre_id
        """, (
            source_id,
            entreprise_id,
            offre_data.get('job_id_source', ''),
            offre_data.get('title', ''),
            offre_data.get('description', ''),
            offre_data.get('url', '')
        ))
        
        offre_id = cursor.fetchone()[0]
        conn.commit()
        
        # Invalider cache Streamlit
        st.cache_data.clear()
        
        return offre_id
        
    except Exception as e:
        conn.rollback()
        st.error(f"Erreur ajout offre: {e}")
        return None
    finally:
        cursor.close()

# ============================================
# UTILITAIRES
# ============================================

def test_connection():
    """Teste la connexion PostgreSQL"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fact_offres")
            count = cursor.fetchone()[0]
            cursor.close()
            return True, f"✅ Connexion OK - {count} offres en base"
        except Exception as e:
            return False, f"❌ Erreur requête: {e}"
    else:
        return False, "❌ Connexion impossible"

def get_db_stats():
    """Statistiques globales base de données"""
    conn = get_db_connection()
    if not conn:
        return {}
    
    stats = {}
    tables = ['dim_source', 'dim_localisation', 'dim_entreprise', 
              'dim_contrat', 'dim_temps', 'fact_offres', 'fact_competences']
    
    try:
        cursor = conn.cursor()
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        cursor.close()
        return stats
    except Exception as e:
        st.error(f"Erreur stats: {e}")
        return {}