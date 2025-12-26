"""
Utilitaires - Chargement des Données
Charge les résultats des analyses NLP pour Streamlit

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import streamlit as st
import pandas as pd
import pickle
import json
from pathlib import Path
import duckdb


@st.cache_data
def load_preprocessed_data():
    """
    Charge les données preprocessées
    
    Returns:
        DataFrame
    """
    path = Path("../resultats_nlp/models/data_with_topics.pkl")
    
    if not path.exists():
        # Fallback
        path = Path("../resultats_nlp/models/data_preprocessed.pkl")
    
    with open(path, 'rb') as f:
        df = pickle.load(f)
    
    return df


@st.cache_data
def load_competences_data():
    """
    Charge les résultats d'extraction de compétences
    
    Returns:
        dict
    """
    path = Path("../resultats_nlp/competences_extracted.json")
    
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None


@st.cache_data
def load_topics_data():
    """
    Charge les résultats du topic modeling
    
    Returns:
        dict
    """
    path = Path("../resultats_nlp/topics_lda.json")
    
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None


@st.cache_data
def load_geo_data():
    """
    Charge les résultats géo-sémantiques
    
    Returns:
        dict
    """
    path = Path("../resultats_nlp/analyse_geo_semantique.json")
    
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None


@st.cache_data
def load_salaires_data():
    """
    Charge les résultats stacks × salaires
    
    Returns:
        dict
    """
    path = Path("../resultats_nlp/stacks_salaires.json")
    
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None


@st.cache_data
def load_stats_globales():
    """
    Charge les statistiques globales
    
    Returns:
        dict
    """
    path = Path("../resultats_nlp/stats_globales.json")
    
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    return None


def filter_dataframe(df, filters):
    """
    Filtre le DataFrame selon les critères
    
    Args:
        df: DataFrame
        filters: dict de filtres
        
    Returns:
        DataFrame filtré
    """
    df_filtered = df.copy()
    
    # Filtre par source
    if 'sources' in filters and filters['sources']:
        df_filtered = df_filtered[df_filtered['source_name'].isin(filters['sources'])]
    
    # Filtre par région
    if 'regions' in filters and filters['regions']:
        df_filtered = df_filtered[df_filtered['region'].isin(filters['regions'])]
    
    # Filtre par type de contrat
    if 'contrats' in filters and filters['contrats']:
        df_filtered = df_filtered[df_filtered['contract_type'].isin(filters['contrats'])]
    
    # Filtre par salaire
    if 'salaire_min' in filters and 'salaire_max' in filters:
        mask = (
            (df_filtered['salary_annual'] >= filters['salaire_min']) & 
            (df_filtered['salary_annual'] <= filters['salaire_max'])
        )
        df_filtered = df_filtered[mask | df_filtered['salary_annual'].isna()]
    
    # Recherche textuelle
    if 'search' in filters and filters['search']:
        search_term = filters['search'].lower()
        mask = (
            df_filtered['title'].str.lower().str.contains(search_term, na=False) |
            df_filtered['description'].str.lower().str.contains(search_term, na=False) |
            df_filtered['company_name'].str.lower().str.contains(search_term, na=False)
        )
        df_filtered = df_filtered[mask]
    
    return df_filtered


def get_kpis(df):
    """
    Calcule les KPIs principaux
    
    Args:
        df: DataFrame
        
    Returns:
        dict de KPIs
    """
    return {
        'total_offres': len(df),
        'nb_entreprises': df['company_name'].nunique(),
        'nb_regions': df['region'].nunique(),
        'nb_villes': df['city'].nunique(),
        'pct_cdi': (df['contract_type'] == 'CDI').sum() / len(df) * 100 if len(df) > 0 else 0,
        'salaire_median': df['salary_annual'].median(),
        'salaire_moyen': df['salary_annual'].mean(),
        'avec_salaire': df['salary_annual'].notna().sum(),
    }