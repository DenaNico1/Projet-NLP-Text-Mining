"""
Utilitaires Streamlit - Chargement optimisé données
Projet NLP Text Mining - Master SISE

Utilise st.session_state pour charger 1 seule fois par session
"""

import streamlit as st
from config_db import load_offres_with_nlp

def get_data():
    """
    Charge données PostgreSQL 1 seule fois par session
    
    Après premier chargement (3-5 sec), accès instantané !
    Données persistent tant que navigateur ouvert.
    
    Returns:
        pd.DataFrame: Offres complètes avec NLP (38 colonnes)
    """
    if 'df_offres' not in st.session_state:
        # Premier chargement - afficher spinner
        with st.spinner("🔄 Chargement données PostgreSQL (première fois)..."):
            st.session_state.df_offres = load_offres_with_nlp()
            
        # Message succès (optionnel)
        if not st.session_state.df_offres.empty:
            st.toast(f"✅ {len(st.session_state.df_offres)} offres chargées !", icon="✅")
    
    return st.session_state.df_offres

def clear_cache():
    """
    Force rechargement données
    Utile après ajout nouvelle offre
    """
    if 'df_offres' in st.session_state:
        del st.session_state.df_offres
    st.cache_data.clear()
    st.rerun()

def get_data_info():
    """
    Retourne infos sur données en cache
    Utile pour debugging
    """
    if 'df_offres' in st.session_state:
        df = st.session_state.df_offres
        return {
            'loaded': True,
            'nb_rows': len(df),
            'nb_cols': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
    else:
        return {'loaded': False}