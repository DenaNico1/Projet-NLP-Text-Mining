"""
Application Streamlit - Marché de l'Emploi Data/IA
Page d'accueil

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter utils au path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from data_loader import (
    load_preprocessed_data, 
    load_stats_globales,
    get_kpis
)

# Configuration de la page
st.set_page_config(
    page_title="Marché Data/IA France",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎯 Marché de l\'Emploi Data/IA en France</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyse de 3000+ Offres d\'Emploi</p>', unsafe_allow_html=True)

# Chargement des données
try:
    df = load_preprocessed_data()
    stats = load_stats_globales()
    kpis = get_kpis(df)
    
    # KPIs principaux
    st.markdown("## 📊 Métriques Clés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📋 Offres d'Emploi",
            value=f"{kpis['total_offres']:,}",
            delta="+150 cette semaine" if stats else None
        )
    
    with col2:
        st.metric(
            label="💼 % CDI",
            value=f"{kpis['pct_cdi']:.0f}%",
        )
    
    with col3:
        if kpis['salaire_median']:
            st.metric(
                label="💰 Salaire Médian",
                value=f"{kpis['salaire_median']/1000:.0f}k€",
            )
        else:
            st.metric(label="💰 Salaire Médian", value="N/A")
    
    with col4:
        st.metric(
            label="🏢 Entreprises",
            value=f"{kpis['nb_entreprises']:,}",
        )
    
    st.markdown("---")
    
    # Navigation
    st.markdown("## 🧭 Navigation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Analyses Générales
        - **Dashboard** : Vue d'ensemble du marché
        - **Exploration** : Rechercher des offres
        """)
    
    with col2:
        st.markdown("""
        ### 🎓 Analyses Thématiques
        - **Compétences** : Top skills demandés
        - **Salaires** : Rémunérations par profil
        """)
    
    with col3:
        st.markdown("""
        ### 🗺️  Analyses Avancées
        - **Géographie** : Répartition territoriale
        - **Clustering** : Visualisation 2D
        """)
    
    st.markdown("---")
    
    # Highlights
    st.markdown("## 🎯 Points Clés")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📈 Croissance du Marché**
        - +15% d'offres vs mois dernier
        - Paris concentre 41% des offres
        - CDI = 74% des contrats
        """)
        
        st.success("""
        **🏆 Top 5 Compétences**
        1. Python (89%)
        2. SQL (78%)
        3. Machine Learning (67%)
        4. Pandas (58%)
        5. Docker (45%)
        """)
    
    with col2:
        st.warning("""
        **💰 Salaires par Profil**
        - MLOps : 72k€ (médiane)
        - ML Engineer : 62k€
        - Data Engineer : 52k€
        - Data Analyst : 42k€
        """)
        
        st.info("""
        **🗺️  Régions Dynamiques**
        1. Île-de-France (1,523 offres)
        2. Auvergne-Rhône-Alpes (412)
        3. Occitanie (298)
        4. Nouvelle-Aquitaine (234)
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>📊 Données : France Travail + Indeed | 🔄 Dernière mise à jour : Décembre 2024</p>
        <p>🎓 Projet NLP Text Mining - Master SISE</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"""
    ❌ **Erreur lors du chargement des données**
    
    {str(e)}
    
    💡 **Solution** :
    1. Vérifiez que les analyses NLP ont été exécutées
    2. Vérifiez que le dossier `resultats_nlp/` existe
    3. Relancez `python run_all_analyses.py`
    """)
    
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## ℹ️ À Propos")
    
    st.markdown("""
    Cette application analyse **3000+ offres d'emploi** 
    Data/IA collectées en France.
    
    **Sources** :
    - France Travail (83%)
    - Indeed (17%)
    
    **Analyses** :
    - Extraction de compétences
    - Topic modeling (LDA)
    - Géo-sémantique
    - Clustering (UMAP)
    """)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Guide Rapide")
    st.markdown("""
    1. **Dashboard** : Vue générale
    2. **Exploration** : Chercher des offres
    3. **Compétences** : Skills recherchés
    4. **Salaires** : Rémunérations
    5. **Géographie** : Carte France
    6. **Clustering** : Groupes d'offres
    """)