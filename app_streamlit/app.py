"""
DataJobs Explorer - Application Streamlit Premium
Analyse NLP du Marché Data/IA en France

Master SISE - Projet NLP Text Mining
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter dossier parent au path
sys.path.insert(0, str(Path(__file__).parent))

# Configuration page (DOIT être la première commande Streamlit)
st.set_page_config(
    page_title="DataJobs Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Analyse NLP de 3,003 offres d'emploi Data/IA en France"
    }
)

# CSS Custom Premium
st.markdown("""
<style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);
        padding-left: 0;
        padding-right: 0;
    }
    
    [data-testid="stAppViewContainer"] {
    padding-left: 0;
    padding-right: 0;
    }

    [data-testid="stMainBlockContainer"] {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 10px 0;
    }
    
    /* Titres */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    h2, h3 {
        color: #667eea;
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1f2937;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: #1f2937;
        border-radius: 8px;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background: #111827;
        padding: 10px;
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        border-top: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR NAVIGATION
# ============================================

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 2.5rem; margin: 0;'></h1>
        <h2 style='margin: 10px 0; font-size: 1.5rem;'>DataJobs Explorer</h2>
        <p style='color: #9ca3af; font-size: 0.9rem;'>Analyse NLP Marché Data/IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("###  Navigation")
    
    page = st.radio(
        "Choisir une page",
        [
            " Dashboard",
            " Exploration Géographique",
            " Profils Métiers",
            " Compétences",
            " Topics & Tendances",
            " Visualisations 3D",
            " Insights Avancés",
            " Matching CV ↔ Offres"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Filtres globaux
    st.markdown("###  Filtres Globaux")
    
    filter_source = st.selectbox(
        "Source",
        ['Toutes', 'France Travail', 'Indeed']
    )
    
    filter_region = st.selectbox(
        "Région",
        ['Toutes', 'Île-de-France', 'Auvergne-Rhône-Alpes', 
         'Nouvelle-Aquitaine', 'Occitanie', 'Provence-Alpes-Côte d\'Azur']
    )
    
    st.markdown("---")
    
    # Stats rapides
    st.markdown("###  Stats Rapides")
    st.metric("Offres", "3,003", help="Total offres analysées")
    st.metric("Classifiées", "56.2%", help="Taux de classification")
    st.metric("Compétences", "158", help="Compétences uniques")
    st.metric("CVs Base", "25", help="CVs démo matching")
    
    st.markdown("---")
    
    # Info projet
    with st.expander("ℹ À propos"):
        st.markdown("""
        **Projet NLP Text Mining**
        
        Master SISE - Décembre 2025
        
        Analyse approfondie de 3,003 offres d'emploi Data/IA collectées en France.
        
        **Techniques utilisées:**
        - Web Scraping (France Travail + Indeed)
        - Classification hybride (56.2%)
        - Embeddings multilingues
        - Topic Modeling (LDA)
        - Clustering (KMeans, HDBSCAN)
        """)

# ============================================
# ROUTING PAGES
# ============================================

# Stocker filtres en session state
if 'filters' not in st.session_state:
    st.session_state.filters = {}

st.session_state.filters = {
    'source': filter_source,
    'region': filter_region
}

# Router vers pages
if page == " Dashboard":
    exec(open(Path(__file__).parent / "pages" / "dashboard.py", encoding='utf-8').read())

elif page == " Exploration Géographique":
    exec(open(Path(__file__).parent / "pages" / "geographique.py", encoding='utf-8').read())

elif page == " Profils Métiers":
    exec(open(Path(__file__).parent / "pages" / "profils.py", encoding='utf-8').read())

elif page == " Compétences":
    exec(open(Path(__file__).parent / "pages" / "competences.py", encoding='utf-8').read())

elif page == " Topics & Tendances":
    exec(open(Path(__file__).parent / "pages" / "topics.py", encoding='utf-8').read())

elif page == " Visualisations 3D":
    exec(open(Path(__file__).parent / "pages" / "viz_3d.py", encoding='utf-8').read())

elif page == " Insights Avancés":
    exec(open(Path(__file__).parent / "pages" / "insights.py", encoding='utf-8').read())

elif page == " Matching CV ↔ Offres":
    exec(open(Path(__file__).parent / "pages" / "matching.py", encoding='utf-8').read())

# Footer
st.markdown("""
<div class='footer'>
    Made with ❤️ using Streamlit | Master SISE 2025 | Projet NLP Text Mining
</div>
""", unsafe_allow_html=True)