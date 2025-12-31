"""
PAGE 7 : INSIGHTS AVANCÉS
Clustering, qualité classification, analyses approfondies
"""

import streamlit as st
import pandas as pd
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, RESULTS_DIR, COLORS

# ============================================
# FONCTION CLASSIFICATION
# ============================================

@st.cache_resource
def load_classifier():
    """Charge le système de classification"""
    try:
        with open(MODELS_DIR / 'classification_system.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

def predict_profile(titre, description, competences_list):
    """
    Prédit le profil d'une offre
    
    Args:
        titre: str
        description: str
        competences_list: list of str
    
    Returns:
        dict: {'profil': str, 'score': float, 'confidence': float, 'details': dict}
    """
    
    # Import TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import re
    import unicodedata
    
    # Normalisation
    def normalize(text):
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        return text.lower().strip()
    
    # Profils avec keywords
    profils_keywords = {
        'Data Scientist': {
            'title': ['data scientist', 'scientist', 'science des données'],
            'core': ['machine learning', 'python', 'statistiques', 'modèles'],
            'tech': ['sklearn', 'tensorflow', 'pytorch', 'r']
        },
        'Data Engineer': {
            'title': ['data engineer', 'engineer', 'ingénieur données'],
            'core': ['pipeline', 'etl', 'sql', 'cloud'],
            'tech': ['spark', 'airflow', 'kafka', 'aws', 'azure']
        },
        'Data Analyst': {
            'title': ['data analyst', 'analyst', 'analyste données'],
            'core': ['sql', 'analyse', 'reporting', 'tableau'],
            'tech': ['excel', 'power bi', 'tableau', 'python']
        },
        'BI Analyst': {
            'title': ['bi analyst', 'business intelligence'],
            'core': ['bi', 'reporting', 'tableau de bord', 'kpi'],
            'tech': ['power bi', 'tableau', 'qlik', 'looker']
        },
        'Data Manager': {
            'title': ['data manager', 'manager', 'responsable'],
            'core': ['management', 'équipe', 'stratégie', 'gouvernance'],
            'tech': ['gestion', 'projet', 'coordination']
        },
        'AI Engineer': {
            'title': ['ai engineer', 'ia', 'intelligence artificielle'],
            'core': ['ia', 'deep learning', 'neural', 'llm'],
            'tech': ['tensorflow', 'pytorch', 'huggingface']
        },
        'ML Engineer': {
            'title': ['ml engineer', 'machine learning engineer'],
            'core': ['ml', 'machine learning', 'modèles', 'production'],
            'tech': ['mlflow', 'kubeflow', 'docker', 'kubernetes']
        },
    }
    
    # Normaliser inputs
    titre_norm = normalize(titre)
    desc_norm = normalize(description)
    comp_norm = [normalize(c) for c in competences_list]
    
    best_profil = None
    best_score = 0
    best_details = {}
    
    for profil, keywords in profils_keywords.items():
        score_titre = 0
        score_desc = 0
        score_comp = 0
        
        # SCORE TITRE (60%)
        for kw in keywords['title']:
            if kw in titre_norm:
                score_titre = 10
                break
        else:
            # Fuzzy
            for kw in keywords['title']:
                if any(word in titre_norm for word in kw.split()):
                    score_titre = 6
                    break
        
        # SCORE DESCRIPTION (20%)
        desc_matches = sum(1 for kw in keywords['core'] if kw in desc_norm)
        score_desc = min(desc_matches * 2, 10)
        
        # SCORE COMPÉTENCES (20%)
        comp_matches = sum(1 for c in comp_norm if any(kw in c for kw in keywords['core'] + keywords['tech']))
        score_comp = min(comp_matches * 2, 10)
        
        # SCORE GLOBAL
        score_global = (score_titre * 0.6) + (score_desc * 0.2) + (score_comp * 0.2)
        
        if score_global > best_score:
            best_score = score_global
            best_profil = profil
            best_details = {
                'score_titre': score_titre,
                'score_description': score_desc,
                'score_competences': score_comp,
                'comp_matches': comp_matches
            }
    
    # Confiance
    if best_score >= 7:
        confidence = 0.9
    elif best_score >= 5:
        confidence = 0.7
    elif best_score >= 3:
        confidence = 0.5
    else:
        confidence = 0.3
    
    return {
        'profil': best_profil if best_profil else 'Non classifié',
        'score': best_score,
        'confidence': confidence,
        'details': best_details
    }

@st.cache_data
def load_data():
    with open(MODELS_DIR / 'data_with_profiles.pkl', 'rb') as f:
        df = pickle.load(f)
    
    try:
        with open(RESULTS_DIR / 'clustering_metrics.json', 'r', encoding='utf-8') as f:
            clustering = json.load(f)
    except:
        clustering = {}
    
    try:
        with open(RESULTS_DIR / 'classification_quality.json', 'r', encoding='utf-8') as f:
            quality = json.load(f)
    except:
        quality = {}
    
    return df, clustering, quality

df, clustering, quality = load_data()

st.title("Insights Avancés")
st.markdown("Analyses approfondies et métriques qualité")

st.markdown("---")

# QUALITÉ CLASSIFICATION
st.subheader("Qualité de Classification")

col1, col2, col3 = st.columns(3)

# Calculer depuis df si pas dans JSON
df_class = df[df['status'] == 'classified']
taux = quality.get('classification_rate', len(df_class) / len(df) * 100)
conf = quality.get('average_confidence', df_class['profil_confidence'].mean() if 'profil_confidence' in df_class.columns else 0)
score = quality.get('average_score', df_class['profil_score'].mean() if 'profil_score' in df_class.columns else 0)

with col1:
    st.metric("Taux Classification", f"{taux:.1f}%")

with col2:
    st.metric("Confiance Moyenne", f"{conf:.2f}")

with col3:
    st.metric("Score Moyen", f"{score:.1f}/10")

st.markdown("---")

# DISTRIBUTION SCORES
st.subheader("Distribution des Scores")

df_class = df[df['status'] == 'classified']

fig_scores = px.histogram(
    df_class,
    x='profil_score',
    nbins=20,
    color_discrete_sequence=['#667eea']
)

fig_scores.update_layout(
    template='plotly_dark',
    height=400,
    xaxis_title='Score de Classification',
    yaxis_title='Nombre d\'offres'
)

st.plotly_chart(fig_scores, use_container_width=True)

st.markdown("---")

# ============================================
# PRÉDICTEUR INTERACTIF
# ============================================

st.subheader("Prédicteur de Profil Interactif")

st.info("💡 Testez le système de classification avec vos propres données")

with st.form("prediction_form"):
    col_input1, col_input2 = st.columns([1, 1])
    
    with col_input1:
        titre_input = st.text_input(
            "Titre de l'offre *",
            placeholder="Ex: Data Scientist Senior",
            help="Titre du poste"
        )
    
    with col_input2:
        # Liste compétences disponibles
        all_comp = set()
        for comp_list in df['competences_found']:
            if isinstance(comp_list, list):
                all_comp.update(comp_list)
        
        competences_input = st.multiselect(
            "Compétences requises",
            options=sorted(list(all_comp)),
            default=['python', 'sql'],
            help="Sélectionner les compétences"
        )
    
    description_input = st.text_area(
        "Description de l'offre *",
        placeholder="Ex: Nous recherchons un Data Scientist pour analyser nos données et créer des modèles prédictifs...",
        height=150,
        help="Description complète du poste"
    )
    
    submitted = st.form_submit_button("🔍 Classifier cette offre", use_container_width=True)

if submitted:
    if titre_input and description_input:
        # Prédiction
        result = predict_profile(titre_input, description_input, competences_input)
        
        st.markdown("---")
        
        # Résultat
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1f2937 0%, #374151 100%); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid #10b981;'>
                <p style='color: #9ca3af; font-size: 0.9rem; margin: 0;'>Profil Prédit</p>
                <p style='font-size: 1.8rem; font-weight: 700; margin: 10px 0; color: #10b981;'>{result['profil']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res2:
            score_color = '#10b981' if result['score'] >= 7 else '#f59e0b' if result['score'] >= 4 else '#ef4444'
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1f2937 0%, #374151 100%); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid {score_color};'>
                <p style='color: #9ca3af; font-size: 0.9rem; margin: 0;'>Score Global</p>
                <p style='font-size: 1.8rem; font-weight: 700; margin: 10px 0; color: {score_color};'>{result['score']:.1f}/10</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res3:
            conf_pct = result['confidence'] * 100
            conf_color = '#10b981' if conf_pct >= 70 else '#f59e0b' if conf_pct >= 50 else '#ef4444'
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1f2937 0%, #374151 100%); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid {conf_color};'>
                <p style='color: #9ca3af; font-size: 0.9rem; margin: 0;'>Confiance</p>
                <p style='font-size: 1.8rem; font-weight: 700; margin: 10px 0; color: {conf_color};'>{conf_pct:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Détails scoring
        st.markdown("---")
        st.markdown("### Détails du Scoring")
        
        details = result['details']
        
        col_det1, col_det2, col_det3 = st.columns(3)
        
        with col_det1:
            st.metric("Score Titre (60%)", f"{details.get('score_titre', 0):.1f}/10")
        
        with col_det2:
            st.metric("Score Description (20%)", f"{details.get('score_description', 0):.1f}/10")
        
        with col_det3:
            st.metric("Score Compétences (20%)", f"{details.get('score_competences', 0):.1f}/10")
        
        # Interprétation
        st.markdown("---")
        
        if result['score'] >= 7:
            st.success(" **Haute confiance** : L'offre correspond clairement au profil détecté.")
        elif result['score'] >= 4:
            st.warning(" **Confiance moyenne** : L'offre présente des caractéristiques du profil mais manque de précision.")
        else:
            st.error(" **Faible confiance** : L'offre est difficile à classifier avec certitude.")
    
    else:
        st.error("⚠️ Veuillez remplir au minimum le titre et la description")

st.markdown("---")

# CLUSTERING METRICS
st.subheader(" Métriques Clustering")

col_km, col_hdb = st.columns(2)

with col_km:
    st.markdown("#### KMeans (k=14)")
    if 'kmeans' in clustering and clustering['kmeans']:
        sil = clustering['kmeans'].get('silhouette', 'N/A')
        db = clustering['kmeans'].get('davies_bouldin', 'N/A')
        
        st.metric("Silhouette", f"{sil:.3f}" if isinstance(sil, float) else sil)
        st.metric("Davies-Bouldin", f"{db:.3f}" if isinstance(db, float) else db)
        
        st.info("""
        **Silhouette** : Plus proche de 1 = mieux séparés  
        **Davies-Bouldin** : Plus proche de 0 = mieux
        """)

with col_hdb:
    st.markdown("#### HDBSCAN")
    if 'hdbscan' in clustering and clustering['hdbscan']:
        n_clust = clustering['hdbscan'].get('n_clusters', 'N/A')
        n_noise = clustering['hdbscan'].get('n_noise', 'N/A')
        
        st.metric("Clusters Trouvés", f"{n_clust}")
        st.metric("Bruit", f"{n_noise}")
        
        st.info("""
        **HDBSCAN** : Détecte automatiquement le nombre optimal de clusters
        """)

st.markdown("---")

# ANALYSE SALAIRES DÉTAILLÉE
st.subheader(" Analyse Salaires Détaillée")

df_salaires = df[df['salary_annual'].notna()]

if len(df_salaires) > 0:
    fig_sal = px.box(
        df_salaires,
        x='source_name',
        y='salary_annual',
        color='source_name',
        color_discrete_sequence=[COLORS['primary'], COLORS['accent']]
    )
    
    fig_sal.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title='Source',
        yaxis_title='Salaire Annuel (€)',
        showlegend=False
    )
    
    st.plotly_chart(fig_sal, use_container_width=True)
    
    # Stats salaires
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.metric("Minimum", f"{df_salaires['salary_annual'].min()/1000:.0f}K€")
    
    with col_s2:
        st.metric("Médiane", f"{df_salaires['salary_annual'].median()/1000:.0f}K€")
    
    with col_s3:
        st.metric("Maximum", f"{df_salaires['salary_annual'].max()/1000:.0f}K€")

st.markdown("---")

# INSIGHTS MÉTHODOLOGIQUES
st.subheader(" Insights Méthodologiques")

tab1, tab2, tab3 = st.tabs(["Classification", "Embeddings", "Limites"])

with tab1:
    st.markdown("""
    ### Approche Classification Hybride
    
    **Système en cascade 4 passes :**
    1. **Passe 1** (seuil 4.5) : Haute confiance
    2. **Passe 2** (seuil 3.5) : Confiance moyenne
    3. **Passe 3** (seuil 2.5) : Confiance faible
    4. **Passe 4** (seuil 0.5) : Fourre-tout Data/IA
    
    **Composantes du score :**
    - 60% Titre (matching variantes + fuzzy)
    - 20% Description (TF-IDF + cosinus)
    - 20% Compétences (couverture core + tech)
    """)

with tab2:
    st.markdown("""
    ### Embeddings Multilingues
    
    **Modèle :** `paraphrase-multilingual-MiniLM-L12-v2`
    - 384 dimensions
    - Support FR + EN
    - Optimisé similarité sémantique
    
    **Réduction dimensionnalité :**
    - UMAP (2D/3D)
    - t-SNE (2D)
    
    **Clustering :**
    - KMeans (k=14)
    - HDBSCAN (automatique)
    """)

with tab3:
    st.markdown("""
    ### Limites Identifiées
    
    **43.8% non classifiés :**
    - Titres trop génériques
    - Descriptions manquantes/courtes
    - Compétences insuffisantes
    
    **Biais géographique :**
    - 41.5% Île-de-France
    - Surreprésentation grandes villes
    
    **Qualité données :**
    - Indeed : descriptions riches mais hétérogènes
    - France Travail : standardisées mais courtes
    """)

st.markdown("---")

st.success("""
 **Résultat final :** 56.2% de classification avec précision estimée à 86% sur validation manuelle
""")
