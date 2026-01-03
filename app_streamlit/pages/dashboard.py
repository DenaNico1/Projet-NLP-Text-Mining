"""
PAGE 1 : DASHBOARD - Vue d'ensemble
KPIs, timeline, top compétences, carte France
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Imports locaux
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, RESULTS_DIR, COLORS

# ============================================
# CHARGEMENT DONNÉES
# ============================================
#from config_db import load_offres_with_nlp
#df = load_offres_with_nlp()

from utils import get_data
df = get_data()



# Appliquer filtres globaux
filters = st.session_state.get('filters', {})
df_filtered = df.copy()

if filters.get('source') and filters['source'] != 'Toutes':
    df_filtered = df_filtered[df_filtered['source'] == filters['source']]

if filters.get('region') and filters['region'] != 'Toutes':
    df_filtered = df_filtered[df_filtered['region'] == filters['region']]

# ============================================
# HEADER
# ============================================

st.title("Dashboard - Vue d'Ensemble")
st.markdown("Analyse globale du marché Data/IA en France")

st.markdown("---")

# ============================================
# KPIs
# ============================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <h3 style='color: #667eea; margin: 0;'>Total Offres</h3>
        <p style='font-size: 2.5rem; font-weight: 700; margin: 10px 0; color: white;'>{len(df_filtered):,}</p>
        <p style='color: #9ca3af; font-size: 0.9rem;'>France Travail + Indeed</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    taux_class = (df_filtered['status'] == 'classified').sum() / len(df_filtered) * 100
    st.markdown(f"""
    <div class='metric-card'>
        <h3 style='color: #10b981; margin: 0;'>Classifiées</h3>
        <p style='font-size: 2.5rem; font-weight: 700; margin: 10px 0; color: white;'>{taux_class:.1f}%</p>
        <p style='color: #9ca3af; font-size: 0.9rem;'>{(df_filtered['status'] == 'classified').sum():,} offres</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    all_comp = []
    for comp_list in df_filtered['competences_found']:
        if isinstance(comp_list, list):
            all_comp.extend(comp_list)
    nb_comp_uniques = len(set(all_comp))
    
    st.markdown(f"""
    <div class='metric-card'>
        <h3 style='color: #f59e0b; margin: 0;'>Compétences</h3>
        <p style='font-size: 2.5rem; font-weight: 700; margin: 10px 0; color: white;'>{nb_comp_uniques}</p>
        <p style='color: #9ca3af; font-size: 0.9rem;'>Uniques identifiées</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    nb_profils = df_filtered[df_filtered['status'] == 'classified']['profil_assigned'].nunique()
    st.markdown(f"""
    <div class='metric-card'>
        <h3 style='color: #ec4899; margin: 0;'>Profils Métiers</h3>
        <p style='font-size: 2.5rem; font-weight: 700; margin: 10px 0; color: white;'>{nb_profils}</p>
        <p style='color: #9ca3af; font-size: 0.9rem;'>Catégories détectées</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# LIGNE 1 : TIMELINE + TOP PROFILS
# ============================================

col_left, col_right = st.columns([6, 4])

with col_left:
    st.subheader("Évolution Temporelle des Offres")
    
    df_timeline = df_filtered.copy()
    df_timeline['date_posted'] = pd.to_datetime(df_timeline['date_posted'], errors='coerce')
    df_timeline = df_timeline.dropna(subset=['date_posted'])
    df_timeline['month'] = df_timeline['date_posted'].dt.to_period('M').astype(str)
    
    timeline_data = df_timeline.groupby('month').size().reset_index(name='count')
    
    fig_timeline = px.line(
        timeline_data,
        x='month',
        y='count',
        title='',
        markers=True
    )
    
    fig_timeline.update_traces(
        line_color=COLORS['primary'],
        line_width=3,
        marker=dict(size=8, color=COLORS['accent'])
    )
    
    fig_timeline.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title='Mois',
        yaxis_title='Nombre d\'offres',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)

with col_right:
    st.subheader("🏆 Top 5 Profils Métiers")
    
    df_class = df_filtered[df_filtered['status'] == 'classified']
    top_profils = df_class['profil_assigned'].value_counts().head(5)
    
    fig_profils = go.Figure(go.Bar(
        x=top_profils.values,
        y=top_profils.index,
        orientation='h',
        marker=dict(
            color=top_profils.values,
            colorscale='Viridis',
            showscale=False
        ),
        text=top_profils.values,
        textposition='auto'
    ))
    
    fig_profils.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title='Nombre d\'offres',
        yaxis_title='',
        showlegend=False
    )
    
    st.plotly_chart(fig_profils, use_container_width=True)

st.markdown("---")

# ============================================
# LIGNE 2 : TOP COMPÉTENCES + DISTRIBUTION SOURCES
# ============================================

col_left2, col_right2 = st.columns([5, 5])

with col_left2:
    st.subheader("Top 15 Compétences Demandées")
    
    from collections import Counter
    comp_counts = Counter(all_comp)
    top_comp = pd.DataFrame(
        comp_counts.most_common(15),
        columns=['competence', 'count']
    )
    top_comp['percentage'] = top_comp['count'] / len(df_filtered) * 100
    
    fig_comp = px.bar(
        top_comp,
        x='count',
        y='competence',
        orientation='h',
        text='count',
        color='percentage',
        color_continuous_scale='Sunset'
    )
    
    fig_comp.update_traces(textposition='outside')
    
    fig_comp.update_layout(
        template='plotly_dark',
        height=500,
        xaxis_title='Nombre d\'offres',
        yaxis_title='',
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)

with col_right2:
    st.subheader("Distribution par Source")
    
    source_counts = df_filtered['source'].value_counts()
    
    fig_sources = go.Figure(data=[go.Pie(
        labels=source_counts.index,
        values=source_counts.values,
        hole=0.4,
        marker=dict(colors=[COLORS['primary'], COLORS['accent']]),
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig_sources.update_layout(
        template='plotly_dark',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_sources, use_container_width=True)

st.markdown("---")

# ============================================
# LIGNE 3 : CARTE FRANCE + STATS RÉGIONALES
# ============================================

st.subheader("Distribution Géographique")

col_map, col_stats = st.columns([6, 4])

with col_map:
    # Top 10 régions
    top_regions = df_filtered['region'].value_counts().head(10).reset_index()
    top_regions.columns = ['region', 'count']
    
    fig_regions = px.bar(
        top_regions,
        x='count',
        y='region',
        orientation='h',
        text='count',
        title='Top 10 Régions'
    )
    
    fig_regions.update_traces(
        marker_color=COLORS['info'],
        textposition='outside'
    )
    
    fig_regions.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title='Nombre d\'offres',
        yaxis_title='',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig_regions, use_container_width=True)

with col_stats:
    st.markdown("### Statistiques Clés")
    
    # Salaire médian
    salaire_median = df_filtered['salary_annual'].median()
    if pd.notna(salaire_median):
        st.metric(
            "Salaire Médian",
            f"{salaire_median/1000:.0f}K€",
            help="Salaire annuel brut médian"
        )
    
    # Tokens moyen
    tokens_moyen = df_filtered['num_tokens'].mean()
    st.metric(
        "Tokens par Offre",
        f"{tokens_moyen:.0f}",
        help="Longueur moyenne descriptions"
    )
    
    # Compétences moyennes
    comp_moyennes = df_filtered['num_competences'].mean()
    st.metric(
        "Compétences par Offre",
        f"{comp_moyennes:.1f}",
        help="Nombre moyen compétences extraites"
    )
    
    # Région dominante
    region_top = df_filtered['region'].value_counts().index[0]
    st.metric(
        "Région Dominante",
        region_top,
        help="Région avec le plus d'offres"
    )

st.markdown("---")

# ============================================
# INSIGHTS RAPIDES
# ============================================

st.subheader("💡 Insights Rapides")

col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("🔥 Tendances Technologiques"):
        st.markdown("""
        **Technologies Émergentes :**
        - Python domine (38.1% des offres)
        - SQL reste essentiel (32.0%)
        - ML/IA en forte demande (21.2%)
        - Cloud (AWS, Azure, GCP) très présent
        - DevOps & CI/CD en croissance
        """)

with col2:
    with st.expander("📍 Concentration Géographique"):
        idf_pct = (df_filtered['region'] == 'Île-de-France').sum() / len(df_filtered) * 100
        st.markdown(f"""
        **Répartition Géographique :**
        - Île-de-France : {idf_pct:.1f}% des offres
        - Lyon : 2ème pôle tech
        - Forte concentration urbaine
        - Télétravail mentionné dans 15%+ offres
        """)

with col3:
    with st.expander("💰 Rémunérations"):
        st.markdown(f"""
        **Fourchettes Salaires :**
        - Médiane : {salaire_median/1000 if pd.notna(salaire_median) else 'N/A'}K€/an
        - Data Scientists : salaires hauts
        - Stages/Alternances : nombreux
        - Écart IDF vs Régions notable
        """)

st.markdown("---")

# ============================================
# CALL TO ACTION
# ============================================

st.info("""
🎯 **Explorez plus en détails :**
- **🗺️ Géographique** : Carte interactive France
- **💼 Profils** : Analyse des 14 profils métiers
- **🎓 Compétences** : Réseau sémantique & co-occurrences
- **🔬 Topics** : Thématiques émergentes (LDA)
- **🌐 3D** : Visualisations embeddings
""")
