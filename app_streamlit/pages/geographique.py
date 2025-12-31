"""
PAGE 2 : EXPLORATION GÉOGRAPHIQUE
Carte France, heatmap régions, stats locales
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, COLORS

@st.cache_data
def load_data():
    with open(MODELS_DIR / 'data_with_profiles.pkl', 'rb') as f:
        return pickle.load(f)

df = load_data()

# ============================================
# HEADER
# ============================================

st.title("Exploration Géographique")
st.markdown("Distribution géographique des offres Data/IA en France")

st.markdown("---")

# ============================================
# SÉLECTION RÉGION
# ============================================

regions_disponibles = ['Toutes'] + sorted(df['region'].dropna().unique().tolist())
region_selectionnee = st.selectbox(
    "Sélectionner une région",
    regions_disponibles,
    index=0
)

if region_selectionnee != 'Toutes':
    df_region = df[df['region'] == region_selectionnee]
else:
    df_region = df.copy()

# ============================================
# MÉTRIQUES RÉGION
# ============================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Offres", f"{len(df_region):,}")

with col2:
    taux_class = (df_region['status'] == 'classified').sum() / len(df_region) * 100
    st.metric("Classifiées", f"{taux_class:.1f}%")

with col3:
    salaire_median = df_region['salary_annual'].median()
    if pd.notna(salaire_median):
        st.metric("Salaire Médian", f"{salaire_median/1000:.0f}K€")
    else:
        st.metric("Salaire Médian", "N/A")

with col4:
    all_comp_region = []
    for comp_list in df_region['competences_found']:
        if isinstance(comp_list, list):
            all_comp_region.extend(comp_list)
    st.metric("Compétences", f"{len(set(all_comp_region))}")

st.markdown("---")

# ============================================
# CARTE INTERACTIVE (Scatter geo France)
# ============================================

st.subheader("📍 Carte Interactive des Offres")

# Filtrer offres avec coordonnées
df_geo = df_region.dropna(subset=['latitude', 'longitude'])

if len(df_geo) > 0:
    # Couleur par profil si classifié
    df_geo_class = df_geo[df_geo['status'] == 'classified'].copy()
    
    if len(df_geo_class) > 0:
        fig_map = px.scatter_mapbox(
            df_geo_class,
            lat='latitude',
            lon='longitude',
            color='profil_assigned',
            hover_name='title',
            hover_data={
                'company_name': True,
                'city': True,
                'profil_assigned': True,
                'latitude': False,
                'longitude': False
            },
            zoom=5,
            height=600,
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            template='plotly_dark',
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Aucune offre classifiée avec coordonnées pour cette région")
else:
    st.warning("Aucune offre géolocalisée pour cette région")

st.markdown("---")

# ============================================
# DISTRIBUTION RÉGIONS
# ============================================

col_left, col_right = st.columns([6, 4])

with col_left:
    st.subheader("Top 15 Régions")
    
    top_regions = df['region'].value_counts().head(15).reset_index()
    top_regions.columns = ['region', 'count']
    top_regions['percentage'] = top_regions['count'] / len(df) * 100
    
    fig_regions = px.bar(
        top_regions,
        x='count',
        y='region',
        orientation='h',
        text='count',
        color='percentage',
        color_continuous_scale='Plasma',
        hover_data=['percentage']
    )
    
    fig_regions.update_traces(textposition='outside')
    
    fig_regions.update_layout(
        template='plotly_dark',
        height=500,
        xaxis_title='Nombre d\'offres',
        yaxis_title='',
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    st.plotly_chart(fig_regions, use_container_width=True)

with col_right:
    st.subheader("Villes Principales")
    
    if region_selectionnee != 'Toutes':
        top_cities = df_region['city'].value_counts().head(10)
    else:
        top_cities = df['city'].value_counts().head(10)
    
    fig_cities = go.Figure(go.Bar(
        x=top_cities.values,
        y=top_cities.index,
        orientation='h',
        marker_color=COLORS['accent'],
        text=top_cities.values,
        textposition='auto'
    ))
    
    fig_cities.update_layout(
        template='plotly_dark',
        height=500,
        xaxis_title='Offres',
        yaxis_title='',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig_cities, use_container_width=True)

st.markdown("---")

# ============================================
# HEATMAP PROFILS × RÉGIONS
# ============================================

st.subheader("Heatmap Profils × Régions (Top 10)")

df_class = df[df['status'] == 'classified']

# Top 10 régions et top 10 profils
top_10_regions = df_class['region'].value_counts().head(10).index
top_10_profils = df_class['profil_assigned'].value_counts().head(10).index

df_heatmap = df_class[
    df_class['region'].isin(top_10_regions) &
    df_class['profil_assigned'].isin(top_10_profils)
]

# Crosstab
heatmap_data = pd.crosstab(
    df_heatmap['region'],
    df_heatmap['profil_assigned']
)

fig_heatmap = px.imshow(
    heatmap_data,
    labels=dict(x="Profil", y="Région", color="Nombre"),
    x=heatmap_data.columns,
    y=heatmap_data.index,
    color_continuous_scale='Turbo',
    aspect='auto'
)

fig_heatmap.update_layout(
    template='plotly_dark',
    height=500,
    xaxis={'side': 'bottom'},
    font=dict(size=10)
)

fig_heatmap.update_xaxes(tickangle=-45)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("---")

# ============================================
# COMPARAISON RÉGIONS
# ============================================

st.subheader("Comparaison Régionale")

tab1, tab2, tab3 = st.tabs(["Profils", "Compétences", "Salaires"])

with tab1:
    st.markdown("#### Distribution Profils par Région")
    
    region_profil = st.selectbox(
        "Choisir région",
        top_10_regions.tolist(),
        key='profil_region'
    )
    
    df_region_profil = df_class[df_class['region'] == region_profil]
    
    profils_counts = df_region_profil['profil_assigned'].value_counts().head(8)
    
    fig_profils_r = go.Figure(data=[go.Pie(
        labels=profils_counts.index,
        values=profils_counts.values,
        hole=0.3
    )])
    
    fig_profils_r.update_layout(
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig_profils_r, use_container_width=True)

with tab2:
    st.markdown("#### Top 5 Compétences par Région")
    
    region_comp = st.selectbox(
        "Choisir région",
        top_10_regions.tolist(),
        key='comp_region'
    )
    
    df_region_comp = df[df['region'] == region_comp]
    
    all_comp_r = []
    for comp_list in df_region_comp['competences_found']:
        if isinstance(comp_list, list):
            all_comp_r.extend(comp_list)
    
    from collections import Counter
    comp_counts_r = Counter(all_comp_r)
    top_comp_r = pd.DataFrame(
        comp_counts_r.most_common(5),
        columns=['competence', 'count']
    )
    top_comp_r['percentage'] = top_comp_r['count'] / len(df_region_comp) * 100
    
    fig_comp_r = px.bar(
        top_comp_r,
        x='percentage',
        y='competence',
        orientation='h',
        text='count',
        color='percentage',
        color_continuous_scale='Blues'
    )
    
    fig_comp_r.update_traces(textposition='outside')
    
    fig_comp_r.update_layout(
        template='plotly_dark',
        height=300,
        xaxis_title='% Offres',
        yaxis_title='',
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    st.plotly_chart(fig_comp_r, use_container_width=True)

with tab3:
    st.markdown("#### Salaires Médians par Région")
    
    # Calculer salaires médians par région
    salaires_regions = df.groupby('region')['salary_annual'].agg([
        ('median', 'median'),
        ('count', 'count')
    ]).reset_index()
    
    salaires_regions = salaires_regions[salaires_regions['count'] >= 3]  # Min 3 offres
    salaires_regions = salaires_regions.sort_values('median', ascending=False).head(10)
    
    fig_salaires = px.bar(
        salaires_regions,
        x='median',
        y='region',
        orientation='h',
        text='median',
        color='median',
        color_continuous_scale='Greens'
    )
    
    fig_salaires.update_traces(
        texttemplate='%{text:.0f}€',
        textposition='outside'
    )
    
    fig_salaires.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title='Salaire Médian (€/an)',
        yaxis_title='',
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    st.plotly_chart(fig_salaires, use_container_width=True)

st.markdown("---")

# ============================================
# TABLEAU DÉTAILS
# ============================================

with st.expander("📋 Voir le tableau détaillé"):
    st.dataframe(
        df_region[[
            'title', 'company_name', 'city', 'region',
            'profil_assigned', 'salary_annual', 'contract_type'
        ]].head(100),
        use_container_width=True
    )
