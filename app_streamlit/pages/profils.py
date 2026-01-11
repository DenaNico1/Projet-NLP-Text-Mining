"""
PAGE 3 : PROFILS MÉTIERS
Analyse des 14 profils, comparateur, radar compétences
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COLORS

# data
from data_loaders import load_profils_data
df, profils_stats = load_profils_data()

# ============================================
# HEADER
# ============================================

st.title(" Profils Métiers Data/IA")
st.markdown("Analyse des 14 profils métiers identifiés")

# ============================================
# VISUALISATIONS GLOBALES (TOUS PROFILS)
# ============================================

st.markdown("---")
st.subheader(" Visualisations Globales - Tous Profils")

# ==========================================
# SANKEY : Compétences → Profils (Global)
# ==========================================

st.markdown("### Flux Compétences → Profils")
st.caption("Les 10 compétences les plus demandées et leur répartition dans les profils")

df_class = df[df['status'] == 'classified']
# Top 10 compétences globales
all_comps_global = []
for comp_list in df_class['competences_found']:
    if isinstance(comp_list, list):
        all_comps_global.extend(comp_list)

from collections import Counter
top_comps_global = [c for c, _ in Counter(all_comps_global).most_common(10)]

# Top 8 profils
top_profils_global = df_class['profil_assigned'].value_counts().head(8).index

# Compter flux compétence → profil
flows_global = []
for comp in top_comps_global:
    for profil in top_profils_global:
        df_match = df_class[
            (df_class['profil_assigned'] == profil) &
            (df_class['competences_found'].apply(lambda x: comp in x if isinstance(x, list) else False))
        ]
        count = len(df_match)
        if count > 10:  # Seuil minimum
            flows_global.append({
                'source': comp,
                'target': profil,
                'value': count
            })

if flows_global:
    df_flows_global = pd.DataFrame(flows_global)
    
    # Créer labels et indices
    all_nodes_global = list(set(df_flows_global['source'].tolist() + df_flows_global['target'].tolist()))
    node_dict_global = {node: idx for idx, node in enumerate(all_nodes_global)}
    
    source_indices_global = [node_dict_global[s] for s in df_flows_global['source']]
    target_indices_global = [node_dict_global[t] for t in df_flows_global['target']]
    
    # Couleurs nodes (compétences vs profils)
    node_colors_global = []
    for node in all_nodes_global:
        if node in top_profils_global:
            node_colors_global.append('rgba(102, 126, 234, 0.8)')  # Profils en violet
        else:
            node_colors_global.append('rgba(34, 197, 94, 0.7)')  # Compétences en vert
    
    fig_sankey_global = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=all_nodes_global,
            color=node_colors_global
        ),
        link=dict(
            source=source_indices_global,
            target=target_indices_global,
            value=df_flows_global['value'],
            color='rgba(102, 126, 234, 0.2)'
        )
    )])
    
    fig_sankey_global.update_layout(
        height=700,
        template='plotly_dark',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig_sankey_global, use_container_width=True)
else:
    st.warning("Pas assez de données pour le Sankey global")

# ==========================================
# SUNBURST : Profils → Contrats → Régions (Global)
# ==========================================

st.markdown("---")
st.markdown("###  Hiérarchie Profils → Contrats → Régions")
st.caption("Distribution des profils par type de contrat et région")

# Top 6 profils, 5 régions
top_profils_sun = df_class['profil_assigned'].value_counts().head(6).index
top_regions_sun = df_class['region'].value_counts().head(5).index

df_viz_sun = df_class[
    (df_class['profil_assigned'].isin(top_profils_sun)) &
    (df_class['region'].isin(top_regions_sun))
]

# Préparer données
data_sun_global = []
for profil in top_profils_sun:
    for contrat in df_viz_sun['contract_type'].unique():
        if pd.notna(contrat):
            for region in top_regions_sun:
                count = len(df_viz_sun[
                    (df_viz_sun['profil_assigned'] == profil) &
                    (df_viz_sun['contract_type'] == contrat) &
                    (df_viz_sun['region'] == region)
                ])
                if count > 0:
                    data_sun_global.append({
                        'Profil': profil,
                        'Contrat': contrat,
                        'Région': region,
                        'Count': count
                    })

df_sun_global = pd.DataFrame(data_sun_global)

if len(df_sun_global) > 0:
    fig_sunburst_global = px.sunburst(
        df_sun_global,
        path=['Profil', 'Contrat', 'Région'],
        values='Count',
        color='Count',
        color_continuous_scale='Purples',
        height=700
    )
    
    fig_sunburst_global.update_layout(
        template='plotly_dark',
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig_sunburst_global, use_container_width=True)
else:
    st.warning("Pas assez de données pour le Sunburst global")



# ============================================
# SÉLECTION PROFIL
# ============================================
st.markdown("---")
st.subheader(" Exploration par Profil")

df_class = df[df['status'] == 'classified']
profils_disponibles = sorted(df_class['profil_assigned'].unique())

profil_choisi = st.selectbox(
    "Sélectionner un profil métier",
    profils_disponibles,
    index=0
)

df_profil = df_class[df_class['profil_assigned'] == profil_choisi]

# ============================================
# MÉTRIQUES PROFIL
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.metric("Offres", f"{len(df_profil):,}")


with col2:
    score_moyen = df_profil['profil_score'].mean()
    st.metric("Score Moyen", f"{score_moyen:.1f}/10")

st.markdown("---")

# ============================================
# DESCRIPTION AUTO
# ============================================

descriptions = {
    'Data Engineer': "Conception et maintenance d'infrastructures de données robustes, pipelines ETL, et systèmes Big Data.",
    'Data Scientist': "Modélisation prédictive, Machine Learning, et extraction d'insights à partir de données complexes.",
    'Data Analyst': "Analyse de données business, création de tableaux de bord, et support décisionnel.",
    'BI Analyst': "Business Intelligence, reporting, et visualisation de données pour pilotage stratégique.",
    'Data Manager': "Management d'équipes data, définition de stratégie data, et gouvernance.",
}

desc = descriptions.get(profil_choisi, "Profil Data/IA spécialisé.")

st.info(f"** {profil_choisi}** : {desc}")

st.markdown("---")

# ============================================
# TOP COMPÉTENCES PROFIL (RADAR)
# ============================================

st.subheader(f" Top 10 Compétences - {profil_choisi}")

all_comp_profil = []
for comp_list in df_profil['competences_found']:
    if isinstance(comp_list, list):
        all_comp_profil.extend(comp_list)

from collections import Counter
comp_counts_profil = Counter(all_comp_profil)
top_comp_profil = pd.DataFrame(
    comp_counts_profil.most_common(10),
    columns=['competence', 'count']
)
top_comp_profil['percentage'] = top_comp_profil['count'] / len(df_profil) * 100

# Radar chart
fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=top_comp_profil['percentage'].tolist(),
    theta=top_comp_profil['competence'].tolist(),
    fill='toself',
    fillcolor='rgba(102, 126, 234, 0.3)',
    line_color='rgb(102, 126, 234)',
    line_width=2
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, top_comp_profil['percentage'].max() * 1.1]
        )
    ),
    template='plotly_dark',
    height=500,
    showlegend=False
)

st.plotly_chart(fig_radar, use_container_width=True)


st.markdown("""----""")


# ============================================
# COMPARATEUR PROFILS
# ============================================

st.subheader("⚖️ Comparateur de Profils")

col_comp1, col_comp2 = st.columns(2)

with col_comp1:
    profil_1 = st.selectbox("Profil 1", profils_disponibles, index=0, key='prof1')

with col_comp2:
    profil_2 = st.selectbox("Profil 2", profils_disponibles, index=min(1, len(profils_disponibles)-1), key='prof2')

if profil_1 != profil_2:
    df_p1 = df_class[df_class['profil_assigned'] == profil_1]
    df_p2 = df_class[df_class['profil_assigned'] == profil_2]
    
    # Comparer compétences
    comp_p1 = []
    for comp_list in df_p1['competences_found']:
        if isinstance(comp_list, list):
            comp_p1.extend(comp_list)
    
    comp_p2 = []
    for comp_list in df_p2['competences_found']:
        if isinstance(comp_list, list):
            comp_p2.extend(comp_list)
    
    counts_p1 = Counter(comp_p1)
    counts_p2 = Counter(comp_p2)
    
    # Top 5 chacun
    top5_p1 = set([c for c, _ in counts_p1.most_common(5)])
    top5_p2 = set([c for c, _ in counts_p2.most_common(5)])
    
    all_comp = top5_p1 | top5_p2
    
    comp_data = []
    for comp in all_comp:
        comp_data.append({
            'Compétence': comp,
            profil_1: counts_p1.get(comp, 0) / len(df_p1) * 100,
            profil_2: counts_p2.get(comp, 0) / len(df_p2) * 100
        })
    
    df_comp = pd.DataFrame(comp_data)
    
    fig_comp = go.Figure()
    
    fig_comp.add_trace(go.Bar(
        name=profil_1,
        x=df_comp['Compétence'],
        y=df_comp[profil_1],
        marker_color=COLORS['primary']
    ))
    
    fig_comp.add_trace(go.Bar(
        name=profil_2,
        x=df_comp['Compétence'],
        y=df_comp[profil_2],
        marker_color=COLORS['accent']
    ))
    
    fig_comp.update_layout(
        barmode='group',
        template='plotly_dark',
        height=400,
        xaxis_title='Compétence',
        yaxis_title='% Offres',
        legend=dict(x=0.7, y=1.0)
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# ============================================
# OFFRES EXEMPLES
# ============================================

with st.expander(f"📋 Exemples d'Offres - {profil_choisi}"):
    st.dataframe(
        df_profil[[
            'title', 'company_name', 'city', 'region',
            'salary_annual', 'profil_score', 'profil_confidence'
        ]].head(20),
        use_container_width=True
    )
