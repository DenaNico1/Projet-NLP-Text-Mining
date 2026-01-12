"""
PAGE 3 : PROFILS MÉTIERS
Analyse des 14 profils, comparateur, radar compétences
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, RESULTS_DIR, COLORS

"""@st.cache_data
def load_data():
    with open(MODELS_DIR / 'data_with_profiles.pkl', 'rb') as f:
        df = pickle.load(f)
    
    with open(RESULTS_DIR / 'profils_distribution.json', 'r', encoding='utf-8') as f:
        profils_stats = json.load(f)
    
    return df, profils_stats

df, profils_stats = load_data()"""

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

# ==========================
# SANKEY : Compétences → Profils (Global) — CORRIGÉ
# ==========================

st.markdown("### Flux Compétences → Profils")
st.caption("Les 10 compétences les plus demandées et leur répartition dans les profils")

df_class = df[df["status"] == "classified"]

# Top 10 compétences
all_comps_global = []
for comp_list in df_class["competences_found"]:
    if isinstance(comp_list, list):
        all_comps_global.extend(comp_list)

from collections import Counter

top_comps_global = [c for c, _ in Counter(all_comps_global).most_common(10)]

# Top 8 profils
top_profils_global = df_class["profil_assigned"].value_counts().head(8).index.tolist()

# Palette de couleurs pour les compétences (couleurs vives et distinctes)
palette_comp = [
    "#00D9FF",  # Cyan
    "#00FF9D",  # Vert menthe
    "#FFD700",  # Or
    "#FF6B9D",  # Rose
    "#A78BFA",  # Violet
    "#FB923C",  # Orange
    "#34D399",  # Vert émeraude
    "#F472B6",  # Rose vif
    "#60A5FA",  # Bleu
    "#FBBF24",  # Jaune
]

# Couleurs pour les profils (plus neutres)
palette_prof = [
    "#8B5CF6",  # Violet
    "#6366F1",  # Indigo
    "#3B82F6",  # Bleu
    "#0EA5E9",  # Cyan
    "#14B8A6",  # Teal
    "#10B981",  # Vert
    "#F59E0B",  # Ambre
    "#EF4444",  # Rouge
]

# Flux
flows_global = []
for comp in top_comps_global:
    for profil in top_profils_global:
        count = len(
            df_class[
                (df_class["profil_assigned"] == profil)
                & df_class["competences_found"].apply(
                    lambda x: comp in x if isinstance(x, list) else False
                )
            ]
        )
        if count > 10:
            flows_global.append({"source": comp, "target": profil, "value": count})

if flows_global:
    df_flows = pd.DataFrame(flows_global)

    nodes = list(set(df_flows["source"]) | set(df_flows["target"]))
    node_idx = {n: i for i, n in enumerate(nodes)}

    sources = df_flows["source"].map(node_idx)
    targets = df_flows["target"].map(node_idx)

    # --------------------
    # COULEURS NODES
    # --------------------
    comp_colors = {
        comp: palette_comp[i % len(palette_comp)]
        for i, comp in enumerate(top_comps_global)
    }

    profil_colors = {
        prof: palette_prof[i % len(palette_prof)]
        for i, prof in enumerate(top_profils_global)
    }

    node_colors = [comp_colors.get(n, profil_colors.get(n, "#888888")) for n in nodes]

    # --------------------
    # COULEURS LIENS = couleur de la compétence source avec transparence
    # --------------------
    def hex_to_rgba(hex_color, alpha=0.25):
        """Convertit une couleur hex en rgba"""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    link_colors = [hex_to_rgba(comp_colors[s], 0.25) for s in df_flows["source"]]

    fig_sankey_global = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=nodes,
                color=node_colors,
                pad=15,
                thickness=20,
                line=dict(color="#0e1117", width=0.5),
                hoverlabel=dict(bgcolor="#111827", font_color="white"),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=df_flows["value"],
                color=link_colors,
                hovertemplate="%{source.label} → %{target.label}<br>%{value} offres<extra></extra>",
            ),
        )
    )

    fig_sankey_global.update_layout(
        height=700,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        hovermode="closest",
        margin=dict(l=10, r=10, t=40, b=10),
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
top_profils_sun = df_class["profil_assigned"].value_counts().head(6).index
top_regions_sun = df_class["region"].value_counts().head(5).index

df_viz_sun = df_class[
    (df_class["profil_assigned"].isin(top_profils_sun))
    & (df_class["region"].isin(top_regions_sun))
]

# Préparer données
data_sun_global = []
for profil in top_profils_sun:
    for contrat in df_viz_sun["contract_type"].unique():
        if pd.notna(contrat):
            for region in top_regions_sun:
                count = len(
                    df_viz_sun[
                        (df_viz_sun["profil_assigned"] == profil)
                        & (df_viz_sun["contract_type"] == contrat)
                        & (df_viz_sun["region"] == region)
                    ]
                )
                if count > 0:
                    data_sun_global.append(
                        {
                            "Profil": profil,
                            "Contrat": contrat,
                            "Région": region,
                            "Count": count,
                        }
                    )

df_sun_global = pd.DataFrame(data_sun_global)

if len(df_sun_global) > 0:
    fig_sunburst_global = px.sunburst(
        df_sun_global,
        path=["Profil", "Contrat", "Région"],
        values="Count",
        color="Count",
        color_continuous_scale="Purples",
        height=700,
    )

    fig_sunburst_global.update_layout(
        template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig_sunburst_global, use_container_width=True)
else:
    st.warning("Pas assez de données pour le Sunburst global")


# ============================================
# SÉLECTION PROFIL
# ============================================
st.markdown("---")
st.subheader(" Exploration par Profil")

df_class = df[df["status"] == "classified"]
profils_disponibles = sorted(df_class["profil_assigned"].unique())

profil_choisi = st.selectbox(
    "Sélectionner un profil métier", profils_disponibles, index=0
)

df_profil = df_class[df_class["profil_assigned"] == profil_choisi]

# ============================================
# MÉTRIQUES PROFIL
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.metric("Offres", f"{len(df_profil):,}")


with col2:
    score_moyen = df_profil["profil_score"].mean()
    st.metric("Score Moyen", f"{score_moyen:.1f}/10")

st.markdown("---")

# ============================================
# DESCRIPTION AUTO
# ============================================

descriptions = {
    "Data Engineer": "Conception et maintenance d'infrastructures de données robustes, pipelines ETL, et systèmes Big Data.",
    "Data Scientist": "Modélisation prédictive, Machine Learning, et extraction d'insights à partir de données complexes.",
    "Data Analyst": "Analyse de données business, création de tableaux de bord, et support décisionnel.",
    "BI Analyst": "Business Intelligence, reporting, et visualisation de données pour pilotage stratégique.",
    "Data Manager": "Management d'équipes data, définition de stratégie data, et gouvernance.",
}

desc = descriptions.get(profil_choisi, "Profil Data/IA spécialisé.")

st.info(f"** {profil_choisi}** : {desc}")

st.markdown("---")

# ============================================
# TOP COMPÉTENCES PROFIL (RADAR)
# ============================================

st.subheader(f" Top 10 Compétences - {profil_choisi}")

all_comp_profil = []
for comp_list in df_profil["competences_found"]:
    if isinstance(comp_list, list):
        all_comp_profil.extend(comp_list)

from collections import Counter

comp_counts_profil = Counter(all_comp_profil)
top_comp_profil = pd.DataFrame(
    comp_counts_profil.most_common(10), columns=["competence", "count"]
)
top_comp_profil["percentage"] = top_comp_profil["count"] / len(df_profil) * 100

# Radar chart
fig_radar = go.Figure()

fig_radar.add_trace(
    go.Scatterpolar(
        r=top_comp_profil["percentage"].tolist(),
        theta=top_comp_profil["competence"].tolist(),
        fill="toself",
        fillcolor="rgba(102, 126, 234, 0.3)",
        line_color="rgb(102, 126, 234)",
        line_width=2,
    )
)

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True, range=[0, top_comp_profil["percentage"].max() * 1.1]
        )
    ),
    template="plotly_dark",
    height=500,
    showlegend=False,
)

st.plotly_chart(fig_radar, use_container_width=True)


st.markdown("""----""")


# ============================================
# COMPARATEUR PROFILS
# ============================================

st.subheader(" Comparateur de Profils")

col_comp1, col_comp2 = st.columns(2)

with col_comp1:
    profil_1 = st.selectbox("Profil 1", profils_disponibles, index=0, key="prof1")

with col_comp2:
    profil_2 = st.selectbox(
        "Profil 2",
        profils_disponibles,
        index=min(1, len(profils_disponibles) - 1),
        key="prof2",
    )

if profil_1 != profil_2:
    df_p1 = df_class[df_class["profil_assigned"] == profil_1]
    df_p2 = df_class[df_class["profil_assigned"] == profil_2]

    # Comparer compétences
    comp_p1 = []
    for comp_list in df_p1["competences_found"]:
        if isinstance(comp_list, list):
            comp_p1.extend(comp_list)

    comp_p2 = []
    for comp_list in df_p2["competences_found"]:
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
        comp_data.append(
            {
                "Compétence": comp,
                profil_1: counts_p1.get(comp, 0) / len(df_p1) * 100,
                profil_2: counts_p2.get(comp, 0) / len(df_p2) * 100,
            }
        )

    df_comp = pd.DataFrame(comp_data)

    fig_comp = go.Figure()

    fig_comp.add_trace(
        go.Bar(
            name=profil_1,
            x=df_comp["Compétence"],
            y=df_comp[profil_1],
            marker_color=COLORS["primary"],
        )
    )

    fig_comp.add_trace(
        go.Bar(
            name=profil_2,
            x=df_comp["Compétence"],
            y=df_comp[profil_2],
            marker_color=COLORS["accent"],
        )
    )

    fig_comp.update_layout(
        barmode="group",
        template="plotly_dark",
        height=400,
        xaxis_title="Compétence",
        yaxis_title="% Offres",
        legend=dict(x=0.7, y=1.0),
    )

    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# ============================================
# OFFRES EXEMPLES
# ============================================

with st.expander(f"📋 Exemples d'Offres - {profil_choisi}"):
    st.dataframe(
        df_profil[
            [
                "title",
                "company_name",
                "city",
                "region",
                "salary_annual",
                "profil_score",
                "profil_confidence",
            ]
        ].head(20),
        use_container_width=True,
    )
