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

from utils import get_data

df = get_data()

# ============================================
# HEADER
# ============================================

st.title("Exploration Géographique")
st.markdown("Distribution géographique des offres Data/IA en France")

st.markdown("---")

# ============================================
# SÉLECTION RÉGION
# ============================================

regions_disponibles = ["Toutes"] + sorted(df["region"].dropna().unique().tolist())
region_selectionnee = st.selectbox(
    "Sélectionner une région", regions_disponibles, index=0
)

if region_selectionnee != "Toutes":
    df_region = df[df["region"] == region_selectionnee]
else:
    df_region = df.copy()

# ============================================
# MÉTRIQUES RÉGION (Style Dashboard)
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
    <div class='metric-card'>
        <h3 style='color: #667eea; margin: 0;'>Offres</h3>
        <p style='font-size: 2.5rem; font-weight: 700; margin: 10px 0; color: white;'>{len(df_region):,}</p>
        <p style='color: #9ca3af; font-size: 0.9rem;'>Offres dans la sélection</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    all_comp_region = []
    for comp_list in df_region["competences_found"]:
        if isinstance(comp_list, list):
            all_comp_region.extend(comp_list)
    nb_comp_uniques = len(set(all_comp_region))

    st.markdown(
        f"""
    <div class='metric-card'>
        <h3 style='color: #ec4899; margin: 0;'>Compétences</h3>
        <p style='font-size: 2.5rem; font-weight: 700; margin: 10px 0; color: white;'>{nb_comp_uniques}</p>
        <p style='color: #9ca3af; font-size: 0.9rem;'>Uniques identifiées</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ============================================
# CARTE INTERACTIVE (Scatter geo France)
# ============================================

st.subheader("📍 Carte Interactive des Offres")

# Filtrer offres avec coordonnées
df_geo = df_region.dropna(subset=["latitude", "longitude"])

if len(df_geo) > 0:
    # Couleur par profil si classifié
    df_geo_class = df_geo[df_geo["status"] == "classified"].copy()

    if len(df_geo_class) > 0:
        fig_map = px.scatter_mapbox(
            df_geo_class,
            lat="latitude",
            lon="longitude",
            color="profil_assigned",
            hover_name="title",
            hover_data={
                "company_name": True,
                "city": True,
                "profil_assigned": True,
                "latitude": False,
                "longitude": False,
            },
            zoom=5,
            height=600,
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )

        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            template="plotly_dark",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )

        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Aucune offre classifiée avec coordonnées pour cette région")
else:
    st.warning("Aucune offre géolocalisée pour cette région")

st.markdown("---")


########################################################################
st.subheader(" Bassins d'Emploi par Ville")

# Préparer données bassins (villes)
df_bassins = df.dropna(subset=["city", "latitude", "longitude"]).copy()

# Compter offres par ville
df_bassins_agg = df_bassins.groupby("city", as_index=False).agg(
    {
        "latitude": "first",
        "longitude": "first",
        "region": "first",
        "title": "count",  # Compte le nombre de lignes (offres)
    }
)

# Renommer 'title' en 'nb_offres'
df_bassins_agg.rename(columns={"title": "nb_offres"}, inplace=True)

# Trier par nombre d'offres
df_bassins_agg = df_bassins_agg.sort_values("nb_offres", ascending=False)

# Filtrer par région si sélectionnée
if region_selectionnee != "Toutes":
    df_bassins_map = df_bassins_agg[df_bassins_agg["region"] == region_selectionnee]
else:
    df_bassins_map = df_bassins_agg.copy()

# ==========================================
# 2 CARTES CÔTE À CÔTE
# ==========================================

if len(df_bassins_map) > 0:
    col_map1, col_map2 = st.columns(2)

    # ==========================================
    # CARTE 1 : TOUTES LES VILLES (Gauche)
    # ==========================================

    with col_map1:
        st.markdown("### Offres par bassin")
        st.caption("Taille des bulles proportionnelle au nombre d'offres")

        fig_bubbles = px.scatter_mapbox(
            df_bassins_map,
            lat="latitude",
            lon="longitude",
            size="nb_offres",
            hover_name="city",
            hover_data={
                "nb_offres": True,
                "region": True,
                "latitude": False,
                "longitude": False,
            },
            color="nb_offres",
            color_continuous_scale="YlOrRd",
            size_max=60,
            zoom=5 if region_selectionnee == "Toutes" else 7,
            height=600,
            labels={"nb_offres": "Nombre d'offres"},
        )

        fig_bubbles.update_layout(
            mapbox_style="carto-darkmatter",
            template="plotly_dark",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Offres", thickness=15, len=0.7),
        )

        st.plotly_chart(fig_bubbles, use_container_width=True)

    # ==========================================
    # CARTE 2 : TOP 20 VILLES (Droite)
    # ==========================================

    with col_map2:
        st.markdown("### Focus Top villes")
        st.caption("Les tops bassins avec le plus d'offres")

        # Top 20 villes
        top_20_villes = df_bassins_map.head(20)

        fig_mini = px.scatter_mapbox(
            top_20_villes,
            lat="latitude",
            lon="longitude",
            size="nb_offres",
            hover_name="city",
            hover_data={
                "nb_offres": True,
                "region": True,
                "latitude": False,
                "longitude": False,
            },
            color="nb_offres",
            color_continuous_scale="YlOrRd",
            size_max=40,
            zoom=4.5,
            height=600,
            labels={"nb_offres": "Nombre d'offres"},
        )

        fig_mini.update_layout(
            mapbox_style="carto-darkmatter",
            template="plotly_dark",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Offres", thickness=15, len=0.7),
        )

        st.plotly_chart(fig_mini, use_container_width=True)

else:
    st.warning("Aucune donnée géolocalisée pour cette sélection")

st.markdown("---")

# ==========================================
# STATISTIQUES BASSINS
# ==========================================
st.markdown("### Statistiques Bassins")

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.metric(
        "Nombre de villes",
        f"{len(df_bassins_map):,}",
        help="Nombre total de villes avec au moins une offre",
    )

with col_stat2:
    st.metric(
        "Ville principale",
        df_bassins_map.iloc[0]["city"] if len(df_bassins_map) > 0 else "N/A",
        (
            f"{df_bassins_map.iloc[0]['nb_offres']} offres"
            if len(df_bassins_map) > 0
            else ""
        ),
    )

with col_stat3:
    # Concentration top 5
    if len(df_bassins_map) >= 5:
        top_5_offres = df_bassins_map.head(5)["nb_offres"].sum()
        total_offres = df_bassins_map["nb_offres"].sum()
        concentration = (top_5_offres / total_offres * 100) if total_offres > 0 else 0
        st.metric(
            "Concentration Top 5",
            f"{concentration:.1f}%",
            help="% d'offres concentrées dans les 5 premières villes",
        )

with col_stat4:
    # Moyenne offres/ville
    avg_offres = df_bassins_map["nb_offres"].mean()
    st.metric(
        "Moyenne offres/ville",
        f"{avg_offres:.1f}",
        help="Nombre moyen d'offres par ville",
    )
st.markdown("---")
#################################################################################################

# ============================================
# DISTRIBUTION RÉGIONS
# ============================================

col_left, col_right = st.columns([6, 4])

with col_left:
    st.subheader("Top 15 Régions")

    top_regions = df["region"].value_counts().head(15).reset_index()
    top_regions.columns = ["region", "count"]
    top_regions["percentage"] = top_regions["count"] / len(df) * 100

    fig_regions = px.bar(
        top_regions,
        x="count",
        y="region",
        orientation="h",
        text="count",
        color="percentage",
        color_continuous_scale="Plasma",
        hover_data=["percentage"],
    )

    fig_regions.update_traces(textposition="outside")

    fig_regions.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Nombre d'offres",
        yaxis_title="",
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
    )

    st.plotly_chart(fig_regions, use_container_width=True)

with col_right:
    st.subheader("Villes Principales")

    if region_selectionnee != "Toutes":
        top_cities = df_region["city"].value_counts().head(10)
    else:
        top_cities = df["city"].value_counts().head(10)

    fig_cities = go.Figure(
        go.Bar(
            x=top_cities.values,
            y=top_cities.index,
            orientation="h",
            marker_color=COLORS["accent"],
            text=top_cities.values,
            textposition="auto",
        )
    )

    fig_cities.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Offres",
        yaxis_title="",
        yaxis={"categoryorder": "total ascending"},
    )

    st.plotly_chart(fig_cities, use_container_width=True)

st.markdown("---")

# ============================================
# HEATMAP PROFILS × RÉGIONS
# ============================================

st.subheader("Heatmap Profils × Régions (Top 10)")

df_class = df[df["status"] == "classified"]

# Top 10 régions et top 10 profils
top_10_regions = df_class["region"].value_counts().head(10).index
top_10_profils = df_class["profil_assigned"].value_counts().head(10).index

df_heatmap = df_class[
    df_class["region"].isin(top_10_regions)
    & df_class["profil_assigned"].isin(top_10_profils)
]

# Crosstab
heatmap_data = pd.crosstab(df_heatmap["region"], df_heatmap["profil_assigned"])

fig_heatmap = px.imshow(
    heatmap_data,
    labels=dict(x="Profil", y="Région", color="Nombre"),
    x=heatmap_data.columns,
    y=heatmap_data.index,
    color_continuous_scale="Turbo",
    aspect="auto",
)

fig_heatmap.update_layout(
    template="plotly_dark", height=500, xaxis={"side": "bottom"}, font=dict(size=10)
)

fig_heatmap.update_xaxes(tickangle=-45)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("---")

# ============================================
# COMPARAISON RÉGIONS
# ============================================

st.subheader("Comparaison Régionale")

tab1, tab2 = st.tabs(["Profils", "Compétences"])

with tab1:
    st.markdown("#### Distribution Profils par Région")

    region_profil = st.selectbox(
        "Choisir région", top_10_regions.tolist(), key="profil_region"
    )

    df_region_profil = df_class[df_class["region"] == region_profil]

    profils_counts = df_region_profil["profil_assigned"].value_counts().head(8)

    fig_profils_r = go.Figure(
        data=[
            go.Pie(labels=profils_counts.index, values=profils_counts.values, hole=0.3)
        ]
    )

    fig_profils_r.update_layout(template="plotly_dark", height=400)

    st.plotly_chart(fig_profils_r, use_container_width=True)

with tab2:
    st.markdown("#### Top 5 Compétences par Région")

    region_comp = st.selectbox(
        "Choisir région", top_10_regions.tolist(), key="comp_region"
    )

    df_region_comp = df[df["region"] == region_comp]

    all_comp_r = []
    for comp_list in df_region_comp["competences_found"]:
        if isinstance(comp_list, list):
            all_comp_r.extend(comp_list)

    from collections import Counter

    comp_counts_r = Counter(all_comp_r)
    top_comp_r = pd.DataFrame(
        comp_counts_r.most_common(5), columns=["competence", "count"]
    )
    top_comp_r["percentage"] = top_comp_r["count"] / len(df_region_comp) * 100

    fig_comp_r = px.bar(
        top_comp_r,
        x="percentage",
        y="competence",
        orientation="h",
        text="count",
        color="percentage",
        color_continuous_scale="Blues",
    )

    fig_comp_r.update_traces(textposition="outside")

    fig_comp_r.update_layout(
        template="plotly_dark",
        height=300,
        xaxis_title="% Offres",
        yaxis_title="",
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
    )

    st.plotly_chart(fig_comp_r, use_container_width=True)

st.markdown("---")

# ============================================
# TABLEAU DÉTAILS
# ============================================

with st.expander(" Voir le tableau détaillé"):
    st.dataframe(
        df_region[
            [
                "title",
                "company_name",
                "city",
                "region",
                "profil_assigned",
                "salary_annual",
                "contract_type",
            ]
        ].head(100),
        use_container_width=True,
    )
