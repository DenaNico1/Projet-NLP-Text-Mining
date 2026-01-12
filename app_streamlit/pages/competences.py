"""
PAGE 4 : COMPÉTENCES
Réseau sémantique, recherche, recommandations
"""

import streamlit as st
import pandas as pd
import pickle
import streamlit.components.v1 as components
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, RESULTS_DIR, VIZ_DIR


from utils import get_data

df = get_data()

# Charger analyse compétences depuis CSV (TEMPORAIRE)
RESULTS_DIR = Path("../resultats_nlp")
df_comp = pd.read_csv(RESULTS_DIR / "competences_analysis.csv")

# Renommer 'count' en 'nb_offres'
df_comp = df_comp.rename(columns={"count": "nb_offres"})

st.title("Analyse des Compétences")
st.markdown("Réseau sémantique, clusters et co-occurrences")

st.markdown("---")

# RÉSEAU SÉMANTIQUE
st.subheader("Réseau Sémantique Interactif")

network_path = VIZ_DIR / "network" / "network_semantic_interactive.html"

if network_path.exists():
    with open(network_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=800, scrolling=True)
else:
    st.warning("Réseau non disponible. Exécuter 8_network_semantic.py")

st.markdown("---")

# # CARTE 3D UMAP
# st.subheader("Carte 3D Compétences (UMAP)")

# comp_3d_path = VIZ_DIR / '3d_interactive' / 'projector_competences_all_3d.html'

# if comp_3d_path.exists():
#     with open(comp_3d_path, 'r', encoding='utf-8') as f:
#         html_3d = f.read()
#     components.html(html_3d, height=800, scrolling=True)
# else:
#     st.info("Visualisation 3D disponible après exécution script 7")

# st.markdown("---")

# CARTE 3D UMAP
st.subheader("Carte 3D Top 100 Compétences (UMAP)")

comp_3d_path_lab = VIZ_DIR / "3d_interactive" / "projector_competences_labels_3d.html"

if comp_3d_path_lab.exists():
    with open(comp_3d_path_lab, "r", encoding="utf-8") as f:
        html_3d = f.read()
    components.html(html_3d, height=800, scrolling=True)
else:
    st.info("Visualisation 3D disponible après exécution script 7")

st.markdown("---")

# RECHERCHE COMPÉTENCE
st.subheader("Recherche de Compétences")

competences_list = sorted(df_comp["competence"].tolist())
comp_recherche = st.selectbox("Sélectionner une compétence", competences_list)

if comp_recherche:
    comp_info = df_comp[df_comp["competence"] == comp_recherche].iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Occurrences", f"{comp_info['nb_offres']}")
    with col2:
        st.metric("Cluster", f"{comp_info['cluster']}")
    with col3:
        pct = comp_info["nb_offres"] / len(df) * 100
        st.metric("% Offres", f"{pct:.1f}%")

# TOP COMPÉTENCES
st.markdown("---")
st.subheader(" Top 20 Compétences")

top_comp = df_comp.nlargest(20, "nb_offres")

import plotly.express as px

fig = px.bar(
    top_comp,
    x="nb_offres",
    y="competence",
    orientation="h",
    color="nb_offres",
    color_continuous_scale="Viridis",
)

fig.update_layout(
    template="plotly_dark",
    height=600,
    yaxis={"categoryorder": "total ascending"},
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)
