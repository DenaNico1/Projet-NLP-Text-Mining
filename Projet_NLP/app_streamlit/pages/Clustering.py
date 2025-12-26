"""Page 6 : Clustering 2D"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

st.set_page_config(page_title="Clustering", page_icon="🔬", layout="wide")
st.title("🔬 Clustering 2D des Offres")

# Charger données clustering
cluster_path = Path("../resultats_nlp/models/data_with_clusters.pkl")

if cluster_path.exists():
    with open(cluster_path, 'rb') as f:
        df_clusters = pickle.load(f)
    
    st.write(f"**{len(df_clusters)} offres analysées**")
    
    # Visualisation intégrée
    viz_path = Path("../resultats_nlp/visualisations/clustering_2d.html")
    if viz_path.exists():
        with open(viz_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=800, scrolling=True)
    
    # Analyse par cluster
    st.subheader("📊 Analyse par Cluster")
    
    cluster_id = st.selectbox("Choisir un cluster", sorted(df_clusters['cluster'].unique()))
    
    df_cluster = df_clusters[df_clusters['cluster'] == cluster_id]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Offres", len(df_cluster))
    col2.metric("% du Total", f"{len(df_cluster)/len(df_clusters)*100:.1f}%")
    
    if df_cluster['salary_annual'].notna().sum() > 0:
        col3.metric("Salaire Médian", f"{df_cluster['salary_annual'].median()/1000:.0f}k€")
    
    # Top mots du cluster
    st.write("**Titres d'offres dans ce cluster**")
    st.write(df_cluster['title'].value_counts().head(10))
    
else:
    st.warning("""
    ⚠️  Analyse de clustering non disponible
    
    Pour générer le clustering, lancez :
    ```
    python fichiers_analyses/6_embeddings_clustering.py
    ```
    """)