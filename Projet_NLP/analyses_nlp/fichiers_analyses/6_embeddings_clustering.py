"""
6. Embeddings + Clustering Visuel
Crée des vecteurs et visualisation 2D des offres

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))
from utils import ResultSaver

def main():
    print("="*70)
    print("🔬 ÉTAPE 6 : EMBEDDINGS + CLUSTERING")
    print("="*70)
    
    saver = ResultSaver()
    
    with open('../resultats_nlp/models/data_with_topics.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Embeddings
    print("\n🔄 Calcul embeddings (peut prendre 5-10 min)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Limiter à 2000 pour rapidité
    df_sample = df.sample(min(2000, len(df)), random_state=42)
    embeddings = model.encode(df_sample['description_clean'].tolist(), show_progress_bar=True)
    
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # UMAP
    print("\n🗺️  Réduction dimensionnelle (UMAP)...")
    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords_2d = umap.fit_transform(embeddings)
    
    # Clustering
    print("\n📊 Clustering K-Means...")
    kmeans = KMeans(n_clusters=8, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Ajouter au DataFrame
    df_sample['x'] = coords_2d[:, 0]
    df_sample['y'] = coords_2d[:, 1]
    df_sample['cluster'] = clusters
    
    # Visualisation
    print("\n📊 Création visualisation interactive...")
    
    fig = px.scatter(
        df_sample,
        x='x',
        y='y',
        color='cluster',
        hover_data=['title', 'company_name', 'city'],
        title='Clustering des Offres Data/IA (UMAP + K-Means)',
        labels={'cluster': 'Cluster'},
        width=1200,
        height=800
    )
    
    saver.save_visualization(fig, 'clustering_2d.html')
    
    # Sauvegardes
    saver.save_numpy(embeddings, 'embeddings.npy')
    saver.save_numpy(coords_2d, 'umap_coords.npy')
    saver.save_pickle(df_sample, 'data_with_clusters.pkl')
    
    # Analyser les clusters
    results = {}
    for cluster_id in range(8):
        df_cluster = df_sample[df_sample['cluster'] == cluster_id]
        
        # Top mots
        all_tokens = [t for tokens in df_cluster['tokens'] for t in tokens]
        from collections import Counter
        counter = Counter(all_tokens)
        
        results[f"Cluster {cluster_id}"] = {
            'size': len(df_cluster),
            'salaire_median': float(df_cluster['salary_annual'].median()),
            'top_mots': counter.most_common(10)
        }
        
        print(f"\nCluster {cluster_id}: {len(df_cluster)} offres")
        print(f"  Top mots: {', '.join([w for w, _ in counter.most_common(5)])}")
    
    saver.save_json(results, 'clusters_analysis.json')
    
    print("\n✅ EMBEDDINGS + CLUSTERING TERMINÉS !")

if __name__ == "__main__":
    main()