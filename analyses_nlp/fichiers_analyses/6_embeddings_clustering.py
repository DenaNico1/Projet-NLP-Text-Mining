"""
5. Embeddings & Clustering - Analyse sémantique offres emploi
Génération embeddings multilingues (FR+EN) + Réduction dimensionnalité + Clustering

Modèle: paraphrase-multilingual-MiniLM-L12-v2 (Hugging Face)
- 384 dimensions
- Support 50+ langues dont français et anglais
- Optimisé pour similarité sémantique

Pipeline:
1. Génération embeddings (titre + description)
2. Réduction dimensionnalité (UMAP 2D/3D, t-SNE)
3. Clustering (KMeans, HDBSCAN)
4. Visualisations interactives Plotly
5. Analyse cohérence avec profils classifiés

Installation requise:
pip install sentence-transformers umap-learn hdbscan plotly scikit-learn

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Embeddings
from sentence_transformers import SentenceTransformer

# Réduction dimensionnalité
import umap
from sklearn.manifold import TSNE

# Clustering
from sklearn.cluster import KMeans
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    print("⚠️  HDBSCAN non disponible. Install: pip install hdbscan")
    HDBSCAN_AVAILABLE = False

# Visualisation
import plotly.express as px

# Métriques
from sklearn.metrics import silhouette_score, davies_bouldin_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import ResultSaver


# ============================================
# CONFIGURATION
# ============================================

RESULTS_DIR = Path('../../resultats_nlp')
VIZ_DIR = RESULTS_DIR / 'visualisations'
MODELS_DIR = RESULTS_DIR / 'models'

# Paramètres embeddings
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'  # Multilingue FR+EN
BATCH_SIZE = 32

# Paramètres UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'cosine'

# Paramètres t-SNE
TSNE_PERPLEXITY = 30
TSNE_LEARNING_RATE = 200

# Paramètres clustering
KMEANS_N_CLUSTERS = 14  # Même nombre que profils classifiés
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES = 10


# ============================================
# CLASSE PRINCIPALE
# ============================================

class EmbeddingAnalyzer:
    """Analyse par embeddings et clustering"""
    
    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"\n🤖 Chargement modèle embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"   ✅ Modèle chargé: {self.model.get_sentence_embedding_dimension()} dimensions")
        
        self.embeddings = None
        self.umap_2d = None
        self.umap_3d = None
        self.tsne_2d = None
        
    def generate_embeddings(self, df, text_column='combined_text'):
        """
        Génère embeddings pour chaque offre
        
        Combine titre + description pour représentation riche
        """
        print(f"\n📊 Génération embeddings pour {len(df)} offres...")
        
        # Créer texte combiné si pas déjà fait
        if text_column not in df.columns:
            print("   📝 Création texte combiné (titre + description)...")
            df['combined_text'] = df.apply(
                lambda row: f"{row['title']}. {row.get('description', '')}",
                axis=1
            )
        
        # Générer embeddings
        texts = df[text_column].fillna('').tolist()
        
        print(f"   🔄 Encoding {len(texts)} textes (batch_size={BATCH_SIZE})...")
        self.embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"   ✅ Embeddings générés: shape {self.embeddings.shape}")
        
        return self.embeddings
    
    def reduce_umap_2d(self):
        """Réduction UMAP 2D pour visualisation"""
        print("\n🔵 Réduction UMAP 2D...")
        
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=42
        )
        
        self.umap_2d = reducer.fit_transform(self.embeddings)
        print(f"   ✅ UMAP 2D: shape {self.umap_2d.shape}")
        
        return self.umap_2d
    
    def reduce_umap_3d(self):
        """Réduction UMAP 3D pour visualisation interactive"""
        print("\n🔵 Réduction UMAP 3D...")
        
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=42
        )
        
        self.umap_3d = reducer.fit_transform(self.embeddings)
        print(f"   ✅ UMAP 3D: shape {self.umap_3d.shape}")
        
        return self.umap_3d
    
    def reduce_tsne_2d(self):
        """Réduction t-SNE 2D (plus lent mais complémentaire)"""
        print("\n🟠 Réduction t-SNE 2D...")
        
        tsne = TSNE(
            n_components=2,
            perplexity=TSNE_PERPLEXITY,
            learning_rate=TSNE_LEARNING_RATE,
            random_state=42,
            n_jobs=-1
        )
        
        self.tsne_2d = tsne.fit_transform(self.embeddings)
        print(f"   ✅ t-SNE 2D: shape {self.tsne_2d.shape}")
        
        return self.tsne_2d
    
    def cluster_kmeans(self, n_clusters=KMEANS_N_CLUSTERS):
        """Clustering KMeans"""
        print(f"\n🔴 Clustering KMeans (k={n_clusters})...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        labels = kmeans.fit_predict(self.embeddings)
        
        # Métriques
        silhouette = silhouette_score(self.embeddings, labels)
        davies_bouldin = davies_bouldin_score(self.embeddings, labels)
        
        print(f"   ✅ KMeans: {n_clusters} clusters")
        print(f"      Silhouette: {silhouette:.3f} (plus proche de 1 = mieux)")
        print(f"      Davies-Bouldin: {davies_bouldin:.3f} (plus proche de 0 = mieux)")
        
        return labels, {'silhouette': silhouette, 'davies_bouldin': davies_bouldin}
    
    def cluster_hdbscan(self):
        """Clustering HDBSCAN (trouve nombre clusters automatiquement)"""
        if not HDBSCAN_AVAILABLE:
            print("\n⚠️  HDBSCAN non disponible, skip")
            return None, None
        
        print("\n🟢 Clustering HDBSCAN...")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(self.embeddings)
        
        # Nombre de clusters (exclut bruit = -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"   ✅ HDBSCAN: {n_clusters} clusters trouvés")
        print(f"      Bruit: {n_noise} offres ({n_noise/len(labels)*100:.1f}%)")
        
        # Métriques (seulement sur offres non-bruit)
        if n_clusters > 1:
            mask_no_noise = labels != -1
            if mask_no_noise.sum() > 0:
                silhouette = silhouette_score(
                    self.embeddings[mask_no_noise],
                    labels[mask_no_noise]
                )
                davies_bouldin = davies_bouldin_score(
                    self.embeddings[mask_no_noise],
                    labels[mask_no_noise]
                )
                
                print(f"      Silhouette: {silhouette:.3f}")
                print(f"      Davies-Bouldin: {davies_bouldin:.3f}")
                
                metrics = {
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette': silhouette,
                    'davies_bouldin': davies_bouldin
                }
            else:
                metrics = {'n_clusters': n_clusters, 'n_noise': n_noise}
        else:
            metrics = {'n_clusters': n_clusters, 'n_noise': n_noise}
        
        return labels, metrics


# ============================================
# VISUALISATIONS
# ============================================

def viz_umap_2d_profils(df, umap_coords, saver):
    """Scatter UMAP 2D coloré par profils classifiés"""
    print("\n   📊 UMAP 2D par profils...")
    
    df_viz = df.copy()
    df_viz['UMAP_1'] = umap_coords[:, 0]
    df_viz['UMAP_2'] = umap_coords[:, 1]
    
    # Seulement offres classifiées
    df_class = df_viz[df_viz['status'] == 'classified']
    
    fig = px.scatter(
        df_class,
        x='UMAP_1',
        y='UMAP_2',
        color='profil_assigned',
        hover_data=['title', 'profil_score', 'profil_confidence'],
        title='Projection UMAP 2D des Offres Classifiées (par Profil)',
        labels={'profil_assigned': 'Profil'},
        height=700,
        width=1000
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    
    saver.save_visualization(fig, 'embeddings_umap_2d_profils.html')


def viz_umap_3d_profils(df, umap_coords, saver):
    """Scatter UMAP 3D interactif coloré par profils"""
    print("\n   📊 UMAP 3D par profils...")
    
    df_viz = df.copy()
    df_viz['UMAP_1'] = umap_coords[:, 0]
    df_viz['UMAP_2'] = umap_coords[:, 1]
    df_viz['UMAP_3'] = umap_coords[:, 2]
    
    df_class = df_viz[df_viz['status'] == 'classified']
    
    fig = px.scatter_3d(
        df_class,
        x='UMAP_1',
        y='UMAP_2',
        z='UMAP_3',
        color='profil_assigned',
        hover_data=['title', 'profil_score'],
        title='Projection UMAP 3D Interactive (par Profil)',
        labels={'profil_assigned': 'Profil'},
        height=800,
        width=1200
    )
    
    fig.update_traces(marker=dict(size=3, opacity=0.6))
    
    saver.save_visualization(fig, 'embeddings_umap_3d_profils.html')


def viz_tsne_2d_profils(df, tsne_coords, saver):
    """Scatter t-SNE 2D coloré par profils"""
    print("\n   📊 t-SNE 2D par profils...")
    
    df_viz = df.copy()
    df_viz['tSNE_1'] = tsne_coords[:, 0]
    df_viz['tSNE_2'] = tsne_coords[:, 1]
    
    df_class = df_viz[df_viz['status'] == 'classified']
    
    fig = px.scatter(
        df_class,
        x='tSNE_1',
        y='tSNE_2',
        color='profil_assigned',
        hover_data=['title', 'profil_score'],
        title='Projection t-SNE 2D des Offres Classifiées (par Profil)',
        labels={'profil_assigned': 'Profil'},
        height=700,
        width=1000
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    
    saver.save_visualization(fig, 'embeddings_tsne_2d_profils.html')


def viz_clusters_kmeans(df, umap_coords, kmeans_labels, saver):
    """Scatter UMAP 2D coloré par clusters KMeans"""
    print("\n   📊 Clusters KMeans...")
    
    df_viz = df.copy()
    df_viz['UMAP_1'] = umap_coords[:, 0]
    df_viz['UMAP_2'] = umap_coords[:, 1]
    df_viz['Cluster_KMeans'] = kmeans_labels.astype(str)
    
    fig = px.scatter(
        df_viz,
        x='UMAP_1',
        y='UMAP_2',
        color='Cluster_KMeans',
        hover_data=['title', 'profil_assigned'],
        title=f'Clustering KMeans (k={KMEANS_N_CLUSTERS}) - Projection UMAP 2D',
        labels={'Cluster_KMeans': 'Cluster'},
        height=700,
        width=1000
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    
    saver.save_visualization(fig, 'embeddings_clusters_kmeans.html')


def viz_clusters_hdbscan(df, umap_coords, hdbscan_labels, saver):
    """Scatter UMAP 2D coloré par clusters HDBSCAN"""
    print("\n   📊 Clusters HDBSCAN...")
    
    df_viz = df.copy()
    df_viz['UMAP_1'] = umap_coords[:, 0]
    df_viz['UMAP_2'] = umap_coords[:, 1]
    df_viz['Cluster_HDBSCAN'] = hdbscan_labels.astype(str)
    df_viz.loc[df_viz['Cluster_HDBSCAN'] == '-1', 'Cluster_HDBSCAN'] = 'Bruit'
    
    fig = px.scatter(
        df_viz,
        x='UMAP_1',
        y='UMAP_2',
        color='Cluster_HDBSCAN',
        hover_data=['title', 'profil_assigned'],
        title='Clustering HDBSCAN (automatique) - Projection UMAP 2D',
        labels={'Cluster_HDBSCAN': 'Cluster'},
        height=700,
        width=1000
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    
    saver.save_visualization(fig, 'embeddings_clusters_hdbscan.html')


def viz_confusion_clusters_profils(df, cluster_labels, cluster_name, saver):
    """Heatmap: Clusters vs Profils classifiés (cohérence)"""
    print(f"\n   📊 Confusion {cluster_name} vs Profils...")
    
    df_viz = df[df['status'] == 'classified'].copy()
    df_viz['Cluster'] = cluster_labels[df_viz.index]
    
    # Crosstab
    ct = pd.crosstab(
        df_viz['Cluster'],
        df_viz['profil_assigned'],
        normalize='index'  # Normaliser par cluster
    ) * 100  # Pourcentages
    
    fig = px.imshow(
        ct,
        labels=dict(x="Profil Classifié", y=f"Cluster {cluster_name}", color="% Offres"),
        x=ct.columns,
        y=ct.index,
        color_continuous_scale='Blues',
        title=f'Cohérence {cluster_name} vs Profils Classifiés (%)',
        height=600,
        width=1000
    )
    
    fig.update_xaxes(tickangle=-45)
    
    filename = f'embeddings_confusion_{cluster_name.lower()}.html'
    saver.save_visualization(fig, filename)


def viz_top_similar_offers(df, embeddings, saver, n_examples=5):
    """Affiche top offres similaires pour quelques exemples"""
    print("\n   📊 Top offres similaires...")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Prendre 5 offres aléatoires classifiées
    df_class = df[df['status'] == 'classified']
    sample_indices = df_class.sample(n=n_examples, random_state=42).index.tolist()
    
    results = []
    
    for idx in sample_indices:
        # Calculer similarité avec toutes les autres
        emb = embeddings[idx].reshape(1, -1)
        similarities = cosine_similarity(emb, embeddings)[0]
        
        # Top 6 (inclut l'offre elle-même en position 1)
        top_indices = similarities.argsort()[-6:][::-1]
        
        for rank, top_idx in enumerate(top_indices[1:], 1):  # Skip première (elle-même)
            results.append({
                'Offre_Source': df.loc[idx, 'title'][:50],
                'Profil_Source': df.loc[idx, 'profil_assigned'],
                'Rang': rank,
                'Offre_Similaire': df.loc[top_idx, 'title'][:50],
                'Profil_Similaire': df.loc[top_idx, 'profil_assigned'],
                'Similarité': similarities[top_idx]
            })
    
    df_sim = pd.DataFrame(results)
    
    # Sauvegarder CSV
    csv_path = VIZ_DIR / 'embeddings_top_similar.csv'
    df_sim.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"      ✅ CSV sauvegardé: {csv_path.name}")
    
    # Visualisation simple
    fig = px.bar(
        df_sim[df_sim['Rang'] == 1],  # Seulement top 1 pour lisibilité
        x='Offre_Source',
        y='Similarité',
        color='Profil_Source',
        hover_data=['Offre_Similaire', 'Profil_Similaire'],
        title='Top Offre la Plus Similaire (par embedding) pour 5 Exemples',
        height=500
    )
    
    fig.update_xaxes(tickangle=-45)
    
    saver.save_visualization(fig, 'embeddings_similarity_examples.html')


# ============================================
# MAIN
# ============================================

def main():
    print("="*70)
    print("🚀 ÉTAPE 5 : EMBEDDINGS & CLUSTERING")
    print("="*70)
    print(f"📁 Répertoire résultats: {RESULTS_DIR}")
    
    # Charger données
    print("\n📥 Chargement data_with_profiles.pkl...")
    with open(MODELS_DIR / 'data_with_profiles.pkl', 'rb') as f:
        df = pickle.load(f)
    
    print(f"   ✅ Offres: {len(df)}")
    print(f"   ✅ Classifiées: {(df['status'] == 'classified').sum()}")
    
    # Initialiser
    saver = ResultSaver(RESULTS_DIR)
    analyzer = EmbeddingAnalyzer(model_name=EMBEDDING_MODEL)
    
    # ========================================
    # 1. GÉNÉRATION EMBEDDINGS
    # ========================================
    
    # Créer texte combiné
    df['combined_text'] = df.apply(
        lambda row: f"{row['title']}. {row.get('description', '')}",
        axis=1
    )
    
    embeddings = analyzer.generate_embeddings(df, text_column='combined_text')
    
    # Sauvegarder embeddings
    embeddings_path = MODELS_DIR / 'embeddings.npy'
    np.save(embeddings_path, embeddings)
    print(f"\n💾 Embeddings sauvegardés: {embeddings_path}")
    
    # ========================================
    # 2. RÉDUCTION DIMENSIONNALITÉ
    # ========================================
    
    umap_2d = analyzer.reduce_umap_2d()
    umap_3d = analyzer.reduce_umap_3d()
    tsne_2d = analyzer.reduce_tsne_2d()
    
    # Sauvegarder
    np.save(MODELS_DIR / 'umap_2d.npy', umap_2d)
    np.save(MODELS_DIR / 'umap_3d.npy', umap_3d)
    np.save(MODELS_DIR / 'tsne_2d.npy', tsne_2d)
    
    # ========================================
    # 3. CLUSTERING
    # ========================================
    
    kmeans_labels, kmeans_metrics = analyzer.cluster_kmeans(n_clusters=KMEANS_N_CLUSTERS)
    
    if HDBSCAN_AVAILABLE:
        hdbscan_labels, hdbscan_metrics = analyzer.cluster_hdbscan()
    else:
        hdbscan_labels = None
    
    # Ajouter au DataFrame
    df['cluster_kmeans'] = kmeans_labels
    if hdbscan_labels is not None:
        df['cluster_hdbscan'] = hdbscan_labels
    
    # ========================================
    # 4. VISUALISATIONS
    # ========================================
    
    print("\n📊 Génération visualisations...")
    
    # Projections par profils
    viz_umap_2d_profils(df, umap_2d, saver)
    viz_umap_3d_profils(df, umap_3d, saver)
    viz_tsne_2d_profils(df, tsne_2d, saver)
    
    # Clusters
    viz_clusters_kmeans(df, umap_2d, kmeans_labels, saver)
    if hdbscan_labels is not None:
        viz_clusters_hdbscan(df, umap_2d, hdbscan_labels, saver)
    
    # Cohérence clusters vs profils
    viz_confusion_clusters_profils(df, kmeans_labels, 'KMeans', saver)
    if hdbscan_labels is not None:
        viz_confusion_clusters_profils(df, hdbscan_labels, 'HDBSCAN', saver)
    
    # Similarité
    viz_top_similar_offers(df, embeddings, saver, n_examples=5)
    
    # ========================================
    # 5. SAUVEGARDE FINALE
    # ========================================
    
    print("\n💾 Sauvegarde données enrichies...")
    
    # DataFrame avec clusters
    with open(MODELS_DIR / 'data_with_clusters.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    print("   ✅ data_with_clusters.pkl")
    
    # Métriques clustering
    metrics = {
        'kmeans': kmeans_metrics,
        'hdbscan': hdbscan_metrics if hdbscan_labels is not None else None
    }
    
    import json
    with open(RESULTS_DIR / 'clustering_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print("   ✅ clustering_metrics.json")
    
    print("\n" + "="*70)
    print("✅ EMBEDDINGS & CLUSTERING TERMINÉS !")
    print("="*70)
    
    print("\n📁 Fichiers créés:")
    print("   - embeddings.npy (représentations vectorielles)")
    print("   - umap_2d.npy / umap_3d.npy (projections UMAP)")
    print("   - tsne_2d.npy (projection t-SNE)")
    print("   - data_with_clusters.pkl (offres + clusters)")
    print("   - clustering_metrics.json (métriques qualité)")
    print("   - 9 visualisations HTML interactives")


if __name__ == "__main__":
    main()