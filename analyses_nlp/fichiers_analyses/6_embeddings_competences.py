"""
6. Embeddings Compétences - Carte sémantique & Co-occurrence
Analyse approfondie des 600+ compétences techniques

Objectifs:
1. Carte sémantique 2D/3D des compétences (proximité sémantique)
2. Analyse co-occurrence (quelles compétences vont ensemble)
3. Clusters technologiques (langages, outils, cloud, frameworks)
4. Profils compétences par métier (signature sémantique)
5. Réseau de compétences (graphe interactif)
6. Recommandations compétences (si tu sais X, apprends Y)

Installation requise:
pip install sentence-transformers umap-learn networkx plotly scikit-learn

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
from collections import Counter, defaultdict

# Embeddings
from sentence_transformers import SentenceTransformer

# Réduction dimensionnalité
import umap

# Clustering
from sklearn.cluster import KMeans, DBSCAN

# Réseau
import networkx as nx

# Visualisation
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Métriques
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from utils import ResultSaver


# ============================================
# CONFIGURATION
# ============================================

RESULTS_DIR = Path('../../resultats_nlp')
VIZ_DIR = RESULTS_DIR / 'visualisations'
MODELS_DIR = RESULTS_DIR / 'models'

# Modèle embeddings
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

# Paramètres UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Paramètres clustering
N_CLUSTERS_COMPETENCES = 10  # Clusters technologiques

# Paramètres co-occurrence
MIN_COOCCURRENCE = 5  # Minimum co-occurrences pour lien


# ============================================
# EXTRACTION COMPÉTENCES
# ============================================

def extract_all_competences(df):
    """Extrait liste unique de toutes les compétences"""
    print("\n📋 Extraction compétences uniques...")
    
    all_competences = []
    
    for comp_list in df['competences_found']:
        if isinstance(comp_list, list):
            all_competences.extend(comp_list)
    
    # Comptage
    comp_counts = Counter(all_competences)
    
    # DataFrame
    df_comp = pd.DataFrame([
        {'competence': comp, 'count': count}
        for comp, count in comp_counts.items()
    ]).sort_values('count', ascending=False)
    
    print(f"   ✅ {len(df_comp)} compétences uniques trouvées")
    print(f"   📊 Top 10:")
    for i, row in df_comp.head(10).iterrows():
        print(f"      {i+1}. {row['competence']}: {row['count']}×")
    
    return df_comp


def build_cooccurrence_matrix(df, min_count=MIN_COOCCURRENCE):
    """Construit matrice co-occurrence compétences"""
    print(f"\n🔗 Calcul co-occurrences (min={min_count})...")
    
    # Matrice co-occurrence
    cooccur = defaultdict(lambda: defaultdict(int))
    
    for comp_list in df['competences_found']:
        if not isinstance(comp_list, list) or len(comp_list) < 2:
            continue
        
        # Toutes paires dans l'offre
        for i, comp1 in enumerate(comp_list):
            for comp2 in comp_list[i+1:]:
                # Ordre alphabétique pour éviter doublons
                if comp1 < comp2:
                    cooccur[comp1][comp2] += 1
                else:
                    cooccur[comp2][comp1] += 1
    
    # Convertir en liste
    edges = []
    for comp1, comp2_dict in cooccur.items():
        for comp2, count in comp2_dict.items():
            if count >= min_count:
                edges.append({
                    'comp1': comp1,
                    'comp2': comp2,
                    'weight': count
                })
    
    df_edges = pd.DataFrame(edges).sort_values('weight', ascending=False)
    
    print(f"   ✅ {len(df_edges)} paires co-occurrentes (>= {min_count}×)")
    print(f"   📊 Top 10 paires:")
    for i, row in df_edges.head(10).iterrows():
        print(f"      {i+1}. {row['comp1']} ↔ {row['comp2']}: {row['weight']}×")
    
    return df_edges


# ============================================
# EMBEDDINGS COMPÉTENCES
# ============================================

class CompetenceEmbeddingAnalyzer:
    """Analyse embeddings compétences"""
    
    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"\n🤖 Chargement modèle embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"   ✅ Modèle chargé: {self.model.get_sentence_embedding_dimension()} dimensions")
        
        self.embeddings = None
        self.competences = None
    
    def generate_embeddings(self, competences_list):
        """Génère embeddings pour liste compétences"""
        print(f"\n📊 Génération embeddings pour {len(competences_list)} compétences...")
        
        self.competences = competences_list
        
        # Encoder
        self.embeddings = self.model.encode(
            competences_list,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"   ✅ Embeddings générés: shape {self.embeddings.shape}")
        
        return self.embeddings
    
    def compute_similarity_matrix(self):
        """Calcule matrice similarité cosinus"""
        print("\n🔢 Calcul matrice similarité...")
        
        sim_matrix = cosine_similarity(self.embeddings)
        
        print(f"   ✅ Matrice similarité: shape {sim_matrix.shape}")
        
        return sim_matrix
    
    def reduce_umap_2d(self):
        """Réduction UMAP 2D"""
        print("\n🔵 Réduction UMAP 2D compétences...")
        
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric='cosine',
            random_state=42
        )
        
        umap_coords = reducer.fit_transform(self.embeddings)
        
        print(f"   ✅ UMAP 2D: shape {umap_coords.shape}")
        
        return umap_coords
    
    def reduce_umap_3d(self):
        """Réduction UMAP 3D"""
        print("\n🔵 Réduction UMAP 3D compétences...")
        
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric='cosine',
            random_state=42
        )
        
        umap_coords = reducer.fit_transform(self.embeddings)
        
        print(f"   ✅ UMAP 3D: shape {umap_coords.shape}")
        
        return umap_coords
    
    def cluster_competences(self, n_clusters=N_CLUSTERS_COMPETENCES):
        """Clustering compétences"""
        print(f"\n🔴 Clustering compétences (k={n_clusters})...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        
        print(f"   ✅ {n_clusters} clusters trouvés")
        
        return labels
    
    def get_most_similar(self, competence, top_n=10):
        """Trouve compétences les plus similaires"""
        if competence not in self.competences:
            return []
        
        idx = self.competences.index(competence)
        emb = self.embeddings[idx].reshape(1, -1)
        
        similarities = cosine_similarity(emb, self.embeddings)[0]
        top_indices = similarities.argsort()[-top_n-1:-1][::-1]  # Exclut elle-même
        
        results = [
            (self.competences[i], similarities[i])
            for i in top_indices
        ]
        
        return results


# ============================================
# PROFILS COMPÉTENCES PAR MÉTIER
# ============================================

def compute_profil_competence_signatures(df, analyzer):
    """Calcule signature compétences de chaque profil métier"""
    print("\n👥 Calcul signatures compétences par profil...")
    
    df_class = df[df['status'] == 'classified']
    profils = df_class['profil_assigned'].unique()
    
    signatures = {}
    
    for profil in profils:
        df_profil = df_class[df_class['profil_assigned'] == profil]
        
        # Toutes compétences du profil
        all_comp = []
        for comp_list in df_profil['competences_found']:
            if isinstance(comp_list, list):
                all_comp.extend(comp_list)
        
        if not all_comp:
            continue
        
        # Fréquences
        comp_counts = Counter(all_comp)
        top_comp = [comp for comp, _ in comp_counts.most_common(20)]  # Top 20
        
        # Embeddings moyens de ces compétences
        comp_in_model = [c for c in top_comp if c in analyzer.competences]
        
        if comp_in_model:
            indices = [analyzer.competences.index(c) for c in comp_in_model]
            signature_emb = analyzer.embeddings[indices].mean(axis=0)
            
            signatures[profil] = {
                'embedding': signature_emb,
                'top_competences': comp_in_model[:10],
                'n_offres': len(df_profil)
            }
    
    print(f"   ✅ Signatures calculées pour {len(signatures)} profils")
    
    return signatures


def compare_profils_similarity(signatures):
    """Compare similarité entre profils via leurs signatures compétences"""
    print("\n🔍 Comparaison similarité profils (via compétences)...")
    
    profils = list(signatures.keys())
    n = len(profils)
    
    sim_matrix = np.zeros((n, n))
    
    for i, p1 in enumerate(profils):
        for j, p2 in enumerate(profils):
            emb1 = signatures[p1]['embedding'].reshape(1, -1)
            emb2 = signatures[p2]['embedding'].reshape(1, -1)
            sim_matrix[i, j] = cosine_similarity(emb1, emb2)[0, 0]
    
    df_sim = pd.DataFrame(sim_matrix, index=profils, columns=profils)
    
    print(f"   ✅ Matrice similarité profils: {df_sim.shape}")
    
    return df_sim


# ============================================
# VISUALISATIONS
# ============================================

def viz_competences_map_2d(df_comp, umap_coords, clusters, saver):
    """Carte 2D compétences avec clusters"""
    print("\n   📊 Carte 2D compétences...")
    
    df_viz = df_comp.copy()
    df_viz['UMAP_1'] = umap_coords[:, 0]
    df_viz['UMAP_2'] = umap_coords[:, 1]
    df_viz['Cluster'] = clusters.astype(str)
    
    fig = px.scatter(
        df_viz,
        x='UMAP_1',
        y='UMAP_2',
        color='Cluster',
        size='count',
        hover_data=['competence', 'count'],
        text='competence',
        title='Carte Sémantique des Compétences (UMAP 2D)',
        height=800,
        width=1200
    )
    
    fig.update_traces(textposition='top center', textfont_size=8)
    
    saver.save_visualization(fig, 'competences_map_2d.html')


def viz_competences_map_3d(df_comp, umap_coords, clusters, saver):
    """Carte 3D compétences interactive"""
    print("\n   📊 Carte 3D compétences...")
    
    df_viz = df_comp.copy()
    df_viz['UMAP_1'] = umap_coords[:, 0]
    df_viz['UMAP_2'] = umap_coords[:, 1]
    df_viz['UMAP_3'] = umap_coords[:, 2]
    df_viz['Cluster'] = clusters.astype(str)
    
    fig = px.scatter_3d(
        df_viz,
        x='UMAP_1',
        y='UMAP_2',
        z='UMAP_3',
        color='Cluster',
        size='count',
        hover_data=['competence', 'count'],
        text='competence',
        title='Carte Sémantique 3D Interactive des Compétences',
        height=900,
        width=1200
    )
    
    fig.update_traces(textfont_size=7)
    
    saver.save_visualization(fig, 'competences_map_3d.html')


def viz_cooccurrence_heatmap(df_edges, top_n=30, saver=None):
    """Heatmap co-occurrence top compétences"""
    print("\n   📊 Heatmap co-occurrence...")
    
    # Top compétences par total co-occurrences
    comp_totals = defaultdict(int)
    for _, row in df_edges.iterrows():
        comp_totals[row['comp1']] += row['weight']
        comp_totals[row['comp2']] += row['weight']
    
    top_comps = sorted(comp_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_comp_names = [c for c, _ in top_comps]
    
    # Matrice
    n = len(top_comp_names)
    matrix = np.zeros((n, n))
    
    comp_to_idx = {c: i for i, c in enumerate(top_comp_names)}
    
    for _, row in df_edges.iterrows():
        if row['comp1'] in comp_to_idx and row['comp2'] in comp_to_idx:
            i = comp_to_idx[row['comp1']]
            j = comp_to_idx[row['comp2']]
            matrix[i, j] = row['weight']
            matrix[j, i] = row['weight']  # Symétrique
    
    fig = px.imshow(
        matrix,
        x=top_comp_names,
        y=top_comp_names,
        color_continuous_scale='Blues',
        title=f'Co-occurrence des Top {top_n} Compétences',
        labels=dict(color="Co-occurrences"),
        height=700,
        width=900
    )
    
    fig.update_xaxes(tickangle=-45)
    
    if saver:
        saver.save_visualization(fig, 'competences_cooccurrence_heatmap.html')
    
    return fig


def viz_network_competences(df_edges, top_n=50, saver=None):
    """Réseau de compétences (graphe interactif)"""
    print("\n   📊 Réseau de compétences...")
    
    # Top edges
    df_top = df_edges.nlargest(top_n, 'weight')
    
    # Créer graphe NetworkX
    G = nx.Graph()
    
    for _, row in df_top.iterrows():
        G.add_edge(row['comp1'], row['comp2'], weight=row['weight'])
    
    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Edges
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Nodes
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        # Taille = degré
        node_sizes.append(G.degree(node) * 3)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            size=node_sizes,
            color='lightblue',
            line_width=2
        ),
        hoverinfo='text'
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Réseau des Top {top_n} Co-occurrences de Compétences',
                        showlegend=False,
                        hovermode='closest',
                        height=800,
                        width=1200,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    if saver:
        saver.save_visualization(fig, 'competences_network.html')
    
    return fig


def viz_profils_similarity_heatmap(df_sim, saver):
    """Heatmap similarité profils (via compétences)"""
    print("\n   📊 Heatmap similarité profils...")
    
    fig = px.imshow(
        df_sim,
        color_continuous_scale='RdBu_r',
        title='Similarité Sémantique entre Profils (via Compétences)',
        labels=dict(color="Similarité"),
        height=700,
        width=900
    )
    
    fig.update_xaxes(tickangle=-45)
    
    saver.save_visualization(fig, 'competences_profils_similarity.html')


def viz_top_similar_competences(analyzer, examples, saver):
    """Table top compétences similaires pour exemples"""
    print("\n   📊 Top compétences similaires...")
    
    results = []
    
    for comp in examples:
        similar = analyzer.get_most_similar(comp, top_n=10)
        
        for rank, (sim_comp, score) in enumerate(similar, 1):
            results.append({
                'Compétence': comp,
                'Rang': rank,
                'Similaire': sim_comp,
                'Similarité': score
            })
    
    df_sim = pd.DataFrame(results)
    
    # Sauvegarder CSV
    csv_path = VIZ_DIR / 'competences_similarities.csv'
    df_sim.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"      ✅ CSV sauvegardé: {csv_path.name}")
    
    # Visualisation
    fig = px.bar(
        df_sim[df_sim['Rang'] <= 5],  # Top 5 seulement
        x='Similaire',
        y='Similarité',
        color='Compétence',
        facet_col='Compétence',
        facet_col_wrap=2,
        title='Top 5 Compétences Similaires (Exemples)',
        height=600,
        width=1200
    )
    
    fig.update_xaxes(tickangle=-45)
    
    saver.save_visualization(fig, 'competences_similarity_examples.html')


def viz_clusters_competences_details(df_comp, clusters, saver):
    """Détails clusters compétences"""
    print("\n   📊 Détails clusters compétences...")
    
    df_viz = df_comp.copy()
    df_viz['Cluster'] = clusters
    
    # Top compétences par cluster
    cluster_details = []
    
    for cluster_id in sorted(df_viz['Cluster'].unique()):
        df_cluster = df_viz[df_viz['Cluster'] == cluster_id]
        top_comp = df_cluster.nlargest(10, 'count')
        
        cluster_details.append({
            'Cluster': cluster_id,
            'N_Competences': len(df_cluster),
            'Top_Competences': ', '.join(top_comp['competence'].tolist())
        })
    
    df_clusters = pd.DataFrame(cluster_details)
    
    # Sauvegarder
    csv_path = VIZ_DIR / 'competences_clusters_details.csv'
    df_clusters.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"      ✅ CSV sauvegardé: {csv_path.name}")
    
    # Bar chart tailles clusters
    fig = px.bar(
        df_clusters,
        x='Cluster',
        y='N_Competences',
        title='Taille des Clusters de Compétences',
        height=500
    )
    
    saver.save_visualization(fig, 'competences_clusters_sizes.html')


# ============================================
# MAIN
# ============================================

def main():
    print("="*70)
    print("🚀 ÉTAPE 6 : EMBEDDINGS COMPÉTENCES")
    print("="*70)
    print(f"📁 Répertoire résultats: {RESULTS_DIR}")
    
    # Charger données
    print(f"\n📥 Chargement data_with_profiles.pkl...")
    with open(MODELS_DIR / 'data_with_profiles.pkl', 'rb') as f:
        df = pickle.load(f)
    
    print(f"   ✅ Offres: {len(df)}")
    
    # Initialiser
    saver = ResultSaver(RESULTS_DIR)
    
    # ========================================
    # 1. EXTRACTION COMPÉTENCES
    # ========================================
    
    df_comp = extract_all_competences(df)
    
    # Garder compétences fréquentes (> 3 occurrences)
    df_comp_filtered = df_comp[df_comp['count'] >= 3].reset_index(drop=True)
    competences_list = df_comp_filtered['competence'].tolist()
    
    print(f"\n   ✅ {len(competences_list)} compétences retenues (>= 3 occurrences)")
    
    # Co-occurrence
    df_edges = build_cooccurrence_matrix(df)
    
    # ========================================
    # 2. EMBEDDINGS COMPÉTENCES
    # ========================================
    
    analyzer = CompetenceEmbeddingAnalyzer(model_name=EMBEDDING_MODEL)
    embeddings = analyzer.generate_embeddings(competences_list)
    
    # Matrice similarité
    sim_matrix = analyzer.compute_similarity_matrix()
    
    # Sauvegarder
    np.save(MODELS_DIR / 'competences_embeddings.npy', embeddings)
    np.save(MODELS_DIR / 'competences_similarity_matrix.npy', sim_matrix)
    
    # ========================================
    # 3. RÉDUCTION DIMENSIONNALITÉ
    # ========================================
    
    umap_2d = analyzer.reduce_umap_2d()
    umap_3d = analyzer.reduce_umap_3d()
    
    np.save(MODELS_DIR / 'competences_umap_2d.npy', umap_2d)
    np.save(MODELS_DIR / 'competences_umap_3d.npy', umap_3d)
    
    # ========================================
    # 4. CLUSTERING
    # ========================================
    
    clusters = analyzer.cluster_competences(n_clusters=N_CLUSTERS_COMPETENCES)
    
    # ========================================
    # 5. PROFILS COMPÉTENCES
    # ========================================
    
    signatures = compute_profil_competence_signatures(df, analyzer)
    df_profils_sim = compare_profils_similarity(signatures)
    
    # ========================================
    # 6. VISUALISATIONS
    # ========================================
    
    print("\n📊 Génération visualisations...")
    
    # Cartes compétences
    viz_competences_map_2d(df_comp_filtered, umap_2d, clusters, saver)
    viz_competences_map_3d(df_comp_filtered, umap_3d, clusters, saver)
    
    # Co-occurrence
    viz_cooccurrence_heatmap(df_edges, top_n=30, saver=saver)
    viz_network_competences(df_edges, top_n=50, saver=saver)
    
    # Similarité profils
    viz_profils_similarity_heatmap(df_profils_sim, saver)
    
    # Exemples similarité
    examples = ['python', 'sql', 'spark', 'docker', 'kubernetes']
    examples_available = [c for c in examples if c in competences_list]
    if examples_available:
        viz_top_similar_competences(analyzer, examples_available, saver)
    
    # Détails clusters
    viz_clusters_competences_details(df_comp_filtered, clusters, saver)
    
    # ========================================
    # 7. SAUVEGARDE
    # ========================================
    
    print("\n💾 Sauvegarde résultats...")
    
    # DataFrame compétences enrichi
    df_comp_enriched = df_comp_filtered.copy()
    df_comp_enriched['cluster'] = clusters
    df_comp_enriched['umap_x'] = umap_2d[:, 0]
    df_comp_enriched['umap_y'] = umap_2d[:, 1]
    
    df_comp_enriched.to_csv(
        RESULTS_DIR / 'competences_analysis.csv',
        index=False,
        encoding='utf-8-sig'
    )
    
    # Co-occurrence
    df_edges.to_csv(
        RESULTS_DIR / 'competences_cooccurrence.csv',
        index=False,
        encoding='utf-8-sig'
    )
    
    # Similarité profils
    df_profils_sim.to_csv(
        RESULTS_DIR / 'profils_similarity_competences.csv',
        encoding='utf-8-sig'
    )
    
    print("\n" + "="*70)
    print("✅ EMBEDDINGS COMPÉTENCES TERMINÉS !")
    print("="*70)
    
    print("\n📁 Fichiers créés:")
    print("   - competences_embeddings.npy")
    print("   - competences_similarity_matrix.npy")
    print("   - competences_umap_2d/3d.npy")
    print("   - competences_analysis.csv")
    print("   - competences_cooccurrence.csv")
    print("   - profils_similarity_competences.csv")
    print("   - 10 visualisations HTML interactives")


if __name__ == "__main__":
    main()