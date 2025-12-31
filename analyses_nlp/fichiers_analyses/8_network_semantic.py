"""
8. Réseau Sémantique de Compétences - Style Graph Interactif
Visualisation réseau avec mise en valeur des hubs centraux

Inspiré de visualisations réseau académiques
- Noeuds = Compétences (taille = fréquence)
- Liens = Co-occurrence dans offres (épaisseur = force)
- Hubs centraux en jaune (Python, SQL, Docker, etc.)
- Layout force-directed pour clustering naturel

Technologie: NetworkX + Plotly + PyVis (HTML interactif)

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json
from collections import defaultdict, Counter

# Réseau
import networkx as nx

# Visualisation
import plotly.graph_objects as go
import plotly.express as px

# PyVis pour réseau interactif HTML
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    print("⚠️  PyVis non disponible. Install: pip install pyvis")
    PYVIS_AVAILABLE = False


# ============================================
# CONFIGURATION
# ============================================

RESULTS_DIR = Path('../../resultats_nlp')
VIZ_DIR = RESULTS_DIR / 'visualisations' / 'network'
MODELS_DIR = RESULTS_DIR / 'models'

# Créer répertoire
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres réseau
MIN_COOCCURRENCE = 10  # Minimum co-occurrences pour lien
TOP_N_NODES = 150  # Top N compétences à afficher
HUB_THRESHOLD = 10  # Seuil degré pour être considéré hub

# Couleurs
COLOR_NORMAL = '#17becf'  # Cyan
COLOR_HUB = '#FFD700'     # Jaune (or)
COLOR_EDGE = '#cccccc'    # Gris clair


# ============================================
# CONSTRUCTION RÉSEAU
# ============================================

def build_network_graph(df, min_cooccur=MIN_COOCCURRENCE, top_n=TOP_N_NODES):
    """
    Construit graphe NetworkX à partir co-occurrences
    
    Returns:
        G: nx.Graph
        node_info: dict avec métadonnées noeuds
    """
    print(f"\n🔗 Construction réseau (min_cooccur={min_cooccur}, top_n={top_n})...")
    
    # ========================================
    # 1. CALCULER CO-OCCURRENCES
    # ========================================
    
    cooccur = defaultdict(lambda: defaultdict(int))
    comp_counts = Counter()
    
    for comp_list in df['competences_found']:
        if not isinstance(comp_list, list) or len(comp_list) < 2:
            continue
        
        # Comptage fréquence
        comp_counts.update(comp_list)
        
        # Co-occurrences (paires)
        for i, comp1 in enumerate(comp_list):
            for comp2 in comp_list[i+1:]:
                if comp1 < comp2:
                    cooccur[comp1][comp2] += 1
                else:
                    cooccur[comp2][comp1] += 1
    
    # ========================================
    # 2. FILTRER TOP COMPÉTENCES
    # ========================================
    
    top_comps = [comp for comp, _ in comp_counts.most_common(top_n)]
    
    print(f"   ✅ Top {len(top_comps)} compétences sélectionnées")
    print(f"   📊 Top 10:")
    for i, (comp, count) in enumerate(comp_counts.most_common(10), 1):
        print(f"      {i}. {comp}: {count}×")
    
    # ========================================
    # 3. CRÉER GRAPHE
    # ========================================
    
    G = nx.Graph()
    
    # Ajouter noeuds
    for comp in top_comps:
        G.add_node(comp, count=comp_counts[comp])
    
    # Ajouter arêtes (co-occurrences)
    n_edges = 0
    for comp1, comp2_dict in cooccur.items():
        if comp1 not in top_comps:
            continue
        
        for comp2, weight in comp2_dict.items():
            if comp2 not in top_comps:
                continue
            
            if weight >= min_cooccur:
                G.add_edge(comp1, comp2, weight=weight)
                n_edges += 1
    
    print(f"   ✅ Graphe construit:")
    print(f"      Noeuds: {G.number_of_nodes()}")
    print(f"      Arêtes: {G.number_of_edges()}")
    
    # ========================================
    # 4. CALCULER MÉTRIQUES CENTRALITÉ
    # ========================================
    
    print("\n📊 Calcul métriques centralité...")
    
    # Degré (nombre connexions)
    degree_centrality = nx.degree_centrality(G)
    
    # Betweenness (importance intermédiaire)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # PageRank (importance globale)
    pagerank = nx.pagerank(G)
    
    # Stocker métadonnées
    node_info = {}
    for node in G.nodes():
        node_info[node] = {
            'count': G.nodes[node]['count'],
            'degree': G.degree(node),
            'degree_centrality': degree_centrality[node],
            'betweenness': betweenness_centrality[node],
            'pagerank': pagerank[node]
        }
    
    # Identifier hubs (degré élevé)
    degrees = [G.degree(node) for node in G.nodes()]
    threshold = np.percentile(degrees, 85)  # Top 15%
    
    hubs = [node for node in G.nodes() if G.degree(node) >= threshold]
    
    print(f"   ✅ Hubs identifiés ({len(hubs)}):")
    hubs_sorted = sorted(hubs, key=lambda x: G.degree(x), reverse=True)
    for i, hub in enumerate(hubs_sorted[:15], 1):
        print(f"      {i}. {hub}: degré {G.degree(hub)}")
    
    # Marquer hubs
    for node in G.nodes():
        node_info[node]['is_hub'] = node in hubs
    
    return G, node_info


# ============================================
# VISUALISATIONS
# ============================================

def viz_network_plotly_2d(G, node_info, output_file='network_semantic_2d.html'):
    """
    Réseau 2D Plotly style académique (comme l'image)
    """
    print(f"\n   📊 Réseau 2D Plotly ({output_file})...")
    
    # ========================================
    # 1. LAYOUT (position noeuds)
    # ========================================
    
    # Spring layout (force-directed)
    pos = nx.spring_layout(
        G,
        k=0.5,           # Distance optimale entre noeuds
        iterations=50,   # Nombre itérations
        seed=42
    )
    
    # ========================================
    # 2. PRÉPARER ARÊTES
    # ========================================
    
    edge_traces = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        
        # Épaisseur ligne proportionnelle au poids
        width = 0.3 + (weight / 50) * 2  # 0.3 à 2.3
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=COLOR_EDGE),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # ========================================
    # 3. PRÉPARER NOEUDS
    # ========================================
    
    # Séparer hubs et normaux
    nodes_normal = [n for n in G.nodes() if not node_info[n]['is_hub']]
    nodes_hub = [n for n in G.nodes() if node_info[n]['is_hub']]
    
    # Noeuds normaux (cyan)
    node_trace_normal = go.Scatter(
        x=[pos[node][0] for node in nodes_normal],
        y=[pos[node][1] for node in nodes_normal],
        mode='markers+text',
        text=[node for node in nodes_normal],
        textposition='top center',
        textfont=dict(size=9, color='#555'),
        hovertext=[
            f"<b>{node}</b><br>"
            f"Occurrences: {node_info[node]['count']}×<br>"
            f"Connexions: {node_info[node]['degree']}<br>"
            f"PageRank: {node_info[node]['pagerank']:.4f}"
            for node in nodes_normal
        ],
        hoverinfo='text',
        marker=dict(
            size=[8 + np.log1p(node_info[node]['count']) * 2 for node in nodes_normal],
            color=COLOR_NORMAL,
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        name='Compétences',
        showlegend=True
    )
    
    # Noeuds hubs (jaune)
    node_trace_hub = go.Scatter(
        x=[pos[node][0] for node in nodes_hub],
        y=[pos[node][1] for node in nodes_hub],
        mode='markers+text',
        text=[node for node in nodes_hub],
        textposition='top center',
        textfont=dict(size=13, color='#333', family='Arial Black'),
        hovertext=[
            f"<b>🌟 HUB: {node}</b><br>"
            f"Occurrences: {node_info[node]['count']}×<br>"
            f"Connexions: {node_info[node]['degree']}<br>"
            f"PageRank: {node_info[node]['pagerank']:.4f}<br>"
            f"Betweenness: {node_info[node]['betweenness']:.4f}"
            for node in nodes_hub
        ],
        hoverinfo='text',
        marker=dict(
            size=[15 + np.log1p(node_info[node]['count']) * 3 for node in nodes_hub],
            color=COLOR_HUB,
            opacity=1.0,
            line=dict(width=2, color='#FFA500')
        ),
        name='Hubs Centraux',
        showlegend=True
    )
    
    # ========================================
    # 4. CRÉER FIGURE
    # ========================================
    
    fig = go.Figure(data=edge_traces + [node_trace_normal, node_trace_hub])
    
    fig.update_layout(
        title={
            'text': 'Réseau Sémantique des Compétences Data/IA<br><sub>Taille = Fréquence | Couleur jaune = Hubs centraux | Liens = Co-occurrence</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#fafafa',
        width=1400,
        height=900
    )
    
    # Sauvegarder
    output_path = VIZ_DIR / output_file
    fig.write_html(output_path)
    
    print(f"      ✅ Sauvegardé: {output_file}")
    
    return fig


def viz_network_pyvis_interactive(G, node_info, output_file='network_semantic_interactive.html'):
    """
    Réseau interactif PyVis (physique temps réel)
    """
    if not PYVIS_AVAILABLE:
        print("   ⚠️  PyVis non disponible, skip")
        return None
    
    print(f"\n   📊 Réseau PyVis interactif ({output_file})...")
    
    # Créer réseau PyVis
    net = Network(
        height='900px',
        width='100%',
        bgcolor='#fafafa',
        font_color='#333',
        notebook=False
    )
    
    # Options physique
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)
    
    # Ajouter noeuds
    for node in G.nodes():
        info = node_info[node]
        
        # Couleur
        color = COLOR_HUB if info['is_hub'] else COLOR_NORMAL
        
        # Taille
        size = 10 + np.log1p(info['count']) * 5
        if info['is_hub']:
            size *= 1.5
        
        # Label
        label = node
        if info['is_hub']:
            label = f"⭐ {node}"
        
        # Tooltip
        title = (
            f"<b>{node}</b><br>"
            f"Occurrences: {info['count']}×<br>"
            f"Connexions: {info['degree']}<br>"
            f"PageRank: {info['pagerank']:.4f}"
        )
        
        net.add_node(
            node,
            label=label,
            size=size,
            color=color,
            title=title,
            font={'size': 14 if info['is_hub'] else 10}
        )
    
    # Ajouter arêtes
    for edge in G.edges(data=True):
        weight = edge[2]['weight']
        
        net.add_edge(
            edge[0],
            edge[1],
            value=weight / 10,  # Épaisseur
            color=COLOR_EDGE
        )
    
    # Sauvegarder
    output_path = VIZ_DIR / output_file
    net.save_graph(str(output_path))
    
    print(f"      ✅ Sauvegardé: {output_file}")
    
    return net


def viz_network_3d_plotly(G, node_info, output_file='network_semantic_3d.html'):
    """
    Réseau 3D Plotly
    """
    print(f"\n   📊 Réseau 3D Plotly ({output_file})...")
    
    # Layout 3D
    pos_3d = nx.spring_layout(G, dim=3, k=0.5, iterations=50, seed=42)
    
    # Arêtes
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        
        edge_trace = go.Scatter3d(
            x=[x0, x1, None],
            y=[y0, y1, None],
            z=[z0, z1, None],
            mode='lines',
            line=dict(width=1, color=COLOR_EDGE),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Noeuds
    nodes_normal = [n for n in G.nodes() if not node_info[n]['is_hub']]
    nodes_hub = [n for n in G.nodes() if node_info[n]['is_hub']]
    
    node_trace_normal = go.Scatter3d(
        x=[pos_3d[n][0] for n in nodes_normal],
        y=[pos_3d[n][1] for n in nodes_normal],
        z=[pos_3d[n][2] for n in nodes_normal],
        mode='markers+text',
        text=[n for n in nodes_normal],
        textfont=dict(size=8),
        marker=dict(
            size=[5 + np.log1p(node_info[n]['count']) for n in nodes_normal],
            color=COLOR_NORMAL,
            opacity=0.7
        ),
        name='Compétences'
    )
    
    node_trace_hub = go.Scatter3d(
        x=[pos_3d[n][0] for n in nodes_hub],
        y=[pos_3d[n][1] for n in nodes_hub],
        z=[pos_3d[n][2] for n in nodes_hub],
        mode='markers+text',
        text=[n for n in nodes_hub],
        textfont=dict(size=12, color='#333'),
        marker=dict(
            size=[10 + np.log1p(node_info[n]['count']) * 2 for n in nodes_hub],
            color=COLOR_HUB,
            opacity=1.0
        ),
        name='Hubs'
    )
    
    fig = go.Figure(data=edge_traces + [node_trace_normal, node_trace_hub])
    
    fig.update_layout(
        title='Réseau Sémantique 3D - Compétences',
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            zaxis=dict(showgrid=False, showticklabels=False)
        ),
        width=1400,
        height=900
    )
    
    output_path = VIZ_DIR / output_file
    fig.write_html(output_path)
    
    print(f"      ✅ Sauvegardé: {output_file}")
    
    return fig


def export_network_stats(G, node_info, output_file='network_stats.json'):
    """
    Exporte statistiques réseau
    """
    print(f"\n   📊 Export statistiques ({output_file})...")
    
    stats = {
        'global': {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G)
        },
        'hubs': [
            {
                'competence': node,
                'count': node_info[node]['count'],
                'degree': node_info[node]['degree'],
                'pagerank': node_info[node]['pagerank'],
                'betweenness': node_info[node]['betweenness']
            }
            for node in G.nodes()
            if node_info[node]['is_hub']
        ],
        'top_edges': [
            {
                'comp1': edge[0],
                'comp2': edge[1],
                'weight': edge[2]['weight']
            }
            for edge in sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:20]
        ]
    }
    
    # Sauvegarder
    output_path = VIZ_DIR / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"      ✅ Sauvegardé: {output_file}")


# ============================================
# MAIN
# ============================================

def main():
    print("="*70)
    print("🌐 RÉSEAU SÉMANTIQUE COMPÉTENCES - STYLE GRAPH")
    print("="*70)
    print(f"📁 Répertoire: {VIZ_DIR}")
    
    # Charger données
    print(f"\n📥 Chargement data_with_profiles.pkl...")
    with open(MODELS_DIR / 'data_with_profiles.pkl', 'rb') as f:
        df = pickle.load(f)
    
    print(f"   ✅ {len(df)} offres chargées")
    
    # ========================================
    # 1. CONSTRUCTION RÉSEAU
    # ========================================
    
    G, node_info = build_network_graph(
        df,
        min_cooccur=MIN_COOCCURRENCE,
        top_n=TOP_N_NODES
    )
    
    # ========================================
    # 2. VISUALISATIONS
    # ========================================
    
    print("\n🎨 Génération visualisations...")
    
    # 2D style image (principal)
    viz_network_plotly_2d(G, node_info, 'network_semantic_2d.html')
    
    # Interactif PyVis
    if PYVIS_AVAILABLE:
        viz_network_pyvis_interactive(G, node_info, 'network_semantic_interactive.html')
    
    # 3D
    viz_network_3d_plotly(G, node_info, 'network_semantic_3d.html')
    
    # ========================================
    # 3. EXPORT STATS
    # ========================================
    
    export_network_stats(G, node_info, 'network_stats.json')
    
    # ========================================
    # 4. INDEX HTML
    # ========================================
    
    print("\n📄 Génération index.html...")
    
    html_index = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Réseau Sémantique Compétences</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; }}
        .viz-card {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .btn {{
            display: inline-block;
            background: #17becf;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin: 10px 5px;
        }}
        .btn:hover {{ background: #128ba3; }}
        .stats {{
            background: #fafafa;
            padding: 15px;
            border-left: 4px solid #FFD700;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <h1>🌐 Réseau Sémantique des Compétences</h1>
    
    <div class="stats">
        <h3>📊 Statistiques Réseau</h3>
        <ul>
            <li><b>Noeuds:</b> {G.number_of_nodes()} compétences</li>
            <li><b>Liens:</b> {G.number_of_edges()} co-occurrences</li>
            <li><b>Hubs:</b> {len([n for n in G.nodes() if node_info[n]['is_hub']])} compétences centrales</li>
        </ul>
    </div>
    
    <div class="viz-card">
        <h2>📍 Vue 2D (Recommandé)</h2>
        <p>Réseau style académique avec hubs en jaune. Layout force-directed.</p>
        <a href="network_semantic_2d.html" class="btn">Ouvrir ➜</a>
    </div>
    
    {"<div class='viz-card'><h2>⚡ Vue Interactive (PyVis)</h2><p>Physique temps réel. Drag, zoom, hover.</p><a href='network_semantic_interactive.html' class='btn'>Ouvrir ➜</a></div>" if PYVIS_AVAILABLE else ""}
    
    <div class="viz-card">
        <h2>🎲 Vue 3D</h2>
        <p>Exploration 3D du réseau.</p>
        <a href="network_semantic_3d.html" class="btn">Ouvrir ➜</a>
    </div>
    
    <p style="color: #999; margin-top: 40px;">Projet NLP Text Mining - Master SISE</p>
</body>
</html>
"""
    
    with open(VIZ_DIR / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html_index)
    
    print(f"   ✅ index.html")
    
    # ========================================
    # RÉSUMÉ
    # ========================================
    
    print("\n" + "="*70)
    print("✅ RÉSEAU SÉMANTIQUE TERMINÉ !")
    print("="*70)
    
    print(f"\n📁 Fichiers créés dans: {VIZ_DIR}")
    print(f"   - network_semantic_2d.html (⭐ principal)")
    if PYVIS_AVAILABLE:
        print(f"   - network_semantic_interactive.html")
    print(f"   - network_semantic_3d.html")
    print(f"   - network_stats.json")
    print(f"   - index.html")
    
    print(f"\n🌐 Ouvrir: {VIZ_DIR / 'index.html'}")


if __name__ == "__main__":
    main()