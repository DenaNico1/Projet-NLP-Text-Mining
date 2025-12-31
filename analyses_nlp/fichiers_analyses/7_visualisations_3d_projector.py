"""
7. Visualisations 3D Interactives - Style TensorFlow Projector
Génère visualisations 3D immersives pour embeddings (offres + compétences)

Inspiré de: https://projector.tensorflow.org/
- Navigation 3D fluide (rotation, zoom, pan)
- Points colorés par catégorie
- Hover affiche labels
- Search/filter interactif
- Export standalone HTML

Technologie: Plotly 3D (WebGL)

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json

# Visualisation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ============================================
# CONFIGURATION
# ============================================

RESULTS_DIR = Path('../../resultats_nlp')
VIZ_DIR = RESULTS_DIR / 'visualisations' / '3d_interactive'
MODELS_DIR = RESULTS_DIR / 'models'

# Créer répertoire 3D
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Couleurs (palette distinctive)
COLORS_PROFILS = px.colors.qualitative.Plotly + px.colors.qualitative.Set3


# ============================================
# VISUALISATIONS 3D INTERACTIVES
# ============================================

def create_3d_scatter_advanced(
    coords_3d,
    labels,
    hover_texts,
    title,
    color_map=None,
    size_values=None,
    output_file='viz_3d.html'
):
    """
    Crée scatter 3D avancé style TensorFlow Projector
    
    Args:
        coords_3d: array (N, 3) - coordonnées 3D
        labels: list - catégories/profils
        hover_texts: list - textes hover détaillés
        title: str - titre visualisation
        color_map: dict - mapping label → couleur (optionnel)
        size_values: list - tailles points (optionnel)
        output_file: str - nom fichier HTML
    """
    
    # Préparer données
    df_viz = pd.DataFrame({
        'x': coords_3d[:, 0],
        'y': coords_3d[:, 1],
        'z': coords_3d[:, 2],
        'label': labels,
        'hover': hover_texts
    })
    
    if size_values is not None:
        df_viz['size'] = size_values
    else:
        df_viz['size'] = 3
    
    # Créer figure
    fig = go.Figure()
    
    # Ajouter points par catégorie (pour légende interactive)
    unique_labels = df_viz['label'].unique()
    
    for i, label in enumerate(unique_labels):
        df_label = df_viz[df_viz['label'] == label]
        
        # Couleur
        if color_map and label in color_map:
            color = color_map[label]
        else:
            color = COLORS_PROFILS[i % len(COLORS_PROFILS)]
        
        fig.add_trace(go.Scatter3d(
            x=df_label['x'],
            y=df_label['y'],
            z=df_label['z'],
            mode='markers',
            name=label,
            text=df_label['hover'],
            hovertemplate='<b>%{text}</b><br>' +
                          'X: %{x:.2f}<br>' +
                          'Y: %{y:.2f}<br>' +
                          'Z: %{z:.2f}<extra></extra>',
            marker=dict(
                size=df_label['size'],
                color=color,
                opacity=0.8,
                line=dict(width=0.5, color='white')
            )
        ))
    
    # Layout style TensorFlow Projector
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#333'}
        },
        scene=dict(
            xaxis=dict(
                title='',
                showgrid=True,
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='#f5f5f5',
                showticklabels=False
            ),
            yaxis=dict(
                title='',
                showgrid=True,
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='#f5f5f5',
                showticklabels=False
            ),
            zaxis=dict(
                title='',
                showgrid=True,
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='#f5f5f5',
                showticklabels=False
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            title='Catégories',
            x=1.02,
            y=0.5,
            font=dict(size=12)
        ),
        hovermode='closest',
        width=1400,
        height=900,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=0, r=200, t=80, b=0)
    )
    
    # Sauvegarder
    output_path = VIZ_DIR / output_file
    fig.write_html(
        output_path,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['toImage'],
            'scrollZoom': True
        }
    )
    
    print(f"   ✅ {output_file}")
    
    return fig


def create_3d_with_labels(
    coords_3d,
    labels,
    texts,
    title,
    show_text=True,
    text_size=8,
    output_file='viz_3d_labels.html'
):
    """
    Scatter 3D avec labels texte visibles (pour compétences)
    """
    
    df_viz = pd.DataFrame({
        'x': coords_3d[:, 0],
        'y': coords_3d[:, 1],
        'z': coords_3d[:, 2],
        'label': labels,
        'text': texts
    })
    
    # Créer figure
    fig = go.Figure()
    
    unique_labels = df_viz['label'].unique()
    
    for i, label in enumerate(unique_labels):
        df_label = df_viz[df_viz['label'] == label]
        
        color = COLORS_PROFILS[i % len(COLORS_PROFILS)]
        
        # Points
        fig.add_trace(go.Scatter3d(
            x=df_label['x'],
            y=df_label['y'],
            z=df_label['z'],
            mode='markers+text' if show_text else 'markers',
            name=label,
            text=df_label['text'] if show_text else None,
            textfont=dict(size=text_size, color=color),
            textposition='top center',
            hovertext=df_label['text'],
            hovertemplate='<b>%{hovertext}</b><extra></extra>',
            marker=dict(
                size=5,
                color=color,
                opacity=0.7,
                line=dict(width=0)
            )
        ))
    
    # Layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22}
        },
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
        ),
        legend=dict(x=1.02, y=0.5),
        width=1400,
        height=900,
        paper_bgcolor='white'
    )
    
    # Sauvegarder
    output_path = VIZ_DIR / output_file
    fig.write_html(output_path)
    
    print(f"   ✅ {output_file}")
    
    return fig


def create_3d_dual_view(
    coords_3d_1, labels_1, title_1,
    coords_3d_2, labels_2, title_2,
    output_file='viz_3d_dual.html'
):
    """
    Vue côte-à-côte de 2 visualisations 3D (ex: UMAP vs t-SNE)
    """
    
    from plotly.subplots import make_subplots
    
    # Créer subplot 1x2
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(title_1, title_2),
        horizontal_spacing=0.05
    )
    
    # Vue 1
    unique_labels_1 = pd.Series(labels_1).unique()
    for i, label in enumerate(unique_labels_1):
        mask = np.array(labels_1) == label
        color = COLORS_PROFILS[i % len(COLORS_PROFILS)]
        
        fig.add_trace(
            go.Scatter3d(
                x=coords_3d_1[mask, 0],
                y=coords_3d_1[mask, 1],
                z=coords_3d_1[mask, 2],
                mode='markers',
                name=label,
                legendgroup=label,
                marker=dict(size=3, color=color, opacity=0.7),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Vue 2
    unique_labels_2 = pd.Series(labels_2).unique()
    for i, label in enumerate(unique_labels_2):
        mask = np.array(labels_2) == label
        color = COLORS_PROFILS[i % len(COLORS_PROFILS)]
        
        fig.add_trace(
            go.Scatter3d(
                x=coords_3d_2[mask, 0],
                y=coords_3d_2[mask, 1],
                z=coords_3d_2[mask, 2],
                mode='markers',
                name=label,
                legendgroup=label,
                marker=dict(size=3, color=color, opacity=0.7),
                showlegend=False  # Déjà dans légende
            ),
            row=1, col=2
        )
    
    # Layout
    fig.update_layout(
        title='Comparaison Visualisations 3D',
        width=1600,
        height=800,
        scene1=dict(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            zaxis=dict(showgrid=False, showticklabels=False)
        ),
        scene2=dict(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            zaxis=dict(showgrid=False, showticklabels=False)
        )
    )
    
    # Sauvegarder
    output_path = VIZ_DIR / output_file
    fig.write_html(output_path)
    
    print(f"   ✅ {output_file}")
    
    return fig


def create_3d_animated_rotation(
    coords_3d,
    labels,
    hover_texts,
    title,
    output_file='viz_3d_animated.html'
):
    """
    Vue 3D avec rotation automatique (effet démo)
    """
    
    df_viz = pd.DataFrame({
        'x': coords_3d[:, 0],
        'y': coords_3d[:, 1],
        'z': coords_3d[:, 2],
        'label': labels,
        'hover': hover_texts
    })
    
    fig = go.Figure()
    
    unique_labels = df_viz['label'].unique()
    
    for i, label in enumerate(unique_labels):
        df_label = df_viz[df_viz['label'] == label]
        color = COLORS_PROFILS[i % len(COLORS_PROFILS)]
        
        fig.add_trace(go.Scatter3d(
            x=df_label['x'],
            y=df_label['y'],
            z=df_label['z'],
            mode='markers',
            name=label,
            text=df_label['hover'],
            hovertemplate='<b>%{text}</b><extra></extra>',
            marker=dict(size=4, color=color, opacity=0.8)
        ))
    
    # Animation rotation
    frames = []
    n_frames = 36  # 360° / 10° = 36 frames
    
    for i in range(n_frames):
        angle = i * 10  # 10° par frame
        eye_x = 2 * np.cos(np.radians(angle))
        eye_y = 2 * np.sin(np.radians(angle))
        
        frames.append(go.Frame(
            layout=dict(
                scene=dict(
                    camera=dict(eye=dict(x=eye_x, y=eye_y, z=1.5))
                )
            )
        ))
    
    fig.frames = frames
    
    # Boutons animation
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            zaxis=dict(showgrid=False, showticklabels=False),
            camera=dict(eye=dict(x=2, y=0, z=1.5))
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='▶ Rotation',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate'
                        }]
                    )
                ],
                x=0.1,
                y=1.1
            )
        ],
        width=1400,
        height=900
    )
    
    # Sauvegarder
    output_path = VIZ_DIR / output_file
    fig.write_html(output_path)
    
    print(f"   ✅ {output_file}")
    
    return fig


# ============================================
# MAIN
# ============================================

def main():
    print("="*70)
    print("🎨 VISUALISATIONS 3D INTERACTIVES - STYLE TENSORFLOW PROJECTOR")
    print("="*70)
    print(f"📁 Répertoire: {VIZ_DIR}")
    
    # ========================================
    # 1. CHARGER DONNÉES EMBEDDINGS OFFRES
    # ========================================
    
    print("\n📥 Chargement embeddings offres...")
    
    # Charger offres
    with open(MODELS_DIR / 'data_with_profiles.pkl', 'rb') as f:
        df_offres = pickle.load(f)
    
    # Charger UMAP 3D offres
    umap_3d_offres = np.load(MODELS_DIR / 'umap_3d.npy')
    
    # Préparer données offres classifiées
    df_class = df_offres[df_offres['status'] == 'classified'].copy()
    indices_class = df_class.index.tolist()
    
    coords_offres = umap_3d_offres[indices_class]
    labels_offres = df_class['profil_assigned'].tolist()
    
    # Textes hover enrichis
    hover_offres = df_class.apply(
        lambda row: f"{row['title']}<br>Profil: {row['profil_assigned']}<br>Score: {row['profil_score']:.1f}/10<br>Confiance: {row['profil_confidence']:.2f}",
        axis=1
    ).tolist()
    
    # Tailles basées sur score
    sizes_offres = (df_class['profil_score'] / 2).tolist()  # Score/2 pour tailles 0-5
    
    print(f"   ✅ {len(coords_offres)} offres chargées")
    
    # ========================================
    # 2. CHARGER DONNÉES EMBEDDINGS COMPÉTENCES
    # ========================================
    
    print("\n📥 Chargement embeddings compétences...")
    
    try:
        # Charger UMAP 3D compétences
        umap_3d_comp = np.load(MODELS_DIR / 'competences_umap_3d.npy')
        
        # Charger CSV compétences
        df_comp = pd.read_csv(RESULTS_DIR / 'competences_analysis.csv')
        
        coords_comp = umap_3d_comp
        labels_comp = df_comp['cluster'].astype(str).tolist()
        texts_comp = df_comp['competence'].tolist()
        hover_comp = df_comp.apply(
            lambda row: f"{row['competence']}<br>Cluster: {row['cluster']}<br>Occurrences: {row['count']}×",
            axis=1
        ).tolist()
        
        print(f"   ✅ {len(coords_comp)} compétences chargées")
        competences_available = True
        
    except FileNotFoundError:
        print("   ⚠️  Compétences non disponibles (exécuter 6_embeddings_competences.py d'abord)")
        competences_available = False
    
    # ========================================
    # 3. GÉNÉRER VISUALISATIONS 3D
    # ========================================
    
    print("\n🎨 Génération visualisations 3D...")
    
    # ========================================
    # VIZ 1: Offres par profil (UMAP 3D)
    # ========================================
    
    print("\n   📊 Vue 1: Offres par profil (UMAP 3D)...")
    create_3d_scatter_advanced(
        coords_3d=coords_offres,
        labels=labels_offres,
        hover_texts=hover_offres,
        title='Projection 3D des Offres d\'Emploi (UMAP) - Par Profil Métier',
        size_values=sizes_offres,
        output_file='projector_offres_profils_3d.html'
    )
    
    # ========================================
    # VIZ 2: Offres avec rotation animée
    # ========================================
    
    print("\n   📊 Vue 2: Offres avec rotation animée...")
    create_3d_animated_rotation(
        coords_3d=coords_offres,
        labels=labels_offres,
        hover_texts=hover_offres,
        title='Projection 3D Animée - Offres d\'Emploi',
        output_file='projector_offres_animated.html'
    )
    
    # ========================================
    # VIZ 3: Compétences avec labels (si disponible)
    # ========================================
    
    if competences_available:
        print("\n   📊 Vue 3: Compétences avec labels...")
        
        # Filtrer top compétences pour lisibilité
        df_comp_top = df_comp.nlargest(100, 'count')
        indices_top = df_comp_top.index.tolist()
        
        create_3d_with_labels(
            coords_3d=coords_comp[indices_top],
            labels=df_comp_top['cluster'].astype(str).tolist(),
            texts=df_comp_top['competence'].tolist(),
            title='Carte 3D des Top 100 Compétences (avec Labels)',
            show_text=True,
            text_size=7,
            output_file='projector_competences_labels_3d.html'
        )
        
        # ========================================
        # VIZ 4: Toutes compétences (sans labels)
        # ========================================
        
        print("\n   📊 Vue 4: Toutes compétences...")
        create_3d_scatter_advanced(
            coords_3d=coords_comp,
            labels=labels_comp,
            hover_texts=hover_comp,
            title='Projection 3D Complète des Compétences - Par Cluster',
            output_file='projector_competences_all_3d.html'
        )
    
    # ========================================
    # VIZ 5: Offres par région (top 5 régions)
    # ========================================
    
    print("\n   📊 Vue 5: Offres par région...")
    
    top_regions = df_class['region'].value_counts().head(5).index.tolist()
    df_regions = df_class[df_class['region'].isin(top_regions)].copy()
    
    indices_regions = df_regions.index.tolist()
    indices_in_class = [indices_class.index(i) for i in indices_regions]
    
    hover_regions = df_regions.apply(
        lambda row: f"{row['title']}<br>Région: {row['region']}<br>Profil: {row['profil_assigned']}",
        axis=1
    ).tolist()
    
    create_3d_scatter_advanced(
        coords_3d=coords_offres[indices_in_class],
        labels=df_regions['region'].tolist(),
        hover_texts=hover_regions,
        title='Projection 3D des Offres - Par Région (Top 5)',
        output_file='projector_offres_regions_3d.html'
    )
    
    # ========================================
    # VIZ 6: Offres par source
    # ========================================
    
    print("\n   📊 Vue 6: Offres par source...")
    
    hover_sources = df_class.apply(
        lambda row: f"{row['title']}<br>Source: {row['source_name']}<br>Profil: {row['profil_assigned']}",
        axis=1
    ).tolist()
    
    create_3d_scatter_advanced(
        coords_3d=coords_offres,
        labels=df_class['source_name'].tolist(),
        hover_texts=hover_sources,
        title='Projection 3D des Offres - Par Source (France Travail vs Indeed)',
        output_file='projector_offres_sources_3d.html'
    )
    
    # ========================================
    # GÉNÉRER INDEX HTML
    # ========================================
    
    print("\n📄 Génération index.html...")
    
    html_index = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualisations 3D Interactives - Projet NLP</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .container {{
            background: white;
            color: #333;
            border-radius: 10px;
            padding: 40px;
            box-shadow: 0 10px 50px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #764ba2;
            margin-top: 30px;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .viz-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 25px;
            border: 2px solid #e9ecef;
            transition: all 0.3s;
        }}
        .viz-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-color: #667eea;
        }}
        .viz-card h3 {{
            color: #667eea;
            margin-top: 0;
        }}
        .viz-card p {{
            color: #666;
            line-height: 1.6;
        }}
        .btn {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s;
        }}
        .btn:hover {{
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .badge {{
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 4px 10px;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Visualisations 3D Interactives</h1>
        <p style="color: #666; font-size: 18px;">
            Explorez les données du projet NLP en 3D - Style TensorFlow Projector
        </p>
        
        <h2>📊 Offres d'Emploi ({len(coords_offres)} offres)</h2>
        <div class="viz-grid">
            <div class="viz-card">
                <h3>Vue par Profil Métier <span class="badge">Principal</span></h3>
                <p>Projection UMAP 3D des offres colorées par profil classifié. Taille = score de classification.</p>
                <a href="projector_offres_profils_3d.html" class="btn">Ouvrir ➜</a>
            </div>
            
            <div class="viz-card">
                <h3>Vue Animée</h3>
                <p>Rotation automatique 360° pour démonstration. Contrôles play/pause.</p>
                <a href="projector_offres_animated.html" class="btn">Ouvrir ➜</a>
            </div>
            
            <div class="viz-card">
                <h3>Vue par Région</h3>
                <p>Top 5 régions françaises. Identifie clusters géographiques.</p>
                <a href="projector_offres_regions_3d.html" class="btn">Ouvrir ➜</a>
            </div>
            
            <div class="viz-card">
                <h3>Vue par Source</h3>
                <p>France Travail vs Indeed. Compare qualité/similarité des sources.</p>
                <a href="projector_offres_sources_3d.html" class="btn">Ouvrir ➜</a>
            </div>
        </div>
"""
    
    if competences_available:
        html_index += f"""
        <h2>🔧 Compétences Techniques ({len(coords_comp)} compétences)</h2>
        <div class="viz-grid">
            <div class="viz-card">
                <h3>Top 100 avec Labels <span class="badge">Recommandé</span></h3>
                <p>Carte sémantique des 100 compétences les plus fréquentes. Labels visibles.</p>
                <a href="projector_competences_labels_3d.html" class="btn">Ouvrir ➜</a>
            </div>
            
            <div class="viz-card">
                <h3>Vue Complète</h3>
                <p>Toutes les compétences par cluster technologique. Hover pour détails.</p>
                <a href="projector_competences_all_3d.html" class="btn">Ouvrir ➜</a>
            </div>
        </div>
"""
    
    html_index += """
        <h2>💡 Instructions</h2>
        <ul style="color: #666; line-height: 1.8;">
            <li><strong>Rotation :</strong> Clic gauche + glisser</li>
            <li><strong>Zoom :</strong> Molette souris ou pincement tactile</li>
            <li><strong>Pan :</strong> Clic droit + glisser (ou Shift + clic gauche)</li>
            <li><strong>Hover :</strong> Survoler points pour infos détaillées</li>
            <li><strong>Légende :</strong> Cliquer sur catégories pour afficher/masquer</li>
            <li><strong>Reset :</strong> Double-clic pour réinitialiser vue</li>
        </ul>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #e9ecef; color: #999; text-align: center;">
            Projet NLP Text Mining - Master SISE - Décembre 2025
        </div>
    </div>
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
    print("✅ VISUALISATIONS 3D TERMINÉES !")
    print("="*70)
    
    print(f"\n📁 Répertoire: {VIZ_DIR}")
    print(f"\n📄 Fichiers créés:")
    print(f"   - index.html (page d'accueil)")
    print(f"   - projector_offres_profils_3d.html")
    print(f"   - projector_offres_animated.html")
    print(f"   - projector_offres_regions_3d.html")
    print(f"   - projector_offres_sources_3d.html")
    
    if competences_available:
        print(f"   - projector_competences_labels_3d.html")
        print(f"   - projector_competences_all_3d.html")
    
    print(f"\n🌐 Pour visualiser:")
    print(f"   Ouvrir: {VIZ_DIR / 'index.html'}")


if __name__ == "__main__":
    main()