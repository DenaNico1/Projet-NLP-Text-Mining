"""
5. Visualisations Profils - 12 Visualisations Interactives
Génère visualisations complètes pour l'analyse des profils métier

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import sys
from collections import Counter

# Visualisations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from utils import ResultSaver
from profils_definitions import get_all_profils


def viz_distribution_profils(df, saver):
    """1. Distribution des profils (bar chart)"""
    print("    Distribution profils...")
    
    df_class = df[df['status'] == 'classified']
    profil_counts = df_class['profil_assigned'].value_counts()
    
    fig = px.bar(
        x=profil_counts.index,
        y=profil_counts.values,
        title='Distribution des Profils Métier',
        labels={'x': 'Profil', 'y': 'Nombre d\'offres'},
        color=profil_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        showlegend=False
    )
    
    saver.save_visualization(fig, 'profils_distribution.html')


def viz_profils_by_region(df, saver):
    """2. Profils par région (grouped bar chart)"""
    print("    Profils par région...")
    
    df_class = df[df['status'] == 'classified']
    top_regions = df_class['region'].value_counts().head(8).index
    
    # Préparer données
    data = []
    for region in top_regions:
        df_region = df_class[df_class['region'] == region]
        profil_counts = df_region['profil_assigned'].value_counts().head(5)
        
        for profil, count in profil_counts.items():
            data.append({
                'Région': region,
                'Profil': profil,
                'Nombre': count
            })
    
    df_viz = pd.DataFrame(data)
    
    fig = px.bar(
        df_viz,
        x='Région',
        y='Nombre',
        color='Profil',
        title='Top 5 Profils par Région',
        barmode='group',
        height=600
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    
    saver.save_visualization(fig, 'profils_by_region.html')


def viz_profils_salaires(df, saver):
    """3. Salaires par profil (box plot)"""
    print("  Salaires par profil...")
    
    df_class = df[df['status'] == 'classified'].copy()
    df_class = df_class[df_class['salary_annual'].notna()]
    
    # Filtrer profils avec >20 offres ayant salaire
    profil_counts = df_class['profil_assigned'].value_counts()
    profils_keep = profil_counts[profil_counts >= 20].index
    df_viz = df_class[df_class['profil_assigned'].isin(profils_keep)]
    
    if len(df_viz) > 0:
        fig = px.box(
            df_viz,
            x='profil_assigned',
            y='salary_annual',
            title='Distribution Salaires par Profil',
            labels={'profil_assigned': 'Profil', 'salary_annual': 'Salaire annuel (€)'},
            color='profil_assigned'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            showlegend=False
        )
        
        saver.save_visualization(fig, 'profils_salaires.html')


def viz_heatmap_region(df, saver):
    """4. Heatmap profils × régions"""
    print("    Heatmap profils × régions...")
    
    df_class = df[df['status'] == 'classified']
    
    # Top 8 régions et top 8 profils
    top_regions = df_class['region'].value_counts().head(8).index
    top_profils = df_class['profil_assigned'].value_counts().head(8).index
    
    # Matrice
    matrix = pd.crosstab(
        df_class[df_class['region'].isin(top_regions)]['profil_assigned'],
        df_class[df_class['region'].isin(top_regions)]['region']
    )
    
    # Garder seulement top profils
    matrix = matrix.loc[matrix.index.isin(top_profils)]
    
    # Normaliser par colonne (région)
    matrix_norm = matrix.div(matrix.sum(axis=0), axis=1) * 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(
        matrix_norm,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Pourcentage (%)'},
        ax=ax
    )
    
    ax.set_title('Distribution Profils par Région (%)', fontsize=16, pad=20)
    ax.set_xlabel('Région', fontsize=12)
    ax.set_ylabel('Profil', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    saver.save_visualization(fig, 'profils_heatmap_region.png')
    plt.close()


def viz_sankey_competences(df, saver):
    """5. Sankey diagram compétences → profils"""
    print("   Sankey compétences → profils...")
    
    df_class = df[df['status'] == 'classified']
    
    # Top 10 compétences et top 8 profils
    all_comps = [comp for comps in df_class['competences_found'] for comp in comps]
    top_comps = [c for c, _ in Counter(all_comps).most_common(10)]
    top_profils = df_class['profil_assigned'].value_counts().head(8).index
    
    # Compter flux compétence → profil
    flows = []
    for comp in top_comps:
        for profil in top_profils:
            df_match = df_class[
                (df_class['profil_assigned'] == profil) &
                (df_class['competences_found'].apply(lambda x: comp in x))
            ]
            count = len(df_match)
            if count > 10:  # Seuil minimum
                flows.append({
                    'source': comp,
                    'target': profil,
                    'value': count
                })
    
    if flows:
        df_flows = pd.DataFrame(flows)
        
        # Créer labels et indices
        all_nodes = list(set(df_flows['source'].tolist() + df_flows['target'].tolist()))
        node_dict = {node: idx for idx, node in enumerate(all_nodes)}
        
        source_indices = [node_dict[s] for s in df_flows['source']]
        target_indices = [node_dict[t] for t in df_flows['target']]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                label=all_nodes
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=df_flows['value']
            )
        )])
        
        fig.update_layout(
            title='Flux Compétences → Profils',
            height=700
        )
        
        saver.save_visualization(fig, 'profils_sankey.html')


def viz_radar_competences(df, saver):
    """6. Radar chart compétences par profil (top 5 profils)"""
    print("    Radar compétences...")
    
    df_class = df[df['status'] == 'classified']
    top_profils = df_class['profil_assigned'].value_counts().head(5).index
    
    # Top 8 compétences globales
    all_comps = [comp for comps in df_class['competences_found'] for comp in comps]
    top_comps = [c for c, _ in Counter(all_comps).most_common(8)]
    
    # Calculer % par profil
    data_radar = []
    
    for profil in top_profils:
        df_profil = df_class[df_class['profil_assigned'] == profil]
        percentages = []
        
        for comp in top_comps:
            count = sum(1 for comps in df_profil['competences_found'] if comp in comps)
            pct = count / len(df_profil) * 100 if len(df_profil) > 0 else 0
            percentages.append(pct)
        
        data_radar.append({
            'profil': profil,
            'competences': top_comps,
            'percentages': percentages
        })
    
    # Créer figure
    fig = go.Figure()
    
    for item in data_radar:
        fig.add_trace(go.Scatterpolar(
            r=item['percentages'],
            theta=item['competences'],
            fill='toself',
            name=item['profil']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title='Profil de Compétences par Métier (Top 5)',
        height=600,
        showlegend=True
    )
    
    saver.save_visualization(fig, 'profils_radar.html')


def viz_treemap_profils(df, saver):
    """7. Treemap profils × sources"""
    print("   Treemap profils × sources...")
    
    df_class = df[df['status'] == 'classified']
    
    # Préparer données
    data = []
    for source in df_class['source_name'].unique():
        for profil in df_class['profil_assigned'].unique():
            count = len(df_class[
                (df_class['source_name'] == source) &
                (df_class['profil_assigned'] == profil)
            ])
            if count > 0:
                data.append({
                    'Source': source,
                    'Profil': profil,
                    'Count': count
                })
    
    df_tree = pd.DataFrame(data)
    
    fig = px.treemap(
        df_tree,
        path=['Source', 'Profil'],
        values='Count',
        title='Répartition Profils par Source',
        height=600
    )
    
    saver.save_visualization(fig, 'profils_treemap.html')


def viz_confidence_distribution(df, saver):
    """8. Distribution confiance classification"""
    print("   Distribution confiance...")
    
    df_class = df[df['status'] == 'classified']
    
    fig = px.histogram(
        df_class,
        x='profil_confidence',
        nbins=50,
        title='Distribution de la Confiance de Classification',
        labels={'profil_confidence': 'Confiance', 'count': 'Nombre d\'offres'},
        color_discrete_sequence=['#636EFA']
    )
    
    fig.add_vline(
        x=df_class['profil_confidence'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Moyenne: {df_class['profil_confidence'].mean():.2f}"
    )
    
    fig.update_layout(height=500)
    
    saver.save_visualization(fig, 'profils_confidence.html')


def viz_sunburst_profils(df, saver):
    """9. Sunburst profils × contrats × régions"""
    print("    Sunburst profils...")
    
    df_class = df[df['status'] == 'classified']
    
    # Top 6 profils, 3 contrats, 5 régions
    top_profils = df_class['profil_assigned'].value_counts().head(6).index
    top_regions = df_class['region'].value_counts().head(5).index
    
    df_viz = df_class[
        (df_class['profil_assigned'].isin(top_profils)) &
        (df_class['region'].isin(top_regions))
    ]
    
    # Préparer données
    data = []
    for profil in top_profils:
        for contrat in df_viz['contract_type'].unique():
            for region in top_regions:
                count = len(df_viz[
                    (df_viz['profil_assigned'] == profil) &
                    (df_viz['contract_type'] == contrat) &
                    (df_viz['region'] == region)
                ])
                if count > 0:
                    data.append({
                        'Profil': profil,
                        'Contrat': contrat,
                        'Région': region,
                        'Count': count
                    })
    
    df_sun = pd.DataFrame(data)
    
    if len(df_sun) > 0:
        fig = px.sunburst(
            df_sun,
            path=['Profil', 'Contrat', 'Région'],
            values='Count',
            title='Profils × Contrats × Régions',
            height=700
        )
        
        saver.save_visualization(fig, 'profils_sunburst.html')


def viz_evolution_temporelle(df, saver):
    """10. Évolution temporelle profils (si dates disponibles)"""
    print("    Évolution temporelle...")
    
    df_class = df[df['status'] == 'classified'].copy()
    df_class = df_class[df_class['date_posted'].notna()]
    
    if len(df_class) > 100:  # Assez de données
        # Top 5 profils
        top_profils = df_class['profil_assigned'].value_counts().head(5).index
        
        df_class['date_posted'] = pd.to_datetime(df_class['date_posted'])
        df_class['year_month'] = df_class['date_posted'].dt.to_period('M').astype(str)
        
        # Compter par mois
        data = []
        for profil in top_profils:
            df_profil = df_class[df_class['profil_assigned'] == profil]
            counts = df_profil.groupby('year_month').size()
            
            for month, count in counts.items():
                data.append({
                    'Mois': month,
                    'Profil': profil,
                    'Nombre': count
                })
        
        df_viz = pd.DataFrame(data)
        
        if len(df_viz) > 0:
            fig = px.line(
                df_viz,
                x='Mois',
                y='Nombre',
                color='Profil',
                title='Évolution Temporelle des Profils (Top 5)',
                markers=True,
                height=600
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            
            saver.save_visualization(fig, 'profils_evolution.html')


def viz_scores_components(df, saver):
    """11. Composantes scores (règles vs ML vs compétences)"""
    print("   Composantes scores...")
    
    df_class = df[df['status'] == 'classified']
    top_profils = df_class['profil_assigned'].value_counts().head(6).index
    df_viz = df_class[df_class['profil_assigned'].isin(top_profils)]
    
    # Moyennes par profil
    data = []
    for profil in top_profils:
        df_profil = df_viz[df_viz['profil_assigned'] == profil]
        data.append({
            'Profil': profil,
            'Titre': df_profil['score_title'].mean(),
            'Description': df_profil['score_description'].mean(),
            'Compétences': df_profil['score_competences'].mean()
        })
    
    df_scores = pd.DataFrame(data)
    
    # Transformer en format long
    df_long = df_scores.melt(
        id_vars=['Profil'],
        value_vars=['Titre', 'Description', 'Compétences'],
        var_name='Composante',
        value_name='Score'
    )
    
    fig = px.bar(
        df_long,
        x='Profil',
        y='Score',
        color='Composante',
        barmode='group',
        title='Contribution des Composantes de Scoring par Profil (Titre 60% + Description 20% + Compétences 20%)',
        height=600
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    
    saver.save_visualization(fig, 'profils_scores_components.html')


def viz_pie_global(df, saver):
    """12. Pie chart distribution globale"""
    print("   Pie chart distribution...")
    
    df_class = df[df['status'] == 'classified']
    profil_counts = df_class['profil_assigned'].value_counts()
    
    fig = px.pie(
        values=profil_counts.values,
        names=profil_counts.index,
        title='Distribution Globale des Profils Métier',
        height=600
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    saver.save_visualization(fig, 'profils_pie.html')


def main():
    """
    Génère toutes les visualisations
    """
    print("="*70)
    print(" ÉTAPE 5 : VISUALISATIONS PROFILS MÉTIER")
    print("="*70)
    
    saver = ResultSaver()
    
    # ==========================================
    # CHARGEMENT
    # ==========================================
    print("\n📥 Chargement data_with_profiles.pkl...")
    
    with open('../resultats_nlp/models/data_with_profiles.pkl', 'rb') as f:
        df = pickle.load(f)
    
    print(f"    Offres: {len(df)}")
    print(f"    Classifiées: {(df['status'] == 'classified').sum()}")
    
    # ==========================================
    # VISUALISATIONS
    # ==========================================
    print("\n Génération visualisations...")
    
    viz_distribution_profils(df, saver)
    viz_profils_by_region(df, saver)
    viz_profils_salaires(df, saver)
    viz_heatmap_region(df, saver)
    viz_sankey_competences(df, saver)
    viz_radar_competences(df, saver)
    viz_treemap_profils(df, saver)
    viz_confidence_distribution(df, saver)
    viz_sunburst_profils(df, saver)
    viz_evolution_temporelle(df, saver)
    viz_scores_components(df, saver)
    viz_pie_global(df, saver)
    
    print("\n VISUALISATIONS TERMINÉES !")
    
    print(f"\n Visualisations créées:")
    print(f"   1. profils_distribution.html")
    print(f"   2. profils_by_region.html")
    print(f"   3. profils_salaires.html")
    print(f"   4. profils_heatmap_region.png")
    print(f"   5. profils_sankey.html")
    print(f"   6. profils_radar.html")
    print(f"   7. profils_treemap.html")
    print(f"   8. profils_confidence.html")
    print(f"   9. profils_sunburst.html")
    print(f"   10. profils_evolution.html")
    print(f"   11. profils_scores_components.html")
    print(f"   12. profils_pie.html")


if __name__ == "__main__":
    main()