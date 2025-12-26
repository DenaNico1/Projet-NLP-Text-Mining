"""
4. Analyse Géo-Sémantique
Compare le vocabulaire par région

Auteur: Projet NLP Text Mining  
Date: Décembre 2025
"""

import pandas as pd
import pickle
from pathlib import Path
import sys
from collections import Counter
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))
from utils import ResultSaver

def main():
    print("="*70)
    print("🗺️  ÉTAPE 4 : ANALYSE GÉO-SÉMANTIQUE")
    print("="*70)
    
    saver = ResultSaver()
    
    with open('../resultats_nlp/models/data_with_topics.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Top 5 régions
    top_regions = df['region'].value_counts().head(5).index
    
    results = {}
    
    for region in top_regions:
        df_region = df[df['region'] == region]
        
        # Termes spécifiques
        all_tokens = [t for tokens in df_region['tokens'] for t in tokens]
        counter = Counter(all_tokens)
        
        # Compétences
        all_comps = [c for cs in df_region['competences_found'] for c in cs]
        comp_counter = Counter(all_comps)
        
        # Stats
        results[region] = {
            'nb_offres': len(df_region),
            'salaire_median': float(df_region['salary_annual'].median()),
            'top_termes': counter.most_common(20),
            'top_competences': comp_counter.most_common(15),
            'contract_types': df_region['contract_type'].value_counts().to_dict()
        }
        
        print(f"\n{region}:")
        print(f"  Offres: {len(df_region)}")
        print(f"  Salaire: {results[region]['salaire_median']:.0f}€")
        print(f"  Top compétences: {', '.join([c for c, _ in comp_counter.most_common(5)])}")
    
    # Carte
    region_stats = df.groupby('region').agg({
        'offre_id': 'count',
        'salary_annual': 'median',
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()
    region_stats.columns = ['region', 'nb_offres', 'salaire', 'lat', 'lon']
    
    fig = px.scatter_geo(
        region_stats,
        lat='lat',
        lon='lon',
        size='nb_offres',
        color='salaire',
        hover_name='region',
        title="Offres Data/IA par Région",
        projection='natural earth'
    )
    saver.save_visualization(fig, 'carte_regions.html')
    
    saver.save_json(results, 'analyse_geo_semantique.json')
    
    print("\n✅ ANALYSE GÉO-SÉMANTIQUE TERMINÉE !")

if __name__ == "__main__":
    main()