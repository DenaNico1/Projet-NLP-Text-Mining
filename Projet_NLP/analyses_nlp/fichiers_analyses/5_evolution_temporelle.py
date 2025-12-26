"""
5. Évolution Temporelle des Compétences
Analyse les tendances temporelles

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
    print("📈 ÉTAPE 5 : ÉVOLUTION TEMPORELLE")
    print("="*70)
    
    saver = ResultSaver()
    
    with open('../resultats_nlp/models/data_with_topics.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Filtrer offres avec date
    df_dated = df[df['scraped_at'].notna()].copy()
    df_dated['date'] = pd.to_datetime(df_dated['scraped_at'])
    df_dated['semaine'] = df_dated['date'].dt.to_period('W')
    
    print(f"   Offres avec date: {len(df_dated)}")
    
    # Évolution par semaine
    weekly = df_dated.groupby('semaine').size()
    
    print(f"\n📊 Offres par semaine:")
    for week, count in weekly.tail(10).items():
        print(f"   {week}: {count}")
    
    # Top compétences émergentes
    comp_lists = df_dated.groupby('semaine')['competences_found'].apply(
        lambda x: [c for cs in x for c in cs]
    )
    
    results = {
        'evolution_hebdo': {str(k): int(v) for k, v in weekly.items()},
        'total_avec_date': len(df_dated)
    }
    
    saver.save_json(results, 'evolution_temporelle.json')
    
    print("\n✅ ÉVOLUTION TEMPORELLE TERMINÉE !")

if __name__ == "__main__":
    main()