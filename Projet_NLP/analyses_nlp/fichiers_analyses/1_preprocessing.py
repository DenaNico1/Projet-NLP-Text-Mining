"""
1. Preprocessing des Données
Charge et prépare les données pour toutes les analyses NLP

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le chemin pour imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import DataLoader, TextPreprocessor, ResultSaver, compute_salary_annual

def main():
    """
    Pipeline de preprocessing
    """
    print("="*70)
    print("📋 ÉTAPE 1 : PREPROCESSING DES DONNÉES")
    print("="*70)
    
    # ==========================================
    # 1. CHARGEMENT DES DONNÉES
    # ==========================================
    print("\n🔄 Chargement des données...")
    
    loader = DataLoader()
    df_offres = loader.load_all_offers()
    df_competences = loader.load_competences()
    loader.disconnect()
    
    print(f"\n📊 Statistiques initiales:")
    print(f"   Total offres: {len(df_offres)}")
    print(f"   Avec description: {df_offres['description'].notna().sum()}")
    print(f"   Compétences structurées: {len(df_competences)}")
    
    # ==========================================
    # 2. NETTOYAGE ET ENRICHISSEMENT
    # ==========================================
    print("\n🧹 Nettoyage des données...")
    
    # Calcul salaire annuel moyen
    df_offres['salary_annual'] = df_offres.apply(compute_salary_annual, axis=1)
    
    # Filtrer les offres avec description
    df_clean = df_offres[df_offres['description'].notna()].copy()
    
    print(f"   Offres après filtrage: {len(df_clean)}")
    
    # ==========================================
    # 3. PREPROCESSING NLP
    # ==========================================
    print("\n🔤 Preprocessing NLP...")
    
    preprocessor = TextPreprocessor(language='french')
    
    # Préprocesser les descriptions
    print("   Nettoyage des textes...")
    df_clean['description_clean'] = df_clean['description'].apply(
        preprocessor.clean_text
    )
    
    # Tokenization
    print("   Tokenization...")
    df_clean['tokens'] = df_clean['description_clean'].apply(
        lambda x: preprocessor.tokenize(x)
    )
    
    # Nombre de tokens
    df_clean['num_tokens'] = df_clean['tokens'].apply(len)
    
    print(f"\n📊 Statistiques texte:")
    print(f"   Tokens moyen par offre: {df_clean['num_tokens'].mean():.0f}")
    print(f"   Tokens médian: {df_clean['num_tokens'].median():.0f}")
    print(f"   Tokens min/max: {df_clean['num_tokens'].min()}/{df_clean['num_tokens'].max()}")
    
    # ==========================================
    # 4. ENRICHISSEMENT - COMPÉTENCES
    # ==========================================
    print("\n🎓 Enrichissement compétences...")
    
    # Créer dictionnaire compétences depuis France Travail
    unique_skills = df_competences['skill_label'].unique()
    print(f"   Compétences uniques (FT): {len(unique_skills)}")
    
    # Ajouter compétences courantes du domaine Data/IA
    additional_skills = [
        # Langages
        'Python', 'R', 'SQL', 'Java', 'Scala', 'Julia',
        # ML/DL
        'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 
        'Scikit-learn', 'Keras', 'XGBoost', 'LightGBM',
        # Data Engineering
        'Spark', 'Hadoop', 'Kafka', 'Airflow', 'DBT',
        # Cloud
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes',
        # BI
        'Power BI', 'Tableau', 'Looker', 'Qlik',
        # Databases
        'PostgreSQL', 'MySQL', 'MongoDB', 'Cassandra', 'Redis',
        # Libs Python
        'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Plotly',
        # MLOps
        'MLflow', 'Kubeflow', 'MLOps', 'CI/CD',
        # NLP
        'NLP', 'NLTK', 'spaCy', 'Transformers', 'BERT', 'GPT',
        'LangChain', 'LlamaIndex',
        # Other
        'Git', 'Linux', 'API', 'REST', 'GraphQL', 'Agile', 'Scrum'
    ]
    
    # Combiner
    all_skills = list(set(list(unique_skills) + additional_skills))
    print(f"   Dictionnaire complet: {len(all_skills)} compétences")
    
    # Sauvegarder dictionnaire
    saver = ResultSaver()
    saver.save_json(
        {'competences': sorted(all_skills)}, 
        'dictionnaire_competences.json'
    )
    
    # ==========================================
    # 5. STATISTIQUES PAR SOURCE
    # ==========================================
    print("\n📊 Statistiques par source:")
    
    for source in df_clean['source_name'].unique():
        df_source = df_clean[df_clean['source_name'] == source]
        print(f"\n   {source}:")
        print(f"      Offres: {len(df_source)}")
        print(f"      Tokens moyen: {df_source['num_tokens'].mean():.0f}")
        print(f"      Avec salaire: {df_source['salary_annual'].notna().sum()}")
        print(f"      Régions uniques: {df_source['region'].nunique()}")
    
    # ==========================================
    # 6. STATISTIQUES PAR RÉGION
    # ==========================================
    print("\n🗺️  Top 10 régions:")
    
    region_stats = df_clean.groupby('region').agg({
        'offre_id': 'count',
        'salary_annual': 'median',
        'num_tokens': 'mean'
    }).round(0)
    region_stats.columns = ['nb_offres', 'salaire_median', 'tokens_moyen']
    region_stats = region_stats.sort_values('nb_offres', ascending=False).head(10)
    
    print(region_stats)
    
    # ==========================================
    # 7. SAUVEGARDE DONNÉES PREPROCESSÉES
    # ==========================================
    print("\n💾 Sauvegarde des données preprocessées...")
    
    # Sauvegarder DataFrame complet
    saver.save_pickle(df_clean, 'data_preprocessed.pkl')
    
    # Sauvegarder CSV allégé (sans tokens)
    df_export = df_clean.drop(columns=['tokens'], errors='ignore')
    saver.save_csv(df_export, 'data_preprocessed.csv')
    
    # Sauvegarder compétences
    saver.save_pickle(df_competences, 'competences_ft.pkl')
    
    # Statistiques globales
    stats = {
        'total_offres': len(df_clean),
        'sources': df_clean['source_name'].value_counts().to_dict(),
        'regions': df_clean['region'].value_counts().head(10).to_dict(),
        'contract_types': df_clean['contract_type'].value_counts().to_dict(),
        'salary_stats': {
            'count': int(df_clean['salary_annual'].notna().sum()),
            'mean': float(df_clean['salary_annual'].mean()),
            'median': float(df_clean['salary_annual'].median()),
            'min': float(df_clean['salary_annual'].min()),
            'max': float(df_clean['salary_annual'].max())
        },
        'text_stats': {
            'tokens_mean': float(df_clean['num_tokens'].mean()),
            'tokens_median': float(df_clean['num_tokens'].median()),
            'tokens_min': int(df_clean['num_tokens'].min()),
            'tokens_max': int(df_clean['num_tokens'].max())
        },
        'competences_uniques': len(all_skills),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    saver.save_json(stats, 'stats_globales.json')
    
    print("\n✅ PREPROCESSING TERMINÉ !")
    print(f"\n📁 Fichiers créés:")
    print(f"   - data_preprocessed.pkl (DataFrame complet)")
    print(f"   - data_preprocessed.csv (export)")
    print(f"   - competences_ft.pkl (compétences FT)")
    print(f"   - dictionnaire_competences.json ({len(all_skills)} compétences)")
    print(f"   - stats_globales.json (statistiques)")
    
    return df_clean, df_competences, all_skills


if __name__ == "__main__":
    df_clean, df_competences, skills_dict = main()