"""
PREPROCESSING MASTER - Unique Point de Traitement
Fait TOUT le preprocessing et génère data_clean.pkl avec toutes les colonnes nécessaires
Fonctionnalités principales:
- Stopwords FR+EN complets
- Lemmatisation activée
- Nettoyage HTML complet
- Pattern matching compétences
- Génère TOUTES les colonnes pour analyses

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Ajouter le chemin pour imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import DataLoader, TextPreprocessor, ResultSaver, compute_salary_annual, extract_competences_from_text

def main():
    """
    Pipeline de preprocessing MASTER
    """
    print("="*70)
    print(" PREPROCESSING MASTER - POINT UNIQUE DE TRAITEMENT")
    print("="*70)
    
    # ==========================================
    # 1. CHARGEMENT DES DONNÉES
    # ==========================================
    print("\n Chargement des données...")
    
    loader = DataLoader()
    df_offres = loader.load_all_offers()
    df_competences = loader.load_competences()
    loader.disconnect()
    
    print(f"\n Statistiques initiales:")
    print(f"   Total offres: {len(df_offres)}")
    print(f"   Avec description: {df_offres['description'].notna().sum()}")
    print(f"   Compétences structurées: {len(df_competences)}")
    
    # ==========================================
    # 2. NETTOYAGE ET ENRICHISSEMENT
    # ==========================================
    print("\n Nettoyage des données...")
    
    # Calcul salaire annuel moyen
    df_offres['salary_annual'] = df_offres.apply(compute_salary_annual, axis=1)
    
    # Filtrer les offres avec description
    df_clean = df_offres[df_offres['description'].notna()].copy()
    
    print(f"   Offres après filtrage: {len(df_clean)}")
    
    # ==========================================
    # 3. PREPROCESSING NLP COMPLET
    # ==========================================
    print("\n Preprocessing NLP complet...")
    
    preprocessor = TextPreprocessor(language='french')
    
    # 3.1 Nettoyage HTML
    print("    Nettoyage HTML (entités &nbsp;, balises)...")
    df_clean['description_clean'] = df_clean['description'].apply(
        preprocessor.clean_text
    )
    
    # 3.2 Tokenisation + Stopwords + Lemmatisation
    print("    Tokenisation + Stopwords FR+EN + Lemmatisation...")
    df_clean['tokens'] = df_clean['description_clean'].apply(
        lambda x: preprocessor.preprocess(x, lemmatize=True)
    )
    
    # 3.3 Texte pour sklearn (rejoint tokens pour TF-IDF/LDA)
    print("    Génération text_for_sklearn (tokens rejoints)...")
    df_clean['text_for_sklearn'] = df_clean['tokens'].apply(lambda x: ' '.join(x))
    
    # 3.4 Nombre de tokens
    df_clean['num_tokens'] = df_clean['tokens'].apply(len)
    
    print(f"\n Statistiques texte:")
    print(f"   Tokens moyen par offre: {df_clean['num_tokens'].mean():.0f}")
    print(f"   Tokens médian: {df_clean['num_tokens'].median():.0f}")
    print(f"   Tokens min/max: {df_clean['num_tokens'].min()}/{df_clean['num_tokens'].max()}")
    
    # ==========================================
    # 4. VÉRIFICATIONS QUALITÉ
    # ==========================================
    print("\n Vérifications qualité preprocessing:")
    
    sample_tokens = df_clean['tokens'].iloc[0]
    print(f"   Exemple tokens (1ère offre): {sample_tokens[:15]}")
    
    # Vérifier absence stopwords
    from nltk.corpus import stopwords
    stop_fr = set(stopwords.words('french'))
    stop_en = set(stopwords.words('english'))
    
    stopwords_found = [t for t in sample_tokens if t in stop_fr or t in stop_en]
    print(f"    Stopwords FR+EN filtrés: {len(stopwords_found) == 0}")
    if stopwords_found:
        print(f"   Trouvés: {stopwords_found[:10]}")
    
    # Vérifier absence 'nbsp'
    nbsp_count = sum(1 for tokens in df_clean['tokens'] if 'nbsp' in tokens)
    print(f"   'nbsp' supprimé: {nbsp_count == 0} ({nbsp_count} offres contiennent 'nbsp')")
    
    # Vérifier longueur mots >= 3
    short_tokens = [t for t in sample_tokens if len(t) < 3]
    print(f"   Tokens >= 3 caractères: {len(short_tokens) == 0}")
    if short_tokens:
        print(f"        Tokens courts trouvés: {short_tokens}")
    
    # ==========================================
    # 5. DICTIONNAIRE COMPÉTENCES
    # ==========================================
    print("\n Création dictionnaire compétences...")
    
    # Compétences France Travail
    unique_skills = df_competences['skill_label'].unique()
    print(f"   Compétences uniques (FT): {len(unique_skills)}")
    
    # Compétences additionnelles Data/IA
    additional_skills = [
        # Langages
        'Python', 'R', 'SQL', 'Java', 'Scala', 'Julia',
        # ML/DL
        'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 
        'Scikit-learn', 'Keras', 'XGBoost', 'LightGBM',
        # Data Engineering
        'Spark', 'Hadoop', 'Kafka', 'Airflow', 'DBT', 'APIs',
        # Cloud
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes',
        # BI
        'Power BI', 'Tableau', 'Looker', 'Qlik',
        # Databases
        'PostgreSQL', 'MySQL', 'MongoDB', 'Cassandra', 'Redis',
        # Libs Python
        'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Plotly',
        # MLOps
        'MLflow', 'Kubeflow', 'MLOps', 'CI/CD', 'Streamlit', 'DevOps', 
        'Databricks', 'Snowflake',
        # NLP
        'NLP', 'NLTK', 'spaCy', 'Transformers', 'BERT', 'GPT',
        'LangChain', 'LlamaIndex',
        # IA /LLM
        'Intelligence Artificielle', 'Large Language Models', 'LLM', 
        'LLMs', 'RAG', 'benchmarks',
        # Other
        'Git', 'Linux', 'API', 'REST', 'GraphQL', 'Agile', 'Scrum'
    ]
    
    # Normaliser tout en lowercase
    all_skills_raw = list(set(list(unique_skills) + additional_skills))
    all_skills = sorted(list(set([skill.lower() for skill in all_skills_raw])))
    
    print(f"   Dictionnaire complet: {len(all_skills)} compétences (normalisées lowercase)")
    
    # ==========================================
    # 6. EXTRACTION COMPÉTENCES
    # ==========================================
    print("\n Extraction compétences par pattern matching...")
    
    df_clean['competences_found'] = df_clean['description'].apply(
        lambda x: extract_competences_from_text(x, all_skills)
    )
    
    df_clean['num_competences'] = df_clean['competences_found'].apply(len)
    
    print(f"   Compétences moyennes par offre: {df_clean['num_competences'].mean():.1f}")
    print(f"   Offres avec compétences: {(df_clean['num_competences'] > 0).sum()}")
    
    # Top compétences
    from collections import Counter
    all_comps = [comp for comps in df_clean['competences_found'] for comp in comps]
    comp_counter = Counter(all_comps)
    
    print(f"\n Top 10 compétences extraites:")
    for comp, count in comp_counter.most_common(10):
        pct = count / len(df_clean) * 100
        print(f"   {comp:<30s}: {count:4d} ({pct:5.1f}%)")
    
    # ==========================================
    # 7. STATISTIQUES PAR SOURCE
    # ==========================================
    print("\n Statistiques par source:")
    
    for source in df_clean['source_name'].unique():
        df_source = df_clean[df_clean['source_name'] == source]
        print(f"\n   {source}:")
        print(f"      Offres: {len(df_source)}")
        print(f"      Tokens moyen: {df_source['num_tokens'].mean():.0f}")
        print(f"      Compétences moyennes: {df_source['num_competences'].mean():.1f}")
        print(f"      Avec salaire: {df_source['salary_annual'].notna().sum()}")
        print(f"      Régions uniques: {df_source['region'].nunique()}")
    
    # ==========================================
    # 8. STATISTIQUES PAR RÉGION
    # ==========================================
    print("\n  Top 10 régions:")
    
    region_stats = df_clean.groupby('region').agg({
        'offre_id': 'count',
        'salary_annual': 'median',
        'num_tokens': 'mean',
        'num_competences': 'mean'
    }).round(1)
    region_stats.columns = ['nb_offres', 'salaire_median', 'tokens_moyen', 'comp_moyennes']
    region_stats = region_stats.sort_values('nb_offres', ascending=False).head(10)
    
    print(region_stats)
    
    # ==========================================
    # 9. SAUVEGARDE DATA_CLEAN.PKL (COMPLET)
    # ==========================================
    print("\n Sauvegarde data_clean.pkl (COMPLET)...")
    
    saver = ResultSaver()
    
    # Colonnes à garder
    colonnes_finales = [
        # IDs
        'offre_id', 'job_id_source',
        # Métadonnées
        'source_name', 'title', 'company_name',
        'city', 'department', 'region', 'latitude', 'longitude',
        'contract_type', 'experience_level', 'duration',
        'salary_min', 'salary_max', 'salary_annual', 'salary_text',
        'date_posted', 'url', 'scraped_at',
        # Textes (TOUS les formats)
        'description',           # Texte brut original
        'description_clean',     # HTML nettoyé
        'tokens',                # Liste tokens (stopwords filtrés, lemmatisés)
        'text_for_sklearn',      # String pour TF-IDF/LDA
        'num_tokens',            # Nombre tokens
        # Compétences
        'competences_found',     # Liste compétences extraites
        'num_competences'        # Nombre compétences
    ]
    
    df_final = df_clean[colonnes_finales].copy()
    
    print(f"\n Colonnes dans data_clean.pkl:")
    for col in colonnes_finales:
        print(f"   - {col}")
    
    # Sauvegarder
    saver.save_pickle(df_final, 'data_clean.pkl')
    
    # Export CSV (sans tokens pour lisibilité)
    df_export = df_final.drop(columns=['tokens', 'competences_found'], errors='ignore')
    saver.save_csv(df_export, 'data_clean.csv')
    
    # Sauvegarder dictionnaire compétences
    saver.save_json({'competences': all_skills}, 'dictionnaire_competences.json')
    
    # Sauvegarder compétences FT
    saver.save_pickle(df_competences, 'competences_ft.pkl')
    
    # ==========================================
    # 10. STATISTIQUES GLOBALES
    # ==========================================
    print("\n Génération statistiques globales...")
    
    stats = {
        'total_offres': len(df_final),
        'sources': df_final['source_name'].value_counts().to_dict(),
        'regions': df_final['region'].value_counts().head(10).to_dict(),
        'contract_types': df_final['contract_type'].value_counts().to_dict(),
        'salary_stats': {
            'count': int(df_final['salary_annual'].notna().sum()),
            'mean': float(df_final['salary_annual'].mean()) if df_final['salary_annual'].notna().any() else 0,
            'median': float(df_final['salary_annual'].median()) if df_final['salary_annual'].notna().any() else 0,
            'min': float(df_final['salary_annual'].min()) if df_final['salary_annual'].notna().any() else 0,
            'max': float(df_final['salary_annual'].max()) if df_final['salary_annual'].notna().any() else 0
        },
        'text_stats': {
            'tokens_mean': float(df_final['num_tokens'].mean()),
            'tokens_median': float(df_final['num_tokens'].median()),
            'tokens_min': int(df_final['num_tokens'].min()),
            'tokens_max': int(df_final['num_tokens'].max())
        },
        'competences_stats': {
            'dictionnaire_size': len(all_skills),
            'comp_mean': float(df_final['num_competences'].mean()),
            'comp_median': float(df_final['num_competences'].median()),
            'offers_with_comp': int((df_final['num_competences'] > 0).sum()),
            'top_10': [
                {'competence': c, 'count': int(n), 'percentage': float(n/len(df_final)*100)}
                for c, n in comp_counter.most_common(10)
            ]
        },
        'preprocessing': {
            'stopwords_count': len(preprocessor.stop_words),
            'stopwords_fr_en': True,
            'lemmatization': True,
            'html_cleaning': True,
            'min_token_length': 3
        },
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    saver.save_json(stats, 'stats_globales.json')
    
    # ==========================================
    # RÉSUMÉ FINAL
    # ==========================================
    print("\n" + "="*70)
    print(" PREPROCESSING MASTER TERMINÉ !")
    print("="*70)
    
    print(f"\n Fichier principal créé:")
    print(f"    data_clean.pkl ({len(df_final)} offres, {len(colonnes_finales)} colonnes)")

    print(f"\n Colonnes disponibles pour analyses:")
    print(f"    description          → Texte brut original")
    print(f"    description_clean    → HTML nettoyé")
    print(f"    tokens               → Liste tokens (clean, lemmatisés)")
    print(f"    text_for_sklearn     → String pour TF-IDF/LDA")
    print(f"    competences_found    → Compétences extraites")
    
    print(f"\n Qualité du preprocessing:")
    print(f"    Stopwords FR+EN filtrés ({len(preprocessor.stop_words)} stopwords)")
    print(f"    Lemmatisation appliquée (technique/techniques → technique)")
    print(f"    HTML nettoyé (&nbsp; supprimé)")
    print(f"    Tokens >= 3 caractères")
    print(f"    Compétences normalisées (lowercase)")
    
    print(f"\n Fichiers additionnels:")
    print(f"   - data_clean.csv (export sans tokens)")
    print(f"   - dictionnaire_competences.json ({len(all_skills)} compétences)")
    print(f"   - competences_ft.pkl (compétences France Travail)")
    print(f"   - stats_globales.json (statistiques complètes)")
    
    print(f"\n Prochaines étapes:")
    print(f"   1. python 2_extraction_competences.py")
    print(f"   2. python 3_topic_modeling.py")
    
    return df_final, all_skills


if __name__ == "__main__":
    df_clean, skills_dict = main()