"""
Analyse des Titres Non Classifiés
Identifie les patterns manquants dans la classification

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import pandas as pd
import pickle
from pathlib import Path
from collections import Counter
import re


def load_data():
    """Charge data_with_profiles.pkl"""
    print("="*70)
    print("📊 ANALYSE DES TITRES NON CLASSIFIÉS")
    print("="*70)
    
    pkl_path = Path('../resultats_nlp/models/data_with_profiles.pkl')
    
    print(f"\n📥 Chargement {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    
    print(f"   ✅ Offres chargées: {len(df)}")
    
    return df


def analyze_classification_stats(df):
    """Statistiques générales de classification"""
    print("\n" + "="*70)
    print("📈 STATISTIQUES GÉNÉRALES")
    print("="*70)
    
    n_total = len(df)
    n_classified = (df['profil_assigned'] != 'Non classifié').sum()
    n_unclassified = (df['profil_assigned'] == 'Non classifié').sum()
    
    print(f"\nTotal offres:     {n_total:,}")
    print(f"Classifiées:      {n_classified:,} ({n_classified/n_total*100:.1f}%)")
    print(f"Non classifiées:  {n_unclassified:,} ({n_unclassified/n_total*100:.1f}%)")
    
    print("\n📊 Distribution profils classifiés:")
    profil_counts = df[df['profil_assigned'] != 'Non classifié']['profil_assigned'].value_counts()
    
    for profil, count in profil_counts.items():
        pct = count / n_total * 100
        print(f"   {profil:<30s}: {count:4d} ({pct:5.1f}%)")


def analyze_unclassified_titles(df):
    """Analyse des titres non classifiés"""
    print("\n" + "="*70)
    print("🔍 TOP 50 TITRES NON CLASSIFIÉS")
    print("="*70)
    
    df_unclassified = df[df['profil_assigned'] == 'Non classifié'].copy()
    
    print(f"\nTotal non classifiés: {len(df_unclassified):,}")
    
    # Top 50 titres
    title_counts = df_unclassified['title'].value_counts().head(50)
    
    print(f"\nTop 50 titres (représentent {title_counts.sum()} offres):\n")
    
    for i, (title, count) in enumerate(title_counts.items(), 1):
        print(f"{i:2d}. [{count:3d}x] {title}")
    
    return df_unclassified, title_counts


def extract_keywords_from_titles(df_unclassified):
    """Extrait les mots-clés fréquents des titres non classifiés"""
    print("\n" + "="*70)
    print("🔑 MOTS-CLÉS FRÉQUENTS DANS TITRES NON CLASSIFIÉS")
    print("="*70)
    
    # Tous les titres en lowercase
    all_titles = ' '.join(df_unclassified['title'].fillna('').str.lower())
    
    # Extraire mots (au moins 4 lettres)
    words = re.findall(r'\b[a-zàâäéèêëïîôùûüç]{4,}\b', all_titles)
    
    # Stopwords basiques
    stopwords = {
        'dans', 'pour', 'avec', 'chez', 'vers', 'sous', 'sans',
        'stage', 'alternance', 'cdi', 'cdd', 'apprentissage',
        'junior', 'senior', 'lead', 'expert',
        'paris', 'lyon', 'marseille', 'toulouse', 'bordeaux',
        'france', 'remote', 'télétravail', 'teletravail'
    }
    
    # Filtrer stopwords
    words_filtered = [w for w in words if w not in stopwords]
    
    # Compter
    word_counts = Counter(words_filtered)
    
    print("\nTop 30 mots-clés:")
    for i, (word, count) in enumerate(word_counts.most_common(30), 1):
        print(f"{i:2d}. {word:<20s}: {count:4d}x")
    
    return word_counts


def identify_patterns(title_counts):
    """Identifie les patterns de titres"""
    print("\n" + "="*70)
    print("🎯 PATTERNS IDENTIFIÉS")
    print("="*70)
    
    patterns = {
        'Ingénieur': [],
        'Développeur': [],
        'Analyste': [],
        'Consultant': [],
        'Architecte': [],
        'Chef de projet': [],
        'Product': [],
        'Manager': [],
        'Autres': []
    }
    
    for title, count in title_counts.items():
        title_lower = title.lower()
        
        matched = False
        for pattern in patterns.keys():
            if pattern.lower() in title_lower:
                patterns[pattern].append((title, count))
                matched = True
                break
        
        if not matched:
            patterns['Autres'].append((title, count))
    
    # Afficher patterns
    for pattern, titles in patterns.items():
        if len(titles) > 0:
            total = sum(count for _, count in titles)
            print(f"\n📌 {pattern} ({len(titles)} titres uniques, {total} offres):")
            
            # Top 10 de ce pattern
            for title, count in sorted(titles, key=lambda x: x[1], reverse=True)[:10]:
                print(f"   [{count:3d}x] {title}")


def analyze_scores_unclassified(df):
    """Analyse les scores des offres non classifiées"""
    print("\n" + "="*70)
    print("📊 ANALYSE SCORES OFFRES NON CLASSIFIÉES")
    print("="*70)
    
    df_unclassified = df[df['profil_assigned'] == 'Non classifié'].copy()
    
    print(f"\nScore moyen:    {df_unclassified['profil_score'].mean():.2f}/10")
    print(f"Score médian:   {df_unclassified['profil_score'].median():.2f}/10")
    print(f"Score min:      {df_unclassified['profil_score'].min():.2f}/10")
    print(f"Score max:      {df_unclassified['profil_score'].max():.2f}/10")
    
    # Distribution scores
    print("\nDistribution scores:")
    bins = [0, 2, 3, 4, 5, 6, 7, 8, 10]
    labels = ['0-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-10']
    
    df_unclassified['score_bin'] = pd.cut(
        df_unclassified['profil_score'], 
        bins=bins, 
        labels=labels,
        include_lowest=True
    )
    
    score_dist = df_unclassified['score_bin'].value_counts().sort_index()
    
    for score_range, count in score_dist.items():
        pct = count / len(df_unclassified) * 100
        bar = '█' * int(pct / 2)
        print(f"   {score_range}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Profils second (qui auraient pu matcher)
    print("\n🥈 Top profils en 2ème position (qui auraient pu matcher):")
    second_counts = df_unclassified['profil_second'].value_counts().head(10)
    
    for profil, count in second_counts.items():
        pct = count / len(df_unclassified) * 100
        print(f"   {profil:<30s}: {count:4d} ({pct:5.1f}%)")


def analyze_by_source(df):
    """Analyse par source"""
    print("\n" + "="*70)
    print("📍 ANALYSE PAR SOURCE")
    print("="*70)
    
    df_unclassified = df[df['profil_assigned'] == 'Non classifié'].copy()
    
    print("\nTaux de non-classification par source:")
    
    for source in df['source_name'].unique():
        df_source = df[df['source_name'] == source]
        df_source_unclass = df_unclassified[df_unclassified['source_name'] == source]
        
        n_total = len(df_source)
        n_unclass = len(df_source_unclass)
        pct = n_unclass / n_total * 100 if n_total > 0 else 0
        
        print(f"   {source:<20s}: {n_unclass:4d}/{n_total:4d} ({pct:5.1f}%)")


def generate_recommendations(df, title_counts, word_counts):
    """Génère des recommandations"""
    print("\n" + "="*70)
    print("💡 RECOMMANDATIONS")
    print("="*70)
    
    df_unclassified = df[df['profil_assigned'] == 'Non classifié'].copy()
    
    # Analyser les mots-clés manquants
    print("\n1️⃣ PROFILS À AJOUTER/ENRICHIR:")
    
    # Identifier patterns
    common_words = [w for w, c in word_counts.most_common(20)]
    
    suggestions = {
        'Data Engineer': ['ingénieur', 'développeur', 'data', 'big', 'cloud', 'plateforme'],
        'Data Scientist': ['scientist', 'scientifique', 'machine', 'learning', 'intelligence'],
        'Data Analyst': ['analyste', 'analyst', 'business', 'reporting'],
        'BI Analyst': ['tableau', 'power', 'looker', 'qlik', 'business intelligence'],
        'DevOps/MLOps': ['devops', 'mlops', 'kubernetes', 'docker', 'infrastructure'],
        'Product Manager': ['product', 'manager', 'chef', 'projet'],
        'Autres profils': ['architecte', 'consultant', 'lead', 'expert']
    }
    
    for profil, keywords in suggestions.items():
        matches = [w for w in common_words if any(k in w for k in keywords)]
        if matches:
            print(f"\n   📌 {profil}:")
            print(f"      Mots-clés trouvés: {', '.join(matches[:5])}")
    
    # Variantes à ajouter
    print("\n2️⃣ VARIANTES DE TITRES À AJOUTER:")
    
    print("\n   Exemples de titres fréquents non matchés:")
    for i, (title, count) in enumerate(list(title_counts.items())[:15], 1):
        print(f"   {i:2d}. [{count:3d}x] {title}")
    
    # Seuils
    print("\n3️⃣ AJUSTEMENT SEUILS:")
    
    score_stats = df_unclassified['profil_score'].describe()
    
    print(f"\n   Score médian non classifiés: {score_stats['50%']:.2f}/10")
    print(f"   Score Q3 non classifiés:     {score_stats['75%']:.2f}/10")
    
    if score_stats['75%'] >= 4.5:
        print(f"\n   ⚠️ 75% des non-classifiés ont score ≥ {score_stats['75%']:.2f}")
        print(f"   → Recommandation: Baisser min_score à {score_stats['75%']:.2f}")


def save_results(df, title_counts, word_counts):
    """Sauvegarde les résultats"""
    print("\n" + "="*70)
    print("💾 SAUVEGARDE RÉSULTATS")
    print("="*70)
    
    output_dir = Path('../resultats_nlp')
    
    # 1. Top titres non classifiés
    df_top_titles = pd.DataFrame({
        'titre': title_counts.index,
        'count': title_counts.values
    })
    
    output_file = output_dir / 'titres_non_classifies.csv'
    df_top_titles.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ {output_file}")
    
    # 2. Mots-clés fréquents
    df_keywords = pd.DataFrame({
        'mot_cle': [w for w, c in word_counts.most_common(100)],
        'count': [c for w, c in word_counts.most_common(100)]
    })
    
    output_file = output_dir / 'mots_cles_titres.csv'
    df_keywords.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ {output_file}")
    
    # 3. Échantillon offres non classifiées
    df_unclassified = df[df['profil_assigned'] == 'Non classifié'].copy()
    
    df_sample = df_unclassified[[
        'title', 'profil_score', 'profil_second', 'profil_second_score',
        'score_title', 'score_description', 'score_competences',
        'region', 'source_name'
    ]].head(200)
    
    output_file = output_dir / 'echantillon_non_classifies.csv'
    df_sample.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ {output_file}")


def main():
    """Pipeline analyse"""
    
    # Charger données
    df = load_data()
    
    # Stats générales
    analyze_classification_stats(df)
    
    # Titres non classifiés
    df_unclassified, title_counts = analyze_unclassified_titles(df)
    
    # Mots-clés
    word_counts = extract_keywords_from_titles(df_unclassified)
    
    # Patterns
    identify_patterns(title_counts)
    
    # Scores
    analyze_scores_unclassified(df)
    
    # Par source
    analyze_by_source(df)
    
    # Recommandations
    generate_recommendations(df, title_counts, word_counts)
    
    # Sauvegarder
    save_results(df, title_counts, word_counts)
    
    print("\n" + "="*70)
    print("✅ ANALYSE TERMINÉE !")
    print("="*70)
    print("\n📁 Fichiers créés:")
    print("   - titres_non_classifies.csv")
    print("   - mots_cles_titres.csv")
    print("   - echantillon_non_classifies.csv")
    print("\n💡 Utilise ces insights pour ajuster les profils et variantes !")


if __name__ == "__main__":
    main()