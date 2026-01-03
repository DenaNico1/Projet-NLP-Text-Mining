"""
2. Extraction de Compétences -
Lit data_clean.pkl (déjà preprocessé) et fait TF-IDF + visualisations

CHANGEMENTS :
- Lit data_clean.pkl au lieu de refaire le preprocessing
- Utilise text_for_sklearn (déjà clean) pour TF-IDF/N-grams
- Compétences déjà extraites dans data_clean.pkl

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

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))
from utils import ResultSaver


def extract_ngrams(texts, n=2, top_k=50):
    """Extrait les n-grams les plus fréquents"""
    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        max_features=top_k
    )
    
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    frequencies = X.sum(axis=0).A1
    
    ngrams = sorted(
        zip(feature_names, frequencies),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ngrams


def compute_tfidf_keywords(texts, top_k=100):
    """Calcule les mots-clés TF-IDF"""
    vectorizer = TfidfVectorizer(
        max_features=top_k,
        min_df=5,
        max_df=0.8
    )
    
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.mean(axis=0).A1
    
    df_tfidf = pd.DataFrame({
        'terme': feature_names,
        'tfidf_score': tfidf_scores
    }).sort_values('tfidf_score', ascending=False)
    
    return df_tfidf


def create_wordcloud(text_data, title="Word Cloud"):
    """Crée un word cloud"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(text_data)
    
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=20, pad=20)
    
    plt.tight_layout()
    return fig


def main():
    """Pipeline extraction compétences SIMPLIFIÉ"""
    print("="*70)
    print("🎓 ÉTAPE 2 : EXTRACTION DE COMPÉTENCES ")
    print("="*70)
    
    saver = ResultSaver()
    
    # ==========================================
    # 1. CHARGEMENT DATA_CLEAN.PKL
    # ==========================================
    print("\n Chargement data_clean.pkl (déjà preprocessé)...")
    
    with open('../resultats_nlp/models/data_clean.pkl', 'rb') as f:
        df = pickle.load(f)
    
    print(f"    Offres: {len(df)}")
    print(f"    Compétences déjà extraites: {df['num_competences'].sum():.0f}")
    print(f"    Text for sklearn disponible: {len(df['text_for_sklearn'])}")
    
    # ==========================================
    # 2. STATISTIQUES COMPÉTENCES
    # ==========================================
    print("\n Statistiques compétences...")
    
    print(f"   Compétences moyennes par offre: {df['num_competences'].mean():.1f}")
    print(f"   Offres avec compétences: {(df['num_competences'] > 0).sum()}")
    
    # Compter toutes les compétences
    all_comps = [comp for comps in df['competences_found'] for comp in comps]
    comp_counter = Counter(all_comps)
    
    print(f"\n Top 20 compétences:")
    for comp, count in comp_counter.most_common(20):
        pct = count / len(df) * 100
        print(f"   {comp:<30s}: {count:4d} ({pct:5.1f}%)")
    
    # ==========================================
    # 3. TF-IDF - TERMES IMPORTANTS
    # ==========================================
    print("\n Calcul TF-IDF (sur text_for_sklearn)...")
    
    df_tfidf = compute_tfidf_keywords(df['text_for_sklearn'], top_k=100)
    
    print(f"\n Top 20 termes TF-IDF:")
    print(df_tfidf.head(20).to_string(index=False))
    
    # ==========================================
    # 4. N-GRAMS
    # ==========================================
    print("\n Extraction N-grams (sur text_for_sklearn)...")
    
    # Bigrams
    bigrams = extract_ngrams(df['text_for_sklearn'], n=2, top_k=50)
    print(f"\n Top 20 Bi-grams:")
    for ngram, freq in bigrams[:20]:
        print(f"   {ngram:<40s}: {freq:4.0f}")
    
    # Trigrams
    trigrams = extract_ngrams(df['text_for_sklearn'], n=3, top_k=30)
    print(f"\n Top 15 Tri-grams:")
    for ngram, freq in trigrams[:15]:
        print(f"   {ngram:<50s}: {freq:4.0f}")
    
    # ==========================================
    # 5. COMPÉTENCES PAR SOURCE
    # ==========================================
    print("\n Compétences par source:")
    
    comp_by_source = {}
    for source in df['source_name'].unique():
        df_source = df[df['source_name'] == source]
        comps = [c for cs in df_source['competences_found'] for c in cs]
        counter = Counter(comps)
        comp_by_source[source] = counter.most_common(10)
        
        print(f"\n   {source}:")
        for comp, count in counter.most_common(10):
            print(f"      {comp:<30s}: {count:4d}")
    
    # ==========================================
    # 6. COMPÉTENCES PAR RÉGION
    # ==========================================
    print("\n  Compétences par région (Top 5 régions):")
    
    top_regions = df['region'].value_counts().head(5).index
    
    comp_by_region = {}
    for region in top_regions:
        df_region = df[df['region'] == region]
        comps = [c for cs in df_region['competences_found'] for c in cs]
        counter = Counter(comps)
        comp_by_region[region] = counter.most_common(10)
        
        print(f"\n   {region}:")
        for comp, count in counter.most_common(10):
            pct = count / len(df_region) * 100
            print(f"      {comp:<30s}: {count:3d} ({pct:5.1f}%)")
    
    # ==========================================
    # 7. CO-OCCURRENCE DE COMPÉTENCES
    # ==========================================
    print("\n Co-occurrences de compétences...")
    
    top_20_comps = [c for c, _ in comp_counter.most_common(20)]
    cooc_matrix = np.zeros((len(top_20_comps), len(top_20_comps)))
    
    for comps_list in df['competences_found']:
        for i, comp1 in enumerate(top_20_comps):
            if comp1 in comps_list:
                for j, comp2 in enumerate(top_20_comps):
                    if comp2 in comps_list and i != j:
                        cooc_matrix[i, j] += 1
    
    # Top paires
    pairs = []
    for i in range(len(top_20_comps)):
        for j in range(i+1, len(top_20_comps)):
            if cooc_matrix[i, j] > 10:
                pairs.append((
                    top_20_comps[i],
                    top_20_comps[j],
                    int(cooc_matrix[i, j])
                ))
    
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    
    print(f"\n🔗 Top 20 paires de compétences:")
    for comp1, comp2, count in pairs_sorted[:20]:
        print(f"   {comp1:<20s} + {comp2:<20s}: {count:3d} offres")
    
    # ==========================================
    # 8. VISUALISATIONS
    # ==========================================
    print("\n Création des visualisations...")
    
    # Word Cloud
    print("   Word cloud compétences...")
    comp_freq = dict(comp_counter.most_common(100))
    fig_wc = create_wordcloud(comp_freq, "Compétences les Plus Demandées")
    saver.save_visualization(fig_wc, 'wordcloud_competences.png')
    plt.close()
    
    # Bar Chart Top 30
    print("   Bar chart top 30...")
    top_30 = comp_counter.most_common(30)
    df_top30 = pd.DataFrame(top_30, columns=['Compétence', 'Fréquence'])
    df_top30['Pourcentage'] = df_top30['Fréquence'] / len(df) * 100
    
    fig = px.bar(
        df_top30.sort_values('Fréquence'),
        x='Fréquence',
        y='Compétence',
        orientation='h',
        title='Top 30 Compétences Demandées',
        labels={'Fréquence': 'Nombre d\'offres'},
        color='Pourcentage',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=800, showlegend=False)
    saver.save_visualization(fig, 'top30_competences.html')
    
    # Heatmap co-occurrence
    print("   Heatmap co-occurrences...")
    fig_heat, ax = plt.subplots(figsize=(14, 12))
    cooc_norm = cooc_matrix / cooc_matrix.max()
    
    sns.heatmap(
        cooc_norm,
        xticklabels=top_20_comps,
        yticklabels=top_20_comps,
        cmap='YlOrRd',
        annot=False,
        square=True,
        cbar_kws={'label': 'Co-occurrence normalisée'},
        ax=ax
    )
    ax.set_title('Co-occurrence des Top 20 Compétences', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    saver.save_visualization(fig_heat, 'heatmap_cooccurrence.png')
    plt.close()
    
    # Compétences par région
    print("   Compétences par région...")
    comp_region_data = []
    for region in top_regions:
        for comp, count in comp_by_region[region][:10]:
            comp_region_data.append({
                'Région': region,
                'Compétence': comp,
                'Nombre': count
            })
    
    df_comp_region = pd.DataFrame(comp_region_data)
    fig = px.bar(
        df_comp_region,
        x='Région',
        y='Nombre',
        color='Compétence',
        title='Top 10 Compétences par Région',
        barmode='group',
        height=600
    )
    saver.save_visualization(fig, 'competences_par_region.html')
    
    # ==========================================
    # 9. SAUVEGARDE RÉSULTATS
    # ==========================================
    print("\n Sauvegarde des résultats...")
    
    results = {
        'top_competences': [
            {'competence': c, 'count': int(n), 'percentage': float(n/len(df)*100)}
            for c, n in comp_counter.most_common(100)
        ],
        'bigrams': [
            {'bigram': bg, 'frequency': int(freq)}
            for bg, freq in bigrams[:50]
        ],
        'trigrams': [
            {'trigram': tg, 'frequency': int(freq)}
            for tg, freq in trigrams[:30]
        ],
        'cooccurrences': [
            {'comp1': c1, 'comp2': c2, 'count': int(cnt)}
            for c1, c2, cnt in pairs_sorted[:50]
        ],
        'competences_par_source': {
            source: [{'competence': c, 'count': int(n)} for c, n in comps]
            for source, comps in comp_by_source.items()
        },
        'competences_par_region': {
            region: [{'competence': c, 'count': int(n)} for c, n in comps]
            for region, comps in comp_by_region.items()
        }
    }
    
    saver.save_json(results, 'competences_extracted.json')
    saver.save_csv(df_tfidf, 'tfidf_keywords.csv')
    
    df_cooc = pd.DataFrame(cooc_matrix, index=top_20_comps, columns=top_20_comps)
    saver.save_csv(df_cooc, 'cooccurrence_matrix.csv')
    
    # Sauvegarder DataFrame avec analyses
    saver.save_pickle(df, 'data_with_analyses.pkl')
    
    print("\n EXTRACTION DE COMPÉTENCES TERMINÉE !")
    print(f"\n Résultats:")
    print(f"   - {len(comp_counter)} compétences uniques")
    print(f"   - {len(bigrams)} bi-grams")
    print(f"   - {len(trigrams)} tri-grams")
    print(f"   - {len(pairs_sorted)} paires co-occurrences")
    
    print(f"\n Fichiers créés:")
    print(f"   - competences_extracted.json")
    print(f"   - data_with_analyses.pkl")
    print(f"   - tfidf_keywords.csv")
    print(f"   - cooccurrence_matrix.csv")
    print(f"   - wordcloud_competences.png")
    print(f"   - top30_competences.html")
    print(f"   - heatmap_cooccurrence.png")
    print(f"   - competences_par_region.html")


if __name__ == "__main__":
    main()