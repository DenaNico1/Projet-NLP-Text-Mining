"""
3. Topic Modeling - Découverte des Profils Métiers
Utilise LDA pour identifier automatiquement les profils cachés

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Visualisation LDA
try:
    import pyLDAvis
    import pyLDAvis.lda_model
    PYLDAVIS_AVAILABLE = True
except:
    PYLDAVIS_AVAILABLE = False
    print("⚠️  pyLDAvis non disponible (optionnel)")

sys.path.insert(0, str(Path(__file__).parent))
from utils import ResultSaver


def train_lda_model(texts, n_topics=6, n_top_words=15):
    """
    Entraîne un modèle LDA
    
    Args:
        texts: Liste de textes
        n_topics: Nombre de topics
        n_top_words: Nombre de mots par topic
    
    Returns:
        model, vectorizer, doc_topics, feature_names
    """
    # Vectorization
    vectorizer = CountVectorizer(
        max_features=1000,
        min_df=5,
        max_df=0.7,
        token_pattern=r'\b[a-zàâäéèêëïîôöùûüÿç]{3,}\b'
    )
    
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method='online'
    )
    
    doc_topics = lda.fit_transform(X)
    
    return lda, vectorizer, doc_topics, feature_names


def display_topics(model, feature_names, n_top_words=15):
    """
    Affiche les topics
    
    Returns:
        dict des topics
    """
    topics = {}
    
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = [topic[i] for i in top_indices]
        
        topics[f"Topic {topic_idx + 1}"] = {
            'words': top_words,
            'scores': top_scores
        }
        
        print(f"\nTopic {topic_idx + 1}:")
        print(f"  Mots clés: {', '.join(top_words[:10])}")
    
    return topics


def main():
    """
    Pipeline Topic Modeling
    """
    print("="*70)
    print("🧠 ÉTAPE 3 : TOPIC MODELING (LDA)")
    print("="*70)
    
    saver = ResultSaver()
    
    # Chargement
    print("\n📥 Chargement des données...")
    with open('../resultats_nlp/models/data_with_competences.pkl', 'rb') as f:
        df = pickle.load(f)
    
    print(f"   Offres: {len(df)}")
    
    # Topic Modeling
    print("\n🔄 Entraînement modèle LDA...")
    print("   (Cela peut prendre quelques minutes...)")
    
    N_TOPICS = 6
    
    lda, vectorizer, doc_topics, feature_names = train_lda_model(
        df['description_clean'],
        n_topics=N_TOPICS,
        n_top_words=15
    )
    
    print(f"\n✅ Modèle entraîné avec {N_TOPICS} topics")
    
    # Afficher topics
    print("\n📋 Topics découverts:")
    topics_dict = display_topics(lda, feature_names, n_top_words=15)
    
    # Assigner topic dominant
    df['topic_dominant'] = doc_topics.argmax(axis=1)
    df['topic_score'] = doc_topics.max(axis=1)
    
    # Statistiques par topic
    print("\n📊 Distribution des topics:")
    for topic_id in range(N_TOPICS):
        count = (df['topic_dominant'] == topic_id).sum()
        pct = count / len(df) * 100
        avg_salary = df[df['topic_dominant'] == topic_id]['salary_annual'].median()
        print(f"   Topic {topic_id + 1}: {count:4d} offres ({pct:5.1f}%) - Salaire médian: {avg_salary:.0f}€")
    
    # Visualisations
    print("\n📊 Création visualisations...")
    
    # Distribution des topics
    topic_counts = df['topic_dominant'].value_counts().sort_index()
    fig = px.bar(
        x=[f"Topic {i+1}" for i in range(N_TOPICS)],
        y=topic_counts.values,
        title="Distribution des Topics (Profils Métiers)",
        labels={'x': 'Topic', 'y': 'Nombre d\'offres'}
    )
    saver.save_visualization(fig, 'topics_distribution.html')
    
    # Sauvegarde
    print("\n💾 Sauvegarde...")
    
    results = {
        'n_topics': N_TOPICS,
        'topics': {
            name: {'words': data['words'], 'scores': [float(s) for s in data['scores']]}
            for name, data in topics_dict.items()
        },
        'distribution': {
            f"Topic {i+1}": int((df['topic_dominant'] == i).sum())
            for i in range(N_TOPICS)
        }
    }
    
    saver.save_json(results, 'topics_lda.json')
    saver.save_pickle(lda, 'lda_model.pkl')
    saver.save_pickle(df, 'data_with_topics.pkl')
    
    print("\n✅ TOPIC MODELING TERMINÉ !")
    print(f"\n📁 Fichiers: topics_lda.json, lda_model.pkl, data_with_topics.pkl")


if __name__ == "__main__":
    main()