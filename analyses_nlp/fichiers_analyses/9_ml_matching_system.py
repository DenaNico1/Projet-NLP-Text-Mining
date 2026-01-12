"""
9. Système ML de Matching CV ↔ Offres
Approche Hybride : Embeddings + Random Forest

Pipeline complet:
1. Génération 25 CV fictifs variés
2. Création dataset 500 paires (CV, Offre) labelisées
3. Feature engineering (6 features)
4. Entraînement Random Forest
5. Évaluation modèle
6. Sauvegarde modèle + assets

Auteur: Projet NLP Text Mining - Master SISE
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from collections import Counter
import random

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLP
from sentence_transformers import SentenceTransformer

# Utils
from tqdm import tqdm
import unicodedata

# ============================================
# CONFIGURATION
# ============================================

RESULTS_DIR = Path('../resultats_nlp')
MODELS_DIR = RESULTS_DIR / 'models'

random.seed(42)
np.random.seed(42)

# ============================================
# GÉNÉRATION CV FICTIFS
# ============================================

def generate_cv_database():
    """
    Génère 25 CV fictifs variés pour démo
    
    Returns:
        list: 25 CVs avec profils variés
    """
    
    print("\n Génération base de 25 CV fictifs...")
    
    # Noms fictifs
    prenoms = ['Alice', 'Bob', 'Claire', 'David', 'Emma', 'François', 'Gabrielle', 
               'Hugo', 'Inès', 'Jules', 'Karim', 'Léa', 'Marc', 'Nina', 'Omar',
               'Paul', 'Qing', 'Rachel', 'Sophie', 'Thomas', 'Ulysse', 'Valérie',
               'William', 'Xavier', 'Yasmine']
    
    noms = ['Martin', 'Bernard', 'Dubois', 'Thomas', 'Robert', 'Richard', 'Petit',
            'Durand', 'Leroy', 'Moreau', 'Simon', 'Laurent', 'Lefebvre', 'Michel',
            'Garcia', 'David', 'Bertrand', 'Roux', 'Vincent', 'Fournier', 'Morel',
            'Girard', 'Andre', 'Lefevre', 'Mercier']
    
    # Profils métiers
    profils = [
        {
            'titre': 'Data Scientist',
            'core_comp': ['python', 'machine learning', 'statistiques', 'sql'],
            'tech_comp': ['sklearn', 'tensorflow', 'pandas', 'numpy', 'jupyter'],
            'formations': ['Master Data Science', 'Master Statistiques', 'Doctorat ML']
        },
        {
            'titre': 'Data Engineer',
            'core_comp': ['python', 'sql', 'etl', 'cloud'],
            'tech_comp': ['spark', 'airflow', 'kafka', 'aws', 'docker', 'kubernetes'],
            'formations': ['Master Informatique', 'Ingénieur Informatique']
        },
        {
            'titre': 'Data Analyst',
            'core_comp': ['sql', 'excel', 'analyse', 'reporting'],
            'tech_comp': ['power bi', 'tableau', 'python', 'r'],
            'formations': ['Master Data Analytics', 'Licence Mathématiques']
        },
        {
            'titre': 'BI Analyst',
            'core_comp': ['sql', 'bi', 'tableau de bord', 'kpi'],
            'tech_comp': ['power bi', 'tableau', 'qlik', 'looker'],
            'formations': ['Master Business Intelligence', 'MBA']
        },
        {
            'titre': 'ML Engineer',
            'core_comp': ['machine learning', 'python', 'mlops', 'production'],
            'tech_comp': ['tensorflow', 'pytorch', 'mlflow', 'kubernetes', 'docker'],
            'formations': ['Master IA', 'Ingénieur ML']
        },
        {
            'titre': 'AI Engineer',
            'core_comp': ['intelligence artificielle', 'deep learning', 'python'],
            'tech_comp': ['tensorflow', 'pytorch', 'huggingface', 'gpu'],
            'formations': ['Doctorat IA', 'Master Deep Learning']
        }
    ]
    
    # Villes
    villes = ['Paris', 'Lyon', 'Toulouse', 'Bordeaux', 'Lille', 'Nantes', 
              'Marseille', 'Nice', 'Rennes', 'Grenoble']
    
    cvs = []
    
    for i in range(25):
        # Profil aléatoire
        profil = random.choice(profils)
        
        # Niveau expérience
        if i < 8:  # Junior
            exp = random.randint(0, 2)
            niveau = 'Junior'
        elif i < 18:  # Confirmé
            exp = random.randint(3, 5)
            niveau = 'Confirmé'
        else:  # Senior
            exp = random.randint(6, 12)
            niveau = 'Senior'
        
        
        # Compétences (core + quelques tech)
        n_tech = random.randint(3, 6)
        n_tech = min(n_tech, len(profil['tech_comp']))  # AJOUT: Limite au max dispo
        competences = profil['core_comp'] + random.sample(profil['tech_comp'], n_tech)

        # Ajouter compétences transverses
        soft_skills = ['autonomie', 'rigueur', 'communication', 'travail équipe', 'anglais']
        competences.extend(random.sample(soft_skills, min(2, len(soft_skills))))
        
        
        cv = {
            'cv_id': i + 1,
            'nom': f"{prenoms[i]} {noms[i]}",
            'email': f"{prenoms[i].lower()}.{noms[i].lower()}@email.fr",
            'titre_recherche': f"{profil['titre']} {niveau}" if niveau != 'Confirmé' else profil['titre'],
            'profil_type': profil['titre'],
            'niveau': niveau,
            'competences': competences,
            'annees_experience': exp,
            'formation': random.choice(profil['formations']),
            'localisation_preferee': random.choice(villes),
            'mobilite': random.choice([True, False]),
            'cv_text': f"{profil['titre']} {niveau} avec {exp} ans d'expérience. Compétences: {', '.join(competences[:5])}. Formation: {random.choice(profil['formations'])}."
        }
        
        cvs.append(cv)
    
    print(f"    {len(cvs)} CV générés")
    
    # Stats
    print(f"\n   Distribution:")
    profils_count = Counter([cv['profil_type'] for cv in cvs])
    for profil, count in profils_count.most_common():
        print(f"      {profil}: {count}")
    
    return cvs


# ============================================
# CRÉATION DATASET PAIRES
# ============================================

def create_matching_dataset(cvs, df_offres, n_pairs=500):
    """
    Crée dataset de paires (CV, Offre) avec labels auto
    
    Args:
        cvs: list de CV
        df_offres: DataFrame offres
        n_pairs: nombre de paires à générer
    
    Returns:
        DataFrame avec features + labels
    """
    
    print(f"\n Création dataset {n_pairs} paires...")
    
    # Normalisation texte
    def normalize(text):
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        return text.lower().strip()
    
    # Auto-labellisation
    def auto_label(cv, offre):
        """Règle pour labelliser automatiquement"""
        
        # Compétences CV
        cv_comp = set([normalize(c) for c in cv['competences']])
        
        # Compétences offre
        if isinstance(offre['competences_found'], list):
            offre_comp = set([normalize(c) for c in offre['competences_found']])
        else:
            offre_comp = set()
        
        if len(offre_comp) == 0:
            return None  # Skip si pas de compétences
        
        # Ratio compétences communes
        common = cv_comp & offre_comp
        comp_ratio = len(common) / len(offre_comp) if len(offre_comp) > 0 else 0
        
        # Match titre
        cv_titre_norm = normalize(cv['titre_recherche'])
        offre_titre_norm = normalize(offre['title'])
        
        title_match = any(word in offre_titre_norm for word in cv_titre_norm.split()[:2])
        
        # Expérience (conversion robuste)
        offre_exp = offre.get('experience_level', 0)
        try:
            offre_exp = int(offre_exp) if offre_exp else 0
        except (ValueError, TypeError):
            offre_exp = 0
        
        # Règle de labellisation
        if comp_ratio >= 0.6 and title_match:
            return 1  # MATCH
        elif comp_ratio < 0.3 or cv['annees_experience'] < (offre_exp - 2):
            return 0  # PAS MATCH
        elif comp_ratio >= 0.4:
            return 1  # MATCH partiel
        else:
            return 0
    
    # Générer paires
    pairs_data = []
    
    # Équilibrer matches/non-matches
    n_positive = n_pairs // 2
    n_negative = n_pairs // 2
    
    positive_pairs = []
    negative_pairs = []
    
    # Échantillonner offres
    offres_sample = df_offres.sample(min(1000, len(df_offres)), random_state=42)
    
    attempts = 0
    max_attempts = 5000
    
    with tqdm(total=n_pairs, desc="Génération paires") as pbar:
        while (len(positive_pairs) < n_positive or len(negative_pairs) < n_negative) and attempts < max_attempts:
            attempts += 1
            
            # Paire aléatoire
            cv = random.choice(cvs)
            offre = offres_sample.sample(1).iloc[0]
            
            label = auto_label(cv, offre)
            
            if label is None:
                continue
            
            pair = (cv, offre.to_dict(), label)
            
            if label == 1 and len(positive_pairs) < n_positive:
                positive_pairs.append(pair)
                pbar.update(1)
            elif label == 0 and len(negative_pairs) < n_negative:
                negative_pairs.append(pair)
                pbar.update(1)
    
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    print(f"    {len(all_pairs)} paires générées")
    print(f"      Matches: {len(positive_pairs)}")
    print(f"      Non-matches: {len(negative_pairs)}")
    
    return all_pairs


# ============================================
# FEATURE ENGINEERING
# ============================================

def extract_features(cv, offre, embeddings_model, tfidf_vectorizer=None, fit_tfidf=False):
    """
    Extrait 6 features d'une paire (CV, Offre)
    
    Returns:
        dict de features
    """
    
    # Normalisation
    def normalize(text):
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        return text.lower().strip()
    
    # 1. EMBEDDING SIMILARITY
    cv_text = cv['cv_text']
    offre_text = f"{offre['title']} {offre.get('description', '')[:500]}"
    
    cv_emb = embeddings_model.encode(cv_text)
    offre_emb = embeddings_model.encode(offre_text)
    
    embedding_sim = float(cosine_similarity([cv_emb], [offre_emb])[0][0])
    
    # 2. TF-IDF SIMILARITY
    if tfidf_vectorizer is None or fit_tfidf:
        tfidf_vectorizer = TfidfVectorizer(max_features=500)
        corpus = [cv_text, offre_text]
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        tfidf_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    else:
        try:
            tfidf_matrix = tfidf_vectorizer.transform([cv_text, offre_text])
            tfidf_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except:
            tfidf_sim = 0.0
    
    # 3. COMPETENCES RATIO
    cv_comp = set([normalize(c) for c in cv['competences']])
    
    if isinstance(offre.get('competences_found'), list):
        offre_comp = set([normalize(c) for c in offre['competences_found']])
    else:
        offre_comp = set()
    
    if len(offre_comp) > 0:
        comp_ratio = len(cv_comp & offre_comp) / len(offre_comp)
    else:
        comp_ratio = 0.0
    
    # 4. COMPETENCES COUNT MATCH
    comp_count = len(cv_comp & offre_comp)
    
    # 5. EXPERIENCE GAP
    offre_exp = offre.get('experience_level', 0)
    try:
        offre_exp = int(offre_exp) if offre_exp else 0
    except (ValueError, TypeError):
        offre_exp = 0

    exp_gap = float(offre_exp - cv['annees_experience'])
    
    # 6. TITLE SIMILARITY (fuzzy)
    cv_title = normalize(cv['titre_recherche'])
    offre_title = normalize(offre['title'])
    
    cv_words = set(cv_title.split())
    offre_words = set(offre_title.split())
    
    if len(offre_words) > 0:
        title_sim = len(cv_words & offre_words) / len(offre_words)
    else:
        title_sim = 0.0
    
    features = {
        'embedding_similarity': embedding_sim,
        'tfidf_similarity': tfidf_sim,
        'comp_ratio': comp_ratio,
        'comp_count_match': comp_count,
        'experience_gap': exp_gap,
        'title_similarity': title_sim
    }
    
    return features, tfidf_vectorizer


# ============================================
# ENTRAÎNEMENT MODÈLE
# ============================================

def train_matching_model(pairs, embeddings_model):
    """
    Entraîne Random Forest sur paires
    
    Returns:
        model, tfidf_vectorizer, metrics
    """
    
    print(f"\n Extraction features + entraînement...")
    
    # Features extraction
    X = []
    y = []
    
    tfidf_vectorizer = None
    
    for i, (cv, offre, label) in enumerate(tqdm(pairs, desc="Features")):
        features, tfidf_vectorizer = extract_features(
            cv, offre, embeddings_model, tfidf_vectorizer, 
            fit_tfidf=(i==0)
        )
        
        feature_values = [
            features['embedding_similarity'],
            features['tfidf_similarity'],
            features['comp_ratio'],
            features['comp_count_match'],
            features['experience_gap'],
            features['title_similarity']
        ]
        
        X.append(feature_values)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"    Features: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entraînement
    print(f"\n   Entraînement Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    # Feature importance
    feature_names = [
        'embedding_similarity',
        'tfidf_similarity', 
        'comp_ratio',
        'comp_count_match',
        'experience_gap',
        'title_similarity'
    ]
    
    feature_importance = {
        name: float(importance)
        for name, importance in zip(feature_names, model.feature_importances_)
    }
    
    metrics['feature_importance'] = feature_importance
    
    print(f"\n   Résultats:")
    print(f"      Accuracy:  {metrics['accuracy']:.3f}")
    print(f"      Precision: {metrics['precision']:.3f}")
    print(f"      Recall:    {metrics['recall']:.3f}")
    print(f"      F1-Score:  {metrics['f1_score']:.3f}")
    print(f"      ROC-AUC:   {metrics['roc_auc']:.3f}")
    
    print(f"\n   Feature Importance:")
    for name, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"      {name:25s}: {importance:.3f}")
    
    return model, tfidf_vectorizer, metrics


# ============================================
# MAIN
# ============================================

def main():
    print("="*70)
    print(" SYSTÈME ML DE MATCHING CV ↔ OFFRES")
    print("="*70)
    
    # 1. Charger offres
    print(f"\n Chargement offres...")
    with open(MODELS_DIR / 'data_with_profiles.pkl', 'rb') as f:
        df_offres = pickle.load(f)
    
    print(f"    {len(df_offres)} offres chargées")
    
    # 2. Générer CV fictifs
    cvs = generate_cv_database()
    
    # 3. Charger modèle embeddings
    print(f"\n Chargement sentence-transformers...")
    embeddings_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print(f"    Modèle chargé")
    
    # 4. Créer dataset
    pairs = create_matching_dataset(cvs, df_offres, n_pairs=500)
    
    # 5. Entraîner modèle
    model, tfidf_vectorizer, metrics = train_matching_model(pairs, embeddings_model)
    
    # 6. Sauvegarder
    print(f"\n Sauvegarde...")
    
    # CV base
    with open(RESULTS_DIR / 'cv_base_fictifs.json', 'w', encoding='utf-8') as f:
        json.dump(cvs, f, indent=2, ensure_ascii=False)
    print(f"    cv_base_fictifs.json")
    
    # Modèle
    with open(MODELS_DIR / 'matching_model.pkl', 'wb') as f:
        pickle.dump({
            'rf_model': model,
            'tfidf_vectorizer': tfidf_vectorizer,
            'embeddings_model_name': 'paraphrase-multilingual-MiniLM-L12-v2'
        }, f)
    print(f"    matching_model.pkl")
    
    # Métriques
    with open(RESULTS_DIR / 'matching_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"    matching_metrics.json")
    
    print("\n" + "="*70)
    print(" SYSTÈME ML MATCHING TERMINÉ !")
    print("="*70)
    print(f"\n Performance: {metrics['accuracy']*100:.1f}% accuracy")
    print(f" Fichiers créés:")
    print(f"   - cv_base_fictifs.json (25 CV)")
    print(f"   - matching_model.pkl (Random Forest)")
    print(f"   - matching_metrics.json (métriques)")


if __name__ == "__main__":
    main()