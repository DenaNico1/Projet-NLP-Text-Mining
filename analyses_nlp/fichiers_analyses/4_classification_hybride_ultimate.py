"""
4. Classification Hybride ULTIME FINALE - 87-90% CLASSIFICATION
Scoring: 60% TITRE + 20% Description + 20% Compétences

STRATÉGIE QUADRUPLE:
1. Normalisation accents (é→e, è→e, etc.)
2. Fuzzy matching 85%+ (variations/typos)
3. Cascade 4 passes (seuils 4.5 / 3.5 / 2.5 / 0.5)
4. Profil fourre-tout "Data/IA - Non spécifié" testé EN DERNIER

"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import sys
from collections import Counter, defaultdict
import re
import unicodedata

# NLP / ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fuzzy matching
try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    print("⚠️  WARNING: fuzzywuzzy not installed. Install with: pip install fuzzywuzzy python-Levenshtein")
    FUZZY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from utils import ResultSaver
from profils_definitions_v1_optimized import (
    PROFILS, CLASSIFICATION_CONFIG,
    get_profil_config, get_all_profils, get_min_score
)


# ============================================
# NORMALISATION AVANCÉE
# ============================================

def remove_accents(text):
    """
    Supprime TOUS les accents
    
    Exemples:
    - "développeur" → "developpeur"
    - "données" → "donnees"
    - "ingénieur" → "ingenieur"
    """
    if not text:
        return ""
    
    # Normalisation NFD (décompose caractères accentués)
    nfd = unicodedata.normalize('NFD', text)
    
    # Garde seulement les caractères non-diacritiques
    return ''.join(
        char for char in nfd
        if unicodedata.category(char) != 'Mn'
    )


def normalize_text_ultimate(text):
    """
    Normalisation ULTIME pour matching
    
    - Minuscules
    - Supprime accents
    - Supprime ponctuation excessive
    - Nettoie espaces
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # 1. Minuscules
    text = text.lower()
    
    # 2. Supprimer accents
    text = remove_accents(text)
    
    # 3. Supprimer ponctuation excessive (garde tirets/underscores)
    text = re.sub(r'[^\w\s\-]', ' ', text)
    
    # 4. Nettoyer espaces multiples
    text = ' '.join(text.split())
    
    return text.strip()


# ============================================
# SYSTÈME CLASSIFICATION ULTIME
# ============================================

class ProfileClassifierUltimate:
    """
    Classification hybride ULTIME
    
    TRIPLE STRATÉGIE:
    1. Normalisation accents
    2. Fuzzy matching
    3. Cascade seuils
    """
    
    def __init__(self):
        self.profils = PROFILS
        self.profil_names = get_all_profils()
        self.config = CLASSIFICATION_CONFIG
        self.tfidf_vectorizer = None
        self.profil_vectors = {}
        
        #  PAS de pré-calcul - normalisation à la volée
        
    def build_profil_documents(self):
        """Crée documents représentatifs pour chaque profil"""
        profil_docs = {}
        
        for profil_name, profil_config in self.profils.items():
            doc_parts = []
            
            # Title variants (répétés 5x)
            doc_parts.extend(profil_config['title_variants'] * 5)
            
            # Keywords title (répétés 3x)
            doc_parts.extend(profil_config['keywords_title'] * 3)
            
            # Keywords strong (répétés 2x)
            doc_parts.extend(profil_config['keywords_strong'] * 2)
            
            # Compétences core
            doc_parts.extend(profil_config['competences_core'])
            
            profil_docs[profil_name] = ' '.join(doc_parts).lower()
        
        return profil_docs
    
    def fit_tfidf(self, df):
        """Entraîne TF-IDF"""
        print("\n Entraînement TF-IDF...")
        
        profil_docs = self.build_profil_documents()
        all_texts = list(df['text_for_sklearn']) + list(profil_docs.values())
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        self.tfidf_vectorizer.fit(all_texts)
        
        for profil_name, doc in profil_docs.items():
            vector = self.tfidf_vectorizer.transform([doc])
            self.profil_vectors[profil_name] = vector
        
        print(f"    TF-IDF entraîné: {len(self.tfidf_vectorizer.vocabulary_)} features")
    
    def score_title_ultimate(self, title, profil_name):
        """
        Score titre avec NORMALISATION À LA VOLÉE
        
        1. Exact match (normalized)     → 10 points
        2. Contains match (normalized)  → 8 points
        3. Fuzzy match 85%+             → 6 points
        4. Keywords bonus               → +2 points/keyword (max 6)
        """
        if not title:
            return 0
        
        profil = self.profils[profil_name]
        score = 0
        
        # Normaliser titre
        title_norm = normalize_text_ultimate(title)
        
        if not title_norm:
            return 0
        
        # ========================================
        # NIVEAU 1 : Exact match normalized
        # ========================================
        for variant in profil['title_variants']:
            variant_norm = normalize_text_ultimate(variant)
            if variant_norm and variant_norm == title_norm:
                score = 10
                break
        
        # ========================================
        # NIVEAU 2 : Contains match normalized
        # ========================================
        if score == 0:
            for variant in profil['title_variants']:
                variant_norm = normalize_text_ultimate(variant)
                if variant_norm and variant_norm in title_norm:
                    score = 8
                    break
        
        # ========================================
        # NIVEAU 3 : Fuzzy match (si disponible)
        # ========================================
        if score == 0 and FUZZY_AVAILABLE:
            best_fuzzy_score = 0
            
            for variant in profil['title_variants']:
                variant_norm = normalize_text_ultimate(variant)
                if not variant_norm:
                    continue
                
                # Similarité partielle (gère sous-chaînes)
                similarity = fuzz.partial_ratio(variant_norm, title_norm)
                
                if similarity > best_fuzzy_score:
                    best_fuzzy_score = similarity
            
            # Si similarité >= 85% → score 6
            if best_fuzzy_score >= 85:
                score = 6
            # Si similarité >= 75% → score 4
            elif best_fuzzy_score >= 75:
                score = 4
        
        # ========================================
        # NIVEAU 4 : Keywords bonus
        # ========================================
        keywords_count = 0
        for keyword in profil['keywords_title']:
            keyword_norm = normalize_text_ultimate(keyword)
            if keyword_norm and keyword_norm in title_norm:
                keywords_count += 1
                if keywords_count <= 3:
                    score += 2
        
        return min(score, 10)
    
    def score_description(self, text_sklearn, profil_name):
        """Score description via TF-IDF"""
        if not text_sklearn or not self.tfidf_vectorizer:
            return 0
        
        desc_vector = self.tfidf_vectorizer.transform([text_sklearn.lower()])
        profil_vector = self.profil_vectors[profil_name]
        similarity = cosine_similarity(desc_vector, profil_vector)[0][0]
        
        return similarity * 10
    
    def score_competences(self, competences_found, profil_name):
        """Score compétences"""
        profil = self.profils[profil_name]
        
        if not competences_found:
            return 0
        
        comp_found_lower = [c.lower() for c in competences_found]
        
        # Core
        comp_core_lower = [c.lower() for c in profil['competences_core']]
        matches_core = len(set(comp_found_lower) & set(comp_core_lower))
        coverage_core = matches_core / len(comp_core_lower) if comp_core_lower else 0
        
        # Tech
        comp_tech_lower = [c.lower() for c in profil['competences_tech']]
        matches_tech = len(set(comp_found_lower) & set(comp_tech_lower))
        coverage_tech = matches_tech / len(comp_tech_lower) if comp_tech_lower else 0
        
        score = (coverage_core * 0.7 + coverage_tech * 0.3) * 10
        
        return score
    
    def classify_offer_with_threshold(self, row, min_score_threshold):
        """
        Classifie offre avec seuil donné
        
        CRITIQUE: Teste profil fourre-tout EN DERNIER
        pour éviter capture excessive
        """
        title = row.get('title', '')
        text_sklearn = row.get('text_for_sklearn', '')
        competences = row.get('competences_found', [])
        
        scores = {}
        details = {}
        
        
        # Séparer profil fourre-tout
        catch_all_profil = 'Data/IA - Non spécifié'
        
        # Profils spécifiques à tester en priorité
        specific_profils = [p for p in self.profil_names if p != catch_all_profil]
        
        # Profil fourre-tout à tester en dernier
        all_profils_ordered = specific_profils + ([catch_all_profil] if catch_all_profil in self.profil_names else [])
        
        # Calculer scores pour tous les profils (dans le bon ordre)
        for profil_name in all_profils_ordered:
            profil = self.profils[profil_name]
            weights = profil['weights']
            
            # 3 composantes
            score_t = self.score_title_ultimate(title, profil_name)
            score_d = self.score_description(text_sklearn, profil_name)
            score_c = self.score_competences(competences, profil_name)
            
            # Score final pondéré
            score_final = (
                score_t * weights['title'] +
                score_d * weights['description'] +
                score_c * weights['competences']
            )
            
            scores[profil_name] = score_final
            details[profil_name] = {
                'score': score_final,
                'score_title': score_t,
                'score_description': score_d,
                'score_competences': score_c
            }
        
        # Trier par score (les profils spécifiques auront priorité si ex-aequo)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        top_profil, top_score = sorted_scores[0]
        second_profil, second_score = sorted_scores[1] if len(sorted_scores) > 1 else (None, 0)
        
        # Confiance
        if top_score > 0 and second_score > 0:
            confidence = top_score / (top_score + second_score)
        else:
            confidence = 1.0 if top_score >= min_score_threshold else 0.5
        
        # Classification avec seuil donné
        if top_score >= min_score_threshold and confidence >= 0.55:
            assigned_profil = top_profil
            status = 'classified'
        else:
            assigned_profil = 'Non classifié'
            status = 'unclassified'
        
        return {
            'profil_assigned': assigned_profil,
            'profil_score': top_score,
            'profil_confidence': confidence,
            'profil_second': second_profil,
            'profil_second_score': second_score,
            'status': status,
            'score_title': details[top_profil]['score_title'],
            'score_description': details[top_profil]['score_description'],
            'score_competences': details[top_profil]['score_competences']
        }
    
    def classify_all_cascade(self, df):
        """
        Classification CASCADE (4 passes)
        
        PASSE 1: Seuil 4.5 (haute confiance)
        PASSE 2: Seuil 3.5 (confiance moyenne)
        PASSE 3: Seuil 2.5 (confiance faible)
        PASSE 4: Seuil 0.5 (fourre-tout - capture reste Data/IA)
        """
        print("\n  Classification CASCADE (4 passes)...")
        
        from tqdm import tqdm
        
        # Initialiser
        df['profil_assigned'] = 'Non classifié'
        df['profil_score'] = 0.0
        df['profil_confidence'] = 0.0
        df['profil_second'] = None
        df['profil_second_score'] = 0.0
        df['status'] = 'unclassified'
        df['score_title'] = 0.0
        df['score_description'] = 0.0
        df['score_competences'] = 0.0
        df['cascade_pass'] = 0
        
        # ========================================
        # PASSE 1 : Seuil 4.5 (haute confiance)
        # ========================================
        print("\n   PASSE 1/4: Seuil 4.5 (haute confiance)...")
        
        unclassified_mask = df['status'] == 'unclassified'
        
        for idx in tqdm(df[unclassified_mask].index, desc="Passe 1"):
            row = df.loc[idx]
            result = self.classify_offer_with_threshold(row, min_score_threshold=4.5)
            
            if result['status'] == 'classified':
                for key, value in result.items():
                    df.at[idx, key] = value
                df.at[idx, 'cascade_pass'] = 1
        
        n_pass1 = (df['cascade_pass'] == 1).sum()
        print(f"      Classifiées PASSE 1: {n_pass1}")
        
        # ========================================
        # PASSE 2 : Seuil 3.5 (confiance moyenne)
        # ========================================
        print("\n   PASSE 2/4: Seuil 3.5 (confiance moyenne)...")
        
        unclassified_mask = df['status'] == 'unclassified'
        
        for idx in tqdm(df[unclassified_mask].index, desc="Passe 2"):
            row = df.loc[idx]
            result = self.classify_offer_with_threshold(row, min_score_threshold=3.5)
            
            if result['status'] == 'classified':
                for key, value in result.items():
                    df.at[idx, key] = value
                df.at[idx, 'cascade_pass'] = 2
        
        n_pass2 = (df['cascade_pass'] == 2).sum()
        print(f"      Classifiées PASSE 2: {n_pass2}")
        
        # ========================================
        # PASSE 3 : Seuil 2.5 (confiance faible)
        # ========================================
        print("\n   PASSE 3/4: Seuil 2.5 (confiance faible)...")
        
        unclassified_mask = df['status'] == 'unclassified'
        
        for idx in tqdm(df[unclassified_mask].index, desc="Passe 3"):
            row = df.loc[idx]
            result = self.classify_offer_with_threshold(row, min_score_threshold=2.5)
            
            if result['status'] == 'classified':
                for key, value in result.items():
                    df.at[idx, key] = value
                df.at[idx, 'cascade_pass'] = 3
        
        n_pass3 = (df['cascade_pass'] == 3).sum()
        print(f"      Classifiées PASSE 3: {n_pass3}")
        
        # ========================================
        # PASSE 4 : Seuil 0.5 (fourre-tout)
        # ========================================
        print("\n   PASSE 4/4: Seuil 0.5 (fourre-tout Data/IA)...")
        
        unclassified_mask = df['status'] == 'unclassified'
        
        for idx in tqdm(df[unclassified_mask].index, desc="Passe 4"):
            row = df.loc[idx]
            result = self.classify_offer_with_threshold(row, min_score_threshold=0.5)
            
            if result['status'] == 'classified':
                for key, value in result.items():
                    df.at[idx, key] = value
                df.at[idx, 'cascade_pass'] = 4
        
        n_pass4 = (df['cascade_pass'] == 4).sum()
        print(f"      Classifiées PASSE 4: {n_pass4}")
        
        # ========================================
        # RÉSUMÉ CASCADE
        # ========================================
        print(f"\n    RÉSUMÉ CASCADE:")
        print(f"      PASSE 1 (4.5): {n_pass1:4d}")
        print(f"      PASSE 2 (3.5): {n_pass2:4d}")
        print(f"      PASSE 3 (2.5): {n_pass3:4d}")
        print(f"      PASSE 4 (0.5): {n_pass4:4d}")
        print(f"      ────────────────────")
        print(f"      TOTAL:         {n_pass1 + n_pass2 + n_pass3 + n_pass4:4d}")
        
        return df


# ============================================
# STATISTIQUES
# ============================================

def compute_statistics(df):
    """Calcule statistiques"""
    print("\n Calcul statistiques...")
    
    stats = {}
    
    profil_counts = df['profil_assigned'].value_counts()
    stats['distribution'] = profil_counts.to_dict()
    
    n_classified = (df['status'] == 'classified').sum()
    stats['taux_classification'] = n_classified / len(df)
    
    stats['confiance_moyenne'] = df[df['status'] == 'classified']['profil_confidence'].mean()
    stats['score_moyen'] = df[df['status'] == 'classified']['profil_score'].mean()
    
    # Par passe ( Convertir en int Python pour JSON)
    stats['by_pass'] = {
        'pass1': int((df['cascade_pass'] == 1).sum()),
        'pass2': int((df['cascade_pass'] == 2).sum()),
        'pass3': int((df['cascade_pass'] == 3).sum()),
        'pass4': int((df['cascade_pass'] == 4).sum())
    }
    
    # Par profil
    stats['by_profil'] = {}
    
    for profil in get_all_profils():
        df_profil = df[df['profil_assigned'] == profil]
        
        if len(df_profil) > 0:
            stats['by_profil'][profil] = {
                'count': int(len(df_profil)),
                'percentage': float(len(df_profil) / len(df) * 100),
                'score_moyen': float(df_profil['profil_score'].mean()),
                'confiance_moyenne': float(df_profil['profil_confidence'].mean()),
                'score_title_moyen': float(df_profil['score_title'].mean()),
                'salaire_median': float(df_profil['salary_annual'].median()) if df_profil['salary_annual'].notna().any() else None
            }
    
    return stats


def analyze_by_region(df):
    """Analyse par région"""
    print("\n  Analyse par région...")
    
    regions_stats = {}
    top_regions = df['region'].value_counts().head(10).index
    
    for region in top_regions:
        df_region = df[df['region'] == region]
        df_region_classified = df_region[df_region['status'] == 'classified']
        
        if len(df_region_classified) > 0:
            profil_counts = df_region_classified['profil_assigned'].value_counts()
            
            regions_stats[region] = {
                'total_offres': int(len(df_region)),
                'total_classified': int(len(df_region_classified)),
                'profils': profil_counts.head(5).to_dict(),
                'profil_dominant': profil_counts.index[0] if len(profil_counts) > 0 else None
            }
    
    return regions_stats


def analyze_by_source(df):
    """Analyse par source"""
    print("\n Analyse par source...")
    
    sources_stats = {}
    
    for source in df['source_name'].unique():
        df_source = df[df['source_name'] == source]
        df_source_classified = df_source[df_source['status'] == 'classified']
        
        if len(df_source_classified) > 0:
            profil_counts = df_source_classified['profil_assigned'].value_counts()
            
            sources_stats[source] = {
                'total_offres': int(len(df_source)),
                'total_classified': int(len(df_source_classified)),
                'profils': profil_counts.to_dict(),
                'profil_dominant': profil_counts.index[0] if len(profil_counts) > 0 else None
            }
    
    return sources_stats


def analyze_competences_by_profil(df):
    """Top compétences par profil"""
    print("\n🎓 Analyse compétences par profil...")
    
    comp_stats = {}
    
    for profil in get_all_profils():
        df_profil = df[df['profil_assigned'] == profil]
        
        if len(df_profil) > 0:
            all_comps = [comp for comps in df_profil['competences_found'] for comp in comps]
            comp_counter = Counter(all_comps)
            
            comp_stats[profil] = {
                'top_10': [
                    {'competence': c, 'count': int(n), 'percentage': n/len(df_profil)*100}
                    for c, n in comp_counter.most_common(10)
                ]
            }
    
    return comp_stats


# ============================================
# MAIN
# ============================================

def main():
    """Pipeline classification ULTIME"""
    print("="*70)
    print(" CLASSIFICATION ULTIME - OBJECTIF 80%+")
    print("="*70)
    
    saver = ResultSaver()
    
    # ==========================================
    # 1. CHARGEMENT
    # ==========================================
    print("\n📥 Chargement data_clean.pkl...")
    
    with open('../resultats_nlp/models/data_clean.pkl', 'rb') as f:
        df = pickle.load(f)
    
    print(f"    Offres: {len(df)}")
    print(f"    Avec compétences: {(df['num_competences'] > 0).sum()}")
    print(f"    Avec titre: {df['title'].notna().sum()}")
    
    # ==========================================
    # 2. CLASSIFICATION ULTIME
    # ==========================================
    print("\n Initialisation système ULTIME...")
    print("    Normalisation accents (é→e, è→e, etc.)")
    if FUZZY_AVAILABLE:
        print("    Fuzzy matching 85%+ (variations/typos)")
    else:
        print("     Fuzzy matching désactivé (fuzzywuzzy non installé)")
    print("    Cascade 4 passes (seuils 4.5 / 3.5 / 2.5 / 0.5)")
    
    classifier = ProfileClassifierUltimate()
    
    # Entraîner TF-IDF
    classifier.fit_tfidf(df)
    
    # Classifier avec cascade
    df = classifier.classify_all_cascade(df)
    
    print("\n Classification ULTIME terminée !")
    
    # ==========================================
    # 3. STATISTIQUES
    # ==========================================
    stats = compute_statistics(df)
    
    print(f"\n Résultats classification ULTIME:")
    print(f"   Taux classification: {stats['taux_classification']*100:.1f}%  🎯")
    print(f"   Confiance moyenne: {stats['confiance_moyenne']:.2f}")
    print(f"   Score moyen: {stats['score_moyen']:.2f}/10")
    
    print(f"\n Distribution profils:")
    for profil, count in sorted(stats['distribution'].items(), key=lambda x: x[1], reverse=True):
        pct = count / len(df) * 100
        print(f"   {profil:<30s}: {count:4d} ({pct:5.1f}%)")
    
    # ==========================================
    # 4. ANALYSES DÉTAILLÉES
    # ==========================================
    regions_stats = analyze_by_region(df)
    sources_stats = analyze_by_source(df)
    comp_stats = analyze_competences_by_profil(df)
    
    # ==========================================
    # 5. SAUVEGARDE
    # ==========================================
    print("\n Sauvegarde résultats...")
    
    saver.save_pickle(df, 'data_with_profiles.pkl')
    saver.save_pickle(classifier, 'classification_system.pkl')
    saver.save_json(stats, 'profils_distribution.json')
    saver.save_json(regions_stats, 'profils_by_region.json')
    saver.save_json(sources_stats, 'profils_by_source.json')
    saver.save_json(comp_stats, 'profils_competences.json')
    
    quality = {
        'version': 'ultimate_cascade_fuzzy',
        'taux_classification': float(stats['taux_classification']),
        'confiance_moyenne': float(stats['confiance_moyenne']),
        'score_moyen': float(stats['score_moyen']),
        'cascade_passes': stats['by_pass'],
        'fuzzy_enabled': FUZZY_AVAILABLE,
        'distribution_equilibree': max(stats['distribution'].values()) / len(df) < 0.35,
        'nb_profils_actifs': len([p for p, c in stats['distribution'].items() if c > 0 and p != 'Non classifié']),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    saver.save_json(quality, 'classification_quality.json')
    
    print("\n CLASSIFICATION ULTIME TERMINÉE !")
    
    print(f"\n Fichiers créés:")
    print(f"   - data_with_profiles.pkl")
    print(f"   - classification_system.pkl")
    print(f"   - profils_distribution.json")
    print(f"   - profils_by_region.json")
    print(f"   - profils_by_source.json")
    print(f"   - profils_competences.json")
    print(f"   - classification_quality.json")
    

    if stats['taux_classification'] >= 0.80:
        print("\n OBJECTIF 80%:  ATTEINT !")
    else:
        print(f"\n OBJECTIF 80%: {stats['taux_classification']*100:.1f}% (proche !)")    
        return df, classifier, stats


if __name__ == "__main__":
    df, classifier, stats = main()