"""
Utilitaires de Recherche et Matching
Fonctions pour la recherche par profil et recommandation CV

Auteur: Projet NLP Text Mining - Master SISE
Date: Décembre 2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re


def jaccard_similarity(set1, set2):
    """
    Calcule la similarité de Jaccard entre deux ensembles
    
    Args:
        set1 (set): Premier ensemble
        set2 (set): Deuxième ensemble
    
    Returns:
        float: Score de similarité (0-1)
    """
    if not set1 or not set2:
        return 0.0
    
    intersection = set1 & set2
    union = set1 | set2
    
    return len(intersection) / len(union) if union else 0.0


def compute_match_score(offre_competences, user_competences, method='jaccard'):
    """
    Calcule le score de correspondance entre offre et compétences utilisateur
    
    Args:
        offre_competences (list): Compétences de l'offre
        user_competences (list): Compétences de l'utilisateur
        method (str): Méthode ('jaccard', 'dice', 'overlap')
    
    Returns:
        float: Score de correspondance (0-1)
    """
    set_offre = set(offre_competences)
    set_user = set(user_competences)
    
    if method == 'jaccard':
        return jaccard_similarity(set_offre, set_user)
    
    elif method == 'dice':
        # Coefficient de Dice (Sørensen–Dice)
        intersection = set_offre & set_user
        return 2 * len(intersection) / (len(set_offre) + len(set_user)) if (set_offre or set_user) else 0.0
    
    elif method == 'overlap':
        # Overlap coefficient
        intersection = set_offre & set_user
        return len(intersection) / min(len(set_offre), len(set_user)) if (set_offre and set_user) else 0.0
    
    else:
        raise ValueError(f"Méthode inconnue : {method}")


def search_by_profile(df, profil, competences_required=None, region=None, top_k=50):
    """
    Recherche des offres par profil, compétences et région
    
    Args:
        df (DataFrame): DataFrame des offres
        profil (str): Profil métier recherché
        competences_required (list): Compétences requises (optionnel)
        region (str): Région (optionnel)
        top_k (int): Nombre de résultats max
    
    Returns:
        DataFrame: Offres matchées avec score
    """
    # Filtrer par profil
    df_filtered = df[df['profil'] == profil].copy()
    
    # Filtrer par région si spécifié
    if region:
        df_filtered = df_filtered[df_filtered['region'] == region]
    
    # Si pas de compétences spécifiées, retourner toutes les offres filtrées
    if not competences_required:
        df_filtered['match_score'] = 1.0
        return df_filtered.head(top_k)
    
    # Calculer score de matching pour chaque offre
    scores = []
    for comps in df_filtered['competences_found']:
        score = compute_match_score(comps, competences_required, method='jaccard')
        scores.append(score)
    
    df_filtered['match_score'] = scores
    
    # Trier par score décroissant
    df_filtered = df_filtered.sort_values('match_score', ascending=False)
    
    return df_filtered.head(top_k)


def get_regional_alerts(df, profil, region, competences_user, top_n=5):
    """
    Génère des alerts sur les compétences régionales spécifiques
    
    Args:
        df (DataFrame): DataFrame des offres
        profil (str): Profil métier
        region (str): Région
        competences_user (list): Compétences de l'utilisateur
        top_n (int): Nombre de suggestions
    
    Returns:
        list: Liste d'alerts
    """
    # Filtrer par profil et région
    df_region = df[(df['profil'] == profil) & (df['region'] == region)]
    df_national = df[df['profil'] == profil]
    
    if len(df_region) < 10:
        return ["⚠️ Données régionales insuffisantes pour analyse"]
    
    # Compter les compétences dans la région
    comps_region = [c for cs in df_region['competences_found'] for c in cs]
    counter_region = Counter(comps_region)
    freq_region = {comp: count/len(df_region) for comp, count in counter_region.items()}
    
    # Compter les compétences au national
    comps_national = [c for cs in df_national['competences_found'] for c in cs]
    counter_national = Counter(comps_national)
    freq_national = {comp: count/len(df_national) for comp, count in counter_national.items()}
    
    # Calculer le "lift" (sur-représentation régionale)
    lifts = {}
    for comp in freq_region:
        if comp in freq_national and freq_national[comp] > 0:
            lift = freq_region[comp] / freq_national[comp]
            if lift > 1.2 and comp not in competences_user:  # Sur-représenté ET pas déjà dans user
                lifts[comp] = (lift, freq_region[comp], freq_national[comp])
    
    # Trier par lift décroissant
    sorted_lifts = sorted(lifts.items(), key=lambda x: x[1][0], reverse=True)
    
    # Générer alerts
    alerts = []
    for comp, (lift, freq_reg, freq_nat) in sorted_lifts[:top_n]:
        alert = f"⚠️ En {region}, **{comp}** est demandé dans {freq_reg*100:.0f}% des offres {profil} (vs {freq_nat*100:.0f}% national, +{(lift-1)*100:.0f}%)"
        alerts.append(alert)
    
    if not alerts:
        alerts.append(f"✅ Vos compétences couvrent bien les spécificités de {region}")
    
    return alerts


def extract_competences_from_cv(cv_text, dict_competences):
    """
    Extrait les compétences d'un CV
    
    Args:
        cv_text (str): Texte du CV
        dict_competences (list): Liste des compétences à rechercher
    
    Returns:
        list: Compétences trouvées
    """
    if not cv_text or pd.isna(cv_text):
        return []
    
    cv_lower = cv_text.lower()
    found = []
    
    for comp in dict_competences:
        comp_lower = comp.lower()
        # Word boundary pour éviter faux positifs
        pattern = r'\b' + re.escape(comp_lower) + r'\b'
        
        if re.search(pattern, cv_lower):
            found.append(comp)
    
    return found


def recommend_offers_by_cv(df, cv_competences, embeddings_cv, embeddings_offres, top_k=10, method='hybrid'):
    """
    Recommande des offres en fonction du CV
    
    Args:
        df (DataFrame): DataFrame des offres
        cv_competences (list): Compétences extraites du CV
        embeddings_cv (np.array): Embedding du CV (si method='semantic' ou 'hybrid')
        embeddings_offres (np.array): Embeddings des offres
        top_k (int): Nombre de recommandations
        method (str): 'competences', 'semantic', 'hybrid'
    
    Returns:
        DataFrame: Offres recommandées avec scores
    """
    df_result = df.copy()
    
    if method == 'competences':
        # Matching basé uniquement sur compétences
        scores = []
        for comps in df['competences_found']:
            score = compute_match_score(comps, cv_competences, method='jaccard')
            scores.append(score)
        
        df_result['recommendation_score'] = scores
    
    elif method == 'semantic':
        # Matching basé uniquement sur similarité sémantique (embeddings)
        if embeddings_cv is None or embeddings_offres is None:
            raise ValueError("Embeddings requis pour méthode 'semantic'")
        
        # Similarité cosinus entre CV et toutes les offres
        similarities = cosine_similarity(embeddings_cv.reshape(1, -1), embeddings_offres)[0]
        df_result['recommendation_score'] = similarities
    
    elif method == 'hybrid':
        # Combinaison des deux approches (50/50)
        if embeddings_cv is None or embeddings_offres is None:
            # Fallback sur compétences uniquement
            return recommend_offers_by_cv(df, cv_competences, None, None, top_k, method='competences')
        
        # Score compétences
        comp_scores = []
        for comps in df['competences_found']:
            score = compute_match_score(comps, cv_competences, method='jaccard')
            comp_scores.append(score)
        
        # Score sémantique
        similarities = cosine_similarity(embeddings_cv.reshape(1, -1), embeddings_offres)[0]
        
        # Combinaison (moyenne pondérée)
        hybrid_scores = 0.5 * np.array(comp_scores) + 0.5 * np.array(similarities)
        df_result['recommendation_score'] = hybrid_scores
        df_result['competences_score'] = comp_scores
        df_result['semantic_score'] = similarities
    
    else:
        raise ValueError(f"Méthode inconnue : {method}")
    
    # Trier par score décroissant
    df_result = df_result.sort_values('recommendation_score', ascending=False)
    
    return df_result.head(top_k)


def compute_gap_analysis(cv_competences, profil, signature_by_profile):
    """
    Analyse les compétences manquantes pour un profil cible
    
    Args:
        cv_competences (list): Compétences du CV
        profil (str): Profil métier cible
        signature_by_profile (dict): Compétences signature par profil
    
    Returns:
        dict: Gap analysis avec compétences manquantes
    """
    if profil not in signature_by_profile:
        return {
            'missing_competences': [],
            'message': f"Profil '{profil}' non trouvé"
        }
    
    # Compétences signature du profil
    signatures = signature_by_profile[profil]
    signature_comps = [s['competence'] for s in signatures]
    
    # Compétences manquantes
    missing = [comp for comp in signature_comps if comp not in cv_competences]
    
    # Compétences présentes
    present = [comp for comp in signature_comps if comp in cv_competences]
    
    # Calcul du taux de couverture
    coverage = len(present) / len(signature_comps) if signature_comps else 0
    
    return {
        'profil': profil,
        'total_signature': len(signature_comps),
        'competences_present': present,
        'competences_missing': missing,
        'coverage': coverage,
        'message': f"Vous maîtrisez {len(present)}/{len(signature_comps)} compétences signature du profil {profil} ({coverage*100:.0f}%)"
    }


def estimate_salary_impact(missing_competences, salary_by_competence):
    """
    Estime l'impact salarial des compétences manquantes
    
    Args:
        missing_competences (list): Compétences manquantes
        salary_by_competence (dict): Salaire médian par compétence
    
    Returns:
        dict: Estimation impact salarial
    """
    # Calculer le salaire médian des compétences manquantes
    salaries = []
    for comp in missing_competences:
        if comp in salary_by_competence:
            salaries.append(salary_by_competence[comp])
    
    if not salaries:
        return {
            'potential_increase': 0,
            'message': "Données salariales insuffisantes"
        }
    
    avg_salary_missing = np.median(salaries)
    
    # Estimation simplifiée : +5% par compétence manquante de haute valeur
    high_value_comps = [c for c in missing_competences if c in salary_by_competence and salary_by_competence[c] > 60000]
    
    potential_increase_pct = min(len(high_value_comps) * 5, 30)  # Max 30%
    
    return {
        'high_value_missing': high_value_comps,
        'potential_increase_pct': potential_increase_pct,
        'avg_salary_missing_comp': avg_salary_missing,
        'message': f"Acquérir ces compétences pourrait augmenter votre salaire de ~{potential_increase_pct}%"
    }