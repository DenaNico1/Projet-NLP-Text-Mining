"""
Classification Hybride 3 Couches pour Profils Data/IA
Système robuste et évolutif pour nouvelles données

Auteur: Projet NLP Text Mining - Master SISE
Date: Décembre 2025
"""

import re
import pickle
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class HybridProfileClassifier:
    """
    Classificateur hybride en 3 couches :
    1. Titre (règles regex) - 70%
    2. Compétences (signatures) - 16%
    3. LDA fallback (modèle figé) - 14%
    """
    
    def __init__(self, lda_model_path=None, config_path=None):
        """
        Initialise le classificateur
        
        Args:
            lda_model_path: Chemin vers modèle LDA figé (optionnel)
            config_path: Chemin vers config JSON (optionnel)
        """
        self.lda_model = None
        self.lda_vectorizer = None
        
        # Charger modèle LDA si disponible
        if lda_model_path and Path(lda_model_path).exists():
            with open(lda_model_path, 'rb') as f:
                self.lda_model = pickle.load(f)
        
        # Charger config personnalisée ou utiliser par défaut
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.REGEX_PROFILS = config.get('regex_profils', self._default_regex_profils())
                self.SIGNATURES_COMPETENCES = config.get('signatures_competences', self._default_signatures())
                self.TOPIC_TO_PROFIL = config.get('topic_to_profil', self._default_topic_mapping())
        else:
            self.REGEX_PROFILS = self._default_regex_profils()
            self.SIGNATURES_COMPETENCES = self._default_signatures()
            self.TOPIC_TO_PROFIL = self._default_topic_mapping()
    
    
    def _default_regex_profils(self) -> Dict[str, List[str]]:
        """Patterns regex par profil (COUCHE 1)"""
        return {
            "Data Scientist": [
                r"data scientist",
                r"scientifique.*donn[ée]es",
                r"scientist.*data",
                r"datascientist"
            ],
            "ML Engineer": [
                r"machine learning engineer",
                r"ml engineer",
                r"ing[ée]nieur.*machine learning",
                r"ing[ée]nieur ml",
                r"machine learning.*ing[ée]nieur"
            ],
            "Data Engineer": [
                r"data engineer",
                r"ing[ée]nieur.*donn[ée]es",
                r"ing[ée]nieur data",
                r"engineer.*data"
            ],
            "MLOps Engineer": [
                r"mlops",
                r"ml.*ops",
                r"machine learning.*ops",
                r"devops.*ml"
            ],
            "Deep Learning Engineer": [
                r"deep learning",
                r"ing[ée]nieur.*deep learning",
                r"dl engineer"
            ],
            "NLP Engineer": [
                r"nlp engineer",
                r"natural language",
                r"traitement.*langage",
                r"ing[ée]nieur.*nlp",
                r"nlp.*ing[ée]nieur"
            ],
            "Computer Vision Engineer": [
                r"computer vision",
                r"vision.*ordinateur",
                r"cv engineer",
                r"image.*processing"
            ],
            "Data Analyst": [
                r"data analyst",
                r"analyste.*donn[ée]es",
                r"analyst.*data",
                r"analyste data"
            ],
            "BI Analyst": [
                r"business intelligence",
                r"bi analyst",
                r"analyste.*bi\b",
                r"analyste.*d[ée]cisionnel"
            ],
            "Analytics Engineer": [
                r"analytics engineer",
                r"ing[ée]nieur.*analytics",
                r"analytics.*ing[ée]nieur"
            ],
            "Big Data Engineer": [
                r"big data",
                r"ing[ée]nieur.*big data",
                r"hadoop.*engineer",
                r"spark.*engineer"
            ],
            "Research Scientist": [
                r"research scientist",
                r"chercheur",
                r"scientist.*research",
                r"ai.*researcher"
            ],
            "Quantitative Analyst": [
                r"quantitative analyst",
                r"quant\b",
                r"analyste.*quantitatif"
            ],
            "Data Architect": [
                r"data architect",
                r"architecte.*donn[ée]es",
                r"architecte data"
            ]
        }
    
    
    def _default_signatures(self) -> Dict[str, Dict]:
        """Signatures de compétences par profil (COUCHE 2)"""
        return {
            "Data Scientist": {
                "must_have": ["python", "machine learning", "scikit-learn", "pandas"],
                "strong_indicators": ["jupyter", "statistiques", "numpy", "matplotlib", "r", "deep learning"],
                "threshold": 0.3
            },
            "ML Engineer": {
                "must_have": ["python", "machine learning", "tensorflow", "pytorch", "scikit-learn"],
                "strong_indicators": ["keras", "xgboost", "lightgbm", "feature engineering", "model deployment"],
                "threshold": 0.35
            },
            "MLOps Engineer": {
                "must_have": ["kubernetes", "docker", "mlflow", "ci/cd"],
                "strong_indicators": ["kubeflow", "airflow", "terraform", "jenkins", "gitlab", "aws", "azure"],
                "threshold": 0.4
            },
            "Deep Learning Engineer": {
                "must_have": ["pytorch", "tensorflow", "deep learning", "neural networks"],
                "strong_indicators": ["cnn", "rnn", "lstm", "gpt", "transformers", "cuda", "gpu"],
                "threshold": 0.4
            },
            "NLP Engineer": {
                "must_have": ["nlp", "transformers", "bert", "spacy", "hugging face"],
                "strong_indicators": ["langchain", "openai", "gpt", "llm", "embedding", "sentiment analysis"],
                "threshold": 0.35
            },
            "Computer Vision Engineer": {
                "must_have": ["computer vision", "opencv", "yolo", "cnn"],
                "strong_indicators": ["segmentation", "detection", "image processing", "pytorch", "tensorflow"],
                "threshold": 0.4
            },
            "Data Engineer": {
                "must_have": ["sql", "python", "etl", "data pipeline"],
                "strong_indicators": ["spark", "airflow", "kafka", "databricks", "aws", "snowflake", "dbt"],
                "threshold": 0.3
            },
            "Big Data Engineer": {
                "must_have": ["spark", "hadoop", "kafka", "big data"],
                "strong_indicators": ["hive", "hbase", "presto", "flink", "scala", "mapreduce"],
                "threshold": 0.4
            },
            "Data Analyst": {
                "must_have": ["sql", "excel", "data analysis"],
                "strong_indicators": ["python", "pandas", "tableau", "power bi", "statistics"],
                "threshold": 0.25
            },
            "BI Analyst": {
                "must_have": ["power bi", "tableau", "sql", "qlik"],
                "strong_indicators": ["dax", "looker", "metabase", "reporting", "dashboard"],
                "threshold": 0.35
            },
            "Analytics Engineer": {
                "must_have": ["dbt", "sql", "data modeling"],
                "strong_indicators": ["looker", "airflow", "python", "snowflake", "bigquery", "git"],
                "threshold": 0.4
            }
        }
    
    
    def _default_topic_mapping(self) -> Dict[int, str]:
        """Mapping topics LDA vers profils (COUCHE 3 - FIGÉ)"""
        return {
            0: "Data Engineering",
            1: "ML Engineering",
            2: "Business Intelligence",
            3: "Deep Learning",
            4: "Data Analysis",
            5: "MLOps"
        }
    
    
    def classify_by_title(self, titre: str) -> Optional[str]:
        """
        COUCHE 1 : Classification par titre (règles regex)
        
        Args:
            titre: Titre de l'offre
            
        Returns:
            Profil détecté ou None
        """
        if not titre:
            return None
        
        titre_lower = titre.lower()
        
        for profil, patterns in self.REGEX_PROFILS.items():
            for pattern in patterns:
                if re.search(pattern, titre_lower):
                    return profil
        
        return None
    
    
    def classify_by_competences(self, competences: List[str]) -> Tuple[Optional[str], float]:
        """
        COUCHE 2 : Classification par compétences (signatures)
        
        Args:
            competences: Liste de compétences détectées
            
        Returns:
            (profil, score) ou (None, 0)
        """
        if not competences:
            return None, 0.0
        
        # Normaliser compétences
        competences_lower = [c.lower() for c in competences]
        
        best_profil = None
        best_score = 0.0
        
        for profil, signature in self.SIGNATURES_COMPETENCES.items():
            # Vérifier must-have (au moins 1 requis)
            has_must = any(
                any(must.lower() in comp for comp in competences_lower)
                for must in signature["must_have"]
            )
            
            if not has_must:
                continue  # Éliminatoire
            
            # Calculer score sur indicators
            score = 0
            for indicator in signature["strong_indicators"]:
                if any(indicator.lower() in comp for comp in competences_lower):
                    score += 1
            
            # Normaliser
            score_normalized = score / len(signature["strong_indicators"])
            
            # Vérifier threshold
            if score_normalized >= signature["threshold"]:
                if score_normalized > best_score:
                    best_score = score_normalized
                    best_profil = profil
        
        return best_profil, best_score
    
    
    def classify_by_lda(self, description: str, topic_dominant: int = None) -> str:
        """
        COUCHE 3 : Classification par LDA (fallback)
        
        Args:
            description: Description de l'offre
            topic_dominant: Topic déjà calculé (optionnel)
            
        Returns:
            Profil (toujours retourne quelque chose)
        """
        # Si topic déjà fourni
        if topic_dominant is not None:
            return self.TOPIC_TO_PROFIL.get(topic_dominant, "ML Engineering")
        
        # Si modèle LDA disponible
        if self.lda_model and self.lda_vectorizer and description:
            try:
                doc_vec = self.lda_vectorizer.transform([description])
                topic_dist = self.lda_model.transform(doc_vec)
                topic_dominant = topic_dist.argmax()
                return self.TOPIC_TO_PROFIL.get(topic_dominant, "ML Engineering")
            except:
                pass
        
        # Fallback ultime
        return "ML Engineering"
    
    
    def classify(self, titre: str, competences: List[str], description: str = "", 
                 topic_dominant: int = None) -> Dict:
        """
        Classification complète en 3 couches (cascade)
        
        Args:
            titre: Titre de l'offre
            competences: Liste de compétences
            description: Description (pour LDA)
            topic_dominant: Topic LDA pré-calculé (optionnel)
            
        Returns:
            dict avec profil, méthode, score
        """
        # COUCHE 1 : Titre
        profil = self.classify_by_title(titre)
        if profil:
            return {
                'profil': profil,
                'methode': 'titre',
                'score': 1.0,
                'confiance': 'haute'
            }
        
        # COUCHE 2 : Compétences
        profil, score = self.classify_by_competences(competences)
        if profil:
            confiance = 'haute' if score >= 0.6 else 'moyenne'
            return {
                'profil': profil,
                'methode': 'competences',
                'score': score,
                'confiance': confiance
            }
        
        # COUCHE 3 : LDA Fallback
        profil = self.classify_by_lda(description, topic_dominant)
        return {
            'profil': profil,
            'methode': 'lda_fallback',
            'score': 0.5,  # Score par défaut
            'confiance': 'faible'
        }
    
    
    def export_config(self, output_path: str):
        """Exporter la configuration actuelle vers JSON"""
        config = {
            'regex_profils': self.REGEX_PROFILS,
            'signatures_competences': self.SIGNATURES_COMPETENCES,
            'topic_to_profil': self.TOPIC_TO_PROFIL,
            'version': '1.0',
            'date': '2024-12-27'
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    
    def get_stats(self, df) -> Dict:
        """Calculer statistiques de classification"""
        # Utiliser profil_hybrid si profil n'existe pas
        profil_col = 'profil_hybrid' if 'profil_hybrid' in df.columns else 'profil'
        
        stats = {
            'total': len(df),
            'par_methode': df['methode'].value_counts().to_dict() if 'methode' in df.columns else {},
            'par_profil': df[profil_col].value_counts().to_dict() if profil_col in df.columns else {},
            'par_confiance': df['confiance'].value_counts().to_dict() if 'confiance' in df.columns else {}
        }
        return stats


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def apply_hybrid_classification(df, classifier: HybridProfileClassifier) -> pd.DataFrame:
    """
    Appliquer la classification hybride sur un DataFrame
    
    Args:
        df: DataFrame avec colonnes titre, competences_found, description, topic_dominant
        classifier: Instance de HybridProfileClassifier
        
    Returns:
        DataFrame avec colonnes profil, methode, score, confiance
    """
    
    results = []
    
    for idx, row in df.iterrows():
        result = classifier.classify(
            titre=row.get('titre', row.get('title', '')),
            competences=row.get('competences_found', []),
            description=row.get('description_clean', row.get('description', '')),
            topic_dominant=row.get('topic_dominant', None)
        )
        results.append(result)
    
    # Ajouter colonnes au DataFrame
    df['profil_hybrid'] = [r['profil'] for r in results]
    df['methode'] = [r['methode'] for r in results]
    df['score_classification'] = [r['score'] for r in results]
    df['confiance'] = [r['confiance'] for r in results]
    
    return df


def detect_emerging_profiles(df, min_occurrences=10) -> Dict:
    """
    Détecter profils émergents dans offres classées en fallback
    
    Args:
        df: DataFrame
        min_occurrences: Nombre minimal d'occurrences
        
    Returns:
        dict avec titres fréquents non classés
    """
    # Filtrer offres en fallback
    if 'methode' not in df.columns:
        return {
            'total_fallback': 0,
            'pct_fallback': 0,
            'emerging_titles': {}
        }
    
    fallback = df[df['methode'] == 'lda_fallback']
    
    # Analyser titres
    titre_col = 'titre' if 'titre' in fallback.columns else 'title'
    
    if titre_col not in fallback.columns:
        return {
            'total_fallback': len(fallback),
            'pct_fallback': len(fallback) / len(df) * 100 if len(df) > 0 else 0,
            'emerging_titles': {}
        }
    
    titres_freq = fallback[titre_col].value_counts()
    
    # Filtrer par seuil
    emerging = titres_freq[titres_freq >= min_occurrences].to_dict()
    
    return {
        'total_fallback': len(fallback),
        'pct_fallback': len(fallback) / len(df) * 100 if len(df) > 0 else 0,
        'emerging_titles': emerging
    }


if __name__ == "__main__":
    # Test
    classifier = HybridProfileClassifier()
    
    # Test 1 : Titre
    result = classifier.classify(
        titre="Data Scientist Senior",
        competences=["python", "scikit-learn"],
        description="Nous cherchons un data scientist..."
    )
    print("Test 1 (Titre):", result)
    
    # Test 2 : Compétences
    result = classifier.classify(
        titre="Ingénieur IA",  # Ambigu
        competences=["nlp", "bert", "transformers", "hugging face", "langchain"],
        description=""
    )
    print("Test 2 (Compétences):", result)
    
    # Test 3 : Fallback
    result = classifier.classify(
        titre="Développeur",  # Très ambigu
        competences=["java"],  # Pas data
        description="",
        topic_dominant=1
    )
    print("Test 3 (Fallback):", result)
    
    # Exporter config
    classifier.export_config("hybrid_classifier_config_v1.json")
    print("\n✅ Config exportée vers hybrid_classifier_config_v1.json")