# nlp_pipeline_wrapper.py - VERSION FINALE CORRIGÉE
"""
Wrapper Pipeline NLP ULTRA-COMPLET
Imports corrigés + Gestion compétences adaptée
"""

import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np

# Ajouter chemins
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analyses_nlp' / 'fichiers_analyses'))

# ============================================
# IMPORTS PIPELINE EXISTANT (CORRIGÉ)
# ============================================

PIPELINE_AVAILABLE = False

try:
    # Utils
    from analyses_nlp.fichiers_analyses.utils import (
        TextPreprocessor,
        extract_competences_from_text
    )
    print("✅ Utils importés")
    
    # Profils
    from analyses_nlp.fichiers_analyses.profils_definitions_v1_optimized import PROFILS
    print("✅ Profils importés")
    
    # Classification (import du MODULE, pas de la classe directement)
    import importlib.util
    
    classif_path = ROOT / 'analyses_nlp' / 'fichiers_analyses' / '4_classification_hybride_ultimate.py'
    spec = importlib.util.spec_from_file_location("classification_module", classif_path)
    classification_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classification_module)
    
    ProfileClassifierUltimate = classification_module.ProfileClassifierUltimate
    normalize_text_ultimate = classification_module.normalize_text_ultimate
    remove_accents = classification_module.remove_accents
    
    print("✅ Classification importée")
    
    PIPELINE_AVAILABLE = True
    print("✅ Pipeline NLP COMPLET importé avec succès")
    
except Exception as e:
    print(f"⚠️ Import pipeline NLP échoué : {e}")
    print("→ Utilisation pipeline basique temporaire")
    import traceback
    traceback.print_exc()


# ============================================
# DICTIONNAIRE COMPÉTENCES
# ============================================

COMPETENCES_DICT = [
    'python', 'r', 'sql', 'java', 'scala', 'julia',
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 
    'scikit-learn', 'keras', 'xgboost', 'lightgbm',
    'spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'apis',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes',
    'power bi', 'tableau', 'looker', 'qlik',
    'postgresql', 'mysql', 'mongodb', 'cassandra', 'redis',
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
    'mlflow', 'kubeflow', 'mlops', 'ci/cd', 'streamlit', 'devops', 
    'databricks', 'snowflake',
    'nlp', 'nltk', 'spacy', 'transformers', 'bert', 'gpt',
    'langchain', 'llamaindex',
    'intelligence artificielle', 'large language models', 'llm', 
    'llms', 'rag', 'benchmarks',
    'git', 'linux', 'api', 'rest', 'graphql', 'agile', 'scrum'
]


# ============================================
# FONCTION WRAPPER PRINCIPALE
# ============================================

def process_single_offre(offre_id, title, description, conn):
    """Pipeline NLP COMPLET sur 1 offre"""
    
    print(f"🔍 Pipeline NLP pour offre #{offre_id}...")
    
    cur = conn.cursor()
    
    # ============================================
    # 1. PREPROCESSING
    # ============================================
    
    print("  📝 Preprocessing (TextPreprocessor)...")
    
    if PIPELINE_AVAILABLE:
        try:
            preprocessor = TextPreprocessor(language='french')
            description_clean = preprocessor.clean_text(description)
            tokens = preprocessor.preprocess(description_clean, lemmatize=True)
            text_for_sklearn = ' '.join(tokens)
            num_tokens = len(tokens)
            print(f"     ✅ Tokens: {num_tokens}, Stopwords filtrés, Lemmatisé")
        except Exception as e:
            print(f"     ⚠️ Erreur preprocessing: {e}")
            text_clean = description.lower()
            text_clean = re.sub(r'[^\w\s]', ' ', text_clean)
            tokens = text_clean.split()
            text_for_sklearn = text_clean
            description_clean = description
            num_tokens = len(tokens)
    else:
        text_clean = description.lower()
        text_clean = re.sub(r'[^\w\s]', ' ', text_clean)
        tokens = text_clean.split()
        text_for_sklearn = text_clean
        description_clean = description
        num_tokens = len(tokens)
    
    # Stocker preprocessing
    try:
        cur.execute("""
            INSERT INTO fact_preprocessing (offre_id, tokens, num_tokens)
            VALUES (%s, %s, %s)
            ON CONFLICT (offre_id) DO UPDATE 
            SET tokens = EXCLUDED.tokens, num_tokens = EXCLUDED.num_tokens
        """, (offre_id, tokens, num_tokens))
    except Exception as e:
        print(f"  ⚠️ Preprocessing storage: {e}")
    
    # ============================================
    # 2. EXTRACTION COMPÉTENCES
    # ============================================
    
    print("  🔧 Extraction compétences (pattern matching)...")
    
    if PIPELINE_AVAILABLE:
        try:
            text_full = f"{title} {description}"
            competences = extract_competences_from_text(text_full, COMPETENCES_DICT)
            print(f"     ✅ Compétences trouvées: {len(competences)}")
        except Exception as e:
            print(f"     ⚠️ Erreur extraction: {e}")
            competences = [comp for comp in COMPETENCES_DICT if comp in text_full.lower()]
    else:
        text_full = f"{title} {description}"
        competences = [comp for comp in COMPETENCES_DICT if comp in text_full.lower()]
    
    num_competences = len(competences)
    
    # Stocker compétences (ADAPTÉ À TA STRUCTURE)
    try:
        for comp in competences:
            skill_code = comp.upper().replace(' ', '_').replace('-', '_')
            cur.execute("""
                INSERT INTO fact_competences (offre_id, skill_label, skill_code, skill_level)
                VALUES (%s, %s, %s, NULL)
                ON CONFLICT (offre_id, skill_label) DO NOTHING
            """, (offre_id, comp, skill_code))
    except Exception as e:
        print(f"  ⚠️ Competences storage: {e}")
    
    # ============================================
    # 3. CLASSIFICATION
    # ============================================
    
    print("  🎯 Classification (ProfileClassifierUltimate)...")
    
    if PIPELINE_AVAILABLE:
        try:
            df_single = pd.DataFrame([{
                'offre_id': offre_id,
                'title': title,
                'description': description,
                'description_clean': description_clean,
                'tokens': tokens,
                'text_for_sklearn': text_for_sklearn,
                'competences_found': competences,
                'num_tokens': num_tokens,
                'num_competences': num_competences
            }])
            
            classifier = ProfileClassifierUltimate()
            classifier.fit_tfidf(df_single)
            
            result = classifier.classify_offer_with_threshold(
                df_single.iloc[0],
                min_score_threshold=2.5
            )
            
            profil = result['profil_assigned']
            score = result['profil_score']
            confidence = result['profil_confidence']
            profil_second = result.get('profil_second')
            profil_second_score = result.get('profil_second_score', 0)
            score_title = result.get('score_title', 0)
            score_description = result.get('score_description', 0)
            score_competences = result.get('score_competences', 0)
            status = result.get('status', 'classified')
            
            print(f"     ✅ Profil: {profil} (score: {score:.2f}, conf: {confidence:.2f})")
            
        except Exception as e:
            print(f"     ⚠️ Erreur classification: {e}")
            import traceback
            traceback.print_exc()
            profil, score, confidence = classify_fallback(title, competences)
            profil_second = None
            profil_second_score = 0
            score_title = 0
            score_description = 0
            score_competences = 0
            status = 'classified'
    else:
        profil, score, confidence = classify_fallback(title, competences)
        profil_second = None
        profil_second_score = 0
        score_title = 0
        score_description = 0
        score_competences = 0
        status = 'classified'
    
    # Stocker profil

    try:
        cur.execute("""
            INSERT INTO fact_profils_nlp (
                offre_id,
                profil_assigned,
                score_classification,
                num_tokens,
                num_competences,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (offre_id) DO UPDATE SET
                profil_assigned = EXCLUDED.profil_assigned,
                score_classification = EXCLUDED.score_classification,
                num_tokens = EXCLUDED.num_tokens,
                num_competences = EXCLUDED.num_competences,
                status = EXCLUDED.status
        """, (
            offre_id,
            result['profil'],
            result['score'],
            result.get('num_tokens', 0),
            result.get('num_competences', 0),
            'classified'
        ))
        
        print(f"     ✅ Profil stocké : {result['profil']} (score: {result['score']:.2f})")
        
    except Exception as e:
        print(f"  ⚠️ Profil storage: {e}")
    
    conn.commit()
    
    print(f"  ✅ Pipeline terminé : {profil} ({score:.2f}), {num_competences} compétences")
    
    return {
        'profil': profil,
        'score': score,
        'confidence': confidence,
        'profil_second': profil_second,
        'profil_second_score': profil_second_score,
        'competences': competences,
        'num_competences': num_competences,
        'num_tokens': num_tokens,
        'tokens': tokens,
        'description_clean': description_clean,
        'text_for_sklearn': text_for_sklearn,
        'score_title': score_title,
        'score_description': score_description,
        'score_competences': score_competences,
        'status': status
    }


def classify_fallback(title, competences):
    """Classification basique si pipeline non disponible"""
    title_lower = title.lower()
    
    if any(kw in title_lower for kw in ['data scientist', 'scientist']):
        return 'Data Scientist', 0.9, 0.85
    elif any(kw in title_lower for kw in ['data engineer', 'engineer data']):
        return 'Data Engineer', 0.9, 0.85
    elif any(kw in title_lower for kw in ['data analyst', 'analyst']):
        return 'Data Analyst', 0.9, 0.85
    elif any(kw in title_lower for kw in ['ml engineer', 'machine learning']):
        return 'ML Engineer', 0.85, 0.80
    else:
        if 'spark' in competences or 'airflow' in competences:
            return 'Data Engineer', 0.7, 0.65
        elif 'machine learning' in competences:
            return 'Data Scientist', 0.7, 0.65
        else:
            return 'Autre', 0.5, 0.50


def ensure_nlp_tables_exist(conn):
    """Crée tables NLP si nécessaire"""
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fact_preprocessing (
            offre_id INTEGER PRIMARY KEY,
            tokens TEXT[],
            num_tokens INTEGER,
            FOREIGN KEY (offre_id) REFERENCES fact_offres(offre_id) ON DELETE CASCADE
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fact_profils_nlp (
            offre_id INTEGER PRIMARY KEY,
            profil_assigned VARCHAR(50),
            score_classification FLOAT,
            profil_confidence FLOAT,
            profil_second VARCHAR(50),
            profil_second_score FLOAT,
            num_tokens INTEGER,
            num_competences INTEGER,
            status VARCHAR(20),
            score_title FLOAT,
            score_description FLOAT,
            score_competences FLOAT,
            cascade_pass INTEGER,
            created_at TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (offre_id) REFERENCES fact_offres(offre_id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    print("✅ Tables NLP vérifiées")


if __name__ == "__main__":
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from config_db import get_db_connection
    
    print("=" * 70)
    print("TEST PIPELINE NLP WRAPPER FINAL")
    print("=" * 70)
    
    conn = get_db_connection()
    if not conn:
        print("❌ Connexion impossible")
        exit(1)
    
    ensure_nlp_tables_exist(conn)
    
    test_data = {
        'offre_id': 9999,
        'title': 'Data Engineer Senior - Spark & AWS',
        'description': """
        Nous recherchons un Data Engineer expérimenté pour construire 
        notre data platform cloud. Compétences : Spark, Python, SQL, 
        Airflow, DBT, AWS, Docker, Kubernetes, PostgreSQL, MongoDB.
        """
    }
    
    try:
        results = process_single_offre(
            offre_id=test_data['offre_id'],
            title=test_data['title'],
            description=test_data['description'],
            conn=conn
        )
        
        print("\n" + "=" * 70)
        print("✅ TEST RÉUSSI")
        print("=" * 70)
        print(f"\n📊 RÉSULTATS:")
        print(f"  Profil : {results['profil']}")
        print(f"  Score : {results['score']:.2f}")
        print(f"  Confiance : {results['confidence']:.2f}")
        print(f"  Compétences : {results['num_competences']}")
        
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()

# # nlp_pipeline_wrapper.py
# """
# Wrapper Pipeline NLP ULTRA-COMPLET
# Réutilise EXACTEMENT le code du projet analyses_nlp

# Modules intégrés :
# 1. utils.py → TextPreprocessor, extract_competences_from_text
# 2. 1_preprocessing.py → Preprocessing complet
# 3. 2_extraction_competences.py → Extraction compétences
# 4. 4_classification_hybride_ultimate.py → Classification ProfileClassifierUltimate

# Pour nouvelle offre ajoutée via Streamlit → Application pipeline en temps réel
# """

# import sys
# from pathlib import Path
# import re
# import pandas as pd
# import numpy as np

# # Ajouter chemins
# ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(ROOT))
# sys.path.insert(0, str(ROOT / 'analyses_nlp' / 'fichiers_analyses'))

# # ============================================
# # IMPORTS PIPELINE EXISTANT
# # ============================================

# try:
#     # Utils (classes preprocessing + extraction)
#     from analyses_nlp.fichiers_analyses.utils import (
#         TextPreprocessor,
#         extract_competences_from_text
#     )
    
#     # Classification hybride ultimate (nom fichier avec "4_")
#     from analyses_nlp.fichiers_analyses.profils_definitions_v1_optimized import PROFILS
    
#     from analyses_nlp.fichiers_analyses import (
#         classification_hybride_ultimate as classification_module
#     )
    
#     ProfileClassifierUltimate = classification_module.ProfileClassifierUltimate
#     normalize_text_ultimate = classification_module.normalize_text_ultimate
#     remove_accents = classification_module.remove_accents
    
#     PIPELINE_AVAILABLE = True
#     print("✅ Pipeline NLP importé avec succès")
    
# except ImportError as e:
#     print(f"⚠️ Import pipeline NLP échoué : {e}")
#     print("→ Utilisation pipeline basique temporaire")
#     PIPELINE_AVAILABLE = False

# # ============================================
# # DICTIONNAIRE COMPÉTENCES (du preprocessing)
# # ============================================

# COMPETENCES_DICT = [
#     # Langages
#     'python', 'r', 'sql', 'java', 'scala', 'julia',
#     # ML/DL
#     'machine learning', 'deep learning', 'tensorflow', 'pytorch', 
#     'scikit-learn', 'keras', 'xgboost', 'lightgbm',
#     # Data Engineering
#     'spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'apis',
#     # Cloud
#     'aws', 'azure', 'gcp', 'docker', 'kubernetes',
#     # BI
#     'power bi', 'tableau', 'looker', 'qlik',
#     # Databases
#     'postgresql', 'mysql', 'mongodb', 'cassandra', 'redis',
#     # Libs Python
#     'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
#     # MLOps
#     'mlflow', 'kubeflow', 'mlops', 'ci/cd', 'streamlit', 'devops', 
#     'databricks', 'snowflake',
#     # NLP
#     'nlp', 'nltk', 'spacy', 'transformers', 'bert', 'gpt',
#     'langchain', 'llamaindex',
#     # IA/LLM
#     'intelligence artificielle', 'large language models', 'llm', 
#     'llms', 'rag', 'benchmarks',
#     # Other
#     'git', 'linux', 'api', 'rest', 'graphql', 'agile', 'scrum'
# ]


# # ============================================
# # FONCTION WRAPPER PRINCIPALE
# # ============================================

# def process_single_offre(offre_id, title, description, conn):
#     """
#     Pipeline NLP COMPLET sur 1 offre
    
#     Applique EXACTEMENT le même traitement que analyses_nlp :
#     1. Preprocessing (TextPreprocessor de utils.py)
#        - Nettoyage HTML
#        - Tokenisation
#        - Stopwords FR+EN
#        - Lemmatisation
    
#     2. Extraction compétences (extract_competences_from_text de utils.py)
#        - Pattern matching sur 60+ compétences
    
#     3. Classification (ProfileClassifierUltimate de 4_classification_hybride_ultimate.py)
#        - Scoring hybride (60% titre + 20% desc + 20% comp)
#        - Normalisation accents
#        - Fuzzy matching
#        - Cascade seuils
    
#     Args:
#         offre_id (int): ID offre dans fact_offres
#         title (str): Titre du poste
#         description (str): Description complète
#         conn: Connexion PostgreSQL
    
#     Returns:
#         dict: Résultats NLP complets
#     """
    
#     print(f"🔍 Pipeline NLP pour offre #{offre_id}...")
    
#     cur = conn.cursor()
    
#     # ============================================
#     # 1. PREPROCESSING (TON CODE EXACT)
#     # ============================================
    
#     print("  📝 Preprocessing (TextPreprocessor)...")
    
#     if PIPELINE_AVAILABLE:
#         try:
#             # Utiliser TON TextPreprocessor (identique à 1_preprocessing.py)
#             preprocessor = TextPreprocessor(language='french')
            
#             # 1.1 Nettoyage HTML (entités &nbsp;, balises)
#             description_clean = preprocessor.clean_text(description)
            
#             # 1.2 Tokenisation + Stopwords FR+EN + Lemmatisation
#             tokens = preprocessor.preprocess(description_clean, lemmatize=True)
            
#             # 1.3 Text for sklearn (rejoint tokens)
#             text_for_sklearn = ' '.join(tokens)
            
#             num_tokens = len(tokens)
            
#             print(f"     ✅ Tokens: {num_tokens}, Stopwords filtrés, Lemmatisé")
            
#         except Exception as e:
#             print(f"     ⚠️ Erreur preprocessing: {e}")
#             # Fallback
#             text_clean = description.lower()
#             text_clean = re.sub(r'[^\w\s]', ' ', text_clean)
#             tokens = text_clean.split()
#             text_for_sklearn = text_clean
#             description_clean = description
#             num_tokens = len(tokens)
#     else:
#         # Fallback basique
#         text_clean = description.lower()
#         text_clean = re.sub(r'[^\w\s]', ' ', text_clean)
#         tokens = text_clean.split()
#         text_for_sklearn = text_clean
#         description_clean = description
#         num_tokens = len(tokens)
    
#     # Stocker preprocessing
#     try:
#         cur.execute("""
#             INSERT INTO fact_preprocessing (offre_id, tokens, num_tokens)
#             VALUES (%s, %s, %s)
#             ON CONFLICT (offre_id) DO UPDATE 
#             SET tokens = EXCLUDED.tokens, num_tokens = EXCLUDED.num_tokens
#         """, (offre_id, tokens, num_tokens))
#     except Exception as e:
#         print(f"  ⚠️ Preprocessing storage: {e}")
    
#     # ============================================
#     # 2. EXTRACTION COMPÉTENCES (TON CODE EXACT)
#     # ============================================
    
#     print("  🔧 Extraction compétences (pattern matching)...")
    
#     if PIPELINE_AVAILABLE:
#         try:
#             # Utiliser TON extract_competences_from_text (identique à 1_preprocessing.py)
#             text_full = f"{title} {description}"
#             competences = extract_competences_from_text(text_full, COMPETENCES_DICT)
            
#             print(f"     ✅ Compétences trouvées: {len(competences)}")
            
#         except Exception as e:
#             print(f"     ⚠️ Erreur extraction: {e}")
#             # Fallback
#             competences = [comp for comp in COMPETENCES_DICT if comp in text_full.lower()]
#     else:
#         # Fallback basique
#         text_full = f"{title} {description}"
#         competences = [comp for comp in COMPETENCES_DICT if comp in text_full.lower()]
    
#     num_competences = len(competences)
    
#     # Stocker compétences
#     # Stocker compétences
#     try:
#         for comp in competences:
#             cur.execute("""
#                 INSERT INTO fact_competences (offre_id, skill_label, skill_code)
#                 VALUES (%s, %s, %s)
#                 ON CONFLICT (offre_id, skill_label) DO NOTHING
#             """, (offre_id, comp, comp.upper().replace(' ', '_')))
#     except Exception as e:
#         print(f"  ⚠️ Competences storage: {e}")
    
#     # ============================================
#     # 3. CLASSIFICATION (TON CODE EXACT)
#     # ============================================
    
#     print("  🎯 Classification (ProfileClassifierUltimate)...")
    
#     if PIPELINE_AVAILABLE:
#         try:
#             # Créer DataFrame minimal (structure identique à analyses_nlp)
#             df_single = pd.DataFrame([{
#                 'offre_id': offre_id,
#                 'title': title,
#                 'description': description,
#                 'description_clean': description_clean,
#                 'tokens': tokens,
#                 'text_for_sklearn': text_for_sklearn,
#                 'competences_found': competences,
#                 'num_tokens': num_tokens,
#                 'num_competences': num_competences
#             }])
            
#             # Utiliser TON ProfileClassifierUltimate (identique à 4_classification_hybride_ultimate.py)
#             classifier = ProfileClassifierUltimate()
            
#             # Entraîner TF-IDF (sur 1 offre + profils)
#             classifier.fit_tfidf(df_single)
            
#             # Classifier avec seuil adaptatif
#             # Utilise scoring hybride : 60% titre + 20% desc + 20% comp
#             result = classifier.classify_offer_with_threshold(
#                 df_single.iloc[0],
#                 min_score_threshold=2.5  # Seuil modéré pour 1 offre
#             )
            
#             profil = result['profil_assigned']
#             score = result['profil_score']
#             confidence = result['profil_confidence']
#             profil_second = result.get('profil_second')
#             profil_second_score = result.get('profil_second_score', 0)
            
#             score_title = result.get('score_title', 0)
#             score_description = result.get('score_description', 0)
#             score_competences = result.get('score_competences', 0)
            
#             status = result.get('status', 'classified')
            
#             print(f"     ✅ Profil: {profil} (score: {score:.2f}, conf: {confidence:.2f})")
#             print(f"        Détail scores - Titre: {score_title:.1f}, Desc: {score_description:.1f}, Comp: {score_competences:.1f}")
            
#         except Exception as e:
#             print(f"     ⚠️ Erreur classification: {e}")
#             import traceback
#             traceback.print_exc()
            
#             # Fallback classification simple
#             profil, score, confidence = classify_fallback(title, competences)
#             profil_second = None
#             profil_second_score = 0
#             score_title = 0
#             score_description = 0
#             score_competences = 0
#             status = 'classified'
#     else:
#         # Fallback classification simple
#         profil, score, confidence = classify_fallback(title, competences)
#         profil_second = None
#         profil_second_score = 0
#         score_title = 0
#         score_description = 0
#         score_competences = 0
#         status = 'classified'
    
#     # Stocker profil
#     try:
#         cur.execute("""
#             INSERT INTO fact_profils_nlp (
#                 offre_id, 
#                 profil_assigned, 
#                 score_classification,
#                 profil_confidence,
#                 profil_second,
#                 profil_second_score,
#                 num_tokens, 
#                 num_competences, 
#                 status,
#                 score_title,
#                 score_description,
#                 score_competences
#             )
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#             ON CONFLICT (offre_id) DO UPDATE SET
#                 profil_assigned = EXCLUDED.profil_assigned,
#                 score_classification = EXCLUDED.score_classification,
#                 profil_confidence = EXCLUDED.profil_confidence,
#                 profil_second = EXCLUDED.profil_second,
#                 profil_second_score = EXCLUDED.profil_second_score,
#                 num_tokens = EXCLUDED.num_tokens,
#                 num_competences = EXCLUDED.num_competences,
#                 status = EXCLUDED.status,
#                 score_title = EXCLUDED.score_title,
#                 score_description = EXCLUDED.score_description,
#                 score_competences = EXCLUDED.score_competences
#         """, (
#             offre_id, 
#             profil, 
#             score, 
#             confidence,
#             profil_second,
#             profil_second_score,
#             num_tokens, 
#             num_competences, 
#             status,
#             score_title,
#             score_description,
#             score_competences
#         ))
#     except Exception as e:
#         print(f"  ⚠️ Profil storage: {e}")
    
#     conn.commit()
    
#     print(f"  ✅ Pipeline terminé : {profil} ({score:.2f}), {num_competences} compétences")
    
#     return {
#         'profil': profil,
#         'score': score,
#         'confidence': confidence,
#         'profil_second': profil_second,
#         'profil_second_score': profil_second_score,
#         'competences': competences,
#         'num_competences': num_competences,
#         'num_tokens': num_tokens,
#         'tokens': tokens,
#         'description_clean': description_clean,
#         'text_for_sklearn': text_for_sklearn,
#         'score_title': score_title,
#         'score_description': score_description,
#         'score_competences': score_competences,
#         'status': status
#     }


# # ============================================
# # FONCTION FALLBACK CLASSIFICATION
# # ============================================

# def classify_fallback(title, competences):
#     """Classification basique si pipeline non disponible"""
    
#     title_lower = title.lower()
    
#     # Règles simples
#     if any(kw in title_lower for kw in ['data scientist', 'scientist', 'scientifique']):
#         return 'Data Scientist', 0.9, 0.85
    
#     elif any(kw in title_lower for kw in ['data engineer', 'engineer data', 'ingénieur données']):
#         return 'Data Engineer', 0.9, 0.85
    
#     elif any(kw in title_lower for kw in ['data analyst', 'analyst', 'analyste']):
#         return 'Data Analyst', 0.9, 0.85
    
#     elif any(kw in title_lower for kw in ['ml engineer', 'machine learning', 'mlops']):
#         return 'ML Engineer', 0.85, 0.80
    
#     elif any(kw in title_lower for kw in ['bi analyst', 'business intelligence']):
#         return 'BI Analyst', 0.85, 0.80
    
#     # Fallback sur compétences
#     else:
#         if 'spark' in competences or 'airflow' in competences or 'kafka' in competences:
#             return 'Data Engineer', 0.7, 0.65
#         elif 'machine learning' in competences or 'deep learning' in competences:
#             return 'Data Scientist', 0.7, 0.65
#         elif 'tableau' in competences or 'power bi' in competences:
#             return 'Data Analyst', 0.7, 0.65
#         else:
#             return 'Autre', 0.5, 0.50


# # ============================================
# # FONCTION HELPER : VÉRIFIER TABLES
# # ============================================

# def ensure_nlp_tables_exist(conn):
#     """Crée tables NLP si nécessaire"""
    
#     cur = conn.cursor()
    
#     # fact_preprocessing
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS fact_preprocessing (
#             offre_id INTEGER PRIMARY KEY,
#             tokens TEXT[],
#             num_tokens INTEGER,
#             FOREIGN KEY (offre_id) REFERENCES fact_offres(offre_id) ON DELETE CASCADE
#         )
#     """)
    
#     # fact_profils_nlp (STRUCTURE COMPLÈTE)
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS fact_profils_nlp (
#             offre_id INTEGER PRIMARY KEY,
#             profil_assigned VARCHAR(50),
#             score_classification FLOAT,
#             profil_confidence FLOAT,
#             profil_second VARCHAR(50),
#             profil_second_score FLOAT,
#             num_tokens INTEGER,
#             num_competences INTEGER,
#             status VARCHAR(20),
#             score_title FLOAT,
#             score_description FLOAT,
#             score_competences FLOAT,
#             cascade_pass INTEGER,
#             created_at TIMESTAMP DEFAULT NOW(),
#             FOREIGN KEY (offre_id) REFERENCES fact_offres(offre_id) ON DELETE CASCADE
#         )
#     """)
    
#     conn.commit()
#     print("✅ Tables NLP vérifiées")


# # ============================================
# # TEST
# # ============================================

# if __name__ == "__main__":
    
#     import sys
#     sys.path.insert(0, str(Path(__file__).parent))
#     from config_db import get_db_connection
    
#     print("=" * 70)
#     print("TEST PIPELINE NLP WRAPPER ULTRA-COMPLET")
#     print("=" * 70)
    
#     conn = get_db_connection()
#     if not conn:
#         print("❌ Connexion impossible")
#         exit(1)
    
#     # Créer tables
#     ensure_nlp_tables_exist(conn)
    
#     # Test
#     test_data = {
#         'offre_id': 9999,
#         'title': 'Data Engineer Senior - Spark & AWS',
#         'description': """
#         Nous recherchons un Data Engineer expérimenté pour construire 
#         notre data platform cloud. 
        
#         Compétences requises :
#         - Spark, Python, SQL
#         - Airflow, DBT
#         - AWS (S3, EMR, Glue)
#         - Docker, Kubernetes
#         - PostgreSQL, MongoDB
        
#         Expérience MLOps et CI/CD appréciée.
#         """
#     }
    
#     try:
#         results = process_single_offre(
#             offre_id=test_data['offre_id'],
#             title=test_data['title'],
#             description=test_data['description'],
#             conn=conn
#         )
        
#         print("\n" + "=" * 70)
#         print("✅ TEST RÉUSSI")
#         print("=" * 70)
#         print(f"\n📊 RÉSULTATS NLP:")
#         print(f"  Profil détecté : {results['profil']}")
#         print(f"  Score : {results['score']:.2f}")
#         print(f"  Confiance : {results['confidence']:.2f}")
#         if results['profil_second']:
#             print(f"  2e profil : {results['profil_second']} ({results['profil_second_score']:.2f})")
        
#         print(f"\n🔧 COMPÉTENCES ({results['num_competences']}):")
#         for i, comp in enumerate(results['competences'][:15], 1):
#             print(f"  {i}. {comp}")
        
#         print(f"\n📝 PREPROCESSING:")
#         print(f"  Tokens : {results['num_tokens']}")
#         print(f"  Exemple tokens : {' '.join(results['tokens'][:15])}...")
        
#         print(f"\n🎯 SCORES DÉTAILLÉS:")
#         print(f"  Titre : {results['score_title']:.1f}/10")
#         print(f"  Description : {results['score_description']:.1f}/10")
#         print(f"  Compétences : {results['score_competences']:.1f}/10")
        
#     except Exception as e:
#         print(f"\n❌ ERREUR TEST : {e}")
#         import traceback
#         traceback.print_exc()
    
#     finally:
#         conn.close()