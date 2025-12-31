"""
Définitions des Profils Métier - VERSION v2 FINALE
14 profils Data/IA dont "Data/IA - Non spécifié" (profil fourre-tout)

NOUVEAUTÉS v2:
- Profil #14: "Data/IA - Non spécifié" (capture reste avec seuil 1.5)
- Seuils optimisés
- Variantes enrichies

Résultat attendu: 85-90% classification

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

# Configuration globale
CLASSIFICATION_CONFIG = {
    'min_score': 4.5,
    'min_confidence': 0.55,
    'weights': {
        'title': 0.6,
        'description': 0.2,
        'competences': 0.2
    }
}

# Profils stricts
STRICT_PROFILES = [
    'AI Engineer',
    'AI Research Scientist',
    'Computer Vision Engineer'
]

# Profils permissifs
PERMISSIVE_PROFILES = [
    'Data Analyst',
    'Data Consultant',
    'Data Manager',
    'Data/IA - Non spécifié'  # ✅ NOUVEAU
]


# ============================================
# PROFILS 1-13 : IDENTIQUES À v1_optimized
# ============================================

PROFILS = {
    
    # [Tous les profils 1-13 de v1_optimized...]
    # Je vais les copier depuis v1_optimized
    
    'Data Engineer': {
        'description': 'Pipelines données, ETL, Big Data, Architecte Data, cloud',
        'title_variants': [
            'data engineer', 'engineer data', 'data engineering',
            'ingénieur données', 'ingenieur donnees',
            'ingénieur data', 'ingenieur data',
            'big data',
            'ingénieur big data', 'ingenieur big data',
            'développeur big data', 'developpeur big data',
            'big data engineer', 'big data developer',
            'lead data engineer',
            'lead engineer data',
            'tech lead data engineer',
            'senior data engineer',
            'architecte data', 'architecte données', 'architecte donnees',
            'data architect',
            'développeur data', 'developpeur data',
            'data ingénieur', 'data ingenieur',
            'ingénieur support data', 'ingenieur support data',
            'ingénieur etl', 'ingenieur etl',
            'etl engineer', 'etl developer',
            'cloud data engineer'
        ],
        'keywords_title': [
            'pipeline', 'etl', 'elt',
            'big data', 'warehouse',
            'cloud', 'plateforme',
            'architecte', 'architect',
            'lead'
        ],
        'keywords_strong': [
            'airflow', 'kafka', 'spark',
            'hadoop', 'hive',
            'data lake', 'lakehouse',
            'dbt', 'streaming'
        ],
        'competences_core': [
            'sql', 'python', 'airflow', 'spark',
            'aws', 'data engineer', 'big data'
        ],
        'competences_tech': [
            'kafka', 'docker', 'kubernetes',
            'postgresql', 'mongodb'
        ],
        'weights': {
            'title': 0.6,
            'description': 0.2,
            'competences': 0.2
        }
    },
    
    'Data Scientist': {
        'description': 'ML classique, statistiques, modèles prédictifs',
        'title_variants': [
            'data scientist', 'scientist data',
            'data science engineer',
            'scientifique données', 'scientifique de données',
            'scientist', 'scientifique',
            'statisticien', 'statisticienne',
            'chargé études statistiques', 'charge etudes statistiques',
            'chargée études statistiques', 'chargee etudes statistiques',
            'analyste statistique',
            'ml scientist', 'machine learning scientist',
            'consultant data scientist',
            'consultante data scientist'
        ],
        'keywords_title': [
            'machine learning', 'ml',
            'statistiques', 'statisticien',
            'prédictif', 'predictive',
            'modèle', 'scientist'
        ],
        'keywords_strong': [
            'scikit-learn', 'sklearn',
            'régression', 'classification',
            'prédiction', 'prediction',
            'xgboost', 'lightgbm'
        ],
        'competences_core': [
            'machine learning', 'python', 'scikit-learn',
            'statistiques', 'r'
        ],
        'competences_tech': [
            'pandas', 'numpy', 'jupyter',
            'matplotlib', 'seaborn', 'sql'
        ],
        'weights': {
            'title': 0.6,
            'description': 0.2,
            'competences': 0.2
        }
    },
    
    'Data Analyst': {
        'description': 'Analyse exploratoire, SQL, Excel, reporting',
        'title_variants': [
            'data analyst', 'analyste données', 'analyste de données',
            'analyst data', 'analyste data',
            'junior data analyst',
            'analyste', 'analyst',
            'data analysis',
            'senior data analyst',
            'lead data analyst'
        ],
        'keywords_title': [
            'analyse', 'analysis',
            'sql', 'excel',
            'reporting',
            'senior'
        ],
        'keywords_strong': [
            'analyse exploratoire',
            'data cleaning',
            'statistiques descriptives',
            'kpi', 'metrics'
        ],
        'competences_core': [
            'sql', 'excel', 'analyse',
            'python'
        ],
        'competences_tech': [
            'pandas', 'sql', 'excel',
            'power bi', 'tableau'
        ],
        'weights': {
            'title': 0.6,
            'description': 0.2,
            'competences': 0.2
        }
    },
    
    'BI Analyst': {
        'description': 'Business Intelligence, dashboards, reporting, business analyst',
        'title_variants': [
            'bi analyst', 'business intelligence analyst',
            'analyste bi', 'analyste business intelligence',
            'business analyst',
            'business analyst data',
            'business analyst (h/f)',
            'business analyst f/h',
            'analyste business',
            'analyste affaires',
            'ba data',
            'analyste métier', 'analyste metier',
            'développeur bi', 'developpeur bi',
            'bi developer', 'business intelligence developer',
            'analyste décisionnel', 'analyste decisionnel',
            'développeur décisionnel', 'developpeur decisionnel',
            'business developer data',
            'business developer',
            'tableau analyst', 'power bi analyst',
            'tableau developer', 'power bi developer'
        ],
        'keywords_title': [
            'tableau', 'power bi', 'powerbi',
            'looker', 'qlik',
            'bi', 'business intelligence',
            'décisionnel', 'decisionnel',
            'dashboard',
            'business analyst'
        ],
        'keywords_strong': [
            'dashboard', 'reporting',
            'visualisation données',
            'data visualization', 'dataviz',
            'kpi', 'metrics',
            'dax', 'powerquery'
        ],
        'competences_core': [
            'power bi', 'tableau', 'sql',
            'excel', 'looker',
            'business analysis'
        ],
        'competences_tech': [
            'dax', 'powerquery', 'qlik',
            'sql', 'excel'
        ],
        'weights': {
            'title': 0.65,
            'description': 0.15,
            'competences': 0.2
        }
    },
    
    'Data Consultant': {
        'description': 'Conseil Data, transformation, accompagnement client',
        'title_variants': [
            'consultant data', 'data consultant',
            'consultant', 'consultante',
            'conseil data',
            'consulting data',
            'consultant data engineer',
            'consultante data engineer'
        ],
        'keywords_title': [
            'consultant', 'conseil',
            'consulting',
            'transformation',
            'advisory'
        ],
        'keywords_strong': [
            'transformation digitale',
            'accompagnement',
            'client', 'mission',
            'esn', 'cabinet'
        ],
        'competences_core': [
            'conseil', 'transformation',
            'management', 'gestion projet'
        ],
        'competences_tech': [
            'python', 'sql', 'excel',
            'power bi'
        ],
        'weights': {
            'title': 0.6,
            'description': 0.25,
            'competences': 0.15
        }
    },
    
    'Data Manager': {
        'description': 'Management équipe data, chef projet data, CDO, direction',
        'title_variants': [
            'data manager', 'manager data',
            'responsable data', 'responsable données', 'responsable donnees',
            'team lead data', 'lead data',
            'chef de projet data', 'chef projet data',
            'chef de projets moa data',
            'chef projet moa data',
            'product manager data', 'pm data',
            'product owner data', 'po data',
            'chief data officer', 'cdo',
            'directeur data', 'directrice data',
            'directeur data ai', 'directrice data ai',
            'directeur data ai factory', 'directrice data ai factory',
            'head of data', 'data director'
        ],
        'keywords_title': [
            'manager', 'responsable',
            'chef', 'director', 'directeur', 'directrice',
            'lead', 'head', 'cdo',
            'product', 'po', 'pm',
            'moa'
        ],
        'keywords_strong': [
            'management', 'équipe', 'team',
            'projet', 'product',
            'stratégie', 'gouvernance',
            'roadmap', 'transformation',
            'factory'
        ],
        'competences_core': [
            'management', 'gestion projet',
            'leadership', 'stratégie'
        ],
        'competences_tech': [
            'sql', 'python', 'agile',
            'jira', 'scrum'
        ],
        'weights': {
            'title': 0.65,
            'description': 0.2,
            'competences': 0.15
        }
    },
    
    'Data Architect': {
        'description': 'Architecture données, gouvernance, stratégie senior',
        'title_variants': [
            'data architect', 'architecte données', 'architecte donnees',
            'architect data',
            'data architect (h/f)',
            'architecte si data',
            'si data architect',
            'architecte solution data',
            'solution architect data',
            'chief data architect',
            'lead architect',
            'data architect confirmé', 'data architect confirme'
        ],
        'keywords_title': [
            'architecture', 'architect',
            'gouvernance', 'governance',
            'stratégie', 'strategy',
            'solution'
        ],
        'keywords_strong': [
            'data architecture',
            'enterprise architecture',
            'data modeling',
            'master data management', 'mdm',
            'data catalog'
        ],
        'competences_core': [
            'architecture', 'gouvernance',
            'data modeling', 'sql'
        ],
        'competences_tech': [
            'sql', 'cloud', 'aws', 'azure',
            'databricks'
        ],
        'weights': {
            'title': 0.65,
            'description': 0.2,
            'competences': 0.15
        }
    },
    
    'AI Engineer': {
        'description': 'IA générative, LLMs, NLP avancé, transformers',
        'title_variants': [
            'ai engineer', 'ingénieur ia', 'ingenieur ia',
            'engineer ai', 'ia engineer',
            'artificial intelligence engineer',
            'llm engineer', 'nlp engineer',
            'tech lead ia', 'tech lead ai',
            'lead ai engineer', 'lead ia engineer',
            'chef de projet ia', 'chef projet ia',
            'chef de projet technique ia'
        ],
        'keywords_title': [
            'llm', 'llms',
            'gpt', 'chatgpt',
            'transformers', 'bert',
            'nlp', 'generative',
            'ia générative', 'generative ai',
            'tech lead'
        ],
        'keywords_strong': [
            'langchain', 'llamaindex',
            'rag', 'retrieval augmented',
            'prompt engineering',
            'fine-tuning',
            'hugging face',
            'chatbot', 'embedding'
        ],
        'competences_core': [
            'intelligence artificielle', 'llm',
            'transformers', 'gpt', 'langchain', 'nlp'
        ],
        'competences_tech': [
            'python', 'pytorch', 'tensorflow',
            'hugging face', 'api'
        ],
        'weights': {
            'title': 0.65,
            'description': 0.2,
            'competences': 0.15
        }
    },
    
    'ML Engineer': {
        'description': 'MLOps, déploiement ML, production, pipelines ML',
        'title_variants': [
            'ml engineer', 'machine learning engineer',
            'ingénieur ml', 'ingenieur ml',
            'engineer ml',
            'mlops engineer', 'ml ops engineer'
        ],
        'keywords_title': [
            'mlops', 'ml ops',
            'déploiement', 'deployment',
            'production',
            'devops ml'
        ],
        'keywords_strong': [
            'mlflow', 'kubeflow',
            'model deployment',
            'kubernetes', 'docker',
            'ci/cd ml'
        ],
        'competences_core': [
            'mlops', 'ci/cd', 'kubernetes', 'docker',
            'machine learning'
        ],
        'competences_tech': [
            'mlflow', 'kubeflow', 'tensorflow',
            'pytorch', 'git', 'linux'
        ],
        'weights': {
            'title': 0.6,
            'description': 0.2,
            'competences': 0.2
        }
    },
    
    'Analytics Engineer': {
        'description': 'Transformation données, dbt, SQL avancé, analytics',
        'title_variants': [
            'analytics engineer',
            'ingénieur analytics', 'ingenieur analytics',
            'dbt engineer'
        ],
        'keywords_title': [
            'analytics', 'dbt',
            'transformation', 'sql'
        ],
        'keywords_strong': [
            'data modeling',
            'data transformation',
            'looker', 'metabase'
        ],
        'competences_core': [
            'dbt', 'sql', 'python',
            'data modeling'
        ],
        'competences_tech': [
            'git', 'postgresql', 'snowflake',
            'databricks'
        ],
        'weights': {
            'title': 0.6,
            'description': 0.2,
            'competences': 0.2
        }
    },
    
    'MLOps Engineer': {
        'description': 'DevOps pour ML, CI/CD ML, infrastructure ML',
        'title_variants': [
            'mlops engineer', 'ml ops engineer',
            'ingénieur mlops', 'ingenieur mlops',
            'devops ml'
        ],
        'keywords_title': [
            'mlops', 'ml ops',
            'devops',
            'kubernetes', 'k8s'
        ],
        'keywords_strong': [
            'ci/cd', 'terraform',
            'infrastructure as code',
            'monitoring'
        ],
        'competences_core': [
            'mlops', 'kubernetes', 'docker',
            'ci/cd', 'devops'
        ],
        'competences_tech': [
            'terraform', 'git', 'linux',
            'aws', 'azure'
        ],
        'weights': {
            'title': 0.65,
            'description': 0.15,
            'competences': 0.2
        }
    },
    
    'AI Research Scientist': {
        'description': 'Recherche IA, publications, PhD, innovation',
        'title_variants': [
            'research scientist', 'chercheur',
            'researcher', 'scientifique recherche',
            'ai researcher', 'ml researcher',
            'research engineer'
        ],
        'keywords_title': [
            'research', 'recherche',
            'phd', 'doctorat',
            'chercheur', 'postdoc'
        ],
        'keywords_strong': [
            'publication', 'paper',
            'conference', 'neurips', 'icml',
            'innovation'
        ],
        'competences_core': [
            'recherche', 'intelligence artificielle',
            'machine learning', 'deep learning'
        ],
        'competences_tech': [
            'python', 'pytorch', 'tensorflow',
            'jupyter'
        ],
        'weights': {
            'title': 0.7,
            'description': 0.15,
            'competences': 0.15
        }
    },
    
    'Computer Vision Engineer': {
        'description': 'Vision par ordinateur, images, vidéo, CNN',
        'title_variants': [
            'computer vision engineer',
            'ingénieur computer vision', 'ingenieur computer vision',
            'cv engineer',
            'vision engineer',
            'image processing engineer'
        ],
        'keywords_title': [
            'computer vision', 'vision',
            'image', 'vidéo', 'video',
            'opencv', 'cv'
        ],
        'keywords_strong': [
            'cnn', 'convolutional',
            'yolo', 'mask r-cnn',
            'object detection',
            'image segmentation'
        ],
        'competences_core': [
            'computer vision', 'deep learning',
            'opencv', 'pytorch'
        ],
        'competences_tech': [
            'python', 'opencv', 'cuda',
            'tensorflow'
        ],
        'weights': {
            'title': 0.7,
            'description': 0.15,
            'competences': 0.15
        }
    },
    
    # ========================================
    # 14. DATA/IA - NON SPÉCIFIÉ (NOUVEAU)
    # ========================================
    'Data/IA - Non spécifié': {
        'description': 'Postes Data/IA sans informations suffisantes pour classification précise',
        
        # ✅ Capture TOUT ce qui contient data/ia
        'title_variants': [
            # Data
            'data', 'donnees', 'donnée',
            'database', 'base de donnees',
            
            # IA
            'ia', 'ai',
            'intelligence artificielle',
            'artificial intelligence',
            
            # ML
            'machine learning', 'ml',
            'deep learning', 'dl',
            
            # Big Data
            'big data',
            
            # Analytics
            'analytics', 'analytique',
            'analyse de donnees', 'analyse donnees'
        ],
        
        'keywords_title': [
            'data', 'donnees', 'donnée',
            'ia', 'ai',
            'machine learning', 'ml',
            'analytics', 'analytique',
            'big data'
        ],
        
        # Pas de filtre sur description/compétences
        'keywords_strong': [],
        
        'competences_core': [],
        
        'competences_tech': [],
        
        # ✅ POIDS ULTRA-PERMISSIF (presque tout sur titre)
        'weights': {
            'title': 0.9,        # 90% titre
            'description': 0.05,
            'competences': 0.05
        }
    }
}


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def get_profil_config(profil_name):
    if profil_name not in PROFILS:
        raise ValueError(f"Profil '{profil_name}' non trouvé")
    return PROFILS[profil_name]


def get_all_profils():
    return list(PROFILS.keys())


def get_min_score(profil_name):
    """Retourne seuil minimum pour un profil"""
    
    # ✅ Profil fourre-tout : seuil TRÈS BAS
    if profil_name == 'Data/IA - Non spécifié':
        return 1.5
    
    # Profils stricts
    if profil_name in STRICT_PROFILES:
        return 5.0
    
    # Profils permissifs
    elif profil_name in PERMISSIVE_PROFILES:
        return 4.0
    
    # Défaut
    else:
        return CLASSIFICATION_CONFIG['min_score']


def export_profils_json(filepath):
    import json
    
    export_data = {
        'version': 'v2_with_catch_all',
        'config': CLASSIFICATION_CONFIG,
        'strict_profiles': STRICT_PROFILES,
        'permissive_profiles': PERMISSIVE_PROFILES,
        'profils': PROFILS
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Profils v2 exportés: {filepath}")


if __name__ == "__main__":
    print("="*70)
    print("📋 VALIDATION DÉFINITIONS PROFILS v2 (avec fourre-tout)")
    print("="*70)
    
    print(f"\n✅ Version: v2 FINALE")
    print(f"✅ Nombre de profils: {len(PROFILS)}")
    print(f"✅ Min score global: {CLASSIFICATION_CONFIG['min_score']}")
    print(f"✅ Min confidence: {CLASSIFICATION_CONFIG['min_confidence']}")
    
    print("\n📊 Nouveautés v2:")
    print("   ✅ Profil #14: 'Data/IA - Non spécifié' (fourre-tout)")
    print("   ✅ Seuil profil fourre-tout: 1.5 (ultra-permissif)")
    print("   ✅ Poids profil fourre-tout: 90% titre")
    print("   ✅ Capture tout ce qui contient 'data', 'ia', 'ml', etc.")
    
    print("\n✅ Validation terminée !")
    print("\n📊 Résultat attendu:")
    print("   ✅ Taux classification: 85-90%")
    print("   ✅ Profils 1-13: classification précise (60-65%)")
    print("   ✅ Profil #14: capture reste Data/IA (20-25%)")


# ============================================
# VALIDATION
# ============================================

if __name__ == "__main__":
    print("="*70)
    print("📋 VALIDATION DÉFINITIONS PROFILS v5 ENRICHIE")
    print("="*70)
    
    print(f"\n✅ Version: v5 - Base nettoyée + enrichissements FR")
    print(f"✅ Nombre de profils: {len(PROFILS)}")
    print(f"✅ Min score global: {CLASSIFICATION_CONFIG['min_score']}")
    
    print("\n📊 Nouveautés v5:")
    print("   ✅ NOUVEAU profil: Data Manager (25 variants)")
    print("   ✅ Data Engineer: + Big Data, Architecte Data, Développeur Data")
    print("   ✅ Data Scientist: + Statisticien")
    print("   ✅ BI Analyst: + Développeur BI, Analyste décisionnel")
    
    print("\n📋 Liste des profils:")
    for i, profil_name in enumerate(PROFILS.keys(), 1):
        profil = PROFILS[profil_name]
        nb_variants = len(profil['title_variants'])
        min_score = get_min_score(profil_name)
        
        nouveau = " ⭐ NOUVEAU" if profil_name == "Data Manager" else ""
        enrichi = " 📈 ENRICHI" if profil_name in ['Data Engineer', 'Data Scientist', 'BI Analyst'] else ""
        
        print(f"\n{i:2d}. {profil_name}{nouveau}{enrichi}")
        print(f"    {profil['description']}")
        print(f"    Variantes titre: {nb_variants}")
        print(f"    Score min: {min_score}/10")
    
    print("\n✅ Validation terminée !")
    print("\n📊 Résultat attendu sur base nettoyée:")
    print("   ✅ Taux classification: 70-80%")
    print("   ✅ Data Manager capture: ~85 offres (Manager, Chef projet, CDO)")
    print("   ✅ Data Engineer capture: + Big Data, Architecte Data")
    print("   ✅ Vocabulaire FR enrichi")
# """
# Définitions des Profils Métier - Classification Hybride
# 12 profils Data/IA avec keywords, compétences et paramètres de scoring

# Auteur: Projet NLP Text Mining
# Date: Décembre 2025
# """

# # Configuration globale
# CLASSIFICATION_CONFIG = {
#     'min_score': 4.0,           # Score minimum pour classifier
#     'min_confidence': 0.6,      # Confiance minimum
#     'required_keywords': 1,     # Nb keywords required minimum
#     'default_weights': {
#         'rules': 0.3,           # Poids règles
#         'tfidf': 0.4,           # Poids ML
#         'competences': 0.3      # Poids compétences
#     }
# }

# # Profils stricts (score min plus élevé)
# STRICT_PROFILES = [
#     'AI Engineer',
#     'AI Research Scientist',
#     'Computer Vision Engineer'
# ]

# # Profils permissifs (score min plus bas)
# PERMISSIVE_PROFILES = [
#     'Data Analyst',
#     'Data Consultant'
# ]


# # ============================================
# # DÉFINITION DES 12 PROFILS
# # ============================================

# PROFILS = {
    
#     # ========================================
#     # 1. DATA SCIENTIST
#     # ========================================
#     'Data Scientist': {
#         'description': 'ML classique, statistiques, modèles prédictifs',
        
#         'keywords_required': [
#             'data scientist', 'data science',
#             'machine learning', 'apprentissage automatique',
#             'modèle prédictif', 'modélisation'
#         ],
        
#         'keywords_strong': [
#             'scikit-learn', 'sklearn',
#             'statistiques', 'statistics',
#             'régression', 'classification',
#             'prédiction', 'prediction',
#             'analyse prédictive',
#             'xgboost', 'lightgbm',
#             'features engineering',
#             'data mining',
#             'clustering', 'segmentation'
#         ],
        
#         'competences_core': [
#             'machine learning', 'python', 'scikit-learn',
#             'statistiques', 'r', 'data science'
#         ],
        
#         'competences_tech': [
#             'pandas', 'numpy', 'jupyter',
#             'matplotlib', 'seaborn', 'plotly',
#             'sql', 'git'
#         ],
        
#         'exclude_keywords': [
#             'llm', 'gpt', 'transformers',  # → AI Engineer
#             'tableau', 'power bi',          # → BI Analyst
#             'airflow', 'kafka', 'pipeline'  # → Data Engineer
#         ],
        
#         'weights': {
#             'rules': 0.3,
#             'tfidf': 0.4,
#             'competences': 0.3
#         }
#     },
    
#     # ========================================
#     # 2. AI ENGINEER
#     # ========================================
#     'AI Engineer': {
#         'description': 'IA générative, LLMs, NLP avancé, transformers',
        
#         'keywords_required': [
#             'intelligence artificielle',
#             'llm', 'llms', 'large language model',
#             'gpt', 'bert', 'transformers'
#         ],
        
#         'keywords_strong': [
#             'langchain', 'llamaindex',
#             'rag', 'retrieval augmented generation',
#             'prompt engineering',
#             'fine-tuning', 'fine tuning',
#             'hugging face', 'huggingface',
#             'openai', 'anthropic', 'claude',
#             'generative ai', 'ia générative',
#             'chatbot', 'conversational ai',
#             'embedding', 'embeddings',
#             'vector database', 'pinecone', 'weaviate'
#         ],
        
#         'competences_core': [
#             'intelligence artificielle', 'llm', 'llms',
#             'transformers', 'gpt', 'bert',
#             'langchain', 'nlp'
#         ],
        
#         'competences_tech': [
#             'python', 'pytorch', 'tensorflow',
#             'hugging face', 'api', 'rest'
#         ],
        
#         'exclude_keywords': [
#             'scikit-learn',  # → Data Scientist
#             'tableau', 'power bi',  # → BI Analyst
#         ],
        
#         'weights': {
#             'rules': 0.4,      # Plus de poids sur règles (profil émergent)
#             'tfidf': 0.3,
#             'competences': 0.3
#         }
#     },
    
#     # ========================================
#     # 3. ML ENGINEER
#     # ========================================
#     'ML Engineer': {
#         'description': 'MLOps, déploiement ML, production, pipelines ML',
        
#         'keywords_required': [
#             'ml engineer', 'machine learning engineer',
#             'mlops', 'ml ops',
#             'déploiement', 'deployment',
#             'production ml'
#         ],
        
#         'keywords_strong': [
#             'mlflow', 'kubeflow',
#             'model deployment', 'model serving',
#             'api ml', 'rest api',
#             'containerization', 'conteneurisation',
#             'monitoring ml', 'model monitoring',
#             'ci/cd ml', 'cicd',
#             'feature store',
#             'model registry',
#             'a/b testing',
#             'model versioning'
#         ],
        
#         'competences_core': [
#             'mlops', 'ci/cd', 'kubernetes', 'docker',
#             'machine learning', 'python'
#         ],
        
#         'competences_tech': [
#             'mlflow', 'kubeflow', 'tensorflow',
#             'pytorch', 'git', 'linux'
#         ],
        
#         'exclude_keywords': [
#             'llm', 'gpt',  # → AI Engineer
#             'airflow',     # → Data Engineer (sauf si ML aussi)
#         ],
        
#         'weights': {
#             'rules': 0.3,
#             'tfidf': 0.4,
#             'competences': 0.3
#         }
#     },
    
#     # ========================================
#     # 4. DATA ENGINEER
#     # ========================================
#     'Data Engineer': {
#         'description': 'Pipelines données, ETL, data warehousing, cloud',
        
#         'keywords_required': [
#             'data engineer', 'ingénieur données',
#             'pipeline', 'etl',
#             'data warehouse', 'entrepôt données'
#         ],
        
#         'keywords_strong': [
#             'airflow', 'kafka', 'spark',
#             'hadoop', 'hive',
#             'data lake', 'lakehouse',
#             'dbt', 'data build tool',
#             'streaming', 'batch processing',
#             'orchestration', 'orchestrateur',
#             'data integration',
#             'data ingestion',
#             'cloud data platform'
#         ],
        
#         'competences_core': [
#             'sql', 'python', 'airflow', 'spark',
#             'aws', 'data engineer'
#         ],
        
#         'competences_tech': [
#             'kafka', 'docker', 'kubernetes',
#             'postgresql', 'mongodb', 'redis'
#         ],
        
#         'exclude_keywords': [
#             'tableau', 'power bi',  # → BI Analyst
#             'scikit-learn',         # → Data Scientist
#         ],
        
#         'weights': {
#             'rules': 0.3,
#             'tfidf': 0.4,
#             'competences': 0.3
#         }
#     },
    
#     # ========================================
#     # 5. ANALYTICS ENGINEER
#     # ========================================
#     'Analytics Engineer': {
#         'description': 'Transformation données, dbt, SQL avancé, analytics',
        
#         'keywords_required': [
#             'analytics engineer',
#             'dbt', 'data build tool',
#             'transformation données'
#         ],
        
#         'keywords_strong': [
#             'sql avancé', 'advanced sql',
#             'data modeling', 'modélisation données',
#             'data transformation',
#             'looker', 'metabase',
#             'analytics',
#             'business intelligence',
#             'data quality',
#             'data testing',
#             'version control sql'
#         ],
        
#         'competences_core': [
#             'dbt', 'sql', 'python',
#             'data modeling'
#         ],
        
#         'competences_tech': [
#             'git', 'postgresql', 'snowflake',
#             'databricks', 'looker'
#         ],
        
#         'exclude_keywords': [
#             'machine learning',  # → Data Scientist
#             'airflow', 'kafka',  # → Data Engineer
#         ],
        
#         'weights': {
#             'rules': 0.35,
#             'tfidf': 0.35,
#             'competences': 0.3
#         }
#     },
    
#     # ========================================
#     # 6. BI ANALYST
#     # ========================================
#     'BI Analyst': {
#         'description': 'Business Intelligence, dashboards, reporting, visualisation',
        
#         'keywords_required': [
#             'bi analyst', 'business intelligence',
#             'tableau', 'power bi',
#             'dashboard', 'reporting'
#         ],
        
#         'keywords_strong': [
#             'looker', 'qlik', 'metabase',
#             'visualisation données', 'data visualization',
#             'kpi', 'metrics',
#             'rapport', 'report',
#             'tableau de bord',
#             'business analyst',
#             'décisionnel',
#             'dataviz'
#         ],
        
#         'competences_core': [
#             'power bi', 'tableau', 'sql',
#             'excel', 'looker'
#         ],
        
#         'competences_tech': [
#             'dax', 'powerquery', 'qlik',
#             'sql', 'excel'
#         ],
        
#         'exclude_keywords': [
#             'machine learning',  # → Data Scientist
#             'python', 'airflow', # → Data Engineer
#         ],
        
#         'weights': {
#             'rules': 0.4,
#             'tfidf': 0.3,
#             'competences': 0.3
#         }
#     },
    
#     # ========================================
#     # 7. MLOPS ENGINEER
#     # ========================================
#     'MLOps Engineer': {
#         'description': 'DevOps pour ML, CI/CD ML, infrastructure ML',
        
#         'keywords_required': [
#             'mlops', 'ml ops',
#             'devops ml', 'ml devops'
#         ],
        
#         'keywords_strong': [
#             'kubernetes', 'docker',
#             'ci/cd', 'cicd',
#             'terraform', 'infrastructure as code',
#             'monitoring ml', 'observability',
#             'gitlab ci', 'github actions',
#             'model serving',
#             'scalability ml',
#             'cloud infrastructure'
#         ],
        
#         'competences_core': [
#             'mlops', 'kubernetes', 'docker',
#             'ci/cd', 'devops'
#         ],
        
#         'competences_tech': [
#             'terraform', 'git', 'linux',
#             'aws', 'azure', 'gcp'
#         ],
        
#         'exclude_keywords': [],
        
#         'weights': {
#             'rules': 0.4,
#             'tfidf': 0.3,
#             'competences': 0.3
#         }
#     },
    
#     # ========================================
#     # 8. AI RESEARCH SCIENTIST
#     # ========================================
#     'AI Research Scientist': {
#         'description': 'Recherche IA, publications, PhD, innovation',
        
#         'keywords_required': [
#             'research scientist', 'chercheur',
#             'recherche', 'research',
#             'phd', 'doctorat'
#         ],
        
#         'keywords_strong': [
#             'publication', 'paper',
#             'conference', 'neurips', 'icml', 'iclr',
#             'innovation',
#             'state-of-the-art', 'sota',
#             'novel algorithm',
#             'academic',
#             'thesis', 'thèse',
#             'arxiv'
#         ],
        
#         'competences_core': [
#             'recherche', 'intelligence artificielle',
#             'machine learning', 'deep learning'
#         ],
        
#         'competences_tech': [
#             'python', 'pytorch', 'tensorflow',
#             'jupyter', 'git'
#         ],
        
#         'exclude_keywords': [
#             'production', 'deployment',  # → ML Engineer
#             'dashboard', 'reporting',    # → BI Analyst
#         ],
        
#         'weights': {
#             'rules': 0.5,  # Très spécifique
#             'tfidf': 0.3,
#             'competences': 0.2
#         }
#     },
    
#     # ========================================
#     # 9. COMPUTER VISION ENGINEER
#     # ========================================
#     'Computer Vision Engineer': {
#         'description': 'Vision par ordinateur, images, vidéo, CNN',
        
#         'keywords_required': [
#             'computer vision', 'vision par ordinateur',
#             'image processing', 'traitement image',
#             'cnn', 'convolutional'
#         ],
        
#         'keywords_strong': [
#             'opencv', 'yolo', 'mask r-cnn',
#             'object detection', 'détection objet',
#             'segmentation image',
#             'face recognition', 'reconnaissance faciale',
#             'video analysis',
#             'image classification',
#             'deep learning vision',
#             'resnet', 'vgg', 'inception'
#         ],
        
#         'competences_core': [
#             'computer vision', 'deep learning',
#             'opencv', 'pytorch', 'tensorflow'
#         ],
        
#         'competences_tech': [
#             'python', 'opencv', 'cuda',
#             'gpu', 'docker'
#         ],
        
#         'exclude_keywords': [
#             'nlp', 'text',  # → AI Engineer / NLP
#             'tableau',      # → BI Analyst
#         ],
        
#         'weights': {
#             'rules': 0.5,  # Très spécifique
#             'tfidf': 0.3,
#             'competences': 0.2
#         }
#     },
    
#     # ========================================
#     # 10. DATA ANALYST
#     # ========================================
#     'Data Analyst': {
#         'description': 'Analyse exploratoire, SQL, Excel, reporting simple',
        
#         'keywords_required': [
#             'data analyst', 'analyste données',
#             'analyse données', 'data analysis'
#         ],
        
#         'keywords_strong': [
#             'excel', 'google sheets',
#             'sql', 'query',
#             'analyse exploratoire', 'exploratory analysis',
#             'statistiques descriptives',
#             'rapport', 'reporting',
#             'data cleaning',
#             'data entry',
#             'spreadsheet'
#         ],
        
#         'competences_core': [
#             'sql', 'excel', 'analyse',
#             'python'
#         ],
        
#         'competences_tech': [
#             'pandas', 'sql', 'excel',
#             'power bi', 'tableau'
#         ],
        
#         'exclude_keywords': [
#             'machine learning',  # → Data Scientist
#             'airflow', 'kafka',  # → Data Engineer
#         ],
        
#         'weights': {
#             'rules': 0.25,  # Profil large
#             'tfidf': 0.45,
#             'competences': 0.3
#         }
#     },
    
#     # ========================================
#     # 11. DATA ARCHITECT
#     # ========================================
#     'Data Architect': {
#         'description': 'Architecture données, gouvernance, stratégie',
        
#         'keywords_required': [
#             'data architect', 'architecte données',
#             'architecture données', 'data architecture'
#         ],
        
#         'keywords_strong': [
#             'gouvernance', 'governance',
#             'data strategy', 'stratégie données',
#             'enterprise architecture',
#             'data modeling', 'modélisation',
#             'master data management', 'mdm',
#             'metadata management',
#             'data catalog',
#             'data quality',
#             'data lineage'
#         ],
        
#         'competences_core': [
#             'architecture', 'gouvernance',
#             'data modeling', 'sql'
#         ],
        
#         'competences_tech': [
#             'sql', 'cloud', 'aws', 'azure',
#             'databricks', 'snowflake'
#         ],
        
#         'exclude_keywords': [
#             'junior', 'stage',  # → Senior role
#         ],
        
#         'weights': {
#             'rules': 0.4,
#             'tfidf': 0.35,
#             'competences': 0.25
#         }
#     },
    
#     # ========================================
#     # 12. DATA CONSULTANT
#     # ========================================
#     'Data Consultant': {
#         'description': 'Conseil Data, transformation, accompagnement client',
        
#         'keywords_required': [
#             'consultant', 'conseil',
#             'consulting', 'advisory'
#         ],
        
#         'keywords_strong': [
#             'transformation', 'transformation digitale',
#             'accompagnement', 'accompagner',
#             'client', 'mission',
#             'conseil stratégique',
#             'change management',
#             'conduite changement',
#             'esn', 'ssii',
#             'cabinet conseil'
#         ],
        
#         'competences_core': [
#             'conseil', 'transformation',
#             'management', 'gestion projet'
#         ],
        
#         'competences_tech': [
#             'python', 'sql', 'excel',
#             'power bi', 'powerpoint'
#         ],
        
#         'exclude_keywords': [
#             'développement', 'coding',  # → Profils techniques
#         ],
        
#         'weights': {
#             'rules': 0.25,  # Profil large
#             'tfidf': 0.45,
#             'competences': 0.3
#         }
#     }
# }


# # ============================================
# # FONCTIONS UTILITAIRES
# # ============================================

# def get_profil_config(profil_name):
#     """Récupère la configuration d'un profil"""
#     if profil_name not in PROFILS:
#         raise ValueError(f"Profil '{profil_name}' non trouvé")
#     return PROFILS[profil_name]


# def get_all_profils():
#     """Retourne la liste de tous les profils"""
#     return list(PROFILS.keys())


# def get_min_score(profil_name):
#     """Retourne le score minimum pour un profil"""
#     if profil_name in STRICT_PROFILES:
#         return 5.0
#     elif profil_name in PERMISSIVE_PROFILES:
#         return 3.5
#     else:
#         return CLASSIFICATION_CONFIG['min_score']


# def export_profils_json(filepath):
#     """Exporte les profils en JSON"""
#     import json
    
#     export_data = {
#         'config': CLASSIFICATION_CONFIG,
#         'strict_profiles': STRICT_PROFILES,
#         'permissive_profiles': PERMISSIVE_PROFILES,
#         'profils': PROFILS
#     }
    
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(export_data, f, ensure_ascii=False, indent=2)
    
#     print(f"✅ Profils exportés: {filepath}")


# # ============================================
# # VALIDATION
# # ============================================

# if __name__ == "__main__":
#     print("="*70)
#     print("📋 VALIDATION DÉFINITIONS PROFILS")
#     print("="*70)
    
#     print(f"\n✅ Nombre de profils définis: {len(PROFILS)}")
#     print(f"✅ Profils stricts: {len(STRICT_PROFILES)}")
#     print(f"✅ Profils permissifs: {len(PERMISSIVE_PROFILES)}")
    
#     print("\n📋 Liste des profils:")
#     for i, profil_name in enumerate(PROFILS.keys(), 1):
#         profil = PROFILS[profil_name]
#         nb_keywords_req = len(profil['keywords_required'])
#         nb_keywords_strong = len(profil['keywords_strong'])
#         nb_comp_core = len(profil['competences_core'])
        
#         print(f"\n{i:2d}. {profil_name}")
#         print(f"    Description: {profil['description']}")
#         print(f"    Keywords required: {nb_keywords_req}")
#         print(f"    Keywords strong: {nb_keywords_strong}")
#         print(f"    Compétences core: {nb_comp_core}")
#         print(f"    Score min: {get_min_score(profil_name)}")
    
#     print("\n✅ Validation terminée !")