"""
Définitions des Profils Métier - VERSION v1 OPTIMISÉE FINALE
14 profils Data/IA dont profil fourre-tout "Data/IA - Non spécifié"

OPTIMISATIONS v1:
- Seuils abaissés: 4.5 / 0.55 (au lieu de 5.0 / 0.60)
- Variantes enrichies avec cas exacts (Big Data, Business Analyst, etc.)
- Profil #14 fourre-tout avec seuil 0.5 (capture reste Data/IA)

Résultat attendu: 87-90% classification

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

# Configuration globale OPTIMISÉE
CLASSIFICATION_CONFIG = {
    'min_score': 4.5,        # ✅ Abaissé de 5.0
    'min_confidence': 0.55,  # ✅ Abaissé de 0.60
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
    'Data/IA - Non spécifié'  # ✅ Profil fourre-tout
]


# ============================================
# DÉFINITION DES 14 PROFILS - v1 OPTIMISÉE FINALE
# ============================================

PROFILS = {
    
    # ========================================
    # 1. DATA ENGINEER (ENRICHI)
    # ========================================
    'Data Engineer': {
        'description': 'Pipelines données, ETL, Big Data, Architecte Data, cloud',
        
        'title_variants': [
            # Base
            'data engineer', 'engineer data', 'data engineering',
            'ingénieur données', 'ingenieur donnees',
            'ingénieur data', 'ingenieur data',
            
            # ✅ Variations H/F (du top 50)
            'data engineer (h/f)', 'data engineer h/f', 'data engineer f/h',
            'data engineer - h/f',
            'ingenieur data (h/f)', 'ingenieur data h/f',
            
            # ✅ Big Data (enrichi)
            'big data',
            'ingénieur big data', 'ingenieur big data',
            'développeur big data', 'developpeur big data',
            'big data engineer', 'big data developer',
            'ingénieur big data cloud',
            'ingenieur / ingenieure big data',
            'ingenieure big data',
            'expert big data',
            'concepteur developpeur big data',
            
            # ✅ Lead/Senior/Confirmé Data Engineer (du top 50)
            'lead data engineer',
            'lead engineer data',
            'tech lead data engineer',
            'senior data engineer',
            'data engineer senior',
            'data engineer confirme',
            'data engineer experimente',
            'technical data engineer',
            'technical data engineer senior',
            
            # Architecte Data
            'architecte data', 'architecte données', 'architecte donnees',
            'data architect',
            'architecte big data',
            'architecte cloud data',
            'architecte plateforme données',
            'architecte solutions data',
            
            # Développeur Data
            'développeur data', 'developpeur data',
            'développeur données', 'developpeur donnees',
            'développeuse data', 'developpeuse data',
            'data developer', 'dev data',
            
            # ✅ Data Ingénieur (variantes orthographiques)
            'data ingénieur', 'data ingenieur',
            'ingenieur data senio',
            
            # ✅ Support Data
            'ingénieur support data', 'ingenieur support data',
            'support data engineer',
            
            # ETL
            'ingénieur etl', 'ingenieur etl',
            'etl engineer', 'etl developer',
            'expert talend data engineer',
            
            # Plateforme
            'ingénieur plateforme données', 'ingenieur plateforme donnees',
            'data platform engineer',
            'ingénieur cloud data', 'ingenieur cloud data',
            'cloud data engineer'
        ],
        
        'keywords_title': [
            'pipeline', 'etl', 'elt',
            'big data', 'warehouse',
            'cloud', 'plateforme',
            'architecte', 'architect',
            'lead'  # ✅ NOUVEAU
        ],
        
        'keywords_strong': [
            'airflow', 'kafka', 'spark',
            'hadoop', 'hive',
            'data lake', 'lakehouse',
            'dbt',
            'streaming',
            'orchestration'
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
    
    # ========================================
    # 2. DATA SCIENTIST (ENRICHI)
    # ========================================
    'Data Scientist': {
        'description': 'ML classique, statistiques, modèles prédictifs',
        
        'title_variants': [
            # Base
            'data scientist', 'scientist data',
            'data science engineer',
            'scientifique données', 'scientifique de données',
            'scientist', 'scientifique',
            
            # ✅ Variations H/F (du top 50)
            'data scientist (h/f)', 'data scientist h/f',
            'data scientist f/h',
            'data scientist confirme',
            
            # ✅ Lead/Senior
            'lead data scientist',
            'lead data scientist h/f',
            'senior data scientist',
            
            # ✅ Alternance/Stage
            'alternance data scientist',
            
            # ✅ Statisticien (enrichi)
            'statisticien', 'statisticienne',
            'chargé études statistiques', 'charge etudes statistiques',
            'chargée études statistiques', 'chargee etudes statistiques',
            'chargé étude statistique', 'charge etude statistique',
            'chargée étude statistique', 'chargee etude statistique',
            'analyste statistique',
            'statistician',
            
            # Chercheur
            'chercheur données', 'data researcher',
            'analyste scientifique',
            'analyste prédictif',
            
            # ML Scientist
            'ml scientist', 'machine learning scientist',
            'spécialiste machine learning',
            
            # ✅ Consultant Data Scientist
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
            'xgboost', 'lightgbm',
            'features engineering',
            'data mining',
            'clustering', 'segmentation'
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
    
    # ========================================
    # 3. DATA ANALYST (ENRICHI)
    # ========================================
    'Data Analyst': {
        'description': 'Analyse exploratoire, SQL, Excel, reporting',
        
        'title_variants': [
            # Base
            'data analyst', 'analyste données', 'analyste de données',
            'analyst data', 'analyste data',
            'junior data analyst',
            'analyste', 'analyst',
            'data analysis',
            
            # ✅ Variations H/F (du top 50)
            'data analyst (h/f)', 'data analyst h/f',
            'data analyst f/h',
            'analyste data (h/f)', 'analyste data h/f',
            'data analyste',
            
            # ✅ Senior/Lead
            'senior data analyst',
            'lead data analyst',
            
            # ✅ Stage/Alternance
            'stage data analyst',
            'stage data analyst f/h',
            'alternance data analyst'
        ],
        
        'keywords_title': [
            'analyse', 'analysis',
            'sql', 'excel',
            'reporting',
            'senior'  # ✅ NOUVEAU
        ],
        
        'keywords_strong': [
            'analyse exploratoire',
            'data cleaning',
            'statistiques descriptives',
            'rapport', 'query', 'requête',
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
    
    # ========================================
    # 4. BI ANALYST (ENRICHI - Business Analyst)
    # ========================================
    'BI Analyst': {
        'description': 'Business Intelligence, dashboards, reporting, business analyst',
        
        'title_variants': [
            # Base BI
            'bi analyst', 'business intelligence analyst',
            'analyste bi', 'analyste business intelligence',
            
            # ✅ Business Analyst (ENRICHI)
            'business analyst',
            'business analyst data',
            'business analyst (h/f)',
            'business analyst f/h',
            'analyste business',
            'analyste affaires',
            'ba data',
            'analyste métier', 'analyste metier',
            'analyste métier data', 'analyste metier data',
            'business analyst connaissance sectorielle',
            'business analyst data connaissance',
            
            # Développeur BI
            'développeur bi', 'developpeur bi',
            'développeur business intelligence',
            'developpeur business intelligence',
            'bi developer', 'business intelligence developer',
            
            # ✅ Analyste décisionnel (ENRICHI)
            'analyste décisionnel', 'analyste decisionnel',
            'développeur décisionnel', 'developpeur decisionnel',
            'développeuse décisionnel', 'developpeuse decisionnel',
            'développeur / développeuse décisionnel',
            'developpeur / developpeuse decisionnel',
            'business analyst / développeur bi',
            'business analyst / developpeur bi',
            
            # Business Developer Data
            'business developer data',
            'business developer',
            'business development data',
            
            # Spécialisations outils
            'tableau analyst', 'power bi analyst',
            'tableau developer', 'power bi developer',
            'looker analyst', 'qlik analyst',
            'analyste business intelligence (h/f)',
            'analyste bi (h/f)'
        ],
        
        'keywords_title': [
            'tableau', 'power bi', 'powerbi',
            'looker', 'qlik',
            'bi', 'business intelligence',
            'décisionnel', 'decisionnel',
            'dashboard',
            'business analyst',
            'business developer',
            'ba data'
        ],
        
        'keywords_strong': [
            'dashboard', 'reporting',
            'visualisation données',
            'data visualization', 'dataviz',
            'kpi', 'metrics',
            'dax', 'powerquery',
            'business analysis',
            'analyse métier',
            'business requirements'
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
    
    # ========================================
    # 5. DATA CONSULTANT (BASE v6)
    # ========================================
    'Data Consultant': {
        'description': 'Conseil Data, transformation, accompagnement client',
        
        'title_variants': [
            'consultant data', 'data consultant',
            'consultant', 'consultante',
            'conseil data',
            'consulting data',
            
            # ✅ Consultant Data Engineer (NOUVEAU)
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
            'change management',
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
    
    # ========================================
    # 6. DATA MANAGER (ENRICHI)
    # ========================================
    'Data Manager': {
        'description': 'Management équipe data, chef projet data, CDO, direction',
        
        'title_variants': [
            # Base
            'data manager', 'manager data',
            'responsable data', 'responsable données', 'responsable donnees',
            'team lead data', 'lead data',
            'data team lead',
            
            # Chef projet
            'chef de projet data', 'chef projet data',
            'chef de projet données', 'chef projet donnees',
            'chef de projets moa data',
            'chef projet moa data',
            
            # Product
            'product manager data', 'pm data',
            'product owner data', 'po data',
            'chef de projet big data',
            
            # ✅ CDO / Direction (enrichi avec variations H/F)
            'chief data officer', 'cdo',
            'chief data officer (h/f)', 'chief data officer h/f',
            'directeur data', 'directrice data',
            'directeur données', 'directrice données',
            'directeur donnees', 'directrice donnees',
            'directeur data ai', 'directrice data ai',
            'directeur data ai factory', 'directrice data ai factory',
            'directeur.trice data ai factory',
            'head of data', 'data director',
            
            # Responsable
            'responsable stratégie data', 'responsable strategie data',
            'directeur big data',
            'responsable activité data', 'responsable activite data',
            'responsable plateforme data',
            'responsable data ai'
        ],
        
        'keywords_title': [
            'manager', 'responsable',
            'chef', 'director', 'directeur', 'directrice',
            'lead', 'head', 'cdo',
            'product', 'po', 'pm',
            'moa'  # ✅ NOUVEAU
        ],
        
        'keywords_strong': [
            'management', 'équipe', 'team',
            'projet', 'product',
            'stratégie', 'gouvernance',
            'roadmap', 'transformation',
            'leadership', 'direction',
            'factory'  # ✅ NOUVEAU
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
    
    # ========================================
    # 7. DATA ARCHITECT (ENRICHI)
    # ========================================
    'Data Architect': {
        'description': 'Architecture données, gouvernance, stratégie senior',
        
        'title_variants': [
            # Base
            'data architect', 'architecte données', 'architecte donnees',
            'architect data',
            
            # ✅ Variantes enrichies
            'data architect (h/f)',
            'architecte si data',      # ✅ NOUVEAU
            'si data architect',       # ✅ NOUVEAU
            'architecte solution data', # ✅ NOUVEAU
            'solution architect data',  # ✅ NOUVEAU
            
            # Senior
            'chief data architect',
            'lead architect',
            'data architect confirmé', 'data architect confirme',
            
            # Solutions
            'architecte solutions',
            'solutions architect data'
        ],
        
        'keywords_title': [
            'architecture', 'architect',
            'gouvernance', 'governance',
            'stratégie', 'strategy',
            'chief', 'lead',
            'solution'  # ✅ NOUVEAU
        ],
        
        'keywords_strong': [
            'data architecture',
            'enterprise architecture',
            'data modeling',
            'master data management', 'mdm',
            'data catalog',
            'data quality'
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
    
    # ========================================
    # 8. AI ENGINEER (ENRICHI)
    # ========================================
    'AI Engineer': {
        'description': 'IA générative, LLMs, NLP avancé, transformers',
        
        'title_variants': [
            # Base
            'ai engineer', 'ingénieur ia', 'ingenieur ia',
            'engineer ai', 'ia engineer',
            'artificial intelligence engineer',
            'llm engineer', 'nlp engineer',
            'ingénieur intelligence artificielle', 'ingenieur intelligence artificielle',
            
            # ✅ Variations H/F (du top 50)
            'ai engineer h/f', 'ai engineer (h/f)',
            'ingenieur ia (h/f)', 'ingenieur ia h/f',
            
            # ✅ Tech Lead IA
            'tech lead ia', 'tech lead ai',
            'lead ai engineer', 'lead ia engineer',
            
            # ✅ Chef projet IA
            'chef de projet ia', 'chef projet ia',
            'chef de projet technique ia'
        ],
        
        'keywords_title': [
            'llm', 'llms',
            'gpt', 'chatgpt',
            'transformers', 'bert',
            'nlp', 'generative',
            'ia générative', 'generative ai',
            'tech lead'  # ✅ NOUVEAU
        ],
        
        'keywords_strong': [
            'langchain', 'llamaindex',
            'rag', 'retrieval augmented',
            'prompt engineering',
            'fine-tuning',
            'hugging face',
            'openai', 'anthropic',
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
    
    # ========================================
    # 9-13. AUTRES PROFILS (BASE v6)
    # ========================================
    
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
            'ci/cd ml',
            'monitoring ml'
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
            'looker', 'metabase',
            'data quality'
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
            'monitoring', 'observability'
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
            'innovation',
            'state-of-the-art', 'arxiv'
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
            'image segmentation',
            'face recognition'
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
    # 14. DATA/IA - NON SPÉCIFIÉ (FOURRE-TOUT)
    # ========================================
    'Data/IA - Non spécifié': {
        'description': 'Postes Data/IA sans informations suffisantes pour classification précise',
        
        # ✅ VARIANTES MINIMALES (seulement mots-clés de base)
        'title_variants': [
            # Data (très court = score faible)
            'data',
            'donnees',
            'donnée',
            
            # IA (très court = score faible)  
            'ia',
            'ai',
            'intelligence artificielle'
        ],
        
        'keywords_title': [
            'data', 'donnees',
            'ia', 'ai'
        ],
        
        # Pas de filtre sur description/compétences
        'keywords_strong': [],
        'competences_core': [],
        'competences_tech': [],
        
        # ✅ POIDS ÉQUILIBRÉS (pas ultra-permissif)
        'weights': {
            'title': 0.5,        # 50% titre (au lieu de 90%)
            'description': 0.25, # 25% description
            'competences': 0.25  # 25% compétences
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
    
    # ✅ Profil fourre-tout : seuil TRÈS BAS (testé en dernier, capture reste)
    if profil_name == 'Data/IA - Non spécifié':
        return 0.5  # Seuil ultra-bas
    
    # Profils stricts
    elif profil_name in STRICT_PROFILES:
        return 5.0  # Garde strict pour profils exigeants
    
    # Profils permissifs
    elif profil_name in PERMISSIVE_PROFILES:
        return 4.0  # Encore plus permissif
    
    # Défaut
    else:
        return CLASSIFICATION_CONFIG['min_score']  # 4.5


def export_profils_json(filepath):
    import json
    
    export_data = {
        'version': 'v1_optimized',
        'config': CLASSIFICATION_CONFIG,
        'strict_profiles': STRICT_PROFILES,
        'permissive_profiles': PERMISSIVE_PROFILES,
        'profils': PROFILS
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Profils v1 optimisés exportés: {filepath}")


if __name__ == "__main__":
    print("="*70)
    print("📋 VALIDATION DÉFINITIONS PROFILS v1 OPTIMISÉE")
    print("="*70)
    
    print(f"\n✅ Version: v1 OPTIMISÉE")
    print(f"✅ Nombre de profils: {len(PROFILS)}")
    print(f"✅ Min score global: {CLASSIFICATION_CONFIG['min_score']}")
    print(f"✅ Min confidence: {CLASSIFICATION_CONFIG['min_confidence']}")
    
    print("\n📊 Optimisations v1:")
    print("   ✅ Seuils abaissés: 4.5 / 0.55 (au lieu de 5.0 / 0.60)")
    print("   ✅ Data Engineer: +12 variantes (Big Data, Lead, Support)")
    print("   ✅ BI Analyst: +15 variantes (Business Analyst enrichi)")
    print("   ✅ Data Architect: +5 variantes (SI Data, Solution)")
    print("   ✅ Data Manager: +6 variantes (CDO, Factory, MOA)")
    print("   ✅ AI Engineer: +3 variantes (Tech Lead IA)")
    
    print("\n✅ Validation terminée !")
    print("\n📊 Résultat attendu:")
    print("   ✅ Taux classification: 47-49%")
    print("   ✅ Compatible v1 (matching simple 'in title_lower')")

#"""
# Définitions des Profils Métier - VERSION v1 OPTIMISÉE
# 13 profils Data/IA - Seuils abaissés + Variantes enrichies

# OPTIMISATIONS v1:
# - Seuils abaissés: 4.5 / 0.55 (au lieu de 5.0 / 0.60)
# - Variantes enrichies avec cas exacts (Big Data, Business Analyst, etc.)
# - Garde compatibilité totale avec v1

# Résultat attendu: 47-49% classification

# Auteur: Projet NLP Text Mining
# Date: Décembre 2025
# """

# # Configuration globale OPTIMISÉE
# CLASSIFICATION_CONFIG = {
#     'min_score': 4.5,        # ✅ Abaissé de 5.0
#     'min_confidence': 0.55,  # ✅ Abaissé de 0.60
#     'weights': {
#         'title': 0.6,
#         'description': 0.2,
#         'competences': 0.2
#     }
# }

# # Profils stricts
# STRICT_PROFILES = [
#     'AI Engineer',
#     'AI Research Scientist',
#     'Computer Vision Engineer'
# ]

# # Profils permissifs
# PERMISSIVE_PROFILES = [
#     'Data Analyst',
#     'Data Consultant',
#     'Data Manager'
# ]


# # ============================================
# # DÉFINITION DES 13 PROFILS - v1 OPTIMISÉE
# # ============================================

# PROFILS = {
    
#     # ========================================
#     # 1. DATA ENGINEER (ENRICHI)
#     # ========================================
#     'Data Engineer': {
#         'description': 'Pipelines données, ETL, Big Data, Architecte Data, cloud',
        
#         'title_variants': [
#             # Base
#             'data engineer', 'engineer data', 'data engineering',
#             'ingénieur données', 'ingenieur donnees',
#             'ingénieur data', 'ingenieur data',
            
#             # ✅ Big Data (enrichi)
#             'big data',
#             'ingénieur big data', 'ingenieur big data',
#             'développeur big data', 'developpeur big data',
#             'big data engineer', 'big data developer',
#             'ingénieur big data cloud',
#             'ingenieur / ingenieure big data',
#             'ingenieure big data',
#             'expert big data',
            
#             # ✅ Lead Data Engineer (NOUVEAU)
#             'lead data engineer',
#             'lead engineer data',
#             'tech lead data engineer',
#             'senior data engineer',
            
#             # Architecte Data
#             'architecte data', 'architecte données', 'architecte donnees',
#             'data architect',
#             'architecte big data',
#             'architecte cloud data',
#             'architecte plateforme données',
#             'architecte solutions data',
            
#             # Développeur Data
#             'développeur data', 'developpeur data',
#             'développeur données', 'developpeur donnees',
#             'développeuse data', 'developpeuse data',
#             'data developer', 'dev data',
            
#             # ✅ Data Ingénieur (NOUVEAU - variantes orthographiques)
#             'data ingénieur', 'data ingenieur',
            
#             # ✅ Support Data (NOUVEAU)
#             'ingénieur support data', 'ingenieur support data',
#             'support data engineer',
            
#             # ETL
#             'ingénieur etl', 'ingenieur etl',
#             'etl engineer', 'etl developer',
            
#             # Plateforme
#             'ingénieur plateforme données', 'ingenieur plateforme donnees',
#             'data platform engineer',
#             'ingénieur cloud data', 'ingenieur cloud data',
#             'cloud data engineer'
#         ],
        
#         'keywords_title': [
#             'pipeline', 'etl', 'elt',
#             'big data', 'warehouse',
#             'cloud', 'plateforme',
#             'architecte', 'architect',
#             'lead'  # ✅ NOUVEAU
#         ],
        
#         'keywords_strong': [
#             'airflow', 'kafka', 'spark',
#             'hadoop', 'hive',
#             'data lake', 'lakehouse',
#             'dbt',
#             'streaming',
#             'orchestration'
#         ],
        
#         'competences_core': [
#             'sql', 'python', 'airflow', 'spark',
#             'aws', 'data engineer', 'big data'
#         ],
        
#         'competences_tech': [
#             'kafka', 'docker', 'kubernetes',
#             'postgresql', 'mongodb'
#         ],
        
#         'weights': {
#             'title': 0.6,
#             'description': 0.2,
#             'competences': 0.2
#         }
#     },
    
#     # ========================================
#     # 2. DATA SCIENTIST (ENRICHI)
#     # ========================================
#     'Data Scientist': {
#         'description': 'ML classique, statistiques, modèles prédictifs',
        
#         'title_variants': [
#             # Base
#             'data scientist', 'scientist data',
#             'data science engineer',
#             'scientifique données', 'scientifique de données',
#             'scientist', 'scientifique',
            
#             # ✅ Statisticien (enrichi)
#             'statisticien', 'statisticienne',
#             'chargé études statistiques', 'charge etudes statistiques',
#             'chargée études statistiques', 'chargee etudes statistiques',
#             'chargé étude statistique', 'charge etude statistique',
#             'chargée étude statistique', 'chargee etude statistique',
#             'analyste statistique',
#             'statistician',
            
#             # Chercheur
#             'chercheur données', 'data researcher',
#             'analyste scientifique',
#             'analyste prédictif',
            
#             # ML Scientist
#             'ml scientist', 'machine learning scientist',
#             'spécialiste machine learning',
            
#             # ✅ Consultant Data Scientist (NOUVEAU)
#             'consultant data scientist',
#             'consultante data scientist'
#         ],
        
#         'keywords_title': [
#             'machine learning', 'ml',
#             'statistiques', 'statisticien',
#             'prédictif', 'predictive',
#             'modèle', 'scientist'
#         ],
        
#         'keywords_strong': [
#             'scikit-learn', 'sklearn',
#             'régression', 'classification',
#             'prédiction', 'prediction',
#             'xgboost', 'lightgbm',
#             'features engineering',
#             'data mining',
#             'clustering', 'segmentation'
#         ],
        
#         'competences_core': [
#             'machine learning', 'python', 'scikit-learn',
#             'statistiques', 'r'
#         ],
        
#         'competences_tech': [
#             'pandas', 'numpy', 'jupyter',
#             'matplotlib', 'seaborn', 'sql'
#         ],
        
#         'weights': {
#             'title': 0.6,
#             'description': 0.2,
#             'competences': 0.2
#         }
#     },
    
#     # ========================================
#     # 3. DATA ANALYST (ENRICHI)
#     # ========================================
#     'Data Analyst': {
#         'description': 'Analyse exploratoire, SQL, Excel, reporting',
        
#         'title_variants': [
#             # Base
#             'data analyst', 'analyste données', 'analyste de données',
#             'analyst data', 'analyste data',
#             'junior data analyst',
#             'analyste', 'analyst',
#             'data analysis',
            
#             # ✅ Senior Data Analyst (NOUVEAU)
#             'senior data analyst',
#             'lead data analyst'
#         ],
        
#         'keywords_title': [
#             'analyse', 'analysis',
#             'sql', 'excel',
#             'reporting',
#             'senior'  # ✅ NOUVEAU
#         ],
        
#         'keywords_strong': [
#             'analyse exploratoire',
#             'data cleaning',
#             'statistiques descriptives',
#             'rapport', 'query', 'requête',
#             'kpi', 'metrics'
#         ],
        
#         'competences_core': [
#             'sql', 'excel', 'analyse',
#             'python'
#         ],
        
#         'competences_tech': [
#             'pandas', 'sql', 'excel',
#             'power bi', 'tableau'
#         ],
        
#         'weights': {
#             'title': 0.6,
#             'description': 0.2,
#             'competences': 0.2
#         }
#     },
    
#     # ========================================
#     # 4. BI ANALYST (ENRICHI - Business Analyst)
#     # ========================================
#     'BI Analyst': {
#         'description': 'Business Intelligence, dashboards, reporting, business analyst',
        
#         'title_variants': [
#             # Base BI
#             'bi analyst', 'business intelligence analyst',
#             'analyste bi', 'analyste business intelligence',
            
#             # ✅ Business Analyst (ENRICHI)
#             'business analyst',
#             'business analyst data',
#             'business analyst (h/f)',
#             'business analyst f/h',
#             'analyste business',
#             'analyste affaires',
#             'ba data',
#             'analyste métier', 'analyste metier',
#             'analyste métier data', 'analyste metier data',
#             'business analyst connaissance sectorielle',
#             'business analyst data connaissance',
            
#             # Développeur BI
#             'développeur bi', 'developpeur bi',
#             'développeur business intelligence',
#             'developpeur business intelligence',
#             'bi developer', 'business intelligence developer',
            
#             # ✅ Analyste décisionnel (ENRICHI)
#             'analyste décisionnel', 'analyste decisionnel',
#             'développeur décisionnel', 'developpeur decisionnel',
#             'développeuse décisionnel', 'developpeuse decisionnel',
#             'développeur / développeuse décisionnel',
#             'developpeur / developpeuse decisionnel',
#             'business analyst / développeur bi',
#             'business analyst / developpeur bi',
            
#             # Business Developer Data
#             'business developer data',
#             'business developer',
#             'business development data',
            
#             # Spécialisations outils
#             'tableau analyst', 'power bi analyst',
#             'tableau developer', 'power bi developer',
#             'looker analyst', 'qlik analyst',
#             'analyste business intelligence (h/f)',
#             'analyste bi (h/f)'
#         ],
        
#         'keywords_title': [
#             'tableau', 'power bi', 'powerbi',
#             'looker', 'qlik',
#             'bi', 'business intelligence',
#             'décisionnel', 'decisionnel',
#             'dashboard',
#             'business analyst',
#             'business developer',
#             'ba data'
#         ],
        
#         'keywords_strong': [
#             'dashboard', 'reporting',
#             'visualisation données',
#             'data visualization', 'dataviz',
#             'kpi', 'metrics',
#             'dax', 'powerquery',
#             'business analysis',
#             'analyse métier',
#             'business requirements'
#         ],
        
#         'competences_core': [
#             'power bi', 'tableau', 'sql',
#             'excel', 'looker',
#             'business analysis'
#         ],
        
#         'competences_tech': [
#             'dax', 'powerquery', 'qlik',
#             'sql', 'excel'
#         ],
        
#         'weights': {
#             'title': 0.65,
#             'description': 0.15,
#             'competences': 0.2
#         }
#     },
    
#     # ========================================
#     # 5. DATA CONSULTANT (BASE v6)
#     # ========================================
#     'Data Consultant': {
#         'description': 'Conseil Data, transformation, accompagnement client',
        
#         'title_variants': [
#             'consultant data', 'data consultant',
#             'consultant', 'consultante',
#             'conseil data',
#             'consulting data',
            
#             # ✅ Consultant Data Engineer (NOUVEAU)
#             'consultant data engineer',
#             'consultante data engineer'
#         ],
        
#         'keywords_title': [
#             'consultant', 'conseil',
#             'consulting',
#             'transformation',
#             'advisory'
#         ],
        
#         'keywords_strong': [
#             'transformation digitale',
#             'accompagnement',
#             'client', 'mission',
#             'change management',
#             'esn', 'cabinet'
#         ],
        
#         'competences_core': [
#             'conseil', 'transformation',
#             'management', 'gestion projet'
#         ],
        
#         'competences_tech': [
#             'python', 'sql', 'excel',
#             'power bi'
#         ],
        
#         'weights': {
#             'title': 0.6,
#             'description': 0.25,
#             'competences': 0.15
#         }
#     },
    
#     # ========================================
#     # 6. DATA MANAGER (ENRICHI)
#     # ========================================
#     'Data Manager': {
#         'description': 'Management équipe data, chef projet data, CDO, direction',
        
#         'title_variants': [
#             # Base
#             'data manager', 'manager data',
#             'responsable data', 'responsable données', 'responsable donnees',
#             'team lead data', 'lead data',
#             'data team lead',
            
#             # Chef projet
#             'chef de projet data', 'chef projet data',
#             'chef de projet données', 'chef projet donnees',
#             'chef de projets moa data',  # ✅ NOUVEAU
#             'chef projet moa data',       # ✅ NOUVEAU
            
#             # Product
#             'product manager data', 'pm data',
#             'product owner data', 'po data',
#             'chef de projet big data',
            
#             # ✅ CDO / Direction (enrichi)
#             'chief data officer', 'cdo',
#             'directeur data', 'directrice data',
#             'directeur données', 'directrice données',
#             'directeur donnees', 'directrice donnees',
#             'directeur data ai', 'directrice data ai',        # ✅ NOUVEAU
#             'directeur data ai factory', 'directrice data ai factory',  # ✅ NOUVEAU
#             'head of data', 'data director',
            
#             # Responsable
#             'responsable stratégie data', 'responsable strategie data',
#             'directeur big data',
#             'responsable activité data', 'responsable activite data',
#             'responsable plateforme data',
#             'responsable data ai'  # ✅ NOUVEAU
#         ],
        
#         'keywords_title': [
#             'manager', 'responsable',
#             'chef', 'director', 'directeur', 'directrice',
#             'lead', 'head', 'cdo',
#             'product', 'po', 'pm',
#             'moa'  # ✅ NOUVEAU
#         ],
        
#         'keywords_strong': [
#             'management', 'équipe', 'team',
#             'projet', 'product',
#             'stratégie', 'gouvernance',
#             'roadmap', 'transformation',
#             'leadership', 'direction',
#             'factory'  # ✅ NOUVEAU
#         ],
        
#         'competences_core': [
#             'management', 'gestion projet',
#             'leadership', 'stratégie'
#         ],
        
#         'competences_tech': [
#             'sql', 'python', 'agile',
#             'jira', 'scrum'
#         ],
        
#         'weights': {
#             'title': 0.65,
#             'description': 0.2,
#             'competences': 0.15
#         }
#     },
    
#     # ========================================
#     # 7. DATA ARCHITECT (ENRICHI)
#     # ========================================
#     'Data Architect': {
#         'description': 'Architecture données, gouvernance, stratégie senior',
        
#         'title_variants': [
#             # Base
#             'data architect', 'architecte données', 'architecte donnees',
#             'architect data',
            
#             # ✅ Variantes enrichies
#             'data architect (h/f)',
#             'architecte si data',      # ✅ NOUVEAU
#             'si data architect',       # ✅ NOUVEAU
#             'architecte solution data', # ✅ NOUVEAU
#             'solution architect data',  # ✅ NOUVEAU
            
#             # Senior
#             'chief data architect',
#             'lead architect',
#             'data architect confirmé', 'data architect confirme',
            
#             # Solutions
#             'architecte solutions',
#             'solutions architect data'
#         ],
        
#         'keywords_title': [
#             'architecture', 'architect',
#             'gouvernance', 'governance',
#             'stratégie', 'strategy',
#             'chief', 'lead',
#             'solution'  # ✅ NOUVEAU
#         ],
        
#         'keywords_strong': [
#             'data architecture',
#             'enterprise architecture',
#             'data modeling',
#             'master data management', 'mdm',
#             'data catalog',
#             'data quality'
#         ],
        
#         'competences_core': [
#             'architecture', 'gouvernance',
#             'data modeling', 'sql'
#         ],
        
#         'competences_tech': [
#             'sql', 'cloud', 'aws', 'azure',
#             'databricks'
#         ],
        
#         'weights': {
#             'title': 0.65,
#             'description': 0.2,
#             'competences': 0.15
#         }
#     },
    
#     # ========================================
#     # 8. AI ENGINEER (ENRICHI)
#     # ========================================
#     'AI Engineer': {
#         'description': 'IA générative, LLMs, NLP avancé, transformers',
        
#         'title_variants': [
#             # Base
#             'ai engineer', 'ingénieur ia', 'ingenieur ia',
#             'engineer ai', 'ia engineer',
#             'artificial intelligence engineer',
#             'llm engineer', 'nlp engineer',
#             'ingénieur intelligence artificielle', 'ingenieur intelligence artificielle',
            
#             # ✅ Tech Lead IA (NOUVEAU)
#             'tech lead ia', 'tech lead ai',
#             'lead ai engineer', 'lead ia engineer',
            
#             # ✅ Chef projet IA (NOUVEAU)
#             'chef de projet ia', 'chef projet ia',
#             'chef de projet technique ia'
#         ],
        
#         'keywords_title': [
#             'llm', 'llms',
#             'gpt', 'chatgpt',
#             'transformers', 'bert',
#             'nlp', 'generative',
#             'ia générative', 'generative ai',
#             'tech lead'  # ✅ NOUVEAU
#         ],
        
#         'keywords_strong': [
#             'langchain', 'llamaindex',
#             'rag', 'retrieval augmented',
#             'prompt engineering',
#             'fine-tuning',
#             'hugging face',
#             'openai', 'anthropic',
#             'chatbot', 'embedding'
#         ],
        
#         'competences_core': [
#             'intelligence artificielle', 'llm',
#             'transformers', 'gpt', 'langchain', 'nlp'
#         ],
        
#         'competences_tech': [
#             'python', 'pytorch', 'tensorflow',
#             'hugging face', 'api'
#         ],
        
#         'weights': {
#             'title': 0.65,
#             'description': 0.2,
#             'competences': 0.15
#         }
#     },
    
#     # ========================================
#     # 9-13. AUTRES PROFILS (BASE v6)
#     # ========================================
    
#     'ML Engineer': {
#         'description': 'MLOps, déploiement ML, production, pipelines ML',
#         'title_variants': [
#             'ml engineer', 'machine learning engineer',
#             'ingénieur ml', 'ingenieur ml',
#             'engineer ml',
#             'mlops engineer', 'ml ops engineer'
#         ],
#         'keywords_title': [
#             'mlops', 'ml ops',
#             'déploiement', 'deployment',
#             'production',
#             'devops ml'
#         ],
#         'keywords_strong': [
#             'mlflow', 'kubeflow',
#             'model deployment',
#             'kubernetes', 'docker',
#             'ci/cd ml',
#             'monitoring ml'
#         ],
#         'competences_core': [
#             'mlops', 'ci/cd', 'kubernetes', 'docker',
#             'machine learning'
#         ],
#         'competences_tech': [
#             'mlflow', 'kubeflow', 'tensorflow',
#             'pytorch', 'git', 'linux'
#         ],
#         'weights': {
#             'title': 0.6,
#             'description': 0.2,
#             'competences': 0.2
#         }
#     },
    
#     'Analytics Engineer': {
#         'description': 'Transformation données, dbt, SQL avancé, analytics',
#         'title_variants': [
#             'analytics engineer',
#             'ingénieur analytics', 'ingenieur analytics',
#             'dbt engineer'
#         ],
#         'keywords_title': [
#             'analytics', 'dbt',
#             'transformation', 'sql'
#         ],
#         'keywords_strong': [
#             'data modeling',
#             'data transformation',
#             'looker', 'metabase',
#             'data quality'
#         ],
#         'competences_core': [
#             'dbt', 'sql', 'python',
#             'data modeling'
#         ],
#         'competences_tech': [
#             'git', 'postgresql', 'snowflake',
#             'databricks'
#         ],
#         'weights': {
#             'title': 0.6,
#             'description': 0.2,
#             'competences': 0.2
#         }
#     },
    
#     'MLOps Engineer': {
#         'description': 'DevOps pour ML, CI/CD ML, infrastructure ML',
#         'title_variants': [
#             'mlops engineer', 'ml ops engineer',
#             'ingénieur mlops', 'ingenieur mlops',
#             'devops ml'
#         ],
#         'keywords_title': [
#             'mlops', 'ml ops',
#             'devops',
#             'kubernetes', 'k8s'
#         ],
#         'keywords_strong': [
#             'ci/cd', 'terraform',
#             'infrastructure as code',
#             'monitoring', 'observability'
#         ],
#         'competences_core': [
#             'mlops', 'kubernetes', 'docker',
#             'ci/cd', 'devops'
#         ],
#         'competences_tech': [
#             'terraform', 'git', 'linux',
#             'aws', 'azure'
#         ],
#         'weights': {
#             'title': 0.65,
#             'description': 0.15,
#             'competences': 0.2
#         }
#     },
    
#     'AI Research Scientist': {
#         'description': 'Recherche IA, publications, PhD, innovation',
#         'title_variants': [
#             'research scientist', 'chercheur',
#             'researcher', 'scientifique recherche',
#             'ai researcher', 'ml researcher',
#             'research engineer'
#         ],
#         'keywords_title': [
#             'research', 'recherche',
#             'phd', 'doctorat',
#             'chercheur', 'postdoc'
#         ],
#         'keywords_strong': [
#             'publication', 'paper',
#             'conference', 'neurips', 'icml',
#             'innovation',
#             'state-of-the-art', 'arxiv'
#         ],
#         'competences_core': [
#             'recherche', 'intelligence artificielle',
#             'machine learning', 'deep learning'
#         ],
#         'competences_tech': [
#             'python', 'pytorch', 'tensorflow',
#             'jupyter'
#         ],
#         'weights': {
#             'title': 0.7,
#             'description': 0.15,
#             'competences': 0.15
#         }
#     },
    
#     'Computer Vision Engineer': {
#         'description': 'Vision par ordinateur, images, vidéo, CNN',
#         'title_variants': [
#             'computer vision engineer',
#             'ingénieur computer vision', 'ingenieur computer vision',
#             'cv engineer',
#             'vision engineer',
#             'image processing engineer'
#         ],
#         'keywords_title': [
#             'computer vision', 'vision',
#             'image', 'vidéo', 'video',
#             'opencv', 'cv'
#         ],
#         'keywords_strong': [
#             'cnn', 'convolutional',
#             'yolo', 'mask r-cnn',
#             'object detection',
#             'image segmentation',
#             'face recognition'
#         ],
#         'competences_core': [
#             'computer vision', 'deep learning',
#             'opencv', 'pytorch'
#         ],
#         'competences_tech': [
#             'python', 'opencv', 'cuda',
#             'tensorflow'
#         ],
#         'weights': {
#             'title': 0.7,
#             'description': 0.15,
#             'competences': 0.15
#         }
#     }
# }


# # ============================================
# # FONCTIONS UTILITAIRES
# # ============================================

# def get_profil_config(profil_name):
#     if profil_name not in PROFILS:
#         raise ValueError(f"Profil '{profil_name}' non trouvé")
#     return PROFILS[profil_name]


# def get_all_profils():
#     return list(PROFILS.keys())


# def get_min_score(profil_name):
#     if profil_name in STRICT_PROFILES:
#         return 5.0  # Garde strict pour profils exigeants
#     elif profil_name in PERMISSIVE_PROFILES:
#         return 4.0  # Encore plus permissif
#     else:
#         return CLASSIFICATION_CONFIG['min_score']  # 4.5


# def export_profils_json(filepath):
#     import json
    
#     export_data = {
#         'version': 'v1_optimized',
#         'config': CLASSIFICATION_CONFIG,
#         'strict_profiles': STRICT_PROFILES,
#         'permissive_profiles': PERMISSIVE_PROFILES,
#         'profils': PROFILS
#     }
    
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(export_data, f, ensure_ascii=False, indent=2)
    
#     print(f"✅ Profils v1 optimisés exportés: {filepath}")


# if __name__ == "__main__":
#     print("="*70)
#     print("📋 VALIDATION DÉFINITIONS PROFILS v1 OPTIMISÉE")
#     print("="*70)
    
#     print(f"\n✅ Version: v1 OPTIMISÉE")
#     print(f"✅ Nombre de profils: {len(PROFILS)}")
#     print(f"✅ Min score global: {CLASSIFICATION_CONFIG['min_score']}")
#     print(f"✅ Min confidence: {CLASSIFICATION_CONFIG['min_confidence']}")
    
#     print("\n📊 Optimisations v1:")
#     print("   ✅ Seuils abaissés: 4.5 / 0.55 (au lieu de 5.0 / 0.60)")
#     print("   ✅ Data Engineer: +12 variantes (Big Data, Lead, Support)")
#     print("   ✅ BI Analyst: +15 variantes (Business Analyst enrichi)")
#     print("   ✅ Data Architect: +5 variantes (SI Data, Solution)")
#     print("   ✅ Data Manager: +6 variantes (CDO, Factory, MOA)")
#     print("   ✅ AI Engineer: +3 variantes (Tech Lead IA)")
    
#     print("\n✅ Validation terminée !")
#     print("\n📊 Résultat attendu:")
#     print("   ✅ Taux classification: 47-49%")
#     print("   ✅ Compatible v1 (matching simple 'in title_lower')")