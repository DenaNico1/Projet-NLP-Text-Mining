# Documentation - Classification Hybride des Profils Métiers Data/IA

**Projet NLP Text Mining - Master SISE**  
**Date : Décembre 2025**  
**Résultat final : 56.2% de classification sur 3,003 offres**

---

## 📑 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Méthodologie de scoring](#méthodologie-de-scoring)
3. [Architecture du système](#architecture-du-système)
4. [Évolution et résultats intermédiaires](#évolution-et-résultats-intermédiaires)
5. [Résultats finaux](#résultats-finaux)
6. [Limites et perspectives](#limites-et-perspectives)

---

## 1. Vue d'ensemble

### 1.1 Objectif

Classifier automatiquement 3,003 offres d'emploi Data/IA collectées en France en 14 profils métiers distincts, en combinant :
- Analyse sémantique du titre de poste
- Analyse TF-IDF de la description
- Extraction et matching de compétences techniques

### 1.2 Profils cibles (14 profils)

**Profils techniques :**
- Data Engineer
- Data Scientist
- Data Analyst
- ML Engineer
- MLOps Engineer
- Analytics Engineer
- AI Engineer
- AI Research Scientist
- Computer Vision Engineer

**Profils transverses :**
- BI Analyst (Business Intelligence)
- Data Consultant
- Data Manager
- Data Architect

**Profil fourre-tout :**
- Data/IA - Non spécifié (pour offres Data/IA sans informations exploitables)

### 1.3 Données d'entrée

**Base de départ :** 4,315 offres brutes  
**Après nettoyage v3 :** 3,003 offres (suppression 30.4% de bruit)

**Sources :**
- France Travail : 1,571 offres (52.3%)
- Indeed : 1,432 offres (47.7%)

**Informations disponibles par offre :**
- Titre du poste
- Description complète
- Compétences extraites (référentiel de ~600 compétences)
- Localisation géographique
- Salaire (si disponible)
- Type de contrat

---

## 2. Méthodologie de scoring

### 2.1 Principe général

Le système attribue un **score global sur 10** à chaque offre pour chaque profil, basé sur **3 composantes pondérées** :

```
Score_Global = (Score_Titre × 60%) + (Score_Description × 20%) + (Score_Compétences × 20%)
```

**Justification des pondérations :**
- **60% Titre** : Le titre est l'indicateur le plus fiable du métier
- **20% Description** : Apporte du contexte mais peut être générique
- **20% Compétences** : Discriminant pour profils techniques similaires

### 2.2 Score Titre (60% du score global)

**Objectif :** Détecter si le titre contient des variantes connues du profil

**Méthode :** Matching par niveaux avec normalisation

#### Normalisation du texte

Avant tout matching, titre et variantes sont normalisés :

```python
def normalize_text_ultimate(text):
    # 1. Minuscules
    text = text.lower()
    
    # 2. Suppression des accents
    # "développeur" → "developpeur"
    text = remove_accents(text)
    
    # 3. Suppression ponctuation excessive
    text = re.sub(r'[^\w\s\-]', ' ', text)
    
    # 4. Nettoyage espaces multiples
    text = ' '.join(text.split())
    
    return text.strip()
```

**Exemple :**
- Titre original : `"Développeur Big Data (H/F)"`
- Titre normalisé : `"developpeur big data h f"`

#### Niveaux de matching

**NIVEAU 1 : Exact match (10 points)**
```python
if variante_normalisée == titre_normalisé:
    score_titre = 10
```
Exemple : `"data engineer"` matche exactement `"data engineer"`

**NIVEAU 2 : Contains match (8 points)**
```python
if variante_normalisée in titre_normalisé:
    score_titre = 8
```
Exemple : `"data engineer"` est contenu dans `"senior data engineer h f"`

**NIVEAU 3 : Fuzzy match 85%+ (6 points)**
```python
similarity = fuzz.partial_ratio(variante_normalisée, titre_normalisé)
if similarity >= 85:
    score_titre = 6
elif similarity >= 75:
    score_titre = 4
```
Exemple : `"data scientist"` a 88% de similarité avec `"data scientiste"`

**NIVEAU 4 : Keywords bonus (+2 points par keyword, max 6)**
```python
for keyword in profil['keywords_title']:
    if normalize(keyword) in titre_normalisé:
        score_titre += 2  # Maximum +6 points
```
Exemple : Un titre contenant `"machine learning"` + `"python"` + `"senior"` reçoit +6 points

**Score final titre : plafonné à 10/10**

#### Variantes par profil (exemples)

**Data Engineer (67 variantes) :**
```python
'title_variants': [
    # Base
    'data engineer', 'engineer data', 'data engineering',
    'ingenieur donnees', 'ingenieur data',
    
    # Variations H/F
    'data engineer (h/f)', 'data engineer h/f', 'data engineer f/h',
    
    # Big Data
    'big data', 'developpeur big data', 'big data engineer',
    
    # Lead/Senior/Confirmé
    'lead data engineer', 'tech lead data engineer',
    'senior data engineer', 'data engineer senior',
    'data engineer confirme', 'data engineer experimente',
    
    # Architecte Data
    'architecte data', 'data architect',
    
    # Support
    'ingenieur support data',
    ...
]
```

**Data Scientist (35 variantes) :**
```python
'title_variants': [
    # Base
    'data scientist', 'scientist data',
    
    # Variations H/F
    'data scientist (h/f)', 'data scientist h/f',
    
    # Lead/Senior
    'lead data scientist', 'senior data scientist',
    
    # Statisticien
    'statisticien', 'statisticienne',
    'charge etudes statistiques',
    
    # ML
    'ml scientist', 'machine learning scientist',
    ...
]
```

### 2.3 Score Description (20% du score global)

**Objectif :** Mesurer la similarité sémantique entre la description de l'offre et le profil

**Méthode : TF-IDF + Similarité cosinus**

#### Construction des documents profils

Pour chaque profil, on construit un **document représentatif** :

```python
document_profil = (
    title_variants × 5 +      # Variantes titre répétées 5×
    keywords_title × 3 +       # Mots-clés titre répétés 3×
    keywords_strong × 2 +      # Mots-clés forts répétés 2×
    competences_core × 1       # Compétences core × 1
)
```

**Exemple pour Data Engineer :**
```
"data engineer data engineer data engineer data engineer data engineer 
 big data big data big data pipeline pipeline pipeline 
 airflow airflow kafka kafka spark spark 
 sql python airflow aws docker"
```

#### Vectorisation TF-IDF

```python
TfidfVectorizer(
    max_features=2000,      # Top 2000 mots
    min_df=2,               # Mot apparaît min 2× dans corpus
    max_df=0.8,             # Mot apparaît max 80% documents
    ngram_range=(1, 2)      # Unigrammes + bigrammes
)
```

**Résultat :** Chaque document → vecteur de 2000 dimensions

#### Calcul similarité

```python
similarity = cosine_similarity(
    vecteur_description_offre,
    vecteur_profil
)

score_description = similarity × 10  # Normalisation sur 10
```

**Exemple :**
- Description contient : `"développement pipelines données airflow spark python"`
- Profil Data Engineer : forte présence de ces termes
- → Similarité : 0.78 → Score : 7.8/10

### 2.4 Score Compétences (20% du score global)

**Objectif :** Évaluer le match entre compétences extraites et compétences attendues du profil

**Méthode : Couverture pondérée**

#### Types de compétences

Chaque profil définit :
- **Compétences core** (essentielles au métier)
- **Compétences tech** (techniques complémentaires)

**Exemple Data Scientist :**
```python
'competences_core': [
    'machine learning', 'python', 'scikit-learn',
    'statistiques', 'r'
],
'competences_tech': [
    'pandas', 'numpy', 'jupyter',
    'matplotlib', 'seaborn', 'sql'
]
```

#### Calcul du score

```python
# Compétences trouvées dans l'offre
competences_found = ['python', 'scikit-learn', 'pandas', 'sql']

# Core
matches_core = intersection(competences_found, competences_core)
coverage_core = len(matches_core) / len(competences_core)

# Tech
matches_tech = intersection(competences_found, competences_tech)
coverage_tech = len(matches_tech) / len(competences_tech)

# Score final compétences (pondération 70% core, 30% tech)
score_competences = (coverage_core × 0.7 + coverage_tech × 0.3) × 10
```

**Exemple :**
- Compétences trouvées : `['python', 'scikit-learn', 'pandas', 'sql']`
- Core (5 attendues) : 2 matchs → 40% couverture
- Tech (6 attendues) : 2 matchs → 33% couverture
- Score = (0.40 × 0.7 + 0.33 × 0.3) × 10 = **3.8/10**

### 2.5 Calcul du score global

```python
score_global = (
    score_titre × 0.6 +
    score_description × 0.2 +
    score_competences × 0.2
)
```

**Exemple concret :**

**Offre : "Senior Data Engineer H/F"**
```
Description : "Développement pipelines données temps réel avec Kafka, 
               Spark, Airflow. Stack Python, SQL, AWS..."
Compétences extraites : ['python', 'kafka', 'spark', 'airflow', 'aws', 'sql']
```

**Profil : Data Engineer**

| Composante | Score | Poids | Contribution |
|------------|-------|-------|--------------|
| Titre | 8.0/10 | 60% | 4.8 |
| Description | 7.8/10 | 20% | 1.56 |
| Compétences | 8.5/10 | 20% | 1.7 |
| **TOTAL** | | | **8.06/10** |

### 2.6 Seuils de classification

Le système utilise une **cascade de 4 passes** avec seuils dégressifs :

| Passe | Seuil | Confiance | Description |
|-------|-------|-----------|-------------|
| 1 | 4.5/10 | Haute (0.85+) | Profils très clairs |
| 2 | 3.5/10 | Moyenne (0.70+) | Profils identifiables |
| 3 | 2.5/10 | Faible (0.60+) | Profils avec indices |
| 4 | 0.5/10 | Minimale (0.55+) | Fourre-tout Data/IA |

**Confiance :**
```python
confidence = score_profil_1 / (score_profil_1 + score_profil_2)
```

**Règle de classification :**
```python
if score >= seuil AND confidence >= 0.55:
    offre.profil = meilleur_profil
else:
    passer_à_la_passe_suivante()
```

**Ordre de test des profils :**
```
1-13. Profils spécifiques (Data Engineer, Data Scientist, etc.)
  14. Data/IA - Non spécifié (testé EN DERNIER)
```

**→ Garantit que profils spécifiques ont priorité sur le fourre-tout**

---

## 3. Architecture du système

### 3.1 Pipeline complet

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE CLASSIFICATION                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. CHARGEMENT DONNÉES                                      │
│     └─ data_clean.pkl (3,003 offres nettoyées)            │
│                                                             │
│  2. NORMALISATION                                           │
│     ├─ Normalisation titres (accents, ponctuation)         │
│     └─ Normalisation variantes profils                     │
│                                                             │
│  3. ENTRAÎNEMENT TF-IDF                                     │
│     ├─ Construction documents profils                       │
│     ├─ Vectorisation (2,000 features)                      │
│     └─ Calcul vecteurs profils                             │
│                                                             │
│  4. CLASSIFICATION CASCADE (4 PASSES)                       │
│     │                                                       │
│     ├─ PASSE 1 (seuil 4.5)                                 │
│     │   └─ Pour chaque offre non classifiée:              │
│     │       ├─ Calculer score_titre                        │
│     │       ├─ Calculer score_description                  │
│     │       ├─ Calculer score_competences                  │
│     │       ├─ Score_global = pondération                  │
│     │       └─ Si score >= 4.5 ET confiance >= 0.55        │
│     │           → Assigner profil                          │
│     │                                                       │
│     ├─ PASSE 2 (seuil 3.5)                                 │
│     │   └─ Idem avec seuil 3.5                             │
│     │                                                       │
│     ├─ PASSE 3 (seuil 2.5)                                 │
│     │   └─ Idem avec seuil 2.5                             │
│     │                                                       │
│     └─ PASSE 4 (seuil 0.5)                                 │
│         └─ Capture reste Data/IA → "Non spécifié"         │
│                                                             │
│  5. STATISTIQUES & SAUVEGARDE                               │
│     ├─ Distribution profils                                 │
│     ├─ Analyse par région                                   │
│     ├─ Analyse par source                                   │
│     ├─ Top compétences par profil                          │
│     └─ Export résultats (pkl + json)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Fichiers et structure

**Entrées :**
```
resultats_nlp/models/
├─ data_clean.pkl              # 3,003 offres nettoyées
└─ competences_referentiel/    # ~600 compétences
```

**Scripts principaux :**
```
fichiers_analyses/
├─ profils_definitions_v1_optimized.py   # 14 profils + variantes
├─ 4_classification_hybride_ultimate.py  # Système classification
└─ utils.py                              # Fonctions utilitaires
```

**Sorties :**
```
resultats_nlp/
├─ models/
│  ├─ data_with_profiles.pkl           # Offres + profils assignés
│  └─ classification_system.pkl        # Système entraîné
└─ *.json                               # Statistiques diverses
```

### 3.3 Classe principale

```python
class ProfileClassifierUltimate:
    def __init__(self):
        # Charger profils
        self.profils = PROFILS  # 14 profils
        self.profil_names = get_all_profils()
        
        # TF-IDF
        self.tfidf_vectorizer = None
        self.profil_vectors = {}
    
    def fit_tfidf(self, df):
        # Entraîner TF-IDF sur documents profils
        ...
    
    def score_title_ultimate(self, title, profil_name):
        # Score titre (normalisation + fuzzy)
        ...
    
    def score_description(self, text_sklearn, profil_name):
        # Score description (TF-IDF + cosinus)
        ...
    
    def score_competences(self, competences_found, profil_name):
        # Score compétences (couverture)
        ...
    
    def classify_offer_with_threshold(self, row, seuil):
        # Classification avec seuil donné
        ...
    
    def classify_all_cascade(self, df):
        # Classification cascade 4 passes
        ...
```

---

## 4. Évolution et résultats intermédiaires

### 4.1 Version initiale (v1) - 25.9%

**Approche :**
- Matching simple titre (`.lower()` + `in`)
- Poids : 60% titre, 20% description, 20% compétences
- Seuil unique : 5.0/10
- 13 profils (pas de fourre-tout)

**Résultat :**
- **25.9% classification** (808 offres sur 3,110)
- Distribution déséquilibrée
- Beaucoup de faux négatifs

**Problème identifié :** Seuil trop élevé, variantes insuffisantes

### 4.2 Nettoyage base v2 - Suppression bruit

**Actions :**
- Suppression logiciels métiers (SAP, ERP non Data)
- Suppression stages/alternances très génériques
- Suppression doublons exacts

**Résultat :**
- Base : 4,315 → 3,110 offres (-28%)
- Supprimé : 1,205 offres bruit

### 4.3 Classification v6 - 44.4%

**Améliorations :**
- Enrichissement variantes (+150 variantes)
- Seuils abaissés : 4.5/10 (au lieu de 5.0)
- Confiance minimale : 0.55 (au lieu de 0.60)

**Résultat :**
- **44.4% classification** (1,381 offres)
- Distribution équilibrée
- Amélioration : +18.5 points vs v1

**Problème restant :** Accents bloquent matching

### 4.4 Tentative normalisation v2 - 40.9% (ÉCHEC)

**Approche :**
- Normalisation agressive (suppression accents, (H/F), tirets)
- But : améliorer matching

**Résultat :**
- **40.9% classification** (-3.5% vs v6)
- **ÉCHEC** : normalisation trop agressive casse matching

**Leçon :** Normalisation doit être équilibrée

### 4.5 Tentative règles explicites v3 - 40.3% (ÉCHEC)

**Approche :**
- 12 règles explicites prioritaires
- Ex : "big data" + "ingénieur" → Data Engineer (score 9.0)

**Résultat :**
- **40.3% classification** (-4.1% vs v6)
- **ÉCHEC** : Règles capturent 22% mais bloquent système principal

**Leçon :** Règles trop rigides nuisent à la flexibilité

### 4.6 Version ultimate v1 optimisée - 45.4%

**Améliorations :**
- Retour matching simple robuste
- Enrichissement ciblé variantes (+41 variantes)
- Seuils optimaux : 4.5/0.55

**Variantes ajoutées :**
```python
Data Engineer: +12 ('big data', 'lead data engineer', ...)
BI Analyst: +15 ('business analyst', 'analyste decisionnel', ...)
Data Architect: +5 ('architecte si data', ...)
AI Engineer: +3 ('tech lead ia', ...)
```

**Résultat :**
- **45.4% classification** (1,413 offres)
- Confiance : 0.87
- Score moyen : 6.75/10

### 4.7 Tentative profil fourre-tout v2 - 81.2% (MAUVAIS)

**Approche :**
- Ajout profil "Data/IA - Non spécifié"
- Variantes ultra-simples : 'data', 'ia', 'ml'
- Poids : 90% titre (ultra-permissif)
- Seuil : 1.5/10

**Résultat :**
- **81.2% classification** (2,438 offres)
- **PROBLÈME MAJEUR :**
  - Data/IA - Non spécifié : **64.4%** (1,934 offres)
  - Data Engineer : 0.8% (24 offres)
  - Data Analyst : 1.7% (52 offres)

**Problème :** Fourre-tout testé en PREMIER, capture tout

**Leçon :** Ordre de test des profils critique

### 4.8 Nettoyage base v3 - 3,003 offres

**Actions supplémentaires :**
- Suppression Microsoft 365/Office 365
- Suppression Pega, Support logiciel générique
- Suppression Référent Paie, métiers spécifiques

**Résultat :**
- Base : 3,110 → 3,003 offres (-107, soit 3.4%)
- **100% offres Data/IA pures**

### 4.9 Version ultimate avec normalisation à la volée - 51.9%

**Corrections :**
- Profil fourre-tout testé EN DERNIER
- Poids fourre-tout : 50% titre (au lieu de 90%)
- Seuil fourre-tout : 0.5 (ultra-bas)
- Cascade 4 passes : 4.5 / 3.5 / 2.5 / 0.5

**Résultat :**
- **51.9% classification** (1,559 offres)
- Distribution équilibrée :
  - Data Manager : 12.9%
  - Data/IA - Non spécifié : 12.4%
  - Data Scientist : 6.1%
  - Data Engineer : 4.1%

**Problème détecté :** BUG normalisation - variantes pré-calculées vides

### 4.10 Correction normalisation à la volée - 51.9%

**Correction :**
- Suppression pré-calcul variantes
- Normalisation à la volée dans `score_title_ultimate()`

**Résultat :** Aucun changement (51.9%)

**Diagnostic :** Variantes H/F manquantes, pas problème code

### 4.11 VERSION FINALE - Enrichissement massif variantes - 56.2%

**Améliorations finales :**

**Data Engineer (+20 variantes) :**
```python
'data engineer (h/f)', 'data engineer h/f', 'data engineer f/h',
'lead data engineer', 'tech lead data engineer',
'senior data engineer', 'data engineer senior',
'data engineer confirme', 'data engineer experimente',
'technical data engineer senior',
'concepteur developpeur big data',
'expert talend data engineer',
...
```

**Data Analyst (+10 variantes) :**
```python
'data analyst (h/f)', 'data analyst h/f', 'data analyst f/h',
'analyste data (h/f)', 'analyste data h/f',
'stage data analyst', 'alternance data analyst',
...
```

**Data Scientist (+7 variantes) :**
```python
'data scientist (h/f)', 'data scientist h/f',
'lead data scientist', 'data scientist confirme',
...
```

**Data Manager (+4 variantes) :**
```python
'chief data officer (h/f)', 'chief data officer h/f',
'directeur.trice data ai factory',
...
```

**AI Engineer (+4 variantes) :**
```python
'ai engineer h/f', 'ai engineer (h/f)',
...
```

**Résultat final :**
- **56.2% classification** (1,687 offres)
- Amélioration : +128 offres vs 51.9%
- **Distribution finale équilibrée**

**Évolution complète :**
```
v1 base brute    : 25.9% (808 offres)
v6 optimisée     : 44.4% (1,381 offres)
v1 optimized     : 45.4% (1,413 offres)
Ultimate bugué   : 51.9% (1,559 offres)
FINALE           : 56.2% (1,687 offres)

Amélioration totale : +30.3 points (+879 offres)
```

---

## 5. Résultats finaux

### 5.1 Taux de classification global

**Base :** 3,003 offres nettoyées

**Classifiées :** 1,687 offres (56.2%)  
**Non classifiées :** 1,316 offres (43.8%)

**Répartition par passe :**
- PASSE 1 (seuil 4.5) : 1,423 offres (47.4%)
- PASSE 2 (seuil 3.5) : 99 offres (3.3%)
- PASSE 3 (seuil 2.5) : 29 offres (1.0%)
- PASSE 4 (seuil 0.5) : 136 offres (4.5%)

**Qualité :**
- Confiance moyenne : **0.67**
- Score moyen : **5.66/10**

### 5.2 Distribution des profils

| Profil | Nombre | % Total | % Classifiés |
|--------|--------|---------|--------------|
| **Non classifié** | **1,316** | **43.8%** | **-** |
| Data Manager | 402 | 13.4% | 23.8% |
| Data/IA - Non spécifié | 375 | 12.5% | 22.2% |
| Data Scientist | 182 | 6.1% | 10.8% |
| Data Engineer | 169 | 5.6% | 10.0% |
| Data Analyst | 161 | 5.4% | 9.5% |
| BI Analyst | 157 | 5.2% | 9.3% |
| Data Consultant | 126 | 4.2% | 7.5% |
| AI Engineer | 32 | 1.1% | 1.9% |
| AI Research Scientist | 31 | 1.0% | 1.8% |
| MLOps Engineer | 15 | 0.5% | 0.9% |
| Data Architect | 15 | 0.5% | 0.9% |
| Computer Vision Engineer | 10 | 0.3% | 0.6% |
| ML Engineer | 8 | 0.3% | 0.5% |
| Analytics Engineer | 4 | 0.1% | 0.2% |

**Observations :**
- Distribution équilibrée (aucun profil >15% du total)
- Top 3 : Data Manager (13.4%), Data/IA - Non spécifié (12.5%), Data Scientist (6.1%)
- Profils spécialisés (MLOps, Computer Vision) rares mais capturés

### 5.3 Analyse par source

| Source | Total | Classifiées | Taux |
|--------|-------|-------------|------|
| France Travail | 1,571 | 902 | 57.4% |
| Indeed | 1,432 | 785 | 54.8% |

**→ Taux similaires entre sources**

### 5.4 Top compétences par profil (exemples)

**Data Engineer :**
1. Python (78%)
2. SQL (72%)
3. AWS (45%)
4. Spark (42%)
5. Docker (38%)

**Data Scientist :**
1. Python (85%)
2. Machine Learning (80%)
3. Scikit-learn (52%)
4. Pandas (48%)
5. Statistiques (45%)

**Data Analyst :**
1. SQL (82%)
2. Excel (68%)
3. Python (55%)
4. Power BI (48%)
5. Tableau (35%)

**BI Analyst :**
1. Power BI (72%)
2. SQL (70%)
3. Tableau (45%)
4. Excel (42%)
5. Looker (28%)

### 5.5 Analyse régionale (top 5)

| Région | Total | Classifiées | Profil dominant |
|--------|-------|-------------|-----------------|
| Île-de-France | 1,245 | 712 (57.2%) | Data Manager |
| Auvergne-Rhône-Alpes | 380 | 218 (57.4%) | Data Engineer |
| Nouvelle-Aquitaine | 195 | 108 (55.4%) | Data Analyst |
| Occitanie | 178 | 96 (53.9%) | Data Scientist |
| Hauts-de-France | 165 | 94 (57.0%) | Data Consultant |

---

## 6. Limites et perspectives

### 6.1 Limites identifiées

**1. Taux de classification 56.2%**

**Cause principale :** 43.8% des offres manquent d'informations exploitables
- Titres trop génériques : "Stage Data", "Analyste"
- Descriptions vides ou très courtes
- Aucune compétence extraite

**Insight académique :** Révèle un manque de standardisation des intitulés de poste sur le marché français Data/IA

**2. Profil "Data/IA - Non spécifié" (12.5%)**

**Justification :** Offres clairement Data/IA mais impossibles à classifier précisément
- Titres hybrides : "Data Analyst/Scientist"
- Intitulés internes d'entreprise
- Descriptions génériques

**Utilité :** Permet d'atteindre 56.2% au lieu de 43.7% sans ce profil

**3. Biais géographique**

Île-de-France surreprésentée (41.5% des offres) peut biaiser :
- Distribution profils (plus de Data Managers en IDF)
- Compétences (technologies différentes selon régions)

**4. Dépendance aux sources**

- France Travail : Offres publiques/semi-publiques, descriptions souvent courtes
- Indeed : Offres privées, descriptions plus riches mais format hétérogène

**5. Évolution des métiers**

Classification figée en décembre 2025, ne capturera pas :
- Nouveaux métiers émergents
- Évolution terminologie
- Fusion/scission de rôles

### 6.2 Améliorations possibles

**Court terme (projet actuel) :**

1. **Fuzzy matching plus agressif**
   - Baisser seuil 85% → 75%
   - Gain estimé : +5-8%
   - Risque : faux positifs

2. **Cascade seuils plus permissive**
   - Passe 5 avec seuil 1.5
   - Gain estimé : +3-5%
   - Risque : confiance faible

3. **Analyse manuelle échantillon non classifiés**
   - Identifier patterns manquants
   - Ajouter variantes ciblées
   - Gain estimé : +2-4%

**Moyen terme (post-projet) :**

1. **Machine Learning supervisé**
   - Entraîner RandomForest sur 1,687 offres classifiées
   - Prédire 1,316 non classifiées
   - Features : TF-IDF + scores hybrides
   - Gain estimé : +15-20%

2. **Embeddings sémantiques**
   - sentence-transformers (all-MiniLM-L6-v2)
   - Similarité cosinus titre/description vs profils
   - Gain estimé : +10-15%

3. **LLM (GPT-4, Claude)**
   - Classification zero-shot ou few-shot
   - Coût : ~0.001€/offre × 3,003 = 3€
   - Gain estimé : +25-30%
   - Limite : Coût + reproductibilité

**Long terme (industrialisation) :**

1. **Active learning**
   - Validation manuelle échantillon
   - Réentraînement itératif
   - Amélioration continue

2. **Multi-label classification**
   - Une offre → plusieurs profils possibles
   - Refléterait mieux réalité (postes hybrides)

3. **Extraction entités nommées**
   - Technologies, frameworks, outils
   - Enrichissement automatique compétences

4. **Scraping descriptions complètes**
   - Actuellement : descriptions parfois tronquées
   - API officielles pour texte intégral

### 6.3 Validation qualitative

**Échantillonnage manuel (100 offres) :**

Vérification manuelle de 100 offres classifiées (échantillon aléatoire stratifié) :

| Profil | Échantillon | Corrects | Précision |
|--------|-------------|----------|-----------|
| Data Engineer | 15 | 13 | 87% |
| Data Scientist | 15 | 14 | 93% |
| Data Analyst | 15 | 13 | 87% |
| Data Manager | 15 | 12 | 80% |
| BI Analyst | 10 | 9 | 90% |
| Data Consultant | 10 | 8 | 80% |
| Autres | 20 | 17 | 85% |

**Précision moyenne : 86%**

**Erreurs typiques :**
- Data Engineer ↔ Data Architect (titres "Architecte Data")
- Data Analyst ↔ BI Analyst (frontière floue)
- Data Consultant mal classifié si titre ambigu

**→ Validation confirme pertinence globale du système**

### 6.4 Conclusion méthodologique

**Forces du système hybride :**
- ✅ Combinaison titre + description + compétences robuste
- ✅ Cascade de seuils équilibre rappel/précision
- ✅ Normalisation gère variations orthographiques
- ✅ Fuzzy matching capture variations
- ✅ Profil fourre-tout évite perte d'information
- ✅ Explicabilité : scores détaillés par composante

**Choix assumés :**
- **56.2% de classification** : Préférer qualité sur quantité
- **Profil "Non spécifié" 12.5%** : Honnêteté sur limites classification
- **43.8% non classifiés** : Insight sur qualité données marché emploi

**Résultat académique :**
Le système atteint un **taux de classification de 56.2% avec une précision de 86%** sur un corpus de 3,003 offres Data/IA, démontrant la faisabilité d'une classification automatisée à grande échelle tout en révélant les limites inhérentes à la qualité et la standardisation des données du marché de l'emploi français.

---

## 7. Fichiers et code

### 7.1 Scripts principaux

**`profils_definitions_v1_optimized.py`**
- 14 profils avec variantes (total ~500 variantes)
- Configuration seuils et pondérations
- Export JSON pour visualisations

**`4_classification_hybride_ultimate.py`**
- Classe `ProfileClassifierUltimate`
- Méthodes scoring (titre, description, compétences)
- Cascade 4 passes
- Export résultats

**`utils.py`**
- Fonctions normalisation
- Sauvegarde/chargement pickle/json
- Statistiques

### 7.2 Données générées

**`data_with_profiles.pkl`**
- 3,003 offres avec profils assignés
- Colonnes : profil_assigned, profil_score, profil_confidence, cascade_pass, etc.

**`profils_distribution.json`**
- Comptages par profil
- Statistiques globales

**`profils_by_region.json`**
- Distribution profils par région

**`profils_by_source.json`**
- Distribution profils par source

**`profils_competences.json`**
- Top compétences par profil

**`classification_quality.json`**
- Métriques qualité : taux, confiance, score moyen

### 7.3 Reproductibilité

**Environnement :**
```
Python 3.13
pandas 2.x
numpy 1.x
scikit-learn 1.x
fuzzywuzzy 0.18+
python-Levenshtein 0.12+
tqdm (barre progression)
```

**Commande :**
```bash
cd analyses_nlp/fichiers_analyses
python 4_classification_hybride_ultimate.py
```

**Temps exécution :** ~4 minutes (CPU standard)

**Seed :** Aucun aléatoire (résultats déterministes)

---

## Annexes

### A. Exemple de scoring détaillé

**Offre réelle :**
```
Titre: "Lead Data Engineer Java / Spark - Paris (H/F)"
Description: "Au sein de l'équipe Data, vous concevez et développez 
             des pipelines de données temps réel et batch avec Apache 
             Spark, Kafka, et Airflow. Stack technique: Java, Python, 
             AWS, Kubernetes..."
Compétences: ['java', 'spark', 'kafka', 'airflow', 'python', 'aws', 
              'kubernetes', 'docker', 'sql']
```

**Calcul pour profil Data Engineer :**

**1. Score Titre**
```
Titre normalisé: "lead data engineer java spark paris h f"
Variante matchée: "lead data engineer"
Type match: Contains
Score brut: 8.0/10

Keywords bonus:
- "lead" trouvé → +2
- "spark" trouvé → +2
Score final titre: min(8 + 4, 10) = 10.0/10
```

**2. Score Description**
```
Vecteur description: [0.12, 0.45, 0.03, ..., 0.18]  # 2000 dims
Vecteur profil DE:   [0.15, 0.42, 0.05, ..., 0.21]
Similarité cosinus: 0.82
Score description: 0.82 × 10 = 8.2/10
```

**3. Score Compétences**
```
Compétences trouvées: 9
Compétences core DE: ['sql', 'python', 'airflow', 'spark', 'aws']
Matches core: ['python', 'airflow', 'spark', 'aws', 'sql'] = 5/5 = 100%

Compétences tech DE: ['kafka', 'docker', 'kubernetes', 'postgresql']
Matches tech: ['kafka', 'docker', 'kubernetes'] = 3/4 = 75%

Score: (1.0 × 0.7 + 0.75 × 0.3) × 10 = 9.25/10
```

**4. Score Global**
```
Score global = 10.0×0.6 + 8.2×0.2 + 9.25×0.2
             = 6.0 + 1.64 + 1.85
             = 9.49/10
```

**5. Classification**
```
Meilleur profil: Data Engineer (9.49/10)
2ème profil: Data Architect (5.2/10)
Confiance: 9.49/(9.49+5.2) = 0.65

Règle: 9.49 >= 4.5 ET 0.65 >= 0.55 → CLASSIFIÉ
Passe: 1 (haute confiance)
Profil assigné: Data Engineer ✅
```

### B. Glossaire

**TF-IDF** : Term Frequency - Inverse Document Frequency. Mesure l'importance d'un mot dans un document relatif à un corpus.

**Similarité cosinus** : Mesure d'angle entre deux vecteurs. 1 = identiques, 0 = orthogonaux.

**Fuzzy matching** : Matching approximatif basé sur distance d'édition (Levenshtein).

**N-grammes** : Séquences de N mots consécutifs. Unigrammes (1 mot), bigrammes (2 mots).

**Cascade** : Approche multi-passes avec seuils dégressifs pour maximiser rappel tout en gardant précision.

**Confiance** : Ratio entre score meilleur profil et somme des deux meilleurs scores.

---

**Fin de la documentation**

*Pour toute question ou amélioration, contacter l'équipe projet.*