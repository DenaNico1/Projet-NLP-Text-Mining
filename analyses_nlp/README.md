# 🔬 ANALYSES NLP - Data IA Talent Observatory

**Pipeline complet de Text Mining pour l'analyse du marché Data/IA**

---

## 📋 VUE D'ENSEMBLE

Ce dossier contient l'ensemble du **pipeline NLP/ML** pour l'analyse des offres d'emploi Data/IA en France. Il implémente **9 analyses scientifiques** allant du preprocessing à la classification supervisée, en passant par le topic modeling et l'extraction de compétences.

### 🎯 Objectifs

- ✅ Prétraiter 3,023 descriptions d'offres (tokenization, nettoyage)
- ✅ Extraire automatiquement 770 compétences techniques
- ✅ Découvrir 6 profils métiers via Topic Modeling (LDA)
- ✅ Classifier offres avec 90% de précision (SVM)
- ✅ Identifier compétences "signature" par profil (Chi²)
- ✅ Analyser spécificités géographiques et temporelles

---

## 🗂️ STRUCTURE DOSSIER

```
analyses_nlp/
│
├── README.md                               # Ce fichier
├── DOCUMENTATION_ANALYSES_NLP.md           # Doc technique complète
│
├── 0_preparation_donnees.py                # Chargement données entrepôt
├── 1_preprocessing.py                      # Nettoyage, tokenization (NLTK)
├── 2_extraction_competences.py             # Extraction 770 compétences
├── 3_topic_modeling.py                     # LDA (k=6, coherence=0.78)
├── 4_analyse_geo_semantique.py             # Spécificités régionales
├── 5_analyse_temporelle.py                 # Évolution tendances
├── 6_clustering.py                         # UMAP + K-Means
├── 7_analyse_stacks_salaires.py            # Correlation compétences-salaires
├── 8_classification_supervisee.py          # SVM (89.6%), MLP (89.4%)
├── 9_selection_features_chi2.py            # Compétences discriminantes
│
├── hybrid_classification.py                # Système hybride 3 couches
├── apply_hybrid_classification.py          # Script application hybride
│
├── dictionnaire_competences.json           # 770 compétences + patterns
├── stopwords_custom.txt                    # Stopwords domaine Data/IA
│
└── resultats_nlp/
    ├── models/
    │   ├── lda_model.pkl                   # Modèle LDA (figé v1)
    │   ├── lda_vectorizer.pkl              # CountVectorizer
    │   ├── model_svm.pkl                   # SVM classifieur
    │   ├── model_mlp.pkl                   # MLP classifieur
    │   ├── vectorizer_classification.pkl   # TF-IDF
    │   ├── umap_model.pkl                  # UMAP embeddings
    │   ├── kmeans_model.pkl                # K-Means clusters
    │   ├── data_preprocessed.pkl           # Données prétraitées
    │   ├── data_with_topics.pkl            # Données + topics LDA
    │   └── data_with_hybrid_profiles.pkl   # Données + profils hybrides
    │
    ├── visualizations/
    │   ├── wordclouds/                     # Nuages de mots par profil
    │   ├── topic_distribution.png          # Distribution topics
    │   ├── confusion_matrix_svm.png        # Matrice confusion
    │   ├── umap_projection.png             # Projection UMAP
    │   └── correlation_competences.png     # Heatmap compétences
    │
    ├── lda_topics.json                     # Topics + top terms
    ├── chi2_selection.json                 # Features Chi² par profil
    ├── classification_results.json         # Métriques SVM/MLP
    ├── geo_analysis.json                   # Spécificités régionales
    ├── temporal_analysis.json              # Tendances temporelles
    ├── cluster_results.json                # Résultats clustering
    ├── hybrid_classification_stats.json    # Stats système hybride
    └── hybrid_classifier_config_v1.json    # Config hybride v1
```

---

## 🚀 GUIDE D'UTILISATION

### **Installation Dépendances**

```bash
pip install -r requirements.txt
```

**Principales librairies** :
```
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
nltk>=3.8.1
spacy>=3.7.0
gensim>=4.3.0
umap-learn>=0.5.5
plotly>=5.18.0
seaborn>=0.13.0
```

**Téléchargement ressources NLTK** :
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

### **Exécution Pipeline Complet**

#### **Option 1 : Pipeline séquentiel (recommandé pour 1ère fois)**

```bash
# Étape 0 : Préparer données depuis entrepôt
python 0_preparation_donnees.py

# Étape 1 : Preprocessing (15 min)
python 1_preprocessing.py

# Étape 2 : Extraction compétences (10 min)
python 2_extraction_competences.py

# Étape 3 : Topic Modeling LDA (20 min)
python 3_topic_modeling.py --n_topics 6

# Étape 4 : Analyse géo-sémantique (5 min)
python 4_analyse_geo_semantique.py

# Étape 5 : Analyse temporelle (5 min)
python 5_analyse_temporelle.py

# Étape 6 : Clustering (10 min)
python 6_clustering.py --n_clusters 8

# Étape 7 : Analyse stacks × salaires (5 min)
python 7_analyse_stacks_salaires.py

# Étape 8 : Classification supervisée (15 min)
python 8_classification_supervisee.py

# Étape 9 : Sélection features Chi² (5 min)
python 9_selection_features_chi2.py

# Étape 10 : Classification hybride (5 min)
python apply_hybrid_classification.py
```

**Durée totale** : ~1h30

---

#### **Option 2 : Pipeline automatisé**

```bash
# Script master qui exécute tout
python run_full_pipeline.py
```

---

### **Exécution Scripts Individuels**

#### **Script 1 : Preprocessing**

```bash
python 1_preprocessing.py
```

**Input** : `resultats_nlp/models/data_raw.pkl` (depuis entrepôt)  
**Output** : `resultats_nlp/models/data_preprocessed.pkl`

**Ce qui est fait** :
- Tokenization (NLTK)
- Lowercasing
- Suppression stopwords (français + custom)
- Suppression ponctuation
- Conservation tokens alphanumériques

**Paramètres modifiables** :
```python
# Dans 1_preprocessing.py
MIN_TOKEN_LENGTH = 2      # Longueur minimale token
STOPWORDS_CUSTOM = [...]  # Stopwords domaine
```

---

#### **Script 2 : Extraction Compétences**

```bash
python 2_extraction_competences.py
```

**Input** : 
- `data_preprocessed.pkl`
- `dictionnaire_competences.json` (770 compétences)

**Output** : 
- `data_preprocessed.pkl` (enrichi avec colonne `competences_found`)

**Méthode** :
- Pattern matching regex (case-insensitive)
- Validation contexte (≥2 caractères, pas dans stopwords)
- Fréquence par offre

**Top 10 compétences détectées** :
```
1. Python       : 2,145 offres (71%)
2. SQL          : 1,987 offres (66%)
3. Machine Learning : 1,456 offres (48%)
4. Pandas       : 1,234 offres (41%)
5. Spark        : 987 offres (33%)
...
```

**Ajouter nouvelle compétence** :
```json
// Dans dictionnaire_competences.json
{
  "langages": {
    "Rust": {
      "patterns": ["\\brust\\b"],
      "categorie": "Langage"
    }
  }
}
```

---

#### **Script 3 : Topic Modeling (LDA)**

```bash
python 3_topic_modeling.py --n_topics 6 --max_iter 1000
```

**Hyperparamètres** :
```bash
--n_topics        # Nombre de topics (défaut: 6)
--alpha           # Prior Dirichlet docs-topics (défaut: 0.1)
--beta            # Prior Dirichlet topics-mots (défaut: 0.01)
--max_iter        # Nombre itérations (défaut: 1000)
--random_state    # Seed (défaut: 42)
```

**Output** :
- `models/lda_model.pkl` (modèle scikit-learn)
- `models/lda_vectorizer.pkl` (CountVectorizer)
- `lda_topics.json` (topics + top 20 termes)
- `data_with_topics.pkl` (données + colonne `topic_dominant`)

**Évaluation** :
```python
# Coherence score (plus élevé = meilleur)
Coherence : 0.78  # Excellent (>0.7)

# Perplexity (plus bas = meilleur)
Perplexity : -8.2  # Bon (<-7)
```

**Topics découverts** :
```
Topic 0 - Data Engineering (24%)
  spark, airflow, sql, etl, kafka, hive, hadoop, python

Topic 1 - ML Engineering (16%)
  machine, learning, scikit, model, python, pandas, tensorflow

Topic 2 - Business Intelligence (13%)
  power, bi, tableau, qlik, dax, sql, excel, reporting

Topic 3 - Deep Learning (24%)
  deep, learning, pytorch, tensorflow, neural, network, cnn

Topic 4 - Data Analysis (7%)
  sql, excel, python, pandas, statistics, analysis

Topic 5 - MLOps (28%)
  kubernetes, docker, mlops, ci, cd, terraform, jenkins
```

**Tester différents k** :
```bash
for k in 4 6 8 10; do
    python 3_topic_modeling.py --n_topics $k
done
# Comparer coherence scores
```

---

#### **Script 8 : Classification Supervisée**

```bash
python 8_classification_supervisee.py --model svm
```

**Options** :
```bash
--model        # svm | mlp | random_forest | gradient_boosting
--cv_folds     # Nombre folds cross-validation (défaut: 5)
--test_size    # Taille test set (défaut: 0.2)
```

**Pipeline** :
1. Split train/test (80/20 stratifié)
2. Vectorisation TF-IDF (max_features=500)
3. GridSearchCV sur hyperparamètres
4. Entraînement meilleur modèle
5. Évaluation (accuracy, precision, recall, F1)
6. Sauvegarde modèle

**Résultats** :

| Modèle | Accuracy | F1 (weighted) | Temps entraînement |
|--------|----------|---------------|---------------------|
| **SVM** | 89.6% | 0.896 | 45s |
| MLP | 89.4% | 0.895 | 120s |
| Random Forest | 87.2% | 0.871 | 30s |
| Gradient Boosting | 88.1% | 0.880 | 90s |

**Meilleur modèle** : SVM (`kernel='rbf', C=2.0`)

**Matrice confusion** : `visualizations/confusion_matrix_svm.png`

---

#### **Script 9 : Chi² Selection**

```bash
python 9_selection_features_chi2.py --top_k 100
```

**Objectif** : Identifier compétences "signature" par profil

**Méthode** :
1. Créer matrice binaire (3,023 × 770) : 1 si compétence présente
2. Chi² test pour chaque (compétence, profil)
3. Sélectionner top k features par χ² score
4. Calculer lift : `P(comp|profil) / P(comp|global)`

**Output** : `chi2_selection.json`
```json
{
  "signature_by_profile": {
    "MLOps": [
      {"competence": "Kubernetes", "chi2": 698.5, "lift": 2.3},
      {"competence": "Docker", "chi2": 645.2, "lift": 2.1},
      ...
    ]
  }
}
```

**Top signatures** :

| Profil | Top 3 Compétences (lift > 1.5) |
|--------|--------------------------------|
| MLOps | Kubernetes (2.3x), Docker (2.1x), Terraform (1.9x) |
| Deep Learning | PyTorch (2.8x), TensorFlow (2.4x), GPU (2.2x) |
| BI | Power BI (3.1x), Tableau (2.7x), Qlik (2.3x) |
| Data Engineering | Spark (2.1x), Airflow (1.9x), Kafka (1.8x) |

**Application** : Gap analysis dans Audit de Profil (Streamlit)

---

#### **Système Hybride 3 Couches**

```bash
python apply_hybrid_classification.py
```

**Ce que fait le script** :
1. ✅ Charge données (`data_with_topics.pkl`)
2. ✅ Applique classification 3 couches (titre → compétences → LDA)
3. ✅ Génère stats détaillées (par méthode, profil, confiance)
4. ✅ Détecte profils émergents (titres fréquents en fallback)
5. ✅ Sauvegarde résultats (`data_with_hybrid_profiles.pkl`)

**Output console** :
```
📊 STATISTIQUES DE CLASSIFICATION
============================================================
Total offres : 3023

Par méthode :
  • titre              : 2116 ( 70.0%)
  • competences        :  484 ( 16.0%)
  • lda_fallback       :  423 ( 14.0%)

Par profil (Top 10) :
  • Data Engineer               :  520 ( 17.2%)
  • Data Scientist              :  470 ( 15.5%)
  • ML Engineer                 :  380 ( 12.6%)
  • MLOps Engineer              :  350 ( 11.6%)
  ...

🔍 DÉTECTION PROFILS ÉMERGENTS
============================================================
Offres en fallback : 423 (14.0%)

Titres fréquents non classés :
  (aucun si < 10 occurrences)
```

**Ajouter nouveau profil** :
```bash
# 1. Éditer config
nano hybrid_classifier_config_v1.json

# 2. Ajouter pattern
{
  "Prompt Engineer": [
    "prompt engineer",
    "prompt.*engineer"
  ]
}

# 3. Reclassifier
python apply_hybrid_classification.py
```

---

## 📊 RÉSULTATS CLÉS

### **Métriques Globales**

| Métrique | Valeur | Détail |
|----------|--------|--------|
| **Corpus** | 3,023 offres | France Travail (83%) + Indeed (17%) |
| **Vocabulaire** | 12,453 tokens | Après preprocessing |
| **Compétences uniques** | 770 | 6 catégories |
| **Compétences/offre** | 12.4 (médiane) | Min: 0, Max: 45 |
| **Topics LDA** | 6 | Coherence: 0.78 |
| **Profils hybrides** | 14 | Data Scientist, ML Engineer... |
| **Précision SVM** | 89.6% | F1: 0.896 |
| **Précision hybride** | 88.7% | Pondérée par méthode |

---

### **Top 10 Compétences**

| Rang | Compétence | Nb Offres | % Corpus |
|------|------------|-----------|----------|
| 1 | Python | 2,145 | 71% |
| 2 | SQL | 1,987 | 66% |
| 3 | Machine Learning | 1,456 | 48% |
| 4 | Pandas | 1,234 | 41% |
| 5 | Spark | 987 | 33% |
| 6 | Docker | 856 | 28% |
| 7 | AWS | 745 | 25% |
| 8 | TensorFlow | 612 | 20% |
| 9 | Kubernetes | 598 | 20% |
| 10 | Tableau | 534 | 18% |

---

### **Distribution Profils Hybrides**

| Profil | Nb Offres | % |
|--------|-----------|---|
| Data Engineering | 520 | 17.2% |
| Data Scientist | 470 | 15.5% |
| ML Engineering | 380 | 12.6% |
| MLOps Engineer | 350 | 11.6% |
| Deep Learning | 280 | 9.3% |
| Data Analyst | 210 | 6.9% |
| BI Analyst | 190 | 6.3% |
| NLP Engineer | 80 | 2.6% |
| **Autres** | 543 | 18.0% |

---

## 🔧 CONFIGURATION

### **Fichiers Configuration**

#### **dictionnaire_competences.json**

Structure :
```json
{
  "langages": {
    "Python": {
      "patterns": ["\\bpython\\b"],
      "categorie": "Langage",
      "type": "Technique"
    }
  },
  "frameworks_ml": {
    "TensorFlow": {
      "patterns": ["tensorflow", "tf\\."],
      "categorie": "Framework ML",
      "type": "Technique"
    }
  }
}
```

**Catégories** :
- `langages` (45 compétences)
- `frameworks_ml` (120)
- `outils_data` (180)
- `cloud_infra` (95)
- `bi_viz` (65)
- `soft_skills` (265)

---

#### **stopwords_custom.txt**

```
data
ia
intelligence
artificielle
recherche
poste
offre
emploi
candidat
profil
experience
annee
...
```

**Usage** :
```python
# Dans 1_preprocessing.py
custom_stopwords = set(open('stopwords_custom.txt').read().split())
```

---

#### **hybrid_classifier_config_v1.json**

```json
{
  "version": "1.0",
  "date": "2024-12-27",
  "regex_profils": {
    "Data Scientist": [
      "data scientist",
      "scientifique.*données"
    ]
  },
  "signatures_competences": {
    "Data Scientist": {
      "must_have": ["python", "machine learning"],
      "strong_indicators": ["pandas", "scikit-learn"],
      "threshold": 0.3
    }
  },
  "topic_to_profil": {
    "0": "Data Engineering",
    "1": "ML Engineering",
    ...
  }
}
```

---

## 📈 VISUALISATIONS

Toutes les visualisations sont sauvegardées dans `resultats_nlp/visualizations/`

### **Disponibles** :

1. **Nuages de mots** (`wordclouds/`)
   - 1 par profil (6 topics LDA)
   - Top 50 termes pondérés

2. **Distribution topics** (`topic_distribution.png`)
   - Bar chart % corpus par topic

3. **Matrice confusion** (`confusion_matrix_svm.png`)
   - Heatmap 6×6 (précision par classe)

4. **Projection UMAP** (`umap_projection.png`)
   - Scatter 2D coloré par profil
   - Visualise séparabilité

5. **Corrélation compétences** (`correlation_competences.png`)
   - Heatmap co-occurrences top 30 compétences

---

## 🐛 TROUBLESHOOTING

### **Erreur : `ModuleNotFoundError: No module named 'nltk'`**

```bash
pip install nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

### **Erreur : `FileNotFoundError: data_preprocessed.pkl`**

→ Exécutez d'abord les scripts dans l'ordre (0 → 1 → 2...)

```bash
python 0_preparation_donnees.py
python 1_preprocessing.py
```

---

### **Erreur : `MemoryError` pendant LDA**

→ Réduire `max_features` dans CountVectorizer

```python
# Dans 3_topic_modeling.py
vectorizer = CountVectorizer(
    max_features=500,  # Au lieu de 1000
    ...
)
```

---

### **Warning : `Coherence score very low (<0.5)`**

→ Tester différents `n_topics` :

```bash
for k in 4 6 8 10 12; do
    python 3_topic_modeling.py --n_topics $k
done
```

→ Vérifier qualité preprocessing (trop de stopwords ?)

---

### **Classification hybride : trop d'offres en fallback (>25%)**

→ Ajouter règles Couche 1 (titre) ou Couche 2 (compétences)

```bash
# 1. Analyser titres fréquents
python apply_hybrid_classification.py

# 2. Éditer config
nano hybrid_classifier_config_v1.json

# 3. Reclassifier
python apply_hybrid_classification.py
```

---

## 📚 DOCUMENTATION COMPLÉMENTAIRE

- 📄 **DOCUMENTATION_ANALYSES_NLP.md** : Documentation technique complète
- 📄 **hybrid_classification.py** : Code commenté système hybride
- 📄 **../entrepot_de_donnees/README.md** : Documentation entrepôt
- 📄 **../app_streamlit/README_DATATALENT_OBSERVATORY.md** : Documentation app

---

## 🧪 TESTS

### **Validation Pipeline**

```bash
# Test complet sur 100 offres échantillon
python test_pipeline.py --sample_size 100

# Output attendu :
# ✅ Preprocessing : 100/100 offres
# ✅ Extraction compétences : 97/100 (≥1 compétence)
# ✅ Topic modeling : Coherence > 0.6
# ✅ Classification : Accuracy > 85%
```

---

### **Validation Manuelle**

1. **Vérifier compétences extraites** :
```python
import pickle
df = pickle.load(open('resultats_nlp/models/data_preprocessed.pkl', 'rb'))
print(df[['titre', 'competences_found']].head(10))
```

2. **Vérifier topics LDA** :
```python
import json
topics = json.load(open('resultats_nlp/lda_topics.json'))
for topic_id, data in topics.items():
    print(f"Topic {topic_id}: {', '.join(data['top_terms'][:10])}")
```

3. **Vérifier profils hybrides** :
```python
df = pickle.load(open('resultats_nlp/models/data_with_hybrid_profiles.pkl', 'rb'))
print(df['profil'].value_counts())
print(df['methode'].value_counts())
```

---

## 🚀 OPTIMISATIONS

### **Pour corpus >10k offres**

1. **Preprocessing** : Parallélisation
```python
from multiprocessing import Pool

def preprocess_batch(batch):
    # ...
    return batch

with Pool(8) as p:
    results = p.map(preprocess_batch, df_chunks)
```

2. **LDA** : Utiliser Gensim (plus rapide)
```python
from gensim.models import LdaModel
# Au lieu de scikit-learn
```

3. **Classification** : Mini-batch learning
```python
from sklearn.linear_model import SGDClassifier
# Au lieu de SVM complet
```

---

## 📊 EXPORTS

### **Exporter résultats vers CSV**

```bash
python export_results.py
```

**Fichiers générés** :
- `resultats_nlp/exports/offres_with_profiles.csv` (toutes colonnes)
- `resultats_nlp/exports/competences_frequency.csv` (top 100)
- `resultats_nlp/exports/topics_distribution.csv` (6 topics)
- `resultats_nlp/exports/classification_report.csv` (métriques)

---

## 👥 CONTRIBUTEURS

**Projet Master SISE - NLP Text Mining**  
Décembre 2025

---

## 📄 LICENCE

Projet académique - Master SISE

---

## 📞 SUPPORT

Pour toute question :
- 📧 Email : [votre email]
- 📂 Repo : [votre repo GitHub]

---

**🔬 DataTalent Observatory - Pipeline NLP Complet**