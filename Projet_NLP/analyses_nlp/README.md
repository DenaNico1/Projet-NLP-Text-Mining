# 📊 Analyses NLP - Marché de l'Emploi Data/IA

Analyses complètes du corpus de 3024 offres d'emploi Data/IA en France.

---

## 🎯 Vue d'Ensemble

Ce dossier contient **7 analyses NLP** qui transforment vos 3024 offres d'emploi en insights actionnables.

### 📋 Les 7 Analyses

| # | Analyse | Objectif | Outputs |
|---|---------|----------|---------|
| **1** | **Preprocessing** | Nettoyage et tokenization | data_preprocessed.pkl |
| **2** | **Extraction Compétences** | TF-IDF, n-grams, word cloud | competences_extracted.json |
| **3** | **Topic Modeling** | Profils métiers (LDA) | topics_lda.json |
| **4** | **Géo-Sémantique** | Spécificités régionales | analyse_geo_semantique.json |
| **5** | **Évolution Temporelle** | Tendances compétences | evolution_temporelle.json |
| **6** | **Embeddings + Clustering** | Vecteurs + visualisation 2D | clustering_2d.html |
| **7** | **Stacks × Salaires** | Corrélations rémunération | stacks_salaires.json |

---

## 🚀 Installation

### 1. Installer les Dépendances

```bash
cd analyses_nlp
pip install -r ../requirements.txt
```

**Temps d'installation** : 5-10 minutes

### 2. Télécharger Données NLTK

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

---

## 📊 Utilisation

### Option A : Tout Lancer Automatiquement ✅ (Recommandé)

```bash
python run_all_analyses.py
```

**Durée** : 15-30 minutes  
**Résultats** : Tous les fichiers dans `../resultats_nlp/`

### Option B : Lancer Analyse par Analyse

```bash
# 1. Preprocessing (obligatoire en premier)
python 1_preprocessing.py

# 2. Extraction compétences
python 2_extraction_competences.py

# 3. Topic modeling
python 3_topic_modeling.py

# 4. Géo-sémantique
python 4_geo_semantique.py

# 5. Évolution temporelle
python 5_evolution_temporelle.py

# 6. Embeddings + clustering
python 6_embeddings_clustering.py

# 7. Stacks × salaires
python 7_stacks_salaires.py
```

### Option C : Sauter Certaines Étapes

```bash
# Sauter preprocessing (déjà fait) et embeddings (long)
python run_all_analyses.py --skip 1,6
```

---

## 📁 Structure des Résultats

```
resultats_nlp/
├── 📄 Données
│   ├── data_preprocessed.pkl          # DataFrame nettoyé
│   ├── data_with_competences.pkl      # + compétences extraites
│   ├── data_with_topics.pkl           # + topics LDA
│   └── data_with_clusters.pkl         # + clusters
│
├── 📊 Analyses JSON
│   ├── stats_globales.json            # Statistiques générales
│   ├── competences_extracted.json     # Top compétences, n-grams
│   ├── topics_lda.json                # Profils métiers
│   ├── analyse_geo_semantique.json    # Par région
│   ├── evolution_temporelle.json      # Tendances
│   ├── clusters_analysis.json         # Clusters
│   └── stacks_salaires.json           # Corrélations
│
├── 🎨 Visualisations
│   ├── wordcloud_competences.png
│   ├── top30_competences.html
│   ├── heatmap_cooccurrence.png
│   ├── topics_distribution.html
│   ├── carte_regions.html
│   ├── clustering_2d.html
│   ├── salaires_par_competence.html
│   └── heatmap_region_competence.html
│
└── 🧠 Modèles
    ├── lda_model.pkl                  # Modèle LDA entraîné
    ├── embeddings.npy                 # Vecteurs (2000×384)
    └── umap_coords.npy                # Coordonnées 2D
```

---

## 🔍 Détails des Analyses

### 1️⃣ Preprocessing

**Entrée** : Entrepôt DuckDB (3024 offres)  
**Sortie** : Texte nettoyé + tokenisé

**Ce qui est fait** :
- Nettoyage HTML/URLs/emails
- Tokenization français
- Suppression stopwords
- Création dictionnaire 150+ compétences

**Fichiers** :
- `data_preprocessed.pkl`
- `dictionnaire_competences.json`
- `stats_globales.json`

---

### 2️⃣ Extraction de Compétences

**Méthodes** :
- Pattern matching (dictionnaire)
- TF-IDF (termes importants)
- N-grams (bi-grams, tri-grams)
- Co-occurrence (paires)

**Résultats attendus** :
- 100+ compétences extraites
- Top bi-grams : "machine learning", "deep learning"
- Paires : Python + SQL, Docker + Kubernetes

**Visualisations** :
- Word cloud
- Top 30 bar chart
- Heatmap co-occurrence
- Compétences par région

---

### 3️⃣ Topic Modeling (LDA)

**Algorithme** : Latent Dirichlet Allocation  
**Nombre de topics** : 6

**Résultats attendus** :
```
Topic 1 (28%): Data Engineering
  → ETL, Spark, Airflow, pipeline

Topic 2 (24%): ML Engineer  
  → model, TensorFlow, deploy, API

Topic 3 (18%): Business Intelligence
  → Power BI, Tableau, dashboard, KPI

Topic 4 (15%): Deep Learning
  → PyTorch, neural network, NLP

Topic 5 (10%): Data Analyst
  → Excel, statistiques, analyse

Topic 6 (5%): MLOps
  → Kubernetes, CI/CD, monitoring
```

**Fichiers** :
- `topics_lda.json`
- `lda_model.pkl`
- `topics_distribution.html`

---

### 4️⃣ Géo-Sémantique

**Analyse** : Vocabulaire spécifique par région

**Résultats attendus** :
- **Île-de-France** : Deep Learning, FinTech, startup
- **Auvergne-RA** : BI, industrie, ERP
- **Occitanie** : Aérospatial, cloud

**Visualisations** :
- Carte interactive France
- Top termes/région
- Salaires/région

---

### 5️⃣ Évolution Temporelle

**Analyse** : Tendances compétences dans le temps

**Résultats attendus** :
- LangChain : +300% (Nov → Déc)
- MLOps : Croissance stable
- GenAI : Émergent

**Fichiers** :
- `evolution_temporelle.json`

---

### 6️⃣ Embeddings + Clustering

**Algorithme** :
- Sentence-BERT (embeddings)
- UMAP (réduction 2D)
- K-Means (8 clusters)

**Résultats attendus** :
- Clustering visuel 2D interactif
- 8 groupes d'offres similaires
- Chaque point = 1 offre (hover = détails)

**⚠️ Note** : Analyse la plus longue (5-10 min)

**Fichiers** :
- `clustering_2d.html` ← Visualisation interactive !
- `embeddings.npy`
- `clusters_analysis.json`

---

### 7️⃣ Stacks × Salaires

**Analyse** : Corrélations compétences ↔ rémunération

**Résultats attendus** :

```
Top compétences rémunérées :
1. Kubernetes    : 72k€
2. MLOps         : 68k€
3. PyTorch       : 65k€
4. TensorFlow    : 62k€
5. Docker        : 58k€

Stacks :
- MLOps Stack    : 72k€ (87 offres)
- ML Engineer    : 62k€ (289 offres)
- Data Analyst   : 42k€ (456 offres)
```

**Visualisations** :
- Box plots par compétence
- Bar chart stacks
- Heatmap région × compétence

---

## 🐛 Dépannage

### Erreur "Module not found"

```bash
pip install -r ../requirements.txt
```

### Erreur NLTK

```python
import nltk
nltk.download('all')
```

### Erreur Mémoire (Embeddings)

Réduire l'échantillon dans `6_embeddings_clustering.py` :
```python
# Ligne 30
df_sample = df.sample(min(1000, len(df)), random_state=42)
```

### Analyse Trop Longue

```bash
# Sauter embeddings
python run_all_analyses.py --skip 6
```

---

## 📊 Utilisation des Résultats

### Dans Python

```python
import pickle
import json

# Charger données avec compétences
with open('../resultats_nlp/data_with_topics.pkl', 'rb') as f:
    df = pickle.load(f)

# Charger résultats JSON
with open('../resultats_nlp/competences_extracted.json', 'r') as f:
    comps = json.load(f)

print(comps['top_competences'][:10])
```

### Visualisations HTML

Ouvrir directement dans navigateur :
- `clustering_2d.html`
- `top30_competences.html`
- `carte_regions.html`

---

## ⏱️ Temps d'Exécution

| Analyse | Durée |
|---------|-------|
| Preprocessing | 1-2 min |
| Extraction compétences | 2-3 min |
| Topic modeling | 3-5 min |
| Géo-sémantique | 1-2 min |
| Évolution temporelle | 1 min |
| Embeddings + clustering | **5-10 min** ⚠️ |
| Stacks × salaires | 2-3 min |
| **TOTAL** | **15-30 min** |

---

## 🎯 Prochaines Étapes

Après avoir lancé les analyses :

1. ✅ Consulter les visualisations HTML
2. ✅ Analyser les fichiers JSON
3. ✅ Passer à l'application Streamlit
4. ✅ Intégrer dans le rapport académique

---

## 📞 Support

En cas de problème :
1. Vérifier les logs dans la console
2. Vérifier que `entrepot_nlp.duckdb` existe
3. Vérifier les dépendances installées

**Tout est prêt ! Lancez** :
```bash
python run_all_analyses.py
```

🚀 **Bon courage !**