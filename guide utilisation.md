# 🚀 GUIDE RAPIDE - PROJET NLP TEXT MINING

## 📦 INSTALLATION

```bash
# 1. Cloner/Télécharger le projet
cd Projet_NLP

# 2. Installer dépendances
pip install -r requirements.txt

# 3. Télécharger modèle spaCy français
python -m spacy download fr_core_news_sm

# 4. Télécharger stopwords NLTK
python -c "import nltk; nltk.download('stopwords')"
```

---

## 🔄 EXÉCUTION PIPELINE COMPLET (ordre)

### **Étape 1 : Preprocessing**
```bash
cd analyses_nlp/fichiers_analyses
python 1_preprocessing.py
```
→ Crée `data_clean.pkl` + extraction compétences

### **Étape 2 : Extraction Compétences**
```bash
python 2_extraction_competences.py
```
→ TF-IDF, bi-grams, co-occurrences

### **Étape 3 : Topic Modeling**
```bash
python 3_topic_modeling.py
```
→ LDA 8 topics

### **Étape 4 : Classification**
```bash
python 4_classification_hybride_ultimate.py
```
→ 56.2% classifiées, 14 profils

### **Étape 5 : Visualisations Profils**
```bash
python 5_visualisations_profils.py
```
→ 12 graphiques HTML

### **Étape 6 : Embeddings Offres**
```bash
python 6_embeddings_clustering.py
```
→ UMAP, t-SNE, KMeans, HDBSCAN (⏱️ 3-5 min)

### **Étape 7 : Embeddings Compétences**
```bash
python 6_embeddings_competences.py
```
→ Carte 3D, réseau sémantique

### **Étape 8 : Visualisations 3D**
```bash
python 7_visualisations_3d_projector.py
```
→ 6 vues 3D style TensorFlow Projector

### **Étape 9 : Réseau Sémantique**
```bash
python 8_network_semantic.py
```
→ Réseau PyVis interactif

---

## 🎯 LANCEMENT APPLICATION STREAMLIT

```bash
cd ../../app_streamlit
streamlit run app.py
```

→ Ouvre http://localhost:8501

**7 Pages disponibles :**
- 🏠 Dashboard (KPIs)
- 🗺️ Géographique (carte France)
- 💼 Profils (14 profils métiers)
- 🎓 Compétences (réseau sémantique)
- 🔬 Topics (LDA)
- 🌐 Visualisations 3D
- 📊 Insights (+ prédicteur interactif)

---

## 📁 STRUCTURE PROJET

```
Projet_NLP/
├─ entrepot_de_donnees/
│  └─ entrepot_nlp.duckdb (base données)
│
├─ analyses_nlp/
│  └─ fichiers_analyses/
│     ├─ 1_preprocessing.py
│     ├─ 2_extraction_competences.py
│     ├─ 3_topic_modeling.py
│     ├─ 4_classification_hybride_ultimate.py
│     ├─ 5_visualisations_profils.py
│     ├─ 6_embeddings_clustering.py
│     ├─ 6_embeddings_competences.py
│     ├─ 7_visualisations_3d_projector.py
│     └─ 8_network_semantic.py
│
├─ resultats_nlp/ (créé automatiquement)
│  ├─ models/ (fichiers .pkl, .npy)
│  ├─ visualisations/ (HTML, PNG)
│  └─ *.json, *.csv
│
└─ app_streamlit/
   ├─ app.py
   ├─ config.py
   └─ pages/ (7 pages)
```

---

## ⚡ EXÉCUTION RAPIDE (tout d'un coup)

```bash
cd analyses_nlp/fichiers_analyses

python 1_preprocessing.py && \
python 2_extraction_competences.py && \
python 3_topic_modeling.py && \
python 4_classification_hybride_ultimate.py && \
python 5_visualisations_profils.py && \
python 6_embeddings_clustering.py && \
python 6_embeddings_competences.py && \
python 7_visualisations_3d_projector.py && \
python 8_network_semantic.py

cd ../../app_streamlit
streamlit run app.py
```

⏱️ **Temps total : ~15-20 minutes**

---

## 🐛 ERREURS FRÉQUENTES

**Erreur : "No module named 'spacy'"**
→ `pip install spacy && python -m spacy download fr_core_news_sm`

**Erreur : "FileNotFoundError: entrepot_nlp.duckdb"**
→ Vérifier chemin base DuckDB dans scripts

**Erreur : "CUDA out of memory" (embeddings)**
→ Normal si pas de GPU, utilise CPU (plus lent)

**Application Streamlit : page blanche**
→ Vérifier tous les scripts 1-8 exécutés avant

---

## 📊 DONNÉES DE SORTIE

**Fichiers principaux :**
- `data_clean.pkl` (3,003 offres preprocessées)
- `data_with_profiles.pkl` (avec classification)
- `embeddings.npy` (vecteurs 384-dim)
- `topics_lda.json` (8 topics)
- Visualisations HTML dans `resultats_nlp/visualisations/`

---

## 💡 CONSEILS

✅ Exécuter scripts dans l'ordre (1→8)
✅ Vérifier `resultats_nlp/` créé
✅ Patience sur embeddings (long)
✅ Streamlit nécessite TOUS les fichiers

---

**Support : Nico - Master SISE 2025**