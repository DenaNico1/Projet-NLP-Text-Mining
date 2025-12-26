# 📱 Application Streamlit - Marché Data/IA

Application web interactive pour explorer les 3000+ offres d'emploi Data/IA en France.

---

## 🚀 Lancement Rapide

### 1. Installation

```bash
# Depuis le dossier app_streamlit/
pip install streamlit plotly pandas pillow
```

### 2. Lancer l'Application

```bash
streamlit run app.py
```

**L'application s'ouvre automatiquement dans votre navigateur** à `http://localhost:8501`

---

## 📊 Les 7 Pages

| Page | Description | Fonctionnalités |
|------|-------------|-----------------|
| **🏠 Accueil** | Vue d'ensemble | KPIs, navigation |
| **📊 Dashboard** | Métriques générales | Graphiques, stats |
| **🔍 Exploration** | Recherche d'offres | Filtres, export CSV |
| **🎓 Compétences** | Top skills | Word cloud, co-occurrences |
| **💰 Salaires** | Rémunérations | Distribution, par région/stack |
| **🗺️  Géographie** | Carte France | Répartition, spécificités |
| **🔬 Clustering** | Visualisation 2D | Groupes similaires |

---

## 🎯 Fonctionnalités

### Filtres Interactifs
- Recherche textuelle
- Filtre par région
- Filtre par type de contrat
- Filtre par source

### Visualisations
- Graphiques Plotly (interactifs)
- Cartes géographiques
- Word clouds
- Clustering 2D

### Export
- Export CSV des résultats filtrés
- Téléchargement graphiques

---

## 📁 Structure

```
app_streamlit/
├── app.py                      # Page d'accueil
├── pages/
│   ├── 1_📊_Dashboard.py
│   ├── 2_🔍_Exploration.py
│   ├── 3_🎓_Competences.py
│   ├── 4_💰_Salaires.py
│   ├── 5_🗺️_Geographie.py
│   └── 6_🔬_Clustering.py
└── utils/
    └── data_loader.py          # Chargement données
```

---

## ⚙️ Configuration

### Prérequis

✅ Analyses NLP terminées (`run_all_analyses.py`)  
✅ Dossier `resultats_nlp/` avec les fichiers  
✅ Python 3.8+

### Dépendances

```bash
pip install streamlit plotly pandas pillow
```

---

## 🐛 Dépannage

### ❌ "Module not found"

```bash
pip install streamlit plotly pandas pillow
```

### ❌ "File not found: resultats_nlp/..."

→ Lancez d'abord les analyses NLP :
```bash
cd ../analyses_nlp
python run_all_analyses.py
```

### ❌ Page blanche / Erreur

→ Vérifiez la console pour les erreurs  
→ Relancez avec `streamlit run app.py --server.headless true`

---

## 🎨 Personnalisation

### Changer le Thème

Créez `.streamlit/config.toml` :

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Modifier le Port

```bash
streamlit run app.py --server.port 8080
```

---

## 📊 Captures d'Écran

### Page d'Accueil
- KPIs en temps réel
- Points clés du marché
- Navigation intuitive

### Dashboard
- Graphiques interactifs
- Répartition par source/région
- Évolution temporelle

### Exploration
- Recherche puissante
- Filtres multiples
- Détails des offres

### Compétences
- Word cloud
- Top 30 skills
- Co-occurrences

### Salaires
- Distribution salariale
- Salaire par région/stack
- Comparaisons

### Géographie
- Carte interactive France
- Top villes
- Spécificités régionales

### Clustering
- Visualisation 2D
- 8 groupes d'offres
- Analyse par cluster

---

## 🚀 Déploiement (Optionnel)

### Streamlit Cloud (Gratuit)

1. Poussez le code sur GitHub
2. Allez sur [share.streamlit.io](https://share.streamlit.io)
3. Connectez votre repo
4. Déployez !

### Heroku

```bash
# Créer Procfile
echo "web: streamlit run app.py --server.port $PORT" > Procfile

# Déployer
heroku create
git push heroku main
```

---

## 💡 Astuces

### Performance

- Les données sont **cachées** (@st.cache_data)
- Premier chargement = lent, ensuite = rapide
- Rafraîchir le cache : `CTRL+R`

### Navigation

- Sidebar gauche = pages
- Filtres = sidebar dans Exploration
- Multi-pages = automatique avec `/pages`

### Développement

```bash
# Mode debug
streamlit run app.py --logger.level debug

# Auto-reload
streamlit run app.py --server.runOnSave true
```

---

## 📞 Support

**En cas de problème** :

1. Vérifiez que les analyses NLP sont terminées
2. Vérifiez que `resultats_nlp/` existe
3. Consultez les logs Streamlit
4. Relancez l'app

---

## ✅ Checklist Avant Lancement

- [ ] Analyses NLP terminées
- [ ] `resultats_nlp/` existe avec fichiers
- [ ] Dépendances installées
- [ ] Port 8501 disponible

**Tout est OK ?** → `streamlit run app.py` 🚀

---

## 🎓 Projet Académique

Cette application fait partie du projet NLP Text Mining (Master SISE).

**Données** : 3000+ offres (France Travail + Indeed)  
**Analyses** : NLP, Topic Modeling, Clustering  
**Technologies** : Python, Streamlit, Plotly, DuckDB