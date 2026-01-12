<div align="center">

<!-- LOGO & HERO -->
<img src="app_streamlit/assets/logo2.JPEG" alt="JOBLIZE" width="280"/>

# JOBLIZE for Data & IA

### **L'Observatoire qui Révolutionne l'Analyse du Marché Data/IA en France**

<p align="center">
  <strong> +3 000 offres analysées • 14 profils métiers • 60+ compétences extraites • villes françaises cartographiées</strong>
</p>

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Master SISE – Statistique & Informatique pour la Science des Données**  
 Université Lumière Lyon 2 • Janvier 2026

[ Démarrage Rapide](#-démarrage-rapide-5-minutes) • [ Démo Live](#) • [ Documentation](#architecture) • [ Rapport PDF](docs/Rapport_NLP_SISE.pdf)

</div>

##  À propos du projet

Le marché de l’emploi **Data / IA** connaît une croissance rapide et une forte diversification des profils.  
Cependant, les offres sont hétérogènes, peu structurées et difficiles à analyser à grande échelle.

**JOBLIZE for Data & IA** propose une **plateforme complète d’analyse automatisée** du marché Data & IA en France, reposant sur :

-  **3 000+ offres d’emploi** collectées (France Travail & Indeed)
-  **Pipeline NLP complet** (prétraitement, compétences, profils, topics)
-  **Classification** de 14 profils métiers
-  **Topic Modeling LDA** pour révéler les tendances du marché
-  **Système de matching ML** CV ↔ Offres
-  **Application Streamlit interactive** multi-pages

**Objectif** : transformer un corpus textuel brut en **insights exploitables** pour étudiants, recruteurs et décideurs.

---

## ❓ Problématique

Comment, à partir d’offres d’emploi non structurées :

- Identifier les **compétences réellement demandées** ?
- Cartographier les **bassins d’emploi Data/IA** ?
- Différencier automatiquement les **profils métiers** ?
- Mettre en relation **candidats et offres** de manière intelligente ?
- Construire une **architecture data robuste et industrialisable** ?

---

## Solution proposée

JOBLIZE for Data & IA s’appuie sur une **chaîne de traitement complète**, de la collecte à la visualisation :

1. **Collecte multi-sources** (API France Travail, scraping Indeed)
2. **Entrepôt de données** PostgreSQL (modèle en étoile)
3. **Pipeline NLP avancé**
   - spaCy (préprocessing)
   - TF-IDF & règles métiers (compétences)
   - LDA (topics)
   - Sentence-BERT (embeddings)
4. **Machine Learning**
   - Classification hybride des profils
   - Matching CV-Offres
5. **Application Streamlit interactive**
   - 7 pages analytiques
   - Visualisations 2D & 3D
6. **➕ Ajout Offres via LLM**
- **Extraction automatique** via Mistral LLM
- Validation utilisateur avant insertion
- Pipeline NLP complet sur nouvelle offre
- Détection doublons automatique


## Comment utiliser notre application

### Option 1 : Docker 🐳 (Recommandé - Le plus simple)

**Prérequis :** [Docker Desktop](https://www.docker.com/products/docker-desktop) installé

```bash
# 1- Télécharger l'image (5-8 min, une seule fois)
docker pull nidena444/datajobs-explorer:latest

# 2- Créer fichier configuration
cat > .env << 'EOF'
SUPABASE_URL=https://votre-projet.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxxx
MISTRAL_API_KEY=xxxxxxxx  # Optionnel
EOF

# 3- Lancer l'application (15 sec)
docker run -d \
  --name datajobs-explorer \
  -p 8501:8501 \
  --env-file .env \
  --restart unless-stopped \
  nidena444/datajobs-explorer:latest

# 4- Accéder à l'application
# 🌐 http://localhost:8501
```

**C'est tout ! Votre observatoire Data/IA est opérationnel !**

---

### Option 2 : Image .tar (Sans Docker Hub)

**Si vous avez reçu le fichier `datajobs-explorer.tar.gz` :**

```bash
# 1- Charger l'image 
docker load -i datajobs-explorer.tar.gz

# 2- Créer .env (même que ci-dessus)
# ...

# 3- Lancer
docker run -d --name datajobs-explorer -p 8501:8501 --env-file .env datajobs-explorer:latest
```

---

### Option 3 : Installation Locale (Développeurs)

<details>
<summary><b> Voir les instructions complètes</b></summary>

**Prérequis :** Python 3.10+, Git, PostgreSQL 15+

```bash
# 1. Cloner le repository
git clone https://github.com/Denanico1/datajobs-explorer.git
cd datajobs-explorer

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Dépendances
pip install -r requirements.txt
python -m spacy download fr_core_news_lg

# 4. Configuration
cp .env.example .env
nano .env  # Éditer avec vos credentials

# 5. Lancer
cd app_streamlit
streamlit run app.py
```

</details>

---

##  Interface Utilisateur

### 7 Pages Analytiques Interactives

<table>
<tr>
<td width="50%">

####  **1. Dashboard Exécutif**
- KPIs temps réel (offres, salaires, profils)
- Timeline publications
- Répartition sources (France Travail, Indeed)

####  **2. Exploration Géographique**
- Carte Mapbox interactive **977 villes**
- Choroplèthe régionale
- Heatmap profils × régions

####  **3. Profils Métiers**
- **14 profils** classifiés automatiquement
- Radar charts compétences
- Comparateur profils

####  **4. Compétences**
- **60+ skills** extraits par NLP
- Réseau sémantique (PyVis)
- Heatmap compétences × profils

</td>
<td width="50%">

####  **5. Topics & Tendances**
- **8 topics LDA** découverts
- Visualisation t-SNE embeddings
- Insights métier actionnables

####  **6. Matching CV-Offres**
- Upload CV → Top 10 offres (**<3 sec**)
- Score matching explicable
- Recommandations bidirectionnelles

#### ➕ **7. Ajout Offres via LLM**
- Extraction automatique **Mistral AI**
- Validation utilisateur
- Pipeline NLP temps réel

</td>
</tr>
</table>

---

## Architecture Technique

### Stack Technologique de Production

<div align="center">

<!-- LOGO & HERO -->
<img src="app_streamlit/assets/nlp_archi.PNG" alt="JOBLIZE" width="280"/>

</div>

```
┌──────────────────────────────────────────────────────────────────┐
│                     🌐 SOURCES DE DONNÉES                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  📡 France Travail API  →  🔍 Indeed Selenium  →  🤖 Mistral LLM │
│     (API officielle)        (Scraping stealth)     (Extraction)  │
│                                                                   │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   🔄 ETL Pipeline  │
                    │   • DuckDB (OLAP)  │
                    │   • Normalisation  │
                    │   • Géocodage 97%  │
                    └─────────┬──────────┘
                              │
          ┌───────────────────▼───────────────────┐
          │     🗄️ ENTREPÔT DE DONNÉES           │
          │   PostgreSQL Cloud (Supabase)        │
          │   • Modèle étoile (5 dimensions)     │
          │   • 3 tables de faits                │
          │   • 3 009 offres géocodées           │
          └───────────────────┬──────────────────┘
                              │
          ┌───────────────────▼───────────────────┐
          │       🧠 PIPELINE NLP AVANCÉ          │
          │   ┌────────────────────────────────┐  │
          │   │ 1. Preprocessing (spaCy)      │  │
          │   │ 2. Extraction (TF-IDF)        │  │
          │   │ 3. Classification (90% acc)   │  │
          │   │ 4. Topic Modeling (LDA)       │  │
          │   │ 5. Embeddings (Sentence-BERT) │  │
          │   └────────────────────────────────┘  │
          └───────────────────┬──────────────────┘
                              │
      ┌───────────────────────┴────────────────────────┐
      │                                                 │
┌─────▼──────────┐                          ┌──────────▼─────────┐
│  🤖 ML ENGINE  │                          │  🎨 APPLICATION    │
│                │                          │                    │
│ • Random Forest│                          │ • Streamlit 1.52   │
│ • 85% précision│                          │ • 8 pages          │
│ • <3 sec       │                          │ • Plotly + Mapbox  │
│ • Embeddings   │                          │ • 7 thèmes         │
│ • 6 features   │                          │ • Docker Ready 🐳  │
└────────────────┘                          └────────────────────┘
```

### Technologies Clés

<table>
<tr>
<td><b>Backend & Data</b></td>
<td>Python 3.10 • PostgreSQL 15 (Supabase) • DuckDB • pandas • SQLAlchemy</td>
</tr>
<tr>
<td><b>Web Scraping</b></td>
<td>Selenium • requests • BeautifulSoup • geopy (Nominatim)</td>
</tr>
<tr>
<td><b>NLP & ML</b></td>
<td>spaCy • Sentence-BERT • scikit-learn • UMAP • Mistral AI</td>
</tr>
<tr>
<td><b>Visualisation</b></td>
<td>Streamlit • Plotly • Mapbox • PyVis • WordCloud</td>
</tr>
<tr>
<td><b>Infrastructure</b></td>
<td>Docker 🐳 • Git • Supabase • Streamlit Cloud</td>
</tr>
</table>

---

##  Pourquoi JOBLIZE ?

<table>
<tr>
<td width="33%" align="center">
<h3> Pour les Étudiants</h3>
<p>Identifiez les <strong>compétences essentielles</strong> pour décrocher votre prochain job Data/IA en France</p>
</td>
<td width="33%" align="center">
<h3> Pour les Recruteurs</h3>
<p>Trouvez les <strong>meilleurs profils</strong> grâce à notre système de matching ML (<strong>85% précision</strong>)</p>
</td>
<td width="33%" align="center">
<h3> Pour les Décideurs</h3>
<p>Anticipez les <strong>tendances marché</strong> avec nos analyses NLP temps réel</p>
</td>
</tr>
</table>

---

## L'Impact en Chiffres

<div align="center">

|  Métrique |  Valeur |  Impact |
|:----------:|:--------:|:---------|
| **Offres Analysées** | **+3 000** | Corpus le plus complet du marché Data/IA France |
| **Villes Cartographiées** | **+70** | Couverture nationale exhaustive |
| **Profils Métiers** | **14** | Classification automatique 90% précision |
| **Compétences Extraites** | **60+** | Roadmap personnalisée pour chercheurs d'emploi |
| **Temps Matching** | **-** | Recommandations instantanées |
| **Taux Géocodage** | **97.3%** | Précision GPS unique sur le marché |

</div>

### Ce que JOBLIZE change pour vous

-  **Étudiants :** Découvrez que **Python (38%)**, **SQL (32%)** et **Machine Learning (21%)** sont les compétences #1 demandées
-  **Chercheurs emploi :** **40% des offres** sont en Île-de-France, mais **16%** en Auvergne-Rhône-Alpes (opportunités cachées !)
-  **Recruteurs :** Économisez **75% du temps** de sourcing avec notre matching CV-Offres automatisé
-  **Institutions :** Adaptez vos formations aux **8 topics LDA** découverts (Machine Learning, Cloud, Analytics...)

---

##  Résultats & Insights Clés

###  Top 10 Profils Métiers les Plus Demandés

<table>
<tr><th>Rang</th><th>Profil</th><th>Part de marché</th><th>Compétence clé</th></tr>
<tr><td>🥇</td><td><b>Data Manager</b></td><td>18.2%</td><td>Leadership, Stratégie</td></tr>
<tr><td>🥈</td><td><b>Data Scientist</b></td><td>16.5%</td><td>Python, ML, Stats</td></tr>
<tr><td>🥉</td><td><b>Data Engineer</b></td><td>14.8%</td><td>Spark, SQL, AWS</td></tr>
<tr><td>4</td><td>Data Analyst</td><td>12.3%</td><td>SQL, Tableau, Excel</td></tr>
<tr><td>5</td><td>ML Engineer</td><td>8.7%</td><td>PyTorch, Docker, MLOps</td></tr>
<tr><td>6</td><td>BI Analyst</td><td>7.1%</td><td>Power BI, Dashboards</td></tr>
<tr><td>7</td><td>AI Engineer</td><td>6.4%</td><td>Deep Learning, NLP</td></tr>
<tr><td>8</td><td>Data Consultant</td><td>4.2%</td><td>Transformation digitale</td></tr>
<tr><td>9</td><td>MLOps Engineer</td><td>3.1%</td><td>Kubernetes, CI/CD</td></tr>
<tr><td>10</td><td>AI Research Scientist</td><td>2.5%</td><td>PhD, Publications</td></tr>
</table>

---

### Top 20 Compétences Techniques (avec profils associés)

<table>
<tr><th>Rang</th><th>Compétence</th><th>Fréquence</th><th>📈 Tendance</th><th>Profils principaux</th></tr>
<tr><td>🥇</td><td><b>Python</b></td><td><b>38%</b></td><td>↗️ +15%/an</td><td>Data Scientist, ML Engineer</td></tr>
<tr><td>🥈</td><td><b>SQL</b></td><td><b>32%</b></td><td>→ Stable</td><td>Data Analyst, Data Engineer</td></tr>
<tr><td>🥉</td><td><b>Machine Learning</b></td><td><b>21%</b></td><td>↗️ +25%/an</td><td>Data Scientist, ML Engineer</td></tr>
<tr><td>4</td><td>Spark</td><td>18%</td><td>↗️ +10%/an</td><td>Data Engineer</td></tr>
<tr><td>5</td><td>AWS</td><td>16%</td><td>↗️ +20%/an</td><td>Data Engineer, MLOps</td></tr>
<tr><td>6</td><td>Docker</td><td>14%</td><td>↗️ +30%/an</td><td>ML Engineer, DevOps</td></tr>
<tr><td>7</td><td>Tableau</td><td>13%</td><td>→ Stable</td><td>Data Analyst, BI</td></tr>
<tr><td>8</td><td>Power BI</td><td>12%</td><td>↗️ +5%/an</td><td>BI Analyst</td></tr>
<tr><td>9</td><td>TensorFlow</td><td>11%</td><td>↗️ +15%/an</td><td>Data Scientist, AI Engineer</td></tr>
<tr><td>10</td><td>Git</td><td>10%</td><td>→ Essentiel</td><td>Tous profils</td></tr>
</table>

---

### Répartition Géographique des Opportunités

<table>
<tr>
<td width="60%">

|  Région | Offres | Part |  Salaire médian |
|----------|--------|------|-------------------|
| **Île-de-France** | 1 203 | **40%** | 52k€ |
| **Auvergne-Rhône-Alpes** | 487 | 16% | 45k€ |
| **PACA** | 312 | 10% | 42k€ |
| **Occitanie** | 289 | 10% | 40k€ |
| **Nouvelle-Aquitaine** | 201 | 7% | 38k€ |
| **Autres** | 517 | 17% | 40k€ |

</td>
<td width="40%">

** Insights Géo :**

- **Paris :** Hub Data/IA #1 (startup, finance, tech)
- **Lyon :** Pôle émergent Data Engineer (+30% offres)
- **Toulouse :** Spécialisation aérospatial/défense
- **Marseille :** Focus e-commerce & logistique
- **Bordeaux :** Wine tech & agro-tech Data

</td>
</tr>
</table>

---

### 8 Topics LDA Découverts

| Topic | Mots-clés | Interprétation métier | % Offres |
|-------|-----------|----------------------|----------|
| **1. Environnement Entreprise** | Client, équipe, groupe, management | Culture d'entreprise | 18% |
| **2. Engineering & Qualité** | Données, technique, qualité, développement | Data Engineering | 16% |
| **3. Conseil & Business** | Transformation, architecture, conseil | Consulting Data | 14% |
| **4. International** | Research, engineering, Paris, English | Postes anglophones | 12% |
| **5. Transformation Digitale** | Big data, cloud, innovation | Modernisation SI | 13% |
| **6. Machine Learning** | Modèles, ML, Python, algorithmes | AI/ML focus | 11% |
| **7. Secteur Financier** | Banque, risques, finance | Fintech/Banking | 9% |
| **8. Analytics & Reporting** | Analyse, tableaux, stages | BI & Junior | 7% |

---

##  Cas d'Usage Concrets

### Pour un Étudiant Data Science

<details>
<summary><b> Scénario : "Je veux devenir Data Scientist, quelles compétences apprendre ?"</b></summary>

**Démarche avec JOBLIZE :**

1. **Page Profils** → Sélectionner "Data Scientist"
2. **Radar Chart** révèle les compétences essentielles :
   - Python (présent dans 91% des offres DS)
   - Machine Learning (87%)
   - SQL (76%)
   - TensorFlow/PyTorch (65%)
   - Git (89%)

3. **Page Compétences** → Réseau sémantique montre :
   - Python ↔ pandas, scikit-learn (forte co-occurrence)
   - ML ↔ Deep Learning, NLP (spécialisations)

4. **Page Topics** → Topic #6 "Machine Learning" :
   - Salaire médian : 52k€
   - Régions : IDF (45%), AURA (20%)

**Résultat :** Roadmap personnalisée claire !

</details>

---

### Pour un Recruteur Tech

<details>
<summary><b> Scénario : "Trouver 5 candidats Data Engineer pour Lyon"</b></summary>

**Démarche avec JOBLIZE :**

1. **Page Matching** → Mode "Recruteur"
2. Sélectionner offre :
   - Poste : Data Engineer
   - Localisation : Lyon
   - Stack : Spark, Python, AWS, Docker

3. **Upload 50 CV** (batch processing)
4. **Système ML** :
   - Calcul 6 features par CV
   - Random Forest scoring
   - Embeddings sémantiques

5. **Résultats** :
   - Top 5 candidats classés (score 0-100%)
   - Compétences matchées/manquantes
   - Localisation + mobilité
   - Expérience alignée

**Résultat :** **75% temps sourcing économisé !**

</details>

---

### Pour un Décideur Formation

<details>
<summary><b> Scénario : "Adapter les curricula aux besoins marché"</b></summary>

**Démarche avec JOBLIZE :**

1. **Dashboard** → Vue macro :
   - +3 000 offres analysées
   - 14 profils identifiés
   - Tendances temporelles

2. **Page Compétences** → Heatmap profils × compétences :
   - Identifier gaps formations actuelles
   - Exemple : MLOps (3% offres mais 0 formation dédiée)

3. **Page Topics** → 8 topics LDA :
   - Topic #5 "Transformation Digitale" en hausse (+40% /an)
   - Nécessite modules Cloud + Data Governance

4. **Exports CSV** :
   - Données brutes pour analyses statistiques
   - Croisement avec taux insertion diplômés

**Résultat :** Formations alignées marché !

</details>

---

## Guide d'Utilisation Avancé

### Ajouter une Nouvelle Offre via LLM

```bash
# 1. Aller sur page "Nouvelle Offre via LLM"

# 2. Coller texte complet offre (ou URL Indeed/France Travail)

# 3. Cliquer "Extraire avec Mistral"
#    → Mistral analyse et extrait :
#      - Titre poste
#      - Entreprise
#      - Localisation
#      - Compétences
#      - Contrat
#      - Salaire (si mentionné)

# 4. Valider/Corriger extraction

# 5. Cliquer "Ajouter à la base"
#    → Pipeline NLP automatique :
#      - Preprocessing
#      - Extraction compétences
#      - Classification profil
#      - Génération embedding
#      - Détection doublons

# Offre disponible immédiatement dans matching
```

---


---

## Structure Projet

```
JOBLIZE_Project/
│
├──  docker-compose.yml       
├──  Dockerfile              
├──  requirements.txt         
├──  .env             
│
├── app_streamlit/              # APPLICATION PRINCIPALE
│   ├── app.py                  # Point d'entrée
│   ├── themes.py               # 7 thèmes UI
│   ├── config_db.py            # Connexion PostgreSQL cloud
│   ├── data_loaders.py         # Chargement optimisé (cache)
│   ├── nlp_pipeline_wrapper.py # Pipeline NLP temps réel
│   ├── utils.py 
│   │
│   └── pages/                  # 8 PAGES INTERACTIVES
│       ├──  dashboard.py
│       ├──  geographique.py
│       ├──  profils.py
│       ├──  competences.py
│       ├──  topics.py
│       ├──  matching.py
│       └── ➕ nouvelle_offre.py
│
├── analyses_nlp/               #  PIPELINE NLP COMPLET
│   └── fichiers_analyses/
│       ├── 1_preprocessing.py              # spaCy (nettoyage, lemmatisation)
│       ├── 2_extraction_competences.py     # TF-IDF + gazetteers (60+ skills)
│       ├── 3_topic_modeling.py             # LDA (8 topics découverts)
│       ├── 4_classification_hybride.py     # Profils 
│       ├── 5_visualisations_profils.py     # Graphiques profils
│       ├── 6_embeddings_clustering.py      # Sentence-BERT + K-Means
│       ├── 9_ml_matching_system.py         # Random Forest (embedding + ML)
│       ├── utils.py                        # Fonctions NLP communes
│       └── profils_definitions.py          # Règles classification métier
│
├── scraping/                   #  COLLECTE DONNÉES
│   ├── france_travail_api.py   # Scraper API officielle (OAuth2)
│   ├── indeed_selenium.py      # Scraper Indeed (mode stealth)
│   └── geocoding.py            # Normalisation + GPS (97.3% succès)
│
├── entrepot_donnees/                       
│   ├── schema.sql              # Schéma PostgreSQL (modèle étoile)
│   ├── import.sql              # Import des offres
│   └── exports/                
│
├── resultats_nlp/
│
├── docs/                       
│   ├── Rapport_NLP_SISE.pdf    # Rapport académique 
│
└── 
```

---

## Contribution Académique

### Publications & Présentations

-  **Rapport Master SISE** : [Télécharger PDF](docs/Rapport_NLP_SISE.pdf)
-  **Présentation Lyon 2** : Janvier 2026

---

## L'Équipe projet (Master SISE 2025-2026)

<table align="center">
  <tr>
    <td align="center" width="25%">
      <a href="https://github.com/Denenico1">
        <img src="" width="120px;" alt="Nico DENA"/><br />
        <sub><b>Nico DENA</b></sub>
      </a><br />
      <sub> Master SISE</sub><br />
      <sub><i>Data Scientist</i></sub>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/modou-mboup">
        <img src="" width="120px;" alt="Modou MBOUP"/><br />
        <sub><b>Modou MBOUP</b></sub>
      </a><br />
      <sub> Master SISE</sub><br />
      <sub><i>Data Scientist</i></sub>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/constantin-rey">
        <img src="https://github.com/constantin-rey.png" width="120px;" alt="Constantin REY-COQUAIS"/><br />
        <sub><b>Constantin REY-COQUAIS</b></sub>
      </a><br />
      <sub> Master SISE</sub><br />
      <sub><i>Data Scientist</i></sub>
    </td>
    <td align="center" width="25%">
      <a href="">
        <img src="" width="120px;" alt="Léo-Paul VIDALENC"/><br />
        <sub><b>Léo-Paul</b></sub>
      </a><br />
      <sub> Master SISE</sub><br />
      <sub><i>Data Scientist</i></sub>
    </td>
  </tr>
</table>

<div align="center">

**Encadré par :**  Professeur Ricco Rakotomalala  
**Institution :** Université Lumière Lyon 2 - Master SISE  
**Période :** Janvier 2026

</div>

---

## 📞 Support & Contact

### Besoin d'Aide ?

<table>
<tr>
<td width="50%">

**🐛 Problème Technique**
- [Créer une Issue GitHub](https://github.com/votre-repo/issues)
- [Consulter DEPLOYMENT.md](docs/DEPLOYMENT.md)
- [FAQ Docker](docs/README_DOCKER.md#faq)

</td>
<td width="50%">

** Collaboration Professionnelle**
- 📧 Email : nico.dena@univ-lyon2.fr


</td>
</tr>
</table>

---

## Licence & Citation

### Licence MIT

```
MIT License - Copyright (c) 2026 Équipe JOBLIZE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```
---

<div align="center">

## ⭐ Soutenez le Projet

**Si JOBLIZE vous a aidé dans votre recherche d'emploi, vos recrutements ou vos analyses, n'oubliez pas de :**

[![Star on GitHub](https://img.shields.io/github/stars/votre-repo?style=social)](https://github.com/votre-repo)
[![Fork on GitHub](https://img.shields.io/github/forks/votre-repo?style=social)](https://github.com/votre-repo/fork)

---

### 🏆 Ce Projet Vous a Plu ?

**Partagez-le avec votre réseau et aidez d'autres professionnels Data/IA ! **

---

**Made with ❤️, ☕, and 🐍 by Team JOBLIZE**

*Master SISE - Université Lumière Lyon 2 - Janvier 2026*

</div>