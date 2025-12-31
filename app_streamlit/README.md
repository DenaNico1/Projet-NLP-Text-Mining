# 📊 DataJobs Explorer - Application Streamlit Premium

Application d'analyse NLP du marché Data/IA en France (3,003 offres).

## 🚀 Installation

```bash
cd app_streamlit
pip install -r requirements.txt
```

## 📦 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
folium>=0.14.0
streamlit-folium>=0.15.0
```

## 🎯 Lancement

```bash
streamlit run app.py
```

L'application s'ouvre à http://localhost:8501

## 📋 Structure

```
app_streamlit/
├── app.py (main)
├── config.py
├── pages/
│   ├── dashboard.py (🏠 Accueil)
│   ├── geographique.py (🗺️)
│   ├── profils.py (💼)
│   ├── competences.py (🎓)
│   ├── topics.py (🔬)
│   ├── viz_3d.py (🌐)
│   └── insights.py (📊)
└── utils/
    └── helpers.py
```

## 🎨 Features

- **Dashboard** : KPIs, timeline, top compétences
- **Géo** : Carte France interactive, heatmap régions
- **Profils** : 14 profils métiers, comparateur
- **Compétences** : Réseau sémantique, UMAP 3D
- **Topics** : LDA, wordclouds, tendances
- **3D** : Projections embeddings interactives
- **Insights** : Clustering, salaires, qualité

## 💡 Utilisation

1. Filtres globaux dans sidebar
2. Navigation par icônes
3. Visualisations interactives Plotly
4. Export PNG/HTML disponible

Enjoy ! 🚀
