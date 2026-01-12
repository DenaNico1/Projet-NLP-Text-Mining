# 🐳 DataJobs Explorer - Docker Quick Start

**Application Streamlit d'analyse NLP du marché Data/IA en France**

---

## ⚡ Démarrage Rapide (5 minutes)

### **1. Télécharger l'image**

```bash
docker pull nicodena/datajobs-explorer:latest
```

### **2. Créer fichier `.env`**

```bash
cat > .env << EOF
SUPABASE_URL=https://votre-projet.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxxx
MISTRAL_API_KEY=xxxxxxxx
EOF
```

**💡 Obtenir credentials Supabase :**
- Se connecter sur [supabase.com](https://supabase.com)
- Projet > Settings > API > Copier URL + anon key

### **3. Lancer l'application**

```bash
docker run -d \
  --name datajobs-explorer \
  -p 8501:8501 \
  --env-file .env \
  --restart unless-stopped \
  nicodena/datajobs-explorer:latest
```

### **4. Accéder à l'application**

**Ouvrir navigateur :** http://localhost:8501

✅ **L'application se charge en 15-20 secondes !**

---

## 📦 Contenu de l'image

- **Application** : Streamlit 8 pages interactives
- **Base de données** : PostgreSQL Supabase (cloud)
- **Données** : 3 009 offres Data/IA France
- **NLP** : spaCy, Sentence-BERT, LDA
- **Modèles** : Random Forest matching, embeddings 384D

---

## 🎯 Fonctionnalités

1. **Dashboard** - KPIs marché emploi Data/IA
2. **Exploration Géographique** - Cartes interactives 977 villes
3. **Profils Métiers** - 14 profils classifiés (Data Scientist, Engineer...)
4. **Compétences** - 60+ skills techniques (Python, SQL, ML...)
5. **Topics & Tendances** - 8 topics LDA découverts
6. **Matching CV-Offres** - Recommandations ML personnalisées
7. **Visualisations 3D** - UMAP/t-SNE embeddings
8. **Ajout Offres LLM** - Extraction automatique Mistral

---

## 🛠️ Commandes Utiles

```bash
# Voir logs
docker logs datajobs-explorer

# Arrêter
docker stop datajobs-explorer

# Redémarrer
docker restart datajobs-explorer

# Supprimer
docker rm -f datajobs-explorer

# Entrer dans conteneur
docker exec -it datajobs-explorer bash
```

---

## 📚 Documentation Complète

**Guide détaillé :** [DEPLOYMENT.md](DEPLOYMENT.md)

**Contient :**
- Installation complète (3 options)
- Configuration avancée
- Troubleshooting
- Performances
- Sécurité

---

## 🔗 Liens

- **Docker Hub** : https://hub.docker.com/r/nicodena/datajobs-explorer
- **GitHub** : https://github.com/nicodena/datajobs-explorer
- **Rapport PDF** : [Rapport_NLP_SISE.pdf](docs/Rapport_NLP_SISE.pdf)

---

## 👥 Auteurs

**Master SISE - Université Lumière Lyon 2**

- Nico DENA
- Modou MBOUP
- Constantin REY-COQUAIS
- Léo-Paul VIDALENC

**Encadrant :** Ricco Rakotomalala

**Janvier 2026**

---

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE)

---

**🎉 Happy Data Analyzing !**