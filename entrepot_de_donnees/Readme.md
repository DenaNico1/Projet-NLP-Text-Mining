# 🏛️ ENTREPÔT DE DONNÉES - DataTalent Observatory

**Système de stockage et gestion des offres d'emploi Data/IA en France**

---

## 📋 VUE D'ENSEMBLE

Cet entrepôt de données implémente un **Data Warehouse** optimisé pour l'analyse NLP et Machine Learning des offres d'emploi Data/IA. Il utilise **DuckDB** comme SGBD analytique pour des performances élevées sur des requêtes complexes.

### 🎯 Objectifs

- ✅ Centraliser toutes les sources de données (France Travail, Indeed, LinkedIn, APEC)
- ✅ Normaliser et dédoublonner les offres
- ✅ Extraire automatiquement les compétences (NLP)
- ✅ Structurer selon modèle en étoile (Star Schema)
- ✅ Faciliter analyses ML et visualisations

---

## 🏗️ ARCHITECTURE

### **Modèle Dimensionnel (Star Schema)**

```
                    ┌─────────────────────┐
                    │   dim_entreprises   │
                    ├─────────────────────┤
                    │ entreprise_id   PK  │
                    │ nom                 │
                    │ secteur             │
                    │ taille              │
                    │ site_web            │
                    └─────────────────────┘
                            ▲
                            │
┌─────────────────────┐    │    ┌─────────────────────┐
│  dim_localisation   │    │    │   dim_competences   │
├─────────────────────┤    │    ├─────────────────────┤
│ localisation_id  PK │    │    │ competence_id    PK │
│ ville               │    │    │ nom                 │
│ departement         │    │    │ categorie           │
│ region              │    │    │ type                │
│ latitude            │    │    │ freq_globale        │
│ longitude           │    │    └─────────────────────┘
└─────────────────────┘    │             ▲
         ▲                 │             │
         │                 │             │
         │    ┌────────────┴──────────────────────┐
         │    │        FAITS_OFFRES               │
         │    ├───────────────────────────────────┤
         └────┤ offre_id                 PK       │
              ├───────────────────────────────────┤
              │ entreprise_id            FK       │
              │ localisation_id          FK       │
              │ titre                             │
              │ description                       │
              │ type_contrat                      │
              │ niveau_experience                 │
              │ salaire_min, salaire_max          │
              │ date_publication                  │
              │ url                               │
              │ source_name                       │
              │                                   │
              │ ── NLP / ML ──                    │
              │ description_clean                 │
              │ tokens                            │
              │ competences_found         (JSON)  │
              │ profil                            │
              │ methode_classification            │
              │ confiance                         │
              │ topic_dominant                    │
              └───────────────────────────────────┘
                            │
                            │
                            ▼
              ┌─────────────────────────┐
              │ rel_offres_competences  │
              ├─────────────────────────┤
              │ offre_id            FK  │
              │ competence_id       FK  │
              │ freq_offre              │
              │ tf_idf_score            │
              └─────────────────────────┘
```

---

## 📊 SCHÉMA DÉTAILLÉ

### **Table de Faits : `faits_offres`**

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `offre_id` | VARCHAR (PK) | Identifiant unique | `"ft_123456"` |
| `job_id_source` | VARCHAR | ID source externe | `"offer_789"` |
| `source_name` | VARCHAR | Source collecte | `"france_travail"` |
| `entreprise_id` | INTEGER (FK) | → dim_entreprises | `42` |
| `localisation_id` | INTEGER (FK) | → dim_localisation | `75` |
| `titre` | VARCHAR | Titre offre | `"Data Scientist Senior"` |
| `description` | TEXT | Description complète | `"Nous recherchons..."` |
| `type_contrat` | VARCHAR | CDI, CDD, Stage... | `"CDI"` |
| `niveau_experience` | VARCHAR | Junior, Senior... | `"5-10 ans"` |
| `duree` | VARCHAR | Durée contrat | `"12 mois"` |
| `salaire_min` | DECIMAL | Salaire minimum annuel | `45000` |
| `salaire_max` | DECIMAL | Salaire maximum annuel | `60000` |
| `salaire_median` | DECIMAL | Médiane calculée | `52500` |
| `salaire_text` | VARCHAR | Texte original | `"45-60k€"` |
| `date_publication` | DATE | Date publi offre | `2024-12-15` |
| `date_scraping` | TIMESTAMP | Date collecte | `2024-12-27 10:30:00` |
| `url` | VARCHAR | Lien offre | `"https://..."` |
| **NLP / ML** | | | |
| `description_clean` | TEXT | Texte nettoyé | (lowercased, sans stopwords) |
| `tokens` | VARCHAR | Tokens extraits (JSON) | `["python", "sql", ...]` |
| `num_tokens` | INTEGER | Nombre tokens | `450` |
| `competences_found` | VARCHAR | Compétences (JSON) | `["Python", "SQL", ...]` |
| `num_competences` | INTEGER | Nombre compétences | `12` |
| `profil` | VARCHAR | Profil hybride | `"Data Scientist"` |
| `methode_classification` | VARCHAR | Méthode classif | `"titre"` |
| `confiance` | VARCHAR | Niveau confiance | `"haute"` |
| `topic_dominant` | INTEGER | Topic LDA | `2` |
| `topic_score` | DECIMAL | Score topic | `0.78` |

---

### **Dimension : `dim_entreprises`**

| Colonne | Type | Description |
|---------|------|-------------|
| `entreprise_id` | INTEGER (PK) | ID auto-incrémenté |
| `nom` | VARCHAR | Nom entreprise |
| `secteur` | VARCHAR | Secteur activité |
| `taille` | VARCHAR | Effectif |
| `site_web` | VARCHAR | URL site |

---

### **Dimension : `dim_localisation`**

| Colonne | Type | Description |
|---------|------|-------------|
| `localisation_id` | INTEGER (PK) | ID auto-incrémenté |
| `ville` | VARCHAR | Ville |
| `code_postal` | VARCHAR | Code postal |
| `departement` | VARCHAR | Département |
| `region` | VARCHAR | Région |
| `latitude` | DECIMAL | Coordonnée GPS |
| `longitude` | DECIMAL | Coordonnée GPS |

---

### **Dimension : `dim_competences`**

| Colonne | Type | Description |
|---------|------|-------------|
| `competence_id` | INTEGER (PK) | ID auto-incrémenté |
| `nom` | VARCHAR | Nom compétence |
| `categorie` | VARCHAR | Langage, Framework, Tool... |
| `type` | VARCHAR | Technique, Soft skill... |
| `freq_globale` | INTEGER | Fréquence dans corpus |

---

### **Relation : `rel_offres_competences`**

| Colonne | Type | Description |
|---------|------|-------------|
| `offre_id` | VARCHAR (FK) | → faits_offres |
| `competence_id` | INTEGER (FK) | → dim_competences |
| `freq_offre` | INTEGER | Nombre occurrences |
| `tf_idf_score` | DECIMAL | Score TF-IDF |

---

## 🗂️ STRUCTURE FICHIERS

```
entrepot_de_donnees/
│
├── entrepot_nlp.duckdb           # Base de données DuckDB
├── README.md                     # Ce fichier
│
├── scripts/
│   ├── create_schema.sql         # Création schéma DDL
│   ├── import_data.py            # Import données brutes
│   ├── extract_competences.py    # Extraction NLP
│   ├── update_profiles.py        # Mise à jour profils (hybride)
│   └── export_to_csv.py          # Export pour analyses
│
├── data/
│   ├── raw/                      # Données brutes (JSON)
│   │   ├── france_travail/
│   │   ├── indeed/
│   │   └── linkedin/
│   │
│   └── processed/                # Données traitées
│       ├── offres_deduplicated.csv
│       └── competences_extracted.csv
│
└── queries/
    ├── stats_globales.sql        # Requêtes statistiques
    ├── top_competences.sql
    └── analyse_geo.sql
```

---

## 🚀 UTILISATION

### **1. Connexion à la base**

```python
import duckdb

# Connexion
con = duckdb.connect('entrepot_de_donnees/entrepot_nlp.duckdb')

# Requête simple
result = con.execute("""
    SELECT profil, COUNT(*) as nb_offres
    FROM faits_offres
    GROUP BY profil
    ORDER BY nb_offres DESC
""").df()

print(result)
```

---

### **2. Import nouvelles données**

```bash
# Collecter données
python collect_indeed.py
python collect_linkedin.py

# Importer dans entrepôt
python scripts/import_data.py --source indeed
python scripts/import_data.py --source linkedin

# Extraire compétences
python scripts/extract_competences.py

# Classifier profils (système hybride)
python scripts/update_profiles.py
```

---

### **3. Requêtes analytiques**

#### **Statistiques globales**

```sql
SELECT 
    COUNT(*) as total_offres,
    COUNT(DISTINCT entreprise_id) as nb_entreprises,
    COUNT(DISTINCT localisation_id) as nb_villes,
    AVG(salaire_median) as salaire_moyen,
    AVG(num_competences) as competences_moyennes
FROM faits_offres;
```

#### **Top 10 compétences**

```sql
SELECT 
    c.nom,
    COUNT(*) as nb_offres,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM faits_offres), 2) as pct
FROM rel_offres_competences roc
JOIN dim_competences c ON c.competence_id = roc.competence_id
GROUP BY c.nom
ORDER BY nb_offres DESC
LIMIT 10;
```

#### **Distribution profils par région**

```sql
SELECT 
    l.region,
    f.profil,
    COUNT(*) as nb_offres,
    AVG(f.salaire_median) as salaire_moyen
FROM faits_offres f
JOIN dim_localisation l ON l.localisation_id = f.localisation_id
WHERE f.profil IS NOT NULL
GROUP BY l.region, f.profil
ORDER BY l.region, nb_offres DESC;
```

#### **Analyse spatio-temporelle**

```sql
SELECT 
    l.region,
    DATE_TRUNC('month', f.date_publication) as mois,
    COUNT(*) as nb_offres,
    AVG(f.salaire_median) as salaire_moyen
FROM faits_offres f
JOIN dim_localisation l ON l.localisation_id = f.localisation_id
WHERE f.date_publication >= '2024-01-01'
GROUP BY l.region, DATE_TRUNC('month', f.date_publication)
ORDER BY mois, l.region;
```

---

## 📈 PERFORMANCES

### **Optimisations appliquées**

- ✅ **Index** sur colonnes clés (offre_id, entreprise_id, localisation_id)
- ✅ **Compression columnar** DuckDB (ratio ~5:1)
- ✅ **Partitionnement temporel** (par mois de publication)
- ✅ **Statistiques** mises à jour automatiquement

### **Benchmarks**

| Requête | Temps (3k offres) | Temps (50k offres estimé) |
|---------|-------------------|----------------------------|
| Stats globales | 15 ms | 80 ms |
| Top compétences | 25 ms | 120 ms |
| Profils × régions | 40 ms | 200 ms |
| Analyse temporelle | 60 ms | 300 ms |

---

## 🔄 PIPELINE DE DONNÉES

```
┌─────────────────────────────────────────────────────────────┐
│ 1. COLLECTE (Scraping / API)                                │
│    - France Travail API                                      │
│    - Indeed (Selenium)                                       │
│    - LinkedIn, APEC, etc.                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. NORMALISATION (ETL)                                       │
│    - Parsing salaires                                        │
│    - Géocodage (API Nominatim)                              │
│    - Dédoublonnage (URL, hash)                              │
│    - Uniformisation format                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ENRICHISSEMENT NLP                                        │
│    - Tokenization (NLTK)                                     │
│    - Extraction compétences (770 patterns)                   │
│    - Classification profils (Système hybride 3 couches)      │
│    - Topic modeling (LDA k=6)                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. STOCKAGE ENTREPÔT (DuckDB)                               │
│    - Insertion dans faits_offres                             │
│    - Mise à jour dimensions                                  │
│    - Création relations                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ANALYSES & VISUALISATION                                  │
│    - Streamlit (interface web)                               │
│    - Notebooks Jupyter (analyses ad-hoc)                     │
│    - Exports CSV/JSON (partage)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ MAINTENANCE

### **Tâches quotidiennes**

```bash
# Collecter nouvelles offres
python collect_all_sources.py

# Mise à jour entrepôt
python scripts/import_data.py --incremental
python scripts/extract_competences.py --new-only
python scripts/update_profiles.py
```

### **Tâches hebdomadaires**

```bash
# Nettoyage offres expirées (>3 mois)
python scripts/cleanup_old_offers.py --threshold 90

# Recalcul statistiques
python scripts/recompute_stats.py

# Backup
python scripts/backup_db.py --output backups/
```

### **Tâches mensuelles**

```bash
# Vérification qualité données
python scripts/data_quality_check.py

# Export pour archivage
python scripts/export_to_csv.py --month 2024-12
```

---

## 📊 STATISTIQUES ACTUELLES

| Métrique | Valeur |
|----------|--------|
| **Total offres** | 3,023 |
| **Entreprises** | 1,450 |
| **Villes** | 312 |
| **Régions** | 13 |
| **Compétences uniques** | 770 |
| **Profils métiers** | 14 |
| **Taille DB** | 28 MB |
| **Période couverte** | Déc 2024 |

---

## 🔗 INTÉGRATIONS

### **Sources de données**

- ✅ France Travail API (officielle)
- ✅ Indeed (web scraping)
- 🟡 LinkedIn (à implémenter)
- 🟡 APEC (à implémenter)
- 🟡 WelcomeToTheJungle (à implémenter)

### **Outils analytiques**

- ✅ Streamlit (DataTalent Observatory)
- ✅ Jupyter Notebooks
- ✅ Plotly/Matplotlib
- 🟡 Power BI / Tableau (export CSV)
- 🟡 API REST (à développer)

---

## 🐛 TROUBLESHOOTING

### **Problème : Base corrompue**

```bash
# Vérifier intégrité
python scripts/check_integrity.py

# Restaurer depuis backup
cp backups/entrepot_nlp_2024-12-27.duckdb entrepot_de_donnees/
```

### **Problème : Lenteur requêtes**

```sql
-- Vérifier statistiques
ANALYZE faits_offres;

-- Vérifier index
SHOW INDEXES FROM faits_offres;

-- Reconstruire index si nécessaire
DROP INDEX idx_offre_id;
CREATE INDEX idx_offre_id ON faits_offres(offre_id);
```

### **Problème : Doublons**

```sql
-- Identifier doublons
SELECT url, COUNT(*) as nb
FROM faits_offres
GROUP BY url
HAVING COUNT(*) > 1;

-- Supprimer doublons (garder plus récent)
DELETE FROM faits_offres
WHERE offre_id IN (
    SELECT offre_id FROM (
        SELECT offre_id,
               ROW_NUMBER() OVER (PARTITION BY url ORDER BY date_scraping DESC) as rn
        FROM faits_offres
    ) WHERE rn > 1
);
```

---

## 📚 DOCUMENTATION COMPLÉMENTAIRE

- 📄 **DOCUMENTATION_ENTREPOT.md** : Documentation technique complète
- 📄 **create_schema.sql** : Script de création DDL
- 📄 **queries/** : Requêtes SQL utiles
- 📄 **../README_DATATALENT_OBSERVATORY.md** : Documentation application Streamlit

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

**🏛️ DataTalent Observatory - Entrepôt de Données**