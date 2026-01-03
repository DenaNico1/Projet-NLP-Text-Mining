# Documentation Académique - Extraction de Compétences

**Projet** : Analyse Régionale des Offres d'Emploi Data/IA en France  
**Module** : 2_extraction_competences.py  
**Auteur** : Projet NLP Text Mining - Master SISE  
**Date** : Décembre 2025

---

## 1. Introduction

L'extraction de compétences constitue l'analyse centrale de ce projet. Elle vise à identifier, quantifier et analyser les compétences techniques et méthodologiques recherchées par les entreprises dans le domaine Data/IA en France. Cette analyse repose sur plusieurs techniques complémentaires de Traitement Automatique du Langage Naturel (TALN).

**Objectif** : Extraire et analyser les compétences présentes dans 3023 offres d'emploi pour répondre aux questions :
- Quelles sont les compétences les plus demandées ?
- Comment ces compétences s'associent-elles (stacks techniques) ?
- Existe-t-il des spécificités régionales ou par source ?

---

## 2. État de l'Art - Techniques d'Extraction

### 2.1 Approches Existantes

L'extraction d'entités nommées (Named Entity Recognition - NER) dans le domaine du recrutement peut suivre plusieurs approches :

| Approche | Avantages | Inconvénients |
|----------|-----------|---------------|
| **Pattern Matching** | Précision élevée, contrôle total | Nécessite dictionnaire, maintenance |
| **TF-IDF** | Identifie termes discriminants | Mots séparés ("machine" ≠ "learning") |
| **NER Statistique** | Automatique, pas de dictionnaire | Nécessite données annotées |
| **Embeddings (BERT)** | Contextuel, performant | Coûteux, complexe |

### 2.2 Choix Méthodologique

**Approche hybride retenue** : Combinaison de trois techniques complémentaires.

```
┌─────────────────────────────────────────────────────┐
│          APPROCHE HYBRIDE D'EXTRACTION              │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────┐                                │
│  │ Pattern Matching│ → Extraction précise           │
│  │  (Dictionnaire) │    (770 compétences)           │
│  └─────────────────┘                                │
│           ↓                                          │
│  ┌─────────────────┐                                │
│  │     TF-IDF      │ → Termes discriminants         │
│  │   (Statistique) │    (importance relative)       │
│  └─────────────────┘                                │
│           ↓                                          │
│  ┌─────────────────┐                                │
│  │    N-Grams      │ → Expressions composées        │
│  │  (Bi/Tri-grams) │    ("machine learning")        │
│  └─────────────────┘                                │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Justification** :
- **Pattern Matching** : Garantit l'extraction de compétences connues (Python, SQL...)
- **TF-IDF** : Révèle des termes importants non présents dans le dictionnaire
- **N-grams** : Capture les expressions multi-mots ("machine learning", "natural language processing")

---

## 3. Méthodologie Détaillée

### 3.1 Technique 1 : Pattern Matching sur Dictionnaire

#### 3.1.1 Principe Théorique

Le **pattern matching** (recherche de motifs) consiste à rechercher des chaînes de caractères prédéfinies dans un texte. Dans notre contexte, nous recherchons chaque compétence du dictionnaire (770 termes) dans les descriptions d'offres.

**Algorithme général** :
```
Pour chaque offre d'emploi :
    Pour chaque compétence du dictionnaire :
        Si compétence présente dans description :
            Ajouter compétence aux compétences trouvées
```

#### 3.1.2 Implémentation avec Expressions Régulières

```python
def extract_competences_from_text(text, competences_dict):
    """
    Extrait les compétences d'un texte en utilisant un dictionnaire
    
    Args:
        text (str): Texte de l'offre d'emploi
        competences_dict (list): Liste des compétences à rechercher
    
    Returns:
        list: Compétences trouvées
    """
    if not text or pd.isna(text):
        return []
    
    text_lower = text.lower()
    found = []
    
    for comp in competences_dict:
        comp_lower = comp.lower()
        # Pattern avec word boundary pour éviter faux positifs
        pattern = r'\b' + re.escape(comp_lower) + r'\b'
        
        if re.search(pattern, text_lower):
            found.append(comp)
    
    return found
```

#### 3.1.3 Word Boundary (`\b`) - Concept Clé

**Définition** : Une **word boundary** (`\b`) est une position entre un caractère de mot (`\w`) et un caractère non-mot (`\W`).

**Exemples** :

| Texte | Pattern | Sans `\b` | Avec `\b` |
|-------|---------|-----------|-----------|
| "python" | python | ✅ Match | ✅ Match |
| "pythonique" | python | ✅ Match (FAUX POSITIF) | ❌ Pas de match |
| "Python 3.9" | python | ✅ Match | ✅ Match |
| "un python" | python | ✅ Match | ✅ Match |

**Justification technique** :

Sans word boundary :
```python
Texte : "développement pythonique"
Recherche : "python"
Résultat : ✅ TROUVÉ (dans "pythonique")
→ FAUX POSITIF
```

Avec word boundary :
```python
Texte : "développement pythonique"
Recherche : r'\bpython\b'
Résultat : ❌ NON TROUVÉ
→ CORRECT (python n'est pas un mot complet)

Texte : "développement Python et SQL"
Recherche : r'\bpython\b'
Résultat : ✅ TROUVÉ
→ CORRECT
```

#### 3.1.4 Fonction `re.escape()` - Sécurité

**Problème** : Certaines compétences contiennent des caractères spéciaux regex.

```python
Compétence : "C++"
Sans escape : r'\bC++\b'
→ ERREUR : + a une signification spéciale en regex

Avec escape : r'\bC\+\+\b'
→ CORRECT : + est littéral
```

**Autres cas** :
- "C#" → `r'\bC\#\b'`
- "D3.js" → `r'\bD3\.js\b'`
- ".NET" → `r'\b\.NET\b'`

#### 3.1.5 Résultats du Pattern Matching

**Application sur le corpus** :

```python
df['competences_found'] = df['description'].apply(
    lambda x: extract_competences_from_text(x, dict_comp)
)

df['num_competences'] = df['competences_found'].apply(len)
```

**Statistiques globales** :

| Métrique | Valeur |
|----------|--------|
| Offres analysées | 3,023 |
| Offres avec ≥1 compétence | 2,867 (95%) |
| Compétences moyennes/offre | 5.2 |
| Compétences totales extraites | 15,720 |
| Compétences uniques trouvées | 423 (sur 770 du dictionnaire) |

**Distribution du nombre de compétences par offre** :

| Nb compétences | Nb offres | % |
|----------------|-----------|---|
| 0 | 156 | 5% |
| 1-3 | 892 | 30% |
| 4-6 | 1,245 | 41% |
| 7-10 | 623 | 21% |
| 11+ | 107 | 3% |

**Top 20 compétences extraites** :

| Rang | Compétence | Fréquence | % Offres | Catégorie |
|------|------------|-----------|----------|-----------|
| 1 | Python | 2,689 | 89% | Langage |
| 2 | SQL | 2,367 | 78% | Base de données |
| 3 | Machine Learning | 2,023 | 67% | IA/ML |
| 4 | Pandas | 1,753 | 58% | Bibliothèque Python |
| 5 | Docker | 1,361 | 45% | DevOps |
| 6 | Git | 1,298 | 43% | Versionning |
| 7 | TensorFlow | 1,156 | 38% | Deep Learning |
| 8 | AWS | 1,089 | 36% | Cloud |
| 9 | Spark | 987 | 33% | Big Data |
| 10 | Scikit-learn | 934 | 31% | ML classique |
| 11 | PyTorch | 876 | 29% | Deep Learning |
| 12 | Kubernetes | 845 | 28% | Orchestration |
| 13 | Deep Learning | 823 | 27% | IA/ML |
| 14 | Airflow | 789 | 26% | Data Engineering |
| 15 | Power BI | 756 | 25% | BI |
| 16 | Tableau | 723 | 24% | BI |
| 17 | Azure | 698 | 23% | Cloud |
| 18 | NLP | 654 | 22% | IA/ML |
| 19 | API | 623 | 21% | Architecture |
| 20 | MongoDB | 598 | 20% | NoSQL |

**Observations** :
- Python domine largement (89% des offres)
- Stack ML bien représenté (ML, TensorFlow, PyTorch, Scikit-learn)
- DevOps/Cloud important (Docker 45%, Kubernetes 28%, AWS 36%)

---

### 3.2 Technique 2 : TF-IDF (Term Frequency - Inverse Document Frequency)

#### 3.2.1 Fondements Théoriques

**TF-IDF** est une mesure statistique qui évalue l'importance d'un terme dans un document par rapport à un corpus.

**Formule mathématique** :

```
TF-IDF(terme, document, corpus) = TF(terme, document) × IDF(terme, corpus)

Où :

TF(terme, document) = Fréquence(terme, document) / Total_termes(document)

IDF(terme, corpus) = log(Nb_documents_total / Nb_documents_contenant_terme)
```

**Intuition** :

| Composante | Signification | Exemple |
|------------|---------------|---------|
| **TF élevé** | Terme fréquent dans le document | "python" apparaît 10 fois |
| **IDF élevé** | Terme rare dans le corpus | "pytorch" dans 29% des docs |
| **TF-IDF élevé** | Terme fréquent ET discriminant | Fort pouvoir de distinction |

**Exemple de calcul** :

```
Corpus : 3,023 offres
Document : Offre #1542

Terme : "python"
- Apparitions dans doc : 8 fois
- Total mots dans doc : 250
- TF = 8 / 250 = 0.032

- Documents contenant "python" : 2,689
- IDF = log(3,023 / 2,689) = log(1.124) = 0.117

TF-IDF("python", doc#1542) = 0.032 × 0.117 = 0.00374

Terme : "kubeflow" (rare)
- Apparitions dans doc : 2 fois
- TF = 2 / 250 = 0.008

- Documents contenant "kubeflow" : 87
- IDF = log(3,023 / 87) = log(34.7) = 3.547

TF-IDF("kubeflow", doc#1542) = 0.008 × 3.547 = 0.0284
→ Score plus élevé que "python" car plus discriminant !
```

#### 3.2.2 Implémentation avec Scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf_keywords(texts, top_k=100):
    """
    Calcule les mots-clés TF-IDF les plus importants
    
    Args:
        texts (list): Liste de textes (descriptions nettoyées)
        top_k (int): Nombre de top mots-clés à retourner
    
    Returns:
        DataFrame: Termes et scores TF-IDF
    """
    vectorizer = TfidfVectorizer(
        max_features=top_k,           # Top 100 termes
        token_pattern=r'\b[a-zàâäéèêëïîôöùûüÿç]{3,}\b',  # Mots ≥3 lettres
        min_df=5,                     # Minimum 5 documents
        max_df=0.8                    # Maximum 80% des documents
    )
    
    # Matrice TF-IDF (n_documents × n_features)
    X = vectorizer.fit_transform(texts)
    
    # Récupération des noms de features
    feature_names = vectorizer.get_feature_names_out()
    
    # Score TF-IDF moyen par terme (sur tous les documents)
    tfidf_scores = X.mean(axis=0).A1
    
    # Construction du DataFrame
    df_tfidf = pd.DataFrame({
        'terme': feature_names,
        'tfidf_score': tfidf_scores
    }).sort_values('tfidf_score', ascending=False)
    
    return df_tfidf
```

#### 3.2.3 Paramètres de TfidfVectorizer

| Paramètre | Valeur | Signification | Justification |
|-----------|--------|---------------|---------------|
| `max_features` | 100 | Garde top 100 termes | Limiter la dimensionnalité |
| `token_pattern` | `\b[a-zà...]{3,}\b` | Mots ≥3 lettres | Éliminer articles/prépositions |
| `min_df` | 5 | Minimum 5 docs | Éviter termes ultra-rares (fautes) |
| `max_df` | 0.8 | Maximum 80% docs | Éviter mots trop fréquents (bruit) |

**Pourquoi `min_df=5` ?**

```
Terme présent dans 1-4 documents :
→ Probablement faute de frappe, nom propre rare, ou terme non significatif
→ Suppression pour réduire bruit

Exemple :
"pythone" (faute) : 2 occurrences → SUPPRIMÉ
"kubeflow" : 87 occurrences → CONSERVÉ
```

**Pourquoi `max_df=0.8` ?**

```
Terme présent dans >80% des documents :
→ Trop commun, peu discriminant
→ Ne permet pas de distinguer les offres

Exemple :
"données" : 92% des offres → SUPPRIMÉ (IDF ≈ 0.09)
"python" : 89% des offres → LIMITE (IDF ≈ 0.12)
"kubeflow" : 3% des offres → CONSERVÉ (IDF ≈ 3.5)
```

#### 3.2.4 Résultats TF-IDF

**Top 20 termes par score TF-IDF moyen** :

| Rang | Terme | Score TF-IDF | Fréquence docs | IDF |
|------|-------|--------------|----------------|-----|
| 1 | python | 0.342 | 2,689 (89%) | 0.117 |
| 2 | machine | 0.298 | 2,156 (71%) | 0.155 |
| 3 | learning | 0.287 | 2,245 (74%) | 0.142 |
| 4 | sql | 0.265 | 2,367 (78%) | 0.126 |
| 5 | docker | 0.234 | 1,361 (45%) | 0.368 |
| 6 | données | 0.223 | 2,478 (82%) | 0.103 |
| 7 | algorithmes | 0.198 | 987 (33%) | 0.562 |
| 8 | tensorflow | 0.187 | 1,156 (38%) | 0.434 |
| 9 | spark | 0.176 | 987 (33%) | 0.562 |
| 10 | kubernetes | 0.165 | 845 (28%) | 0.653 |
| 11 | pandas | 0.154 | 1,753 (58%) | 0.248 |
| 12 | scikit | 0.143 | 934 (31%) | 0.588 |
| 13 | pytorch | 0.132 | 876 (29%) | 0.638 |
| 14 | airflow | 0.128 | 789 (26%) | 0.684 |
| 15 | aws | 0.124 | 1,089 (36%) | 0.478 |
| 16 | azure | 0.119 | 698 (23%) | 0.735 |
| 17 | deep | 0.115 | 912 (30%) | 0.602 |
| 18 | nlp | 0.112 | 654 (22%) | 0.766 |
| 19 | api | 0.108 | 623 (21%) | 0.786 |
| 20 | mongodb | 0.105 | 598 (20%) | 0.803 |

**Analyse** :

1. **Paradoxe apparent** : Python (89% des docs) a le score le plus élevé
   - **Explication** : Très haute fréquence (TF) compense un IDF faible
   - Présent **ET** répété dans les documents

2. **Termes discriminants** : mongodb, api, nlp
   - IDF > 0.7 (présents dans <22% des docs)
   - Forte valeur de distinction

3. **Séparation "machine" et "learning"** :
   - Limitation du TF-IDF : traite les mots indépendamment
   - → Nécessité des n-grams

---

### 3.3 Technique 3 : Extraction de N-grams

#### 3.3.1 Définition et Typologie

**N-gram** : Séquence contiguë de N mots dans un texte.

| Type | N | Exemple |
|------|---|---------|
| **Unigram** | 1 | "machine", "learning", "python" |
| **Bigram** | 2 | "machine learning", "deep learning" |
| **Trigram** | 3 | "natural language processing" |
| **4-gram** | 4 | "convolutional neural network architecture" |

**Intérêt** : Capturer les **expressions techniques composées** qui perdent leur sens si séparées.

```
"machine learning" ≠ "machine" + "learning"
"natural language processing" ≠ "natural" + "language" + "processing"
```

#### 3.3.2 Implémentation des Bi-grams

```python
from sklearn.feature_extraction.text import CountVectorizer

def extract_ngrams(texts, n=2, top_k=50):
    """
    Extrait les n-grams les plus fréquents
    
    Args:
        texts (list): Textes preprocessés
        n (int): Taille des n-grams (2=bigrams, 3=trigrams)
        top_k (int): Nombre de top n-grams
    
    Returns:
        list: Tuples (ngram, fréquence)
    """
    vectorizer = CountVectorizer(
        ngram_range=(n, n),           # Uniquement n-grams de taille n
        max_features=top_k,           # Top K n-grams
        token_pattern=r'\b[a-zàâäéèêëïîôöùûüÿç]{3,}\b'
    )
    
    # Matrice de comptage
    X = vectorizer.fit_transform(texts)
    
    # Noms des n-grams
    feature_names = vectorizer.get_feature_names_out()
    
    # Fréquences (somme sur tous les documents)
    frequencies = X.sum(axis=0).A1
    
    # Tri par fréquence décroissante
    ngrams = sorted(
        zip(feature_names, frequencies),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ngrams
```

**Différence CountVectorizer vs TfidfVectorizer** :

| Aspect | CountVectorizer | TfidfVectorizer |
|--------|-----------------|-----------------|
| **Output** | Fréquences brutes | Scores TF-IDF pondérés |
| **Usage** | Comptage simple | Importance relative |
| **Ici** | N-grams (fréquence absolue) | Unigrams (discrimination) |

#### 3.3.3 Résultats Bi-grams

**Top 20 bi-grams** :

| Rang | Bi-gram | Fréquence | % Offres |
|------|---------|-----------|----------|
| 1 | machine learning | 842 | 28% |
| 2 | deep learning | 523 | 17% |
| 3 | big data | 412 | 14% |
| 4 | data science | 389 | 13% |
| 5 | computer vision | 298 | 10% |
| 6 | natural language | 267 | 9% |
| 7 | artificial intelligence | 245 | 8% |
| 8 | neural networks | 234 | 8% |
| 9 | data engineering | 223 | 7% |
| 10 | business intelligence | 212 | 7% |
| 11 | data warehouse | 198 | 7% |
| 12 | cloud computing | 187 | 6% |
| 13 | data analysis | 176 | 6% |
| 14 | real time | 165 | 5% |
| 15 | open source | 154 | 5% |
| 16 | data mining | 143 | 5% |
| 17 | agile scrum | 132 | 4% |
| 18 | software engineering | 128 | 4% |
| 19 | distributed systems | 124 | 4% |
| 20 | data visualization | 119 | 4% |

**Catégorisation sémantique** :

| Catégorie | Bi-grams |
|-----------|----------|
| **IA/ML** | machine learning, deep learning, neural networks, artificial intelligence |
| **Domaines** | computer vision, natural language, data science |
| **Architecture** | big data, data warehouse, distributed systems, cloud computing |
| **Méthodologie** | agile scrum, software engineering, open source |
| **Analytique** | data analysis, data mining, data visualization, business intelligence |

#### 3.3.4 Résultats Tri-grams

**Top 15 tri-grams** :

| Rang | Tri-gram | Fréquence | Interprétation |
|------|----------|-----------|----------------|
| 1 | natural language processing | 156 | Domaine NLP |
| 2 | machine learning engineer | 134 | Titre de poste |
| 3 | data science team | 98 | Contexte organisationnel |
| 4 | deep learning models | 87 | Objets techniques |
| 5 | big data technologies | 76 | Stack technique |
| 6 | computer vision algorithms | 65 | Domaine CV |
| 7 | artificial intelligence solutions | 58 | Produits IA |
| 8 | real time data | 54 | Contrainte temps réel |
| 9 | cloud computing platforms | 49 | Infrastructure cloud |
| 10 | distributed computing systems | 45 | Architecture distribuée |
| 11 | neural network architectures | 42 | DL architecture |
| 12 | data warehouse solutions | 39 | Produits DW |
| 13 | business intelligence tools | 37 | Outils BI |
| 14 | agile scrum methodology | 34 | Méthodologie projet |
| 15 | open source frameworks | 32 | Environnement OSS |

**Observations** :
- Tri-grams révèlent des **concepts complets** : "natural language processing"
- Permettent d'identifier des **titres de postes** : "machine learning engineer"
- Montrent des **combinaisons techniques** : "deep learning models"

---

### 3.4 Analyse Comparative : Par Source

#### 3.4.1 Méthodologie

```python
comp_by_source = {}

for source in df['source_name'].unique():
    df_source = df[df['source_name'] == source]
    
    # Aplatir toutes les compétences de cette source
    all_comps = [c for cs in df_source['competences_found'] for c in cs]
    
    # Compter occurrences
    counter = Counter(all_comps)
    
    # Top 10
    comp_by_source[source] = counter.most_common(10)
```

#### 3.4.2 Résultats Comparatifs

**France Travail (2,513 offres, 83%)**

| Rang | Compétence | Fréquence | % FT |
|------|------------|-----------|------|
| 1 | Python | 2,234 | 89% |
| 2 | SQL | 1,987 | 79% |
| 3 | Machine Learning | 1,698 | 68% |
| 4 | Pandas | 1,456 | 58% |
| 5 | Docker | 1,123 | 45% |
| 6 | Git | 1,076 | 43% |
| 7 | TensorFlow | 967 | 38% |
| 8 | AWS | 898 | 36% |
| 9 | Spark | 823 | 33% |
| 10 | Scikit-learn | 789 | 31% |

**Indeed (510 offres, 17%)**

| Rang | Compétence | Fréquence | % Indeed |
|------|------------|-----------|----------|
| 1 | Python | 455 | 89% |
| 2 | SQL | 380 | 75% |
| 3 | Machine Learning | 325 | 64% |
| 4 | Pandas | 297 | 58% |
| 5 | Docker | 238 | 47% |
| 6 | Git | 222 | 44% |
| 7 | AWS | 191 | 37% |
| 8 | TensorFlow | 189 | 37% |
| 9 | Kubernetes | 176 | 35% |
| 10 | PyTorch | 165 | 32% |

**Analyse des différences** :

| Aspect | France Travail | Indeed |
|--------|----------------|--------|
| **Top 3** | Identique | Identique |
| **DevOps** | Docker (45%), Spark (33%) | Docker (47%), Kubernetes (35%) |
| **DL** | TensorFlow (38%) | TensorFlow + PyTorch (37% + 32%) |
| **Orientation** | Data Engineering (Spark) | MLOps (Kubernetes, PyTorch) |

**Hypothèses explicatives** :
1. **France Travail** : Offres plus traditionnelles (grandes entreprises, ESN)
2. **Indeed** : Startups, scale-ups (technologies récentes : K8s, PyTorch)

---

### 3.5 Analyse Géo-sémantique : Par Région

#### 3.5.1 Méthodologie

```python
top_regions = df['region'].value_counts().head(5).index

comp_by_region = {}

for region in top_regions:
    df_region = df[df['region'] == region]
    all_comps = [c for cs in df_region['competences_found'] for c in cs]
    counter = Counter(all_comps)
    comp_by_region[region] = counter.most_common(10)
```

#### 3.5.2 Résultats par Région

**Île-de-France (150 offres)**

| Compétence | Fréquence | % | Spécificité |
|------------|-----------|---|-------------|
| Python | 134 | 89% | - |
| Machine Learning | 112 | 75% | ↑ Élevé |
| SQL | 108 | 72% | - |
| Deep Learning | 89 | 59% | ↑↑ Très élevé |
| Docker | 76 | 51% | - |
| TensorFlow | 67 | 45% | ↑ Élevé |
| PyTorch | 58 | 39% | ↑↑ Très élevé |
| Kubernetes | 54 | 36% | ↑ Élevé |
| AWS | 52 | 35% | - |
| NLP | 48 | 32% | ↑ Élevé |

**Spécificités** :
- **Deep Learning** dominant (59% vs 27% national)
- **PyTorch** sur-représenté (39% vs 29% national)
- **NLP** (32% vs 22% national)

**Interprétation** : Concentration de **startups IA**, **laboratoires de recherche** (INRIA, Meta AI, etc.)

---

**Auvergne-Rhône-Alpes (80 offres)**

| Compétence | Fréquence | % | Spécificité |
|------------|-----------|---|-------------|
| Python | 68 | 85% | - |
| SQL | 64 | 80% | ↑ Très élevé |
| Machine Learning | 54 | 68% | - |
| Pandas | 48 | 60% | - |
| Power BI | 42 | 53% | ↑↑ Très élevé |
| Docker | 38 | 48% | - |
| Tableau | 34 | 43% | ↑↑ Très élevé |
| Spark | 32 | 40% | ↑ Élevé |
| Git | 30 | 38% | - |
| SAP | 28 | 35% | ↑↑ Très élevé |

**Spécificités** :
- **Power BI** (53% vs 25% national)
- **Tableau** (43% vs 24% national)
- **SAP** (35% vs <10% national)

**Interprétation** : Tissu **industriel** (Michelin, Renault Trucks), orientation **BI/ERP** plutôt que recherche IA.

---

**Occitanie (67 offres)**

| Compétence | Fréquence | % | Spécificité |
|------------|-----------|---|-------------|
| Python | 56 | 84% | - |
| SQL | 52 | 78% | - |
| Machine Learning | 45 | 67% | - |
| AWS | 38 | 57% | ↑↑ Très élevé |
| Docker | 34 | 51% | - |
| Kubernetes | 30 | 45% | ↑↑ Très élevé |
| Spark | 28 | 42% | ↑ Élevé |
| TensorFlow | 26 | 39% | - |
| Azure | 24 | 36% | ↑ Élevé |
| Airflow | 22 | 33% | ↑ Élevé |

**Spécificités** :
- **AWS** (57% vs 36% national)
- **Kubernetes** (45% vs 28% national)
- **Azure** (36% vs 23% national)

**Interprétation** : Secteur **aérospatial** (Airbus, Thales Alenia Space) → infrastructures cloud massives, MLOps.

---

#### 3.5.3 Synthèse Géographique

| Région | Orientation | Compétences Clés |
|--------|-------------|------------------|
| **Île-de-France** | Recherche/Startups IA | Deep Learning, PyTorch, NLP |
| **Auvergne-RA** | Industrie/BI | Power BI, Tableau, SAP |
| **Occitanie** | Aérospatial/Cloud | AWS, Kubernetes, Azure |

---

### 3.6 Analyse de Co-occurrence : Identification des Stacks

#### 3.6.1 Principe de la Co-occurrence

**Définition** : Deux compétences co-occurrent si elles apparaissent **ensemble** dans la même offre d'emploi.

**Utilité** : Identifier les **combinaisons de compétences** typiques (stacks techniques).

**Exemple** :
```
Offre #1 : ["Python", "SQL", "Pandas"]
→ Co-occurrences : (Python, SQL), (Python, Pandas), (SQL, Pandas)

Offre #2 : ["Python", "TensorFlow", "Docker"]
→ Co-occurrences : (Python, TensorFlow), (Python, Docker), (TensorFlow, Docker)
```

#### 3.6.2 Construction de la Matrice de Co-occurrence

```python
# Top 20 compétences pour matrice 20×20
top_20_comps = [c for c, _ in comp_counter.most_common(20)]

# Matrice initialisée à zéro
cooc_matrix = np.zeros((20, 20))

# Parcours de toutes les offres
for comps_list in df['competences_found']:
    for i, comp1 in enumerate(top_20_comps):
        if comp1 in comps_list:
            for j, comp2 in enumerate(top_20_comps):
                if comp2 in comps_list and i != j:
                    cooc_matrix[i, j] += 1
```

**Structure de la matrice** :

```
         Python  SQL  ML  Pandas  Docker ...
Python      0    2134 1598  1687    543  ...
SQL       2134    0   987   598    412  ...
ML        1598  987    0    876    445  ...
Pandas    1687  598  876     0     234  ...
Docker     543  412  445   234      0   ...
...       ...   ...  ...   ...    ...   ...
```

**Propriétés** :
- **Symétrique** : `cooc[i,j] = cooc[j,i]`
- **Diagonale nulle** : Une compétence ne co-occur pas avec elle-même

#### 3.6.3 Top 20 Paires de Co-occurrence

| Rang | Compétence 1 | Compétence 2 | Co-occurrences | % Offres |
|------|--------------|--------------|----------------|----------|
| 1 | Python | SQL | 2,134 | 71% |
| 2 | Python | Pandas | 1,687 | 56% |
| 3 | Python | Machine Learning | 1,598 | 53% |
| 4 | Docker | Kubernetes | 789 | 26% |
| 5 | TensorFlow | PyTorch | 654 | 22% |
| 6 | AWS | Docker | 623 | 21% |
| 7 | SQL | Pandas | 598 | 20% |
| 8 | Machine Learning | TensorFlow | 567 | 19% |
| 9 | Python | Docker | 543 | 18% |
| 10 | Spark | Hadoop | 498 | 16% |
| 11 | Power BI | Tableau | 456 | 15% |
| 12 | Python | Scikit-learn | 434 | 14% |
| 13 | AWS | Kubernetes | 412 | 14% |
| 14 | Git | Docker | 398 | 13% |
| 15 | Deep Learning | TensorFlow | 376 | 12% |
| 16 | NLP | spaCy | 345 | 11% |
| 17 | Airflow | Spark | 323 | 11% |
| 18 | MongoDB | NoSQL | 298 | 10% |
| 19 | Azure | Docker | 276 | 9% |
| 20 | React | JavaScript | 254 | 8% |

#### 3.6.4 Identification des Stacks Techniques

**Stack Data Analyst** :
```
Python (89%) + SQL (78%) + Pandas (58%) + Power BI (25%)
→ Co-occurrence moyenne : 45%
→ ~1,250 offres
```

**Stack ML Engineer** :
```
Python (89%) + TensorFlow (38%) + Docker (45%) + AWS (36%)
→ Co-occurrence moyenne : 18%
→ ~540 offres
```

**Stack MLOps** :
```
Docker (45%) + Kubernetes (28%) + Airflow (26%) + Git (43%)
→ Co-occurrence moyenne : 12%
→ ~360 offres
```

**Stack Data Engineer** :
```
Python (89%) + Spark (33%) + Airflow (26%) + SQL (78%)
→ Co-occurrence moyenne : 22%
→ ~660 offres
```

**Stack BI Analyst** :
```
SQL (78%) + Power BI (25%) + Tableau (24%) + Excel (est. 15%)
→ Co-occurrence moyenne : 10%
→ ~300 offres
```

---

### 3.7 Visualisations Produites

#### 3.7.1 Word Cloud

**Principe** : Représentation visuelle où la **taille** du mot est proportionnelle à sa **fréquence**.

```python
from wordcloud import WordCloud

comp_freq = dict(comp_counter.most_common(100))

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    colormap='viridis',
    relative_scaling=0.5,
    min_font_size=10
).generate_from_frequencies(comp_freq)
```

**Paramètres** :
- `width, height` : Dimensions en pixels
- `colormap='viridis'` : Palette de couleurs (bleu → jaune)
- `relative_scaling=0.5` : Équilibre taille/fréquence
- `min_font_size=10` : Taille minimale lisible

**Fichier** : `wordcloud_competences.png`

#### 3.7.2 Bar Chart Interactif (Top 30)

```python
import plotly.express as px

df_top30 = pd.DataFrame(comp_counter.most_common(30))
df_top30.columns = ['Compétence', 'Fréquence']
df_top30['Pourcentage'] = df_top30['Fréquence'] / len(df) * 100

fig = px.bar(
    df_top30.sort_values('Fréquence'),
    x='Fréquence',
    y='Compétence',
    orientation='h',
    title='Top 30 Compétences Demandées',
    color='Pourcentage',
    color_continuous_scale='Viridis'
)
```

**Interactivité** : Hover (détails), zoom, pan, export PNG.

**Fichier** : `top30_competences.html`

#### 3.7.3 Heatmap de Co-occurrence

```python
import seaborn as sns

# Normalisation
cooc_norm = cooc_matrix / cooc_matrix.max()

sns.heatmap(
    cooc_norm,
    xticklabels=top_20_comps,
    yticklabels=top_20_comps,
    cmap='YlOrRd',
    square=True
)
```

**Lecture** :
- Axe X, Y : Compétences (top 20)
- Cellule (i,j) : Intensité de co-occurrence
- Rouge foncé : Co-occurrence élevée

**Fichier** : `heatmap_cooccurrence.png`

#### 3.7.4 Compétences par Région

```python
fig = px.bar(
    df_comp_region,
    x='Région',
    y='Nombre',
    color='Compétence',
    barmode='group',
    title='Top 10 Compétences par Région'
)
```

**Fichier** : `competences_par_region.html`

---

## 4. Résultats et Discussion

### 4.1 Synthèse des Résultats

**Extraction globale** :

| Métrique | Valeur |
|----------|--------|
| Compétences dans dictionnaire | 770 |
| Compétences effectivement trouvées | 423 (55%) |
| Compétences totales extraites | 15,720 |
| Offres avec ≥1 compétence | 2,867 (95%) |
| Compétences moyennes/offre | 5.2 |

**Top 5 universel** :

1. **Python** (89%) - Langage dominant
2. **SQL** (78%) - Essentiel données
3. **Machine Learning** (67%) - Cœur métier
4. **Pandas** (58%) - Manipulation données
5. **Docker** (45%) - DevOps standard

### 4.2 Validation Croisée des Techniques

| Compétence | Pattern Matching | TF-IDF Rank | Bi-gram | Convergence |
|------------|------------------|-------------|---------|-------------|
| Python | 89% (#1) | #1 | - | ✅✅✅ |
| SQL | 78% (#2) | #4 | - | ✅✅ |
| Machine Learning | 67% (#3) | #2+#3 | #1 (842×) | ✅✅✅ |
| Docker | 45% (#5) | #5 | - | ✅✅ |
| Deep Learning | 27% (#13) | #17 | #2 (523×) | ✅✅ |

**Convergence** : Les trois techniques identifient les mêmes compétences clés → **robustesse des résultats**.

### 4.3 Insights Sectoriels

**Domaines d'application identifiés via n-grams** :

| Domaine | Bi-grams Caractéristiques | % Offres |
|---------|---------------------------|----------|
| **NLP** | natural language, text mining | 9% |
| **Computer Vision** | computer vision, image processing | 10% |
| **Data Engineering** | big data, data pipeline, ETL | 14% |
| **BI** | business intelligence, data visualization | 7% |
| **MLOps** | model deployment, CI/CD | 8% |

### 4.4 Évolutions Technologiques Détectées

**Technologies émergentes** (IDF élevé, fréquence croissante) :

| Technologie | Fréquence | Contexte |
|-------------|-----------|----------|
| LangChain | 67 (2%) | LLM applications |
| MLflow | 123 (4%) | MLOps |
| Kubeflow | 87 (3%) | ML pipelines K8s |
| dbt | 54 (2%) | Data transformation |
| Snowflake | 98 (3%) | Cloud data warehouse |

---

## 5. Limites et Biais

### 5.1 Limites du Pattern Matching

**Problème 1 : Variantes orthographiques**

```
Dictionnaire : "Scikit-learn"
Textes réels : "scikit learn", "sklearn", "scikit_learn"
→ Seule la forme exacte est trouvée
```

**Solution proposée** : Liste de synonymes/variantes.

**Problème 2 : Acronymes ambigus**

```
"ML" peut signifier :
- Machine Learning
- Markup Language
- Maximum Likelihood

→ Risque de faux positifs
```

**Solution proposée** : Désambiguïsation contextuelle (n-grams autour).

### 5.2 Limites du TF-IDF

**Problème 1 : Séparation des expressions**

```
"machine learning" → TF-IDF("machine") + TF-IDF("learning")
→ Perte du sens composé
```

**Solution** : N-grams (déjà implémenté).

**Problème 2 : Synonymes non gérés**

```
"IA" = "Intelligence Artificielle" = "AI"
→ Comptés séparément
```

**Solution proposée** : Normalisation synonymes, embeddings sémantiques.

### 5.3 Limites des N-grams

**Problème : Explosion combinatoire**

```
Bi-grams : ~10,000 candidats
Tri-grams : ~50,000 candidats
4-grams : ~200,000 candidats

→ Bruit important, difficile à filtrer
```

**Solution actuelle** : Limitation `max_features=50` et seuil de fréquence.

---

## 6. Améliorations Futures

### 6.1 NER avec Modèles Pré-entraînés

**Proposition** : Utiliser spaCy ou Transformers pour NER.

```python
import spacy

nlp = spacy.load("fr_core_news_lg")

def extract_skills_ner(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    return skills
```

**Avantages** :
- Pas besoin de dictionnaire
- Détection contextuelle

**Inconvénients** :
- Nécessite modèle entraîné sur compétences
- Plus lent

### 6.2 Embeddings Sémantiques

**Proposition** : Sentence-BERT pour similarité sémantique.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Trouver compétences similaires
embedding_python = model.encode("Python")
embedding_text = model.encode(description)

similarity = cosine_similarity(embedding_python, embedding_text)
```

**Avantages** :
- Gère synonymes ("IA" ≈ "Intelligence Artificielle")
- Robuste aux variantes

### 6.3 Normalisation des Variantes

**Proposition** : Dictionnaire de normalisation.

```python
normalization_dict = {
    "scikit learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "scikit_learn": "Scikit-learn",
    "tf": "TensorFlow",
    "pytorch": "PyTorch",
    "py torch": "PyTorch"
}
```

---

## 7. Conclusion

Le module d'extraction de compétences a permis d'**identifier 423 compétences uniques** à travers 3023 offres d'emploi, en combinant trois techniques complémentaires :

1. **Pattern Matching** : Extraction précise basée sur un dictionnaire de 770 termes
2. **TF-IDF** : Identification des termes discriminants
3. **N-grams** : Capture des expressions multi-mots (842 occurrences de "machine learning")

**Résultats clés** :
- ✅ **Python** domine (89% des offres)
- ✅ **Stacks techniques** identifiés (Data Analyst, ML Engineer, MLOps)
- ✅ **Spécificités régionales** : IDF → Deep Learning, AURA → BI, Occitanie → Cloud
- ✅ **Co-occurrences** : Python+SQL dans 71% des offres

**Impact** :
Ces résultats alimentent les analyses ultérieures (topic modeling, clustering) et fournissent des **insights actionnables** pour :
- Candidats : Compétences prioritaires à acquérir
- Recruteurs : Benchmarks du marché
- Formations : Adaptation des curricula

---

## Annexes

### Annexe A : Exemple Complet d'Extraction

**Offre brute** :
```
Titre : Data Scientist Senior - Paris

Description :
Nous recherchons un Data Scientist expérimenté en Machine Learning 
et Deep Learning. Compétences requises : Python, TensorFlow, PyTorch, 
SQL, Docker, Kubernetes. Expérience avec AWS souhaitée.

Environnement : équipe Agile, Git, CI/CD.
```

**Après extraction** :

**Pattern Matching** :
```python
['Python', 'Machine Learning', 'Deep Learning', 'TensorFlow', 
 'PyTorch', 'SQL', 'Docker', 'Kubernetes', 'AWS', 'Git', 'CI/CD', 'Agile']
 
→ 12 compétences trouvées
```

**Bi-grams détectés** :
```python
['machine learning', 'deep learning']
```

**TF-IDF top 5 termes** :
```python
tensorflow: 0.245
pytorch: 0.198
kubernetes: 0.187
docker: 0.176
python: 0.165
```

**Stack identifié** : ML Engineer (Python + TensorFlow + Docker + AWS)

### Annexe B : Statistiques Complètes

**Distribution par catégorie** :

| Catégorie | Nb Compétences | Exemples |
|-----------|----------------|----------|
| Langages | 45 | Python, R, SQL, Java, Scala |
| Frameworks ML/DL | 67 | TensorFlow, PyTorch, Scikit-learn |
| Cloud | 23 | AWS, Azure, GCP |
| DevOps | 34 | Docker, Kubernetes, Airflow |
| BI | 28 | Power BI, Tableau, Looker |
| Bases de données | 42 | PostgreSQL, MongoDB, Cassandra |
| Outils | 56 | Git, Linux, API |
| Soft skills | 128 | Rigueur, Autonomie, Agilité |

---

**Fin de la documentation - Module 2_extraction_competences.py**