"""
Analyse 8 : Classification Supervisée des Profils Métiers
Entraîne des modèles (SVM, MLP) pour prédire le profil métier d'une offre

Auteur: Projet NLP Text Mining - Master SISE
Date: Décembre 2025
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter utils au path (gérer le fait qu'on est dans fichiers_analyses/)
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from utils import ResultSaver

print("=" * 80)
print("ANALYSE 8 : CLASSIFICATION SUPERVISÉE DES PROFILS MÉTIERS")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================

print("\n📥 Chargement des données...")

# Charger les données avec topics
data_path = Path(__file__).parent.parent / "resultats_nlp" / "models" / "data_with_topics.pkl"

# Si le fichier n'existe pas, essayer dans le dossier parent
if not data_path.exists():
    data_path = Path(__file__).parent.parent.parent / "resultats_nlp" / "models" / "data_with_topics.pkl"

if not data_path.exists():
    print("❌ Erreur : fichier data_with_topics.pkl introuvable")
    print(f"   Cherché dans : {data_path}")
    print("💡 Lancez d'abord le script 3_topic_modeling.py")
    sys.exit(1)

with open(data_path, 'rb') as f:
    df = pickle.load(f)

print(f"✅ {len(df)} offres chargées")

# Vérifier qu'on a bien les topics
if 'topic_dominant' not in df.columns:
    print("❌ Erreur : colonne 'topic_dominant' manquante")
    print("💡 Lancez d'abord le script 3_topic_modeling.py")
    sys.exit(1)

# ============================================================================
# 2. PRÉPARATION DES DONNÉES
# ============================================================================

print("\n🔧 Préparation des données...")

# Mapper les topics vers des labels textuels
topic_labels = {
    0: "Data Engineering",
    1: "ML Engineering", 
    2: "Business Intelligence",
    3: "Deep Learning",
    4: "Data Analysis",
    5: "MLOps"
}

# Créer la variable cible
df['profil'] = df['topic_dominant'].map(topic_labels)

# Distribution des profils
print("\n📊 Distribution des profils :")
print(df['profil'].value_counts())
print()

# Filtrer les offres avec description
df_clean = df[df['description_clean'].notna()].copy()
print(f"✅ {len(df_clean)} offres avec description")

# ============================================================================
# 3. VECTORISATION TF-IDF
# ============================================================================

print("\n🔤 Vectorisation TF-IDF...")

# On utilise TF-IDF au lieu de BoW pour mieux capturer l'importance
vectorizer = TfidfVectorizer(
    max_features=500,  # Top 500 termes (compromis performance/qualité)
    min_df=5,          # Minimum 5 documents
    max_df=0.7,        # Maximum 70% des documents
    token_pattern=r'\b[a-zàâäéèêëïîôöùûüÿç]{3,}\b'
)

# Fit et transform sur toutes les données
X = vectorizer.fit_transform(df_clean['description_clean'])

print(f"✅ Matrice : {X.shape[0]} documents × {X.shape[1]} features")

# Variable cible
y = df_clean['profil']

# ============================================================================
# 4. DÉCOUPAGE TRAIN / TEST
# ============================================================================

print("\n✂️  Découpage Train / Test (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Garder les mêmes proportions
)

print(f"✅ Train : {X_train.shape[0]} offres")
print(f"✅ Test  : {X_test.shape[0]} offres")

# Vérifier stratification
print("\n📊 Distribution Train :")
print(y_train.value_counts(normalize=True).round(3))
print("\n📊 Distribution Test :")
print(y_test.value_counts(normalize=True).round(3))

# ============================================================================
# 5. MODÈLE 1 : SVM avec GridSearchCV
# ============================================================================

print("\n" + "=" * 80)
print("MODÈLE 1 : SVM (Support Vector Machine)")
print("=" * 80)

print("\n🔍 Recherche des meilleurs hyperparamètres (GridSearchCV)...")

# Grille de paramètres à tester
param_grid_svm = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 0.5, 1.0, 2.0, 10.0]
}

# Instance SVM
svm = SVC(random_state=42)

# GridSearchCV avec validation croisée 5-fold
grid_svm = GridSearchCV(
    estimator=svm,
    param_grid=param_grid_svm,
    scoring='f1_weighted',  # F1 pondéré pour classes déséquilibrées
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Entraînement (peut prendre quelques minutes)
print("⏳ Entraînement en cours...")
grid_svm.fit(X_train, y_train)

# Meilleurs paramètres
print(f"\n✅ Meilleurs paramètres : {grid_svm.best_params_}")
print(f"✅ F1-score (CV) : {grid_svm.best_score_:.3f}")

# Évaluation sur le test
y_pred_svm = grid_svm.predict(X_test)

print("\n📊 RÉSULTATS SVM SUR TEST :")
print(f"Accuracy : {accuracy_score(y_test, y_pred_svm):.3f}")
print("\nClassification Report :")
print(classification_report(y_test, y_pred_svm))

# Matrice de confusion
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=grid_svm.classes_)

# ============================================================================
# 6. MODÈLE 2 : Perceptron Multi-Couches (MLP)
# ============================================================================

print("\n" + "=" * 80)
print("MODÈLE 2 : PERCEPTRON MULTI-COUCHES (MLP)")
print("=" * 80)

print("\n🔍 Recherche des meilleurs hyperparamètres (GridSearchCV)...")

# Grille de paramètres
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 25)],
    'activation': ['tanh', 'relu'],
    'alpha': [0.0001, 0.001, 0.01]
}

# Instance MLP
mlp = MLPClassifier(random_state=42, max_iter=500)

# GridSearchCV
grid_mlp = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid_mlp,
    scoring='f1_weighted',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Entraînement
print("⏳ Entraînement en cours...")
grid_mlp.fit(X_train, y_train)

# Meilleurs paramètres
print(f"\n✅ Meilleurs paramètres : {grid_mlp.best_params_}")
print(f"✅ F1-score (CV) : {grid_mlp.best_score_:.3f}")

# Évaluation sur le test
y_pred_mlp = grid_mlp.predict(X_test)

print("\n📊 RÉSULTATS MLP SUR TEST :")
print(f"Accuracy : {accuracy_score(y_test, y_pred_mlp):.3f}")
print("\nClassification Report :")
print(classification_report(y_test, y_pred_mlp))

# Matrice de confusion
cm_mlp = confusion_matrix(y_test, y_pred_mlp, labels=grid_mlp.classes_)

# ============================================================================
# 7. COMPARAISON DES MODÈLES
# ============================================================================

print("\n" + "=" * 80)
print("COMPARAISON DES MODÈLES")
print("=" * 80)

# Scores
scores = {
    'SVM': accuracy_score(y_test, y_pred_svm),
    'MLP': accuracy_score(y_test, y_pred_mlp)
}

print("\n📊 Accuracy sur Test :")
for model, score in scores.items():
    print(f"  {model} : {score:.3f}")

# Meilleur modèle
best_model_name = max(scores, key=scores.get)
best_model = grid_svm if best_model_name == 'SVM' else grid_mlp

print(f"\n🏆 Meilleur modèle : {best_model_name} ({scores[best_model_name]:.3f})")

# ============================================================================
# 8. VISUALISATIONS
# ============================================================================

print("\n📊 Création des visualisations...")

# Créer dossier visualisations
viz_dir = Path(__file__).parent.parent / "resultats_nlp" / "visualisations"
viz_dir.mkdir(parents=True, exist_ok=True)  # parents=True crée aussi resultats_nlp si besoin

# 8.1 Matrices de confusion
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# SVM
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=grid_svm.classes_, 
            yticklabels=grid_svm.classes_,
            ax=axes[0])
axes[0].set_title('Matrice de Confusion - SVM')
axes[0].set_xlabel('Prédit')
axes[0].set_ylabel('Réel')

# MLP
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens',
            xticklabels=grid_mlp.classes_,
            yticklabels=grid_mlp.classes_,
            ax=axes[1])
axes[1].set_title('Matrice de Confusion - MLP')
axes[1].set_xlabel('Prédit')
axes[1].set_ylabel('Réel')

plt.tight_layout()
plt.savefig(viz_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Matrices de confusion : {viz_dir / 'confusion_matrices.png'}")

# 8.2 Comparaison des scores par classe
from sklearn.metrics import f1_score

classes = grid_svm.classes_

scores_by_class = {
    'SVM': [f1_score(y_test, y_pred_svm, labels=[c], average='weighted') 
            for c in classes],
    'MLP': [f1_score(y_test, y_pred_mlp, labels=[c], average='weighted') 
            for c in classes]
}

df_scores = pd.DataFrame(scores_by_class, index=classes)

fig, ax = plt.subplots(figsize=(12, 6))
df_scores.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('F1-Score par Profil Métier')
ax.set_xlabel('Profil')
ax.set_ylabel('F1-Score')
ax.set_ylim(0, 1)
ax.legend(['SVM', 'MLP'])
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(viz_dir / 'f1_scores_by_profile.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ F1-scores par profil : {viz_dir / 'f1_scores_by_profile.png'}")

# ============================================================================
# 9. SAUVEGARDE DES RÉSULTATS
# ============================================================================

print("\n💾 Sauvegarde des résultats...")

saver = ResultSaver()

# Sauvegarder les modèles
saver.save_pickle(grid_svm, 'model_svm.pkl')
saver.save_pickle(grid_mlp, 'model_mlp.pkl')
saver.save_pickle(vectorizer, 'vectorizer_classification.pkl')

print(f"✅ Modèles sauvegardés dans resultats_nlp/models/")

# Résultats JSON
results = {
    'svm': {
        'best_params': grid_svm.best_params_,
        'cv_f1_score': float(grid_svm.best_score_),
        'test_accuracy': float(accuracy_score(y_test, y_pred_svm)),
        'classification_report': classification_report(y_test, y_pred_svm, output_dict=True)
    },
    'mlp': {
        'best_params': grid_mlp.best_params_,
        'cv_f1_score': float(grid_mlp.best_score_),
        'test_accuracy': float(accuracy_score(y_test, y_pred_mlp)),
        'classification_report': classification_report(y_test, y_pred_mlp, output_dict=True)
    },
    'best_model': best_model_name,
    'topic_labels': topic_labels
}

saver.save_json(results, 'classification_results.json')

print(f"✅ Résultats sauvegardés : resultats_nlp/classification_results.json")

# ============================================================================
# 10. RÉCAPITULATIF
# ============================================================================

print("\n" + "=" * 80)
print("✅ CLASSIFICATION SUPERVISÉE TERMINÉE")
print("=" * 80)

print(f"""
📊 Résultats finaux :

SVM :
  - Accuracy Test : {scores['SVM']:.3f}
  - F1-Score CV   : {grid_svm.best_score_:.3f}
  - Params        : {grid_svm.best_params_}

MLP :
  - Accuracy Test : {scores['MLP']:.3f}
  - F1-Score CV   : {grid_mlp.best_score_:.3f}
  - Params        : {grid_mlp.best_params_}

🏆 Meilleur modèle : {best_model_name}

📁 Fichiers créés :
  - models/model_svm.pkl
  - models/model_mlp.pkl
  - models/vectorizer_classification.pkl
  - classification_results.json
  - visualisations/confusion_matrices.png
  - visualisations/f1_scores_by_profile.png
""")

print("✨ Les modèles peuvent maintenant être utilisés dans l'application Streamlit !")