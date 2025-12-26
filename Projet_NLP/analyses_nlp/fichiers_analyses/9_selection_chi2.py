"""
Analyse 9 : Sélection de Variables (Chi²)
Identifie les compétences "signature" de chaque profil métier

Auteur: Projet NLP Text Mining - Master SISE
Date: Décembre 2025
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter utils au path (gérer le fait qu'on est dans fichiers_analyses/)
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from utils import ResultSaver

print("=" * 80)
print("ANALYSE 9 : SÉLECTION DE VARIABLES (CHI²)")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================

print("\n📥 Chargement des données...")

# Charger données avec topics
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

# Charger dictionnaire de compétences
dict_path = Path(__file__).parent.parent / "resultats_nlp" / "dictionnaire_competences.json"

# Si le fichier n'existe pas, essayer dans le dossier parent
if not dict_path.exists():
    dict_path = Path(__file__).parent.parent.parent / "resultats_nlp" / "dictionnaire_competences.json"

if not dict_path.exists():
    print("❌ Erreur : dictionnaire_competences.json introuvable")
    print("💡 Lancez d'abord le script 1_preprocessing.py")
    sys.exit(1)

with open(dict_path, 'r', encoding='utf-8') as f:
    dict_comp = json.load(f)['competences']

print(f"✅ {len(dict_comp)} compétences dans le dictionnaire")

# ============================================================================
# 2. PRÉPARATION : MATRICE COMPÉTENCES × PROFILS
# ============================================================================

print("\n🔧 Création de la matrice compétences × profils...")

# Mapper topics vers labels
topic_labels = {
    0: "Data Engineering",
    1: "ML Engineering", 
    2: "Business Intelligence",
    3: "Deep Learning",
    4: "Data Analysis",
    5: "MLOps"
}

df['profil'] = df['topic_dominant'].map(topic_labels)

# Créer une matrice binaire : offre × compétence
# 1 si compétence présente, 0 sinon

n_offres = len(df)
n_competences = len(dict_comp)

# Matrice binaire
X = np.zeros((n_offres, n_competences), dtype=int)

print("⏳ Construction de la matrice...")

for i, comps in enumerate(df['competences_found']):
    for comp in comps:
        if comp in dict_comp:
            j = dict_comp.index(comp)
            X[i, j] = 1

print(f"✅ Matrice : {X.shape[0]} offres × {X.shape[1]} compétences")
print(f"   Taux de remplissage : {X.mean():.1%}")

# Variable cible
y = df['profil']

# ============================================================================
# 3. CALCUL DU CHI² POUR CHAQUE COMPÉTENCE
# ============================================================================

print("\n📊 Calcul du Chi² pour chaque compétence...")

# Chi² entre chaque compétence et la variable cible
chi2_scores, p_values = chi2(X, y)

print(f"✅ {len(chi2_scores)} scores Chi² calculés")

# Créer DataFrame avec résultats
df_chi2 = pd.DataFrame({
    'competence': dict_comp,
    'chi2_score': chi2_scores,
    'p_value': p_values
})

# Trier par Chi² décroissant
df_chi2 = df_chi2.sort_values('chi2_score', ascending=False)

# Top 30 compétences les plus discriminantes
print("\n🏆 Top 30 Compétences les Plus Discriminantes (Chi²) :")
print(df_chi2.head(30).to_string(index=False))

# ============================================================================
# 4. COMPÉTENCES SIGNATURE PAR PROFIL
# ============================================================================

print("\n" + "=" * 80)
print("COMPÉTENCES SIGNATURE PAR PROFIL MÉTIER")
print("=" * 80)

# Pour chaque profil, identifier ses compétences signature

signature_by_profile = {}

for profil in topic_labels.values():
    print(f"\n{'=' * 80}")
    print(f"📌 PROFIL : {profil}")
    print(f"{'=' * 80}")
    
    # Filtrer les offres de ce profil
    mask_profil = df['profil'] == profil
    df_profil = df[mask_profil]
    
    print(f"\n📊 {len(df_profil)} offres dans ce profil")
    
    # Compter les compétences dans ce profil
    all_comps = [c for cs in df_profil['competences_found'] for c in cs]
    from collections import Counter
    comp_counts = Counter(all_comps)
    
    # Calculer le % d'apparition dans ce profil
    comp_freq_profil = {comp: count/len(df_profil) for comp, count in comp_counts.items()}
    
    # Calculer le % d'apparition global
    all_comps_global = [c for cs in df['competences_found'] for c in cs]
    comp_counts_global = Counter(all_comps_global)
    comp_freq_global = {comp: count/len(df) for comp, count in comp_counts_global.items()}
    
    # "Lift" = fréquence dans profil / fréquence globale
    # Lift > 1 → Sur-représentée dans ce profil
    lifts = {}
    for comp in comp_freq_profil:
        if comp in comp_freq_global:
            lifts[comp] = comp_freq_profil[comp] / comp_freq_global[comp]
    
    # Trier par lift décroissant
    sorted_lifts = sorted(lifts.items(), key=lambda x: x[1], reverse=True)
    
    # Top 15 compétences signature (lift > 1.2)
    signatures = [(comp, lift) for comp, lift in sorted_lifts if lift > 1.2][:15]
    
    print(f"\n🎯 Top 15 Compétences Signature (sur-représentées) :")
    print(f"{'Compétence':<25} {'% Profil':<12} {'% Global':<12} {'Lift':<8}")
    print("-" * 60)
    
    for comp, lift in signatures:
        pct_profil = comp_freq_profil[comp] * 100
        pct_global = comp_freq_global[comp] * 100
        print(f"{comp:<25} {pct_profil:>6.1f}%      {pct_global:>6.1f}%      {lift:>5.2f}x")
    
    # Sauvegarder
    signature_by_profile[profil] = [
        {
            'competence': comp,
            'freq_profil': float(comp_freq_profil[comp]),
            'freq_global': float(comp_freq_global[comp]),
            'lift': float(lift)
        }
        for comp, lift in signatures
    ]

# ============================================================================
# 5. SÉLECTION DES K MEILLEURES VARIABLES
# ============================================================================

print("\n" + "=" * 80)
print("SÉLECTION DES K MEILLEURES VARIABLES")
print("=" * 80)

# Sélectionner top 100 compétences selon Chi²
k_best = 100

print(f"\n📊 Sélection des {k_best} meilleures compétences...")

selector = SelectKBest(chi2, k=k_best)
X_selected = selector.fit_transform(X, y)

# Récupérer les indices des compétences sélectionnées
selected_indices = selector.get_support(indices=True)
selected_competences = [dict_comp[i] for i in selected_indices]

print(f"✅ {len(selected_competences)} compétences sélectionnées")
print(f"\n🎯 Top 30 :")
for i, comp in enumerate(selected_competences[:30], 1):
    chi2_score = df_chi2[df_chi2['competence'] == comp]['chi2_score'].values[0]
    print(f"  {i:2d}. {comp:<30} (Chi² = {chi2_score:.1f})")

# ============================================================================
# 6. VISUALISATIONS
# ============================================================================

print("\n📊 Création des visualisations...")

viz_dir = Path(__file__).parent.parent / "resultats_nlp" / "visualisations"
viz_dir.mkdir(parents=True, exist_ok=True)

# 6.1 Distribution Chi² (courbe du coude)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(1, len(chi2_scores) + 1), sorted(chi2_scores, reverse=True), linewidth=2)
ax.axvline(k_best, color='red', linestyle='--', label=f'k={k_best}')
ax.set_xlabel('Rang de la Compétence')
ax.set_ylabel('Score Chi²')
ax.set_title('Distribution des Scores Chi² (Méthode du Coude)')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(viz_dir / 'chi2_elbow.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Courbe du coude : {viz_dir / 'chi2_elbow.png'}")

# 6.2 Top 30 compétences (bar chart)
fig, ax = plt.subplots(figsize=(12, 8))
top_30 = df_chi2.head(30).sort_values('chi2_score')
ax.barh(range(len(top_30)), top_30['chi2_score'])
ax.set_yticks(range(len(top_30)))
ax.set_yticklabels(top_30['competence'])
ax.set_xlabel('Score Chi²')
ax.set_title('Top 30 Compétences les Plus Discriminantes')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(viz_dir / 'top30_chi2.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Top 30 Chi² : {viz_dir / 'top30_chi2.png'}")

# 6.3 Heatmap : Compétences signature par profil
# Créer matrice : profils × top 20 compétences signature

all_signatures = []
for profil, sigs in signature_by_profile.items():
    for sig in sigs[:10]:  # Top 10 par profil
        all_signatures.append({
            'profil': profil,
            'competence': sig['competence'],
            'lift': sig['lift']
        })

df_heatmap = pd.DataFrame(all_signatures)
df_pivot = df_heatmap.pivot_table(
    index='competence', 
    columns='profil', 
    values='lift', 
    fill_value=0
)

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(df_pivot, annot=True, fmt='.2f', cmap='YlOrRd', 
            linewidths=0.5, cbar_kws={'label': 'Lift'}, ax=ax)
ax.set_title('Compétences Signature par Profil (Lift)')
ax.set_xlabel('Profil Métier')
ax.set_ylabel('Compétence')
plt.tight_layout()
plt.savefig(viz_dir / 'heatmap_signatures.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Heatmap signatures : {viz_dir / 'heatmap_signatures.png'}")

# ============================================================================
# 7. SAUVEGARDE DES RÉSULTATS
# ============================================================================

print("\n💾 Sauvegarde des résultats...")

saver = ResultSaver()

# Sauvegarder le sélecteur
saver.save_pickle(selector, 'chi2_selector.pkl')

# Résultats JSON
results = {
    'top_100_competences': df_chi2.head(k_best).to_dict('records'),
    'selected_competences': selected_competences,
    'signature_by_profile': signature_by_profile,
    'statistics': {
        'total_competences': len(dict_comp),
        'selected_k': k_best,
        'mean_chi2': float(chi2_scores.mean()),
        'max_chi2': float(chi2_scores.max())
    }
}

saver.save_json(results, 'chi2_selection.json')

# CSV pour Excel
df_chi2.to_csv(
    Path(__file__).parent.parent / "resultats_nlp" / "chi2_scores.csv",
    index=False,
    encoding='utf-8-sig'
)

print(f"✅ Résultats sauvegardés :")
print(f"   - models/chi2_selector.pkl")
print(f"   - chi2_selection.json")
print(f"   - chi2_scores.csv")

# ============================================================================
# 8. RÉCAPITULATIF
# ============================================================================

print("\n" + "=" * 80)
print("✅ SÉLECTION DE VARIABLES (CHI²) TERMINÉE")
print("=" * 80)

print(f"""
📊 Résultats :

Compétences analysées : {len(dict_comp)}
Compétences sélectionnées (k={k_best}) : {len(selected_competences)}

🏆 Top 5 Compétences les Plus Discriminantes :
{chr(10).join(f"  {i}. {row['competence']:<30} Chi² = {row['chi2_score']:.1f}" 
              for i, (_, row) in enumerate(df_chi2.head(5).iterrows(), 1))}

📁 Fichiers créés :
  - models/chi2_selector.pkl
  - chi2_selection.json
  - chi2_scores.csv
  - visualisations/chi2_elbow.png
  - visualisations/top30_chi2.png
  - visualisations/heatmap_signatures.png

💡 Utilisation :
  Ces compétences signature peuvent être utilisées pour :
  - Améliorer la classification (features réduites)
  - Identifier les compétences clés par profil
  - Recommander des formations ciblées
""")

print("✨ Analyse terminée avec succès !")