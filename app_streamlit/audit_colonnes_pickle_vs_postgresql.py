"""
AUDIT COMPLET - PICKLE vs POSTGRESQL
Identifie toutes les colonnes manquantes dans PostgreSQL

Projet NLP Text Mining - Master SISE
"""

import pickle
import psycopg2
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("🔍 AUDIT COLONNES - PICKLE vs POSTGRESQL")
print("="*80)

# ============================================
# CHARGEMENT PICKLE
# ============================================

PICKLE_PATH = Path('../resultats_nlp/models/data_with_profiles.pkl')

print(f"\n📥 Chargement pickle: {PICKLE_PATH}")

try:
    with open(PICKLE_PATH, 'rb') as f:
        df_pickle = pickle.load(f)
    
    print(f"✅ {len(df_pickle)} lignes chargées")
    print(f"✅ {len(df_pickle.columns)} colonnes")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    exit(1)

# ============================================
# CONNEXION POSTGRESQL
# ============================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

print("\n🔗 Connexion PostgreSQL...")

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✅ Connecté")
except Exception as e:
    print(f"❌ Erreur: {e}")
    exit(1)

# Charger données PostgreSQL
query = """
SELECT * FROM v_offres_nlp_complete LIMIT 1
"""

df_pg = pd.read_sql(query, conn)

print(f"✅ {len(df_pg.columns)} colonnes dans PostgreSQL")

# ============================================
# COMPARAISON COLONNES
# ============================================

print("\n" + "="*80)
print("📊 ANALYSE COLONNES")
print("="*80)

# Colonnes pickle
cols_pickle = set(df_pickle.columns)
print(f"\n📦 Colonnes PICKLE ({len(cols_pickle)}):")
for i, col in enumerate(sorted(cols_pickle), 1):
    print(f"   {i:2d}. {col}")

# Colonnes PostgreSQL
cols_pg = set(df_pg.columns)
print(f"\n☁️  Colonnes POSTGRESQL ({len(cols_pg)}):")
for i, col in enumerate(sorted(cols_pg), 1):
    print(f"   {i:2d}. {col}")

# ============================================
# COLONNES MANQUANTES
# ============================================

missing_cols = cols_pickle - cols_pg

print("\n" + "="*80)
print("⚠️  COLONNES MANQUANTES DANS POSTGRESQL")
print("="*80)

if missing_cols:
    print(f"\n❌ {len(missing_cols)} colonnes à migrer:\n")
    
    for i, col in enumerate(sorted(missing_cols), 1):
        # Analyse type et valeurs
        dtype = df_pickle[col].dtype
        non_null = df_pickle[col].notna().sum()
        pct_non_null = (non_null / len(df_pickle)) * 100
        
        # Exemples valeurs
        sample_values = df_pickle[col].dropna().head(3).tolist()
        
        print(f"{i:2d}. {col}")
        print(f"    Type: {dtype}")
        print(f"    Non-null: {non_null}/{len(df_pickle)} ({pct_non_null:.1f}%)")
        print(f"    Exemples: {sample_values}")
        print()
else:
    print("\n✅ Toutes les colonnes pickle sont dans PostgreSQL !")

# ============================================
# COLONNES SUPPLÉMENTAIRES (PG uniquement)
# ============================================

extra_cols = cols_pg - cols_pickle

if extra_cols:
    print("\n" + "="*80)
    print("ℹ️  COLONNES SUPPLÉMENTAIRES DANS POSTGRESQL")
    print("="*80)
    print("\n(Colonnes ajoutées par le modèle étoile)\n")
    for col in sorted(extra_cols):
        print(f"   • {col}")

# ============================================
# GÉNÉRATION SCRIPT MIGRATION
# ============================================

print("\n" + "="*80)
print("🔧 RECOMMANDATIONS")
print("="*80)

if missing_cols:
    print(f"""
🎯 ACTION REQUISE:

1. Ajouter {len(missing_cols)} colonnes dans fact_nlp_analysis
2. Migrer données depuis pickle
3. Modifier vue v_offres_nlp_complete

Je vais générer le script de migration automatique...
""")
    
    # Sauvegarder liste colonnes manquantes
    with open('colonnes_manquantes.txt', 'w', encoding='utf-8') as f:
        for col in sorted(missing_cols):
            dtype = df_pickle[col].dtype
            f.write(f"{col}\t{dtype}\n")
    
    print("✅ Liste sauvegardée: colonnes_manquantes.txt")
    
else:
    print("\n✅ AUCUNE ACTION REQUISE - Base de données complète !")

# ============================================
# FERMETURE
# ============================================

cursor.close()
conn.close()

print("\n" + "="*80)
print("✅ AUDIT TERMINÉ")
print("="*80)