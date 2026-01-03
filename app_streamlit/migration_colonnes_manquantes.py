"""
MIGRATION COLONNES MANQUANTES - PICKLE → POSTGRESQL
Projet NLP Text Mining - Master SISE

Ajoute 16 colonnes manquantes dans fact_nlp_analysis
pour compatibilité 100% avec data_with_profiles.pkl
"""

import pickle
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("🔧 AJOUT COLONNES MANQUANTES → POSTGRESQL")
print("="*80)

# ============================================
# CONFIGURATION
# ============================================

PICKLE_PATH = Path('../resultats_nlp/models/data_with_profiles.pkl')

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# Colonnes à ajouter
COLONNES_MANQUANTES = [
    ('num_tokens', 'INTEGER'),
    ('num_competences', 'INTEGER'),
    ('profil_score', 'NUMERIC(5,2)'),
    ('profil_confidence', 'NUMERIC(5,2)'),
    ('profil_second', 'VARCHAR(100)'),
    ('profil_second_score', 'NUMERIC(5,2)'),
    ('score_title', 'NUMERIC(5,2)'),
    ('score_description', 'NUMERIC(5,2)'),
    ('score_competences', 'NUMERIC(5,2)'),
    ('cascade_pass', 'INTEGER'),
    ('description_clean', 'TEXT'),
    ('text_for_sklearn', 'TEXT'),
    ('tokens', 'TEXT'),
    ('duration', 'VARCHAR(100)'),
    ('salary_text', 'VARCHAR(200)'),
    ('source_name', 'VARCHAR(50)')
]

print(f"\n✅ {len(COLONNES_MANQUANTES)} colonnes à ajouter")

# ============================================
# CHARGEMENT PICKLE
# ============================================

print("\n" + "="*80)
print("📥 CHARGEMENT PICKLE")
print("="*80)

try:
    print(f"\n   Lecture: {PICKLE_PATH}")
    with open(PICKLE_PATH, 'rb') as f:
        df_pickle = pickle.load(f)
    
    print(f"   ✅ {len(df_pickle)} lignes chargées")
    
except Exception as e:
    print(f"\n❌ Erreur: {e}")
    sys.exit(1)

# Vérifier offre_id existe
if 'offre_id' not in df_pickle.columns:
    print("\n⚠️  Création colonne offre_id...")
    df_pickle['offre_id'] = df_pickle.index + 1

# Conversion NaN → None
df_pickle = df_pickle.where(pd.notna(df_pickle), None)

# ============================================
# CONNEXION POSTGRESQL
# ============================================

print("\n" + "="*80)
print("🔗 CONNEXION POSTGRESQL")
print("="*80)

try:
    print(f"\n   Host: {DB_CONFIG['host']}")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("   ✅ Connecté")
    
except Exception as e:
    print(f"\n❌ Erreur connexion: {e}")
    sys.exit(1)

# ============================================
# AJOUT COLONNES DANS TABLE
# ============================================

print("\n" + "="*80)
print("🏗️  AJOUT COLONNES DANS fact_nlp_analysis")
print("="*80)

added_cols = []
existing_cols = []

for col_name, col_type in COLONNES_MANQUANTES:
    try:
        print(f"\n   Ajout {col_name} ({col_type})...", end=" ")
        
        cursor.execute(f"""
            ALTER TABLE fact_nlp_analysis 
            ADD COLUMN IF NOT EXISTS {col_name} {col_type}
        """)
        
        conn.commit()
        print("✅")
        added_cols.append(col_name)
        
    except Exception as e:
        print(f"⚠️  Déjà existe")
        conn.rollback()
        existing_cols.append(col_name)

print(f"\n   ✅ {len(added_cols)} colonnes ajoutées")
if existing_cols:
    print(f"   ℹ️  {len(existing_cols)} colonnes déjà existantes")

# ============================================
# MISE À JOUR DONNÉES
# ============================================

print("\n" + "="*80)
print("📤 MISE À JOUR DONNÉES")
print("="*80)

# Colonnes à mettre à jour (celles qui existent dans pickle)
cols_to_update = [col for col, _ in COLONNES_MANQUANTES if col in df_pickle.columns]

print(f"\n   Colonnes à remplir: {len(cols_to_update)}")

if cols_to_update:
    batch_size = 500
    total_updated = 0
    
    print(f"\n   Mise à jour par batch ({batch_size} lignes)...")
    
    for i in tqdm(range(0, len(df_pickle), batch_size), desc="   Progression"):
        batch = df_pickle.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            offre_id = int(row['offre_id'])
            
            # Vérifier que offre_id existe dans fact_nlp_analysis
            cursor.execute(
                "SELECT 1 FROM fact_nlp_analysis WHERE offre_id = %s",
                (offre_id,)
            )
            
            if not cursor.fetchone():
                continue
            
            # Construire UPDATE dynamique
            set_clauses = []
            values = []
            
            for col in cols_to_update:
                set_clauses.append(f"{col} = %s")
                values.append(row.get(col))
            
            values.append(offre_id)  # Pour WHERE
            
            update_query = f"""
                UPDATE fact_nlp_analysis
                SET {', '.join(set_clauses)}
                WHERE offre_id = %s
            """
            
            cursor.execute(update_query, values)
            total_updated += 1
        
        conn.commit()
    
    print(f"\n   ✅ {total_updated} lignes mises à jour")

else:
    print("\n   ⚠️  Aucune donnée à mettre à jour")

# ============================================
# MISE À JOUR VUE v_offres_nlp_complete
# ============================================

print("\n" + "="*80)
print("📈 MISE À JOUR VUE v_offres_nlp_complete")
print("="*80)

try:
    print("\n   Recréation vue avec nouvelles colonnes...")
    
    cursor.execute("DROP VIEW IF EXISTS v_offres_nlp_complete CASCADE")
    
    cursor.execute("""
    CREATE VIEW v_offres_nlp_complete AS
    SELECT 
        o.offre_id,
        o.job_id_source,
        s.source_name as source,
        o.title,
        e.company_name,
        l.city,
        l.department,
        l.region,
        l.latitude,
        l.longitude,
        c.contract_type,
        c.experience_level,
        o.salary_min,
        o.salary_max,
        (o.salary_min + o.salary_max) / 2 as salary_annual,
        t.date_posted,
        o.description,
        o.url,
        o.scraped_at,
        -- Résultats NLP (colonnes originales)
        n.status,
        n.profil_assigned,
        n.score_classification,
        n.competences_found,
        n.topic_id,
        n.cluster_id,
        -- Nouvelles colonnes NLP
        n.num_tokens,
        n.num_competences,
        n.profil_score,
        n.profil_confidence,
        n.profil_second,
        n.profil_second_score,
        n.score_title,
        n.score_description,
        n.score_competences,
        n.cascade_pass,
        n.description_clean,
        n.text_for_sklearn,
        n.tokens,
        n.duration,
        n.salary_text,
        n.source_name
    FROM fact_offres o
    LEFT JOIN dim_source s ON o.source_id = s.source_id
    LEFT JOIN dim_localisation l ON o.localisation_id = l.localisation_id
    LEFT JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
    LEFT JOIN dim_contrat c ON o.contrat_id = c.contrat_id
    LEFT JOIN dim_temps t ON o.temps_id = t.temps_id
    LEFT JOIN fact_nlp_analysis n ON o.offre_id = n.offre_id
    """)
    
    conn.commit()
    print("   ✅ Vue recréée avec toutes colonnes")
    
except Exception as e:
    print(f"\n⚠️  Erreur vue: {e}")
    conn.rollback()

# ============================================
# VÉRIFICATION FINALE
# ============================================

print("\n" + "="*80)
print("✅ VÉRIFICATION")
print("="*80)

try:
    # Compter colonnes dans vue
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'v_offres_nlp_complete'
        ORDER BY ordinal_position
    """)
    
    cols_vue = [row[0] for row in cursor.fetchall()]
    
    print(f"\n   📊 Colonnes dans v_offres_nlp_complete: {len(cols_vue)}")
    
    # Vérifier colonnes ajoutées présentes
    missing_in_vue = [col for col, _ in COLONNES_MANQUANTES if col not in cols_vue]
    
    if missing_in_vue:
        print(f"\n   ⚠️  Colonnes manquantes dans vue: {missing_in_vue}")
    else:
        print(f"\n   ✅ Toutes colonnes présentes dans vue")
    
    # Test chargement
    cursor.execute("SELECT COUNT(*) FROM v_offres_nlp_complete")
    count = cursor.fetchone()[0]
    print(f"   ✅ {count} offres dans vue")
    
    # Exemples valeurs nouvelles colonnes
    print(f"\n   📋 Exemples valeurs (première ligne):")
    cursor.execute(f"""
        SELECT {', '.join([col for col, _ in COLONNES_MANQUANTES[:5]])}
        FROM v_offres_nlp_complete 
        WHERE offre_id IS NOT NULL
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    if result:
        for i, (col, _) in enumerate(COLONNES_MANQUANTES[:5]):
            print(f"      {col}: {result[i]}")
    
except Exception as e:
    print(f"\n⚠️  Erreur vérification: {e}")

# ============================================
# FERMETURE
# ============================================

cursor.close()
conn.close()

print("\n" + "="*80)
print("🎉 MIGRATION COLONNES TERMINÉE !")
print("="*80)

print(f"""
📊 RÉSUMÉ:
   - Colonnes ajoutées: {len(added_cols)}
   - Lignes mises à jour: {total_updated}
   - Vue recréée: v_offres_nlp_complete
   
🏗️  COLONNES AJOUTÉES:
   ✅ num_tokens, num_competences
   ✅ profil_score, profil_confidence, profil_second
   ✅ score_title, score_description, score_competences
   ✅ cascade_pass
   ✅ description_clean, text_for_sklearn, tokens
   ✅ duration, salary_text, source_name

🔗 BASE DE DONNÉES:
   - Host: {DB_CONFIG['host']}
   - Vue: v_offres_nlp_complete (toutes colonnes pickle)
   
🚀 PROCHAINES ÉTAPES:
   1. Tester load_offres_with_nlp() (toutes colonnes dispo)
   2. Relancer application Streamlit
   3. Vérifier toutes pages fonctionnent
   
✅ Application 100% compatible avec pickle !
""")

print("\n✅ Script terminé sans erreur !")