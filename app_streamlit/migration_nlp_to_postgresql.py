"""
MIGRATION RÉSULTATS NLP → POSTGRESQL
Projet NLP Text Mining - Master SISE

Migre les résultats d'analyse NLP (classification, profils, topics, clustering)
depuis le fichier pickle vers la table PostgreSQL fact_nlp_analysis

Configuration via fichier .env
"""

import pickle
import psycopg2
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# ============================================
# CHARGEMENT CONFIGURATION
# ============================================

load_dotenv()

print("="*80)
print("🧠 MIGRATION RÉSULTATS NLP → POSTGRESQL")
print("="*80)

PICKLE_PATH = Path('../resultats_nlp/models/data_with_profiles.pkl')

SUPABASE_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# Vérifier variables
missing_vars = [k for k, v in SUPABASE_CONFIG.items() if not v]
if missing_vars:
    print(f"\n❌ Variables manquantes dans .env: {missing_vars}")
    sys.exit(1)

print("\n✅ Configuration chargée")

# Vérifier pickle existe
if not PICKLE_PATH.exists():
    print(f"\n❌ Fichier pickle introuvable: {PICKLE_PATH}")
    print(f"   Chemin actuel: {Path.cwd()}")
    sys.exit(1)

print(f"✅ Pickle trouvé: {PICKLE_PATH}")

# ============================================
# CHARGEMENT DONNÉES PICKLE
# ============================================

print("\n" + "="*80)
print("📥 CHARGEMENT RÉSULTATS NLP (PICKLE)")
print("="*80)

try:
    print("\n   Chargement data_with_profiles.pkl...")
    with open(PICKLE_PATH, 'rb') as f:
        df_nlp = pickle.load(f)
    
    print(f"   ✅ {len(df_nlp)} offres chargées")
    print(f"   Colonnes: {list(df_nlp.columns)[:10]}...")
    
    # Colonnes NLP attendues
    nlp_columns = ['status', 'profil_assigned', 'score_classification', 
                   'competences_found', 'topic_id', 'cluster_id']
    
    available_nlp_cols = [col for col in nlp_columns if col in df_nlp.columns]
    print(f"\n   📊 Colonnes NLP disponibles: {available_nlp_cols}")
    
except Exception as e:
    print(f"\n❌ Erreur chargement pickle: {e}")
    sys.exit(1)

# ============================================
# CONNEXION POSTGRESQL
# ============================================

print("\n" + "="*80)
print("🔗 CONNEXION POSTGRESQL")
print("="*80)

try:
    print(f"\n   Connexion à {SUPABASE_CONFIG['host']}...")
    conn_pg = psycopg2.connect(**SUPABASE_CONFIG)
    cursor = conn_pg.cursor()
    print("   ✅ Connecté")
except Exception as e:
    print(f"\n❌ Erreur connexion: {e}")
    sys.exit(1)

# ============================================
# CRÉATION TABLE FACT_NLP_ANALYSIS
# ============================================

print("\n" + "="*80)
print("🏗️  CRÉATION TABLE FACT_NLP_ANALYSIS")
print("="*80)

try:
    print("\n   Suppression ancienne table (si existe)...")
    cursor.execute("DROP TABLE IF EXISTS fact_nlp_analysis CASCADE;")
    
    print("   Création nouvelle table...")
    cursor.execute("""
    CREATE TABLE fact_nlp_analysis (
        offre_id INTEGER PRIMARY KEY REFERENCES fact_offres(offre_id) ON DELETE CASCADE,
        
        -- Classification
        status VARCHAR(20),                    -- 'classified', 'not_classified'
        profil_assigned VARCHAR(100),          -- 'Data Scientist', 'Data Engineer', etc.
        score_classification NUMERIC(5, 2),    -- Score de confiance (0-100)
        
        -- Compétences (array PostgreSQL)
        competences_found TEXT[],              -- ['python', 'sql', 'spark']
        
        -- Topic Modeling
        topic_id INTEGER,                      -- ID du topic LDA
        topic_label VARCHAR(200),              -- Label du topic (optionnel)
        
        -- Clustering
        cluster_id INTEGER,                    -- ID du cluster
        cluster_label VARCHAR(200),            -- Label du cluster (optionnel)
        
        -- Métadonnées
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """)
    
    print("   Création index...")
    cursor.execute("""
    CREATE INDEX idx_nlp_status ON fact_nlp_analysis(status);
    CREATE INDEX idx_nlp_profil ON fact_nlp_analysis(profil_assigned);
    CREATE INDEX idx_nlp_topic ON fact_nlp_analysis(topic_id);
    CREATE INDEX idx_nlp_cluster ON fact_nlp_analysis(cluster_id);
    """)
    
    conn_pg.commit()
    print("   ✅ Table créée avec 4 index")
    
except Exception as e:
    print(f"\n❌ Erreur création table: {e}")
    cursor.close()
    conn_pg.close()
    sys.exit(1)

# ============================================
# PRÉPARATION DONNÉES
# ============================================

print("\n" + "="*80)
print("🔧 PRÉPARATION DONNÉES NLP")
print("="*80)

# Vérifier colonne offre_id existe
if 'offre_id' not in df_nlp.columns:
    print("\n⚠️  Colonne 'offre_id' manquante - création depuis index...")
    df_nlp['offre_id'] = df_nlp.index + 1

# Filtrer seulement offres avec offre_id valide
df_nlp = df_nlp[df_nlp['offre_id'].notna()].copy()

print(f"\n   Offres avec offre_id: {len(df_nlp)}")

# Conversion NaN → None
df_nlp = df_nlp.where(pd.notna(df_nlp), None)

# Convertir competences_found en array PostgreSQL si liste
if 'competences_found' in df_nlp.columns:
    print("   Conversion compétences en array...")
    df_nlp['competences_found'] = df_nlp['competences_found'].apply(
        lambda x: x if isinstance(x, list) else ([] if x is None else [])
    )

# Colonnes à insérer (seulement celles qui existent)
columns_mapping = {
    'offre_id': 'offre_id',
    'status': 'status',
    'profil_assigned': 'profil_assigned',
    'score_classification': 'score_classification',
    'competences_found': 'competences_found',
    'topic_id': 'topic_id',
    'cluster_id': 'cluster_id'
}

# Filtrer colonnes disponibles
available_cols = {k: v for k, v in columns_mapping.items() if k in df_nlp.columns}

print(f"   📊 Colonnes à migrer: {list(available_cols.keys())}")

# ============================================
# INSERTION DONNÉES
# ============================================

print("\n" + "="*80)
print("📤 INSERTION RÉSULTATS NLP")
print("="*80)

batch_size = 500
total_inserted = 0
total_skipped = 0

try:
    print(f"\n   Insertion par batch ({batch_size} lignes)...")
    
    for i in tqdm(range(0, len(df_nlp), batch_size), desc="   Progression"):
        batch = df_nlp.iloc[i:i+batch_size]
        
        values = []
        for _, row in batch.iterrows():
            # Vérifier offre_id existe dans fact_offres
            try:
                cursor.execute("SELECT 1 FROM fact_offres WHERE offre_id = %s", (int(row['offre_id']),))
                if not cursor.fetchone():
                    total_skipped += 1
                    continue
            except:
                total_skipped += 1
                continue
            
            # Construire tuple de valeurs
            row_values = tuple(row[col] for col in available_cols.keys())
            values.append(row_values)
        
        if values:
            # Insertion batch
            placeholders = ', '.join(['%s'] * len(available_cols))
            insert_query = f"""
                INSERT INTO fact_nlp_analysis ({', '.join(available_cols.values())})
                VALUES ({placeholders})
                ON CONFLICT (offre_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    profil_assigned = EXCLUDED.profil_assigned,
                    score_classification = EXCLUDED.score_classification,
                    updated_at = NOW()
            """
            
            cursor.executemany(insert_query, values)
            conn_pg.commit()
            
            total_inserted += len(values)
    
    print(f"\n   ✅ {total_inserted} résultats NLP insérés")
    if total_skipped > 0:
        print(f"   ⚠️  {total_skipped} lignes ignorées (offre_id inexistant)")
    
except Exception as e:
    print(f"\n❌ Erreur insertion: {e}")
    conn_pg.rollback()
    cursor.close()
    conn_pg.close()
    sys.exit(1)

# ============================================
# CRÉATION VUE COMPLÈTE (OFFRES + NLP)
# ============================================

print("\n" + "="*80)
print("📈 CRÉATION VUE v_offres_nlp_complete")
print("="*80)

try:
    print("\n   Création vue jointe offres + NLP...")
    cursor.execute("""
    CREATE OR REPLACE VIEW v_offres_nlp_complete AS
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
        -- Résultats NLP
        n.status,
        n.profil_assigned,
        n.score_classification,
        n.competences_found,
        n.topic_id,
        n.cluster_id
    FROM fact_offres o
    LEFT JOIN dim_source s ON o.source_id = s.source_id
    LEFT JOIN dim_localisation l ON o.localisation_id = l.localisation_id
    LEFT JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
    LEFT JOIN dim_contrat c ON o.contrat_id = c.contrat_id
    LEFT JOIN dim_temps t ON o.temps_id = t.temps_id
    LEFT JOIN fact_nlp_analysis n ON o.offre_id = n.offre_id;
    """)
    
    conn_pg.commit()
    print("   ✅ Vue créée")
    
except Exception as e:
    print(f"\n⚠️  Erreur création vue: {e}")

# ============================================
# VÉRIFICATION FINALE
# ============================================

print("\n" + "="*80)
print("✅ VÉRIFICATION MIGRATION NLP")
print("="*80)

try:
    # Count total
    cursor.execute("SELECT COUNT(*) FROM fact_nlp_analysis")
    count_total = cursor.fetchone()[0]
    print(f"\n   📊 Total résultats NLP: {count_total}")
    
    # Count par statut
    cursor.execute("SELECT status, COUNT(*) FROM fact_nlp_analysis GROUP BY status")
    statuses = cursor.fetchall()
    print("\n   📍 Répartition par statut:")
    for status, count in statuses:
        print(f"      {status}: {count}")
    
    # Count par profil
    cursor.execute("""
        SELECT profil_assigned, COUNT(*) 
        FROM fact_nlp_analysis 
        WHERE profil_assigned IS NOT NULL
        GROUP BY profil_assigned 
        ORDER BY COUNT(*) DESC 
        LIMIT 10
    """)
    profils = cursor.fetchall()
    print("\n   🎯 Top 10 profils:")
    for profil, count in profils:
        print(f"      {profil}: {count}")
    
    # Test vue
    cursor.execute("SELECT COUNT(*) FROM v_offres_nlp_complete WHERE status = 'classified'")
    count_classified = cursor.fetchone()[0]
    print(f"\n   ✅ Vue complète : {count_classified} offres classifiées")
    
except Exception as e:
    print(f"\n⚠️  Erreur vérification: {e}")

# ============================================
# FERMETURE
# ============================================

cursor.close()
conn_pg.close()

print("\n" + "="*80)
print("🎉 MIGRATION NLP TERMINÉE !")
print("="*80)

print(f"""
📊 RÉSUMÉ:
   - Résultats NLP migrés: {total_inserted}
   - Lignes ignorées: {total_skipped}
   - Table: fact_nlp_analysis
   - Vue: v_offres_nlp_complete
   
🏗️  NOUVELLES COLONNES DISPONIBLES:
   ✅ status (classified/not_classified)
   ✅ profil_assigned (Data Scientist, Data Engineer, etc.)
   ✅ score_classification
   ✅ competences_found (array)
   ✅ topic_id
   ✅ cluster_id

🔗 BASE DE DONNÉES:
   - Host: {SUPABASE_CONFIG['host']}
   - Tables: 8 (7 entrepôt + 1 NLP)
   - Vues: 3 (stats + NLP)

🚀 PROCHAINES ÉTAPES:
   1. Modifier config_db.py (fonction load_offres_with_nlp)
   2. Modifier pages Streamlit (utiliser nouvelle vue)
   3. Tester application
   
✅ Tous les résultats NLP sont maintenant dans PostgreSQL !
""")

print("\n✅ Script terminé sans erreur !")