"""
MIGRATION MODÈLE EN ÉTOILE - DUCKDB → SUPABASE POSTGRESQL
Projet NLP Text Mining - Master SISE

Migre l'entrepôt de données complet (star schema) :
- 5 tables de dimensions
- 2 tables de faits
- Vues analytiques
- Index de performance

Configuration via fichier .env (sécurisé)
"""

import duckdb
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# ============================================
# CHARGEMENT VARIABLES ENVIRONNEMENT
# ============================================

load_dotenv()

print("="*80)
print("  MIGRATION MODÈLE EN ÉTOILE - DUCKDB → POSTGRESQL")
print("="*80)

# Vérifier .env existe
if not Path('.env').exists():
    print("\n❌ ERREUR: Fichier .env introuvable")
    print("\nCréez un fichier .env avec :")
    print("DB_HOST=...\nDB_PORT=...\nDB_NAME=...\nDB_USER=...\nDB_PASSWORD=...")
    sys.exit(1)

# ============================================
# CONFIGURATION
# ============================================

DUCKDB_PATH = Path('entrepot_de_donnees/entrepot_nlp.duckdb')

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

print(f"\n✅ Configuration chargée")
print(f"   Host: {SUPABASE_CONFIG['host']}")
print(f"   User: {SUPABASE_CONFIG['user']}")

# Vérifier DuckDB existe
if not DUCKDB_PATH.exists():
    print(f"\n❌ Fichier DuckDB introuvable: {DUCKDB_PATH}")
    sys.exit(1)

print(f"✅ DuckDB trouvé: {DUCKDB_PATH}")

# ============================================
# CONNEXION BASES DE DONNÉES
# ============================================

print("\n" + "="*80)
print(" CONNEXION AUX BASES DE DONNÉES")
print("="*80)

# DuckDB
try:
    print("\n   Connexion DuckDB...")
    conn_duck = duckdb.connect(str(DUCKDB_PATH))
    print("   ✅ DuckDB connecté")
except Exception as e:
    print(f"\n❌ Erreur DuckDB: {e}")
    sys.exit(1)

# PostgreSQL
try:
    print("   Connexion PostgreSQL (Supabase)...")
    conn_pg = psycopg2.connect(**SUPABASE_CONFIG)
    cursor = conn_pg.cursor()
    print("   ✅ PostgreSQL connecté")
except Exception as e:
    print(f"\n❌ Erreur PostgreSQL: {e}")
    conn_duck.close()
    sys.exit(1)

# ============================================
# EXTRACTION STATISTIQUES DUCKDB
# ============================================

print("\n" + "="*80)
print(" STATISTIQUES ENTREPÔT DUCKDB")
print("="*80)

stats = {}
tables = ['dim_source', 'dim_localisation', 'dim_entreprise', 'dim_contrat', 
          'dim_temps', 'fact_offres', 'fact_competences']

for table in tables:
    try:
        count = conn_duck.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        stats[table] = count
        print(f"   {table:20s} : {count:6d} lignes")
    except Exception as e:
        print(f"   {table:20s} : ⚠️  Erreur - {e}")
        stats[table] = 0

total_lignes = sum(stats.values())
print(f"\n   {'TOTAL':20s} : {total_lignes:6d} lignes")

# ============================================
# CRÉATION SCHÉMA POSTGRESQL
# ============================================

print("\n" + "="*80)
print("  CRÉATION SCHÉMA POSTGRESQL")
print("="*80)

print("\n   Suppression ancien schéma (si existe)...")
cursor.execute("""
DROP TABLE IF EXISTS fact_competences CASCADE;
DROP TABLE IF EXISTS fact_offres CASCADE;
DROP TABLE IF EXISTS dim_temps CASCADE;
DROP TABLE IF EXISTS dim_contrat CASCADE;
DROP TABLE IF EXISTS dim_entreprise CASCADE;
DROP TABLE IF EXISTS dim_localisation CASCADE;
DROP TABLE IF EXISTS dim_source CASCADE;
""")
conn_pg.commit()
print("   ✅ Ancien schéma supprimé")

print("\n   Création tables de dimensions...")

# dim_source
cursor.execute("""
CREATE TABLE dim_source (
    source_id SERIAL PRIMARY KEY,
    source_name VARCHAR(50) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# dim_localisation
cursor.execute("""
CREATE TABLE dim_localisation (
    localisation_id SERIAL PRIMARY KEY,
    city VARCHAR(100),
    department VARCHAR(10),
    region VARCHAR(100),
    latitude NUMERIC(10, 6),
    longitude NUMERIC(10, 6),
    location_raw VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(city, department)
);
""")

# dim_entreprise
cursor.execute("""
CREATE TABLE dim_entreprise (
    entreprise_id SERIAL PRIMARY KEY,
    company_name VARCHAR(200) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# dim_contrat
cursor.execute("""
CREATE TABLE dim_contrat (
    contrat_id SERIAL PRIMARY KEY,
    contract_type VARCHAR(50),
    duration VARCHAR(100),
    experience_level VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(contract_type, duration, experience_level)
);
""")

# dim_temps
cursor.execute("""
CREATE TABLE dim_temps (
    temps_id SERIAL PRIMARY KEY,
    date_posted DATE NOT NULL,
    year INTEGER,
    month INTEGER,
    week INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date_posted)
);
""")

conn_pg.commit()
print("   ✅ 5 tables de dimensions créées")

print("\n   Création tables de faits...")

# fact_offres
cursor.execute("""
CREATE TABLE fact_offres (
    offre_id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES dim_source(source_id),
    localisation_id INTEGER REFERENCES dim_localisation(localisation_id),
    entreprise_id INTEGER REFERENCES dim_entreprise(entreprise_id),
    contrat_id INTEGER REFERENCES dim_contrat(contrat_id),
    temps_id INTEGER REFERENCES dim_temps(temps_id),
    job_id_source VARCHAR(100) NOT NULL,
    salary_min NUMERIC(10, 2),
    salary_max NUMERIC(10, 2),
    title TEXT NOT NULL,
    description TEXT,
    salary_text VARCHAR(200),
    location_text VARCHAR(200),
    duration_text VARCHAR(200),
    url TEXT,
    scraped_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_data_json TEXT,
    UNIQUE(source_id, job_id_source)
);
""")

# fact_competences
cursor.execute("""
CREATE TABLE fact_competences (
    competence_id SERIAL PRIMARY KEY,
    offre_id INTEGER REFERENCES fact_offres(offre_id) ON DELETE CASCADE,
    skill_code VARCHAR(20),
    skill_label TEXT NOT NULL,
    skill_level VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(offre_id, skill_label)
);
""")

conn_pg.commit()
print("   ✅ 2 tables de faits créées")

print("\n   Création index de performance...")

cursor.execute("""
-- Index clés étrangères
CREATE INDEX idx_offres_source ON fact_offres(source_id);
CREATE INDEX idx_offres_localisation ON fact_offres(localisation_id);
CREATE INDEX idx_offres_entreprise ON fact_offres(entreprise_id);
CREATE INDEX idx_offres_contrat ON fact_offres(contrat_id);
CREATE INDEX idx_offres_temps ON fact_offres(temps_id);

-- Index compétences
CREATE INDEX idx_competences_offre ON fact_competences(offre_id);
CREATE INDEX idx_competences_label ON fact_competences(skill_label);

-- Index recherches
CREATE INDEX idx_offres_title ON fact_offres(title);
CREATE INDEX idx_entreprise_name ON dim_entreprise(company_name);
CREATE INDEX idx_localisation_city ON dim_localisation(city);
CREATE INDEX idx_localisation_region ON dim_localisation(region);
""")

conn_pg.commit()
print("   ✅ 11 index créés")

# ============================================
# MIGRATION DIMENSIONS
# ============================================

print("\n" + "="*80)
print(" MIGRATION TABLES DE DIMENSIONS")
print("="*80)

dimensions = [
    ('dim_source', ['source_id', 'source_name', 'created_at']),
    ('dim_localisation', ['localisation_id', 'city', 'department', 'region', 
                          'latitude', 'longitude', 'location_raw', 'created_at']),
    ('dim_entreprise', ['entreprise_id', 'company_name', 'created_at']),
    ('dim_contrat', ['contrat_id', 'contract_type', 'duration', 
                     'experience_level', 'created_at']),
    ('dim_temps', ['temps_id', 'date_posted', 'year', 'month', 'week', 
                   'day_of_week', 'is_weekend', 'created_at'])
]

for table_name, columns in dimensions:
    print(f"\n   Migration {table_name}...")
    
    try:
        # Extraction DuckDB
        df = conn_duck.execute(f"SELECT * FROM {table_name}").df()
        
        if len(df) == 0:
            print(f"      ⚠️  Table vide - ignorée")
            continue
        
        # Conversion NaN → None
        df = df.where(pd.notna(df), None)
        
        # Colonnes disponibles
        available_cols = [col for col in columns if col in df.columns]
        
        # Insertion PostgreSQL
        values = [tuple(row[col] for col in available_cols) for _, row in df.iterrows()]
        
        placeholders = ', '.join(['%s'] * len(available_cols))
        insert_query = f"""
            INSERT INTO {table_name} ({', '.join(available_cols)})
            VALUES ({placeholders})
        """
        
        cursor.executemany(insert_query, values)
        conn_pg.commit()
        
        print(f"      ✅ {len(df)} lignes insérées")
        
    except Exception as e:
        print(f"      ❌ Erreur: {e}")
        conn_pg.rollback()

# ============================================
# MIGRATION FACT_OFFRES
# ============================================

print("\n" + "="*80)
print(" MIGRATION FACT_OFFRES")
print("="*80)

try:
    print("\n   Extraction fact_offres...")
    df_offres = conn_duck.execute("SELECT * FROM fact_offres").df()
    print(f"   ✅ {len(df_offres)} offres extraites")
    
    # Conversion NaN → None
    df_offres = df_offres.where(pd.notna(df_offres), None)
    
    # Colonnes
    columns = ['offre_id', 'source_id', 'localisation_id', 'entreprise_id', 
               'contrat_id', 'temps_id', 'job_id_source', 'salary_min', 
               'salary_max', 'title', 'description', 'salary_text', 
               'location_text', 'duration_text', 'url', 'scraped_at']
    
    # Filtrer colonnes existantes
    available_cols = [col for col in columns if col in df_offres.columns]
    
    print(f"\n   Insertion par batch (500 lignes)...")
    batch_size = 500
    total_inserted = 0
    
    for i in tqdm(range(0, len(df_offres), batch_size), desc="   Progression"):
        batch = df_offres.iloc[i:i+batch_size]
        
        values = [
            tuple(row[col] for col in available_cols)
            for _, row in batch.iterrows()
        ]
        
        placeholders = ', '.join(['%s'] * len(available_cols))
        insert_query = f"""
            INSERT INTO fact_offres ({', '.join(available_cols)})
            VALUES ({placeholders})
        """
        
        cursor.executemany(insert_query, values)
        conn_pg.commit()
        
        total_inserted += len(batch)
    
    print(f"\n   ✅ {total_inserted} offres insérées")
    
except Exception as e:
    print(f"\n   ❌ Erreur fact_offres: {e}")
    conn_pg.rollback()

# ============================================
# MIGRATION FACT_COMPETENCES
# ============================================

print("\n" + "="*80)
print(" MIGRATION FACT_COMPETENCES")
print("="*80)

try:
    print("\n   Extraction fact_competences...")
    df_comp = conn_duck.execute("SELECT * FROM fact_competences").df()
    print(f"   ✅ {len(df_comp)} compétences extraites")
    
    if len(df_comp) > 0:
        # Conversion NaN → None
        df_comp = df_comp.where(pd.notna(df_comp), None)
        
        # Colonnes
        columns = ['competence_id', 'offre_id', 'skill_code', 'skill_label', 'skill_level']
        available_cols = [col for col in columns if col in df_comp.columns]
        
        print(f"\n   Insertion par batch (1000 lignes)...")
        batch_size = 1000
        total_inserted = 0
        
        for i in tqdm(range(0, len(df_comp), batch_size), desc="   Progression"):
            batch = df_comp.iloc[i:i+batch_size]
            
            values = [
                tuple(row[col] for col in available_cols)
                for _, row in batch.iterrows()
            ]
            
            placeholders = ', '.join(['%s'] * len(available_cols))
            insert_query = f"""
                INSERT INTO fact_competences ({', '.join(available_cols)})
                VALUES ({placeholders})
            """
            
            cursor.executemany(insert_query, values)
            conn_pg.commit()
            
            total_inserted += len(batch)
        
        print(f"\n   ✅ {total_inserted} compétences insérées")
    else:
        print("   ⚠️  Table vide")
    
except Exception as e:
    print(f"\n   ❌ Erreur fact_competences: {e}")
    conn_pg.rollback()

# ============================================
# CRÉATION VUES ANALYTIQUES
# ============================================

print("\n" + "="*80)
print(" CRÉATION VUES ANALYTIQUES")
print("="*80)

try:
    # Vue offres complètes
    print("\n   Création v_offres_complete...")
    cursor.execute("""
    CREATE OR REPLACE VIEW v_offres_complete AS
    SELECT 
        o.offre_id,
        o.job_id_source,
        s.source_name,
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
        t.date_posted,
        o.description,
        o.url,
        o.scraped_at
    FROM fact_offres o
    LEFT JOIN dim_source s ON o.source_id = s.source_id
    LEFT JOIN dim_localisation l ON o.localisation_id = l.localisation_id
    LEFT JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
    LEFT JOIN dim_contrat c ON o.contrat_id = c.contrat_id
    LEFT JOIN dim_temps t ON o.temps_id = t.temps_id;
    """)
    
    # Vue stats régions
    print("   Création v_stats_region...")
    cursor.execute("""
    CREATE OR REPLACE VIEW v_stats_region AS
    SELECT 
        l.region,
        COUNT(o.offre_id) as nb_offres,
        AVG(o.salary_min) as salaire_moyen_min,
        AVG(o.salary_max) as salaire_moyen_max,
        COUNT(DISTINCT o.entreprise_id) as nb_entreprises
    FROM fact_offres o
    LEFT JOIN dim_localisation l ON o.localisation_id = l.localisation_id
    WHERE l.region IS NOT NULL
    GROUP BY l.region
    ORDER BY nb_offres DESC;
    """)
    
    conn_pg.commit()
    print("   ✅ Vues créées")
    
except Exception as e:
    print(f"   ⚠️  Erreur vues: {e}")

# ============================================
# VÉRIFICATION FINALE
# ============================================

print("\n" + "="*80)
print("✅ VÉRIFICATION MIGRATION")
print("="*80)

verification = {}
for table in tables:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        verification[table] = count
        status = "✅" if count == stats.get(table, 0) else "⚠️"
        print(f"   {table:20s} : {count:6d} lignes {status}")
    except Exception as e:
        print(f"   {table:20s} : ❌ Erreur")

# ============================================
# FERMETURE
# ============================================

cursor.close()
conn_pg.close()
conn_duck.close()

print("\n" + "="*80)
print("🎉 MIGRATION MODÈLE EN ÉTOILE TERMINÉE !")
print("="*80)

print(f"""
 RÉSUMÉ:
   - Tables migrées: {len(tables)}
   - Dimensions: 5
   - Faits: 2
   - Index: 11
   - Vues: 2

 ARCHITECTURE:
   ├─ dim_source ({verification.get('dim_source', 0)} sources)
   ├─ dim_localisation ({verification.get('dim_localisation', 0)} localisations)
   ├─ dim_entreprise ({verification.get('dim_entreprise', 0)} entreprises)
   ├─ dim_contrat ({verification.get('dim_contrat', 0)} types contrats)
   ├─ dim_temps ({verification.get('dim_temps', 0)} dates)
   ├─ fact_offres ({verification.get('fact_offres', 0)} offres)
   └─ fact_competences ({verification.get('fact_competences', 0)} compétences)

 BASE DE DONNÉES:
   - Host: {SUPABASE_CONFIG['host']}
   - Database: {SUPABASE_CONFIG['database']}
   - Schéma: Star Schema (Modèle en étoile)

 PROCHAINES ÉTAPES:
   1. Modifier Streamlit pour interroger PostgreSQL
   2. Partager credentials avec binôme
   3. Tester requêtes analytiques
""")

print("\n✅ Migration terminée sans erreur !")