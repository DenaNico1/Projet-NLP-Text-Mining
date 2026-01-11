"""
EXPORT SUPABASE → SQL DUMP pour PostgreSQL Docker
Projet NLP Text Mining - Master SISE

Exporte toutes les données de Supabase vers un fichier SQL
qui sera utilisé pour initialiser le PostgreSQL embarqué
"""

import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv
from pathlib import Path
import time

load_dotenv()

# Configuration Supabase
SUPABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-1-eu-north-1.pooler.supabase.com'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.znkulobexqmrshfkgynv'),
    'password': os.getenv('DB_PASSWORD', ''),
    'connect_timeout': 30,
    'keepalives': 1,
    'keepalives_idle': 30,
    'keepalives_interval': 10,
    'keepalives_count': 5
}

# Configuration export
BATCH_SIZE = 100  # Lignes par INSERT
FETCH_SIZE = 1000  # Lignes à fetcher à la fois (server-side cursor)

OUTPUT_DIR = Path("docker_init")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================
# FONCTIONS HELPER
# ============================================

def format_sql_value(val, col_type):
    """Formate une valeur Python en valeur SQL"""
    if val is None:
        return "NULL"
    
    # Gérer NaN/Infinity pour types numériques
    if col_type in ['numeric', 'real', 'double precision', 'integer', 'bigint']:
        # Vérifier si c'est NaN ou Infinity
        try:
            import math
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return "NULL"
        except:
            pass
        # Vérifier string 'NaN'
        if str(val).upper() in ['NAN', 'INF', '-INF', 'INFINITY', '-INFINITY']:
            return "NULL"
    
    if col_type == 'bytea':
        # Embeddings en format bytea
        hex_val = val.hex() if isinstance(val, bytes) else val
        return f"'\\x{hex_val}'::bytea"
    elif col_type == 'ARRAY':
        # Tableaux PostgreSQL
        if isinstance(val, list):
            # Convertir liste Python en format array PostgreSQL
            if not val:
                return "'{}'::text[]"
            # Échapper éléments
            escaped_items = [str(item).replace("'", "''").replace("\\", "\\\\") for item in val]
            array_str = "'{" + ",".join(escaped_items) + "}'"
            return array_str
        elif isinstance(val, str) and val.startswith('{'):
            # Déjà au format PostgreSQL array
            escaped = val.replace("'", "''")
            return f"'{escaped}'"
        else:
            return f"'{val}'"
    elif col_type in ['character varying', 'text']:
        # Échapper quotes et backslashes
        escaped = str(val).replace("\\", "\\\\").replace("'", "''")
        return f"'{escaped}'"
    elif col_type in ['timestamp without time zone', 'date']:
        return f"'{val}'"
    elif col_type in ['jsonb', 'json']:
        escaped = str(val).replace("\\", "\\\\").replace("'", "''")
        return f"'{escaped}'::jsonb"
    elif col_type == 'boolean':
        return "TRUE" if val else "FALSE"
    elif col_type in ['integer', 'bigint', 'numeric', 'real', 'double precision']:
        return str(val)
    else:
        # Par défaut : échapper et quoter
        escaped = str(val).replace("'", "''")
        return f"'{escaped}'"

def write_insert_batch(f, table, column_names, column_types, batch):
    """Écrit un batch INSERT dans le fichier SQL"""
    if not batch:
        return
    
    f.write(f"INSERT INTO {table} ({', '.join(column_names)}) VALUES\n")
    
    values_list = []
    for row in batch:
        values = []
        for idx, val in enumerate(row):
            col_name = column_names[idx]
            col_type = column_types[col_name]
            values.append(format_sql_value(val, col_type))
        
        values_list.append(f"({', '.join(values)})")
    
    f.write(",\n".join(values_list))
    f.write(";\n\n")

# ============================================
# EXPORT
# ============================================

print("🚀 Export Supabase vers SQL...")
print(f"Host: {SUPABASE_CONFIG['host']}")
print(f"Batch size: {BATCH_SIZE} | Fetch size: {FETCH_SIZE}\n")

# Connexion Supabase
try:
    conn = psycopg2.connect(**SUPABASE_CONFIG)
    cursor = conn.cursor()
    print("✅ Connexion Supabase réussie\n")
except Exception as e:
    print(f"❌ Erreur connexion: {e}")
    exit(1)

# Fichier SQL de sortie
sql_file = OUTPUT_DIR / "01_init_data.sql"

with open(sql_file, 'w', encoding='utf-8') as f:
    f.write("-- Export Supabase → PostgreSQL Docker\n")
    f.write("-- Généré automatiquement\n\n")
    f.write("SET client_encoding = 'UTF8';\n")
    f.write("SET standard_conforming_strings = on;\n\n")
    
    # ==========================================
    # 1. SCHÉMA ET TABLES
    # ==========================================
    
    print("📋 Récupération schéma des tables...")
    
    # Lister toutes les tables (hors système)
    cursor.execute("""
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY tablename
    """)
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"   Tables trouvées: {', '.join(tables)}\n")
    
    # Créer tables
    for table in tables:
        print(f"📊 Export table: {table}")
        
        # Récupérer DDL complet (avec UDT pour ARRAY)
        cursor.execute(f"""
            SELECT 
                column_name, 
                data_type, 
                character_maximum_length, 
                is_nullable,
                udt_name
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        
        # Générer CREATE TABLE
        f.write(f"\n-- Table: {table}\n")
        f.write(f"DROP TABLE IF EXISTS {table} CASCADE;\n")
        f.write(f"CREATE TABLE {table} (\n")
        
        col_defs = []
        for col_name, data_type, max_length, nullable, udt_name in columns:
            col_def = f"    {col_name} "
            
            # Type de données
            if data_type == 'ARRAY':
                # Gérer les tableaux (ex: _text -> TEXT[], _int4 -> INTEGER[])
                if udt_name.startswith('_'):
                    base_type = udt_name[1:]  # Retirer underscore
                    if base_type == 'text':
                        col_def += "TEXT[]"
                    elif base_type == 'varchar':
                        col_def += "VARCHAR[]"
                    elif base_type == 'int4':
                        col_def += "INTEGER[]"
                    elif base_type == 'int8':
                        col_def += "BIGINT[]"
                    elif base_type == 'float4':
                        col_def += "REAL[]"
                    elif base_type == 'float8':
                        col_def += "DOUBLE PRECISION[]"
                    else:
                        col_def += f"{base_type.upper()}[]"
                else:
                    col_def += "TEXT[]"  # Par défaut
            elif data_type == 'character varying':
                col_def += f"VARCHAR({max_length})" if max_length else "VARCHAR"
            elif data_type == 'bytea':
                col_def += "BYTEA"
            elif data_type == 'integer':
                col_def += "INTEGER"
            elif data_type == 'bigint':
                col_def += "BIGINT"
            elif data_type == 'timestamp without time zone':
                col_def += "TIMESTAMP"
            elif data_type == 'date':
                col_def += "DATE"
            elif data_type == 'boolean':
                col_def += "BOOLEAN"
            elif data_type == 'numeric':
                col_def += "NUMERIC"
            elif data_type == 'real':
                col_def += "REAL"
            elif data_type == 'double precision':
                col_def += "DOUBLE PRECISION"
            elif data_type == 'text':
                col_def += "TEXT"
            elif data_type == 'jsonb':
                col_def += "JSONB"
            elif data_type == 'json':
                col_def += "JSON"
            else:
                col_def += data_type.upper()
            
            # Nullable
            if nullable == 'NO':
                col_def += " NOT NULL"
            
            col_defs.append(col_def)
        
        f.write(",\n".join(col_defs))
        f.write("\n);\n\n")
    
    # ==========================================
    # 2. DONNÉES
    # ==========================================
    
    print("\n📦 Export données...")
    
    for table in tables:
        print(f"   Extraction: {table}")
        
        # Compter lignes
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("      ⚠️  Table vide")
            continue
        
        print(f"      {count:,} lignes")
        
        # Récupérer métadonnées colonnes (avec UDT pour détecter ARRAY)
        cursor.execute(f"""
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """)
        columns_info = cursor.fetchall()
        column_names = [c[0] for c in columns_info]
        # Mapper type complet (ARRAY si data_type = 'ARRAY')
        column_types = {}
        for col_name, data_type, udt_name in columns_info:
            if data_type == 'ARRAY':
                column_types[col_name] = 'ARRAY'
            else:
                column_types[col_name] = data_type
        
        # Générer header INSERT
        f.write(f"-- Données table: {table} ({count:,} lignes)\n")
        
        # Utiliser curseur nommé (server-side) pour éviter timeout
        # Nom unique pour éviter conflits
        cursor_name = f"export_{table}_{int(time.time())}"
        
        # Créer connexion dédiée avec autocommit=False pour curseur serveur
        export_conn = psycopg2.connect(**SUPABASE_CONFIG)
        export_conn.set_session(readonly=True, autocommit=False)
        
        try:
            # Curseur server-side avec fetchmany
            with export_conn.cursor(name=cursor_name) as export_cursor:
                export_cursor.itersize = FETCH_SIZE
                
                # Query toutes les données
                export_cursor.execute(f"SELECT * FROM {table}")
                
                rows_processed = 0
                batch = []
                
                # Fetch par chunks
                while True:
                    chunk = export_cursor.fetchmany(FETCH_SIZE)
                    if not chunk:
                        break
                    
                    # Traiter chunk
                    for row in chunk:
                        batch.append(row)
                        rows_processed += 1
                        
                        # Écrire batch complet
                        if len(batch) >= BATCH_SIZE:
                            write_insert_batch(f, table, column_names, column_types, batch)
                            batch = []
                        
                        # Afficher progression tous les 1000
                        if rows_processed % 1000 == 0:
                            print(f"      → {rows_processed:,}/{count:,} lignes", end='\r')
                
                # Écrire batch restant
                if batch:
                    write_insert_batch(f, table, column_names, column_types, batch)
                
                print(f"      ✅ {rows_processed:,} lignes exportées")
        
        finally:
            export_conn.close()
        
        f.write("\n")
    
    # ==========================================
    # 3. CONTRAINTES ET INDEX
    # ==========================================
    
    print("\n🔑 Export contraintes et index...")
    
    # Primary keys
    cursor.execute("""
        SELECT tc.table_name, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
        WHERE tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_schema = 'public'
    """)
    
    pks = {}
    for table, column in cursor.fetchall():
        if table not in pks:
            pks[table] = []
        pks[table].append(column)
    
    for table, columns in pks.items():
        f.write(f"ALTER TABLE {table} ADD PRIMARY KEY ({', '.join(columns)});\n")
    
    # Index
    cursor.execute("""
        SELECT indexname, tablename, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
            AND indexname NOT LIKE '%_pkey'
    """)
    
    for idx_name, table, idx_def in cursor.fetchall():
        f.write(f"{idx_def};\n")
    
    # ==========================================
    # 4. VUES
    # ==========================================
    
    print("\n👁️  Export vues...")
    
    cursor.execute("""
        SELECT table_name, view_definition
        FROM information_schema.views
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    
    vues = cursor.fetchall()
    
    if vues:
        print(f"   {len(vues)} vue(s) trouvée(s)")
        f.write("\n-- ==========================================\n")
        f.write("-- VUES\n")
        f.write("-- ==========================================\n\n")
        
        for view_name, view_def in vues:
            print(f"   - {view_name}")
            f.write(f"-- Vue: {view_name}\n")
            f.write(f"DROP VIEW IF EXISTS {view_name} CASCADE;\n")
            f.write(f"CREATE OR REPLACE VIEW {view_name} AS\n")
            # La définition peut se terminer par ";"
            view_def_clean = view_def.strip().rstrip(';')
            f.write(f"{view_def_clean};\n\n")
    else:
        print("   ⚠️  Aucune vue trouvée")
    
    f.write("\n-- Export terminé\n")

conn.close()

print(f"\n✅ Export terminé : {sql_file}")
print(f"   Taille: {sql_file.stat().st_size / 1024 / 1024:.1f} MB")
print("\n📋 Fichiers générés:")
print(f"   - {sql_file}")
print("\n🚀 Prochaine étape: docker-compose up --build")
