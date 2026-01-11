"""
DUMP SUPABASE → DOCKER POSTGRESQL
Projet NLP Text Mining - Master SISE

Exporte toutes les données de Supabase et les importe dans PostgreSQL Docker local
pour permettre le travail offline.

Usage:
    python sync_supabase_to_docker.py
"""

import psycopg2
import os
from dotenv import load_dotenv
import sys

# ============================================
# CONFIGURATION
# ============================================

# Charger Supabase credentials
load_dotenv()

SUPABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-1-eu-north-1.pooler.supabase.com'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

DOCKER_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'entrepot_nlp',
    'user': 'nlp_user',
    'password': 'nlp_password_2026'
}

# ============================================
# TABLES À MIGRER
# ============================================

DIMENSIONS = [
    'dim_source',
    'dim_localisation', 
    'dim_entreprise',
    'dim_contrat',
    'dim_temps'
]

FACTS = [
    'fact_offres',
    'fact_nlp_analysis',
    'fact_competences',
    'fact_preprocessing',
    'fact_profils_nlp',
    'fact_offres_external',
    'offres_embeddings'
]

ALL_TABLES = DIMENSIONS + FACTS

# ============================================
# FONCTIONS
# ============================================

def test_connection(config, name):
    """Test connexion PostgreSQL"""
    try:
        conn = psycopg2.connect(**config)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        conn.close()
        print(f"✅ {name}: Connecté")
        print(f"   {version[:50]}...")
        return True
    except Exception as e:
        print(f"❌ {name}: Erreur connexion")
        print(f"   {e}")
        return False

def get_table_count(conn, table_name):
    """Compte les lignes d'une table"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        return count
    except:
        return 0

def copy_table_data(source_conn, dest_conn, table_name):
    """Copie toutes les données d'une table"""
    print(f"\n📋 Copie {table_name}...")
    
    source_cursor = source_conn.cursor()
    dest_cursor = dest_conn.cursor()
    
    try:
        # Compter lignes source
        source_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = source_cursor.fetchone()[0]
        
        if total_rows == 0:
            print("   ⚠️  Table vide, skip")
            return
        
        print(f"   📊 {total_rows:,} lignes à copier")
        
        # Récupérer colonnes
        source_cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """)
        columns = [row[0] for row in source_cursor.fetchall()]
        columns_str = ', '.join(columns)
        
        # Vider table destination
        dest_cursor.execute(f"TRUNCATE TABLE {table_name} CASCADE")
        
        # Copier par batch de 1000
        batch_size = 1000
        offset = 0
        
        while offset < total_rows:
            # Lire batch
            source_cursor.execute(f"""
                SELECT {columns_str} 
                FROM {table_name} 
                ORDER BY 1
                LIMIT {batch_size} OFFSET {offset}
            """)
            rows = source_cursor.fetchall()
            
            if not rows:
                break
            
            # Insérer batch
            placeholders = ', '.join(['%s'] * len(columns))
            insert_query = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
            """
            dest_cursor.executemany(insert_query, rows)
            dest_conn.commit()
            
            offset += batch_size
            progress = min(offset, total_rows)
            print(f"   ⏳ {progress:,}/{total_rows:,} lignes ({progress*100//total_rows}%)", end='\r')
        
        print(f"   ✅ {total_rows:,} lignes copiées" + " "*20)
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        dest_conn.rollback()
        raise

def create_views(conn):
    """Crée les vues nécessaires"""
    print("\n📈 Création des vues...")
    
    cursor = conn.cursor()
    
    try:
        # D'abord supprimer la vue existante si elle existe
        cursor.execute("DROP VIEW IF EXISTS v_offres_nlp_complete CASCADE")
        conn.commit()
        
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
            n.status,
            n.profil_assigned,
            n.score_classification,
            n.competences_found,
            n.topic_id,
            n.topic_label,
            n.cluster_id,
            n.cluster_label,
            n.num_tokens,
            n.num_competences,
            n.description_clean,
            n.text_for_sklearn,
            n.tokens,
            n.duration,
            n.salary_text,
            n.profil_score,
            n.profil_confidence,
            n.profil_second,
            n.profil_second_score,
            n.score_title,
            n.score_description,
            n.score_competences,
            n.cascade_pass
        FROM fact_offres o
        LEFT JOIN dim_source s ON o.source_id = s.source_id
        LEFT JOIN dim_localisation l ON o.localisation_id = l.localisation_id
        LEFT JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
        LEFT JOIN dim_contrat c ON o.contrat_id = c.contrat_id
        LEFT JOIN dim_temps t ON o.temps_id = t.temps_id
        LEFT JOIN fact_nlp_analysis n ON o.offre_id = n.offre_id
        """)
        
        conn.commit()
        print("   ✅ Vue v_offres_nlp_complete créée")
        
    except Exception as e:
        conn.rollback()  # Rollback pour récupérer de l'erreur
        print(f"   ❌ Erreur création vues: {e}")

# ============================================
# MAIN
# ============================================

def main():
    print("="*80)
    print("  SYNCHRONISATION SUPABASE → DOCKER POSTGRESQL")
    print("="*80)
    
    # Vérifier credentials Supabase
    if not SUPABASE_CONFIG['password']:
        print("\n❌ ERREUR: DB_PASSWORD manquant dans .env")
        print("\nAjoutez votre mot de passe Supabase dans le fichier .env :")
        print("DB_PASSWORD=votre_mot_de_passe")
        sys.exit(1)
    
    # Test connexions
    print("\n🔌 Test des connexions...")
    print("-"*80)
    
    if not test_connection(SUPABASE_CONFIG, "Supabase"):
        sys.exit(1)
    
    if not test_connection(DOCKER_CONFIG, "Docker PostgreSQL"):
        print("\n💡 Assurez-vous que Docker est démarré:")
        print("   docker-compose up -d postgres")
        sys.exit(1)
    
    # Connexions
    print("\n🔗 Connexion aux bases...")
    source_conn = psycopg2.connect(**SUPABASE_CONFIG)
    dest_conn = psycopg2.connect(**DOCKER_CONFIG)
    
    print("✅ Connexions établies")
    
    # Copier tables dimensions d'abord
    print("\n" + "="*80)
    print("📦 COPIE DES DIMENSIONS")
    print("="*80)
    
    for table in DIMENSIONS:
        copy_table_data(source_conn, dest_conn, table)
    
    # Puis tables de faits
    print("\n" + "="*80)
    print("📊 COPIE DES FAITS")
    print("="*80)
    
    for table in FACTS:
        copy_table_data(source_conn, dest_conn, table)
    
    # Créer vues
    print("\n" + "="*80)
    print("🔧 CONFIGURATION")
    print("="*80)
    create_views(dest_conn)
    
    # Vérification finale
    print("\n" + "="*80)
    print("✅ VÉRIFICATION")
    print("="*80)
    
    dest_cursor = dest_conn.cursor()
    dest_cursor.execute("SELECT COUNT(*) FROM v_offres_nlp_complete")
    total = dest_cursor.fetchone()[0]
    
    print(f"\n📊 Total offres dans Docker: {total:,}")
    
    if total > 0:
        dest_cursor.execute("""
            SELECT status, COUNT(*) 
            FROM fact_nlp_analysis 
            GROUP BY status
        """)
        for status, count in dest_cursor.fetchall():
            print(f"   {status}: {count:,}")
    
    # Fermer connexions
    source_conn.close()
    dest_conn.close()
    
    print("\n" + "="*80)
    print("✅ SYNCHRONISATION TERMINÉE")
    print("="*80)
    print("\n💡 Vous pouvez maintenant utiliser Docker en mode offline")
    print("   Modifiez .env pour pointer sur localhost ou Supabase selon besoin")

if __name__ == '__main__':
    main()
