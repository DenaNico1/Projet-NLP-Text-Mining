"""
SCRIPT 2 : MIGRATION EMBEDDINGS.NPY → POSTGRESQL
Exécuter : python 02_migrate_embeddings_to_postgres.py
"""

import numpy as np
import psycopg2
from pathlib import Path
import sys
from tqdm import tqdm

# Ajouter path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'app_streamlit'))
from config_db import get_db_connection

print("=" * 60)
print("MIGRATION EMBEDDINGS → POSTGRESQL")
print("=" * 60)

# ============================================
# 1. CHARGER EMBEDDINGS LOCAUX
# ============================================

print("\n📦 Chargement embeddings.npy...")
embeddings_path = Path(__file__).parent.parent / 'resultats_nlp' / 'models' / 'embeddings.npy'

if not embeddings_path.exists():
    print(f"❌ Fichier non trouvé: {embeddings_path}")
    sys.exit(1)

embeddings = np.load(embeddings_path)
print(f"✅ {len(embeddings)} embeddings chargés")
print(f"   Shape: {embeddings.shape}")
print(f"   Dtype: {embeddings.dtype}")

# ============================================
# 2. CONNEXION POSTGRESQL
# ============================================

print("\n🔌 Connexion PostgreSQL...")
try:
    conn = get_db_connection()
    cur = conn.cursor()
    print("✅ Connexion établie")
except Exception as e:
    print(f"❌ Erreur connexion: {e}")
    sys.exit(1)

# ============================================
# 3. RÉCUPÉRER IDS OFFRES (ORDRE)
# ============================================

print("\n🔍 Récupération IDs offres...")
cur.execute("""
    SELECT offre_id 
    FROM fact_offres 
    ORDER BY offre_id 
    LIMIT %s
""", (len(embeddings),))

offre_ids = [row[0] for row in cur.fetchall()]
print(f"✅ {len(offre_ids)} IDs récupérés")

if len(offre_ids) != len(embeddings):
    print(f"⚠️ Warning: {len(embeddings)} embeddings mais {len(offre_ids)} offres")
    print("   → Utilisation des premiers IDs disponibles")

# ============================================
# 4. MIGRATION (BATCH)
# ============================================

print(f"\n🚀 Migration vers PostgreSQL...")
print(f"   Batch size: 100 offres")

inserted = 0
errors = 0

for i in tqdm(range(0, len(offre_ids), 100), desc="Migration"):
    batch_ids = offre_ids[i:i+100]
    batch_embeddings = embeddings[i:i+100]
    
    try:
        # Préparer données batch
        values = []
        for offre_id, embedding in zip(batch_ids, batch_embeddings):
            # Convertir numpy array → list Python
            embedding_list = embedding.tolist()
            values.append((offre_id, embedding_list))
        
        # Insert batch avec ON CONFLICT (upsert)
        cur.executemany("""
            INSERT INTO offres_embeddings (offre_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (offre_id) 
            DO UPDATE SET 
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
        """, values)
        
        conn.commit()
        inserted += len(batch_ids)
        
    except Exception as e:
        print(f"\n❌ Erreur batch {i}: {e}")
        conn.rollback()
        errors += 1
        
        if errors > 5:
            print("❌ Trop d'erreurs, arrêt migration")
            break

print(f"\n✅ Migration terminée !")
print(f"   - Insérés: {inserted}")
print(f"   - Erreurs: {errors}")

# ============================================
# 5. VÉRIFICATION
# ============================================

print("\n🔍 Vérification...")

cur.execute("""
    SELECT 
        COUNT(*) as total,
        MIN(array_length(embedding, 1)) as min_dim,
        MAX(array_length(embedding, 1)) as max_dim,
        AVG(array_length(embedding, 1)) as avg_dim
    FROM offres_embeddings
""")

result = cur.fetchone()
print(f"   - Total embeddings: {result[0]}")
print(f"   - Dimensions: min={result[1]}, max={result[2]}, avg={result[3]:.0f}")

# Exemple embedding
cur.execute("SELECT offre_id, embedding[1:5] FROM offres_embeddings LIMIT 1")
sample = cur.fetchone()
print(f"   - Exemple offre {sample[0]}: {sample[1][:5]}...")

# Jointure test
cur.execute("""
    SELECT COUNT(*) 
    FROM fact_offres o
    INNER JOIN offres_embeddings e ON o.offre_id = e.offre_id
""")
joined = cur.fetchone()[0]
print(f"   - Offres avec embeddings: {joined}")

conn.close()

print("\n" + "=" * 60)
print("🎉 MIGRATION RÉUSSIE !")
print("=" * 60)
print("\n📝 Prochaines étapes:")
print("   1. Modifier data_loaders.py")
print("   2. Modifier matching.py")
print("   3. Tester matching (<5 sec attendu)")