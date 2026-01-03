"""
Nettoyage Entrepôt DuckDB - VERSION CASCADE
Supprime offres hors scope Data/AI + leurs compétences

Stratégie CASCADE (respect Foreign Keys):
1. Backup fact_offres + fact_competences
2. Analyse offres à supprimer
3. DELETE fact_competences (compétences liées)
4. DELETE fact_offres (offres)
5. Vérification résultats

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime


def connect_db():
    """Connexion à l'entrepôt DuckDB"""
    db_path = Path('../entrepot_de_donnees/entrepot_nlp.duckdb')
    
    if not db_path.exists():
        raise FileNotFoundError(f"Entrepôt non trouvé: {db_path}")
    
    print(f"📁 Connexion à: {db_path}")
    return duckdb.connect(str(db_path))


def get_current_stats(conn):
    """Statistiques actuelles"""
    print("\n" + "="*70)
    print("📊 STATISTIQUES ACTUELLES")
    print("="*70)
    
    # Total offres
    n_offres = conn.execute("SELECT COUNT(*) FROM fact_offres").fetchone()[0]
    print(f"\nTotal offres dans entrepôt: {n_offres:,}")
    
    # Total compétences
    n_competences = conn.execute("SELECT COUNT(*) FROM fact_competences").fetchone()[0]
    print(f"Total compétences:          {n_competences:,}")
    
    # Par source
    print("\nOffres par source:")
    df_sources = conn.execute("""
        SELECT 
            s.source_name,
            COUNT(*) as count
        FROM fact_offres o
        JOIN dim_source s ON o.source_id = s.source_id
        GROUP BY s.source_name
        ORDER BY count DESC
    """).df()
    
    for _, row in df_sources.iterrows():
        print(f"   {row['source_name']:<20s}: {row['count']:,}")
    
    return n_offres, n_competences


def analyze_to_delete(conn):
    """Analyse offres à supprimer"""
    print("\n" + "="*70)
    print("🔍 ANALYSE OFFRES À SUPPRIMER")
    print("="*70)
    
    # Conditions d'exclusion
    exclusion_conditions = """
    WHERE 
        -- 1. Comptabilité (pas Data/AI)
        (LOWER(title) LIKE '%comptable%' 
         OR LOWER(title) LIKE '%comptabilité%'
         OR LOWER(title) LIKE '%comptabilite%'
         OR LOWER(title) LIKE '%gestionnaire comptable%'
         OR LOWER(title) LIKE '%assistant comptable%')
        
        -- 2. Logiciel embarqué (pas Data)
        OR (LOWER(title) LIKE '%logiciel embarqué%'
            OR LOWER(title) LIKE '%logiciel embarque%'
            OR LOWER(title) LIKE '%embarqué%'
            OR LOWER(title) LIKE '%embarque%'
            OR LOWER(title) LIKE '%embedded%')
        
        -- 3. Développeur/Ingénieur SANS keyword Data
        OR ((LOWER(title) LIKE '%développeur%' OR LOWER(title) LIKE '%developpeur%'
             OR LOWER(title) LIKE '%ingénieur%' OR LOWER(title) LIKE '%ingenieur%')
            AND LOWER(title) NOT LIKE '%data%'
            AND LOWER(title) NOT LIKE '%données%'
            AND LOWER(title) NOT LIKE '%donnees%'
            AND LOWER(title) NOT LIKE '%big data%'
            AND LOWER(title) NOT LIKE '%bi%'
            AND LOWER(title) NOT LIKE '%business intelligence%'
            AND LOWER(title) NOT LIKE '%machine learning%'
            AND LOWER(title) NOT LIKE '%ml%'
            AND LOWER(title) NOT LIKE '%statisticien%'
            AND LOWER(title) NOT LIKE '%scientist%')
        
        -- 4. RH / Recrutement
        OR LOWER(title) LIKE '%recruitment%'
        OR LOWER(title) LIKE '%recrutement%'
        OR LOWER(title) LIKE '%business partner rh%'
        
        -- 5. Autres métiers hors Data
        OR LOWER(title) LIKE '%facteur%'
        OR LOWER(title) LIKE '%technicien data center%'
        OR LOWER(title) LIKE '%gestionnaire%'
        
        -- 6. Manufacturing/Energy sans Data
        OR ((LOWER(title) LIKE '%manufacturing%'
             OR LOWER(title) LIKE '%utilities%'
             OR LOWER(title) LIKE '%energy%')
            AND LOWER(title) NOT LIKE '%data%')
        
        -- 7. Formations génériques
        OR LOWER(title) LIKE '%formation%concepteur%'
        OR LOWER(title) LIKE '%copy of%'
    """
    
    # Compter offres à supprimer
    query_count = f"""
        SELECT COUNT(*) as count
        FROM fact_offres
        {exclusion_conditions}
    """
    
    n_offres_to_delete = conn.execute(query_count).fetchone()[0]
    
    print(f"\n📉 Offres à supprimer: {n_offres_to_delete:,}")
    
    # Compter compétences associées
    query_competences = f"""
        SELECT COUNT(*) as count
        FROM fact_competences
        WHERE offre_id IN (
            SELECT offre_id FROM fact_offres
            {exclusion_conditions}
        )
    """
    
    n_competences_to_delete = conn.execute(query_competences).fetchone()[0]
    
    print(f"📉 Compétences associées: {n_competences_to_delete:,}")
    
    # Top 50 titres à supprimer
    query_titles = f"""
        SELECT 
            title,
            COUNT(*) as count
        FROM fact_offres
        {exclusion_conditions}
        GROUP BY title
        ORDER BY count DESC
        LIMIT 50
    """
    
    df_titles = conn.execute(query_titles).df()
    
    print(f"\n🔍 Top 50 titres à supprimer (représentent {df_titles['count'].sum():,} offres):\n")
    
    for i, row in df_titles.iterrows():
        print(f"{i+1:2d}. [{row['count']:3d}x] {row['title']}")
    
    # Sauvegarder liste complète
    output_path = Path('../resultats_nlp/offres_a_supprimer.csv')
    
    query_all = f"""
        SELECT 
            o.offre_id,
            o.title,
            o.description,
            s.source_name
        FROM fact_offres o
        JOIN dim_source s ON o.source_id = s.source_id
        {exclusion_conditions}
        ORDER BY o.title
    """
    
    df_all = conn.execute(query_all).df()
    df_all.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 Liste complète sauvegardée: {output_path}")
    
    return n_offres_to_delete, n_competences_to_delete, exclusion_conditions


def backup_tables(conn):
    """Créer backups avant suppression"""
    print("\n" + "="*70)
    print("📦 CRÉATION BACKUPS")
    print("="*70)
    
    # Backup fact_offres
    try:
        conn.execute("DROP TABLE IF EXISTS fact_offres_backup")
        print("\n   ℹ️  Ancien backup fact_offres supprimé")
    except:
        pass
    
    print("   ⏳ Backup fact_offres...")
    conn.execute("CREATE TABLE fact_offres_backup AS SELECT * FROM fact_offres")
    n_offres = conn.execute("SELECT COUNT(*) FROM fact_offres_backup").fetchone()[0]
    print(f"   ✅ Backup fact_offres créé: {n_offres:,} offres")
    
    # Backup fact_competences
    try:
        conn.execute("DROP TABLE IF EXISTS fact_competences_backup")
        print("\n   ℹ️  Ancien backup fact_competences supprimé")
    except:
        pass
    
    print("   ⏳ Backup fact_competences...")
    conn.execute("CREATE TABLE fact_competences_backup AS SELECT * FROM fact_competences")
    n_comp = conn.execute("SELECT COUNT(*) FROM fact_competences_backup").fetchone()[0]
    print(f"   ✅ Backup fact_competences créé: {n_comp:,} compétences")
    
    print(f"\n   💡 Pour restaurer:")
    print(f"      DROP TABLE fact_offres;")
    print(f"      CREATE TABLE fact_offres AS SELECT * FROM fact_offres_backup;")
    print(f"      DROP TABLE fact_competences;")
    print(f"      CREATE TABLE fact_competences AS SELECT * FROM fact_competences_backup;")


def delete_cascade(conn, exclusion_conditions):
    """Supprimer en cascade: compétences PUIS offres"""
    print("\n" + "="*70)
    print("🗑️  SUPPRESSION CASCADE (COMPÉTENCES → OFFRES)")
    print("="*70)
    
    # ÉTAPE 1: Supprimer compétences
    print("\n   1️⃣ Suppression compétences associées...")
    
    delete_competences = f"""
        DELETE FROM fact_competences
        WHERE offre_id IN (
            SELECT offre_id FROM fact_offres
            {exclusion_conditions}
        )
    """
    
    conn.execute(delete_competences)
    print("   ✅ Compétences supprimées")
    
    # ÉTAPE 2: Supprimer offres
    print("\n   2️⃣ Suppression offres...")
    
    delete_offres = f"""
        DELETE FROM fact_offres
        {exclusion_conditions}
    """
    
    conn.execute(delete_offres)
    print("   ✅ Offres supprimées")


def verify_results(conn, n_offres_before, n_comp_before, n_offres_to_delete, n_comp_to_delete):
    """Vérifier résultats du nettoyage"""
    print("\n" + "="*70)
    print("✅ VÉRIFICATION RÉSULTATS")
    print("="*70)
    
    # Stats après
    n_offres_after = conn.execute("SELECT COUNT(*) FROM fact_offres").fetchone()[0]
    n_comp_after = conn.execute("SELECT COUNT(*) FROM fact_competences").fetchone()[0]
    
    n_offres_deleted = n_offres_before - n_offres_after
    n_comp_deleted = n_comp_before - n_comp_after
    
    print(f"\n📊 OFFRES:")
    print(f"   AVANT:      {n_offres_before:,}")
    print(f"   APRÈS:      {n_offres_after:,}")
    print(f"   Supprimées: {n_offres_deleted:,} ({n_offres_deleted/n_offres_before*100:.1f}%)")
    
    print(f"\n📊 COMPÉTENCES:")
    print(f"   AVANT:      {n_comp_before:,}")
    print(f"   APRÈS:      {n_comp_after:,}")
    print(f"   Supprimées: {n_comp_deleted:,} ({n_comp_deleted/n_comp_before*100:.1f}%)")
    
    # Vérifier cohérence
    if n_offres_deleted == n_offres_to_delete:
        print(f"\n✅ Cohérence offres OK")
    else:
        print(f"\n⚠️  Différence offres: {abs(n_offres_deleted - n_offres_to_delete)}")
    
    if n_comp_deleted == n_comp_to_delete:
        print(f"✅ Cohérence compétences OK")
    else:
        print(f"⚠️  Différence compétences: {abs(n_comp_deleted - n_comp_to_delete)}")
    
    # Vérifier intégrité FK
    print(f"\n🔍 Vérification intégrité FK...")
    
    orphan_check = conn.execute("""
        SELECT COUNT(*) FROM fact_competences c
        WHERE NOT EXISTS (
            SELECT 1 FROM fact_offres o
            WHERE o.offre_id = c.offre_id
        )
    """).fetchone()[0]
    
    if orphan_check == 0:
        print(f"   ✅ Pas de compétences orphelines")
    else:
        print(f"   ⚠️  {orphan_check} compétences orphelines trouvées !")
    
    # Distribution par source
    print("\n📊 Distribution après nettoyage:")
    df_sources = conn.execute("""
        SELECT 
            s.source_name,
            COUNT(*) as count
        FROM fact_offres o
        JOIN dim_source s ON o.source_id = s.source_id
        GROUP BY s.source_name
        ORDER BY count DESC
    """).df()
    
    for _, row in df_sources.iterrows():
        pct = row['count'] / n_offres_after * 100
        print(f"   {row['source_name']:<20s}: {row['count']:,} ({pct:.1f}%)")
    
    # Top titres restants
    print("\n📋 Top 20 titres restants:")
    df_top = conn.execute("""
        SELECT 
            title,
            COUNT(*) as count
        FROM fact_offres
        GROUP BY title
        ORDER BY count DESC
        LIMIT 20
    """).df()
    
    for i, row in df_top.iterrows():
        print(f"{i+1:2d}. [{row['count']:3d}x] {row['title']}")


def save_log(n_offres_before, n_offres_after, n_offres_deleted, 
             n_comp_before, n_comp_after, n_comp_deleted):
    """Sauvegarder log du nettoyage"""
    log_path = Path('../resultats_nlp/nettoyage_log.txt')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    log_content = f"""
NETTOYAGE ENTREPÔT DUCKDB - CASCADE
Date: {timestamp}

RÉSULTATS:
---------

OFFRES:
  AVANT:      {n_offres_before:,}
  APRÈS:      {n_offres_after:,}
  Supprimées: {n_offres_deleted:,} ({n_offres_deleted/n_offres_before*100:.1f}%)

COMPÉTENCES:
  AVANT:      {n_comp_before:,}
  APRÈS:      {n_comp_after:,}
  Supprimées: {n_comp_deleted:,} ({n_comp_deleted/n_comp_before*100:.1f}%)

BACKUPS:
--------
Tables:
  - fact_offres_backup ({n_offres_before:,} offres)
  - fact_competences_backup ({n_comp_before:,} compétences)

Pour restaurer:
DROP TABLE fact_offres;
CREATE TABLE fact_offres AS SELECT * FROM fact_offres_backup;
DROP TABLE fact_competences;
CREATE TABLE fact_competences AS SELECT * FROM fact_competences_backup;
"""
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    print(f"\n💾 Log sauvegardé: {log_path}")


def main():
    """Pipeline complet de nettoyage CASCADE"""
    
    print("="*70)
    print("🧹 NETTOYAGE ENTREPÔT - OFFRES DATA/AI + CASCADE FK")
    print("="*70)
    
    # Connexion
    conn = connect_db()
    
    # Stats actuelles
    n_offres_before, n_comp_before = get_current_stats(conn)
    
    # Analyser offres à supprimer
    n_offres_to_delete, n_comp_to_delete, exclusion_conditions = analyze_to_delete(conn)
    
    # Demander confirmation
    print("\n" + "="*70)
    print("⚠️  CONFIRMATION REQUISE")
    print("="*70)
    
    print(f"\nVous allez supprimer:")
    print(f"   • {n_offres_to_delete:,} offres")
    print(f"   • {n_comp_to_delete:,} compétences associées")
    
    print(f"\nDes backups seront créés:")
    print(f"   • fact_offres_backup")
    print(f"   • fact_competences_backup")
    
    print(f"\n⚠️  Cette action est IRRÉVERSIBLE (sauf via backup).")
    
    confirm = input("\n👉 Confirmer la suppression CASCADE ? (tapez 'OUI' en majuscules): ")
    
    if confirm != 'OUI':
        print("\n❌ Nettoyage annulé")
        conn.close()
        return
    
    # Créer backups
    backup_tables(conn)
    
    # Supprimer en cascade
    delete_cascade(conn, exclusion_conditions)
    
    # Stats après
    n_offres_after = conn.execute("SELECT COUNT(*) FROM fact_offres").fetchone()[0]
    n_comp_after = conn.execute("SELECT COUNT(*) FROM fact_competences").fetchone()[0]
    
    n_offres_deleted = n_offres_before - n_offres_after
    n_comp_deleted = n_comp_before - n_comp_after
    
    # Vérifier
    verify_results(conn, n_offres_before, n_comp_before, 
                  n_offres_to_delete, n_comp_to_delete)
    
    # Sauvegarder log
    save_log(n_offres_before, n_offres_after, n_offres_deleted,
            n_comp_before, n_comp_after, n_comp_deleted)
    
    # Fermer connexion
    conn.close()
    
    print("\n" + "="*70)
    print("✅ NETTOYAGE CASCADE TERMINÉ !")
    print("="*70)
    
    print(f"\n📊 RÉSUMÉ:")
    print(f"   Offres supprimées:       {n_offres_deleted:,} ({n_offres_deleted/n_offres_before*100:.1f}%)")
    print(f"   Compétences supprimées:  {n_comp_deleted:,} ({n_comp_deleted/n_comp_before*100:.1f}%)")
    print(f"   Offres restantes:        {n_offres_after:,}")
    
    print(f"\n📁 Fichiers créés:")
    print(f"   - offres_a_supprimer.csv")
    print(f"   - nettoyage_log.txt")
    
    print(f"\n🚀 PROCHAINES ÉTAPES:")
    print(f"   1. cd analyses_nlp/fichiers_analyses")
    print(f"   2. python 1_extraction_competences.py")
    print(f"   3. python 2_topic_modeling.py")
    print(f"   4. python 3_text_preprocessing.py")
    print(f"   5. python 4_classification_hybride.py")
    
    print(f"\n💡 Entrepôt propre (offres Data/AI + compétences cohérentes) !")


if __name__ == "__main__":
    main()