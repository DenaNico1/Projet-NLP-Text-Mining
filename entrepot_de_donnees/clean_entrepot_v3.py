"""
Nettoyage Entrepôt v3 - SUPPRESSION BRUIT RÉSIDUEL
Supprime profils non-Data identifiés dans analyse des non classifiés

NOUVEAUTÉS v3:
- Supprime Architecte SANS data (Microsoft 365, Pega, etc.)
- Supprime Support/Technicien logiciel générique
- Supprime Référent Paie/RH
- Supprime métiers spécifiques (dessinateur, planner, etc.)
- Supprime Software Engineer trop générique

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import duckdb
import pandas as pd
from pathlib import Path

def clean_entrepot_v3():
    """
    Supprime bruit résiduel identifié dans analyse non classifiés
    """
    print("="*70)
    print("🧹 NETTOYAGE ENTREPÔT v3 - BRUIT RÉSIDUEL")
    print("="*70)
    
    # ✅ Chemin DuckDB correct
    db_path = Path('../entrepot_de_donnees/entrepot_nlp.duckdb')
    
    if not db_path.exists():
        print(f"❌ Base non trouvée: {db_path}")
        return
    
    # ✅ Connexion DuckDB
    conn = duckdb.connect(str(db_path))
    
    # ==========================================
    # 1. ÉTAT AVANT
    # ==========================================
    print("\n📊 État AVANT nettoyage v3:")
    
    count_before = conn.execute("SELECT COUNT(*) as n FROM fact_offres").fetchdf().iloc[0]['n']
    print(f"   Total offres: {count_before}")
    
    # ==========================================
    # 2. BACKUP v3
    # ==========================================
    print("\n💾 Création backup v3...")
    
    conn.execute("DROP TABLE IF EXISTS fact_offres_backup_v3")
    conn.execute("DROP TABLE IF EXISTS fact_competences_backup_v3")
    
    conn.execute("CREATE TABLE fact_offres_backup_v3 AS SELECT * FROM fact_offres")
    conn.execute("CREATE TABLE fact_competences_backup_v3 AS SELECT * FROM fact_competences")
    
    print("   ✅ Backup v3 créé")
    
    # ==========================================
    # 3. IDENTIFICATION BRUIT RÉSIDUEL
    # ==========================================
    print("\n🔍 Identification bruit résiduel...")
    
    query_bruit = """
    SELECT offre_id, title 
    FROM fact_offres
    WHERE 
        -- ========================================
        -- GROUPE 1: Architecte NON Data
        -- ========================================
        
        -- Microsoft 365 / Office 365
        (LOWER(title) LIKE '%microsoft 365%'
         OR LOWER(title) LIKE '%office 365%'
         OR LOWER(title) LIKE '%m365%'
         OR LOWER(title) LIKE '%o365%')
        
        -- Pega
        OR LOWER(title) LIKE '%pega%'
        
        -- Architecte SI/Solution/Technique SANS data/ia
        OR (LOWER(title) LIKE '%architecte%'
            AND (LOWER(title) LIKE '%solution%' OR LOWER(title) LIKE '%si%' OR LOWER(title) LIKE '%technique%')
            AND LOWER(title) NOT LIKE '%data%'
            AND LOWER(title) NOT LIKE '%donnees%'
            AND LOWER(title) NOT LIKE '%donnée%'
            AND LOWER(title) NOT LIKE '%ia%'
            AND LOWER(title) NOT LIKE '%ai%'
            AND LOWER(title) NOT LIKE '%intelligence artificielle%')
        
        -- ========================================
        -- GROUPE 2: Support/Technicien logiciel
        -- ========================================
        
        -- Support applicatif/logiciel
        OR LOWER(title) LIKE '%support%applicatif%'
        OR LOWER(title) LIKE '%support%logiciel%'
        OR LOWER(title) LIKE '%technicien%support%'
        
        -- Technicien ERP/SAP
        OR LOWER(title) LIKE '%technicien%erp%'
        OR LOWER(title) LIKE '%technicien%sap%'
        OR (LOWER(title) LIKE '%stagiaire%' AND LOWER(title) LIKE '%sap%')
        
        -- Paramétreur logiciel
        OR LOWER(title) LIKE '%parametreur%logiciel%'
        OR LOWER(title) LIKE '%parametreuse%logiciel%'
        
        -- Intégration/validation logiciel
        OR LOWER(title) LIKE '%integration%logiciel%'
        OR LOWER(title) LIKE '%validation%logiciel%'
        OR LOWER(title) LIKE '%test%logiciel%'
        
        -- ========================================
        -- GROUPE 3: Référent Paie/RH
        -- ========================================
        
        OR LOWER(title) LIKE '%referent%paie%'
        OR LOWER(title) LIKE '%référent%paie%'
        OR LOWER(title) LIKE '%charge%paie%'
        OR LOWER(title) LIKE '%chargé%paie%'
        
        -- ========================================
        -- GROUPE 4: Métiers spécifiques
        -- ========================================
        
        -- Dessinateur/CAO
        OR LOWER(title) LIKE '%dessinateur%'
        OR LOWER(title) LIKE '%dessinatrice%'
        OR LOWER(title) LIKE '%caneco%'
        
        -- Planner
        OR (LOWER(title) LIKE '%planner%'
            AND LOWER(title) NOT LIKE '%data%')
        
        -- Alpiniste (!?)
        OR LOWER(title) LIKE '%alpiniste%'
        
        -- EIA/Utilities
        OR LOWER(title) LIKE '%eia%utilites%'
        OR LOWER(title) LIKE '%eia%utilities%'
        
        -- Bio-informatique (sauf si data)
        OR (LOWER(title) LIKE '%bio-informatique%'
            AND LOWER(title) NOT LIKE '%data%')
        
        -- Protection données personnelles (DPO)
        OR LOWER(title) LIKE '%protection%donnees%personnelles%'
        OR LOWER(title) LIKE '%dpo%'
        
        -- Chef/Cheffe produits numériques (trop vague)
        OR (LOWER(title) LIKE '%chef%produit%'
            AND LOWER(title) LIKE '%numerique%'
            AND LOWER(title) NOT LIKE '%data%')
        
        -- ========================================
        -- GROUPE 5: Titres trop génériques
        -- ========================================
        
        -- Software Engineer seul (sans data/ia)
        OR (title = 'Software Engineer'
            OR title = 'Software Developer'
            OR title = 'Développeur Logiciel'
            OR title = 'Ingénieur Logiciel')
        
        -- Intervenant scolaire
        OR LOWER(title) LIKE '%intervenant%scolaire%'
        
        -- A-Player Interns (trop vague)
        OR LOWER(title) LIKE '%a-player%intern%'
    """
    
    df_bruit = conn.execute(query_bruit).fetchdf()
    
    print(f"\n📋 Offres bruit résiduel identifiées: {len(df_bruit)}")
    
    if len(df_bruit) > 0:
        print("\n🔍 Exemples de bruit résiduel:")
        for i, row in df_bruit.head(20).iterrows():
            print(f"   - {row['title']}")
    
    # ==========================================
    # 4. SUPPRESSION BRUIT
    # ==========================================
    if len(df_bruit) > 0:
        print(f"\n🗑️  Suppression {len(df_bruit)} offres bruit résiduel...")
        
        # Supprimer compétences associées
        conn.execute(f"""
            DELETE FROM fact_competences 
            WHERE offre_id IN ({','.join(map(str, df_bruit['offre_id']))})
        """)
        
        # Supprimer offres
        conn.execute(f"""
            DELETE FROM fact_offres 
            WHERE offre_id IN ({','.join(map(str, df_bruit['offre_id']))})
        """)
        
        print("   ✅ Bruit résiduel supprimé")
    else:
        print("\n✅ Aucun bruit résiduel détecté")
    
    # ==========================================
    # 5. ÉTAT APRÈS
    # ==========================================
    print("\n📊 État APRÈS nettoyage v3:")
    
    count_after = conn.execute("SELECT COUNT(*) as n FROM fact_offres").fetchdf().iloc[0]['n']
    count_comp = conn.execute("SELECT COUNT(*) as n FROM fact_competences").fetchdf().iloc[0]['n']
    
    print(f"   Total offres: {count_after}")
    print(f"   Total compétences: {count_comp}")
    print(f"   Offres supprimées: {count_before - count_after}")
    
    # ==========================================
    # 6. TOP TITRES RESTANTS
    # ==========================================
    print("\n🏆 Top 20 titres restants:")
    
    top_titles = conn.execute("""
        SELECT title, COUNT(*) as count 
        FROM fact_offres 
        GROUP BY title 
        ORDER BY count DESC 
        LIMIT 20
    """).fetchdf()
    
    for i, row in top_titles.iterrows():
        print(f"   {i+1:2d}. [{row['count']:3d}x] {row['title']}")
    
    conn.close()
    
    print("\n✅ NETTOYAGE v3 TERMINÉ !")
    print(f"\n📊 RÉSUMÉ:")
    print(f"   Base initiale: {count_before} offres")
    print(f"   Base finale:   {count_after} offres")
    print(f"   Supprimées:    {count_before - count_after} offres ({(count_before - count_after)/count_before*100:.1f}%)")
    
    print("\n💡 Prochaine étape:")
    print("   python 1_preprocessing.py")
    print("   (Relancer preprocessing avec base nettoyée v3)")


if __name__ == "__main__":
    clean_entrepot_v3()