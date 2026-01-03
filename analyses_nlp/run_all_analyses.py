"""
Script Maître - Lancement de Toutes les Analyses NLP
Exécute toutes les analyses dans l'ordre

Auteur: Projet NLP Text Mining
Date: Décembre 2025

Usage:
    python run_all_analyses.py
    python run_all_analyses.py --skip 1,2  # Sauter étapes 1 et 2
"""

import argparse
import sys
import subprocess
from pathlib import Path
import time

def run_analysis(script_name, step_num):
    """Exécute une analyse"""
    print(f"\n{'='*70}")
    print(f"🚀 LANCEMENT ÉTAPE {step_num} : {script_name}")
    print(f"{'='*70}\n")
    
    try:
        # Exécuter directement le fichier Python
        import subprocess
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ Étape {step_num} terminée avec succès !")
            return True
        else:
            print(f"\n❌ Étape {step_num} a échoué (code: {result.returncode})")
            return False
        
    except Exception as e:
        print(f"\n❌ ERREUR dans étape {step_num}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Pipeline complet"""
    parser = argparse.ArgumentParser(description='Lancement analyses NLP')
    parser.add_argument('--skip', type=str, default='', 
                       help='Étapes à sauter (ex: 1,2,5)')
    args = parser.parse_args()
    
    # Étapes à sauter
    skip_steps = set()
    if args.skip:
        skip_steps = set(int(x.strip()) for x in args.skip.split(','))
    
    # Liste des analyses
    analyses = [
        (1, 'fichiers_analyses/1_preprocessing.py'),
        (2, 'fichiers_analyses/2_extraction_competences.py'),
        (3, 'fichiers_analyses/3_topic_modeling.py'),
        (4, 'fichiers_analyses/4_geo_semantique.py'),
        (5, 'fichiers_analyses/5_evolution_temporelle.py'),
        (6, 'fichiers_analyses/6_embeddings_clustering.py'),
        (7, 'fichiers_analyses/7_stacks_salaires.py')
    ]
    
    print("="*70)
    print("🎯 PIPELINE COMPLET D'ANALYSES NLP")
    print("="*70)
    print(f"\n📋 {len(analyses)} analyses à exécuter")
    
    if skip_steps:
        print(f"⏭️  Étapes à sauter: {sorted(skip_steps)}")
    
    print(f"\n⏱️  Temps estimé: 15-30 minutes")
    print(f"\nAppuyez sur Ctrl+C pour annuler...")
    
    time.sleep(3)
    
    # Exécution
    start_time = time.time()
    results = {}
    
    for step_num, script_name in analyses:
        if step_num in skip_steps:
            print(f"\n⏭️  Étape {step_num} sautée ({script_name})")
            continue
        
        success = run_analysis(script_name, step_num)
        results[step_num] = success
        
        if not success:
            print(f"\n⚠️  Voulez-vous continuer malgré l'erreur ? (o/n)")
            response = input().strip().lower()
            if response != 'o':
                break
    
    # Résumé
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DES ANALYSES")
    print("="*70)
    
    for step_num, success in results.items():
        status = "✅ Succès" if success else "❌ Échec"
        print(f"   Étape {step_num}: {status}")
    
    successful = sum(1 for s in results.values() if s)
    total = len(results)
    
    print(f"\n🎯 {successful}/{total} analyses réussies")
    print(f"⏱️  Temps total: {elapsed/60:.1f} minutes")
    
    if successful == total:
        print("\n🎉 TOUTES LES ANALYSES TERMINÉES AVEC SUCCÈS !")
        print(f"\n📁 Résultats disponibles dans: ../resultats_nlp/")
    else:
        print(f"\n⚠️  Certaines analyses ont échoué")


if __name__ == "__main__":
    # Ajouter le chemin
    sys.path.insert(0, str(Path(__file__).parent))
    
    main()