"""
Script Maitre - Lancement de Toutes les Analyses NLP
Execute sequentiellement toutes les analyses du projet

Auteur: Projet NLP Text Mining - Master SISE
Date: Decembre 2025
"""

import subprocess
import sys
from pathlib import Path
import time
import os

# Forcer UTF-8 sur Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("=" * 80)
print("LANCEMENT DE TOUTES LES ANALYSES NLP")
print("=" * 80)

# Liste des scripts a executer dans l'ordre
scripts = [
    'fichiers_analyses/1_preprocessing.py',
    'fichiers_analyses/2_extraction_competences.py',
    'fichiers_analyses/3_topic_modeling.py',
    'fichiers_analyses/4_geo_semantique.py',
    'fichiers_analyses/5_evolution_temporelle.py',
    'fichiers_analyses/6_embeddings_clustering.py',
    'fichiers_analyses/7_stacks_salaires.py',
    'fichiers_analyses/8_classification_supervisee.py',
    'fichiers_analyses/9_selection_chi2.py'
]

# Dossier de base
base_dir = Path(__file__).parent

# Statistiques d'execution
start_time_total = time.time()
results = []

for i, script in enumerate(scripts, 1):
    script_path = base_dir / script
    
    print(f"\n{'=' * 80}")
    print(f"[{i}/{len(scripts)}] Execution de {script}")
    print(f"{'=' * 80}\n")
    
    start_time = time.time()
    
    try:
        # Executer le script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=600  # 10 minutes max par script
        )
        
        # Afficher la sortie
        if result.stdout:
            print(result.stdout)
        
        # Verifier erreurs
        if result.returncode != 0:
            print(f"\n[ERREUR] dans {script}")
            if result.stderr:
                print(result.stderr)
            results.append({
                'script': script,
                'status': 'ERROR',
                'time': time.time() - start_time
            })
        else:
            elapsed = time.time() - start_time
            print(f"\n[OK] {script} termine en {elapsed:.1f}s")
            results.append({
                'script': script,
                'status': 'SUCCESS',
                'time': elapsed
            })
    
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] {script} a depasse 10 minutes")
        results.append({
            'script': script,
            'status': 'TIMEOUT',
            'time': 600
        })
    
    except Exception as e:
        print(f"\n[EXCEPTION] {e}")
        results.append({
            'script': script,
            'status': 'EXCEPTION',
            'time': time.time() - start_time
        })

# Recapitulatif
total_time = time.time() - start_time_total

print("\n" + "=" * 80)
print("RECAPITULATIF")
print("=" * 80)

success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
error_count = sum(1 for r in results if r['status'] == 'ERROR')
timeout_count = sum(1 for r in results if r['status'] == 'TIMEOUT')

print(f"\nResultats :")
print(f"  [OK]      : {success_count}/{len(scripts)}")
print(f"  [ERREUR]  : {error_count}/{len(scripts)}")
print(f"  [TIMEOUT] : {timeout_count}/{len(scripts)}")

print(f"\nTemps total : {total_time/60:.1f} minutes")

print(f"\nDetails par script :")
for r in results:
    status_icon = {
        'SUCCESS': '[OK]     ',
        'ERROR': '[ERREUR] ',
        'TIMEOUT': '[TIMEOUT]',
        'EXCEPTION': '[CRASH]  '
    }.get(r['status'], '[?]     ')
    
    print(f"  {status_icon} {r['script']:<50} ({r['time']:.1f}s)")

if success_count == len(scripts):
    print("\n" + "=" * 80)
    print("TOUTES LES ANALYSES SONT TERMINEES AVEC SUCCES !")
    print("=" * 80)
    print("\nVous pouvez maintenant lancer l'application Streamlit :")
    print("   cd app_streamlit")
    print("   streamlit run app.py")
else:
    print("\n" + "=" * 80)
    print("CERTAINES ANALYSES ONT ECHOUE")
    print("=" * 80)
    print("\nVerifiez les erreurs ci-dessus et relancez les scripts concernes.")

sys.exit(0 if success_count == len(scripts) else 1)