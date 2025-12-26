"""
Script de Vérification et Configuration
Vérifie que tout est en place avant de lancer les analyses

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import sys
from pathlib import Path
import os

def check_structure():
    """Vérifie la structure du projet"""
    
    print("="*70)
    print("🔍 VÉRIFICATION DE LA STRUCTURE DU PROJET")
    print("="*70)
    
    # Répertoire courant
    current_dir = Path.cwd()
    print(f"\n📁 Répertoire courant: {current_dir}")
    
    # Vérifications
    checks = {
        'entrepot_de_donnees': False,
        'base_duckdb': False,
        'resultats_nlp': False
    }
    
    # 1. Chercher entrepot_de_donnees
    print("\n🔎 Recherche de l'entrepôt de données...")
    
    possible_paths = [
        Path("../entrepot_de_donnees"),
        Path("../../entrepot_de_donnees"),
        Path("entrepot_de_donnees")
    ]
    
    entrepot_path = None
    for path in possible_paths:
        if path.exists():
            entrepot_path = path
            checks['entrepot_de_donnees'] = True
            print(f"   ✅ Trouvé: {path.absolute()}")
            break
    
    if not checks['entrepot_de_donnees']:
        print(f"   ❌ Dossier entrepot_de_donnees non trouvé !")
        return False
    
    # 2. Vérifier base DuckDB
    print("\n🔎 Recherche de la base DuckDB...")
    
    db_file = entrepot_path / "entrepot_nlp.duckdb"
    if db_file.exists():
        checks['base_duckdb'] = True
        size_mb = db_file.stat().st_size / (1024*1024)
        print(f"   ✅ Base trouvée: {db_file.absolute()}")
        print(f"   📊 Taille: {size_mb:.1f} MB")
    else:
        print(f"   ❌ Fichier entrepot_nlp.duckdb non trouvé !")
        print(f"   Cherché dans: {db_file.absolute()}")
        return False
    
    # 3. Créer dossier resultats_nlp
    print("\n📁 Vérification dossier résultats...")
    
    results_dir = Path("../resultats_nlp")
    if not results_dir.exists():
        print(f"   📁 Création de: {results_dir.absolute()}")
        results_dir.mkdir(parents=True, exist_ok=True)
    
    checks['resultats_nlp'] = True
    print(f"   ✅ Dossier résultats: {results_dir.absolute()}")
    
    # Créer sous-dossiers
    (results_dir / "visualisations").mkdir(exist_ok=True)
    (results_dir / "models").mkdir(exist_ok=True)
    
    # 4. Vérifier dépendances Python
    print("\n🐍 Vérification des dépendances Python...")
    
    required_modules = [
        'pandas', 'numpy', 'sklearn', 'nltk', 
        'gensim', 'plotly', 'wordcloud', 'duckdb'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - MANQUANT")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Modules manquants: {', '.join(missing)}")
        print(f"\n💡 Pour installer:")
        print(f"   pip install -r ../requirements.txt")
        return False
    
    # 5. Vérifier NLTK data
    print("\n📚 Vérification données NLTK...")
    
    try:
        import nltk
        
        nltk_data = ['punkt', 'stopwords', 'wordnet']
        nltk_ok = True
        
        for data in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}')
                print(f"   ✅ {data}")
            except LookupError:
                print(f"   ❌ {data} - MANQUANT")
                nltk_ok = False
        
        if not nltk_ok:
            print(f"\n💡 Pour télécharger:")
            print(f"   python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')\"")
            return False
            
    except ImportError:
        print(f"   ❌ NLTK non installé")
        return False
    
    # 6. Test connexion DuckDB
    print("\n🔌 Test connexion base de données...")
    
    try:
        import duckdb
        conn = duckdb.connect(str(db_file), read_only=True)
        
        # Compter offres
        count = conn.execute("SELECT COUNT(*) FROM fact_offres").fetchone()[0]
        print(f"   ✅ Connexion OK")
        print(f"   📊 Offres dans la base: {count}")
        
        conn.close()
    except Exception as e:
        print(f"   ❌ Erreur connexion: {e}")
        return False
    
    # Résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ")
    print("="*70)
    
    all_ok = all(checks.values())
    
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {check}")
    
    if all_ok:
        print("\n🎉 TOUT EST PRÊT !")
        print("\n🚀 Vous pouvez lancer les analyses:")
        print("   python run_all_analyses.py")
    else:
        print("\n⚠️  Certaines vérifications ont échoué")
        print("   Corrigez les problèmes ci-dessus avant de continuer")
    
    return all_ok


if __name__ == "__main__":
    success = check_structure()
    sys.exit(0 if success else 1)