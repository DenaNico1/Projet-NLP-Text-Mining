"""
DIAGNOSTIC : Vérifier chargement embeddings
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("DIAGNOSTIC EMBEDDINGS")
print("=" * 60)

# 1. Vérifier data_loaders
try:
    from data_loaders import load_matching_data
    print("✅ data_loaders importé")
    
    # 2. Charger données
    print("\n📦 Chargement données...")
    df, embeddings, rf_model, tfidf, emb_model, cv_base, metrics = load_matching_data()
    
    # 3. Vérifier embeddings
    print(f"\n🔍 Vérification embeddings:")
    print(f"   - Type: {type(embeddings)}")
    print(f"   - Est None: {embeddings is None}")
    
    if embeddings is not None:
        import numpy as np
        print(f"   - Shape: {embeddings.shape}")
        print(f"   - Dtype: {embeddings.dtype}")
        print(f"   ✅ Embeddings OK ({len(embeddings)} offres)")
    else:
        print(f"   ❌ Embeddings = None (PROBLÈME !)")
        
        # 4. Vérifier fichier
        from config import MODELS_DIR
        emb_path = MODELS_DIR / 'embeddings.npy'
        print(f"\n📁 Vérification fichier:")
        print(f"   - Chemin: {emb_path}")
        print(f"   - Existe: {emb_path.exists()}")
        
        if emb_path.exists():
            import numpy as np
            emb_test = np.load(emb_path)
            print(f"   - Taille fichier: {emb_path.stat().st_size / 1024 / 1024:.1f} MB")
            print(f"   - Shape: {emb_test.shape}")
            print(f"   ⚠️ Fichier existe mais load_matching_data() retourne None !")
        else:
            print(f"   ❌ Fichier embeddings.npy MANQUANT !")
            print(f"\n💡 SOLUTION:")
            print(f"   1. Exécuter: python resultats_nlp/7_embeddings.py")
            print(f"   2. Ou copier embeddings.npy depuis autre dossier")
    
    # 5. Vérifier autres composants
    print(f"\n🔍 Autres composants:")
    print(f"   - df: {len(df) if df is not None else 'None'} offres")
    print(f"   - rf_model: {'✅ OK' if rf_model is not None else '❌ None'}")
    print(f"   - emb_model: {'✅ OK' if emb_model is not None else '❌ None'}")
    print(f"   - cv_base: {len(cv_base) if cv_base else 'None'} CVs")

except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)