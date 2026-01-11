"""
Analyse descriptions détaillées des offres non classifiées
Affiche 10 exemples pour comprendre pourquoi elles ne matchent pas

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import pandas as pd
import pickle

def analyze_descriptions():
    """Affiche descriptions complètes des non classifiés"""
    
    print("="*70)
    print("📋 DESCRIPTIONS OFFRES NON CLASSIFIÉES")
    print("="*70)
    
    # Charger données
    with open('../resultats_nlp/models/data_with_profiles.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Filtrer non classifiés
    df_unclass = df[df['status'] == 'unclassified'].copy()
    
    print(f"\n📊 Total non classifiés: {len(df_unclass)}")
    
    # Prendre 10 exemples variés
    print("\n" + "="*70)
    print("📄 10 EXEMPLES DÉTAILLÉS")
    print("="*70)
    
    for i, (idx, row) in enumerate(df_unclass.head(10).iterrows(), 1):
        print(f"\n{'='*70}")
        print(f"OFFRE #{i}")
        print(f"{'='*70}")
        
        print("\n📌 TITRE:")
        print(f"   {row['title']}")
        
        print("\n📍 SOURCE:")
        print(f"   {row['source_name']}")
        
        print("\n📊 SCORES:")
        print(f"   Score final: {row['profil_score']:.2f}/10")
        print(f"   Score titre: {row['score_title']:.2f}/10")
        print(f"   Score description: {row['score_description']:.2f}/10")
        print(f"   Score compétences: {row['score_competences']:.2f}/10")
        
        print(f"\n🔑 COMPÉTENCES EXTRAITES ({row['num_competences']} trouvées):")
        if row['num_competences'] > 0:
            comps = row['competences_found'][:10]  # Max 10
            for comp in comps:
                print(f"   - {comp}")
        else:
            print("   Aucune")
        
        print("\n📝 DESCRIPTION (premiers 500 caractères):")
        desc = row.get('description', '')
        if desc and not pd.isna(desc):
            print(f"   {desc[:500]}...")
        else:
            print("   [Pas de description]")
        
        print("\n📝 TEXT_FOR_SKLEARN (premiers 300 caractères):")
        text_sk = row.get('text_for_sklearn', '')
        if text_sk and not pd.isna(text_sk):
            print(f"   {text_sk[:300]}...")
        else:
            print("   [Pas de texte]")
        
        print("\n🎯 PROFIL 2ÈME POSITION:")
        print(f"   {row.get('profil_second', 'Aucun')} (score: {row.get('profil_second_score', 0):.2f})")
    
    print("\n" + "="*70)
    print("✅ ANALYSE TERMINÉE")
    print("="*70)


if __name__ == "__main__":
    analyze_descriptions()