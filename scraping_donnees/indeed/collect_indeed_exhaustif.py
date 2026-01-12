"""
Collecte Indeed STEALTH - EXHAUSTIVE
TOUS LES DÉPARTEMENTS (adaptation du code qui marche)

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indeed_stealth_scraper import IndeedStealthScraper
from collections import Counter
from datetime import datetime
import time
import json

# ============================================
# DÉPARTEMENTS - AU LIEU DE 5 VILLES
# ============================================

# TOUS LES DÉPARTEMENTS DE FRANCE (101 départements)
TOUS_DEPARTEMENTS = {
    # Île-de-France
    "75": "Paris", "77": "Seine-et-Marne", "78": "Yvelines", "91": "Essonne",
    "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis", "94": "Val-de-Marne", "95": "Val-d'Oise",
    
    # Auvergne-Rhône-Alpes
    "01": "Ain", "03": "Allier", "07": "Ardèche", "15": "Cantal", "26": "Drôme",
    "38": "Isère", "42": "Loire", "43": "Haute-Loire", "63": "Puy-de-Dôme",
    "69": "Rhône", "73": "Savoie", "74": "Haute-Savoie",
    
    # Bourgogne-Franche-Comté
    "21": "Côte-d'Or", "25": "Doubs", "39": "Jura", "58": "Nièvre",
    "70": "Haute-Saône", "71": "Saône-et-Loire", "89": "Yonne", "90": "Territoire de Belfort",
    
    # Bretagne
    "22": "Côtes-d'Armor", "29": "Finistère", "35": "Ille-et-Vilaine", "56": "Morbihan",
    
    # Centre-Val de Loire
    "18": "Cher", "28": "Eure-et-Loir", "36": "Indre", "37": "Indre-et-Loire",
    "41": "Loir-et-Cher", "45": "Loiret",
    
    # Corse
    "2A": "Corse-du-Sud", "2B": "Haute-Corse",
    
    # Grand Est
    "08": "Ardennes", "10": "Aube", "51": "Marne", "52": "Haute-Marne",
    "54": "Meurthe-et-Moselle", "55": "Meuse", "57": "Moselle", "67": "Bas-Rhin",
    "68": "Haut-Rhin", "88": "Vosges",
    
    # Hauts-de-France
    "02": "Aisne", "59": "Nord", "60": "Oise", "62": "Pas-de-Calais", "80": "Somme",
    
    # Normandie
    "14": "Calvados", "27": "Eure", "50": "Manche", "61": "Orne", "76": "Seine-Maritime",
    
    # Nouvelle-Aquitaine
    "16": "Charente", "17": "Charente-Maritime", "19": "Corrèze", "23": "Creuse",
    "24": "Dordogne", "33": "Gironde", "40": "Landes", "47": "Lot-et-Garonne",
    "64": "Pyrénées-Atlantiques", "79": "Deux-Sèvres", "86": "Vienne", "87": "Haute-Vienne",
    
    # Occitanie
    "09": "Ariège", "11": "Aude", "12": "Aveyron", "30": "Gard", "31": "Haute-Garonne",
    "32": "Gers", "34": "Hérault", "46": "Lot", "48": "Lozère", "65": "Hautes-Pyrénées",
    "66": "Pyrénées-Orientales", "81": "Tarn", "82": "Tarn-et-Garonne",
    
    # Pays de la Loire
    "44": "Loire-Atlantique", "49": "Maine-et-Loire", "53": "Mayenne",
    "72": "Sarthe", "85": "Vendée",
    
    # Provence-Alpes-Côte d'Azur
    "04": "Alpes-de-Haute-Provence", "05": "Hautes-Alpes", "06": "Alpes-Maritimes",
    "13": "Bouches-du-Rhône", "83": "Var", "84": "Vaucluse"
}

# ============================================
# STRATÉGIES
# ============================================

def get_villes_test():
    """5 villes pour test (comme avant)"""
    return {
        "75": "Paris",
        "69": "Lyon",
        "13": "Marseille", 
        "31": "Toulouse",
        "33": "Bordeaux"
    }

def get_villes_prioritaires():
    """20 départements tech"""
    return {
        "75": "Paris", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
        "94": "Val-de-Marne", "91": "Essonne", "78": "Yvelines",
        "69": "Rhône", "31": "Haute-Garonne", "33": "Gironde",
        "59": "Nord", "13": "Bouches-du-Rhône", "44": "Loire-Atlantique",
        "67": "Bas-Rhin", "34": "Hérault", "35": "Ille-et-Vilaine",
        "38": "Isère", "06": "Alpes-Maritimes", "76": "Seine-Maritime",
        "54": "Meurthe-et-Moselle", "74": "Haute-Savoie"
    }

# ============================================
# MÉTIERS (identiques à ton code)
# ============================================

METIERS = [
    # Core Data Science
    "Data Scientist",
    "Data Analyst", 
    "Data Engineer",
    "Data Architect",
    
    # Machine Learning / IA
    "Machine Learning Engineer",
    "ML Engineer",
    "Deep Learning Engineer",
    "Ingénieur Machine Learning",
    "Ingénieur IA",
    "AI Engineer",
    "Research Scientist",
    "Computer Vision Engineer",
    "NLP Engineer",
    
    # Analytics & Business Intelligence
    "Business Intelligence",
    "BI Analyst",
    "Analytics Engineer",
    "Analyste de données",
    "Analyste décisionnel",
    
    # Big Data
    "Big Data Engineer",
    "Big Data Architect",
    "Data Platform Engineer",
    "Ingénieur Big Data",
    
    # Spécialisations
    "MLOps Engineer",
    "Quantitative Analyst",
    "Statisticien",
    "Data Science Manager",
    "Chief Data Officer",
    "Lead Data Scientist",
    
    # Recherches génériques
    "Data",
    "Intelligence Artificielle",
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Python développeur data",
]

# Variants (Stage, Alternance)
MODIFIERS = [
    "",           # Normal
    "Stage",      # Stages
    "Alternance", # Alternances
]

# Offres par recherche
RESULTS_PER_SEARCH = 20

# ============================================
# FONCTION PRINCIPALE (TON CODE)
# ============================================

def main():
    print("="*70)
    print("  COLLECTE INDEED STEALTH - EXHAUSTIVE")
    print("="*70)
    
    # CHOIX STRATÉGIE
    print("\n Choix de la stratégie:")
    print("  1. TEST (5 villes) - ~2-3h - 1,000-2,000 offres")
    print("  2. PRIORITAIRE (20 départements) - ~10-15h - 4,000-8,000 offres")
    print("  3. EXHAUSTIVE (101 départements) - ~50-80h - 10,000-20,000 offres")
    
    choix = input("\n  Votre choix (1/2/3): ")
    
    if choix == "1":
        VILLES = get_villes_test()
        nom_strategie = "TEST"
    elif choix == "2":
        VILLES = get_villes_prioritaires()
        nom_strategie = "PRIORITAIRE"
    elif choix == "3":
        VILLES = TOUS_DEPARTEMENTS
        nom_strategie = "EXHAUSTIVE"
    else:
        print("❌ Choix invalide")
        return
    
    total_searches = len(VILLES) * len(METIERS) * len(MODIFIERS)
    estimated_offers = total_searches * RESULTS_PER_SEARCH * 0.4
    estimated_minutes = (total_searches * 1.3)
    
    print(f"\n Configuration {nom_strategie}:")
    print(f"  • Villes/Départements: {len(VILLES)}")
    print(f"  • Métiers: {len(METIERS)}")
    print(f"  • Modifiers: {len(MODIFIERS)} (normal, stage, alternance)")
    print(f"  • Offres par recherche: {RESULTS_PER_SEARCH}")
    
    print(f"\n Estimations:")
    print(f"  • Nombre de recherches: {total_searches}")
    print(f"  • Offres estimées: ~{int(estimated_offers)}")
    print(f"  • Temps estimé: ~{estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f}h)")
    
    print(f"\n⚠️  IMPORTANT:")
    print(f"  • Mode STEALTH (lent mais fonctionne)")
    print(f"  • ~4 secondes par offre (comportement humain)")
    print(f"  • Sauvegardes automatiques tous les 100 offres")
    print(f"  • Vous pouvez interrompre (Ctrl+C) sans perdre les données")
    
    response = input(f"\n▶️  Démarrer la collecte {nom_strategie} ? (o/n): ")
    if response.lower() != 'o':
        print("❌ Collecte annulée")
        return
    
    # Initialiser le scraper
    print("\n  Initialisation du scraper stealth...")
    scraper = IndeedStealthScraper(headless=False)  # Mode VISIBLE pour debug
    
    all_jobs = []
    seen_ids = set()
    current_search = 0
    errors = 0
    max_consecutive_errors = 5
    consecutive_errors = 0
    
    start_time = time.time()
    
    try:
        # BOUCLE SUR DÉPARTEMENTS (au lieu de VILLES)
        for dept_code, ville in VILLES.items():
            for metier in METIERS:
                for modifier in MODIFIERS:
                    current_search += 1
                    
                    # Construire la requête
                    if modifier:
                        query = f"{metier} {modifier}"
                    else:
                        query = metier
                    
                    print(f"\n{'='*70}")
                    print(f"Recherche {current_search}/{total_searches}")
                    print(f"🎯 {query} - {ville}")
                    print(f"{'='*70}")
                    
                    try:
                        # Scraper STEALTH
                        jobs = scraper.search_jobs_stealth(
                            keywords=query,
                            location=ville,
                            max_results=RESULTS_PER_SEARCH
                        )
                        
                        # Dédupliquer
                        new_jobs = 0
                        for job in jobs:
                            job_id = job.get('job_id', f"{job.get('title')}-{job.get('company')}")
                            
                            if job_id not in seen_ids:
                                seen_ids.add(job_id)
                                all_jobs.append(job)
                                new_jobs += 1
                        
                        duplicates = len(jobs) - new_jobs
                        print(f"✅ Nouvelles: {new_jobs} | Doublons: {duplicates}")
                        print(f" Total cumulé: {len(all_jobs)} offres uniques")
                        
                        # Reset erreurs consécutives
                        consecutive_errors = 0
                        
                        # Backup tous les 100 offres
                        if len(all_jobs) > 0 and len(all_jobs) % 100 == 0:
                            backup_file = f"backup_indeed_stealth_{len(all_jobs)}_offres.json"
                            with open(backup_file, 'w', encoding='utf-8') as f:
                                json.dump(all_jobs, f, ensure_ascii=False, indent=2)
                            print(f" Backup: {backup_file}")
                        
                        # Petite pause entre recherches
                        if current_search < total_searches:
                            print("  Pause 5 secondes...")
                            time.sleep(5)
                    
                    except KeyboardInterrupt:
                        print("\n\n⚠️  Interruption utilisateur")
                        raise
                    
                    except Exception as e:
                        errors += 1
                        consecutive_errors += 1
                        print(f"❌ Erreur: {str(e)[:100]}")
                        
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"\n⚠️  Trop d'erreurs consécutives ({consecutive_errors})")
                            print("Arrêt de sécurité")
                            break
                        
                        print("  Passage à la recherche suivante...")
                        time.sleep(10)
                        continue
                
                if consecutive_errors >= max_consecutive_errors:
                    break
            
            if consecutive_errors >= max_consecutive_errors:
                break
        
        # Temps écoulé
        elapsed_time = time.time() - start_time
        
        # Sauvegarder résultat final
        print(f"\n{'='*70}")
        print(" SAUVEGARDE DES RÉSULTATS")
        print(f"{'='*70}")
        
        if all_jobs:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"corpus_indeed_stealth_{nom_strategie.lower()}_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_jobs, f, ensure_ascii=False, indent=2)
            
            # Statistiques
            print(f"\n STATISTIQUES FINALES")
            print(f"{'='*70}")
            print(f"Total d'offres: {len(all_jobs)}")
            print(f"Recherches effectuées: {current_search}/{total_searches}")
            print(f"Erreurs: {errors}")
            print(f"Temps: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")
            print(f"Vitesse: {len(all_jobs) / (elapsed_time / 60):.0f} offres/min")
            
            # Top villes
            cities = [j.get('city') or j.get('location', 'N/A') for j in all_jobs]
            print(f"\n Top 10 Villes:")
            for i, (city, count) in enumerate(Counter(cities).most_common(10), 1):
                print(f"  {i:2d}. {city[:30]:<30}: {count}")
            
            # Top entreprises
            companies = [j.get('company', 'N/A') for j in all_jobs if j.get('company') != 'N/A']
            print(f"\n Top 10 Entreprises:")
            for i, (comp, count) in enumerate(Counter(companies).most_common(10), 1):
                print(f"  {i:2d}. {comp[:30]:<30}: {count}")
            
            # Types de contrat
            contracts = [j.get('contract_type', 'Non spécifié') for j in all_jobs]
            print(f"\n Types de contrat:")
            for contract, count in Counter(contracts).most_common():
                pct = (count / len(all_jobs)) * 100
                print(f"  {contract:<20}: {count:4d} ({pct:5.1f}%)")
            
            # Qualité des données
            with_desc = sum(1 for j in all_jobs if j.get('description'))
            with_salary = sum(1 for j in all_jobs if j.get('salary_min'))
            with_gps = sum(1 for j in all_jobs if j.get('latitude'))
            
            print(f"\n Qualité des données:")
            print(f"  Description complète: {with_desc}/{len(all_jobs)} ({with_desc/len(all_jobs)*100:.1f}%)")
            print(f"  Salaire renseigné: {with_salary}/{len(all_jobs)} ({with_salary/len(all_jobs)*100:.1f}%)")
            print(f"  GPS calculé: {with_gps}/{len(all_jobs)} ({with_gps/len(all_jobs)*100:.1f}%)")
            
            # Régions
            regions = [j.get('region', 'N/A') for j in all_jobs if j.get('region')]
            if regions:
                print(f"\n Régions:")
                for i, (region, count) in enumerate(Counter(regions).most_common(), 1):
                    pct = (count / len(all_jobs)) * 100
                    print(f"  {i}. {region:<30}: {count:4d} ({pct:5.1f}%)")
            
            print(f"\n Fichier: {filename}")
            print(f"{'='*70}")
            print(f"\n COLLECTE {nom_strategie} TERMINÉE AVEC SUCCÈS !")
            
            # Exemple
            print(f"\n Exemple d'offre:")
            print("-"*70)
            if all_jobs:
                ex = all_jobs[0]
                for key, value in ex.items():
                    if key == 'description':
                        preview = value[:150] + "..." if value and len(value) > 150 else value
                        print(f"  {key:20s}: {preview}")
                    elif key not in ['summary', 'skills_mentioned']:
                        print(f"  {key:20s}: {value}")
        
        else:
            print("⚠️  Aucune offre collectée")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Collecte interrompue")
        if all_jobs:
            filename = f"corpus_indeed_interrompu_{len(all_jobs)}_offres.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_jobs, f, ensure_ascii=False, indent=2)
            print(f" {len(all_jobs)} offres sauvegardées: {filename}")
    
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        
        if all_jobs:
            filename = f"corpus_indeed_erreur_{len(all_jobs)}_offres.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_jobs, f, ensure_ascii=False, indent=2)
            print(f" {len(all_jobs)} offres sauvegardées: {filename}")
    
    finally:
        print("\n Fermeture du navigateur...")
        scraper.close()


if __name__ == "__main__":
    main()