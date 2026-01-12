"""
Collecte EXHAUSTIVE - France Travail API
TOUTE LA FRANCE + TOUS LES MÉTIERS DATA/IA
Objectif: 2000-5000 offres
"""

from france_travail_scraper import FranceTravailScraper
from collections import Counter
from datetime import datetime
import time
import json

# ============================================
# VOS IDENTIFIANTS
# ============================================

CLIENT_ID = "PAR_projetnlptextmining_f2a0c35b68b0d2308fd06cf29c6d2b1c0a11711c12989f3e7a7a9faba6a2e8e7"
CLIENT_SECRET = "9d94b981172b01bdc5ebf978d20632284d868f675376e95f660ac1cb239c47d0"

# ============================================
# CONFIGURATION EXHAUSTIVE
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

# TOUS LES MÉTIERS DATA/IA (liste exhaustive)
TOUS_METIERS = [
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

# ============================================
# STRATÉGIES DE COLLECTE
# ============================================

def strategie_departements_prioritaires():
    """
    Stratégie 1: Départements à forte concentration tech
    Plus rapide, couvre 70-80% des offres
    """
    return {
        # Top 20 départements tech
        "75": "Paris", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
        "94": "Val-de-Marne", "91": "Essonne", "78": "Yvelines",
        "69": "Rhône", "31": "Haute-Garonne", "33": "Gironde",
        "59": "Nord", "13": "Bouches-du-Rhône", "44": "Loire-Atlantique",
        "67": "Bas-Rhin", "34": "Hérault", "35": "Ille-et-Vilaine",
        "38": "Isère", "06": "Alpes-Maritimes", "76": "Seine-Maritime",
        "54": "Meurthe-et-Moselle", "74": "Haute-Savoie"
    }

def strategie_metiers_core():
    """
    Stratégie 2: Métiers principaux seulement
    Plus rapide, moins de doublons
    """
    return [
        "Data Scientist", "Data Engineer", "Data Analyst",
        "Machine Learning Engineer", "Data Architect",
        "BI Analyst", "Big Data Engineer", "MLOps Engineer"
    ]

# ============================================
# FONCTION PRINCIPALE
# ============================================

def main():
    print("="*70)
    print(" COLLECTE EXHAUSTIVE - FRANCE TRAVAIL API")
    print("="*70)
    
    print("\n Choix de la stratégie de collecte:")
    print("\n  EXHAUSTIVE (Recommandé)")
    print("    • TOUS les départements (101)")
    print("    • Métiers principaux (8)")
    print("    • Estimation: 2000-3000 offres en ~15-20 minutes")
    
    print("\n  MAXIMALE")
    print("    • TOUS les départements (101)")
    print("    • TOUS les métiers (30+)")
    print("    • Estimation: 4000-6000 offres en ~40-60 minutes")
    
    print("\n  DÉPARTEMENTS PRIORITAIRES")
    print("    • 20 départements tech")
    print("    • TOUS les métiers (30+)")
    print("    • Estimation: 3000-4000 offres en ~25-35 minutes")
    
    print("\n  RAPIDE")
    print("    • 20 départements tech")
    print("    • 8 métiers principaux")
    print("    • Estimation: 1500-2000 offres en ~10-12 minutes")
    
    choix = input("\n  Choisissez votre stratégie (1/2/3/4): ")
    
    if choix == "1":
        departements = TOUS_DEPARTEMENTS
        metiers = strategie_metiers_core()
        nom_strategie = "EXHAUSTIVE"
    elif choix == "2":
        departements = TOUS_DEPARTEMENTS
        metiers = TOUS_METIERS
        nom_strategie = "MAXIMALE"
    elif choix == "3":
        departements = strategie_departements_prioritaires()
        metiers = TOUS_METIERS
        nom_strategie = "PRIORITAIRE"
    elif choix == "4":
        departements = strategie_departements_prioritaires()
        metiers = strategie_metiers_core()
        nom_strategie = "RAPIDE"
    else:
        print("❌ Choix invalide")
        return
    
    total_searches = len(departements) * len(metiers)
    
    print(f"\n Configuration {nom_strategie}:")
    print(f"  • Départements: {len(departements)}")
    print(f"  • Métiers: {len(metiers)}")
    print(f"  • Total recherches: {total_searches}")
    print(f"  • Temps estimé: ~{total_searches * 2 // 60} minutes")
    
    response = input("\n  Confirmer et démarrer ? (o/n): ")
    if response.lower() != 'o':
        print("❌ Collecte annulée")
        return
    
    # Initialiser le scraper
    print("\n  Initialisation du scraper...")
    scraper = FranceTravailScraper(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
    
    all_jobs = []
    seen_ids = set()
    current_search = 0
    errors = 0
    
    start_time = time.time()
    
    try:
        for dept_code, dept_name in departements.items():
            for keyword in metiers:
                current_search += 1
                
                # Affichage compact
                progress = f"[{current_search}/{total_searches}]"
                print(f"\n{progress}  {keyword[:25]:<25} - {dept_name[:20]:<20}", end=" ")
                
                try:
                    jobs = scraper.search_jobs(
                        keywords=keyword,
                        location=dept_code,
                        max_results=150
                    )
                    
                    # Dédupliquer
                    new_jobs = 0
                    for job in jobs:
                        job_id = job.get('id')
                        if job_id and job_id not in seen_ids:
                            seen_ids.add(job_id)
                            all_jobs.append(job)
                            new_jobs += 1
                    
                    print(f" +{new_jobs:3d} (Total: {len(all_jobs):4d})")
                    
                    # Sauvegardes intermédiaires tous les 500 offres
                    if len(all_jobs) > 0 and len(all_jobs) % 500 == 0:
                        backup_file = f"backup_{len(all_jobs)}_offres.json"
                        with open(backup_file, 'w', encoding='utf-8') as f:
                            json.dump(all_jobs, f, ensure_ascii=False, indent=2)
                        print(f"     Backup: {backup_file}")
                    
                    # Pause courte
                    time.sleep(0.5)
                
                except KeyboardInterrupt:
                    print("\n\n⚠️  Interruption par l'utilisateur")
                    raise
                
                except Exception as e:
                    errors += 1
                    print(f"❌ Erreur")
                    if errors > 10:
                        print(f"\n⚠️  Trop d'erreurs ({errors}), arrêt de sécurité")
                        break
                    continue
        
        elapsed_time = time.time() - start_time
        
        # Sauvegarder le résultat final
        print(f"\n\n{'='*70}")
        print(" SAUVEGARDE DES RÉSULTATS")
        print(f"{'='*70}")
        
        if all_jobs:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Fichier normalisé principal
            filename = f"corpus_france_travail_{nom_strategie.lower()}_{timestamp}.json"
            scraper.save_to_json(all_jobs, filename, normalize=True)
            
            # Statistiques détaillées
            print(f"\n STATISTIQUES FINALES")
            print(f"{'='*70}")
            print(f"Stratégie: {nom_strategie}")
            print(f"Offres collectées: {len(all_jobs)}")
            print(f"Recherches effectuées: {current_search}")
            print(f"Erreurs: {errors}")
            print(f"Temps: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f}s")
            print(f"Vitesse: {len(all_jobs) / (elapsed_time / 60):.0f} offres/min")
            
            # Top 15 localisations
            locations = [j.get('lieuTravail', {}).get('libelle', 'N/A') for j in all_jobs]
            print(f"\n Top 15 Localisations:")
            for i, (loc, count) in enumerate(Counter(locations).most_common(15), 1):
                print(f"  {i:2d}. {loc[:40]:<40}: {count:4d}")
            
            # Top 15 entreprises
            companies = [j.get('entreprise', {}).get('nom', 'N/A') for j in all_jobs]
            print(f"\n Top 15 Entreprises:")
            for i, (comp, count) in enumerate(Counter(companies).most_common(15), 1):
                print(f"  {i:2d}. {comp[:40]:<40}: {count:4d}")
            
            # Répartition par région (estimation)
            regions = {
                'Île-de-France': ['75', '77', '78', '91', '92', '93', '94', '95'],
                'Auvergne-Rhône-Alpes': ['69', '38', '42', '73', '74', '01', '03', '07', '15', '26', '43', '63'],
                'Nouvelle-Aquitaine': ['33', '64', '40', '47', '24', '19', '87', '16', '17', '23', '79', '86'],
                'Occitanie': ['31', '34', '30', '11', '66', '81', '82', '09', '12', '32', '46', '48', '65'],
                'Hauts-de-France': ['59', '62', '60', '02', '80'],
                'PACA': ['13', '06', '83', '84', '04', '05'],
                'Grand Est': ['67', '68', '54', '57', '51', '10', '52', '08', '55', '88'],
                'Bretagne': ['35', '29', '56', '22'],
                'Pays de la Loire': ['44', '49', '85', '72', '53'],
                'Normandie': ['76', '14', '27', '50', '61']
            }
            
            print(f"\n  Répartition par région:")
            region_counts = {}
            for job in all_jobs:
                dept = job.get('lieuTravail', {}).get('codePostal', '')[:2]
                for region, depts in regions.items():
                    if dept in depts:
                        region_counts[region] = region_counts.get(region, 0) + 1
                        break
            
            for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(all_jobs)) * 100
                print(f"  {region:<25}: {count:4d} ({percentage:5.1f}%)")
            
            # Types de contrat
            contracts = [j.get('typeContrat', 'N/A') for j in all_jobs]
            print(f"\n Types de contrat:")
            for contract, count in Counter(contracts).most_common():
                percentage = (count / len(all_jobs)) * 100
                print(f"  {contract:<15}: {count:4d} ({percentage:5.1f}%)")
            
            # Statistiques salaire
            with_salary = sum(1 for j in all_jobs if j.get('salaire'))
            print(f"\n Offres avec salaire: {with_salary} ({with_salary/len(all_jobs)*100:.1f}%)")
            
            # Statistiques compétences
            with_skills = sum(1 for j in all_jobs if j.get('competences'))
            print(f" Offres avec compétences: {with_skills} ({with_skills/len(all_jobs)*100:.1f}%)")
            
            # Statistiques GPS
            with_gps = sum(1 for j in all_jobs if j.get('lieuTravail', {}).get('latitude'))
            print(f" Offres avec GPS: {with_gps} ({with_gps/len(all_jobs)*100:.1f}%)")
            
            print(f"\n✅ Fichier sauvegardé: {filename}")
            print(f"{'='*70}")
            print("\n🎉 COLLECTE EXHAUSTIVE TERMINÉE AVEC SUCCÈS !")
        
        else:
            print("⚠️  Aucune offre collectée")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Collecte interrompue")
        if all_jobs:
            filename = f"corpus_interrompu_{len(all_jobs)}_offres.json"
            scraper.save_to_json(all_jobs, filename, normalize=True)
            print(f" {len(all_jobs)} offres sauvegardées dans {filename}")
    
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()