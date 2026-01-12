"""
France Travail API Scraper - Collecte d'offres d'emploi via l'API officielle
Auteur: Projet NLP Text Mining
Date: Décembre 2024

API Documentation: https://francetravail.io/data/api/offres-emploi
"""

import requests
import json
import time
from typing import List, Dict, Optional
from datetime import datetime
import os


class FranceTravailScraper:
    """
    Scraper utilisant l'API officielle de France Travail (ex-Pôle Emploi)
    
    Avantages:
    - 100% légal et gratuit
    - Données structurées de qualité
    - Pas de risque de blocage
    - Informations complètes et fiables
    """
    
    API_BASE_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2"
    AUTH_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=%2Fpartenaire"
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        """
        Initialise le scraper France Travail
        
        Args:
            client_id: Identifiant client API (optionnel si variable d'environnement)
            client_secret: Secret client API (optionnel si variable d'environnement)
        
        Note:
            Pour obtenir vos identifiants API:
            1. Créer un compte sur https://francetravail.io/
            2. Créer une application
            3. Récupérer client_id et client_secret
        """
        # Récupérer les credentials depuis les variables d'environnement ou paramètres
        self.client_id = client_id or os.getenv('FRANCE_TRAVAIL_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('FRANCE_TRAVAIL_CLIENT_SECRET')
        
        self.access_token = None
        self.token_expiry = None
        
        if not self.client_id or not self.client_secret:
            print("⚠️  ATTENTION: Identifiants API non fournis")
            print(" Pour obtenir vos identifiants:")
            print("   1. Allez sur https://francetravail.io/")
            print("   2. Créez un compte développeur")
            print("   3. Créez une application")
            print("   4. Récupérez client_id et client_secret")
            print("\n💡 Ensuite, utilisez:")
            print("   scraper = FranceTravailScraper(client_id='...', client_secret='...')")
    
    def _get_access_token(self) -> bool:
        """
        Obtient un token d'accès OAuth2
        
        Returns:
            True si succès, False sinon
        """
        if not self.client_id or not self.client_secret:
            print("❌ Impossible d'obtenir un token sans identifiants")
            return False
        
        try:
            print(" Obtention du token d'accès...")
            
            data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'scope': 'api_offresdemploiv2 o2dsoffre'
            }
            
            response = requests.post(
                self.AUTH_URL,
                data=data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.token_expiry = time.time() + token_data.get('expires_in', 3600)
                print("✅ Token obtenu avec succès")
                return True
            else:
                print(f"❌ Erreur d'authentification: {response.status_code}")
                print(f"   Réponse: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de l'authentification: {e}")
            return False
    
    def _ensure_token(self) -> bool:
        """Vérifie et renouvelle le token si nécessaire"""
        if not self.access_token or (self.token_expiry and time.time() > self.token_expiry - 60):
            return self._get_access_token()
        return True
    
    def search_jobs(self,
                    keywords: str,
                    location: str = None,
                    contract_type: str = None,
                    max_results: int = 150,
                    experience: str = None) -> List[Dict]:
        """
        Recherche des offres d'emploi via l'API France Travail
        
        Args:
            keywords: Mots-clés de recherche (ex: "Data Scientist")
            location: Code département ou commune (ex: "69" pour Rhône, "75" pour Paris)
            contract_type: Type de contrat ("CDI", "CDD", "MIS", "SAI")
            max_results: Nombre maximum de résultats (max 150 par requête)
            experience: Niveau d'expérience ("D" débutant, "E" expérimenté, "S" expert)
        
        Returns:
            Liste des offres d'emploi
        """
        if not self._ensure_token():
            print("❌ Impossible de continuer sans token valide")
            return []
        
        all_jobs = []
        
        # Construire les paramètres de recherche
        params = {
            'motsCles': keywords,
            'range': f'0-{min(max_results, 150) - 1}',  # API limite à 150
            'sort': '2'  # Trier par date (plus récent d'abord)
        }
        
        if location:
            # Déterminer si c'est un département ou une commune
            if location.isdigit() and len(location) <= 3:
                params['departement'] = location
            else:
                params['commune'] = location
        
        if contract_type:
            params['typeContrat'] = contract_type
        
        if experience:
            params['experience'] = experience
        
        try:
            print(f"\n Recherche France Travail")
            print(f"   Mots-clés: {keywords}")
            if location:
                print(f"   Localisation: {location}")
            if contract_type:
                print(f"   Type de contrat: {contract_type}")
            print(f"   Max résultats: {max_results}")
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            url = f"{self.API_BASE_URL}/offres/search"
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'resultats' in data:
                    jobs = data['resultats']
                    print(f"✅ {len(jobs)} offres trouvées")
                    
                    # Enrichir les données avec métadonnées
                    for job in jobs:
                        job['scraped_at'] = datetime.now().isoformat()
                        job['source'] = 'France Travail'
                        job['search_keywords'] = keywords
                    
                    all_jobs.extend(jobs)
                else:
                    print("⚠️  Aucune offre trouvée")
            
            elif response.status_code == 401:
                print("❌ Token invalide, tentative de renouvellement...")
                if self._get_access_token():
                    return self.search_jobs(keywords, location, contract_type, max_results, experience)
            
            else:
                print(f"❌ Erreur API: {response.status_code}")
                print(f"   Réponse: {response.text[:200]}")
        
        except Exception as e:
            print(f"❌ Erreur lors de la recherche: {e}")
        
        return all_jobs
    
    def get_job_details(self, job_id: str) -> Optional[Dict]:
        """
        Récupère les détails complets d'une offre
        
        Args:
            job_id: Identifiant de l'offre
        
        Returns:
            Dictionnaire avec les détails de l'offre
        """
        if not self._ensure_token():
            return None
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            url = f"{self.API_BASE_URL}/offres/{job_id}"
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Erreur lors de la récupération des détails: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return None
    
    def normalize_job_data(self, job: Dict) -> Dict:
        """
        Normalise les données pour correspondre au format du projet
        
        Args:
            job: Offre brute de l'API France Travail
        
        Returns:
            Offre normalisée
        """
        normalized = {
            'job_id': job.get('id'),
            'title': job.get('intitule'),
            'company': job.get('entreprise', {}).get('nom', 'N/A'),
            'location': job.get('lieuTravail', {}).get('libelle', 'N/A'),
            'contract_type': job.get('typeContrat', 'N/A'),
            'description': job.get('description', ''),
            'skills': job.get('competences', []),
            'experience': job.get('experienceExige', 'N/A'),
            'education': job.get('formations', []),
            'salary': job.get('salaire', {}).get('libelle', 'N/A'),
            'duration': job.get('dureeTravailLibelle', 'N/A'),
            'date_posted': job.get('dateCreation'),
            'url': f"https://candidat.francetravail.fr/offres/recherche/detail/{job.get('id')}",
            'source': 'France Travail',
            'scraped_at': job.get('scraped_at', datetime.now().isoformat())
        }
        
        # Ajouter coordonnées GPS si disponibles
        lieu = job.get('lieuTravail', {})
        if 'latitude' in lieu and 'longitude' in lieu:
            normalized['latitude'] = lieu['latitude']
            normalized['longitude'] = lieu['longitude']
        
        return normalized
    
    def save_to_json(self, jobs: List[Dict], filename: str, normalize: bool = True):
        """
        Sauvegarde les offres au format JSON
        
        Args:
            jobs: Liste des offres
            filename: Nom du fichier de sortie
            normalize: Si True, normalise les données avant sauvegarde
        """
        if normalize:
            jobs = [self.normalize_job_data(job) for job in jobs]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
        
        print(f" {len(jobs)} offres sauvegardées dans {filename}")


def demo_without_credentials():
    """
    Démo montrant comment utiliser le scraper (sans vraies requêtes)
    """
    print("="*70)
    print(" DÉMONSTRATION - France Travail API Scraper")
    print("="*70)
    
    print("\n⚠️  Pour utiliser ce scraper, vous avez besoin d'identifiants API")
    print("\n📝 Étapes pour obtenir vos identifiants:")
    print("   1. Allez sur https://francetravail.io/")
    print("   2. Cliquez sur 'Espace développeur' ou 'S'inscrire'")
    print("   3. Créez un compte (gratuit)")
    print("   4. Créez une nouvelle application")
    print("   5. Sélectionnez l'API 'Offres d'emploi v2'")
    print("   6. Récupérez votre client_id et client_secret")
    
    print("\n💻 Une fois les identifiants obtenus, utilisez:")
    print("""
# Méthode 1: Passer directement les identifiants
scraper = FranceTravailScraper(
    client_id='VOTRE_CLIENT_ID',
    client_secret='VOTRE_CLIENT_SECRET'
)

# Méthode 2: Variables d'environnement (recommandé)
# export FRANCE_TRAVAIL_CLIENT_ID='...'
# export FRANCE_TRAVAIL_CLIENT_SECRET='...'
scraper = FranceTravailScraper()

# Rechercher des offres
jobs = scraper.search_jobs(
    keywords="Data Scientist",
    location="69",  # Rhône (Lyon)
    contract_type="CDI",
    max_results=150
)

# Sauvegarder
scraper.save_to_json(jobs, "offres_france_travail.json")
""")
    
    print("\n Codes de localisation utiles:")
    print("   • '69' = Rhône (Lyon)")
    print("   • '75' = Paris")
    print("   • '31' = Haute-Garonne (Toulouse)")
    print("   • '33' = Gironde (Bordeaux)")
    print("   • '59' = Nord (Lille)")
    print("   • '13' = Bouches-du-Rhône (Marseille)")
    
    print("\n Types de contrat:")
    print("   • 'CDI' = Contrat à Durée Indéterminée")
    print("   • 'CDD' = Contrat à Durée Déterminée")
    print("   • 'MIS' = Mission d'intérim")
    print("   • 'SAI' = Contrat saisonnier")

    print("\n Niveaux d'expérience:")
    print("   • 'D' = Débutant accepté")
    print("   • 'E' = Expérience exigée")
    print("   • 'S' = Expérience souhaitée")
    
    print("\n Avantages de France Travail API:")
    print("   • 100% légal et gratuit")
    print("   • Données structurées et de qualité")
    print("   • Coordonnées GPS incluses")
    print("   • Compétences extraites automatiquement")
    print("   • Pas de limite de requêtes (usage raisonnable)")
    print("   • Pas de risque de blocage")
    
    print("\n" + "="*70)


def main():
    """Fonction principale de démonstration"""
    demo_without_credentials()


if __name__ == "__main__":
    main()