"""
Indeed Scraper ANTI-DÉTECTION ULTIME
Utilise undetected-chromedriver pour contourner la détection bot
Version: 4.0 - Stealth Mode
Date: Décembre 2025
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
import time
import random
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import os
import re

# Import géocodage
try:
    from villes_france import geocode_location
except:
    print("⚠️  Module villes_france non trouvé, géocodage désactivé")
    geocode_location = None


class IndeedStealthScraper:
    """
    Scraper Indeed ANTI-DÉTECTION avec undetected-chromedriver
    Imite le comportement humain pour éviter les CAPTCHAs
    """
    
    BASE_URL = "https://fr.indeed.com"
    
    def __init__(self, headless: bool = False):
        """
        Initialise le scraper en mode stealth
        
        Args:
            headless: Mode sans interface (False recommandé pour debug)
        """
        print(" Initialisation du navigateur anti-détection...")
        
        options = uc.ChromeOptions()
        
        if headless:
            options.add_argument('--headless=new')
        
        # Options supplémentaires
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--start-maximized')
        
        # Créer le driver avec undetected-chromedriver
        self.driver = uc.Chrome(options=options, version_main=None)
        
        # Configurer les timeouts
        self.driver.set_page_load_timeout(30)
        
        print("✅ Navigateur anti-détection initialisé")
    
    def close(self):
        """Ferme le navigateur"""
        self.driver.quit()
        print(" Navigateur fermé")
    
    def _human_delay(self, min_sec: float = 1, max_sec: float = 3):
        """Pause aléatoire comme un humain"""
        time.sleep(random.uniform(min_sec, max_sec))
    
    def _human_scroll(self):
        """Scroll aléatoire comme un humain"""
        scroll_amount = random.randint(100, 400)
        self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
        self._human_delay(0.5, 1.5)
    
    def _move_mouse_randomly(self):
        """Bouge la souris aléatoirement"""
        try:
            action = ActionChains(self.driver)
            # Mouvement aléatoire
            x_offset = random.randint(-50, 50)
            y_offset = random.randint(-50, 50)
            action.move_by_offset(x_offset, y_offset).perform()
        except:
            pass
    
    def _accept_cookies(self):
        """Accepte les cookies si popup présent"""
        try:
            cookie_button = WebDriverWait(self.driver, 3).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            self._human_delay(0.5, 1)
            cookie_button.click()
            print(" Cookies acceptés")
            self._human_delay(1, 2)
        except:
            pass
    
    def _parse_salary(self, salary_text: str) -> tuple:
        """Parse le texte du salaire"""
        if not salary_text:
            return None, None
        
        try:
            pattern = r'(\d+(?:\s*\d+)*)\s*€\s*-\s*(\d+(?:\s*\d+)*)\s*€'
            match = re.search(pattern, salary_text)
            
            if match:
                min_sal = int(match.group(1).replace(' ', ''))
                max_sal = int(match.group(2).replace(' ', ''))
                
                if 'mois' in salary_text.lower():
                    min_sal *= 12
                    max_sal *= 12
                elif 'heure' in salary_text.lower():
                    min_sal *= 35 * 52
                    max_sal *= 35 * 52
                
                return min_sal, max_sal
            
            pattern2 = r'(\d+(?:\s*\d+)*)\s*€'
            match2 = re.search(pattern2, salary_text)
            
            if match2:
                sal = int(match2.group(1).replace(' ', ''))
                
                if 'mois' in salary_text.lower():
                    sal *= 12
                elif 'heure' in salary_text.lower():
                    sal *= 35 * 52
                
                return sal, sal
        except:
            pass
        
        return None, None
    
    def _extract_contract_type(self, text: str) -> Optional[str]:
        """Extrait le type de contrat"""
        if not text:
            return None
        
        text_lower = text.lower()
        
        if 'cdi' in text_lower:
            return 'CDI'
        elif 'cdd' in text_lower:
            return 'CDD'
        elif 'stage' in text_lower:
            return 'Stage'
        elif 'alternance' in text_lower or 'apprentissage' in text_lower:
            return 'Alternance'
        elif 'freelance' in text_lower or 'indépendant' in text_lower:
            return 'Freelance'
        elif 'intérim' in text_lower or 'interim' in text_lower:
            return 'Intérim'
        elif 'temps partiel' in text_lower:
            return 'Temps partiel'
        elif 'temps plein' in text_lower:
            return 'Temps plein'
        
        return None
    
    def _parse_date(self, date_text: str) -> Optional[str]:
        """Convertit dates relatives en ISO"""
        if not date_text:
            return None
        
        today = datetime.now()
        
        try:
            text_lower = date_text.lower()
            
            if 'aujourd\'hui' in text_lower or 'ajd' in text_lower:
                return today.strftime('%Y-%m-%d')
            
            if 'hier' in text_lower:
                return (today - timedelta(days=1)).strftime('%Y-%m-%d')
            
            match = re.search(r'il y a (\d+) jour', text_lower)
            if match:
                days = int(match.group(1))
                return (today - timedelta(days=days)).strftime('%Y-%m-%d')
            
            match = re.search(r'il y a (\d+) semaine', text_lower)
            if match:
                weeks = int(match.group(1))
                return (today - timedelta(weeks=weeks)).strftime('%Y-%m-%d')
            
            match = re.search(r'il y a (\d+) mois', text_lower)
            if match:
                months = int(match.group(1))
                return (today - timedelta(days=months*30)).strftime('%Y-%m-%d')
        except:
            pass
        
        return None
    
    def search_jobs_stealth(self, 
                           keywords: str, 
                           location: str = "France",
                           max_results: int = 50) -> List[Dict]:
        """
        Recherche Indeed en MODE STEALTH
        Imite un humain pour éviter détection
        
        Args:
            keywords: Mots-clés
            location: Localisation
            max_results: Nombre max d'offres
        
        Returns:
            Liste des offres avec descriptions
        """
        all_jobs = []
        
        print(f"\n  Recherche Indeed MODE STEALTH")
        print(f"   Mots-clés: {keywords}")
        print(f"   Localisation: {location}")
        print(f"   Objectif: {max_results} offres")
        print(f"     Mode lent (comportement humain)")
        print("-" * 70)
        
        search_url = f"{self.BASE_URL}/jobs?q={keywords.replace(' ', '+')}&l={location.replace(' ', '+')}"
        
        try:
            print(f" Chargement de la page...")
            self.driver.get(search_url)
            self._human_delay(3, 5)
            
            # Scroll un peu comme un humain
            self._human_scroll()
            
            self._accept_cookies()
            
            # Petit scroll supplémentaire
            self._human_scroll()
            
            page = 0
            max_pages = (max_results // 15) + 1
            
            while len(all_jobs) < max_results and page < max_pages:
                print(f"\n Page {page + 1}/{max_pages}")
                
                # Attendre et scroller
                self._human_delay(2, 3)
                self._human_scroll()
                
                # Compter les offres
                try:
                    job_cards = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_all_elements_located((By.CLASS_NAME, "job_seen_beacon"))
                    )
                    nb_cards = len(job_cards)
                except TimeoutException:
                    print("⚠️  Aucune offre trouvée")
                    break
                
                print(f" {nb_cards} offres trouvées")
                
                # Traiter chaque offre avec comportement humain
                for i in range(min(nb_cards, max_results - len(all_jobs))):
                    try:
                        # Re-chercher les cartes (évite stale element)
                        job_cards = self.driver.find_elements(By.CLASS_NAME, "job_seen_beacon")
                        
                        if i >= len(job_cards):
                            break
                        
                        card = job_cards[i]
                        
                        # Scroll vers la carte
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", card)
                        self._human_delay(0.5, 1)
                        
                        # Extraire titre
                        try:
                            title_elem = card.find_element(By.CSS_SELECTOR, "h2.jobTitle span")
                            title = title_elem.get_attribute('title') or title_elem.text
                        except:
                            title = "N/A"
                        
                        print(f"   [{i+1}/{nb_cards}] {title[:40]:<40}", end=" ", flush=True)
                        
                        # Bouger la souris un peu
                        self._move_mouse_randomly()
                        self._human_delay(0.3, 0.7)
                        
                        # CLIQUER sur la carte
                        try:
                            title_link = card.find_element(By.CSS_SELECTOR, "h2.jobTitle a")
                            # Click avec ActionChains (plus humain)
                            action = ActionChains(self.driver)
                            action.move_to_element(title_link).pause(0.2).click().perform()
                        except:
                            try:
                                card.click()
                            except:
                                print("❌ (impossible de cliquer)")
                                continue
                        
                        # Attendre chargement panneau droit (comportement humain)
                        self._human_delay(2, 4)
                        
                        # Petit scroll dans le panneau
                        self.driver.execute_script("window.scrollBy(0, 200);")
                        self._human_delay(0.5, 1)
                        
                        # Extraire depuis panneau droit
                        job_data = self._extract_from_right_panel()
                        
                        # Ajouter infos de base
                        job_data['title'] = title
                        
                        # Entreprise
                        try:
                            company_elem = card.find_element(By.CSS_SELECTOR, "span[data-testid='company-name']")
                            job_data['company'] = company_elem.text
                        except:
                            try:
                                company_elem = card.find_element(By.CLASS_NAME, "companyName")
                                job_data['company'] = company_elem.text
                            except:
                                job_data['company'] = "N/A"
                        
                        # Localisation
                        try:
                            location_elem = card.find_element(By.CSS_SELECTOR, "div[data-testid='text-location']")
                            job_data['location'] = location_elem.text
                        except:
                            try:
                                location_elem = card.find_element(By.CLASS_NAME, "companyLocation")
                                job_data['location'] = location_elem.text
                            except:
                                job_data['location'] = "N/A"
                        
                        # URL
                        try:
                            href = title_link.get_attribute('href')
                            match = re.search(r'jk=([a-f0-9]+)', href)
                            if match:
                                job_data['job_id'] = match.group(1)
                                job_data['url'] = f"https://fr.indeed.com/viewjob?jk={job_data['job_id']}"
                        except:
                            job_data['job_id'] = f"indeed_{i}_{int(time.time())}"
                            job_data['url'] = None
                        
                        job_data['scraped_at'] = datetime.now().isoformat()
                        job_data['source'] = 'Indeed'
                        
                        # Géocodage
                        if geocode_location and job_data.get('location'):
                            lat, lon, city, dept, region = geocode_location(job_data['location'])
                            job_data['latitude'] = lat
                            job_data['longitude'] = lon
                            job_data['city'] = city
                            job_data['department'] = dept
                            job_data['region'] = region
                        
                        # Afficher résultat
                        if job_data.get('description'):
                            desc_len = len(job_data['description'])
                            print(f" ({desc_len:4d} car.)")
                        else:
                            print(f"  (0 car.)")
                        
                        all_jobs.append(job_data)
                        
                        # Pause humaine entre offres
                        self._human_delay(2, 4)
                    
                    except Exception as e:
                        print(f"❌ {str(e)[:40]}")
                        continue
                
                # Stats
                with_desc = sum(1 for j in all_jobs if j.get('description'))
                print(f"\n Page terminée: {len(all_jobs)} total | {with_desc} avec desc. ({with_desc/len(all_jobs)*100 if all_jobs else 0:.0f}%)")
                
                # Page suivante
                if len(all_jobs) < max_results:
                    if not self._go_to_next_page():
                        print(" Dernière page")
                        break
                    page += 1
                    self._human_delay(3, 5)
            
            print(f"\n Collecte terminée: {len(all_jobs)} offres")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
        
        return all_jobs
    
    def _extract_from_right_panel(self) -> Dict:
        """Extrait depuis le panneau de droite"""
        details = {}
        
        # Description
        description_selectors = [
            (By.CLASS_NAME, "jobsearch-JobComponent-description"),
            (By.CSS_SELECTOR, ".jobsearch-JobComponent-description"),
            (By.ID, "jobDescriptionText"),
            (By.CLASS_NAME, "jobsearch-jobDescriptionText"),
        ]
        
        for by_method, selector in description_selectors:
            try:
                desc_elem = self.driver.find_element(by_method, selector)
                details['description'] = desc_elem.text.strip()
                if details['description']:
                    break
            except:
                continue
        
        if not details.get('description'):
            details['description'] = None
        
        # Salaire
        try:
            salary_elem = self.driver.find_element(By.CSS_SELECTOR, "#salaryInfoAndJobType")
            salary_text = salary_elem.text
            details['salary_text'] = salary_text
            details['salary_min'], details['salary_max'] = self._parse_salary(salary_text)
            details['contract_type'] = self._extract_contract_type(salary_text)
        except:
            details['salary_text'] = None
            details['salary_min'] = None
            details['salary_max'] = None
            details['contract_type'] = None
        
        # Date
        try:
            date_elem = self.driver.find_element(By.CLASS_NAME, "jobsearch-JobMetadataFooter")
            details['date_posted'] = self._parse_date(date_elem.text)
        except:
            details['date_posted'] = None
        
        # Compétences
        try:
            skills_elem = self.driver.find_element(By.XPATH, "//div[contains(text(), 'Compétences')]/parent::div")
            details['skills_mentioned'] = skills_elem.text
        except:
            details['skills_mentioned'] = None
        
        return details
    
    def _go_to_next_page(self) -> bool:
        """Va à la page suivante"""
        try:
            # Scroller un peu avant
            self._human_scroll()
            self._human_delay(1, 2)
            
            next_button = self.driver.find_element(By.CSS_SELECTOR, "a[data-testid='pagination-page-next']")
            
            # Click humain
            action = ActionChains(self.driver)
            action.move_to_element(next_button).pause(0.5).click().perform()
            
            self._human_delay(2, 4)
            return True
        except:
            return False
    
    def save_to_json(self, jobs: List[Dict], filename: str):
        """Sauvegarde en JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
        print(f" {len(jobs)} offres sauvegardées dans {filename}")


# Test
if __name__ == "__main__":
    print("="*70)
    print("  TEST INDEED STEALTH SCRAPER")
    print("="*70)
    
    scraper = IndeedStealthScraper(headless=False)
    
    try:
        jobs = scraper.search_jobs_stealth(
            keywords="Data Scientist",
            location="Lyon",
            max_results=10
        )
        
        if jobs:
            filename = f"indeed_stealth_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            scraper.save_to_json(jobs, filename)
            
            with_desc = sum(1 for j in jobs if j.get('description'))
            print(f"\n RÉSUMÉ:")
            print(f"   Total: {len(jobs)}")
            print(f"   Avec description: {with_desc} ({with_desc/len(jobs)*100:.0f}%)")
            
            if with_desc >= len(jobs) * 0.8:
                print(f"\n MODE STEALTH FONCTIONNE ! ({with_desc/len(jobs)*100:.0f}% de succès)")
            elif with_desc > 0:
                print(f"\n Succès partiel ({with_desc/len(jobs)*100:.0f}%)")
            else:
                print(f"\n Aucune description extraite")
    
    finally:
        scraper.close()