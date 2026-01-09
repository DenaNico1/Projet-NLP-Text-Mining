# scraper.py
"""
Module de scraping d'offres depuis URL
Support : Indeed, LinkedIn, APEC, France Travail, etc.
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

# ============================================
# SCRAPING SIMPLE (BeautifulSoup)
# ============================================

def scrape_simple(url):
    """
    Scraping simple pour sites statiques
    
    Fonctionne pour : APEC, France Travail, certains sites carrière
    """
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extraire texte brut
        # Retirer scripts/styles
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        
        # Nettoyer
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Supprimer lignes vides multiples
        text = re.sub(r' +', ' ', text)  # Supprimer espaces multiples
        
        return {
            'success': True,
            'text': text,
            'method': 'simple'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Erreur scraping simple : {str(e)}"
        }


# ============================================
# SCRAPING DYNAMIQUE (Selenium)
# ============================================

def scrape_dynamic(url):
    """
    Scraping avec Selenium pour sites JavaScript (Indeed, LinkedIn)
    
    Nécessite Chrome/Chromium installé
    """
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Mode sans interface
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Attendre chargement (max 10 sec)
        time.sleep(3)
        
        # Récupérer texte de la page
        page_source = driver.page_source
        driver.quit()
        
        # Parser avec BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Nettoyer
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return {
            'success': True,
            'text': text,
            'method': 'selenium'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Erreur Selenium : {str(e)}"
        }


# ============================================
# DÉTECTION AUTO MÉTHODE
# ============================================

def detect_site_type(url):
    """
    Détecte le type de site et méthode appropriée
    """
    
    url_lower = url.lower()
    
    # Sites nécessitant Selenium (JavaScript)
    dynamic_sites = ['indeed', 'linkedin', 'glassdoor', 'monster']
    
    for site in dynamic_sites:
        if site in url_lower:
            return 'dynamic'
    
    # Sites simples (statiques)
    simple_sites = ['apec', 'francetravail', 'pole-emploi', 'meteojob']
    
    for site in simple_sites:
        if site in url_lower:
            return 'simple'
    
    # Par défaut : essayer simple d'abord
    return 'simple'


# ============================================
# FONCTION PRINCIPALE
# ============================================

def scrape_job_offer(url):
    """
    Scrape offre depuis URL (détection auto méthode)
    
    Args:
        url (str): URL offre
    
    Returns:
        dict: {
            'success': bool,
            'text': str,  # Texte brut extrait
            'method': str,  # Méthode utilisée
            'error': str  # Si échec
        }
    """
    
    # Détection méthode
    site_type = detect_site_type(url)
    
    if site_type == 'dynamic':
        # Essayer Selenium
        result = scrape_dynamic(url)
        
        # Fallback simple si échec
        if not result['success']:
            result = scrape_simple(url)
    else:
        # Essayer simple
        result = scrape_simple(url)
        
        # Fallback Selenium si échec
        if not result['success']:
            result = scrape_dynamic(url)
    
    return result


# ============================================
# EXTRACTEURS SPÉCIFIQUES (BONUS)
# ============================================

def extract_indeed(url):
    """Extracteur optimisé Indeed"""
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Attendre élément spécifique Indeed
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "jobsearch-JobComponent")))
        
        # Extraire éléments structurés
        title = driver.find_element(By.CSS_SELECTOR, "h1.jobsearch-JobInfoHeader-title").text
        company = driver.find_element(By.CSS_SELECTOR, "[data-company-name='true']").text
        
        description_elem = driver.find_element(By.ID, "jobDescriptionText")
        description = description_elem.text
        
        driver.quit()
        
        # Formatter pour LLM
        text = f"""
Titre : {title}
Entreprise : {company}

Description :
{description}
"""
        
        return {
            'success': True,
            'text': text,
            'method': 'indeed_specific'
        }
        
    except Exception as e:
        driver.quit()
        return {
            'success': False,
            'error': f"Erreur Indeed : {str(e)}"
        }


def extract_linkedin(url):
    """Extracteur optimisé LinkedIn (nécessite login)"""
    
    # LinkedIn bloque scraping sans login
    # Alternative : utiliser API LinkedIn ou scraping authentifié
    
    return {
        'success': False,
        'error': "LinkedIn nécessite authentification. Utilisez copier-coller manuel."
    }


# ============================================
# WRAPPER INTELLIGENT
# ============================================

def smart_scrape(url):
    """
    Scraping intelligent avec extracteurs spécifiques
    
    1. Détecte site
    2. Utilise extracteur optimisé si disponible
    3. Fallback scraping générique
    """
    
    url_lower = url.lower()
    
    # Extracteurs spécifiques
    if 'indeed.com' in url_lower or 'indeed.fr' in url_lower:
        result = extract_indeed(url)
        if result['success']:
            return result
    
    elif 'linkedin.com' in url_lower:
        return extract_linkedin(url)
    
    # Fallback générique
    return scrape_job_offer(url)


# ============================================
# EXEMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Test
    url = "https://fr.indeed.com/viewjob?jk=exemple"
    
    result = smart_scrape(url)
    
    if result['success']:
        print(f"✅ Texte extrait ({result['method']}) :")
        print(result['text'][:500])
    else:
        print(f"❌ Erreur : {result['error']}")