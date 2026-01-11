# scraper.py (VERSION SIMPLIFIÉE - INDEED + FRANCE TRAVAIL UNIQUEMENT)
"""
Module de scraping d'offres depuis URL

SOURCES SUPPORTÉES :
✅ Indeed France
✅ France Travail (API + Scraping)

ARCHITECTURE COHÉRENTE :
- Formats données standardisés
- Extraction LLM optimisée
- Maintenance simplifiée
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
import hashlib

# ============================================
# VALIDATION SOURCE
# ============================================

SOURCES_AUTORISEES = ['indeed', 'francetravail', 'pole-emploi']

def validate_source(url):
    """
    Vérifie que l'URL provient d'une source autorisée
    
    Returns:
        tuple: (is_valid: bool, source: str, message: str)
    """
    
    url_lower = url.lower()
    
    # Indeed
    if 'indeed.fr' in url_lower or 'indeed.com' in url_lower:
        return True, 'indeed', "✅ Source Indeed autorisée"
    
    # France Travail / Pole Emploi
    if 'francetravail' in url_lower or 'pole-emploi' in url_lower:
        return True, 'francetravail', "✅ Source France Travail autorisée"
    
    # Source non autorisée
    return False, 'unknown', """
    ❌ Source non supportée
    
    **Sources acceptées :**
    - Indeed France (fr.indeed.com)
    - France Travail (francetravail.fr)
    
    **Cette URL provient d'un autre site.**
    
    💡 Utilisez le mode "Texte manuel" si vous souhaitez quand même ajouter cette offre.
    """


# ============================================
# EXTRACTION ID OFFRE
# ============================================

def extract_job_id_from_url(url):
    """
    Extrait l'identifiant unique offre depuis URL
    
    Supporte : Indeed, France Travail
    """
    
    url_lower = url.lower()
    
    # ============================================
    # INDEED
    # ============================================
    if 'indeed' in url_lower:
        # Chercher jk= ou vjk= (formats standards)
        match = re.search(r'[v]?jk=([a-f0-9]+)', url, re.IGNORECASE)
        if match:
            job_id = match.group(1)
            return {
                'source': 'indeed',
                'job_id': job_id,
                'unique_key': f'indeed_{job_id}'
            }
        
        # CORRECTION : Si URL Indeed mais pas de jk= trouvé
        # (ex: URL redirect Indeed complexe)
        # → Utiliser hash URL mais garder source 'indeed'
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return {
            'source': 'indeed',
            'job_id': url_hash,
            'unique_key': f'indeed_{url_hash}'
        }
    
    # ============================================
    # FRANCE TRAVAIL / POLE EMPLOI
    # ============================================
    elif 'francetravail' in url_lower or 'pole-emploi' in url_lower:
        # Format : /detail/123ABC456 ou /offre/123ABC456
        match = re.search(r'/(?:detail|offre)/([A-Z0-9]+)', url, re.IGNORECASE)
        if match:
            job_id = match.group(1)
            return {
                'source': 'francetravail',
                'job_id': job_id,
                'unique_key': f'francetravail_{job_id}'
            }
        
        # CORRECTION : Si URL France Travail mais pas d'ID trouvé
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return {
            'source': 'francetravail',
            'job_id': url_hash,
            'unique_key': f'francetravail_{url_hash}'
        }
    
    # Fallback : Hash URL + source unknown
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    
    return {
        'source': 'unknown',
        'job_id': url_hash,
        'unique_key': f'url_{url_hash}'
    }


# ============================================
# CORRECTION URL INDEED
# ============================================

def fix_indeed_url(url):
    """
    Corrige URL Indeed malformées
    
    Exemples :
    - https://fr.indeed.com/?vjk=abc123 → https://fr.indeed.com/viewjob?jk=abc123
    """
    
    if 'indeed' not in url.lower():
        return url
    
    # Extraire job ID
    match = re.search(r'[v]?jk=([a-f0-9]+)', url, re.IGNORECASE)
    
    if match:
        job_id = match.group(1)
        
        # Reconstruire URL correcte
        if 'indeed.fr' in url or 'fr.indeed.com' in url:
            corrected = f"https://fr.indeed.com/viewjob?jk={job_id}"
        else:
            corrected = f"https://www.indeed.com/viewjob?jk={job_id}"
        
        if corrected != url:
            print(f"🔧 URL corrigée : {corrected}")
        
        return corrected
    
    return url


# ============================================
# SCRAPING INDEED (Selenium optimisé)
# ============================================

def scrape_indeed(url):
    """
    Scraping optimisé Indeed avec sélecteurs CSS précis
    
    Extrait :
    - Titre du poste
    - Entreprise
    - Localisation
    - Description complète
    """
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    driver = None
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Attendre chargement contenu
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.jobsearch-JobComponent")))
        
        time.sleep(2)  # Sécurité
        
        # Extraire éléments structurés
        try:
            title = driver.find_element(By.CSS_SELECTOR, "h1.jobsearch-JobInfoHeader-title span").text
        except:
            title = "Non spécifié"
        
        try:
            company = driver.find_element(By.CSS_SELECTOR, "[data-company-name='true']").text
        except:
            company = "Non spécifié"
        
        try:
            location = driver.find_element(By.CSS_SELECTOR, "[data-testid='job-location']").text
        except:
            location = "Non spécifié"
        
        try:
            description_elem = driver.find_element(By.ID, "jobDescriptionText")
            description = description_elem.text
        except:
            # Fallback : tout le texte
            description = driver.find_element(By.TAG_NAME, "body").text
        
        driver.quit()
        
        # Formatter pour LLM
        text = f"""
Titre du poste : {title}
Entreprise : {company}
Localisation : {location}

Description complète :
{description}
"""
        
        return {
            'success': True,
            'text': text,
            'method': 'indeed_selenium',
            'url': url
        }
        
    except Exception as e:
        if driver:
            driver.quit()
        
        return {
            'success': False,
            'error': f"Erreur scraping Indeed : {str(e)}"
        }


# ============================================
# SCRAPING FRANCE TRAVAIL (BeautifulSoup)
# ============================================

def scrape_francetravail(url):
    """
    Scraping France Travail (site public)
    
    Alternative : Utiliser API officielle France Travail
    """
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Nettoyer
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        
        # Nettoyer lignes vides
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return {
            'success': True,
            'text': text,
            'method': 'francetravail_beautifulsoup',
            'url': url
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Erreur scraping France Travail : {str(e)}"
        }


# ============================================
# FONCTION PRINCIPALE (SIMPLIFIÉE)
# ============================================

def smart_scrape(url):
    """
    Scraping intelligent pour Indeed + France Travail UNIQUEMENT
    
    1. Valide source autorisée
    2. Extrait ID offre
    3. Corrige URL si nécessaire
    4. Scrape avec méthode optimisée
    
    Returns:
        dict: {
            'success': bool,
            'text': str,
            'method': str,
            'url': str,
            'job_info': dict,
            'error': str
        }
    """
    
    # ============================================
    # 1. VALIDATION SOURCE
    # ============================================
    
    is_valid, source, message = validate_source(url)
    
    if not is_valid:
        return {
            'success': False,
            'error': message
        }
    
    # ============================================
    # 2. EXTRACTION ID
    # ============================================
    
    job_info = extract_job_id_from_url(url)
    
    # ============================================
    # 3. CORRECTION URL (Indeed)
    # ============================================
    
    if source == 'indeed':
        url = fix_indeed_url(url)
    
    # ============================================
    # 4. SCRAPING
    # ============================================
    
    if source == 'indeed':
        result = scrape_indeed(url)
    elif source == 'francetravail':
        result = scrape_francetravail(url)
    else:
        result = {
            'success': False,
            'error': "Source non reconnue"
        }
    
    # Ajouter job_info au résultat
    if result['success']:
        result['job_info'] = job_info
    
    return result


# ============================================
# EXEMPLE USAGE
# ============================================

if __name__ == "__main__":
    
    # Test URLs valides
    test_urls = [
        "https://fr.indeed.com/viewjob?jk=abc123",
        "https://candidat.francetravail.fr/offres/recherche/detail/123ABC456"
    ]
    
    # Test URL invalide
    invalid_urls = [
        "https://www.linkedin.com/jobs/view/123456",
        "https://www.apec.fr/offre/123456"
    ]
    
    print("=" * 60)
    print("TEST URLS VALIDES")
    print("=" * 60)
    
    for url in test_urls:
        is_valid, source, msg = validate_source(url)
        print(f"\n✅ {url}")
        print(f"   Source: {source}")
    
    print("\n" + "=" * 60)
    print("TEST URLS INVALIDES")
    print("=" * 60)
    
    for url in invalid_urls:
        is_valid, source, msg = validate_source(url)
        print(f"\n❌ {url}")
        print(f"   {msg}")



# # scraper.py (VERSION SIMPLIFIÉE - INDEED + FRANCE TRAVAIL UNIQUEMENT)
# """
# Module de scraping d'offres depuis URL

# SOURCES SUPPORTÉES :
# ✅ Indeed France
# ✅ France Travail (API + Scraping)

# ARCHITECTURE COHÉRENTE :
# - Formats données standardisés
# - Extraction LLM optimisée
# - Maintenance simplifiée
# """

# import requests
# from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# import time
# import re
# import hashlib

# # ============================================
# # VALIDATION SOURCE
# # ============================================

# SOURCES_AUTORISEES = ['indeed', 'francetravail', 'pole-emploi']

# def validate_source(url):
#     """
#     Vérifie que l'URL provient d'une source autorisée
    
#     Returns:
#         tuple: (is_valid: bool, source: str, message: str)
#     """
    
#     url_lower = url.lower()
    
#     # Indeed
#     if 'indeed.fr' in url_lower or 'indeed.com' in url_lower:
#         return True, 'indeed', "✅ Source Indeed autorisée"
    
#     # France Travail / Pole Emploi
#     if 'francetravail' in url_lower or 'pole-emploi' in url_lower:
#         return True, 'francetravail', "✅ Source France Travail autorisée"
    
#     # Source non autorisée
#     return False, 'unknown', f"""
#     ❌ Source non supportée
    
#     **Sources acceptées :**
#     - Indeed France (fr.indeed.com)
#     - France Travail (francetravail.fr)
    
#     **Cette URL provient d'un autre site.**
    
#     💡 Utilisez le mode "Texte manuel" si vous souhaitez quand même ajouter cette offre.
#     """


# # ============================================
# # EXTRACTION ID OFFRE
# # ============================================

# def extract_job_id_from_url(url):
#     """
#     Extrait l'identifiant unique offre depuis URL
    
#     Supporte : Indeed, France Travail
#     """
    
#     url_lower = url.lower()
    
#     # ============================================
#     # INDEED
#     # ============================================
#     if 'indeed' in url_lower:
#         match = re.search(r'[v]?jk=([a-f0-9]+)', url, re.IGNORECASE)
#         if match:
#             job_id = match.group(1)
#             return {
#                 'source': 'indeed',
#                 'job_id': job_id,
#                 'unique_key': f'indeed_{job_id}'
#             }
    
#     # ============================================
#     # FRANCE TRAVAIL / POLE EMPLOI
#     # ============================================
#     elif 'francetravail' in url_lower or 'pole-emploi' in url_lower:
#         # Format : /detail/123ABC456 ou /offre/123ABC456
#         match = re.search(r'/(?:detail|offre)/([A-Z0-9]+)', url, re.IGNORECASE)
#         if match:
#             job_id = match.group(1)
#             return {
#                 'source': 'francetravail',
#                 'job_id': job_id,
#                 'unique_key': f'francetravail_{job_id}'
#             }
    
#     # Fallback : Hash URL
#     url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    
#     return {
#         'source': 'unknown',
#         'job_id': url_hash,
#         'unique_key': f'url_{url_hash}'
#     }


# # ============================================
# # CORRECTION URL INDEED
# # ============================================

# def fix_indeed_url(url):
#     """
#     Corrige URL Indeed malformées
    
#     Exemples :
#     - https://fr.indeed.com/?vjk=abc123 → https://fr.indeed.com/viewjob?jk=abc123
#     """
    
#     if 'indeed' not in url.lower():
#         return url
    
#     # Extraire job ID
#     match = re.search(r'[v]?jk=([a-f0-9]+)', url, re.IGNORECASE)
    
#     if match:
#         job_id = match.group(1)
        
#         # Reconstruire URL correcte
#         if 'indeed.fr' in url or 'fr.indeed.com' in url:
#             corrected = f"https://fr.indeed.com/viewjob?jk={job_id}"
#         else:
#             corrected = f"https://www.indeed.com/viewjob?jk={job_id}"
        
#         if corrected != url:
#             print(f"🔧 URL corrigée : {corrected}")
        
#         return corrected
    
#     return url


# # ============================================
# # SCRAPING INDEED (Selenium optimisé)
# # ============================================

# def scrape_indeed(url):
#     """
#     Scraping optimisé Indeed avec sélecteurs CSS précis
    
#     Extrait :
#     - Titre du poste
#     - Entreprise
#     - Localisation
#     - Description complète
#     """
    
#     chrome_options = Options()
#     chrome_options.add_argument('--headless')
#     chrome_options.add_argument('--no-sandbox')
#     chrome_options.add_argument('--disable-dev-shm-usage')
#     chrome_options.add_argument('--disable-gpu')
#     chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
#     driver = None
    
#     try:
#         driver = webdriver.Chrome(options=chrome_options)
#         driver.get(url)
        
#         # Attendre chargement contenu
#         wait = WebDriverWait(driver, 10)
#         wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.jobsearch-JobComponent")))
        
#         time.sleep(2)  # Sécurité
        
#         # Extraire éléments structurés
#         try:
#             title = driver.find_element(By.CSS_SELECTOR, "h1.jobsearch-JobInfoHeader-title span").text
#         except:
#             title = "Non spécifié"
        
#         try:
#             company = driver.find_element(By.CSS_SELECTOR, "[data-company-name='true']").text
#         except:
#             company = "Non spécifié"
        
#         try:
#             location = driver.find_element(By.CSS_SELECTOR, "[data-testid='job-location']").text
#         except:
#             location = "Non spécifié"
        
#         try:
#             description_elem = driver.find_element(By.ID, "jobDescriptionText")
#             description = description_elem.text
#         except:
#             # Fallback : tout le texte
#             description = driver.find_element(By.TAG_NAME, "body").text
        
#         driver.quit()
        
#         # Formatter pour LLM
#         text = f"""
# Titre du poste : {title}
# Entreprise : {company}
# Localisation : {location}

# Description complète :
# {description}
# """
        
#         return {
#             'success': True,
#             'text': text,
#             'method': 'indeed_selenium',
#             'url': url
#         }
        
#     except Exception as e:
#         if driver:
#             driver.quit()
        
#         return {
#             'success': False,
#             'error': f"Erreur scraping Indeed : {str(e)}"
#         }


# # ============================================
# # SCRAPING FRANCE TRAVAIL (BeautifulSoup)
# # ============================================

# def scrape_francetravail(url):
#     """
#     Scraping France Travail (site public)
    
#     Alternative : Utiliser API officielle France Travail
#     """
    
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#     }
    
#     try:
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
        
#         soup = BeautifulSoup(response.content, 'html.parser')
        
#         # Nettoyer
#         for script in soup(["script", "style", "nav", "footer", "header"]):
#             script.decompose()
        
#         text = soup.get_text(separator='\n', strip=True)
        
#         # Nettoyer lignes vides
#         text = re.sub(r'\n\s*\n', '\n\n', text)
#         text = re.sub(r' +', ' ', text)
        
#         return {
#             'success': True,
#             'text': text,
#             'method': 'francetravail_beautifulsoup',
#             'url': url
#         }
        
#     except Exception as e:
#         return {
#             'success': False,
#             'error': f"Erreur scraping France Travail : {str(e)}"
#         }


# # ============================================
# # FONCTION PRINCIPALE (SIMPLIFIÉE)
# # ============================================

# def smart_scrape(url):
#     """
#     Scraping intelligent pour Indeed + France Travail UNIQUEMENT
    
#     1. Valide source autorisée
#     2. Extrait ID offre
#     3. Corrige URL si nécessaire
#     4. Scrape avec méthode optimisée
    
#     Returns:
#         dict: {
#             'success': bool,
#             'text': str,
#             'method': str,
#             'url': str,
#             'job_info': dict,
#             'error': str
#         }
#     """
    
#     # ============================================
#     # 1. VALIDATION SOURCE
#     # ============================================
    
#     is_valid, source, message = validate_source(url)
    
#     if not is_valid:
#         return {
#             'success': False,
#             'error': message
#         }
    
#     # ============================================
#     # 2. EXTRACTION ID
#     # ============================================
    
#     job_info = extract_job_id_from_url(url)
    
#     # ============================================
#     # 3. CORRECTION URL (Indeed)
#     # ============================================
    
#     if source == 'indeed':
#         url = fix_indeed_url(url)
    
#     # ============================================
#     # 4. SCRAPING
#     # ============================================
    
#     if source == 'indeed':
#         result = scrape_indeed(url)
#     elif source == 'francetravail':
#         result = scrape_francetravail(url)
#     else:
#         result = {
#             'success': False,
#             'error': "Source non reconnue"
#         }
    
#     # Ajouter job_info au résultat
#     if result['success']:
#         result['job_info'] = job_info
    
#     return result


# # ============================================
# # EXEMPLE USAGE
# # ============================================

# if __name__ == "__main__":
    
#     # Test URLs valides
#     test_urls = [
#         "https://fr.indeed.com/viewjob?jk=abc123",
#         "https://candidat.francetravail.fr/offres/recherche/detail/123ABC456"
#     ]
    
#     # Test URL invalide
#     invalid_urls = [
#         "https://www.linkedin.com/jobs/view/123456",
#         "https://www.apec.fr/offre/123456"
#     ]
    
#     print("=" * 60)
#     print("TEST URLS VALIDES")
#     print("=" * 60)
    
#     for url in test_urls:
#         is_valid, source, msg = validate_source(url)
#         print(f"\n✅ {url}")
#         print(f"   Source: {source}")
    
#     print("\n" + "=" * 60)
#     print("TEST URLS INVALIDES")
#     print("=" * 60)
    
#     for url in invalid_urls:
#         is_valid, source, msg = validate_source(url)
#         print(f"\n❌ {url}")
#         print(f"   {msg}")