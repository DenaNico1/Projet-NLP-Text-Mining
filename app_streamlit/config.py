"""
Configuration Application Data IA Talent Observatory
"""

from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR.parent / 'resultats_nlp'
MODELS_DIR = RESULTS_DIR / 'models'
VIZ_DIR = RESULTS_DIR / 'visualisations'

# Couleurs
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#FFD700',
    'success': '#10b981',
    'info': '#3b82f6',
}

# Config
PAGE_CONFIG = {
    'page_title': 'Data IA Talent Observatory',
    'page_icon': '',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}
