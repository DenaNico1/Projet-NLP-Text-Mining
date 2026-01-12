import streamlit as st
import sys
from pathlib import Path


# Ajouter dossier parent au path
sys.path.insert(0, str(Path(__file__).parent))

# Import thèmes
from themes import THEMES, get_theme_css, get_logo_html

# Configuration page (DOIT être la première commande Streamlit)
base_dir = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Data IA Talents Observatory",
    # page_icon="📊",
    page_icon=(base_dir / "assets" / "_icone.png").resolve(),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Analyse NLP des offres d'emploi Data/IA en France"},
)

# ============================================
# GESTION DE LA PERSISTENCE DE PAGE VIA URL
# ============================================

# Récupérer les query params de l'URL
query_params = st.query_params

# Liste des pages disponibles
pages = [
    "Dashboard",
    "Exploration Géographique",
    "Profils Métiers",
    "Compétences",
    "Topics & Tendances",
    "Matching CV - Offres",
    "Nouvelle Offre via LLM",
]

# Initialiser la page depuis l'URL ou mettre Dashboard par défaut
if "page" in query_params:
    url_page = query_params["page"]
    if url_page in pages:
        initial_page = url_page
    else:
        initial_page = "Dashboard"
else:
    initial_page = "Dashboard"

# Initialiser session_state avec la page de l'URL
if "current_page" not in st.session_state:
    st.session_state.current_page = initial_page

# Récupérer le thème depuis l'URL s'il existe
if "theme" in query_params:
    url_theme = query_params["theme"]
    if url_theme in THEMES.keys():
        initial_theme = url_theme
    else:
        initial_theme = "Dark Purple (Défaut)"
else:
    initial_theme = "Dark Purple (Défaut)"

# ============================================
# SIDEBAR - SÉLECTION THÈME + LOGO
# ============================================

with st.sidebar:
    # Logo en haut
    st.markdown(get_logo_html(size="210px"), unsafe_allow_html=True)

    st.markdown("---")

    # Sélecteur de thème
    st.markdown("### Thème")

    # Initialiser thème dans session_state
    if "theme" not in st.session_state:
        st.session_state.theme = initial_theme

    theme_choice = st.selectbox(
        "Choisir un thème",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme),
        key="theme_selector",
    )

    # Mettre à jour si changement
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        # Mettre à jour l'URL avec le nouveau thème
        st.query_params["theme"] = theme_choice
        st.rerun()

    st.markdown("---")

    # Navigation stylisée
    st.markdown("### Navigation")

    # CSS pour les boutons de navigation
    st.markdown(
        """
    <style>
    .nav-button {
        display: block;
        width: 100%;
        padding: 12px 16px;
        margin: 6px 0;
        border: none;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.05);
        color: #ffffff;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        font-weight: 500;
    }
    
    .nav-button:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(4px);
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        font-weight: 600;
    }
    
    .nav-icon {
        margin-right: 10px;
        font-size: 16px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Créer les boutons de navigation
    for page_name in pages:
        if st.button(
            page_name,
            key=f"nav_{page_name}",
            use_container_width=True,
            type=(
                "primary" if st.session_state.current_page == page_name else "secondary"
            ),
        ):
            st.session_state.current_page = page_name
            # Mettre à jour l'URL avec le nouveau paramètre de page
            st.query_params["page"] = page_name
            st.rerun()

    page = st.session_state.current_page

    st.markdown("---")

    # Filtres globaux
    st.markdown("### Filtres Globaux")

    filter_source = st.selectbox("Source", ["Toutes", "France Travail", "Indeed"])

    filter_region = st.selectbox(
        "Région",
        [
            "Toutes",
            "Île-de-France",
            "Auvergne-Rhône-Alpes",
            "Nouvelle-Aquitaine",
            "Occitanie",
            "Provence-Alpes-Côte d'Azur",
        ],
    )

    st.markdown("---")

    # Info projet
    with st.expander("À propos"):
        st.markdown(
            """
        **Projet NLP Text Mining**
        
        Master SISE - Janvier 2026
        
        Analyse approfondie des offres d'emploi Data/IA collectées en France.
        
        **Réalisé par:**
        - Nico DENA
        - Modou MBOUP
        - Constantin REY-COQUAIS
        - Léo-Paul
        """
        )

# ============================================
# APPLIQUER THÈME SÉLECTIONNÉ
# ============================================

current_theme = THEMES[st.session_state.theme]
st.markdown(get_theme_css(current_theme), unsafe_allow_html=True)

# ============================================
# ROUTING PAGES
# ============================================

# Stocker filtres en session state
if "filters" not in st.session_state:
    st.session_state.filters = {}

st.session_state.filters = {"source": filter_source, "region": filter_region}

# Router vers pages
if page == "Dashboard":
    exec(
        open(Path(__file__).parent / "pages" / "dashboard.py", encoding="utf-8").read()
    )

elif page == "Exploration Géographique":
    exec(
        open(
            Path(__file__).parent / "pages" / "geographique.py", encoding="utf-8"
        ).read()
    )

elif page == "Profils Métiers":
    exec(open(Path(__file__).parent / "pages" / "profils.py", encoding="utf-8").read())

elif page == "Compétences":
    exec(
        open(
            Path(__file__).parent / "pages" / "competences.py", encoding="utf-8"
        ).read()
    )

elif page == "Topics & Tendances":
    exec(open(Path(__file__).parent / "pages" / "topics.py", encoding="utf-8").read())

elif page == "Matching CV - Offres":
    exec(open(Path(__file__).parent / "pages" / "matching.py", encoding="utf-8").read())

elif page == "Nouvelle Offre via LLM":
    exec(
        open(
            Path(__file__).parent / "pages" / "nouvelle_offre.py", encoding="utf-8"
        ).read()
    )

# Footer
st.markdown(
    f"""
<div class='footer'>
    Made with Streamlit | Master SISE 2026 | Projet NLP Text Mining | Thème : {current_theme['name']}
</div>
""",
    unsafe_allow_html=True,
)
