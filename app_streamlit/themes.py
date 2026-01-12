# themes.py
"""
Système de thèmes personnalisables pour DataJobs Explorer
"""

THEMES = {
    "Dark Purple (Défaut)": {
        "name": "Dark Purple",
        "app_bg": "linear-gradient(135deg, #0e1117 0%, #1a1d29 100%)",
        "sidebar_bg": "#111827",
        "card_bg": "linear-gradient(135deg, #1f2937 0%, #374151 100%)",
        "primary_color": "#667eea",
        "secondary_color": "#764ba2",
        "accent_color": "#FFD700",
        "text_color": "#ffffff",
        "border_color": "#374151",
    },
    "Ocean Blue": {
        "name": "Ocean Blue",
        "app_bg": "linear-gradient(135deg, #0a1929 0%, #1a2332 100%)",
        "sidebar_bg": "#0f1419",
        "card_bg": "linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%)",
        "primary_color": "#3b82f6",
        "secondary_color": "#1e40af",
        "accent_color": "#60a5fa",
        "text_color": "#ffffff",
        "border_color": "#2563eb",
    },
    "Emerald Green": {
        "name": "Emerald Green",
        "app_bg": "linear-gradient(135deg, #0a1612 0%, #0f2419 100%)",
        "sidebar_bg": "#0a1612",
        "card_bg": "linear-gradient(135deg, #1a3e2e 0%, #2d5f4a 100%)",
        "primary_color": "#10b981",
        "secondary_color": "#047857",
        "accent_color": "#34d399",
        "text_color": "#ffffff",
        "border_color": "#059669",
    },
    "Sunset Orange": {
        "name": "Sunset Orange",
        "app_bg": "linear-gradient(135deg, #1a0f0a 0%, #2b1810 100%)",
        "sidebar_bg": "#1a0f0a",
        "card_bg": "linear-gradient(135deg, #3e2617 0%, #5a3a1f 100%)",
        "primary_color": "#f59e0b",
        "secondary_color": "#d97706",
        "accent_color": "#fbbf24",
        "text_color": "#ffffff",
        "border_color": "#f59e0b",
    },
    "Rose Pink": {
        "name": "Rose Pink",
        "app_bg": "linear-gradient(135deg, #1a0a14 0%, #2b1020 100%)",
        "sidebar_bg": "#1a0a14",
        "card_bg": "linear-gradient(135deg, #3e1728 0%, #5a1f38 100%)",
        "primary_color": "#ec4899",
        "secondary_color": "#be185d",
        "accent_color": "#f472b6",
        "text_color": "#ffffff",
        "border_color": "#db2777",
    },
    "Cyber Neon": {
        "name": "Cyber Neon",
        "app_bg": "linear-gradient(135deg, #000000 0%, #0a0a1a 100%)",
        "sidebar_bg": "#000000",
        "card_bg": "linear-gradient(135deg, #0f0f2e 0%, #1a1a3e 100%)",
        "primary_color": "#00ffff",
        "secondary_color": "#ff00ff",
        "accent_color": "#00ff00",
        "text_color": "#ffffff",
        "border_color": "#00ffff",
    },
    "Light Mode": {
        "name": "Light Mode",
        "app_bg": "linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)",
        "sidebar_bg": "#ffffff",
        "card_bg": "linear-gradient(135deg, #ffffff 0%, #f9fafb 100%)",
        "primary_color": "#667eea",
        "secondary_color": "#764ba2",
        "accent_color": "#f59e0b",
        "text_color": "#1f2937",
        "border_color": "#d1d5db",
    },
}


def get_theme_css(theme_config):
    """
    Génère le CSS complet pour un thème

    Args:
        theme_config (dict): Configuration du thème

    Returns:
        str: CSS formaté
    """

    is_light = theme_config["name"] == "Light Mode"
    text_muted = "#6b7280" if is_light else "#9ca3af"
    hover_bg = "#f3f4f6" if is_light else "#374151"

    css = f"""
<style>
    /* ============================================
       BACKGROUND & LAYOUT
    ============================================ */
    .stApp {{
        background: {theme_config['app_bg']};
        padding-left: 0;
        padding-right: 0;
    }}
    
    [data-testid="stAppViewContainer"] {{
        padding-left: 0;
        padding-right: 0;
    }}

    [data-testid="stMainBlockContainer"] {{
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }}
    
    /* ============================================
       SIDEBAR
    ============================================ */
    [data-testid="stSidebar"] {{
        background: {theme_config['sidebar_bg']} !important;
        border-right: 1px solid {theme_config['border_color']};
    }}
    
    [data-testid="stSidebar"] .stRadio > label {{
        color: {theme_config['text_color']};
        padding: 8px 12px;
        border-radius: 8px;
        transition: all 0.3s;
    }}
    
    [data-testid="stSidebar"] .stRadio > label:hover {{
        background: {hover_bg};
    }}
    
    /* ============================================
       CARDS & CONTAINERS
    ============================================ */
    .metric-card {{
        background: {theme_config['card_bg']};
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid {theme_config['primary_color']};
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 10px 0;
    }}
    
    div[data-testid="stMetricValue"] {{
        background: {theme_config['card_bg']};
        padding: 15px;
        border-radius: 8px;
        border: 1px solid {theme_config['border_color']};
    }}
    
    /* ============================================
       TYPOGRAPHY
    ============================================ */
    h1 {{
        background: linear-gradient(135deg, {theme_config['primary_color']} 0%, {theme_config['secondary_color']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }}
    
    h2, h3 {{
        color: {theme_config['primary_color']};
    }}
    
    p, span, div {{
        color: {theme_config['text_color']};
    }}
    
    /* ============================================
       BUTTONS
    ============================================ */
    .stButton>button {{
        background: linear-gradient(135deg, {theme_config['primary_color']} 0%, {theme_config['secondary_color']} 100%);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
        transition: all 0.3s;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }}
    
    /* ============================================
       METRICS
    ============================================ */
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: {theme_config['primary_color']};
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {text_muted};
        font-size: 0.9rem;
    }}
    
    /* ============================================
       INPUTS & SELECTS
    ============================================ */
    .stSelectbox > div > div {{
        background: {theme_config['card_bg']};
        border-radius: 8px;
        border: 1px solid {theme_config['border_color']};
        color: {theme_config['text_color']};
    }}
    
    .stTextInput > div > div > input {{
        background: {theme_config['card_bg']};
        border: 1px solid {theme_config['border_color']};
        color: {theme_config['text_color']};
        border-radius: 8px;
    }}
    
    /* ============================================
       EXPANDER
    ============================================ */
    .streamlit-expanderHeader {{
        background: {theme_config['card_bg']};
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid {theme_config['border_color']};
        color: {theme_config['text_color']};
    }}
    
    .streamlit-expanderContent {{
        background: {theme_config['card_bg']};
        border: 1px solid {theme_config['border_color']};
        border-top: none;
        border-radius: 0 0 8px 8px;
    }}
    
    /* ============================================
       DATAFRAME
    ============================================ */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid {theme_config['border_color']};
    }}
    
    /* ============================================
       TABS
    ============================================ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {theme_config['card_bg']};
        border-radius: 8px;
        padding: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 6px;
        color: {text_muted};
        padding: 8px 16px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {theme_config['primary_color']} !important;
        color: white !important;
    }}
    
    /* ============================================
       FOOTER
    ============================================ */
    .footer {{
        position: fixed;
        bottom: 0;
        width: 100%;
        background: {theme_config['sidebar_bg']};
        padding: 10px;
        text-align: center;
        color: {text_muted};
        font-size: 0.8rem;
        border-top: 1px solid {theme_config['border_color']};
        z-index: 999;
    }}
    
    /* ============================================
       SCROLLBAR (pour thèmes sombres)
    ============================================ */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {theme_config['sidebar_bg']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {theme_config['primary_color']};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {theme_config['secondary_color']};
    }}
</style>
"""

    return css


import base64
from pathlib import Path


def get_logo_html(size="140px"):
    base_dir = Path(__file__).resolve().parent
    #    logo_path = (base_dir / "assets" / "logo2.JPEG").resolve()
    logo_path = (base_dir / "assets" / "logo.png").resolve()

    if not logo_path.exists():
        return f"<p style='color:red;'>Logo introuvable : {logo_path}</p>"

    encoded = base64.b64encode(logo_path.read_bytes()).decode()

    return f"""
    <div style="text-align: center; padding: 20px 0;">
        <div style="
            width: {size};
            height: {size};
            margin: 0 auto 15px auto;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
        ">
            <img src="data:image/jpeg;base64,{encoded}"
                 style="width: 100%; height: 100%; object-fit: cover; border-radius: 20px;" />
        </div>
    </div>
    """
