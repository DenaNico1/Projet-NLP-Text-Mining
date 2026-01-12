FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Nico DENA <nico.dena@univ-lyon2.fr>"
LABEL description="Analyse NLP marché Data/IA France"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Répertoire de travail
WORKDIR /app

# ============================================
# INSTALLATION DÉPENDANCES SYSTÈME
# ============================================

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# COPIE REQUIREMENTS ET INSTALLATION
# ============================================

# Copier requirements.txt en premier (optimisation cache Docker)
COPY requirements.txt .

# Installer dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Télécharger modèle spaCy français
RUN python -m spacy download fr_core_news_lg

# ============================================
# COPIE CODE APPLICATION
# ============================================

# Copier structure complète projet (dossiers principaux)
COPY app_streamlit/ /app/app_streamlit/
COPY analyses_nlp/ /app/analyses_nlp/
COPY resultats_nlp/ /app/resultats_nlp/
COPY scraping_donnees/ /app/scraping_donnees/

# Note: .streamlit/ et guide_utilisation.md sont optionnels
# Le Dockerfile les créera si absents

# ============================================
# CONFIGURATION STREAMLIT
# ============================================

# Créer répertoire config Streamlit
RUN mkdir -p /app/.streamlit

# Créer fichier config Streamlit
RUN echo '[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
base = "dark"\n\
primaryColor = "#667eea"\n\
backgroundColor = "#0e1117"\n\
secondaryBackgroundColor = "#1e2130"' > /app/.streamlit/config.toml

# ============================================
# PERMISSIONS ET NETTOYAGE
# ============================================

# Donner permissions lecture/exécution
RUN chmod -R 755 /app

# Créer utilisateur non-root (sécurité)
RUN useradd -m -u 1000 streamlit && \
    chown -R streamlit:streamlit /app

USER streamlit

# ============================================
# HEALTHCHECK
# ============================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ============================================
# EXPOSITION PORT
# ============================================

EXPOSE 8501

# ============================================
# POINT D'ENTRÉE
# ============================================

# Répertoire de démarrage
WORKDIR /app/app_streamlit

# Commande lancement Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

