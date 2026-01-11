# ==================================================
# MULTI-STAGE BUILD - Application Streamlit NLP
# ==================================================

# --------------------------------------------------
# Stage 1: Builder - Installation dépendances
# --------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /build

# Dépendances système pour compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements (root + app_streamlit)
COPY requirements.txt ./requirements.txt
COPY app_streamlit/requirements.txt ./app_streamlit_requirements.txt

# Fusionner les deux fichiers requirements avec newline explicite entre eux
RUN cat requirements.txt > all_requirements.txt && \
    echo "" >> all_requirements.txt && \
    cat app_streamlit_requirements.txt >> all_requirements.txt && \
    echo "Merged requirements:" && cat all_requirements.txt | grep -i mistral

# Installer packages Python
RUN pip install --user --no-cache-dir -r all_requirements.txt

# Pré-télécharger modèles SentenceTransformer (pour offline)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# --------------------------------------------------
# Stage 2: Runtime - Image finale légère
# --------------------------------------------------
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Master SISE"
LABEL description="Application NLP Text Mining avec PostgreSQL embarqué"

WORKDIR /app

# Dépendances runtime uniquement
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Créer utilisateur non-root AVANT de copier les packages
RUN useradd -m -u 1000 streamlit

# Copier packages Python du builder vers home de streamlit
COPY --from=builder --chown=streamlit:streamlit /root/.local /home/streamlit/.local
ENV PATH=/home/streamlit/.local/bin:$PATH

# Copier code application
COPY --chown=streamlit:streamlit app_streamlit/ ./app_streamlit/
COPY --chown=streamlit:streamlit analyses_nlp/ ./analyses_nlp/
COPY --chown=streamlit:streamlit resultats_nlp/ ./resultats_nlp/

# Copier scripts utilitaires
COPY --chown=streamlit:streamlit matching.py nouvelle_offre.py scraper.py ./

# Créer répertoires nécessaires
RUN mkdir -p \
    /app/resultats_nlp \
    /app/analyses_nlp/resultats_nlp \
    /app/analyses_nlp/fichiers_analyses && \
    chown -R streamlit:streamlit /app

# Configuration Streamlit
RUN mkdir -p /home/streamlit/.streamlit && \
    chown -R streamlit:streamlit /home/streamlit/.streamlit

RUN echo '\
[server]\n\
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
primaryColor = "#FF6B6B"\n\
backgroundColor = "#0E1117"\n\
secondaryBackgroundColor = "#262730"\n\
textColor = "#FAFAFA"\n\
' > /home/streamlit/.streamlit/config.toml && \
    chown streamlit:streamlit /home/streamlit/.streamlit/config.toml

# Port Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Basculer sur utilisateur non-root
USER streamlit

# Démarrage application
ENTRYPOINT ["streamlit", "run"]
CMD ["app_streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
