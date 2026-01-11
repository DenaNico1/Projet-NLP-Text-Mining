#!/bin/bash
# ==================================================
# Script de Build et Lancement Docker
# Linux / macOS / Git Bash
# ==================================================

set -e

echo "🚀 Build et déploiement Docker - Application NLP"
echo "================================================="
echo ""

# --------------------------------------------------
# 1. Vérifier Docker
# --------------------------------------------------
echo "1️⃣  Vérification Docker..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    echo "   Installez Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

docker --version
echo "✅ Docker OK"
echo ""

# --------------------------------------------------
# 2. Export Supabase → SQL
# --------------------------------------------------
echo "2️⃣  Export données Supabase..."

# Vérifier .env
if [ ! -f .env ]; then
    echo "❌ Fichier .env introuvable"
    echo "   Créez un fichier .env avec vos credentials Supabase"
    exit 1
fi

# Créer dossier docker_init
mkdir -p docker_init

# Lancer export
echo "   Export en cours (peut prendre 2-5 min)..."
python export_supabase_to_sql.py

# Vérifier fichier généré
SQL_FILE="docker_init/01_init_data.sql"
if [ ! -f "$SQL_FILE" ]; then
    echo "❌ Fichier SQL non généré"
    exit 1
fi

FILE_SIZE=$(du -h "$SQL_FILE" | cut -f1)
echo "✅ Export réussi ($FILE_SIZE)"
echo ""

# --------------------------------------------------
# 3. Build Images Docker
# --------------------------------------------------
echo "3️⃣  Build images Docker..."

docker-compose build --no-cache

echo "✅ Build terminé"
echo ""

# --------------------------------------------------
# 4. Démarrer services
# --------------------------------------------------
echo "4️⃣  Démarrage services..."
echo "   PostgreSQL + Streamlit"
echo ""

docker-compose up -d

# --------------------------------------------------
# 5. Attendre initialisation
# --------------------------------------------------
echo "5️⃣  Initialisation PostgreSQL..."
echo "   Import SQL en cours (peut prendre 3-10 min selon taille)..."

sleep 5

# Attendre healthcheck PostgreSQL
MAX_ATTEMPTS=60
ATTEMPT=0
HEALTHY=false

while [ $ATTEMPT -lt $MAX_ATTEMPTS ] && [ "$HEALTHY" = false ]; do
    ATTEMPT=$((ATTEMPT + 1))
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' nlp_postgres 2>/dev/null || echo "starting")
    
    if [ "$STATUS" = "healthy" ]; then
        HEALTHY=true
        echo "✅ PostgreSQL prêt"
    else
        echo "   Tentative $ATTEMPT/$MAX_ATTEMPTS - Status: $STATUS"
        sleep 5
    fi
done

if [ "$HEALTHY" = false ]; then
    echo "❌ PostgreSQL n'a pas démarré à temps"
    echo "   Vérifiez les logs: docker-compose logs postgres"
    exit 1
fi

echo ""

# --------------------------------------------------
# 6. Vérifier Streamlit
# --------------------------------------------------
echo "6️⃣  Vérification Streamlit..."

sleep 10

MAX_ATTEMPTS=30
ATTEMPT=0
HEALTHY=false

while [ $ATTEMPT -lt $MAX_ATTEMPTS ] && [ "$HEALTHY" = false ]; do
    ATTEMPT=$((ATTEMPT + 1))
    
    if curl -f http://localhost:8501/_stcore/health &>/dev/null; then
        HEALTHY=true
        echo "✅ Streamlit prêt"
    else
        echo "   Tentative $ATTEMPT/$MAX_ATTEMPTS..."
        sleep 5
    fi
done

if [ "$HEALTHY" = false ]; then
    echo "⚠️  Streamlit prend plus de temps que prévu"
    echo "   Vérifiez les logs: docker-compose logs streamlit"
fi

echo ""

# --------------------------------------------------
# 7. Résumé
# --------------------------------------------------
echo "================================================="
echo "✅ DÉPLOIEMENT TERMINÉ"
echo "================================================="
echo ""
echo "🌐 Application disponible:"
echo "   http://localhost:8501"
echo ""
echo "📊 Services actifs:"
docker-compose ps
echo ""
echo "📝 Commandes utiles:"
echo "   Logs:         docker-compose logs -f"
echo "   Arrêter:      docker-compose down"
echo "   Redémarrer:   docker-compose restart"
echo "   Rebuild:      docker-compose up --build -d"
echo ""
