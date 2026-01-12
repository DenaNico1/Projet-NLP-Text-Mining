#!/bin/bash

# ============================================
# SCRIPT BUILD & PUSH DOCKER HUB
# DataJobs Explorer
# ============================================

set -e  # Arrêter si erreur

# Variables
IMAGE_NAME="nicodena/datajobs-explorer"
VERSION="1.0.0"
LATEST="latest"

echo "🐳 BUILD & PUSH DOCKER IMAGE"
echo "=============================="
echo ""

# ============================================
# 1. BUILD IMAGE
# ============================================

echo "📦 Étape 1/4 : Build image Docker..."
docker build \
    --platform linux/amd64 \
    -t ${IMAGE_NAME}:${VERSION} \
    -t ${IMAGE_NAME}:${LATEST} \
    .

echo "✅ Build terminé !"
echo ""

# ============================================
# 2. TEST LOCAL
# ============================================

echo "🧪 Étape 2/4 : Test image locale..."

# Lancer conteneur test
docker run -d \
    --name datajobs-test \
    -p 8501:8501 \
    -e SUPABASE_URL="${SUPABASE_URL}" \
    -e SUPABASE_KEY="${SUPABASE_KEY}" \
    ${IMAGE_NAME}:${LATEST}

echo "⏳ Attente démarrage (30 sec)..."
sleep 30

# Test healthcheck
if docker ps | grep -q datajobs-test; then
    echo "✅ Conteneur test OK !"
else
    echo "❌ Erreur conteneur test"
    docker logs datajobs-test
    exit 1
fi

# Nettoyer conteneur test
docker rm -f datajobs-test
echo ""

# ============================================
# 3. PUSH DOCKER HUB
# ============================================

echo "📤 Étape 3/4 : Push vers Docker Hub..."

# Login Docker Hub (si pas déjà connecté)
if ! docker info | grep -q "Username"; then
    echo "🔐 Login Docker Hub requis..."
    docker login
fi

# Push version
echo "📤 Push version ${VERSION}..."
docker push ${IMAGE_NAME}:${VERSION}

# Push latest
echo "📤 Push latest..."
docker push ${IMAGE_NAME}:${LATEST}

echo "✅ Push terminé !"
echo ""

# ============================================
# 4. EXPORT TAR (optionnel)
# ============================================

echo "💾 Étape 4/4 : Export image .tar..."
docker save ${IMAGE_NAME}:${LATEST} -o datajobs-explorer.tar
echo "✅ Fichier créé : datajobs-explorer.tar ($(du -h datajobs-explorer.tar | cut -f1))"
echo ""

# ============================================
# RÉSUMÉ
# ============================================

echo "=============================="
echo "✅ DÉPLOIEMENT TERMINÉ !"
echo "=============================="
echo ""
echo "📊 Informations image :"
docker images | grep ${IMAGE_NAME}
echo ""
echo "🔗 Docker Hub :"
echo "   https://hub.docker.com/r/${IMAGE_NAME}"
echo ""
echo "📥 Commande pull :"
echo "   docker pull ${IMAGE_NAME}:${LATEST}"
echo ""
echo "🚀 Commande run :"
echo "   docker run -d -p 8501:8501 --env-file .env ${IMAGE_NAME}:${LATEST}"
echo ""
echo "💾 Fichier tar disponible :"
echo "   datajobs-explorer.tar"
echo ""