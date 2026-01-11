#!/usr/bin/env pwsh
# ==================================================
# Script de Build et Lancement Docker
# Windows PowerShell
# ==================================================

Write-Host "🚀 Build et déploiement Docker - Application NLP" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# --------------------------------------------------
# 1. Vérifier Docker
# --------------------------------------------------
Write-Host "1️⃣  Vérification Docker..." -ForegroundColor Yellow

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker n'est pas installé ou non accessible" -ForegroundColor Red
    Write-Host "   Installez Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

docker --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker ne fonctionne pas correctement" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker OK" -ForegroundColor Green
Write-Host ""

# --------------------------------------------------
# 2. Export Supabase → SQL
# --------------------------------------------------
Write-Host "2️⃣  Export données Supabase..." -ForegroundColor Yellow

# Vérifier .env
if (-not (Test-Path .env)) {
    Write-Host "❌ Fichier .env introuvable" -ForegroundColor Red
    Write-Host "   Créez un fichier .env avec vos credentials Supabase" -ForegroundColor Yellow
    exit 1
}

# Créer dossier docker_init
if (-not (Test-Path docker_init)) {
    New-Item -ItemType Directory -Path docker_init | Out-Null
}

# Lancer export
Write-Host "   Export en cours (peut prendre 2-5 min)..." -ForegroundColor Cyan
python export_supabase_to_sql.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors de l'export" -ForegroundColor Red
    exit 1
}

# Vérifier fichier généré
$sqlFile = "docker_init/01_init_data.sql"
if (-not (Test-Path $sqlFile)) {
    Write-Host "❌ Fichier SQL non généré" -ForegroundColor Red
    exit 1
}

$fileSize = (Get-Item $sqlFile).Length / 1MB
Write-Host "✅ Export réussi ($([Math]::Round($fileSize, 1)) MB)" -ForegroundColor Green
Write-Host ""

# --------------------------------------------------
# 3. Build Images Docker
# --------------------------------------------------
Write-Host "3️⃣  Build images Docker..." -ForegroundColor Yellow

docker-compose build --no-cache

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors du build" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Build terminé" -ForegroundColor Green
Write-Host ""

# --------------------------------------------------
# 4. Démarrer services
# --------------------------------------------------
Write-Host "4️⃣  Démarrage services..." -ForegroundColor Yellow
Write-Host "   PostgreSQL + Streamlit" -ForegroundColor Cyan
Write-Host ""

docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur au démarrage" -ForegroundColor Red
    exit 1
}

# --------------------------------------------------
# 5. Attendre initialisation
# --------------------------------------------------
Write-Host "5️⃣  Initialisation PostgreSQL..." -ForegroundColor Yellow
Write-Host "   Import SQL en cours (peut prendre 3-10 min selon taille)..." -ForegroundColor Cyan

Start-Sleep -Seconds 5

# Attendre healthcheck PostgreSQL
$maxAttempts = 60
$attempt = 0
$healthy = $false

while ($attempt -lt $maxAttempts -and -not $healthy) {
    $attempt++
    $status = docker inspect --format='{{.State.Health.Status}}' nlp_postgres 2>$null
    
    if ($status -eq "healthy") {
        $healthy = $true
        Write-Host "✅ PostgreSQL prêt" -ForegroundColor Green
    } else {
        Write-Host "   Tentative $attempt/$maxAttempts - Status: $status" -ForegroundColor Gray
        Start-Sleep -Seconds 5
    }
}

if (-not $healthy) {
    Write-Host "❌ PostgreSQL n'a pas démarré à temps" -ForegroundColor Red
    Write-Host "   Vérifiez les logs: docker-compose logs postgres" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# --------------------------------------------------
# 6. Vérifier Streamlit
# --------------------------------------------------
Write-Host "6️⃣  Vérification Streamlit..." -ForegroundColor Yellow

Start-Sleep -Seconds 10

$maxAttempts = 30
$attempt = 0
$healthy = $false

while ($attempt -lt $maxAttempts -and -not $healthy) {
    $attempt++
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $healthy = $true
            Write-Host "✅ Streamlit prêt" -ForegroundColor Green
        }
    } catch {
        Write-Host "   Tentative $attempt/$maxAttempts..." -ForegroundColor Gray
        Start-Sleep -Seconds 5
    }
}

if (-not $healthy) {
    Write-Host "⚠️  Streamlit prend plus de temps que prévu" -ForegroundColor Yellow
    Write-Host "   Vérifiez les logs: docker-compose logs streamlit" -ForegroundColor Yellow
}

Write-Host ""

# --------------------------------------------------
# 7. Résumé
# --------------------------------------------------
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "✅ DÉPLOIEMENT TERMINÉ" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🌐 Application disponible:" -ForegroundColor Yellow
Write-Host "   http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Services actifs:" -ForegroundColor Yellow
docker-compose ps
Write-Host ""
Write-Host "📝 Commandes utiles:" -ForegroundColor Yellow
Write-Host "   Logs:         docker-compose logs -f" -ForegroundColor Gray
Write-Host "   Arrêter:      docker-compose down" -ForegroundColor Gray
Write-Host "   Redémarrer:   docker-compose restart" -ForegroundColor Gray
Write-Host "   Rebuild:      docker-compose up --build -d" -ForegroundColor Gray
Write-Host ""

# Ouvrir navigateur (optionnel)
$response = Read-Host "Ouvrir le navigateur ? (O/n)"
if ($response -ne "n" -and $response -ne "N") {
    Start-Process "http://localhost:8501"
}
