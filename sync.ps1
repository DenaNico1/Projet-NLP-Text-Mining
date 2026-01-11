# Script de synchronisation Supabase → Docker PostgreSQL
# Projet NLP Text Mining

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "  SYNCHRONISATION SUPABASE → DOCKER POSTGRESQL" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Vérifier que Docker est démarré
Write-Host "🔍 Vérification Docker..." -ForegroundColor Yellow
$dockerRunning = docker ps 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker n'est pas démarré" -ForegroundColor Red
    Write-Host ""
    Write-Host "Démarrez Docker Desktop et relancez ce script" -ForegroundColor Yellow
    exit 1
}

# Démarrer PostgreSQL si nécessaire
Write-Host "🚀 Démarrage PostgreSQL Docker..." -ForegroundColor Yellow
docker-compose up -d postgres
Start-Sleep -Seconds 5

# Vérifier que le container est prêt
$ready = $false
$attempts = 0
while (-not $ready -and $attempts -lt 30) {
    $healthcheck = docker exec nlp_postgres pg_isready -U nlp_user -d entrepot_nlp 2>&1
    if ($LASTEXITCODE -eq 0) {
        $ready = $true
    } else {
        Write-Host "⏳ Attente PostgreSQL... ($attempts/30)" -ForegroundColor Gray
        Start-Sleep -Seconds 2
        $attempts++
    }
}

if (-not $ready) {
    Write-Host "❌ PostgreSQL ne démarre pas" -ForegroundColor Red
    Write-Host "Vérifiez les logs: docker-compose logs postgres" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ PostgreSQL prêt" -ForegroundColor Green
Write-Host ""

# Vérifier fichier .env
if (-not (Test-Path ".env")) {
    Write-Host "❌ Fichier .env introuvable" -ForegroundColor Red
    Write-Host ""
    Write-Host "Créez un fichier .env à partir du template :" -ForegroundColor Yellow
    Write-Host "  Copy-Item .env.template .env" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Puis ajoutez votre mot de passe Supabase" -ForegroundColor Yellow
    exit 1
}

# Vérifier que DB_PASSWORD est défini
$envContent = Get-Content .env -Raw
if ($envContent -notmatch 'DB_PASSWORD=\S+') {
    Write-Host "⚠️  DB_PASSWORD semble vide dans .env" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continuer quand même ? (o/N)"
    if ($continue -ne 'o' -and $continue -ne 'O') {
        exit 0
    }
}

# Lancer la synchronisation
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "🔄 Lancement de la synchronisation..." -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

python sync_supabase_to_docker.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host "✅ SYNCHRONISATION TERMINÉE" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Vérification rapide :" -ForegroundColor Cyan
    docker exec -it nlp_postgres psql -U nlp_user -d entrepot_nlp -c "SELECT COUNT(*) as total_offres FROM fact_offres;"
    Write-Host ""
    Write-Host "💡 Vous pouvez maintenant :" -ForegroundColor Yellow
    Write-Host "   1. Lancer l'app en mode local: docker-compose up" -ForegroundColor White
    Write-Host "   2. Ou forcer le mode local dans .env: USE_LOCAL_DB=true" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Erreur lors de la synchronisation" -ForegroundColor Red
    Write-Host "Vérifiez les messages d'erreur ci-dessus" -ForegroundColor Yellow
}
