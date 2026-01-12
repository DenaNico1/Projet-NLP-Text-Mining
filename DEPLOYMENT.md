# 🐳 GUIDE DÉPLOIEMENT DOCKER - DataJobs Explorer

**Application Streamlit d'analyse du marché Data/IA en France**

---

## 📋 Table des matières

- [Prérequis](#prérequis)
- [Option 1 : Docker Hub (Recommandé)](#option-1--docker-hub-recommandé)
- [Option 2 : Fichier .tar](#option-2--fichier-tar)
- [Option 3 : Build Local](#option-3--build-local)
- [Configuration](#configuration)
- [Lancement](#lancement)
- [Vérification](#vérification)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Prérequis

### **Logiciels requis**

- **Docker** : Version 20.10+ ([Installation](https://docs.docker.com/get-docker/))
- **Docker Compose** : Version 2.0+ (inclus avec Docker Desktop)

**Vérifier installation :**
```bash
docker --version
# Docker version 24.0.7, build afdd53b

docker compose version
# Docker Compose version v2.23.0
```

### **Configuration système minimale**

- **CPU** : 2 cores
- **RAM** : 4 GB minimum (8 GB recommandé)
- **Disque** : 5 GB libres
- **OS** : Linux, macOS, Windows 10/11 (WSL2)

---

## 🚀 Option 1 : Docker Hub (Recommandé)

### **Étape 1 : Télécharger l'image**

```bash
docker pull nicodena/datajobs-explorer:latest
```

**Taille image :** ~2.0 GB (premier téléchargement 5-10 min selon connexion)

### **Étape 2 : Créer fichier `.env`**

```bash
# Créer fichier .env
nano .env
```

**Contenu (remplacer par vos credentials) :**

```env
# Supabase
SUPABASE_URL=https://xxxxxxxxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxxx

# Mistral AI (optionnel)
MISTRAL_API_KEY=xxxxxxxxxxxxxxxxxxxxxxx
```

**💡 Obtenir credentials Supabase :**
1. Se connecter sur [supabase.com](https://supabase.com)
2. Ouvrir votre projet
3. Settings > API > Project API keys
4. Copier `URL` et `anon public` key

### **Étape 3 : Lancer conteneur**

```bash
docker run -d \
  --name datajobs-explorer \
  -p 8501:8501 \
  --env-file .env \
  --restart unless-stopped \
  nicodena/datajobs-explorer:latest
```

**Paramètres :**
- `-d` : Mode détaché (background)
- `--name` : Nom conteneur
- `-p 8501:8501` : Port mapping (hôte:conteneur)
- `--env-file` : Charger variables depuis .env
- `--restart unless-stopped` : Redémarrage automatique

### **Étape 4 : Vérifier lancement**

```bash
# Voir logs démarrage
docker logs datajobs-explorer

# Attendre ce message :
# You can now view your Streamlit app in your browser.
# URL: http://0.0.0.0:8501
```

### **Étape 5 : Accéder à l'application**

Ouvrir navigateur : **http://localhost:8501**

✅ **L'application devrait se charger en 10-15 secondes !**

---

## 📦 Option 2 : Fichier .tar

### **Étape 1 : Télécharger fichier .tar**

**Récupérer :** `datajobs-explorer.tar` (fourni séparément, ~2 GB)

### **Étape 2 : Charger image**

```bash
# Charger dans Docker local
docker load -i datajobs-explorer.tar

# Vérifier chargement
docker images | grep datajobs
# nicodena/datajobs-explorer   latest   abc123def456   2.1GB
```

### **Étape 3 : Configuration et lancement**

**Même procédure qu'Option 1 (étapes 2-5)**

---

## 🛠️ Option 3 : Build Local

### **Étape 1 : Cloner repository**

```bash
git clone https://github.com/nicodena/datajobs-explorer.git
cd datajobs-explorer
```

### **Étape 2 : Créer `.env`**

```bash
cp .env.example .env
nano .env  # Éditer avec vos credentials
```

### **Étape 3 : Build image**

```bash
docker build -t datajobs-explorer:local .
```

**Temps build :** 10-15 minutes (téléchargement dépendances)

### **Étape 4 : Lancer avec Docker Compose**

```bash
docker compose up -d
```

**OU lancement manuel :**

```bash
docker run -d \
  --name datajobs-explorer \
  -p 8501:8501 \
  --env-file .env \
  datajobs-explorer:local
```

---

## ⚙️ Configuration

### **Variables d'environnement**

| Variable | Description | Obligatoire | Exemple |
|----------|-------------|-------------|---------|
| `SUPABASE_URL` | URL projet Supabase | ✅ Oui | `https://xxx.supabase.co` |
| `SUPABASE_KEY` | Clé API publique | ✅ Oui | `eyJhbGci...` |
| `MISTRAL_API_KEY` | Clé Mistral LLM | ❌ Non | `xxxxxxxx` |
| `STREAMLIT_PORT` | Port application | ❌ Non | `8501` (défaut) |

### **Fichier `.env` complet**

```env
# Supabase (OBLIGATOIRE)
SUPABASE_URL=https://votre-projet.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxxx
SUPABASE_PASSWORD=votre_password  # Optionnel

# Mistral AI (OPTIONNEL - pour ajout offres LLM)
MISTRAL_API_KEY=xxxxxxxxxxxxxxxxxxxxxxx

# Config application
STREAMLIT_PORT=8501
DEBUG=false
LANGUAGE=fr
```

### **Changer port d'écoute**

**Si port 8501 déjà utilisé :**

```bash
docker run -d \
  --name datajobs-explorer \
  -p 8080:8501 \  # ← Changer port hôte
  --env-file .env \
  nicodena/datajobs-explorer:latest
```

**Accès :** http://localhost:8080

---

## 🎯 Lancement

### **Commandes essentielles**

```bash
# Démarrer conteneur
docker start datajobs-explorer

# Arrêter conteneur
docker stop datajobs-explorer

# Redémarrer conteneur
docker restart datajobs-explorer

# Voir logs temps réel
docker logs -f datajobs-explorer

# Supprimer conteneur
docker rm -f datajobs-explorer
```

### **Docker Compose**

```bash
# Démarrer
docker compose up -d

# Arrêter
docker compose down

# Logs
docker compose logs -f

# Rebuild + restart
docker compose up -d --build
```

---

## ✅ Vérification

### **Healthcheck**

```bash
# Vérifier état conteneur
docker ps | grep datajobs

# Status: healthy = OK
```

### **Test connexion Supabase**

```bash
# Entrer dans conteneur
docker exec -it datajobs-explorer bash

# Tester connexion
python -c "
import psycopg2
import os
conn = psycopg2.connect(os.getenv('SUPABASE_URL'))
print('✅ Connexion OK')
"
```

### **Test application**

1. Ouvrir http://localhost:8501
2. Vérifier page **Dashboard** se charge
3. Tester page **Exploration Géographique** (carte)
4. Vérifier page **Matching CV-Offres**

**✅ Si tout fonctionne → Déploiement réussi !**

---

## 🐛 Troubleshooting

### **Problème 1 : Port déjà utilisé**

**Erreur :**
```
Error: bind: address already in use
```

**Solution :**
```bash
# Trouver processus utilisant port 8501
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Tuer processus OU changer port Docker
docker run -p 8080:8501 ...
```

### **Problème 2 : Credentials invalides**

**Erreur logs :**
```
psycopg2.OperationalError: connection to server failed
```

**Solution :**
1. Vérifier `.env` (URL et KEY corrects)
2. Tester connexion Supabase (dashboard web)
3. Vérifier firewall/proxy

**Test manuel :**
```bash
curl https://votre-projet.supabase.co/rest/v1/
# Devrait retourner 404 (normal, endpoint existe)
```

### **Problème 3 : Image trop volumineuse**

**Si disque plein :**
```bash
# Nettoyer images inutilisées
docker system prune -a

# Libérer ~10-20 GB
```

### **Problème 4 : Lenteur chargement**

**Si application lente (>30 sec) :**

```bash
# Vérifier ressources allouées Docker
docker stats datajobs-explorer

# Augmenter RAM Docker Desktop :
# Settings > Resources > Memory → 8 GB
```

### **Problème 5 : Modèles NLP manquants**

**Erreur :**
```
Can't find model 'fr_core_news_lg'
```

**Solution (rebuild avec téléchargement forcé) :**
```bash
docker exec -it datajobs-explorer bash
python -m spacy download fr_core_news_lg
exit
docker restart datajobs-explorer
```

### **Problème 6 : Fichiers .pkl manquants**

**Erreur :**
```
FileNotFoundError: resultats_nlp/models/xxx.pkl
```

**Vérifier fichiers inclus :**
```bash
docker exec -it datajobs-explorer ls -la /app/resultats_nlp/models/
```

**Si vide → Rebuild image :**
```bash
docker build --no-cache -t datajobs-explorer:local .
```

---

## 📊 Performances

### **Temps de démarrage**

```
Lancement conteneur     : 2 sec
Chargement dépendances  : 3 sec
Connexion Supabase      : 1 sec
Chargement modèles NLP  : 8 sec
Lancement Streamlit     : 3 sec
─────────────────────────────
TOTAL                   : ~15-20 sec
```

### **Utilisation ressources (normal)**

```
CPU  : 5-15% (idle), 30-50% (calculs)
RAM  : 1.5-2.5 GB
Disk : 2.1 GB (image)
```

---

## 🔒 Sécurité

### **Bonnes pratiques**

✅ **NE JAMAIS commiter `.env` sur Git**
✅ **Utiliser secrets Docker** (production) :
```bash
docker secret create supabase_key supabase.key
docker service create --secret supabase_key ...
```

✅ **Limiter ressources conteneur :**
```bash
docker run --cpus="2" --memory="4g" ...
```

✅ **Activer HTTPS** (production avec reverse proxy) :
```
nginx/traefik → HTTPS → conteneur Docker
```

---

## 📞 Support

**Problème non résolu ?**

1. **Vérifier logs :**
   ```bash
   docker logs datajobs-explorer > logs.txt
   ```

2. **Créer issue GitHub :**
   https://github.com/nicodena/datajobs-explorer/issues

3. **Contact :**
   - Email : nico.dena@univ-lyon2.fr
   - LinkedIn : [linkedin.com/in/nico-dena](https://linkedin.com/in/nico-dena)

---

## 🎉 Félicitations !

**Votre application DataJobs Explorer est déployée ! 🚀**

**Prochaines étapes :**
- Explorer les 3 009 offres d'emploi
- Tester le matching CV-Offres
- Analyser les compétences et profils métiers
- Consulter les visualisations géographiques

**Happy Data Analyzing! 📊**