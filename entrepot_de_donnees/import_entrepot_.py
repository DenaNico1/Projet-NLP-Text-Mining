"""
Script universel d'import dans l'entrepôt
ADAPTÉ AU SCHÉMA ANGLAIS (fact_offres, dim_entreprise.company_name, etc.)

Usage:
    python import_universal.py --source indeed --file corpus.json
    python import_universal.py --source france_travail --file corpus.json

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import duckdb
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import hashlib

# ============================================
# GESTION DES DIMENSIONS (ADAPTÉ AU SCHÉMA)
# ============================================

def get_or_create_entreprise(conn, company_name):
    """Récupère ou crée une entreprise"""
    if not company_name or company_name == 'N/A':
        return 1  # ID par défaut "Non spécifié"
    
    # Cherche si existe
    result = conn.execute("""
        SELECT entreprise_id FROM dim_entreprise WHERE company_name = ?
    """, [company_name]).fetchone()
    
    if result:
        return result[0]
    
    # Sinon crée avec ID manuel
    max_id = conn.execute("SELECT COALESCE(MAX(entreprise_id), 1) FROM dim_entreprise").fetchone()[0]
    next_id = max_id + 1
    
    conn.execute("""
        INSERT INTO dim_entreprise (entreprise_id, company_name)
        VALUES (?, ?)
    """, [next_id, company_name])
    
    return next_id


def get_or_create_localisation(conn, city, department, region, latitude, longitude, location_raw):
    """Récupère ou crée une localisation"""
    if not city or city == 'N/A':
        return 1  # ID par défaut "Non spécifié"
    
    # Cherche si existe (sur city + department)
    result = conn.execute("""
        SELECT localisation_id FROM dim_localisation 
        WHERE city = ? AND department IS NOT DISTINCT FROM ?
    """, [city, department]).fetchone()
    
    if result:
        return result[0]
    
    # Sinon crée avec ID manuel
    max_id = conn.execute("SELECT COALESCE(MAX(localisation_id), 1) FROM dim_localisation").fetchone()[0]
    next_id = max_id + 1
    
    conn.execute("""
        INSERT INTO dim_localisation (localisation_id, city, department, region, latitude, longitude, location_raw)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [next_id, city, department, region, latitude, longitude, location_raw])
    
    return next_id


def get_or_create_contrat(conn, contract_type, duration, experience_level):
    """Récupère ou crée un contrat"""
    if not contract_type:
        return 1  # ID par défaut
    
    # Cherche si existe
    result = conn.execute("""
        SELECT contrat_id FROM dim_contrat 
        WHERE contract_type = ? 
        AND duration IS NOT DISTINCT FROM ? 
        AND experience_level IS NOT DISTINCT FROM ?
    """, [contract_type, duration, experience_level]).fetchone()
    
    if result:
        return result[0]
    
    # Sinon crée
    max_id = conn.execute("SELECT COALESCE(MAX(contrat_id), 1) FROM dim_contrat").fetchone()[0]
    next_id = max_id + 1
    
    conn.execute("""
        INSERT INTO dim_contrat (contrat_id, contract_type, duration, experience_level)
        VALUES (?, ?, ?, ?)
    """, [next_id, contract_type, duration, experience_level])
    
    return next_id


def get_or_create_temps(conn, date_posted):
    """Récupère ou crée une date"""
    if not date_posted:
        return 1  # ID par défaut
    
    # Parser la date
    try:
        if isinstance(date_posted, str):
            dt = datetime.fromisoformat(date_posted.replace('Z', '+00:00'))
        else:
            dt = date_posted
        
        date_only = dt.date()
    except:
        return 1
    
    # Cherche si existe
    result = conn.execute("""
        SELECT temps_id FROM dim_temps WHERE date_posted = ?
    """, [date_only]).fetchone()
    
    if result:
        return result[0]
    
    # Sinon crée
    max_id = conn.execute("SELECT COALESCE(MAX(temps_id), 1) FROM dim_temps").fetchone()[0]
    next_id = max_id + 1
    
    year = dt.year
    month = dt.month
    week = dt.isocalendar()[1]
    day_of_week = dt.weekday()
    is_weekend = day_of_week >= 5
    
    conn.execute("""
        INSERT INTO dim_temps (temps_id, date_posted, year, month, week, day_of_week, is_weekend)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [next_id, date_only, year, month, week, day_of_week, is_weekend])
    
    return next_id


def get_source_id(conn, source_name):
    """Récupère l'ID de la source (ne crée PAS, doit exister)"""
    result = conn.execute("""
        SELECT source_id FROM dim_source WHERE source_name = ?
    """, [source_name]).fetchone()
    
    if result:
        return result[0]
    
    # Si n'existe pas, erreur
    raise ValueError(f"Source '{source_name}' n'existe pas. Sources valides: France Travail, Indeed")


# ============================================
# NORMALISATEURS PAR SOURCE
# ============================================

def normalize_indeed(offer):
    """Normalise une offre Indeed"""
    # Extraction ville
    city = offer.get('city') or offer.get('location', 'N/A')
    if city == 'N/A':
        city = 'Non spécifié'
    
    # Type contrat
    contract_type = offer.get('contract_type', 'Non spécifié')
    
    # Date
    date_posted = offer.get('posted_date') or offer.get('date_posted') or datetime.now().isoformat()
    
    return {
        'job_id_source': offer.get('job_id', ''),
        'title': offer.get('title', 'N/A'),
        'description': offer.get('description', ''),
        'company_name': offer.get('company', 'Non spécifié'),
        'city': city,
        'department': offer.get('departement_code') or offer.get('department'),
        'region': offer.get('region'),
        'latitude': offer.get('latitude'),
        'longitude': offer.get('longitude'),
        'location_raw': offer.get('location', city),
        'contract_type': contract_type,
        'duration': None,
        'experience_level': None,
        'salary_min': offer.get('salary_min'),
        'salary_max': offer.get('salary_max'),
        'salary_text': offer.get('salary'),
        'date_posted': date_posted,
        'url': offer.get('url')
    }


def normalize_france_travail(offer):
    """Normalise une offre France Travail"""
    # Extraction entreprise
    company_name = 'Non spécifié'
    if isinstance(offer.get('entreprise'), dict):
        company_name = offer['entreprise'].get('nom', 'Non spécifié')
    elif offer.get('entreprise'):
        company_name = offer['entreprise']
    
    # Extraction localisation
    lieu = offer.get('lieuTravail', {})
    city = lieu.get('libelle', 'Non spécifié')
    code_postal = lieu.get('codePostal', '')
    department = code_postal[:2] if code_postal and len(code_postal) >= 2 else None
    latitude = lieu.get('latitude')
    longitude = lieu.get('longitude')
    
    # Type contrat
    type_contrat = offer.get('typeContrat', 'Non spécifié')
    contract_mapping = {
        'CDI': 'CDI',
        'CDD': 'CDD',
        'SAI': 'CDD',
        'MIS': 'Intérim',
        'TTI': 'Intérim',
        'ALT': 'Alternance',
        'LIB': 'Freelance'
    }
    contract_type = contract_mapping.get(type_contrat, type_contrat)
    
    # Expérience
    experience = offer.get('experienceLibelle', '')
    experience_level = None
    if 'Débutant' in experience or '1 an' in experience:
        experience_level = 'D'
    elif 'Expérimenté' in experience or '2' in experience:
        experience_level = 'E'
    
    # Salaire
    salaire = offer.get('salaire', {})
    salary_text = None
    salary_min = None
    salary_max = None
    if isinstance(salaire, dict):
        salary_text = salaire.get('libelle')
    
    # Date
    date_posted = offer.get('dateCreation', datetime.now().isoformat())
    
    return {
        'job_id_source': offer.get('id', ''),
        'title': offer.get('intitule', 'N/A'),
        'description': offer.get('description', ''),
        'company_name': company_name,
        'city': city,
        'department': department,
        'region': None,  # À calculer si besoin
        'latitude': latitude,
        'longitude': longitude,
        'location_raw': city,
        'contract_type': contract_type,
        'duration': offer.get('dureeTravailLibelle'),
        'experience_level': experience_level,
        'salary_min': salary_min,
        'salary_max': salary_max,
        'salary_text': salary_text,
        'date_posted': date_posted,
        'url': offer.get('origineOffre', {}).get('urlOrigine')
    }


# Mapping source → fonction de normalisation
NORMALIZERS = {
    'indeed': normalize_indeed,
    'france_travail': normalize_france_travail,
    'france travail': normalize_france_travail,
    'ft': normalize_france_travail,
}


# ============================================
# UTILITAIRES
# ============================================

def generate_offer_hash(title, company_name, city):
    """Génère un hash unique pour détecter doublons"""
    key = f"{title}|{company_name}|{city}".lower()
    return hashlib.md5(key.encode()).hexdigest()


def extract_offers_from_json(data):
    """Extrait les offres d'un JSON (gère différents formats)"""
    if isinstance(data, list):
        return data
    
    if isinstance(data, dict):
        if 'jobs' in data:
            return data['jobs']
        elif 'offres' in data:
            return data['offres']
        elif 'resultats' in data:
            return data['resultats']
        elif 'data' in data:
            return data['data']
        else:
            return [data]
    
    return []


# ============================================
# FONCTION PRINCIPALE D'IMPORT
# ============================================

def import_data(source, json_file, db_file='entrepot_nlp.duckdb', verbose=True):
    """Importe des données dans l'entrepôt"""
    
    if verbose:
        print("="*70)
        print("📥 IMPORT UNIVERSEL - ENTREPÔT NLP")
        print("="*70)
        print(f"\n📋 Source: {source}")
        print(f"📂 Fichier: {json_file}")
        print(f"💾 Base: {db_file}")
    
    # Vérification source
    source_lower = source.lower()
    if source_lower not in NORMALIZERS:
        print(f"\n❌ Source inconnue: {source}")
        print(f"Sources disponibles: {', '.join(NORMALIZERS.keys())}")
        return False
    
    normalizer = NORMALIZERS[source_lower]
    
    # Mapping nom source → nom dans DB
    source_name_mapping = {
        'indeed': 'Indeed',
        'france_travail': 'France Travail',
        'france travail': 'France Travail',
        'ft': 'France Travail'
    }
    source_name_db = source_name_mapping.get(source_lower, source)
    
    # Connexion
    if verbose:
        print("\n✅ Connexion à la base...")
    conn = duckdb.connect(db_file)
    
    # Lecture du fichier
    if verbose:
        print("📖 Lecture du fichier...")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Erreur lecture fichier: {e}")
        return False
    
    # Extraction offres
    offers = extract_offers_from_json(data)
    
    if not offers:
        print("❌ Aucune offre trouvée dans le fichier")
        return False
    
    if verbose:
        print(f"✅ {len(offers)} offres extraites")
    
    # ID source
    try:
        source_id = get_source_id(conn, source_name_db)
        if verbose:
            print(f"✅ Source ID: {source_id}")
    except ValueError as e:
        print(f"❌ {e}")
        return False
    
    # Chargement offres existantes
    if verbose:
        print("\n📊 Vérification doublons...")
    
    existing_hashes = set()
    existing = conn.execute("""
        SELECT o.title, e.company_name, l.city
        FROM fact_offres o
        JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
        JOIN dim_localisation l ON o.localisation_id = l.localisation_id
        WHERE o.source_id = ?
    """, [source_id]).fetchall()
    
    for title, company_name, city in existing:
        existing_hashes.add(generate_offer_hash(title, company_name, city))
    
    if verbose:
        print(f"✅ {len(existing_hashes)} offres déjà en base (source: {source_name_db})")
    
    # Statistiques
    inserted = 0
    duplicates = 0
    errors = 0
    error_details = []
    
    # Import
    if verbose:
        print("\n📥 Import en cours...")
    
    for i, offer in enumerate(offers, 1):
        try:
            # Normalisation selon la source
            normalized = normalizer(offer)
            
            # Vérification doublon
            offer_hash = generate_offer_hash(
                normalized['title'],
                normalized['company_name'],
                normalized['city']
            )
            
            if offer_hash in existing_hashes:
                duplicates += 1
                if verbose and i % 100 == 0:
                    print(f"  [{i}/{len(offers)}] Insérées: {inserted}, Doublons: {duplicates}, Erreurs: {errors}")
                continue
            
            # Récupération/création entreprise
            entreprise_id = get_or_create_entreprise(conn, normalized['company_name'])
            
            # Récupération/création localisation
            localisation_id = get_or_create_localisation(
                conn,
                normalized['city'],
                normalized['department'],
                normalized['region'],
                normalized['latitude'],
                normalized['longitude'],
                normalized['location_raw']
            )
            
            # Récupération/création contrat
            contrat_id = get_or_create_contrat(
                conn,
                normalized['contract_type'],
                normalized['duration'],
                normalized['experience_level']
            )
            
            # Récupération/création temps
            temps_id = get_or_create_temps(conn, normalized['date_posted'])
            
            # Insertion offre
            max_offre_id = conn.execute("SELECT COALESCE(MAX(offre_id), 0) FROM fact_offres").fetchone()[0]
            next_offre_id = max_offre_id + 1
            
            conn.execute("""
                INSERT INTO fact_offres (
                    offre_id, source_id, localisation_id, entreprise_id, 
                    contrat_id, temps_id, job_id_source, 
                    salary_min, salary_max, title, description,
                    salary_text, location_text, duration_text,
                    url, scraped_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                next_offre_id,
                source_id,
                localisation_id,
                entreprise_id,
                contrat_id,
                temps_id,
                normalized['job_id_source'],
                normalized['salary_min'],
                normalized['salary_max'],
                normalized['title'],
                normalized['description'],
                normalized['salary_text'],
                normalized['location_raw'],
                normalized['duration'],
                normalized['url'],
                datetime.now()
            ])
            
            inserted += 1
            existing_hashes.add(offer_hash)
            
            # Affichage progression
            if verbose and i % 100 == 0:
                print(f"  [{i}/{len(offers)}] Insérées: {inserted}, Doublons: {duplicates}, Erreurs: {errors}")
        
        except Exception as e:
            errors += 1
            error_msg = str(e)[:150]
            if errors <= 5:
                error_details.append(f"Offre {i}: {error_msg}")
    
    # Statistiques finales
    if verbose:
        print(f"\n{'='*70}")
        print("📊 RÉSULTATS D'IMPORT")
        print(f"{'='*70}")
        print(f"Source             : {source_name_db}")
        print(f"Offres lues        : {len(offers)}")
        print(f"Offres insérées    : {inserted}")
        print(f"Doublons ignorés   : {duplicates}")
        print(f"Erreurs            : {errors}")
        
        if error_details:
            print("\n⚠️  Détails erreurs:")
            for err in error_details:
                print(f"  • {err}")
        
        # Vérification totaux
        total_source = conn.execute("""
            SELECT COUNT(*) FROM fact_offres WHERE source_id = ?
        """, [source_id]).fetchone()[0]
        
        total_general = conn.execute("""
            SELECT COUNT(*) FROM fact_offres
        """).fetchone()[0]
        
        print(f"\n📈 Total offres {source_name_db} en base : {total_source}")
        print(f"📈 Total général en base              : {total_general}")
        
        # Top villes
        print(f"\n📍 Top 10 villes ({source_name_db}):")
        top_cities = conn.execute("""
            SELECT l.city, COUNT(*) as total
            FROM fact_offres o
            JOIN dim_localisation l ON o.localisation_id = l.localisation_id
            WHERE o.source_id = ?
            GROUP BY l.city
            ORDER BY total DESC
            LIMIT 10
        """, [source_id]).fetchall()
        
        for i, (city, count) in enumerate(top_cities, 1):
            print(f"  {i:2d}. {city:<30}: {count:4d}")
        
        # Top entreprises
        print(f"\n🏢 Top 10 entreprises ({source_name_db}):")
        top_companies = conn.execute("""
            SELECT e.company_name, COUNT(*) as total
            FROM fact_offres o
            JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
            WHERE o.source_id = ?
            GROUP BY e.company_name
            ORDER BY total DESC
            LIMIT 10
        """, [source_id]).fetchall()
        
        for i, (company, count) in enumerate(top_companies, 1):
            print(f"  {i:2d}. {company:<30}: {count:4d}")
    
    conn.close()
    
    if verbose:
        print(f"\n{'='*70}")
        if errors == 0:
            print("✅ IMPORT TERMINÉ AVEC SUCCÈS !")
        else:
            print(f"⚠️  IMPORT TERMINÉ AVEC {errors} ERREURS")
        print(f"{'='*70}")
    
    return True


# ============================================
# LIGNE DE COMMANDE
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='Import universel dans l\'entrepôt NLP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python import_universal.py --source indeed --file corpus_indeed.json
  python import_universal.py --source france_travail --file corpus_ft.json
  python import_universal.py -s ft -f data.json --db custom.duckdb
        """
    )
    
    parser.add_argument('-s', '--source', required=True, help='Source (indeed, france_travail)')
    parser.add_argument('-f', '--file', required=True, help='Fichier JSON')
    parser.add_argument('-d', '--db', default='entrepot_nlp.duckdb', help='Base DuckDB')
    parser.add_argument('-q', '--quiet', action='store_true', help='Mode silencieux')
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"❌ Fichier introuvable: {args.file}")
        sys.exit(1)
    
    success = import_data(
        source=args.source,
        json_file=args.file,
        db_file=args.db,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()