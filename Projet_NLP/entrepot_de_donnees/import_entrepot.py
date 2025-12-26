"""
Script d'Import - Entrepôt de Données NLP
Importe Indeed + France Travail dans DuckDB

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import duckdb
import json
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, Optional, Tuple

# Import géocodage
try:
    import sys
    sys.path.insert(0, 'scraping')
    from villes_france import geocode_location
except:
    print("⚠️  Module villes_france non trouvé")
    geocode_location = None


class EntrepotImporter:
    """
    Import des données dans l'entrepôt DuckDB
    """
    
    def __init__(self, db_path: str = "entrepot_nlp.duckdb"):
        """
        Initialise la connexion DuckDB
        
        Args:
            db_path: Chemin vers la base DuckDB
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
        print(f"✅ Connexion DuckDB: {db_path}")
        
        # Compteurs pour les IDs
        self.next_offre_id = 1
        self.next_localisation_id = 2  # 1 = "Non spécifié"
        self.next_entreprise_id = 2
        self.next_contrat_id = 2
        self.next_temps_id = 2
        self.next_competence_id = 1
        
        # Caches pour éviter les doublons
        self.cache_localisation = {}
        self.cache_entreprise = {}
        self.cache_contrat = {}
        self.cache_temps = {}
    
    def create_schema(self, schema_file: str = "schema_entrepot.sql"):
        """
        Crée le schéma de l'entrepôt
        
        Args:
            schema_file: Fichier SQL contenant le schéma
        """
        print(f"\n📋 Création du schéma...")
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            sql_schema = f.read()
        
        # Exécuter le schéma
        try:
            self.conn.execute(sql_schema)
            print("✅ Schéma créé avec succès")
        except Exception as e:
            print(f"⚠️  Schéma déjà existant ou erreur: {e}")
    
    def _get_or_create_localisation(self, city: str, department: str, 
                                     region: str, latitude: float, 
                                     longitude: float, location_raw: str) -> int:
        """
        Récupère ou crée une localisation
        
        Returns:
            localisation_id
        """
        # Géocodage si manquant
        if not city or not latitude:
            if geocode_location and location_raw:
                lat, lon, city_geo, dept, reg = geocode_location(location_raw)
                if city_geo:
                    city = city_geo
                    latitude = lat
                    longitude = lon
                    department = dept
                    region = reg
        
        # Clé de cache
        cache_key = f"{city}_{department}"
        
        if cache_key in self.cache_localisation:
            return self.cache_localisation[cache_key]
        
        # Vérifier si existe déjà
        result = self.conn.execute("""
            SELECT localisation_id FROM dim_localisation 
            WHERE city = ? AND department = ?
        """, [city, department]).fetchone()
        
        if result:
            loc_id = result[0]
        else:
            # Créer nouvelle localisation
            loc_id = self.next_localisation_id
            self.conn.execute("""
                INSERT INTO dim_localisation 
                (localisation_id, city, department, region, latitude, longitude, location_raw)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [loc_id, city, department, region, latitude, longitude, location_raw])
            
            self.next_localisation_id += 1
        
        self.cache_localisation[cache_key] = loc_id
        return loc_id
    
    def _get_or_create_entreprise(self, company_name: str) -> int:
        """Récupère ou crée une entreprise"""
        if not company_name or company_name == "N/A":
            return 1  # ID par défaut
        
        if company_name in self.cache_entreprise:
            return self.cache_entreprise[company_name]
        
        result = self.conn.execute("""
            SELECT entreprise_id FROM dim_entreprise WHERE company_name = ?
        """, [company_name]).fetchone()
        
        if result:
            ent_id = result[0]
        else:
            ent_id = self.next_entreprise_id
            self.conn.execute("""
                INSERT INTO dim_entreprise (entreprise_id, company_name) VALUES (?, ?)
            """, [ent_id, company_name])
            self.next_entreprise_id += 1
        
        self.cache_entreprise[company_name] = ent_id
        return ent_id
    
    def _get_or_create_contrat(self, contract_type: str, duration: str, 
                                experience_level: str) -> int:
        """Récupère ou crée un contrat"""
        cache_key = f"{contract_type}_{duration}_{experience_level}"
        
        if cache_key in self.cache_contrat:
            return self.cache_contrat[cache_key]
        
        result = self.conn.execute("""
            SELECT contrat_id FROM dim_contrat 
            WHERE contract_type = ? AND duration IS NOT DISTINCT FROM ? 
            AND experience_level IS NOT DISTINCT FROM ?
        """, [contract_type, duration, experience_level]).fetchone()
        
        if result:
            cont_id = result[0]
        else:
            cont_id = self.next_contrat_id
            self.conn.execute("""
                INSERT INTO dim_contrat (contrat_id, contract_type, duration, experience_level)
                VALUES (?, ?, ?, ?)
            """, [cont_id, contract_type, duration, experience_level])
            self.next_contrat_id += 1
        
        self.cache_contrat[cache_key] = cont_id
        return cont_id
    
    def _get_or_create_temps(self, date_posted: str) -> int:
        """Récupère ou crée une date"""
        if not date_posted:
            return 1  # ID par défaut
        
        if date_posted in self.cache_temps:
            return self.cache_temps[date_posted]
        
        result = self.conn.execute("""
            SELECT temps_id FROM dim_temps WHERE date_posted = ?
        """, [date_posted]).fetchone()
        
        if result:
            temps_id = result[0]
        else:
            # Parser la date
            try:
                dt = datetime.fromisoformat(date_posted.replace('Z', '+00:00'))
                
                temps_id = self.next_temps_id
                self.conn.execute("""
                    INSERT INTO dim_temps 
                    (temps_id, date_posted, year, month, week, day_of_week, is_weekend)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [
                    temps_id,
                    dt.date(),
                    dt.year,
                    dt.month,
                    dt.isocalendar()[1],  # week
                    dt.weekday(),
                    dt.weekday() >= 5  # weekend
                ])
                
                self.next_temps_id += 1
            except:
                return 1  # Date invalide → défaut
        
        self.cache_temps[date_posted] = temps_id
        return temps_id
    
    def _parse_salary_france_travail(self, salary_text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse le salaire France Travail
        Ex: "Mensuel de 3166.0 Euros sur 12.0 mois"
        """
        if not salary_text:
            return None, None
        
        try:
            # Pattern: Mensuel de X Euros sur Y mois
            match = re.search(r'Mensuel de ([\d.]+) Euros sur ([\d.]+) mois', salary_text)
            if match:
                mensuel = float(match.group(1))
                mois = float(match.group(2))
                annuel = mensuel * mois
                return annuel, annuel
            
            # Pattern: De X € à Y € par an/mois
            match2 = re.search(r'De ([\d ]+)\s*€ à ([\d ]+)\s*€', salary_text)
            if match2:
                min_sal = float(match2.group(1).replace(' ', ''))
                max_sal = float(match2.group(2).replace(' ', ''))
                
                if 'mois' in salary_text.lower() and 'par an' not in salary_text.lower():
                    min_sal *= 12
                    max_sal *= 12
                elif 'heure' in salary_text.lower():
                    min_sal *= 35 * 52
                    max_sal *= 35 * 52
                
                return min_sal, max_sal
            
            # Pattern: Mensuel de X Euros à Y Euros sur Z mois
            match3 = re.search(r'Mensuel de ([\d.]+) Euros à ([\d.]+) Euros sur ([\d.]+) mois', salary_text)
            if match3:
                min_mensuel = float(match3.group(1))
                max_mensuel = float(match3.group(2))
                mois = float(match3.group(3))
                return min_mensuel * mois, max_mensuel * mois
            
        except:
            pass
        
        return None, None
    
    def _parse_salary_indeed(self, salary_text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse le salaire Indeed
        Ex: "De 40 000 € à 55 000 € par an"
        """
        if not salary_text:
            return None, None
        
        try:
            # Pattern: "De X € à Y € par an/mois"
            pattern = r'De ([\d\s]+)\s*€ à ([\d\s]+)\s*€'
            match = re.search(pattern, salary_text)
            
            if match:
                min_sal = float(match.group(1).replace(' ', ''))
                max_sal = float(match.group(2).replace(' ', ''))
                
                # Convertir en annuel si mensuel
                if 'mois' in salary_text.lower() and 'par an' not in salary_text.lower():
                    min_sal *= 12
                    max_sal *= 12
                elif 'heure' in salary_text.lower():
                    min_sal *= 35 * 52
                    max_sal *= 35 * 52
                
                return min_sal, max_sal
            
            # Pattern 2: "X€ - Y€"
            pattern2 = r'([\d\s]+)\s*€\s*-\s*([\d\s]+)\s*€'
            match2 = re.search(pattern2, salary_text)
            
            if match2:
                min_sal = float(match2.group(1).replace(' ', ''))
                max_sal = float(match2.group(2).replace(' ', ''))
                
                if 'mois' in salary_text.lower():
                    min_sal *= 12
                    max_sal *= 12
                
                return min_sal, max_sal
            
        except:
            pass
        
        return None, None
        """
        Parse le salaire France Travail
        Ex: "Mensuel de 3166.0 Euros sur 12.0 mois"
        """
        if not salary_text:
            return None, None
        
        try:
            # Pattern: Mensuel de X Euros sur Y mois
            match = re.search(r'Mensuel de ([\d.]+) Euros sur ([\d.]+) mois', salary_text)
            if match:
                mensuel = float(match.group(1))
                mois = float(match.group(2))
                annuel = mensuel * mois
                return annuel, annuel
            
            # Pattern: De X € à Y € par an/mois
            match2 = re.search(r'De ([\d ]+)\s*€ à ([\d ]+)\s*€', salary_text)
            if match2:
                min_sal = float(match2.group(1).replace(' ', ''))
                max_sal = float(match2.group(2).replace(' ', ''))
                
                if 'mois' in salary_text.lower():
                    min_sal *= 12
                    max_sal *= 12
                
                return min_sal, max_sal
            
        except:
            pass
        
        return None, None
    
    def import_france_travail(self, json_file: str):
        """
        Importe les offres France Travail
        
        Args:
            json_file: Fichier JSON France Travail
        """
        print(f"\n📥 Import France Travail: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            offres = json.load(f)
        
        source_id = 1  # France Travail
        imported = 0
        errors = 0
        skipped = 0
        
        for offre in offres:
            try:
                job_id = offre.get('job_id', '')
                
                # Vérifier si existe déjà
                existing = self.conn.execute("""
                    SELECT offre_id FROM fact_offres 
                    WHERE source_id = ? AND job_id_source = ?
                """, [source_id, job_id]).fetchone()
                
                if existing:
                    skipped += 1
                    continue
                
                # Dimensions
                loc_id = self._get_or_create_localisation(
                    offre.get('location', 'N/A'),
                    None,  # department pas dans FT
                    None,  # region pas dans FT
                    offre.get('latitude'),
                    offre.get('longitude'),
                    offre.get('location', '')
                )
                
                ent_id = self._get_or_create_entreprise(offre.get('company', 'N/A'))
                
                cont_id = self._get_or_create_contrat(
                    offre.get('contract_type'),
                    offre.get('duration'),
                    offre.get('experience')
                )
                
                temps_id = self._get_or_create_temps(offre.get('date_posted'))
                
                # Salaire
                salary_min, salary_max = self._parse_salary_france_travail(offre.get('salary', ''))
                
                # Insertion offre
                offre_id = self.next_offre_id
                self.conn.execute("""
                    INSERT INTO fact_offres 
                    (offre_id, source_id, localisation_id, entreprise_id, contrat_id, temps_id,
                     job_id_source, salary_min, salary_max, title, description, salary_text,
                     location_text, duration_text, url, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    offre_id, source_id, loc_id, ent_id, cont_id, temps_id,
                    offre.get('job_id', ''),
                    salary_min, salary_max,
                    offre.get('title', ''),
                    offre.get('description', ''),
                    offre.get('salary', ''),
                    offre.get('location', ''),  # location_text
                    offre.get('duration', ''),  # duration_text
                    offre.get('url', ''),
                    offre.get('scraped_at', datetime.now().isoformat())
                ])
                
                # Compétences
                if offre.get('skills'):
                    for skill in offre['skills']:
                        self.conn.execute("""
                            INSERT INTO fact_competences 
                            (competence_id, offre_id, skill_code, skill_label, skill_level)
                            VALUES (?, ?, ?, ?, ?)
                        """, [
                            self.next_competence_id,
                            offre_id,
                            skill.get('code'),
                            skill.get('libelle', ''),
                            skill.get('exigence', '')
                        ])
                        self.next_competence_id += 1
                
                self.next_offre_id += 1
                imported += 1
                
            except Exception as e:
                errors += 1
                print(f"   ❌ Erreur offre {offre.get('job_id', '?')}: {str(e)[:50]}")
                continue
        
        print(f"✅ France Travail: {imported} offres importées, {errors} erreurs, {skipped} doublons ignorés")
    
    def import_indeed(self, json_file: str):
        """
        Importe les offres Indeed
        
        Args:
            json_file: Fichier JSON Indeed
        """
        print(f"\n📥 Import Indeed: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            offres = json.load(f)
        
        source_id = 2  # Indeed
        imported = 0
        errors = 0
        skipped = 0
        
        for offre in offres:
            try:
                job_id = offre.get('job_id', '')
                
                # Vérifier si existe déjà (éviter doublons)
                existing = self.conn.execute("""
                    SELECT offre_id FROM fact_offres 
                    WHERE source_id = ? AND job_id_source = ?
                """, [source_id, job_id]).fetchone()
                
                if existing:
                    skipped += 1
                    continue
                
                # Dimensions
                loc_id = self._get_or_create_localisation(
                    offre.get('city', 'N/A'),
                    offre.get('department'),
                    offre.get('region'),
                    offre.get('latitude'),
                    offre.get('longitude'),
                    offre.get('location', '')
                )
                
                ent_id = self._get_or_create_entreprise(offre.get('company', 'N/A'))
                
                # Indeed n'a pas toujours experience
                cont_id = self._get_or_create_contrat(
                    offre.get('contract_type'),
                    None,  # duration pas dans Indeed
                    None   # experience pas dans Indeed
                )
                
                # Date = scraped_at car Indeed n'a pas date_posted
                temps_id = self._get_or_create_temps(offre.get('scraped_at'))
                
                # Salaire - Parser depuis salary_text si salary_min/max sont NULL
                salary_min = offre.get('salary_min')
                salary_max = offre.get('salary_max')
                
                if not salary_min and not salary_max and offre.get('salary_text'):
                    salary_min, salary_max = self._parse_salary_indeed(offre.get('salary_text', ''))
                
                # Insertion offre
                offre_id = self.next_offre_id
                self.conn.execute("""
                    INSERT INTO fact_offres 
                    (offre_id, source_id, localisation_id, entreprise_id, contrat_id, temps_id,
                     job_id_source, salary_min, salary_max, title, description, salary_text,
                     location_text, duration_text, url, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    offre_id, source_id, loc_id, ent_id, cont_id, temps_id,
                    offre.get('job_id', ''),
                    salary_min, salary_max,
                    offre.get('title', ''),
                    offre.get('description', ''),
                    offre.get('salary_text', ''),
                    offre.get('location', ''),  # location_text
                    None,  # duration_text (pas dans Indeed)
                    offre.get('url', ''),
                    offre.get('scraped_at', datetime.now().isoformat())
                ])
                
                self.next_offre_id += 1
                imported += 1
                
            except Exception as e:
                errors += 1
                print(f"   ❌ Erreur offre {offre.get('job_id', '?')}: {str(e)[:50]}")
                continue
        
        print(f"✅ Indeed: {imported} offres importées, {errors} erreurs, {skipped} doublons ignorés")
    
    def get_stats(self):
        """Affiche les statistiques de l'entrepôt"""
        print(f"\n📊 STATISTIQUES ENTREPÔT")
        print("="*70)
        
        # Total offres
        total = self.conn.execute("SELECT COUNT(*) FROM fact_offres").fetchone()[0]
        print(f"Total offres: {total}")
        
        # Par source
        print(f"\n📌 Par source:")
        result = self.conn.execute("""
            SELECT s.source_name, COUNT(o.offre_id) as nb
            FROM fact_offres o
            JOIN dim_source s ON o.source_id = s.source_id
            GROUP BY s.source_name
        """).fetchall()
        for source, nb in result:
            print(f"  {source:<20}: {nb}")
        
        # Par région
        print(f"\n🗺️  Top 10 régions:")
        result = self.conn.execute("""
            SELECT l.region, COUNT(o.offre_id) as nb
            FROM fact_offres o
            JOIN dim_localisation l ON o.localisation_id = l.localisation_id
            WHERE l.region IS NOT NULL
            GROUP BY l.region
            ORDER BY nb DESC
            LIMIT 10
        """).fetchall()
        for region, nb in result:
            print(f"  {region:<30}: {nb}")
        
        # Par type de contrat
        print(f"\n📝 Par type de contrat:")
        result = self.conn.execute("""
            SELECT c.contract_type, COUNT(o.offre_id) as nb
            FROM fact_offres o
            JOIN dim_contrat c ON o.contrat_id = c.contrat_id
            WHERE c.contract_type IS NOT NULL
            GROUP BY c.contract_type
            ORDER BY nb DESC
        """).fetchall()
        for contrat, nb in result:
            print(f"  {contrat:<20}: {nb}")
        
        # Compétences
        nb_comp = self.conn.execute("SELECT COUNT(*) FROM fact_competences").fetchone()[0]
        print(f"\n🎓 Total compétences: {nb_comp}")
        
        if nb_comp > 0:
            print(f"\n🏆 Top 10 compétences:")
            result = self.conn.execute("""
                SELECT skill_label, COUNT(*) as nb
                FROM fact_competences
                GROUP BY skill_label
                ORDER BY nb DESC
                LIMIT 10
            """).fetchall()
            for skill, nb in result:
                print(f"  {skill:<40}: {nb}")
    
    def close(self):
        """Ferme la connexion"""
        self.conn.close()
        print(f"\n🔒 Connexion fermée")


# ============================================
# SCRIPT PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("="*70)
    print("🏗️  IMPORT ENTREPÔT NLP TEXT MINING")
    print("="*70)
    
    # Initialiser
    importer = EntrepotImporter("entrepot_nlp.duckdb")
    
    try:
        # Créer schéma
        importer.create_schema("schema_entrepot.sql")
        
        # Importer France Travail
        ft_file = input("\n📁 Fichier JSON France Travail (ou Enter pour passer): ").strip()
        if ft_file and Path(ft_file).exists():
            importer.import_france_travail(ft_file)
        
        # Importer Indeed
        indeed_file = input("\n📁 Fichier JSON Indeed (ou Enter pour passer): ").strip()
        if indeed_file and Path(indeed_file).exists():
            importer.import_indeed(indeed_file)
        
        # Stats
        importer.get_stats()
        
        print(f"\n✅ IMPORT TERMINÉ !")
        print(f"📂 Base de données: entrepot_nlp.duckdb")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        importer.close()