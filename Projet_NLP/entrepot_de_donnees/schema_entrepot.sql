-- ============================================
-- Schéma SQL - Entrepôt de Données NLP Text Mining
-- DuckDB - Modèle en Étoile (Star Schema)
-- 
-- Auteur: Projet NLP Text Mining
-- Date: Décembre 2025
-- ============================================

-- ============================================
-- DIMENSIONS
-- ============================================

-- Dimension SOURCE
CREATE TABLE dim_source (
    source_id INTEGER PRIMARY KEY,
    source_name VARCHAR(50) NOT NULL UNIQUE,  -- 'France Travail', 'Indeed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension LOCALISATION
CREATE TABLE dim_localisation (
    localisation_id INTEGER PRIMARY KEY,
    city VARCHAR(100),
    department VARCHAR(10),
    region VARCHAR(100),
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6),
    location_raw VARCHAR(200),  -- Localisation brute originale
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(city, department)
);

-- Dimension ENTREPRISE
CREATE TABLE dim_entreprise (
    entreprise_id INTEGER PRIMARY KEY,
    company_name VARCHAR(200) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dimension CONTRAT
CREATE TABLE dim_contrat (
    contrat_id INTEGER PRIMARY KEY,
    contract_type VARCHAR(50),      -- CDI, CDD, Stage, Alternance, Freelance
    duration VARCHAR(100),           -- "35H/semaine"
    experience_level VARCHAR(10),    -- D (débutant), E (expérimenté), S (senior)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(contract_type, duration, experience_level)
);

-- Dimension TEMPS
CREATE TABLE dim_temps (
    temps_id INTEGER PRIMARY KEY,
    date_posted DATE NOT NULL,
    year INTEGER,
    month INTEGER,
    week INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date_posted)
);


-- ============================================
-- TABLE DE FAITS PRINCIPALE
-- ============================================

CREATE TABLE fact_offres (
    -- Clé primaire
    offre_id INTEGER PRIMARY KEY,
    
    -- Clés étrangères (dimensions)
    source_id INTEGER REFERENCES dim_source(source_id),
    localisation_id INTEGER REFERENCES dim_localisation(localisation_id),
    entreprise_id INTEGER REFERENCES dim_entreprise(entreprise_id),
    contrat_id INTEGER REFERENCES dim_contrat(contrat_id),
    temps_id INTEGER REFERENCES dim_temps(temps_id),
    
    -- Identifiant source
    job_id_source VARCHAR(100) NOT NULL,  -- ID original de la source
    
    -- Mesures numériques
    salary_min DECIMAL(10, 2),            -- Salaire minimum annuel en €
    salary_max DECIMAL(10, 2),            -- Salaire maximum annuel en €
    
    -- Informations textuelles
    title TEXT NOT NULL,                   -- Titre du poste
    description TEXT,                      -- Description complète (pour NLP)
    
    -- NOUVEAU : Champs texte bruts pour flexibilité
    salary_text VARCHAR(200),              -- Texte salaire brut
    location_text VARCHAR(200),            -- Localisation texte brut
    duration_text VARCHAR(200),            -- Durée texte (ex: "35H/semaine")
    
    -- Métadonnées
    url TEXT,                              -- URL de l'offre
    scraped_at TIMESTAMP NOT NULL,         -- Date de scraping
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- NOUVEAU : Source data (JSON brut optionnel pour traçabilité)
    source_data_json TEXT,                 -- JSON original (optionnel)
    
    -- Contrainte d'unicité
    UNIQUE(source_id, job_id_source)
);


-- ============================================
-- TABLE DE FAITS - COMPÉTENCES
-- ============================================

CREATE TABLE fact_competences (
    competence_id INTEGER PRIMARY KEY,
    offre_id INTEGER REFERENCES fact_offres(offre_id),
    
    skill_code VARCHAR(20),                -- Code compétence (France Travail)
    skill_label TEXT NOT NULL,             -- Nom de la compétence
    skill_level VARCHAR(10),               -- E (exigée), S (souhaitée)
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(offre_id, skill_label)
);


-- ============================================
-- INDEX POUR PERFORMANCE
-- ============================================

-- Index sur clés étrangères
CREATE INDEX idx_offres_source ON fact_offres(source_id);
CREATE INDEX idx_offres_localisation ON fact_offres(localisation_id);
CREATE INDEX idx_offres_entreprise ON fact_offres(entreprise_id);
CREATE INDEX idx_offres_contrat ON fact_offres(contrat_id);
CREATE INDEX idx_offres_temps ON fact_offres(temps_id);

-- Index sur compétences
CREATE INDEX idx_competences_offre ON fact_competences(offre_id);
CREATE INDEX idx_competences_label ON fact_competences(skill_label);

-- Index pour recherches textuelles
CREATE INDEX idx_offres_title ON fact_offres(title);
CREATE INDEX idx_entreprise_name ON dim_entreprise(company_name);

-- Index géographique
CREATE INDEX idx_localisation_city ON dim_localisation(city);
CREATE INDEX idx_localisation_region ON dim_localisation(region);


-- ============================================
-- VUES UTILES
-- ============================================

-- Vue complète des offres avec toutes les dimensions
CREATE VIEW v_offres_complete AS
SELECT 
    o.offre_id,
    o.job_id_source,
    s.source_name,
    o.title,
    e.company_name,
    l.city,
    l.department,
    l.region,
    l.latitude,
    l.longitude,
    c.contract_type,
    c.experience_level,
    o.salary_min,
    o.salary_max,
    t.date_posted,
    o.description,
    o.url,
    o.scraped_at
FROM fact_offres o
LEFT JOIN dim_source s ON o.source_id = s.source_id
LEFT JOIN dim_localisation l ON o.localisation_id = l.localisation_id
LEFT JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
LEFT JOIN dim_contrat c ON o.contrat_id = c.contrat_id
LEFT JOIN dim_temps t ON o.temps_id = t.temps_id;


-- Vue des offres avec compétences
CREATE VIEW v_offres_competences AS
SELECT 
    o.offre_id,
    o.title,
    e.company_name,
    c.skill_label,
    c.skill_level,
    s.source_name
FROM fact_offres o
LEFT JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
LEFT JOIN fact_competences c ON o.offre_id = c.offre_id
LEFT JOIN dim_source s ON o.source_id = s.source_id
WHERE c.skill_label IS NOT NULL;


-- Vue statistiques par région
CREATE VIEW v_stats_region AS
SELECT 
    l.region,
    COUNT(o.offre_id) as nb_offres,
    AVG(o.salary_min) as salaire_moyen_min,
    AVG(o.salary_max) as salaire_moyen_max,
    COUNT(DISTINCT o.entreprise_id) as nb_entreprises
FROM fact_offres o
LEFT JOIN dim_localisation l ON o.localisation_id = l.localisation_id
GROUP BY l.region
ORDER BY nb_offres DESC;


-- Vue statistiques par type de contrat
CREATE VIEW v_stats_contrat AS
SELECT 
    c.contract_type,
    c.experience_level,
    COUNT(o.offre_id) as nb_offres,
    AVG(o.salary_min) as salaire_moyen_min,
    AVG(o.salary_max) as salaire_moyen_max
FROM fact_offres o
LEFT JOIN dim_contrat c ON o.contrat_id = c.contrat_id
GROUP BY c.contract_type, c.experience_level
ORDER BY nb_offres DESC;


-- ============================================
-- INSERTIONS INITIALES
-- ============================================

-- Sources
INSERT INTO dim_source (source_id, source_name) VALUES 
    (1, 'France Travail'),
    (2, 'Indeed');

-- Valeur par défaut pour données manquantes
INSERT INTO dim_localisation (localisation_id, city, department, region, location_raw) 
VALUES (1, 'Non spécifié', NULL, NULL, 'Non spécifié');

INSERT INTO dim_entreprise (entreprise_id, company_name) 
VALUES (1, 'Non spécifié');

INSERT INTO dim_contrat (contrat_id, contract_type, duration, experience_level) 
VALUES (1, 'Non spécifié', NULL, NULL);

INSERT INTO dim_temps (temps_id, date_posted, year, month, week, day_of_week, is_weekend) 
VALUES (1, '1900-01-01', 1900, 1, 1, 1, false);