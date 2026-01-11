"""
Base de données des villes françaises avec coordonnées GPS
Pour géocodage Indeed
"""

# Top 100 villes de France avec coordonnées GPS
VILLES_FRANCE = {
    # Île-de-France
    "Paris": {"lat": 48.8566, "lon": 2.3522, "dept": "75", "region": "Île-de-France"},
    "Boulogne-Billancourt": {"lat": 48.8356, "lon": 2.2397, "dept": "92", "region": "Île-de-France"},
    "Saint-Denis": {"lat": 48.9362, "lon": 2.3574, "dept": "93", "region": "Île-de-France"},
    "Argenteuil": {"lat": 48.9474, "lon": 2.2466, "dept": "95", "region": "Île-de-France"},
    "Montreuil": {"lat": 48.8634, "lon": 2.4431, "dept": "93", "region": "Île-de-France"},
    "Créteil": {"lat": 48.7908, "lon": 2.4551, "dept": "94", "region": "Île-de-France"},
    "Nanterre": {"lat": 48.8925, "lon": 2.2064, "dept": "92", "region": "Île-de-France"},
    "Versailles": {"lat": 48.8014, "lon": 2.1301, "dept": "78", "region": "Île-de-France"},
    
    # Auvergne-Rhône-Alpes
    "Lyon": {"lat": 45.7640, "lon": 4.8357, "dept": "69", "region": "Auvergne-Rhône-Alpes"},
    "Grenoble": {"lat": 45.1885, "lon": 5.7245, "dept": "38", "region": "Auvergne-Rhône-Alpes"},
    "Villeurbanne": {"lat": 45.7661, "lon": 4.8795, "dept": "69", "region": "Auvergne-Rhône-Alpes"},
    "Saint-Étienne": {"lat": 45.4397, "lon": 4.3872, "dept": "42", "region": "Auvergne-Rhône-Alpes"},
    "Annecy": {"lat": 45.8992, "lon": 6.1294, "dept": "74", "region": "Auvergne-Rhône-Alpes"},
    "Clermont-Ferrand": {"lat": 45.7772, "lon": 3.0870, "dept": "63", "region": "Auvergne-Rhône-Alpes"},
    
    # Provence-Alpes-Côte d'Azur
    "Marseille": {"lat": 43.2965, "lon": 5.3698, "dept": "13", "region": "Provence-Alpes-Côte d'Azur"},
    "Nice": {"lat": 43.7102, "lon": 7.2620, "dept": "06", "region": "Provence-Alpes-Côte d'Azur"},
    "Toulon": {"lat": 43.1242, "lon": 5.9280, "dept": "83", "region": "Provence-Alpes-Côte d'Azur"},
    "Aix-en-Provence": {"lat": 43.5297, "lon": 5.4474, "dept": "13", "region": "Provence-Alpes-Côte d'Azur"},
    "Avignon": {"lat": 43.9493, "lon": 4.8055, "dept": "84", "region": "Provence-Alpes-Côte d'Azur"},
    
    # Occitanie
    "Toulouse": {"lat": 43.6047, "lon": 1.4442, "dept": "31", "region": "Occitanie"},
    "Montpellier": {"lat": 43.6108, "lon": 3.8767, "dept": "34", "region": "Occitanie"},
    "Nîmes": {"lat": 43.8367, "lon": 4.3601, "dept": "30", "region": "Occitanie"},
    "Perpignan": {"lat": 42.6886, "lon": 2.8948, "dept": "66", "region": "Occitanie"},
    "Béziers": {"lat": 43.3414, "lon": 3.2150, "dept": "34", "region": "Occitanie"},
    
    # Nouvelle-Aquitaine
    "Bordeaux": {"lat": 44.8378, "lon": -0.5792, "dept": "33", "region": "Nouvelle-Aquitaine"},
    "Limoges": {"lat": 45.8336, "lon": 1.2611, "dept": "87", "region": "Nouvelle-Aquitaine"},
    "Poitiers": {"lat": 46.5802, "lon": 0.3404, "dept": "86", "region": "Nouvelle-Aquitaine"},
    "La Rochelle": {"lat": 46.1603, "lon": -1.1511, "dept": "17", "region": "Nouvelle-Aquitaine"},
    "Pau": {"lat": 43.2951, "lon": -0.3708, "dept": "64", "region": "Nouvelle-Aquitaine"},
    
    # Hauts-de-France
    "Lille": {"lat": 50.6292, "lon": 3.0573, "dept": "59", "region": "Hauts-de-France"},
    "Amiens": {"lat": 49.8942, "lon": 2.2958, "dept": "80", "region": "Hauts-de-France"},
    "Roubaix": {"lat": 50.6942, "lon": 3.1746, "dept": "59", "region": "Hauts-de-France"},
    "Tourcoing": {"lat": 50.7242, "lon": 3.1609, "dept": "59", "region": "Hauts-de-France"},
    "Dunkerque": {"lat": 51.0343, "lon": 2.3768, "dept": "59", "region": "Hauts-de-France"},
    
    # Grand Est
    "Strasbourg": {"lat": 48.5734, "lon": 7.7521, "dept": "67", "region": "Grand Est"},
    "Reims": {"lat": 49.2583, "lon": 4.0317, "dept": "51", "region": "Grand Est"},
    "Metz": {"lat": 49.1193, "lon": 6.1757, "dept": "57", "region": "Grand Est"},
    "Mulhouse": {"lat": 47.7508, "lon": 7.3359, "dept": "68", "region": "Grand Est"},
    "Nancy": {"lat": 48.6921, "lon": 6.1844, "dept": "54", "region": "Grand Est"},
    
    # Bretagne
    "Rennes": {"lat": 48.1173, "lon": -1.6778, "dept": "35", "region": "Bretagne"},
    "Brest": {"lat": 48.3905, "lon": -4.4860, "dept": "29", "region": "Bretagne"},
    "Quimper": {"lat": 47.9960, "lon": -4.0970, "dept": "29", "region": "Bretagne"},
    "Saint-Malo": {"lat": 48.6500, "lon": -2.0260, "dept": "35", "region": "Bretagne"},
    "Lorient": {"lat": 47.7482, "lon": -3.3650, "dept": "56", "region": "Bretagne"},
    
    # Pays de la Loire
    "Nantes": {"lat": 47.2184, "lon": -1.5536, "dept": "44", "region": "Pays de la Loire"},
    "Angers": {"lat": 47.4784, "lon": -0.5632, "dept": "49", "region": "Pays de la Loire"},
    "Le Mans": {"lat": 48.0077, "lon": 0.1984, "dept": "72", "region": "Pays de la Loire"},
    "Saint-Nazaire": {"lat": 47.2730, "lon": -2.2137, "dept": "44", "region": "Pays de la Loire"},
    
    # Normandie
    "Rouen": {"lat": 49.4432, "lon": 1.0993, "dept": "76", "region": "Normandie"},
    "Le Havre": {"lat": 49.4944, "lon": 0.1079, "dept": "76", "region": "Normandie"},
    "Caen": {"lat": 49.1829, "lon": -0.3707, "dept": "14", "region": "Normandie"},
    "Cherbourg": {"lat": 49.6337, "lon": -1.6163, "dept": "50", "region": "Normandie"},
    
    # Bourgogne-Franche-Comté
    "Dijon": {"lat": 47.3220, "lon": 5.0415, "dept": "21", "region": "Bourgogne-Franche-Comté"},
    "Besançon": {"lat": 47.2380, "lon": 6.0243, "dept": "25", "region": "Bourgogne-Franche-Comté"},
    
    # Centre-Val de Loire
    "Tours": {"lat": 47.3941, "lon": 0.6848, "dept": "37", "region": "Centre-Val de Loire"},
    "Orléans": {"lat": 47.9029, "lon": 1.9093, "dept": "45", "region": "Centre-Val de Loire"},
    "Bourges": {"lat": 47.0844, "lon": 2.3964, "dept": "18", "region": "Centre-Val de Loire"},
    
    # Corse
    "Ajaccio": {"lat": 41.9267, "lon": 8.7369, "dept": "2A", "region": "Corse"},
    "Bastia": {"lat": 42.7025, "lon": 9.4501, "dept": "2B", "region": "Corse"},
    
    # Villes moyennes importantes pour tech/data
    "La Roche-sur-Yon": {"lat": 46.6708, "lon": -1.4269, "dept": "85", "region": "Pays de la Loire"},
    "Valence": {"lat": 44.9334, "lon": 4.8924, "dept": "26", "region": "Auvergne-Rhône-Alpes"},
    "Chambéry": {"lat": 45.5646, "lon": 5.9178, "dept": "73", "region": "Auvergne-Rhône-Alpes"},
    "Troyes": {"lat": 48.2973, "lon": 4.0744, "dept": "10", "region": "Grand Est"},
    "Niort": {"lat": 46.3236, "lon": -0.4595, "dept": "79", "region": "Nouvelle-Aquitaine"},
    "Angoulême": {"lat": 45.6484, "lon": 0.1562, "dept": "16", "region": "Nouvelle-Aquitaine"},
    "Périgueux": {"lat": 45.1845, "lon": 0.7213, "dept": "24", "region": "Nouvelle-Aquitaine"},
    "Bayonne": {"lat": 43.4933, "lon": -1.4748, "dept": "64", "region": "Nouvelle-Aquitaine"},
    "Mâcon": {"lat": 46.3066, "lon": 4.8286, "dept": "71", "region": "Bourgogne-Franche-Comté"},
    "Colmar": {"lat": 48.0797, "lon": 7.3579, "dept": "68", "region": "Grand Est"},
    "Épinal": {"lat": 48.1747, "lon": 6.4503, "dept": "88", "region": "Grand Est"},
    "Chartres": {"lat": 48.4469, "lon": 1.4892, "dept": "28", "region": "Centre-Val de Loire"},
    "Blois": {"lat": 47.5859, "lon": 1.3292, "dept": "41", "region": "Centre-Val de Loire"},
    "Vannes": {"lat": 47.6584, "lon": -2.7606, "dept": "56", "region": "Bretagne"},
    "Albi": {"lat": 43.9277, "lon": 2.1480, "dept": "81", "region": "Occitanie"},
    "Tarbes": {"lat": 43.2330, "lon": 0.0780, "dept": "65", "region": "Occitanie"},
    "Carcassonne": {"lat": 43.2130, "lon": 2.3491, "dept": "11", "region": "Occitanie"},
    "Montauban": {"lat": 44.0178, "lon": 1.3555, "dept": "82", "region": "Occitanie"},
    "Rodez": {"lat": 44.3498, "lon": 2.5752, "dept": "12", "region": "Occitanie"},
    "Gap": {"lat": 44.5596, "lon": 6.0799, "dept": "05", "region": "Provence-Alpes-Côte d'Azur"},
    "Digne-les-Bains": {"lat": 44.0929, "lon": 6.2361, "dept": "04", "region": "Provence-Alpes-Côte d'Azur"},
}


def get_city_info(city_name: str) -> dict:
    """
    Récupère les infos GPS d'une ville
    
    Args:
        city_name: Nom de la ville
    
    Returns:
        Dict avec lat, lon, dept, region ou None
    """
    # Normaliser le nom (enlever accents, casse)
    city_normalized = city_name.strip().title()
    
    # Recherche directe
    if city_normalized in VILLES_FRANCE:
        return VILLES_FRANCE[city_normalized]
    
    # Recherche approximative (commence par)
    for ville, info in VILLES_FRANCE.items():
        if ville.startswith(city_normalized[:3]):
            return info
    
    return None


def geocode_location(location_str: str) -> tuple:
    """
    Géocode une chaîne de localisation Indeed
    
    Args:
        location_str: Ex: "69003 Lyon 3e", "Lyon (69)", "Paris"
    
    Returns:
        (latitude, longitude, city, department, region)
    """
    import re
    
    # Extraire le nom de la ville
    # Patterns possibles:
    # "69003 Lyon 3e" → Lyon
    # "Lyon (69)" → Lyon
    # "Paris 15e Arrondissement" → Paris
    
    # Pattern 1: Code postal + Ville
    match = re.search(r'\d{5}\s+([A-Za-zÀ-ÿ-]+)', location_str)
    if match:
        city = match.group(1)
    else:
        # Pattern 2: Ville (Dept)
        match = re.search(r'([A-Za-zÀ-ÿ-]+)\s*\(', location_str)
        if match:
            city = match.group(1)
        else:
            # Pattern 3: Juste le nom
            city = location_str.split()[0] if location_str else None
    
    if not city:
        return None, None, None, None, None
    
    # Nettoyer le nom de ville
    city = city.strip().title()
    
    # Rechercher dans la base
    info = get_city_info(city)
    
    if info:
        return (
            info['lat'],
            info['lon'],
            city,
            info['dept'],
            info['region']
        )
    
    # Si pas trouvé, essayer avec Nominatim (optionnel, lent)
    return None, None, city, None, None


def get_all_cities() -> list:
    """
    Retourne la liste de toutes les villes pour collecte
    
    Returns:
        Liste des noms de villes
    """
    return list(VILLES_FRANCE.keys())


def get_cities_by_region(region: str) -> list:
    """
    Retourne les villes d'une région
    
    Args:
        region: Nom de la région
    
    Returns:
        Liste des villes
    """
    return [
        ville for ville, info in VILLES_FRANCE.items()
        if info['region'] == region
    ]


def get_top_cities(n: int = 20) -> list:
    """
    Retourne les N plus grandes villes
    (ordre actuel = ordre d'importance approximatif)
    
    Args:
        n: Nombre de villes
    
    Returns:
        Liste des N premières villes
    """
    return list(VILLES_FRANCE.keys())[:n]


if __name__ == "__main__":
    # Tests
    print("="*70)
    print("TEST GÉOCODAGE VILLES FRANÇAISES")
    print("="*70)
    
    # Test 1: Recherche directe
    print("\n1. Recherche directe:")
    info = get_city_info("Lyon")
    print(f"Lyon: {info}")
    
    # Test 2: Géocodage depuis Indeed
    print("\n2. Géocodage format Indeed:")
    test_locations = [
        "69003 Lyon 3e",
        "Lyon (69)",
        "Paris 15e Arrondissement",
        "Marseille",
        "75001 Paris"
    ]
    
    for loc in test_locations:
        lat, lon, city, dept, region = geocode_location(loc)
        print(f"{loc:30s} → {city:15s} ({lat}, {lon}) - {dept} - {region}")
    
    # Test 3: Liste des villes
    print(f"\n3. Nombre total de villes: {len(VILLES_FRANCE)}")
    print(f"   Top 10 villes: {get_top_cities(10)}")
    
    # Test 4: Villes par région
    print("\n4. Villes Auvergne-Rhône-Alpes:")
    print(f"   {get_cities_by_region('Auvergne-Rhône-Alpes')}")