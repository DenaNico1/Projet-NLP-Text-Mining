"""
Fonctions Utilitaires - Analyses NLP
Fonctions communes pour toutes les analyses

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import re
from typing import List, Optional

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


class DataLoader:
    """
    Chargeur de données depuis l'entrepôt DuckDB
    """
    
    def __init__(self, db_path: str = "../entrepot_de_donnees/entrepot_nlp.duckdb"):
        """
        Initialise le chargeur
        
        Args:
            db_path: Chemin vers la base DuckDB
        """
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connexion à la base"""
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            print(f"✅ Connecté à: {self.db_path}")
    
    def disconnect(self):
        """Déconnexion"""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("🔒 Connexion fermée")
    
    def load_all_offers(self) -> pd.DataFrame:
        """
        Charge toutes les offres avec leurs dimensions
        
        Returns:
            DataFrame avec toutes les colonnes enrichies
        """
        self.connect()
        
        query = """
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
            c.duration,
            o.salary_min,
            o.salary_max,
            o.salary_text,
            t.date_posted,
            o.description,
            o.url,
            o.scraped_at
        FROM fact_offres o
        LEFT JOIN dim_source s ON o.source_id = s.source_id
        LEFT JOIN dim_localisation l ON o.localisation_id = l.localisation_id
        LEFT JOIN dim_entreprise e ON o.entreprise_id = e.entreprise_id
        LEFT JOIN dim_contrat c ON o.contrat_id = c.contrat_id
        LEFT JOIN dim_temps t ON o.temps_id = t.temps_id
        WHERE o.description IS NOT NULL
        """
        
        df = self.conn.execute(query).df()
        
        print(f"📊 Chargé: {len(df)} offres")
        print(f"   - France Travail: {len(df[df['source_name'] == 'France Travail'])}")
        print(f"   - Indeed: {len(df[df['source_name'] == 'Indeed'])}")
        
        return df
    
    def load_competences(self) -> pd.DataFrame:
        """
        Charge les compétences structurées (France Travail)
        
        Returns:
            DataFrame compétences
        """
        self.connect()
        
        query = """
        SELECT 
            c.competence_id,
            c.offre_id,
            c.skill_code,
            c.skill_label,
            c.skill_level,
            o.title,
            s.source_name
        FROM fact_competences c
        JOIN fact_offres o ON c.offre_id = o.offre_id
        JOIN dim_source s ON o.source_id = s.source_id
        """
        
        df = self.conn.execute(query).df()
        
        print(f"🎓 Chargé: {len(df)} compétences")
        
        return df


class TextPreprocessor:
    """
    Préprocessing du texte pour analyses NLP
    """
    
    def __init__(self, language: str = 'french'):
        """
        Initialise le preprocessor
        
        Args:
            language: Langue pour stopwords
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        
        # Ajouter stopwords personnalisés
        custom_stops = {
            'emploi', 'offre', 'poste', 'recherche', 'recrute',
            'missions', 'profil', 'travail', 'entreprise', 'equipe',
            'description', 'competences', 'experience', 'formation',
            'cv', 'candidature', 'nbsp', 'ref', 'code'
        }
        self.stop_words.update(custom_stops)
        
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte
        
        Args:
            text: Texte brut
            
        Returns:
            Texte nettoyé
        """
        if not text or pd.isna(text):
            return ""
        
        # Minuscules
        text = text.lower()
        
        # Supprimer HTML/XML
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Supprimer URLs
        text = re.sub(r'http\S+|www\S+', ' ', text)
        
        # Supprimer emails
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Garder uniquement lettres et espaces
        text = re.sub(r'[^a-zàâäéèêëïîôöùûüÿç\s]', ' ', text)
        
        # Supprimer espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize et filtre le texte
        
        Args:
            text: Texte nettoyé
            
        Returns:
            Liste de tokens
        """
        # Tokenize
        tokens = word_tokenize(text, language=self.language)
        
        # Filtrer
        tokens = [
            t for t in tokens 
            if len(t) > 2  # Au moins 3 caractères
            and t not in self.stop_words
        ]
        
        return tokens
    
    def preprocess(self, text: str, lemmatize: bool = False) -> List[str]:
        """
        Pipeline complet de preprocessing
        
        Args:
            text: Texte brut
            lemmatize: Appliquer la lemmatisation
            
        Returns:
            Liste de tokens preprocessés
        """
        # Nettoyer
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Lemmatize (optionnel)
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
    
    def preprocess_corpus(self, texts: List[str], lemmatize: bool = False) -> List[List[str]]:
        """
        Preprocess un corpus entier
        
        Args:
            texts: Liste de textes
            lemmatize: Appliquer lemmatisation
            
        Returns:
            Liste de listes de tokens
        """
        from tqdm import tqdm
        
        corpus = []
        for text in tqdm(texts, desc="Preprocessing"):
            tokens = self.preprocess(text, lemmatize=lemmatize)
            corpus.append(tokens)
        
        return corpus


class ResultSaver:
    """
    Sauvegarde des résultats des analyses
    """
    
    def __init__(self, output_dir: str = "../resultats_nlp"):
        """
        Initialise le saver
        
        Args:
            output_dir: Répertoire de sortie
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Sous-répertoires
        self.viz_dir = self.output_dir / "visualisations"
        self.viz_dir.mkdir(exist_ok=True)
        
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        print(f"📁 Répertoire résultats: {self.output_dir}")
    
    def save_json(self, data: dict, filename: str):
        """Sauvegarde JSON"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Sauvegardé: {filepath}")
    
    def save_pickle(self, obj, filename: str):
        """Sauvegarde Pickle"""
        filepath = self.models_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"💾 Sauvegardé: {filepath}")
    
    def save_numpy(self, array: np.ndarray, filename: str):
        """Sauvegarde NumPy array"""
        filepath = self.output_dir / filename
        np.save(filepath, array)
        print(f"💾 Sauvegardé: {filepath}")
    
    def save_csv(self, df: pd.DataFrame, filename: str):
        """Sauvegarde CSV"""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"💾 Sauvegardé: {filepath}")
    
    def save_visualization(self, fig, filename: str):
        """
        Sauvegarde visualisation (Plotly ou Matplotlib)
        
        Args:
            fig: Figure Plotly ou Matplotlib
            filename: Nom du fichier (avec extension)
        """
        filepath = self.viz_dir / filename
        
        # Déterminer le type
        if hasattr(fig, 'write_html'):
            # Plotly
            fig.write_html(filepath)
        elif hasattr(fig, 'savefig'):
            # Matplotlib
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
        else:
            # Wordcloud ou autre
            fig.to_file(filepath)
        
        print(f"📊 Visualisation sauvegardée: {filepath}")


def compute_salary_annual(row: pd.Series) -> Optional[float]:
    """
    Calcule le salaire annuel moyen
    
    Args:
        row: Ligne du DataFrame avec salary_min et salary_max
        
    Returns:
        Salaire annuel moyen ou None
    """
    if pd.notna(row['salary_min']) and pd.notna(row['salary_max']):
        return (row['salary_min'] + row['salary_max']) / 2
    elif pd.notna(row['salary_min']):
        return row['salary_min']
    elif pd.notna(row['salary_max']):
        return row['salary_max']
    return None


def extract_competences_from_text(text: str, competences_dict: List[str]) -> List[str]:
    """
    Extrait les compétences présentes dans un texte
    
    Args:
        text: Texte à analyser
        competences_dict: Liste de compétences à chercher
        
    Returns:
        Liste des compétences trouvées
    """
    if not text or pd.isna(text):
        return []
    
    text_lower = text.lower()
    found = []
    
    for comp in competences_dict:
        comp_lower = comp.lower()
        # Recherche avec word boundary
        if re.search(r'\b' + re.escape(comp_lower) + r'\b', text_lower):
            found.append(comp)
    
    return found


# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    print("="*70)
    print("🧪 TEST DES UTILITAIRES")
    print("="*70)
    
    # Test DataLoader
    print("\n1️⃣ Test DataLoader")
    loader = DataLoader()
    df = loader.load_all_offers()
    print(f"\n   Colonnes: {list(df.columns)}")
    print(f"   Aperçu:\n{df[['title', 'source_name', 'city']].head()}")
    
    # Test TextPreprocessor
    print("\n2️⃣ Test TextPreprocessor")
    preprocessor = TextPreprocessor()
    
    sample_text = """
    Nous recherchons un Data Scientist expérimenté en Python, 
    Machine Learning et SQL pour notre équipe à Paris.
    Compétences: TensorFlow, Pandas, AWS.
    """
    
    tokens = preprocessor.preprocess(sample_text)
    print(f"\n   Texte: {sample_text[:100]}...")
    print(f"   Tokens: {tokens}")
    
    # Test ResultSaver
    print("\n3️⃣ Test ResultSaver")
    saver = ResultSaver()
    test_data = {"test": "ok", "count": 123}
    saver.save_json(test_data, "test.json")
    
    print("\n✅ Tests terminés !")
    
    loader.disconnect()