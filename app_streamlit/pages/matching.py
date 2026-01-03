"""
PAGE 8 : MATCHING CV ↔ OFFRES
Système ML hybride (Embeddings + Random Forest)
"""

import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, RESULTS_DIR, COLORS

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata

# ============================================
# CHARGEMENT DONNÉES & MODÈLE
# ============================================

# ============================================
# CHARGEMENT EMBEDDINGS PRÉ-CALCULÉS (CACHE)
# ============================================

# @st.cache_data
# def load_precomputed_embeddings():
#     """Charge embeddings pré-calculés des offres"""
#     try:
#         embeddings_path = MODELS_DIR / 'embeddings.npy'
#         embeddings = np.load(embeddings_path)
#         st.success(f" Embeddings chargés ({len(embeddings)} offres)")
#         return embeddings
#     except Exception as e:
#         st.warning(f" Embeddings non trouvés : {e}")
#         return None

# # Charger au démarrage
# OFFRES_EMBEDDINGS = load_precomputed_embeddings()

# @st.cache_resource
# def load_matching_system():
#     """Charge modèle ML + assets"""
#     with open(MODELS_DIR / 'matching_model.pkl', 'rb') as f:
#         system = pickle.load(f)
    
#     embeddings_model = SentenceTransformer(system['embeddings_model_name'])
    
#     return system['rf_model'], system['tfidf_vectorizer'], embeddings_model

# @st.cache_data
# def load_cv_base():
#     """Charge base CV fictifs"""
#     with open(RESULTS_DIR / 'cv_base_fictifs.json', 'r', encoding='utf-8') as f:
#         return json.load(f)

# @st.cache_data
# def load_offres():
#     """Charge offres réelles"""
#     with open(MODELS_DIR / 'data_with_profiles.pkl', 'rb') as f:
#         return pickle.load(f)

# @st.cache_data
# def load_metrics():
#     """Charge métriques modèle"""
#     try:
#         with open(RESULTS_DIR / 'matching_metrics.json', 'r', encoding='utf-8') as f:
#             return json.load(f)
#     except:
#         return {}

from data_loaders import load_matching_data
df, embeddings, rf_model, tfidf, embeddings_model, cv_base, metrics = load_matching_data()

# Alias pour compatibilité (si besoin)
OFFRES_EMBEDDINGS = embeddings
tfidf_vec = tfidf
emb_model = embeddings_model
df_offres = df

# Vérifier chargement
SYSTEM_LOADED = (
    embeddings is not None and 
    rf_model is not None and 
    embeddings_model is not None
)

if not SYSTEM_LOADED:
    st.error(" Système matching non initialisé")
    st.info(" Exécuter: `python 9_ml_matching_system.py`")
    st.stop()

# ============================================
# FONCTIONS MATCHING
# ============================================

def normalize(text):
    """Normalisation texte"""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    return text.lower().strip()

def extract_features_predict(cv, offre, emb_model, tfidf_vec):
    """Extrait features pour prédiction"""
    
    # Textes
    cv_text = f"{cv['titre_recherche']} {' '.join(cv['competences'][:10])}"
    offre_text = f"{offre['title']} {offre.get('description', '')[:500]}"
    
    # 1. Embedding similarity
    cv_emb = emb_model.encode(cv_text)
    offre_emb = emb_model.encode(offre_text)
    embedding_sim = float(cosine_similarity([cv_emb], [offre_emb])[0][0])
    
    # 2. TF-IDF similarity
    try:
        tfidf_matrix = tfidf_vec.transform([cv_text, offre_text])
        tfidf_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except:
        tfidf_sim = 0.0
    
    # 3. Compétences ratio
    cv_comp = set([normalize(c) for c in cv['competences']])
    
    if isinstance(offre.get('competences_found'), list):
        offre_comp = set([normalize(c) for c in offre['competences_found']])
    else:
        offre_comp = set()
    
    common_comp = cv_comp & offre_comp
    
    if len(offre_comp) > 0:
        comp_ratio = len(common_comp) / len(offre_comp)
    else:
        comp_ratio = 0.0
    
    # 4. Compétences count
    comp_count = len(common_comp)
    
    
    # 5. Experience gap
    offre_exp = offre.get('experience_level', 0)
    cv_exp = cv.get('annees_experience', 0)

    # Conversion robuste
    try:
        offre_exp = int(offre_exp) if offre_exp else 0
    except (ValueError, TypeError):
        offre_exp = 0

    try:
        cv_exp = int(cv_exp) if cv_exp else 0
    except (ValueError, TypeError):
        cv_exp = 0

    exp_gap = float(offre_exp - cv_exp)
    
    # 6. Title similarity
    cv_title = normalize(cv['titre_recherche'])
    offre_title = normalize(offre['title'])
    
    cv_words = set(cv_title.split())
    offre_words = set(offre_title.split())
    
    if len(offre_words) > 0:
        title_sim = len(cv_words & offre_words) / len(offre_words)
    else:
        title_sim = 0.0
    
    features = np.array([[
        embedding_sim,
        tfidf_sim,
        comp_ratio,
        comp_count,
        exp_gap,
        title_sim
    ]])
    
    details = {
        'embedding_sim': embedding_sim,
        'tfidf_sim': tfidf_sim,
        'comp_ratio': comp_ratio,
        'comp_count': comp_count,
        'common_comp': list(common_comp),
        'missing_comp': list(offre_comp - cv_comp),
        'exp_gap': exp_gap,
        'title_sim': title_sim
    }
    
    return features, details

def predict_matches(cv, df_offres, rf_model, emb_model, tfidf_vec, top_n=10):
    """Prédit top N offres matching pour un CV donné"""
    
    # Encoder CV UNE SEULE FOIS
    cv_text = f"{cv['titre_recherche']} {' '.join(cv['competences'][:10])}"
    cv_emb = emb_model.encode(cv_text)
    
    results = []
    
    # Utiliser embeddings pré-calculés
    use_precomputed = OFFRES_EMBEDDINGS is not None and len(OFFRES_EMBEDDINGS) >= len(df_offres)
    
    for idx, offre in df_offres.iterrows():
        # === EMBEDDING SIMILARITY ===
        if use_precomputed:
            offre_emb = OFFRES_EMBEDDINGS[idx]
            embedding_sim = float(cosine_similarity([cv_emb], [offre_emb])[0][0])
        else:
            # Fallback lent
            offre_text = f"{offre['title']} {offre.get('description', '')[:300]}"
            offre_emb = emb_model.encode(offre_text)
            embedding_sim = float(cosine_similarity([cv_emb], [offre_emb])[0][0])
        
        # === COMPÉTENCES ===
        cv_comp = set([normalize(c) for c in cv['competences']])
        
        if isinstance(offre.get('competences_found'), list):
            offre_comp = set([normalize(c) for c in offre['competences_found']])
        else:
            offre_comp = set()
        
        common_comp = cv_comp & offre_comp
        comp_ratio = len(common_comp) / len(offre_comp) if len(offre_comp) > 0 else 0.0
        comp_count = len(common_comp)
        
        # === EXPERIENCE ===
        offre_exp = offre.get('experience_level', 0)
        cv_exp = cv.get('annees_experience', 0)
        
        try:
            offre_exp = int(offre_exp) if offre_exp else 0
        except (ValueError, TypeError):
            offre_exp = 0
        
        try:
            cv_exp = int(cv_exp) if cv_exp else 0
        except (ValueError, TypeError):
            cv_exp = 0
        
        exp_gap = float(offre_exp - cv_exp)
        
        # === TITRE ===
        cv_title = normalize(cv['titre_recherche'])
        offre_title = normalize(offre['title'])
        
        cv_words = set(cv_title.split())
        offre_words = set(offre_title.split())
        title_sim = len(cv_words & offre_words) / len(offre_words) if len(offre_words) > 0 else 0.0
        
        # === FEATURES ===
        features = np.array([[
            embedding_sim,
            0.0,  # tfidf_sim (skip pour vitesse)
            comp_ratio,
            comp_count,
            exp_gap,
            title_sim
        ]])
        
        # === PRÉDICTION ===
        proba = rf_model.predict_proba(features)[0][1]
        score = proba * 100
        
        # === BONUS TITRE ===
        if cv_title in offre_title or offre_title in cv_title:
            score = min(score * 1.3, 100)
        elif len(cv_title.split()) >= 2:
            key_words = cv_title.split()[:2]
            if all(word in offre_title for word in key_words):
                score = min(score * 1.15, 100)
        
        # === DÉTAILS ===
        details = {
            'embedding_sim': embedding_sim,
            'comp_ratio': comp_ratio,
            'comp_count': comp_count,
            'common_comp': list(common_comp),
            'missing_comp': list(offre_comp - cv_comp),
            'exp_gap': exp_gap,
            'title_sim': title_sim
        }
        
        results.append({
            'offre': offre.to_dict(),
            'score': score,
            'details': details
        })
    
    # === FILTRAGE INTELLIGENT ===
    results_filtered = []
    cv_title_check = normalize(cv['titre_recherche'])
    
    for result in results:
        offre_title_check = normalize(result['offre']['title'])
        should_keep = True
        
        if 'data engineer' in cv_title_check or 'engineer' in cv_title_check:
            required_kw = ['data', 'engineer', 'etl', 'pipeline', 'big data', 'données', 'cloud', 'spark', 'kafka']
            excluded_kw = ['développeur web', 'développeur mobile', 'front-end', 'frontend', 'programmeur c', 'c++', 'c#', '.net']
            should_keep = any(kw in offre_title_check for kw in required_kw) and not any(kw in offre_title_check for kw in excluded_kw)
        
        elif 'data scientist' in cv_title_check or 'scientist' in cv_title_check:
            required_kw = ['data', 'scientist', 'science', 'machine learning', 'ml', 'ia', 'intelligence artificielle', 'research']
            excluded_kw = ['développeur', 'programmeur', 'ingénieur système']
            should_keep = any(kw in offre_title_check for kw in required_kw) and not any(kw in offre_title_check for kw in excluded_kw)
        
        elif 'data analyst' in cv_title_check or 'analyst' in cv_title_check:
            required_kw = ['data', 'analyst', 'analyse', 'bi', 'business intelligence', 'reporting', 'tableau']
            should_keep = any(kw in offre_title_check for kw in required_kw)
        
        if should_keep:
            results_filtered.append(result)
    
    if len(results_filtered) < 5:
        results_filtered = results[:top_n]
    
    results_filtered = sorted(results_filtered, key=lambda x: x['score'], reverse=True)[:top_n]
    
    return results_filtered

# ============================================
# UI COMPONENTS
# ============================================

def display_match_result(match, rank):
    """Affiche un résultat de matching"""
    
    offre = match['offre']
    score = match['score']
    details = match['details']
    
    # Couleur selon score
    if score >= 70:
        color = '#10b981'
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else '✅'
    elif score >= 50:
        color = '#f59e0b'
        medal = '⚠️'
    else:
        color = '#ef4444'
        medal = '❌'
    
    with st.expander(f"{medal} **#{rank} - {offre['title']}** ({score:.0f}%)", expanded=(rank==1)):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**🏢 Entreprise:** {offre.get('company_name', 'N/A')}")
            st.markdown(f"**📍 Localisation:** {offre.get('city', 'N/A')}, {offre.get('region', 'N/A')}")
            st.markdown(f"**📝 Contrat:** {offre.get('contract_type', 'N/A')}")
            
            if pd.notna(offre.get('salary_annual')):
                st.markdown(f"**💰 Salaire:** {offre['salary_annual']/1000:.0f}K€/an")
            
            # URL offre
            if pd.notna(offre.get('url')):
                st.markdown(f"**🔗 Lien:** [Voir l'offre complète]({offre['url']})")
            elif pd.notna(offre.get('source_url')):
                st.markdown(f"**🔗 Lien:** [Voir l'offre complète]({offre['source_url']})")
        
        with col2:
            # Score visuel
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #1f2937 0%, #374151 100%); 
                        border-radius: 10px; border-left: 4px solid {color};'>
                <p style='color: #9ca3af; font-size: 0.9rem; margin: 0;'>Score Match</p>
                <p style='font-size: 2.5rem; font-weight: 700; margin: 10px 0; color: {color};'>{score:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Détails matching
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("**✅ Compétences Matchées:**")
            if details['common_comp']:
                for comp in details['common_comp'][:8]:
                    st.markdown(f"- {comp}")
            else:
                st.caption("Aucune")
        
        with col_comp2:
            st.markdown("**❌ Compétences Manquantes:**")
            if details['missing_comp']:
                for comp in details['missing_comp'][:8]:
                    st.markdown(f"- {comp}")
            else:
                st.caption("Aucune")
        
        # Description
        if pd.notna(offre.get('description')):
            with st.expander("📄 Voir description complète"):
                st.text(offre['description'][:500] + "...")

# ============================================
# HEADER
# ============================================

st.title(" Système de Matching CV ↔ Offres")
st.markdown("**Intelligence Artificielle hybride** : Embeddings + Random Forest")

if not SYSTEM_LOADED:
    st.stop()

# Métriques modèle
# if metrics:
#     col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
#     with col_m1:
#         st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
    
#     with col_m2:
#         st.metric("Précision", f"{metrics.get('precision', 0)*100:.1f}%")
    
#     with col_m3:
#         st.metric("Recall", f"{metrics.get('recall', 0)*100:.1f}%")
    
#     with col_m4:
#         st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")

st.markdown("---")

# ============================================
# TABS
# ============================================

tab1, tab2, tab3 = st.tabs([" Je cherche un emploi", " Je recrute", " Base de CV"])

# ============================================
# TAB 1 : CHERCHEUR EMPLOI
# ============================================

with tab1:
    st.subheader(" Trouvez les offres qui vous correspondent")
    
    st.info("💡 Remplissez le formulaire pour trouver les meilleures offres matching")
    
    with st.form("cv_form"):
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            nom = st.text_input("Nom *", placeholder="Ex: Jean Dupont")
            titre = st.text_input("Titre/Poste recherché *", placeholder="Ex: Data Scientist")
            
            # Compétences disponibles
            all_comp = set()
            for comp_list in df_offres['competences_found']:
                if isinstance(comp_list, list):
                    all_comp.update(comp_list)
            
            competences = st.multiselect(
                "Compétences *",
                options=sorted(list(all_comp)),
                default=['python', 'sql'],
                help="Sélectionnez vos compétences principales"
            )
        
        with col_f2:
            experience = st.number_input("Années d'expérience", min_value=0, max_value=20, value=3)
            
            formations = ['Bac+2', 'Bac+3 (Licence)', 'Bac+5 (Master)', 'Doctorat', 'Autre']
            formation = st.selectbox("Formation", formations, index=2)
            
            villes = ['Paris', 'Lyon', 'Toulouse', 'Bordeaux', 'Lille', 'Marseille', 'Nantes', 'Autre']
            ville = st.selectbox("Localisation préférée", villes)
        
        submitted = st.form_submit_button(" Trouver mes offres", use_container_width=True)
    
    if submitted:
        if not titre or not competences:
            st.error("⚠️ Veuillez remplir au minimum le titre et les compétences")
        else:
            # Créer CV utilisateur
            cv_user = {
                'nom': nom,
                'titre_recherche': titre,
                'competences': competences,
                'annees_experience': experience,
                'formation': formation,
                'localisation_preferee': ville
            }
            
            # Matching
            with st.spinner(" Analyse de vos compétences et recherche des meilleures offres..."):
                matches = predict_matches(cv_user, df_offres, rf_model, emb_model, tfidf_vec, top_n=10)
            
            st.success(f"✅ {len(matches)} offres trouvées !")
            
            st.markdown("---")
            st.subheader(" Vos meilleures opportunités")
            
            # Affichage résultats
            for i, match in enumerate(matches, 1):
                display_match_result(match, i)

# ============================================
# TAB 2 : RECRUTEUR
# ============================================

with tab2:
    st.subheader(" Trouvez les meilleurs profils")
    
    st.info("💡 Décrivez votre offre pour trouver les candidats correspondants")
    
    with st.form("offre_form"):
        col_o1, col_o2 = st.columns(2)
        
        with col_o1:
            titre_offre = st.text_input("Titre du poste *", placeholder="Ex: Data Engineer Senior")
            entreprise = st.text_input("Entreprise", placeholder="Ex: Startup IA")
            
            all_comp = set()
            for comp_list in df_offres['competences_found']:
                if isinstance(comp_list, list):
                    all_comp.update(comp_list)
            
            comp_requises = st.multiselect(
                "Compétences requises *",
                options=sorted(list(all_comp)),
                default=['python', 'spark'],
                help="Compétences indispensables"
            )
        
        with col_o2:
            exp_min = st.number_input("Expérience minimum (années)", min_value=0, max_value=15, value=3)
            
            ville_offre = st.selectbox("Localisation poste", villes, key='ville_offre')
            
            contrat = st.selectbox("Type contrat", ['CDI', 'CDD', 'Freelance', 'Stage'])
        
        description_offre = st.text_area(
            "Description du poste",
            placeholder="Ex: Nous recherchons un Data Engineer pour...",
            height=150
        )
        
        submitted_offre = st.form_submit_button("🔍 Trouver des candidats", use_container_width=True)
    
    if submitted_offre:
        if not titre_offre or not comp_requises:
            st.error(" Veuillez remplir au minimum le titre et les compétences")
        else:
            # Créer offre
            offre_new = {
                'title': titre_offre,
                'company_name': entreprise,
                'description': description_offre,
                'competences_found': comp_requises,
                'experience_level': exp_min,
                'city': ville_offre,
                'contract_type': contrat
            }
            
            # Matching contre base CV
            matches_cv = []
            
            with st.spinner(" Recherche des profils correspondants..."):
                for cv in cv_base:
                    features, details = extract_features_predict(
                        cv, offre_new, emb_model, tfidf_vec
                    )
                    
                    proba = rf_model.predict_proba(features)[0][1]
                    score = proba * 100
                    
                    matches_cv.append({
                        'cv': cv,
                        'score': score,
                        'details': details
                    })
                
                matches_cv = sorted(matches_cv, key=lambda x: x['score'], reverse=True)[:10]
            
            st.success(f"✅ {len(matches_cv)} profils trouvés dans la base !")
            
            st.markdown("---")
            st.subheader(" Meilleurs candidats")
            
            # Affichage
            for i, match in enumerate(matches_cv, 1):
                cv = match['cv']
                score = match['score']
                details = match['details']
                
                if score >= 70:
                    color = '#10b981'
                    medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '✅'
                elif score >= 50:
                    color = '#f59e0b'
                    medal = '⚠️'
                else:
                    color = '#ef4444'
                    medal = '❌'
                
                with st.expander(f"{medal} **#{i} - {cv['nom']}** ({score:.0f}%)", expanded=(i==1)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"** Profil:** {cv['profil_type']} - {cv['niveau']}")
                        st.markdown(f"** Expérience:** {cv['annees_experience']} ans")
                        st.markdown(f"** Formation:** {cv['formation']}")
                        st.markdown(f"** Localisation:** {cv['localisation_preferee']}")
                    
                    with col2:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #1f2937 0%, #374151 100%); 
                                    border-radius: 10px; border-left: 4px solid {color};'>
                            <p style='color: #9ca3af; font-size: 0.9rem; margin: 0;'>Score Match</p>
                            <p style='font-size: 2.5rem; font-weight: 700; margin: 10px 0; color: {color};'>{score:.0f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    col_c1, col_c2 = st.columns(2)
                    
                    with col_c1:
                        st.markdown("**✅ Compétences Matchées:**")
                        for comp in details['common_comp'][:8]:
                            st.markdown(f"- {comp}")
                    
                    with col_c2:
                        st.markdown("**❌ Compétences Manquantes:**")
                        if details['missing_comp']:
                            for comp in details['missing_comp'][:8]:
                                st.markdown(f"- {comp}")
                        else:
                            st.caption("Aucune")

# ============================================
# TAB 3 : BASE CV
# ============================================

with tab3:
    st.subheader(" Base de CV Démo")
    
    st.info(f"**{len(cv_base)} CV fictifs** disponibles pour tests de matching")
    
    # Stats
    col_s1, col_s2, col_s3 = st.columns(3)
    
    profils_count = {}
    niveaux_count = {}
    
    for cv in cv_base:
        profils_count[cv['profil_type']] = profils_count.get(cv['profil_type'], 0) + 1
        niveaux_count[cv['niveau']] = niveaux_count.get(cv['niveau'], 0) + 1
    
    with col_s1:
        st.metric("Total CVs", len(cv_base))
    
    with col_s2:
        st.metric("Profils différents", len(profils_count))
    
    with col_s3:
        avg_exp = np.mean([cv['annees_experience'] for cv in cv_base])
        st.metric("Expérience moyenne", f"{avg_exp:.1f} ans")
    
    # Distribution
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("**Distribution Profils:**")
        for profil, count in sorted(profils_count.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"- {profil}: {count}")
    
    with col_d2:
        st.markdown("**Distribution Niveaux:**")
        for niveau, count in sorted(niveaux_count.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"- {niveau}: {count}")
    
    st.markdown("---")
    
    # Tableau CVs
    df_cvs = pd.DataFrame([
        {
            'Nom': cv['nom'],
            'Profil': cv['profil_type'],
            'Niveau': cv['niveau'],
            'Expérience': f"{cv['annees_experience']} ans",
            'Formation': cv['formation'],
            'Ville': cv['localisation_preferee'],
            'Nb Compétences': len(cv['competences'])
        }
        for cv in cv_base
    ])
    
    st.dataframe(df_cvs, use_container_width=True, hide_index=True)