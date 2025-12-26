"""Page 8 : Analyse de CV
Upload CV → Extraction compétences → Recommandation offres
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from data_loader import load_preprocessed_data
from search_utils import (
    extract_competences_from_cv,
    recommend_offers_by_cv,
    compute_gap_analysis,
    estimate_salary_impact
)

st.set_page_config(page_title="Analyse CV", page_icon="📄", layout="wide")

# Titre
st.title("📄 Analyse de CV & Recommandations")
st.markdown("Analysez votre CV et obtenez des recommandations d'offres personnalisées")

# Chargement données
try:
    df = load_preprocessed_data()
    
    # Charger dictionnaire compétences
    dict_path = Path(__file__).parent.parent.parent / "resultats_nlp" / "dictionnaire_competences.json"
    with open(dict_path, 'r', encoding='utf-8') as f:
        dict_comp = json.load(f)['competences']
    
    # Charger modèle de classification
    model_path = Path(__file__).parent.parent.parent / "resultats_nlp" / "models" / "model_svm.pkl"
    vectorizer_path = Path(__file__).parent.parent.parent / "resultats_nlp" / "models" / "vectorizer_classification.pkl"
    
    with open(model_path, 'rb') as f:
        model_classif = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer_classif = pickle.load(f)
    
    # Charger compétences signature
    chi2_path = Path(__file__).parent.parent.parent / "resultats_nlp" / "chi2_selection.json"
    with open(chi2_path, 'r', encoding='utf-8') as f:
        chi2_data = json.load(f)
        signature_by_profile = chi2_data['signature_by_profile']
    
    # Charger données salariales
    sal_path = Path(__file__).parent.parent.parent / "resultats_nlp" / "stacks_salaires.json"
    with open(sal_path, 'r', encoding='utf-8') as f:
        sal_data = json.load(f)
        salary_by_comp = {
            item['competence']: item['salary_median']
            for item in sal_data.get('salaire_par_competence', [])
        }
    
except Exception as e:
    st.error(f"❌ Erreur chargement : {e}")
    st.stop()

# ============================================================================
# INPUT : CV
# ============================================================================

st.markdown("## 📝 Votre CV")

# Méthode 1 : Copier-coller le texte
cv_text = st.text_area(
    "Collez le texte de votre CV ici",
    height=300,
    placeholder="""Exemple :
Data Scientist avec 3 ans d'expérience en Machine Learning.

Compétences :
- Python, TensorFlow, PyTorch
- SQL, Pandas, NumPy
- Docker, Kubernetes
- AWS

Expérience :
- Développement de modèles de recommandation
- Déploiement de modèles en production
- ...
""",
    help="Copiez-collez le contenu de votre CV (texte brut)"
)

# Bouton d'analyse
if st.button("🔍 Analyser mon CV", type="primary", disabled=not cv_text):
    
    with st.spinner("🔄 Analyse en cours..."):
        
        # ====================================================================
        # ÉTAPE 1 : Extraction compétences
        # ====================================================================
        
        st.markdown("---")
        st.markdown("## 🎓 Compétences Extraites")
        
        cv_competences = extract_competences_from_cv(cv_text, dict_comp)
        
        if not cv_competences:
            st.warning("⚠️ Aucune compétence reconnue dans le CV. Essayez d'ajouter plus de détails.")
            st.stop()
        
        st.success(f"✅ {len(cv_competences)} compétences extraites")
        
        # Afficher les compétences
        cols = st.columns(4)
        for i, comp in enumerate(sorted(cv_competences)):
            cols[i % 4].markdown(f"✅ {comp}")
        
        # ====================================================================
        # ÉTAPE 2 : Classification du profil
        # ====================================================================
        
        st.markdown("---")
        st.markdown("## 🎯 Profil Détecté")
        
        # Vectoriser le CV
        cv_vec = vectorizer_classif.transform([cv_text])
        
        # Prédire le profil
        profil_pred = model_classif.predict(cv_vec)[0]
        
        # Probabilités (si SVM avec probability=True)
        try:
            probas = model_classif.predict_proba(cv_vec)[0]
            classes = model_classif.classes_
            
            # Trier par probabilité décroissante
            top_indices = np.argsort(probas)[::-1]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(
                    "Profil Principal",
                    profil_pred,
                    f"{probas[top_indices[0]]*100:.0f}% confiance"
                )
            
            with col2:
                st.markdown("**Probabilités par profil :**")
                df_probas = pd.DataFrame({
                    'Profil': [classes[i] for i in top_indices],
                    'Probabilité': [probas[i] * 100 for i in top_indices]
                })
                st.dataframe(df_probas, hide_index=True, use_container_width=True)
        
        except:
            # Si pas de probabilités disponibles
            st.metric("Profil Détecté", profil_pred)
        
        # ====================================================================
        # ÉTAPE 3 : Gap Analysis
        # ====================================================================
        
        st.markdown("---")
        st.markdown("## 💡 Gap Analysis")
        
        gap = compute_gap_analysis(cv_competences, profil_pred, signature_by_profile)
        
        st.info(gap['message'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Compétences Maîtrisées")
            if gap['competences_present']:
                for comp in gap['competences_present'][:10]:
                    st.markdown(f"- {comp}")
                if len(gap['competences_present']) > 10:
                    st.markdown(f"*... et {len(gap['competences_present']) - 10} autres*")
            else:
                st.write("Aucune compétence signature identifiée")
        
        with col2:
            st.markdown("### ❌ Compétences Manquantes")
            if gap['competences_missing']:
                for comp in gap['competences_missing'][:10]:
                    st.markdown(f"- {comp}")
                if len(gap['competences_missing']) > 10:
                    st.markdown(f"*... et {len(gap['competences_missing']) - 10} autres*")
                
                # Estimation impact salarial
                salary_impact = estimate_salary_impact(gap['competences_missing'], salary_by_comp)
                
                if salary_impact['potential_increase_pct'] > 0:
                    st.success(f"💰 {salary_impact['message']}")
            else:
                st.write("✅ Vous maîtrisez toutes les compétences signature !")
        
        # ====================================================================
        # ÉTAPE 4 : Recommandation d'offres
        # ====================================================================
        
        st.markdown("---")
        st.markdown("## 🏆 Offres Recommandées")
        
        # Recommandation basée sur compétences
        recommendations = recommend_offers_by_cv(
            df=df,
            cv_competences=cv_competences,
            embeddings_cv=None,  # Pas d'embeddings pour version simple
            embeddings_offres=None,
            top_k=10,
            method='competences'
        )
        
        st.success(f"✅ {len(recommendations)} offres recommandées")
        
        # Afficher les offres
        for idx, row in recommendations.iterrows():
            score_pct = row['recommendation_score'] * 100
            
            # Couleur selon le score
            if score_pct >= 80:
                icon = "🟢"
            elif score_pct >= 60:
                icon = "🟡"
            else:
                icon = "🟠"
            
            with st.expander(f"{icon} **{row['title']}** - {row['company_name']} ({score_pct:.0f}% match)"):
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**📍 Lieu**")
                    if pd.notna(row['city']):
                        st.write(row['city'])
                    else:
                        st.write("Non spécifié")
                
                with col2:
                    st.markdown("**📝 Contrat**")
                    st.write(row['contract_type'] if pd.notna(row['contract_type']) else "N/A")
                
                with col3:
                    st.markdown("**🎯 Profil**")
                    st.write(row['profil'] if pd.notna(row['profil']) else "N/A")
                
                with col4:
                    st.markdown("**💰 Salaire**")
                    if pd.notna(row['salary_annual']):
                        st.write(f"{row['salary_annual']/1000:.0f}k€")
                    else:
                        st.write("N/A")
                
                # Compétences
                st.markdown("**🎓 Compétences demandées**")
                
                offre_comps = row['competences_found']
                
                # Compétences que vous avez
                comps_you_have = [c for c in offre_comps if c in cv_competences]
                # Compétences que vous n'avez pas
                comps_you_need = [c for c in offre_comps if c not in cv_competences]
                
                if comps_you_have:
                    st.markdown("✅ **Vous avez** : " + ", ".join(comps_you_have))
                
                if comps_you_need:
                    st.markdown("❌ **À acquérir** : " + ", ".join(comps_you_need[:5]))
                    if len(comps_you_need) > 5:
                        st.markdown(f"*... et {len(comps_you_need) - 5} autres*")
                
                # Lien
                if pd.notna(row.get('url')):
                    st.markdown(f"[🔗 Voir l'offre]({row['url']})")
        
        # Export
        st.markdown("---")
        if st.button("📥 Exporter les recommandations en CSV"):
            csv = recommendations.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="⬇️ Télécharger CSV",
                data=csv,
                file_name="recommandations_cv.csv",
                mime="text/csv"
            )

else:
    # Message initial
    st.info("👆 Collez le texte de votre CV ci-dessus et cliquez sur 'Analyser mon CV'")
    
    # Guide
    st.markdown("### 💡 Guide d'Utilisation")
    
    st.markdown("""
    **Comment ça marche ?**
    
    1. **Copiez votre CV** (format texte)
    2. **Collez-le** dans la zone de texte ci-dessus
    3. **Cliquez sur 'Analyser'**
    
    **Ce que vous obtiendrez :**
    
    - ✅ Extraction automatique de vos compétences
    - 🎯 Détection de votre profil métier
    - 💡 Analyse des compétences manquantes
    - 💰 Estimation de l'impact salarial
    - 🏆 Top 10 offres qui vous correspondent
    
    **Conseils :**
    
    - Incluez vos **compétences techniques** (Python, SQL, Docker...)
    - Mentionnez vos **projets** et **réalisations**
    - Indiquez votre **expérience** (années, contexte)
    - Plus le CV est détaillé, meilleure sera l'analyse !
    """)
    
    # Exemple
    with st.expander("📋 Voir un exemple de CV"):
        st.code("""
Data Scientist Senior
5 ans d'expérience

COMPÉTENCES TECHNIQUES
- Langages : Python, R, SQL
- ML/DL : TensorFlow, PyTorch, Scikit-learn, XGBoost
- Data : Pandas, NumPy, Spark
- DevOps : Docker, Kubernetes, Git
- Cloud : AWS (SageMaker, Lambda), Azure
- BI : Power BI, Tableau

EXPÉRIENCE
Data Scientist Senior - Startup FinTech (2020-2024)
- Développement de modèles de détection de fraude (XGBoost, recall 95%)
- Déploiement de 15 modèles en production (Docker + Kubernetes)
- Mise en place pipeline MLOps (Airflow + MLflow)

Machine Learning Engineer - Grande Entreprise (2018-2020)
- Création système de recommandation (collaborative filtering)
- Optimisation modèles NLP (BERT fine-tuning)
- Mentoring 3 data scientists juniors

PROJETS
- Chatbot NLP pour support client (GPT-3 + LangChain)
- Prédiction churn clients (feature engineering avancé)
        """, language="text")