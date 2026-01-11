# nouvelle_offre.py (VERSION 2.0 - AVEC SCRAPING URL)
"""
Ajout offres avec 2 modes :
1. URL automatique (scraping + LLM)
2. Texte manuel (copier-coller + LLM)
"""

import streamlit as st
import scraper  # Module scraping créé
import llm_extraction as ai
from data_loaders import ajouter_offre_avec_embedding
from sentence_transformers import SentenceTransformer

st.markdown("## 📥 Ajouter une nouvelle offre (Assisté par IA)")

# ============================================
# SÉLECTION MODE
# ============================================

mode = st.radio(
    "Mode d'ajout",
    ["🔗 URL automatique (scraping)", "📝 Texte manuel (copier-coller)"],
    horizontal=True
)

st.markdown("---")

# ============================================
# MODE 1 : URL AUTOMATIQUE
# ============================================

if mode == "🔗 URL automatique (scraping)":
    
    st.info("🚀 Collez l'URL d'une offre. L'IA va extraire et structurer les données automatiquement.")
    
    url_input = st.text_input(
        "URL de l'offre",
        placeholder="https://fr.indeed.com/viewjob?jk=...",
        help="Indeed, APEC, France Travail, sites carrière..."
    )
    
    if st.button("🔍 Analyser l'offre"):
        if not url_input:
            st.warning("⚠️ Veuillez coller une URL")
        else:
            # ÉTAPE 1 : SCRAPING
            with st.spinner("📡 Récupération de l'offre..."):
                scrape_result = scraper.smart_scrape(url_input)
            
            if not scrape_result['success']:
                st.error(f"❌ Impossible de récupérer l'offre : {scrape_result['error']}")
                st.info("💡 Essayez le mode 'Texte manuel' en copiant-collant le contenu")
            else:
                st.success(f"✅ Offre récupérée ({scrape_result['method']})")
                
                # Afficher aperçu texte
                with st.expander("👀 Aperçu texte extrait"):
                    st.text(scrape_result['text'][:1000] + "...")
                
                # ÉTAPE 2 : EXTRACTION LLM
                with st.spinner("🤖 Extraction des données via Mistral..."):
                    extracted_data = ai.extract_job_info(scrape_result['text'])
                
                if "error" in extracted_data:
                    st.error(f"❌ Erreur LLM : {extracted_data['error']}")
                else:
                    st.success("✅ Données extraites !")
                    
                    # Ajouter URL source
                    extracted_data['url'] = url_input
                    
                    # Stocker dans session
                    st.session_state['draft_offer'] = extracted_data
                    st.rerun()

# ============================================
# MODE 2 : TEXTE MANUEL
# ============================================

else:
    st.info("📋 Copiez le texte complet d'une offre. Mistral va extraire les champs clés.")
    
    raw_text = st.text_area(
        "Texte de l'offre",
        height=200,
        placeholder="Collez le texte complet ici..."
    )
    
    if st.button("✨ Analyser avec Mistral"):
        if not raw_text:
            st.warning("⚠️ Veuillez coller du texte d'abord.")
        else:
            with st.spinner("🤖 Extraction des données via Mistral..."):
                extracted_data = ai.extract_job_info(raw_text)
            
            if "error" in extracted_data:
                st.error(extracted_data["error"])
            else:
                st.success("✅ Extraction réussie !")
                st.session_state['draft_offer'] = extracted_data
                st.rerun()

# ============================================
# AFFICHAGE FORMULAIRE VALIDATION
# ============================================

if 'draft_offer' in st.session_state:
    
    # JSON brut
    with st.expander("🕵️ Voir le JSON brut généré par Mistral"):
        st.json(st.session_state['draft_offer'])
    
    st.divider()
    st.markdown("### 🔍 Vérification avant insertion")
    
    data = st.session_state['draft_offer']
    
    with st.form("validation_form"):
        col1, col2 = st.columns(2)
        
        # Champs formulaire
        title = col1.text_input("Titre du poste", value=data.get('title', ''))
        company = col2.text_input("Entreprise", value=data.get('company_name', ''))
        
        col3, col4 = st.columns(2)
        city = col3.text_input("Ville", value=data.get('city', ''))
        
        contract_options = ["CDI", "CDD", "Freelance", "Stage", "Alternance", "Non spécifié"]
        found_contract = data.get('contract_type', 'Non spécifié')
        try:
            idx = contract_options.index(found_contract)
        except ValueError:
            idx = 5
        
        contract = col4.selectbox("Type de contrat", contract_options, index=idx)
        
        col5, col6 = st.columns(2)
        sal_min = col5.number_input("Salaire Min (€/an)", value=int(data.get('salary_min') or 0))
        sal_max = col6.number_input("Salaire Max (€/an)", value=int(data.get('salary_max') or 0))
        
        desc = st.text_area("Description courte", value=data.get('description', ''), height=150)
        url = st.text_input("URL d'origine", value=data.get('url', ''))
        
        submitted = st.form_submit_button("💾 Sauvegarder dans la Base")
        
        if submitted:
            # Construction données finales
            final_data = {
                'title': title,
                'company_name': company,
                'city': city,
                'contract_type': contract,
                'salary_min': sal_min if sal_min > 0 else None,
                'salary_max': sal_max if sal_max > 0 else None,
                'description': desc,
                'url': url,
                'source': 'Import IA (Mistral + Scraping)' if mode.startswith("🔗") else 'Import IA (Mistral)'
            }
            
            # ============================================
            # VÉRIFICATION DOUBLONS (OPTIONNEL)
            # ============================================
            
            # TODO : Intégrer check_all_duplicates() ici si souhaité
            
            # ============================================
            # INSERTION + EMBEDDING AUTO
            # ============================================
            
            try:
                # Charger modèle embeddings
                emb_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                
                # Insertion avec embedding automatique
                new_id = ajouter_offre_avec_embedding(final_data, emb_model)
                
                if new_id:
                    # Nettoyer brouillon
                    del st.session_state['draft_offer']
                    
                    # Messages succès
                    st.toast("✅ Offre ajoutée avec succès !", icon="✅")
                    st.success(f"🎉 Offre #{new_id} enregistrée avec embedding automatique !")
                    
                    # Invalider cache pour mise à jour immédiate
                    from data_loaders import load_offres_with_embeddings
                    load_offres_with_embeddings.clear()
                    
                    st.info("💡 L'offre apparaîtra dans le matching immédiatement")
                    
                    # Bouton ajouter autre offre
                    if st.button("➕ Ajouter une autre offre"):
                        st.rerun()
                else:
                    st.error("❌ Erreur lors de l'insertion")
                    
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")

# ============================================
# AIDE
# ============================================

with st.expander("❓ Aide - Sites supportés"):
    st.markdown("""
    ### 🌐 Sites compatibles
    
    **✅ Scraping automatique :**
    - Indeed France/International
    - APEC
    - France Travail
    - Sites carrière entreprises
    
    **⚠️ Scraping limité :**
    - LinkedIn (nécessite login)
    - Glassdoor (anti-bot fort)
    
    **💡 Astuce :**
    Si le scraping échoue, utilisez le mode "Texte manuel" :
    1. Ouvrir l'offre dans navigateur
    2. Sélectionner tout (Ctrl+A)
    3. Copier (Ctrl+C)
    4. Coller dans le formulaire
    """)