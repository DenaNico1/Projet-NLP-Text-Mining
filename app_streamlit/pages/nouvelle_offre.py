import streamlit as st
import llm_extraction as ai
import config_db as db

st.markdown("## 📥 Ajouter une nouvelle offre (Assisté par IA)")

# --- ÉTAPE 1 : INPUT ---
st.info("Copiez le texte complet d'une offre. Mistral va extraire les champs clés.")
raw_text = st.text_area("Texte de l'offre", height=200, placeholder="Collez le texte ici...")

# --- ÉTAPE 2 : TRAITEMENT (LOGIQUE) ---
if st.button("✨ Analyser avec Mistral"):
    if not raw_text:
        st.warning("Veuillez coller du texte d'abord.")
    else:
        with st.spinner("Extraction des données via Mistral..."):
            extracted_data = ai.extract_job_info(raw_text)
            
            if "error" in extracted_data:
                st.error(extracted_data["error"])
            else:
                st.success("Extraction réussie !")
                st.session_state['draft_offer'] = extracted_data

# --- ÉTAPE 3 : VISUALISATION JSON (NOUVEAU) ---
if 'draft_offer' in st.session_state:
    # Bloc dépliant pour voir le JSON brut sans polluer l'interface
    with st.expander("🕵️ Voir le JSON brut généré par Mistral (Contrôle Qualité)"):
        st.json(st.session_state['draft_offer'])

    st.divider()
    st.markdown("### 🔍 Vérification avant insertion")
    
    data = st.session_state['draft_offer']
    
    with st.form("validation_form"):
        col1, col2 = st.columns(2)
        
        title = col1.text_input("Titre du poste", value=data.get('title', ''))
        company = col2.text_input("Entreprise", value=data.get('company_name', ''))
        
        col3, col4 = st.columns(2)
        city = col3.text_input("Ville", value=data.get('city', ''))
        
        # Astuce : on essaie de pré-sélectionner le bon index si Mistral a trouvé le type
        contract_options = ["CDI", "CDD", "Freelance", "Stage", "Alternance", "Non spécifié"]
        found_contract = data.get('contract_type', 'Non spécifié')
        try:
            idx = contract_options.index(found_contract)
        except ValueError:
            idx = 5 # Par défaut "Non spécifié"
            
        contract = col4.selectbox("Type de contrat", contract_options, index=idx)
        
        col5, col6 = st.columns(2)
        sal_min = col5.number_input("Salaire Min", value=int(data.get('salary_min') or 0))
        sal_max = col6.number_input("Salaire Max", value=int(data.get('salary_max') or 0))
        
        desc = st.text_area("Description courte", value=data.get('description', ''))
        url = st.text_input("URL d'origine (optionnel)", value=data.get('url', ''))

        submitted = st.form_submit_button("💾 Sauvegarder dans la Base")
        
        if submitted:
            # Construction du paquet final
            final_data = {
                'title': title,
                'company_name': company,
                'city': city,
                'contract_type': contract,
                'salary_min': sal_min if sal_min > 0 else None,
                'salary_max': sal_max if sal_max > 0 else None,
                'description': desc,
                'url': url,
                'source': 'Import Manuel + IA'
            }
            
            # Appel BDD
            new_id = db.add_offre(final_data)
            
            if new_id:
                # 1. On nettoie le brouillon pour fermer le formulaire au prochain chargement
                del st.session_state['draft_offer']
                
                # 2. Message de succès
                st.toast("Offre ajoutée avec succès !", icon="✅")
                st.success(f"🎉 Offre enregistrée (ID: {new_id}) ! Vous pouvez saisir la suivante ci-dessus.")
            