# nouvelle_offre.py (CORRECTION FINALE - CONNEXIONS ISOLÉES)
"""
CORRECTION : Créer nouvelle connexion à chaque opération
Ne PAS partager connexion entre fonctions
"""

import streamlit as st
from app_streamlit import scraper
import llm_extraction as ai
import config_db as db
from data_loaders import ajouter_offre_avec_embedding
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.markdown("## Ajouter une nouvelle offre")
st.markdown("**Sources acceptées :** Indeed France 🇫🇷 • France Travail")

# ============================================
# FONCTION VÉRIFICATION DOUBLON (CONNEXION ISOLÉE)
# ============================================


def check_duplicate_by_external_id(source, job_id):
    """
    Vérifie si offre existe déjà via external_job_id

    CORRECTION : Crée SA PROPRE connexion (isolée)
    """

    # CORRECTION : Créer connexion dédiée
    conn = None

    try:
        conn = db.get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT 
                f.offre_id,
                f.title,
                e.company_name,
                l.city,
                f.created_at
            FROM fact_offres f
            LEFT JOIN dim_entreprise e 
                ON f.entreprise_id = e.entreprise_id
            LEFT JOIN dim_localisation l 
                ON f.localisation_id = l.localisation_id
            LEFT JOIN fact_offres_external fe
                ON fe.offre_id = f.offre_id
            WHERE fe.external_source = %s
            AND fe.external_job_id = %s
            LIMIT 1;
        """,
            (source, job_id),
        )

        result = cur.fetchone()

        if result:
            return (
                True,
                result[0],
                {
                    "offre_id": result[0],
                    "title": result[1],
                    "company": result[2],
                    "city": result[3],
                    "created_at": result[4],
                },
            )

        return False, None, None

    except Exception as e:
        st.error(f"❌ Erreur vérification doublon : {str(e)}")
        return False, None, None

    finally:
        # IMPORTANT : Fermer connexion dédiée
        if conn:
            try:
                conn.close()
            except:
                pass


# ============================================
# SÉLECTION MODE
# ============================================

mode = st.radio(
    "Mode d'ajout", [" URL (Indeed / France Travail)", " Texte manuel"], horizontal=True
)

st.markdown("---")

# ============================================
# MODE 1 : URL
# ============================================

if mode == " URL (Indeed / France Travail)":

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.info(
            "**Indeed France**\n\nFormat URL :\n`https://fr.indeed.com/viewjob?jk=...`"
        )

    with col_info2:
        st.info(
            "**France Travail**\n\nFormat URL :\n`https://francetravail.fr/offre/...`"
        )

    url_input = st.text_input(
        "URL de l'offre",
        placeholder="Collez l'URL ici...",
        help="💡 Clic droit sur titre offre → Copier l'adresse du lien",
    )

    # Afficher correction Indeed
    if url_input and "Indeed" in url_input.lower():
        url_fixed = scraper.fix_indeed_url(url_input)
        if url_fixed != url_input:
            st.caption(f" URL corrigée : {url_fixed}")

    if st.button(" Analyser l'offre", type="primary"):
        if not url_input:
            st.warning(" Veuillez coller une URL")
        else:

            # ============================================
            # VALIDATION SOURCE
            # ============================================

            is_valid, source, message = scraper.validate_source(url_input)

            if not is_valid:
                st.error(message)

                st.markdown(
                    """
                ---
                ### 💡 Alternative
                
                Si vous souhaitez ajouter une offre d'un autre site :
                1. Ouvrir l'offre dans votre navigateur
                2. Sélectionner tout le texte (Ctrl+A)
                3. Copier (Ctrl+C)
                4. Utiliser le mode " Texte manuel"
                """
                )

                st.stop()

            st.success(f"✅ {message}")

            # ============================================
            # VÉRIFICATION DOUBLON (CONNEXION ISOLÉE)
            # ============================================

            with st.spinner("🔍 Vérification doublons..."):
                job_info = scraper.extract_job_id_from_url(url_input)

                # CORRECTION : Pas besoin passer connexion
                is_dup, dup_id, dup_data = check_duplicate_by_external_id(
                    job_info["source"], job_info["job_id"]
                )

            if is_dup:
                st.error("❌ Cette offre existe déjà dans la base !")

                st.markdown(
                    f"""
                **Offre existante :**
                -  ID : #{dup_data['offre_id']}
                -  Titre : {dup_data['title']}
                -  Entreprise : {dup_data['company']}
                -  Ville : {dup_data['city']}
                -  Ajoutée le : {dup_data['created_at'].strftime('%d/%m/%Y')}
                """
                )

                st.info(
                    f" Source : **{job_info['source'].upper()}** | ID : `{job_info['job_id']}`"
                )
                st.stop()

            st.success(f" Offre unique ({job_info['source']}: {job_info['job_id']})")

            # ============================================
            # SCRAPING
            # ============================================

            with st.spinner(f"📡 Récupération offre {source.upper()}..."):
                scrape_result = scraper.smart_scrape(url_input)

            if not scrape_result["success"]:
                st.error(f" {scrape_result['error']}")
                st.info(" Essayez le mode 'Texte manuel'")
            else:
                st.success(f" Offre récupérée ({scrape_result['method']})")

                with st.expander(" Aperçu texte extrait"):
                    st.text(scrape_result["text"][:1000] + "...")

                # ============================================
                # EXTRACTION LLM
                # ============================================

                with st.spinner(" Extraction données Mistral..."):
                    extracted_data = ai.extract_job_info(scrape_result["text"])

                if "error" in extracted_data:
                    st.error(f"❌ Erreur LLM : {extracted_data['error']}")
                else:
                    st.success(" Données extraites !")

                    extracted_data["url"] = url_input
                    extracted_data["job_info"] = job_info

                    st.session_state["draft_offer"] = extracted_data
                    st.rerun()

# ============================================
# MODE 2 : TEXTE MANUEL
# ============================================

else:
    st.info(" Copiez le texte complet d'une offre (tous sites acceptés)")

    raw_text = st.text_area(
        "Texte de l'offre", height=250, placeholder="Collez le texte complet ici..."
    )

    if st.button("✨ Analyser avec Mistral"):
        if not raw_text:
            st.warning("⚠️ Veuillez coller du texte")
        else:
            with st.spinner("🤖 Extraction données Mistral..."):
                extracted_data = ai.extract_job_info(raw_text)

            if "error" in extracted_data:
                st.error(extracted_data["error"])
            else:
                st.success("✅ Extraction réussie !")
                st.session_state["draft_offer"] = extracted_data
                st.rerun()

# ============================================
# FORMULAIRE VALIDATION
# ============================================

if "draft_offer" in st.session_state:

    with st.expander("🕵️ JSON brut Mistral"):
        st.json(st.session_state["draft_offer"])

    st.divider()
    st.markdown("###  Vérification avant insertion")

    data = st.session_state["draft_offer"]

    with st.form("validation_form"):
        col1, col2 = st.columns(2)

        title = col1.text_input("Titre du poste *", value=data.get("title", ""))
        company = col2.text_input("Entreprise *", value=data.get("company_name", ""))

        col3, col4 = st.columns(2)
        city = col3.text_input("Ville *", value=data.get("city", ""))

        contract_options = [
            "CDI",
            "CDD",
            "Freelance",
            "Stage",
            "Alternance",
            "Non spécifié",
        ]
        found_contract = data.get("contract_type", "Non spécifié")
        try:
            idx = contract_options.index(found_contract)
        except ValueError:
            idx = 5

        contract = col4.selectbox("Type de contrat", contract_options, index=idx)

        col5, col6 = st.columns(2)
        sal_min = col5.number_input(
            "Salaire Min (€/an)", value=int(data.get("salary_min") or 0)
        )
        sal_max = col6.number_input(
            "Salaire Max (€/an)", value=int(data.get("salary_max") or 0)
        )

        desc = st.text_area(
            "Description", value=data.get("description", ""), height=150
        )
        url = st.text_input("URL d'origine", value=data.get("url", ""))

        submitted = st.form_submit_button(" Sauvegarder dans la Base", type="primary")

    # ← IMPORTANT : Pas d'indentation ici (fin du formulaire)

    if submitted:  # ← 0 espaces (même niveau que "with st.form")

        if not title or not company or not city:
            st.error(" Titre, Entreprise et Ville sont obligatoires")
        else:

            final_data = {
                "title": title,
                "company_name": company,
                "city": city,
                "contract_type": contract,
                "salary_min": sal_min if sal_min > 0 else None,
                "salary_max": sal_max if sal_max > 0 else None,
                "description": desc,
                "url": url,
                "source": f"Import IA",
            }

            if "job_info" in data:
                final_data["external_source"] = data["job_info"]["source"]
                final_data["external_job_id"] = data["job_info"]["job_id"]

            # Dans nouvelle_offre.py, avant insertion

            # Extraire ville du titre si LLM a échoué
            if not city or city == "Non spécifié":
                #
                import re

                match = re.search(r"-\s*([A-ZÀ-Ý]+(?:\s+[A-ZÀ-Ý]+)?)\s*$", title)
                if match:
                    city = match.group(1).title()
            # ============================================
            # INSERTION + EMBEDDING
            # ============================================

            try:
                emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

                with st.spinner(" Insertion + Calcul embedding..."):
                    new_id = ajouter_offre_avec_embedding(final_data, emb_model)

                if new_id:
                    del st.session_state["draft_offer"]

                    st.balloons()
                    st.success(f" Offre #{new_id} ajoutée avec succès !")

                    from data_loaders import load_offres_with_embeddings

                    load_offres_with_embeddings.clear()

                    st.info(" Disponible immédiatement dans le matching (<3 sec)")

                    if "job_info" in data:
                        st.caption(
                            f"Source : {data['job_info']['source']} | ID : {data['job_info']['job_id']}"
                        )

                    if st.button("➕ Ajouter une autre offre"):
                        st.rerun()
                else:
                    st.error("❌ Erreur insertion")

            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")
                with st.expander(" Détails erreur"):
                    st.exception(e)
