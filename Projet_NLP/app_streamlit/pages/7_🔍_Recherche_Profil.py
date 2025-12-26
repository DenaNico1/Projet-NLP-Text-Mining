"""Page 7 : Recherche par Profil
Recherche d'offres par profil métier, compétences et région
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from data_loader import load_preprocessed_data
from search_utils import search_by_profile, get_regional_alerts

st.set_page_config(page_title="Recherche par Profil", page_icon="🔍", layout="wide")

# Titre
st.title("🔍 Recherche par Profil Métier")
st.markdown("Trouvez les offres qui correspondent à vos critères")

# Chargement données
try:
    df = load_preprocessed_data()
    
    # Charger dictionnaire compétences
    dict_path = Path(__file__).parent.parent.parent / "resultats_nlp" / "dictionnaire_competences.json"
    with open(dict_path, 'r', encoding='utf-8') as f:
        dict_comp = json.load(f)['competences']
    
except Exception as e:
    st.error(f"❌ Erreur chargement données : {e}")
    st.stop()

# Vérifier qu'on a les profils
if 'profil' not in df.columns:
    st.warning("⚠️ Colonne 'profil' manquante. Lancez d'abord les analyses.")
    st.stop()

# ============================================================================
# SIDEBAR : FILTRES
# ============================================================================

st.sidebar.header("🎯 Critères de Recherche")

# Profil métier
profils_disponibles = sorted(df['profil'].dropna().unique())
profil_selectionne = st.sidebar.selectbox(
    "Profil Métier",
    options=profils_disponibles,
    help="Sélectionnez le profil métier recherché"
)

# Région (optionnel)
regions_disponibles = ['Toutes'] + sorted(df['region'].dropna().unique().tolist())
region_selectionnee = st.sidebar.selectbox(
    "Région",
    options=regions_disponibles,
    help="Filtrer par région (optionnel)"
)

if region_selectionnee == 'Toutes':
    region_selectionnee = None

# Compétences requises
st.sidebar.markdown("### 🎓 Compétences Requises")

# Méthode 1 : Sélection dans une liste
competences_selectionnees = st.sidebar.multiselect(
    "Choisir des compétences",
    options=sorted(dict_comp),
    default=[],
    help="Sélectionnez les compétences recherchées"
)

# Méthode 2 : Saisie manuelle (séparées par virgules)
competences_manuelles = st.sidebar.text_input(
    "Ou saisir manuellement (séparées par ,)",
    placeholder="Python, SQL, Docker...",
    help="Séparer par des virgules"
)

# Combiner les deux méthodes
if competences_manuelles:
    comps_manual_list = [c.strip() for c in competences_manuelles.split(',') if c.strip()]
    competences_requises = list(set(competences_selectionnees + comps_manual_list))
else:
    competences_requises = competences_selectionnees

# Afficher les compétences sélectionnées
if competences_requises:
    st.sidebar.success(f"✅ {len(competences_requises)} compétence(s) sélectionnée(s)")

# Nombre de résultats
top_k = st.sidebar.slider(
    "Nombre de résultats",
    min_value=10,
    max_value=100,
    value=50,
    step=10
)

# ============================================================================
# RECHERCHE
# ============================================================================

if st.sidebar.button("🔍 Lancer la Recherche", type="primary"):
    
    with st.spinner("🔄 Recherche en cours..."):
        # Recherche
        results = search_by_profile(
            df=df,
            profil=profil_selectionne,
            competences_required=competences_requises if competences_requises else None,
            region=region_selectionnee,
            top_k=top_k
        )
    
    # Afficher résultats
    st.markdown("---")
    st.subheader(f"📊 Résultats : {len(results)} offre(s) trouvée(s)")
    
    # Alerts régionales
    if region_selectionnee and competences_requises:
        st.markdown("### 🚨 Alerts Régionales")
        
        alerts = get_regional_alerts(
            df=df,
            profil=profil_selectionne,
            region=region_selectionnee,
            competences_user=competences_requises,
            top_n=3
        )
        
        for alert in alerts:
            st.info(alert)
    
    st.markdown("---")
    
    # Afficher les offres
    st.markdown("### 🏆 Offres Recommandées")
    
    for idx, row in results.iterrows():
        # Score de matching
        match_pct = row['match_score'] * 100
        
        # Couleur selon le score
        if match_pct >= 80:
            score_color = "🟢"
        elif match_pct >= 60:
            score_color = "🟡"
        else:
            score_color = "🔴"
        
        with st.expander(f"{score_color} **{row['title']}** - {row['company_name']} ({match_pct:.0f}% match)"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**📍 Localisation**")
                if pd.notna(row['city']) and pd.notna(row['region']):
                    st.write(f"{row['city']}, {row['region']}")
                elif pd.notna(row['city']):
                    st.write(row['city'])
                else:
                    st.write("Non spécifié")
            
            with col2:
                st.markdown(f"**📝 Contrat**")
                st.write(row['contract_type'] if pd.notna(row['contract_type']) else "Non spécifié")
            
            with col3:
                st.markdown(f"**💰 Salaire**")
                if pd.notna(row['salary_annual']):
                    st.write(f"{row['salary_annual']/1000:.0f}k€/an")
                else:
                    st.write("Non spécifié")
            
            # Compétences de l'offre
            st.markdown("**🎓 Compétences demandées**")
            
            if isinstance(row['competences_found'], list) and row['competences_found']:
                # Séparer compétences matchées vs non matchées
                comps_matched = [c for c in row['competences_found'] if c in competences_requises]
                comps_other = [c for c in row['competences_found'] if c not in competences_requises]
                
                # Afficher compétences matchées en vert
                if comps_matched:
                    st.markdown("✅ **Correspondent à vos critères** : " + ", ".join(comps_matched))
                
                # Afficher autres compétences
                if comps_other:
                    st.markdown("📌 **Autres compétences** : " + ", ".join(comps_other[:10]))
                    if len(comps_other) > 10:
                        st.markdown(f"*... et {len(comps_other) - 10} autres*")
            else:
                st.write("Aucune compétence extraite")
            
            # Description (extrait)
            st.markdown("**📄 Description**")
            desc_preview = str(row['description'])[:400] + "..." if len(str(row['description'])) > 400 else str(row['description'])
            st.write(desc_preview)
            
            # Lien vers l'offre
            if pd.notna(row.get('url')):
                st.markdown(f"[🔗 Voir l'offre complète]({row['url']})")
    
    # Export CSV
    st.markdown("---")
    if st.button("📥 Exporter les résultats en CSV"):
        csv = results.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="⬇️ Télécharger CSV",
            data=csv,
            file_name=f"offres_{profil_selectionne.replace(' ', '_')}.csv",
            mime="text/csv"
        )

else:
    # Message initial
    st.info("👈 Configurez vos critères dans la barre latérale et cliquez sur 'Lancer la Recherche'")
    
    # Statistiques générales
    st.markdown("### 📊 Statistiques Générales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Offres Total", f"{len(df):,}")
    
    with col2:
        st.metric("Profils Métiers", len(profils_disponibles))
    
    with col3:
        st.metric("Compétences", len(dict_comp))
    
    # Distribution par profil
    st.markdown("#### 📈 Répartition par Profil")
    
    df_profils = df['profil'].value_counts().reset_index()
    df_profils.columns = ['Profil', 'Nombre']
    df_profils['Pourcentage'] = (df_profils['Nombre'] / len(df) * 100).round(1)
    
    st.dataframe(df_profils, use_container_width=True, hide_index=True)