"""Page 3 : Analyse Compétences"""
import streamlit as st
import sys
from pathlib import Path
import plotly.express as px
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from data_loader import load_competences_data

st.set_page_config(page_title="Compétences", page_icon="🎓", layout="wide")
st.title("🎓 Analyse des Compétences")

comp_data = load_competences_data()

if comp_data:
    # Top compétences
    st.subheader("🏆 Top 30 Compétences")
    top_comps = comp_data['top_competences'][:30]
    
    df_comps = pd.DataFrame(top_comps)
    fig = px.bar(df_comps, x='percentage', y='competence', orientation='h',
                 title="Pourcentage d'Offres par Compétence")
    st.plotly_chart(fig, use_container_width=True)
    
    # Word Cloud
    st.subheader("☁️ Word Cloud")
    wc_path = Path("../resultats_nlp/visualisations/wordcloud_competences.png")
    if wc_path.exists():
        img = Image.open(wc_path)
        st.image(img, use_column_width=True)
    
    # Co-occurrences
    st.subheader("🔗 Top Co-occurrences")
    if 'cooccurrences' in comp_data:
        coocs = comp_data['cooccurrences'][:20]
        for cooc in coocs:
            st.write(f"• **{cooc['comp1']}** + **{cooc['comp2']}** : {cooc['count']} offres")
else:
    st.warning("Données de compétences non disponibles")