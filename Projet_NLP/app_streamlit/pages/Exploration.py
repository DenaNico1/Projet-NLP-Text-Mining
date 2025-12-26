"""
Page 2 : Exploration des Offres
Recherche et filtrage
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from data_loader import load_preprocessed_data, filter_dataframe

st.set_page_config(page_title="Exploration", page_icon="🔍", layout="wide")
st.title("🔍 Explorer les Offres")

df = load_preprocessed_data()

# Filtres
with st.sidebar:
    st.header("🎯 Filtres")
    
    search = st.text_input("🔎 Recherche", placeholder="Python, Machine Learning...")
    
    regions = st.multiselect(
        "🗺️  Régions",
        options=df['region'].dropna().unique(),
        default=[]
    )
    
    contrats = st.multiselect(
        "📝 Types de Contrat",
        options=df['contract_type'].dropna().unique(),
        default=[]
    )
    
    sources = st.multiselect(
        "🏢 Sources",
        options=df['source_name'].unique(),
        default=[]
    )

# Appliquer filtres
filters = {
    'search': search,
    'regions': regions,
    'contrats': contrats,
    'sources': sources
}

df_filtered = filter_dataframe(df, filters)

st.write(f"**{len(df_filtered)} offres trouvées**")

# Affichage
for idx, row in df_filtered.head(50).iterrows():
    with st.expander(f"🏢 {row['title']} - {row['company_name']}"):
        col1, col2, col3 = st.columns(3)
        col1.write(f"📍 **Lieu** : {row['city']}, {row['region']}")
        col2.write(f"📝 **Contrat** : {row['contract_type']}")
        
        if pd.notna(row['salary_annual']):
            col3.write(f"💰 **Salaire** : {row['salary_annual']/1000:.0f}k€")
        
        st.write("**Description** :")
        st.write(row['description'][:500] + "..." if len(str(row['description'])) > 500 else row['description'])
        
        if pd.notna(row['url']):
            st.markdown(f"[🔗 Voir l'offre]({row['url']})")

# Export
if st.button("📥 Exporter en CSV"):
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="Télécharger CSV",
        data=csv,
        file_name="offres_filtrees.csv",
        mime="text/csv"
    )