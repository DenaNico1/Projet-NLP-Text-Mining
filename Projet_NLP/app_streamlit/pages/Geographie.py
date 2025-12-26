"""Page 5 : Analyse Géographique"""
import streamlit as st
import sys
from pathlib import Path
import plotly.express as px
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from data_loader import load_preprocessed_data, load_geo_data

st.set_page_config(page_title="Géographie", page_icon="🗺️", layout="wide")
st.title("🗺️  Analyse Géographique")

df = load_preprocessed_data()
geo_data = load_geo_data()

# Top villes
st.subheader("🏙️ Top 10 Villes")
city_counts = df['city'].value_counts().head(10)
fig = px.bar(x=city_counts.values, y=city_counts.index, orientation='h',
             title="Offres par Ville")
st.plotly_chart(fig, use_container_width=True)

# Carte
st.subheader("🗺️  Carte de France")
df_geo = df.groupby('region').agg({
    'offre_id': 'count',
    'salary_annual': 'median',
    'latitude': 'mean',
    'longitude': 'mean'
}).reset_index()
df_geo.columns = ['region', 'nb_offres', 'salaire', 'lat', 'lon']

fig = px.scatter_geo(
    df_geo,
    lat='lat',
    lon='lon',
    size='nb_offres',
    hover_name='region',
    hover_data=['nb_offres', 'salaire'],
    title="Répartition Géographique des Offres",
    projection='natural earth',
    scope='europe'
)
st.plotly_chart(fig, use_container_width=True)

# Spécificités régionales
if geo_data:
    st.subheader("🎯 Spécificités Régionales")
    
    selected_region = st.selectbox("Choisir une région", list(geo_data.keys()))
    
    if selected_region:
        region_info = geo_data[selected_region]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Offres", region_info['nb_offres'])
            st.metric("Salaire Médian", f"{region_info['salaire_median']/1000:.0f}k€")
        
        with col2:
            st.write("**Top Compétences**")
            for comp, count in region_info['top_competences'][:5]:
                st.write(f"• {comp}: {count}")