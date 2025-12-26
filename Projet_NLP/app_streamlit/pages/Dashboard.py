"""
Page 1 : Dashboard Général
Vue d'ensemble du marché
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from data_loader import load_preprocessed_data, get_kpis

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Dashboard Général")

# Chargement
df = load_preprocessed_data()
kpis = get_kpis(df)

# KPIs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Offres", f"{kpis['total_offres']:,}")
col2.metric("Entreprises", f"{kpis['nb_entreprises']:,}")
col3.metric("Régions", kpis['nb_regions'])
col4.metric("% CDI", f"{kpis['pct_cdi']:.0f}%")
col5.metric("Salaire Médian", f"{kpis['salaire_median']/1000:.0f}k€" if kpis['salaire_median'] else "N/A")

st.markdown("---")

# Graphiques
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Répartition par Source")
    source_counts = df['source_name'].value_counts()
    fig = px.pie(values=source_counts.values, names=source_counts.index, 
                 title="Sources des Offres")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📝 Types de Contrats")
    contract_counts = df['contract_type'].value_counts().head(10)
    fig = px.bar(x=contract_counts.values, y=contract_counts.index, 
                 orientation='h', title="Top 10 Types de Contrats")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🗺️  Top 10 Régions")
    region_counts = df['region'].value_counts().head(10)
    fig = px.bar(x=region_counts.values, y=region_counts.index,
                 orientation='h', title="Offres par Région")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🏢 Top 10 Entreprises")
    company_counts = df['company_name'].value_counts().head(10)
    fig = px.bar(x=company_counts.values, y=company_counts.index,
                 orientation='h', title="Entreprises qui Recrutent")
    st.plotly_chart(fig, use_container_width=True)

# Évolution temporelle
st.markdown("---")
st.subheader("📈 Évolution Temporelle")

df_temp = df[df['scraped_at'].notna()].copy()
if len(df_temp) > 0:
    df_temp['date'] = pd.to_datetime(df_temp['scraped_at'])
    df_temp['week'] = df_temp['date'].dt.to_period('W').astype(str)
    weekly = df_temp.groupby('week').size().reset_index(name='count')
    
    fig = px.line(weekly, x='week', y='count', title="Offres par Semaine")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Données temporelles non disponibles")