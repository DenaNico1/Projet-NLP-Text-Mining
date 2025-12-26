"""Page 4 : Analyse Salariale"""
import streamlit as st
import sys
from pathlib import Path
import plotly.express as px
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from data_loader import load_preprocessed_data, load_salaires_data

st.set_page_config(page_title="Salaires", page_icon="💰", layout="wide")
st.title("💰 Analyse Salariale")

df = load_preprocessed_data()
sal_data = load_salaires_data()

# Statistiques
df_sal = df[df['salary_annual'].notna()]
if len(df_sal) > 0:
    col1, col2, col3 = st.columns(3)
    col1.metric("Médian", f"{df_sal['salary_annual'].median()/1000:.0f}k€")
    col2.metric("Moyen", f"{df_sal['salary_annual'].mean()/1000:.0f}k€")
    col3.metric("Offres avec salaire", len(df_sal))
    
    # Distribution
    st.subheader("📊 Distribution des Salaires")
    fig = px.histogram(df_sal, x='salary_annual', nbins=30,
                       title="Distribution Salariale")
    st.plotly_chart(fig, use_container_width=True)
    
    # Par région
    st.subheader("🗺️  Salaire par Région")
    sal_region = df_sal.groupby('region')['salary_annual'].median().sort_values(ascending=False).head(10)
    fig = px.bar(x=sal_region.values/1000, y=sal_region.index, orientation='h',
                 labels={'x': 'Salaire Médian (k€)', 'y': 'Région'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Stacks
    if sal_data and 'stacks_techniques' in sal_data:
        st.subheader("🔧 Salaire par Stack Technique")
        stacks = sal_data['stacks_techniques']
        df_stacks = pd.DataFrame([
            {'Stack': k, 'Salaire': v['salary_median']/1000}
            for k, v in stacks.items()
        ]).sort_values('Salaire', ascending=False)
        
        fig = px.bar(df_stacks, x='Salaire', y='Stack', orientation='h')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Données salariales insuffisantes")