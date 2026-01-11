"""
PAGE 5 : TOPICS & TENDANCES
LDA, wordclouds, évolution
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR

# data
from data_loaders import load_topics_data
df, topics = load_topics_data()

st.title(" Topics & Tendances")
st.markdown("Thématiques découvertes par LDA")

st.markdown("---")

# TOPICS LDA
st.subheader(" Topics Découverts (LDA)")

if topics and isinstance(topics, dict) and 'topics' in topics:
    topics_dict = topics['topics']
    n_topics = topics.get('n_topics', len(topics_dict))
    
    st.info(f"**{n_topics} topics** identifiés automatiquement")
    
    for topic_name, topic_data in topics_dict.items():
        words = topic_data.get('words', [])
        scores = topic_data.get('scores', [])
        
        # Compter offres si distribution disponible
        count = None
        if 'distribution' in topics and topic_name in topics['distribution']:
            count = topics['distribution'][topic_name]
        
        with st.expander(f"📌 {topic_name}"):
            if words:
                st.markdown(f"**Mots-clés principaux :** {', '.join(words[:10])}")
                
                if count:
                    st.metric("Nombre d'offres", f"{count}")
                
                # Mini bar chart des scores
                if scores and len(scores) >= 10:
                    df_words = pd.DataFrame({
                        'Mot': words[:10],
                        'Score': scores[:10]
                    })
                    
                    fig_words = px.bar(
                        df_words,
                        x='Score',
                        y='Mot',
                        orientation='h',
                        color='Score',
                        color_continuous_scale='Blues'
                    )
                    
                    fig_words.update_layout(
                        template='plotly_dark',
                        height=300,
                        showlegend=False,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(fig_words, use_container_width=True)
else:
    st.warning("❌ Fichier topics_lda.json non disponible ou structure invalide")

st.markdown("---")

# DISTRIBUTION TOPICS
st.subheader(" Distribution des Topics")

if topics and 'distribution' in topics:
    distribution = topics['distribution']
    
    df_dist = pd.DataFrame([
        {'Topic': topic, 'Count': count}
        for topic, count in distribution.items()
    ])
    
    # Trier par count
    df_dist = df_dist.sort_values('Count', ascending=True)
    
    fig = px.bar(
        df_dist,
        x='Count',
        y='Topic',
        orientation='h',
        text='Count',
        color='Count',
        color_continuous_scale='Sunset'
    )
    
    fig.update_traces(textposition='outside')
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title='Nombre d\'offres',
        yaxis_title='',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        topic_max = max(distribution, key=distribution.get)
        st.metric("Topic Dominant", topic_max, f"{distribution[topic_max]} offres")
    
    with col2:
        total = sum(distribution.values())
        st.metric("Total Classé", f"{total:,}")
    
    with col3:
        avg = total / len(distribution)
        st.metric("Moyenne", f"{avg:.0f} offres/topic")

else:
    st.info("ℹ Distribution topics non disponible")

st.markdown("---")

# TF-IDF
st.subheader(" Top Termes TF-IDF")

try:
    df_tfidf = pd.read_csv(RESULTS_DIR / 'tfidf_keywords.csv')
    
    top_tfidf = df_tfidf.head(20)
    
    fig_tfidf = px.bar(
        top_tfidf,
        x='tfidf_score',
        y='terme',
        orientation='h',
        color='tfidf_score',
        color_continuous_scale='Blues'
    )
    
    fig_tfidf.update_layout(
        template='plotly_dark',
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    st.plotly_chart(fig_tfidf, use_container_width=True)

except FileNotFoundError:
    st.warning(" Données TF-IDF non disponibles")

st.markdown("---")

# TENDANCES
st.subheader(" Tendances Émergentes")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🔥 Technologies Montantes**
    - **LLMs & IA Générative** : GPT, Claude, Llama
    - **MLOps** : Industrialisation ML
    - **Data Mesh** : Architecture décentralisée
    - **Streaming** : Kafka, Flink temps réel
    """)

with col2:
    st.markdown("""
    **☁️ Ecosystème Cloud**
    - **AWS** : Leader marché
    - **Azure** : Microsoft IA/ML
    - **GCP** : BigQuery, Vertex AI
    - **Hybrid** : Multi-cloud croissance
    """)

st.markdown("---")

# INTERPRETATION TOPICS
st.subheader("💡 Interprétation des Topics")

if topics and 'topics' in topics:
    interpretations = {
        "Topic 1": " **Environnement Entreprise** - Focus client, équipe, groupe, expérience",
        "Topic 2": " **Engineering & Qualité** - Données, technique, gestion, développement",
        "Topic 3": " **Conseil & Business** - Data, client, transformation, architecture",
        "Topic 4": " **International / Anglophone** - Science, research, engineering, Paris",
        "Topic 5": " **Transformation Digitale** - Inetum, Sopra Steria, big data, cloud",
        "Topic 6": " **Machine Learning** - Modèles, ML, Python, science, stage",
        "Topic 7": " **Secteur Financier/Risques** - Banque, risques, Crédit Agricole, Socotec",
        "Topic 8": " **Analytics & Reporting** - Analyse, tableaux, gestion, stages"
    }
    
    for topic_name, interp in interpretations.items():
        if topic_name in topics['topics']:
            st.markdown(f"- {interp}")