"""
7. Analyse Stacks × Salaires
Corrélations entre stacks techniques et salaires

Auteur: Projet NLP Text Mining
Date: Décembre 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from utils import ResultSaver

def main():
    print("="*70)
    print("💰 ÉTAPE 7 : STACKS × SALAIRES")
    print("="*70)
    
    saver = ResultSaver()
    
    with open('../resultats_nlp/models/data_with_topics.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Filtrer offres avec salaire
    df_sal = df[df['salary_annual'].notna()].copy()
    
    print(f"\n📊 Offres avec salaire: {len(df_sal)} ({len(df_sal)/len(df)*100:.1f}%)")
    print(f"   Salaire médian global: {df_sal['salary_annual'].median():.0f}€")
    
    # ==========================================
    # 1. SALAIRE PAR COMPÉTENCE
    # ==========================================
    print("\n💼 Salaire par compétence (Top 30)...")
    
    # Compter compétences
    all_comps = [c for cs in df_sal['competences_found'] for c in cs]
    comp_counter = Counter(all_comps)
    top_30_comps = [c for c, _ in comp_counter.most_common(30)]
    
    comp_salaries = {}
    for comp in top_30_comps:
        # Offres avec cette compétence
        mask = df_sal['competences_found'].apply(lambda x: comp in x)
        df_comp = df_sal[mask]
        
        if len(df_comp) >= 10:  # Au moins 10 offres
            comp_salaries[comp] = {
                'count': len(df_comp),
                'median': df_comp['salary_annual'].median(),
                'mean': df_comp['salary_annual'].mean(),
                'q25': df_comp['salary_annual'].quantile(0.25),
                'q75': df_comp['salary_annual'].quantile(0.75)
            }
    
    # Trier par salaire médian
    sorted_comps = sorted(
        comp_salaries.items(),
        key=lambda x: x[1]['median'],
        reverse=True
    )
    
    print(f"\n🏆 Top 15 compétences les mieux rémunérées:")
    for comp, stats in sorted_comps[:15]:
        print(f"   {comp:<30s}: {stats['median']:6.0f}€ (n={stats['count']})")
    
    # ==========================================
    # 2. STACKS TECHNIQUES TYPIQUES
    # ==========================================
    print("\n🔧 Identification des stacks techniques...")
    
    # Définir des stacks prédéfinis
    stacks_definition = {
        'Data Analyst': ['Python', 'SQL', 'Pandas', 'Excel'],
        'ML Engineer': ['Python', 'TensorFlow', 'Docker', 'Kubernetes'],
        'Data Engineer': ['Python', 'Spark', 'Airflow', 'SQL'],
        'MLOps': ['Docker', 'Kubernetes', 'MLflow', 'CI/CD'],
        'BI Analyst': ['Power BI', 'Tableau', 'SQL'],
        'Deep Learning': ['PyTorch', 'TensorFlow', 'Deep Learning'],
        'NLP Engineer': ['NLP', 'Transformers', 'Python']
    }
    
    stack_results = {}
    
    for stack_name, required_comps in stacks_definition.items():
        # Trouver offres avec au moins 2 compétences du stack
        mask = df_sal['competences_found'].apply(
            lambda comps: sum(1 for c in required_comps if c in comps) >= 2
        )
        
        df_stack = df_sal[mask]
        
        if len(df_stack) >= 5:
            stack_results[stack_name] = {
                'count': len(df_stack),
                'salary_median': df_stack['salary_annual'].median(),
                'salary_mean': df_stack['salary_annual'].mean(),
                'competences': required_comps
            }
            
            print(f"\n   {stack_name}:")
            print(f"      Offres: {len(df_stack)}")
            print(f"      Salaire médian: {stack_results[stack_name]['salary_median']:.0f}€")
    
    # ==========================================
    # 3. SALAIRE PAR NIVEAU D'EXPÉRIENCE
    # ==========================================
    print("\n🎓 Salaire par expérience...")
    
    exp_mapping = {
        'D': 'Débutant',
        'E': 'Expérimenté',
        'S': 'Senior'
    }
    
    exp_salaries = {}
    for code, label in exp_mapping.items():
        df_exp = df_sal[df_sal['experience_level'] == code]
        if len(df_exp) >= 10:
            exp_salaries[label] = {
                'count': len(df_exp),
                'median': df_exp['salary_annual'].median()
            }
            print(f"   {label:<15s}: {exp_salaries[label]['median']:6.0f}€ (n={len(df_exp)})")
    
    # ==========================================
    # 4. SALAIRE PAR TYPE DE CONTRAT
    # ==========================================
    print("\n📝 Salaire par type de contrat...")
    
    contract_salaries = {}
    for contract in ['CDI', 'CDD', 'Stage', 'Alternance']:
        df_contract = df_sal[df_sal['contract_type'] == contract]
        if len(df_contract) >= 10:
            contract_salaries[contract] = {
                'count': len(df_contract),
                'median': df_contract['salary_annual'].median()
            }
            print(f"   {contract:<15s}: {contract_salaries[contract]['median']:6.0f}€ (n={len(df_contract)})")
    
    # ==========================================
    # 5. SALAIRE PAR RÉGION
    # ==========================================
    print("\n🗺️  Salaire par région (Top 10)...")
    
    region_salaries = {}
    top_regions = df_sal['region'].value_counts().head(10).index
    
    for region in top_regions:
        df_region = df_sal[df_sal['region'] == region]
        if len(df_region) >= 10:
            region_salaries[region] = {
                'count': len(df_region),
                'median': df_region['salary_annual'].median()
            }
            print(f"   {region:<30s}: {region_salaries[region]['median']:6.0f}€ (n={len(df_region)})")
    
    # ==========================================
    # 6. VISUALISATIONS
    # ==========================================
    print("\n📊 Création visualisations...")
    
    # 6.1 Box plot compétences
    df_viz_comp = []
    for comp, stats in sorted_comps[:20]:
        mask = df_sal['competences_found'].apply(lambda x: comp in x)
        salaries = df_sal[mask]['salary_annual'].tolist()
        for sal in salaries:
            df_viz_comp.append({'Compétence': comp, 'Salaire': sal})
    
    df_comp_plot = pd.DataFrame(df_viz_comp)
    
    fig = px.box(
        df_comp_plot,
        x='Compétence',
        y='Salaire',
        title='Distribution Salariale par Compétence (Top 20)',
        labels={'Salaire': 'Salaire Annuel (€)'}
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=600)
    saver.save_visualization(fig, 'salaires_par_competence.html')
    
    # 6.2 Bar chart stacks
    if stack_results:
        df_stacks = pd.DataFrame([
            {'Stack': name, 'Salaire': data['salary_median'], 'Nb_offres': data['count']}
            for name, data in stack_results.items()
        ]).sort_values('Salaire', ascending=False)
        
        fig = px.bar(
            df_stacks,
            x='Salaire',
            y='Stack',
            orientation='h',
            title='Salaire Médian par Stack Technique',
            labels={'Salaire': 'Salaire Médian (€)'},
            color='Nb_offres',
            color_continuous_scale='Viridis'
        )
        saver.save_visualization(fig, 'salaires_par_stack.html')
    
    # 6.3 Heatmap région × compétence
    print("   Heatmap région × top compétences...")
    
    top_10_comps = [c for c, _ in comp_counter.most_common(10)]
    top_5_regions = df_sal['region'].value_counts().head(5).index
    
    heatmap_data = []
    for region in top_5_regions:
        row = {'Région': region}
        df_region = df_sal[df_sal['region'] == region]
        
        for comp in top_10_comps:
            # % offres avec cette compétence dans cette région
            mask = df_region['competences_found'].apply(lambda x: comp in x)
            pct = mask.sum() / len(df_region) * 100 if len(df_region) > 0 else 0
            row[comp] = pct
        
        heatmap_data.append(row)
    
    df_heatmap = pd.DataFrame(heatmap_data).set_index('Région')
    
    fig = go.Figure(data=go.Heatmap(
        z=df_heatmap.values,
        x=df_heatmap.columns,
        y=df_heatmap.index,
        colorscale='YlOrRd',
        text=df_heatmap.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Présence des Compétences par Région (%)',
        xaxis_title='Compétence',
        yaxis_title='Région',
        height=500
    )
    
    saver.save_visualization(fig, 'heatmap_region_competence.html')
    
    # ==========================================
    # 7. SAUVEGARDE RÉSULTATS
    # ==========================================
    print("\n💾 Sauvegarde...")
    
    results = {
        'salaires_par_competence': {
            comp: {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                   for k, v in stats.items()}
            for comp, stats in sorted_comps[:50]
        },
        'stacks_techniques': {
            name: {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                   for k, v in data.items()}
            for name, data in stack_results.items()
        },
        'salaires_par_experience': {
            label: {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                    for k, v in stats.items()}
            for label, stats in exp_salaries.items()
        },
        'salaires_par_contrat': {
            contract: {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in stats.items()}
            for contract, stats in contract_salaries.items()
        },
        'salaires_par_region': {
            region: {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                     for k, v in stats.items()}
            for region, stats in region_salaries.items()
        }
    }
    
    saver.save_json(results, 'stacks_salaires.json')
    
    print("\n✅ ANALYSE STACKS × SALAIRES TERMINÉE !")
    print(f"\n📊 Résumé:")
    print(f"   - {len(comp_salaries)} compétences analysées")
    print(f"   - {len(stack_results)} stacks identifiés")
    print(f"   - {len(exp_salaries)} niveaux d'expérience")
    print(f"   - {len(region_salaries)} régions")
    
    print(f"\n📁 Fichiers créés:")
    print(f"   - stacks_salaires.json")
    print(f"   - salaires_par_competence.html")
    print(f"   - salaires_par_stack.html")
    print(f"   - heatmap_region_competence.html")


if __name__ == "__main__":
    main()