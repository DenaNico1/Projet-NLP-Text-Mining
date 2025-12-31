"""
PAGE 6 : VISUALISATIONS 3D
Embeddings projections interactives
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VIZ_DIR

st.title(" Visualisations 3D Interactives")
st.markdown("Projections embeddings style TensorFlow Projector")

st.markdown("---")

# INDEX 3D
index_3d_path = VIZ_DIR / '3d_interactive' / 'index.html'

if index_3d_path.exists():
    st.info(" Contrôles : Rotation (clic gauche) | Zoom (molette) | Pan (clic droit)")
    
    tab1, tab2, tab3 = st.tabs(["Offres par Profil", "Offres Animées", "Compétences"])
    
    with tab1:
        st.subheader(" Offres par Profil (UMAP 3D)")
        path1 = VIZ_DIR / '3d_interactive' / 'projector_offres_profils_3d.html'
        if path1.exists():
            with open(path1, 'r', encoding='utf-8') as f:
                html1 = f.read()
            components.html(html1, height=900, scrolling=True)
    
    with tab2:
        st.subheader(" Rotation Animée")
        path2 = VIZ_DIR / '3d_interactive' / 'projector_offres_animated.html'
        if path2.exists():
            with open(path2, 'r', encoding='utf-8') as f:
                html2 = f.read()
            components.html(html2, height=900, scrolling=True)
    
    with tab3:
        st.subheader(" Compétences 3D")
        path3 = VIZ_DIR / '3d_interactive' / 'projector_competences_all_3d.html'
        if path3.exists():
            with open(path3, 'r', encoding='utf-8') as f:
                html3 = f.read()
            components.html(html3, height=900, scrolling=True)
else:
    st.warning("""
     Visualisations 3D non disponibles.
    
    Exécuter : `python 7_visualisations_3d_projector.py`
    """)

st.markdown("---")

st.success("""
💡 **Astuces Navigation 3D :**
- Double-clic pour reset vue
- Hover sur points pour détails
- Légende cliquable pour filtrer
""")
