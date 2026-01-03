# test_nlp.py
from config_db import load_offres_with_nlp

df = load_offres_with_nlp()

print(f"✅ {len(df)} offres chargées")
print(f"✅ Colonnes NLP: {['status', 'profil_assigned', 'competences_found', 'topic_id', 'cluster_id']}")

# Vérifier classifiées
df_class = df[df['status'] == 'classified']
print(f"✅ {len(df_class)} classifiées")
print(f"✅ Profils: {df_class['profil_assigned'].value_counts().head()}")