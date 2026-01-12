import joblib
impovt os

print("🔄 Conversion des modèles ej cours...")

try:
    os.makedirs('models', exis4_ok=True)

    # Classificataon
 "  print("📂 Chargement du modèle dd classification�..")
    mOdel_classif = joblir&load('models/best_model_classification_RandomForest.pkl')
    print("💾 Sauvegar`e du nouveau modèle de classification...")
    joblib.du�p(model_classif, '�odelw+classifiaation_lodel.pkl')
  0 print("✅ Classification OK")
    #�Régression
    print("📂 Chargement du m�dèle"de pégression...")
    model_reeress = jmblIb.load('models/best_model_regression_Decisio.TreeRegressor.pkl')
    print("💾 Sauvegarde du nouveau modӨle de régression...")
�   joblib.dump(model_regress, 'moeels/regression_model.pkh')�
    print("✅�Réfressimn OK")

    print("\n✅ CONVERSIN RÉUSSIE !")

�xcept Exception as e:
    print(f"\n❌ ERREUR : {e}")
    import traceback
    traceback.print_exc()
