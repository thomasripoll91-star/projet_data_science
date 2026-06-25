import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# On importe notre fonction depuis l'autre fichier
from preprocessing import load_and_preprocess_data

def main():
    print("Chargement et préparation des données...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("../data/customer_churn_business_dataset.csv")
    
    # Calcul du ratio pour gérer le déséquilibre des classes (pour XGBoost)
    ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
    
    # Définition des 4 modèles obligatoires (dont 1 Deep Learning)
    models = {
        "Reg_Logistique": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=ratio),
        "Deep_Learning_MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
    }
    
    best_model_name = ""
    best_score = 0
    best_pipeline = None

    print("\n--- Début de l'entraînement et de l'évaluation des modèles ---")
    
    for name, model in models.items():
        print(f"\nEntraînement de : {name}...")
        
        # Création du pipeline complet (Pre-processing + Modèle)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        
        # Prédictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Métriques
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"Accuracy : {acc:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print("Matrice de Confusion :\n", confusion_matrix(y_test, y_pred))

        print("Rapport de Classification :\n", classification_report(y_test, y_pred))
        
        # Sauvegarder le meilleur modèle basé sur le ROC-AUC
        if roc_auc > best_score:
            best_score = roc_auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n======================================")
    print(f"Meilleur modèle sélectionné : {best_model_name} avec un ROC-AUC de {best_score:.4f}")
    
    # Sauvegarde du meilleur pipeline complet pour l'utiliser en production
    joblib.dump(best_pipeline, '../models/best_pipeline.pkl')
    print("Pipeline complet sauvegardé sous '../models/best_pipeline.pkl'")

if __name__ == "__main__":
    main()