import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(filepath="../data/customer_churn_business_dataset.csv"):
    # Créer un dossier pour sauvegarder les graphiques si on le souhaite
    os.makedirs("../plots", exist_ok=True)
    
    print("--- DÉMARRAGE DE L'ANALYSE EXPLORATOIRE (EDA) ---")
    df = pd.read_csv(filepath)
    
    print(f"\n1. Aperçu du dataset : {df.shape[0]} lignes et {df.shape[1]} colonnes.")
    print("\n2. Valeurs manquantes par colonne :\n", df.isnull().sum()[df.isnull().sum() > 0])
    
    # Configuration globale de Seaborn
    sns.set_theme(style="whitegrid")
    
    # --- GRAPHIQUE 1 : Distribution de la variable cible (Churn) ---
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='churn', data=df, palette='Set2')
    plt.title("Distribution de la variable cible (Churn)")
    plt.xlabel("Churn (0 = Non, 1 = Oui)")
    plt.ylabel("Nombre de clients")
    plt.savefig("../plots/1_churn_distribution.png", bbox_inches='tight')
    print("\nAffichage du Graphique 1 : Fermez la fenêtre pour continuer...")
    plt.show()

    # --- GRAPHIQUE 2 : Corrélation des variables numériques ---
    # On sélectionne uniquement les colonnes numériques pour la matrice
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['churn'], errors='ignore')
    
    plt.figure(figsize=(12, 10))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title("Matrice de corrélation des variables numériques")
    plt.savefig("../plots/2_correlation_matrix.png", bbox_inches='tight')
    print("Affichage du Graphique 2 : Fermez la fenêtre pour continuer...")
    plt.show()

    # --- GRAPHIQUE 3 : Churn en fonction du type de contrat ---
    plt.figure(figsize=(8, 5))
    sns.countplot(x='contract_type', hue='churn', data=df, palette='Set1')
    plt.title("Impact du type de contrat sur le Churn")
    plt.xlabel("Type de contrat")
    plt.ylabel("Nombre de clients")
    plt.savefig("../plots/3_churn_by_contract.png", bbox_inches='tight')
    print("Affichage du Graphique 3 : Fermez la fenêtre pour terminer l'EDA.")
    plt.show()
    
    print("\n--- FIN DE L'EDA. Les graphiques ont été sauvegardés dans le dossier 'plots/'. ---")

if __name__ == "__main__":
    run_eda()