## 👥 Auteurs

- **Bintou DIALLO**
- **Thomas RIPOLL**
---


# 📊 CRM Predict : Anticipation du Risque de Churn

Ce projet a été réalisé dans le cadre du **Mastère Data Engineering et IA (EFREI, 2025-2026)** pour le bloc de certification **RNCP40875 : Pilotage et implémentation de solutions IA**.

---

## 🎯 Contexte et Objectif Métier

Le taux de désabonnement (*churn*) est un défi critique pour les entreprises de services.

L'objectif de ce projet est de fournir aux équipes **Customer Success** un outil d'aide à la décision en temps réel permettant de :

- Prédire la probabilité de résiliation d'un client à partir de son comportement et de ses données financières.
- Comprendre les variables ayant mené à cette prédiction afin de déclencher les bonnes actions de rétention (appels proactifs, remises, campagnes de fidélisation).
- Analyser globalement la santé de la base client via des indicateurs clés (KPIs).

---

## 🚀 Fonctionnalités du Dashboard

L'application **Streamlit** propose une interface interactive comprenant :

- **Un simulateur de profil client** : saisie des données (ancienneté, CSAT, nombre de tickets, usage, etc.) dans une barre latérale.
- **Un double moteur d'Intelligence Artificielle** : comparaison en temps réel des prédictions entre un modèle de Machine Learning classique (**XGBoost**) et un réseau de neurones artificiels (**Deep Learning avec PyTorch**).
- **Une explicabilité des modèles** : affichage d'un graphique d'importance des variables (pour XGBoost) afin d'assurer la transparence des décisions de l'IA.
- **Une analyse exploratoire globale** : visualisation des tendances de churn et de l'engagement client grâce à des graphiques interactifs réalisés avec **Plotly**.

---

## 🛠️ Technologies Utilisées

- **Langage :** Python 3.12
- **Interface Web :** Streamlit
- **Machine Learning :** Scikit-Learn (prétraitement, SMOTE), XGBoost
- **Deep Learning :** PyTorch (Perceptron Multicouche)
- **Visualisation :** Plotly Express, Matplotlib, Seaborn
- **Manipulation de données :** Pandas, NumPy

---

## 📁 Structure du Projet

```text
.
├── data/
│   └── silver_customer_churn_business_dataset.csv
│
├── models/
│   ├── preprocessor.pkl
│   ├── xgb_model.pkl
│   └── dl_model.pth
│
├── plots/
│   └── ...
│
├── eda_notebook_customer.ipynb
├── dashboard.py
├── requirements.txt
└── README.md
```

### Description des fichiers

| Élément | Description |
|---------|-------------|
| `data/` | Dataset nettoyé utilisé pour l'entraînement et les prédictions. |
| `models/` | Contient les modèles entraînés et le pipeline de prétraitement. |
| `plots/` | Graphiques générés lors de l'analyse exploratoire (EDA). |
| `eda_notebook_customer.ipynb` | Notebook d'analyse exploratoire, préparation des données et entraînement des modèles. |
| `dashboard.py` | Application Streamlit. |
| `requirements.txt` | Dépendances Python nécessaires au projet. |
| `README.md` | Documentation du projet. |

---

## ⚙️ Installation et Exécution

### 1. Cloner le dépôt

```bash
git clone <votre-lien-github>
cd <nom-du-dossier>
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
```

Sous **Windows** :

```bash
venv\Scripts\activate
```

Sous **Mac / Linux** :

```bash
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> **Remarque :** Sous Python 3.12, si vous rencontrez des problèmes de compatibilité avec PyTorch, l'application désactive automatiquement **TorchDynamo** afin d'assurer le bon fonctionnement du modèle Deep Learning.

### 4. Lancer l'application

```bash
streamlit run dashboard.py
```

Le dashboard sera accessible à l'adresse :

```text
http://localhost:8501
```

---
