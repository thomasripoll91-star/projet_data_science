import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(filepath="data/customer_churn_business_dataset.csv"):
    # 1. Chargement des données
    df = pd.read_csv(filepath)
    
    # 2. Séparation Target / Features
    # customer_id n'a aucune valeur prédictive, on le retire.
    X = df.drop(columns=['churn', 'customer_id'])
    y = df['churn']
    
    # 3. Identification des colonnes
    numeric_features = [
        'age', 'tenure_months', 'monthly_logins', 'weekly_active_days', 
        'avg_session_time', 'features_used', 'usage_growth_rate', 
        'last_login_days_ago', 'monthly_fee', 'total_revenue', 
        'payment_failures', 'support_tickets', 'avg_resolution_time', 
        'csat_score', 'escalations', 'email_open_rate', 
        'marketing_click_rate', 'nps_score', 'referral_count'
    ]
    
    categorical_features = [
        'gender', 'country', 'city', 'customer_segment', 
        'signup_channel', 'contract_type', 'payment_method', 
        'discount_applied', 'price_increase_last_3m', 
        'complaint_type', 'survey_response'
    ]
    
    # 4. Création des Pipelines de transformation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Remplace les valeurs manquantes par la médiane
        ('scaler', StandardScaler()) # Indispensable pour le Deep Learning
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 5. Assemblage avec ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 6. Split Train/Test (Stratifié pour garder la proportion de churners)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor