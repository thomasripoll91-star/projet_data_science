import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px  # <-- L'arme secrète pour des graphiques magnifiques

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="CRM Predict - Rétention Client", layout="wide", page_icon="📊")

# --- 2. CHARGEMENT DES MODÈLES & DES VRAIES DONNÉES ---
@st.cache_resource
def load_models():
    preprocessor = joblib.load('API/preprocessor.pkl')
    xgb_model = joblib.load('API/xgb_model.pkl')
    return preprocessor, xgb_model

preprocessor, xgb_model = load_models()

@st.cache_data
def load_data():
    try:
        return pd.read_csv('silver_customer_churn_business_dataset.csv')
    except:
        return pd.read_csv('data/silver_customer_churn_business_dataset.csv')

df = load_data()
df['Statut Client'] = df['churn'].map({0: 'Fidèle', 1: 'Résilié'}) # Préparation pour les graphiques

# --- 3. INTERFACE GRAPHIQUE ---
st.title("📊 CRM Predict : Anticipation du risque de Churn")
st.markdown("""
Bienvenue sur l'outil d'aide à la décision métier. 
Cette interface permet aux équipes Customer Success d'évaluer en temps réel la probabilité de résiliation d'un client et de comprendre les leviers d'action pour le retenir.
""")

# --- 4. BARRE LATÉRALE : SAISIE DES DONNÉES (SIMULATEUR) ---
st.sidebar.header("⚙️ Simuler un profil client")

with st.sidebar.expander("👤 Profil & Contrat", expanded=True):
    customer_segment = st.selectbox("Segment Client", ["SME", "Individual", "Enterprise"])
    contract_type = st.selectbox("Type de contrat", ["Monthly", "Yearly"]) 
    signup_channel = st.selectbox("Canal d'acquisition", ["Web", "Mobile", "Referral"])
    tenure_months = st.number_input("Ancienneté (mois)", min_value=0, value=12)

with st.sidebar.expander("💰 Finances", expanded=True):
    monthly_fee = st.number_input("Abonnement mensuel (€)", min_value=0.0, value=49.99)
    total_revenue = st.number_input("Chiffre d'affaires généré (€)", min_value=0.0, value=500.0)
    discount_applied = st.selectbox("Réduction appliquée ?", ["Yes", "No"])
    price_increase_last_3m = st.selectbox("Hausse de prix (3 derniers mois) ?", ["No", "Yes"])
    payment_failures = st.selectbox("Échecs de paiement récents ?", [0, 1, 2, 3, 4, 5])

with st.sidebar.expander("🛠️ Support & Satisfaction", expanded=True):
    csat_score = st.slider("Satisfaction Client (CSAT)", 1.0, 5.0, 3.0, step=0.5)
    survey_response = st.selectbox("Note au dernier sondage", ["Satisfied", "Neutral", "Unsatisfied"])
    support_tickets = st.number_input("Tickets de support ouverts", min_value=0, value=1)
    escalations = st.number_input("Nombre de plaintes (Escalations)", min_value=0, value=0)
    avg_resolution_time = st.number_input("Temps moyen de résolution (heures)", min_value=0.0, value=24.0)

with st.sidebar.expander("📈 Engagement (Usage)", expanded=True):
    last_login_days_ago = st.number_input("Jours depuis dernière connexion", min_value=0.0, value=15.0)
    monthly_logins = st.number_input("Connexions ce mois-ci", min_value=0, value=15)
    weekly_active_days = st.slider("Jours actifs par semaine", 0, 7, 3)
    features_used = st.number_input("Fonctionnalités utilisées", min_value=1, value=5)
    referral_count = st.number_input("Nombre de parrainages", min_value=0, value=0)

predict_btn = st.sidebar.button("🔮 Calculer le risque", use_container_width=True)

# --- 5. SECTION PRINCIPALE : VRAIS KPIs ---
st.header("📈 Indicateurs Globaux (Base de données actuelle)")
col1, col2, col3 = st.columns(3)

total_clients = len(df)
taux_churn = df['churn'].mean() * 100
csat_moyen = df['csat_score'].mean()

col1.metric("Clients Analysés", f"{total_clients:,}".replace(',', ' '))
col2.metric("Taux de Churn Réel", f"{taux_churn:.1f} %")
col3.metric("Score CSAT Moyen", f"{csat_moyen:.2f} / 5")
st.divider()

# --- 6. ANALYSE DESCRIPTIVE AVANCÉE (AVEC PLOTLY) ---
st.header("📊 Comportement global de la Base Client")
st.markdown("Identifiez les moments critiques et les comportements à risque à l'échelle de l'entreprise.")

col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    # Graphique 1 : Courbe de tendance lissée
    churn_trend = df.groupby('tenure_months')['churn'].mean().reset_index()
    churn_trend['churn'] = churn_trend['churn'] * 100
    
    fig1 = px.line(churn_trend, x='tenure_months', y='churn', 
                   title="<b>Évolution du risque selon l'ancienneté</b>",
                   labels={'tenure_months': 'Mois d\'ancienneté', 'churn': 'Taux d\'attrition (%)'},
                   markers=True, line_shape='spline')
    fig1.update_traces(line_color='#FF4B4B', marker=dict(size=4))
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig1, use_container_width=True)

with col_graph2:
    # Graphique 2 (NOUVEAU) : Comparaison des moyennes d'engagement
    engagement_metrics = df.groupby('Statut Client')[['monthly_logins', 'weekly_active_days', 'features_used']].mean().reset_index()
    engagement_melted = engagement_metrics.melt(id_vars='Statut Client', var_name='Métrique', value_name='Moyenne')
    
    # Renommer les variables pour faire propre
    engagement_melted['Métrique'] = engagement_melted['Métrique'].replace({
        'monthly_logins': 'Connexions / Mois',
        'weekly_active_days': 'Jours Actifs / Semaine',
        'features_used': 'Fonctionnalités Utilisées'
    })
    
    fig2 = px.bar(engagement_melted, x='Métrique', y='Moyenne', color='Statut Client', barmode='group',
                  title="<b>Profil d'engagement moyen</b>",
                  color_discrete_map={'Fidèle': '#0068C9', 'Résilié': '#FF4B4B'})
                  
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=40, b=0), 
                       legend_title=None, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    st.plotly_chart(fig2, use_container_width=True)

st.write("") # Espace

# Graphique 3 : Répartition CSAT en barres élégantes
csat_dist = df['csat_score'].value_counts().reset_index()
csat_dist.columns = ['Score CSAT', 'Nombre de Clients']
csat_dist = csat_dist.sort_values('Score CSAT')

fig3 = px.bar(csat_dist, x='Score CSAT', y='Nombre de Clients', text='Nombre de Clients',
              title="<b>Volume de clients par Niveau de Satisfaction (CSAT)</b>",
              color='Score CSAT', color_continuous_scale='Blues')
fig3.update_traces(textposition='outside', marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_type='category', margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig3, use_container_width=True)
st.divider()

# --- 7. RÉSULTAT DE LA PRÉDICTION ---
st.header("🎯 Résultat de la simulation")

if predict_btn:
    try:
        input_data = pd.DataFrame({
            'customer_segment': [customer_segment],
            'tenure_months': [tenure_months],
            'signup_channel': [signup_channel],
            'contract_type': [contract_type],
            'monthly_logins': [monthly_logins],
            'weekly_active_days': [weekly_active_days],
            'features_used': [features_used],
            'last_login_days_ago': [np.log(max(1.0, last_login_days_ago))],
            'monthly_fee': [monthly_fee],
            'total_revenue': [np.log(max(1.0, total_revenue))],
            'payment_failures': [payment_failures],
            'discount_applied': [discount_applied],
            'price_increase_last_3m': [price_increase_last_3m],
            'support_tickets': [support_tickets],
            'avg_resolution_time': [avg_resolution_time],
            'csat_score': [csat_score],
            'escalations': [escalations],
            'survey_response': [survey_response],
            'referral_count': [referral_count]
        })
        
        input_scaled = preprocessor.transform(input_data)
        prediction = xgb_model.predict(input_scaled)[0]
        probabilite = xgb_model.predict_proba(input_scaled)[0][1]
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction == 1:
                st.error(f"⚠️ RISQUE ÉLEVÉ DE DÉPART ({probabilite*100:.1f}%)")
                st.write("Ce profil correspond à notre **Cluster 0** (Clients en détresse).")
            else:
                st.success(f"✅ CLIENT SÉCURISÉ ({probabilite*100:.1f}% de risque)")
                st.write("Ce profil correspond à notre **Cluster 1 ou 2**.")

        with res_col2:
            st.subheader("💡 Interprétation Métier")
            if prediction == 1:
                st.markdown("- **Alerte :** Le modèle a détecté des signaux forts de résiliation.")
                st.markdown("**Action recommandée :** Appel pro-actif avec remise de rétention.")
            else:
                st.markdown("- **Sécurisé :** Le client montre des signaux d'engagement stables.")
                st.markdown("**Action recommandée :** Campagne de fidélisation classique (Upsell).")
                
        # --- 8. LE GRAPHIQUE D'IMPORTANCE EN PLOTLY ---
        st.divider()
        st.subheader("🧠 Ce qui a influencé l'algorithme (Top 5 des critères)")
        
        try:
            final_model = xgb_model.best_estimator_ if hasattr(xgb_model, 'best_estimator_') else xgb_model
            
            if hasattr(final_model, 'feature_importances_'):
                importances = final_model.feature_importances_
            elif hasattr(final_model, 'coef_'):
                importances = final_model.coef_[0]
            else:
                importances = []

            def get_real_names(prep):
                ct = prep.steps[0][1] if hasattr(prep, 'steps') else prep
                names = []
                for name, transformer, columns in ct.transformers_:
                    if name == 'remainder' and transformer == 'drop':
                        continue
                    if hasattr(transformer, 'get_feature_names_out'):
                        try:
                            names.extend(transformer.get_feature_names_out(columns))
                        except:
                            names.extend(columns)
                    else:
                        names.extend(columns)
                return names
            
            feature_names = get_real_names(preprocessor)
            
            if len(importances) > 0 and len(feature_names) == len(importances):
                df_importance = pd.DataFrame({
                    'Critère': feature_names,
                    'Poids (%)': abs(np.array(importances)) * 100
                }).sort_values(by='Poids (%)', ascending=False).head(5)
                
                df_importance['Critère'] = (df_importance['Critère']
                                            .astype(str)
                                            .str.replace('cat__', '')
                                            .str.replace('num__', '')
                                            .str.replace('num_log__', '')
                                            .str.replace('_', ' ')
                                            .str.title()) 
                
                # Remplacement du vieux st.bar_chart par un beau Plotly horizontal
                df_importance = df_importance.sort_values(by='Poids (%)', ascending=True) # Nécessaire pour Plotly horizontal
                fig4 = px.bar(df_importance, x='Poids (%)', y='Critère', orientation='h',
                              text=df_importance['Poids (%)'].apply(lambda x: f"{x:.1f}%"),
                              color='Poids (%)', color_continuous_scale='Reds')
                
                fig4.update_traces(textposition='outside')
                fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=20, t=20, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig4, use_container_width=True)
                
                st.caption("Ce graphique illustre de manière transparente les variables réelles ayant le plus pesé dans la décision de l'IA.")
            else:
                st.warning("Oups, les noms de colonnes n'ont pas pu être alignés avec les coefficients.")
                
        except Exception as e_graph:
            st.error(f"Le graphique n'a pas pu être généré : {e_graph}")

    except Exception as e:
        st.error(f"Erreur inattendue lors de la prédiction : {e}")
        
else:
    st.info("👈 Renseignez le profil client à gauche et cliquez sur 'Calculer le risque' pour lancer une simulation en direct.")