import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os


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

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="CRM Predict - Rétention Client", layout="wide", page_icon="📊")

# --- 2. CHARGEMENT DES MODÈLES & DES VRAIES DONNÉES ---
@st.cache_resource
def load_models():
    # Chargement du preprocessor et du modèle de Régression Logistique
    preprocessor = joblib.load('models/preprocessor.pkl')
    # Assurez-vous que le modèle de régression logistique a été sauvegardé sous ce nom
    logreg_model = joblib.load('models/logreg.pkl') 
    
    return preprocessor, logreg_model

preprocessor, logreg_model = load_models()

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
st.sidebar.markdown("---")
st.sidebar.info("🧠 **Modèle actif :** Régression Logistique")
st.sidebar.markdown("---")

with st.sidebar.expander("👤 Profil & Contrat", expanded=True):
    customer_segment = st.selectbox("Segment Client", ['SME', 'Individual', 'Enterprise'])
    contract_type = st.selectbox("Type de contrat", ['Monthly', 'Yearly', 'Quarterly']) 
    signup_channel = st.selectbox("Canal d'acquisition", ['Web', 'Mobile', 'Referral'])
    # Champ conservé car non listé dans les modifications :
    tenure_months = st.number_input("Ancienneté (mois)", min_value=0, value=12)

with st.sidebar.expander("💰 Finances", expanded=True):
    monthly_fee = st.selectbox("Abonnement mensuel (€)", [10, 20, 30, 50, 70, 100], index=2)
    # Champ conservé car non listé dans les modifications :
    total_revenue = st.number_input("Chiffre d'affaires généré (€)", min_value=0.0, value=500.0)
    discount_applied = st.selectbox("Réduction appliquée ?", ['Yes', 'No'])
    price_increase_last_3m = st.selectbox("Hausse de prix (3 derniers mois) ?", ['No', 'Yes'])
    payment_failures = st.selectbox("Échecs de paiement récents ?", [0, 1, 2, 3, 4, 5])

with st.sidebar.expander("🛠️ Support & Satisfaction", expanded=True):
    csat_score = st.selectbox("Satisfaction Client (CSAT)", [1.0, 2.0, 3.0, 4.0, 5.0], index=2)
    survey_response = st.selectbox("Note au dernier sondage", ['Satisfied', 'Neutral', 'Unsatisfied'])
    support_tickets = st.selectbox("Tickets de support ouverts", [0, 1, 2, 3, 4, 5, 6, 7])
    escalations = st.selectbox("Nombre de plaintes (Escalations)", [0, 1, 2, 3, 4])
    # Champ conservé car non listé dans les modifications :
    avg_resolution_time = st.number_input("Temps moyen de résolution (heures)", min_value=0.0, value=24.0)

with st.sidebar.expander("📈 Engagement (Usage)", expanded=True):
    # Champs conservés car non listés dans les modifications :
    last_login_days_ago = st.number_input("Jours depuis dernière connexion", min_value=0.0, value=15.0)
    monthly_logins = st.number_input("Connexions ce mois-ci", min_value=0, value=15)
    
    weekly_active_days = st.selectbox("Jours actifs par semaine", [0, 1, 2, 3, 4, 5, 6, 7], index=3)
    features_used = st.selectbox("Fonctionnalités utilisées", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], index=4)
    referral_count = st.selectbox("Nombre de parrainages", [0, 1, 2, 3, 4, 5, 6, 7])

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
    # Graphique 2 : Comparaison des moyennes d'engagement
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
            # On retire np.log car le pipeline s'en charge déjà
            'last_login_days_ago': [last_login_days_ago],
            'monthly_fee': [monthly_fee],
            'total_revenue': [total_revenue],
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
        
        # 1. Transformation des données saisies
        input_scaled = preprocessor.transform(input_data)
        
        # 2. Prédiction par la Régression Logistique
        probabilite = logreg_model.predict_proba(input_scaled)[0][1]

        # Définition de la classe prédite (Seuil standard de 50%)
        prediction = 1 if probabilite >= 0.5 else 0
        
        # --- AFFICHAGE DES RÉSULTATS ---
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction == 1:
                st.error(f"⚠️ RISQUE ÉLEVÉ DE DÉPART ({probabilite*100:.1f}%)")
                st.write("Ce profil est considéré comme **à risque**.")
            else:
                st.success(f"✅ CLIENT SÉCURISÉ ({probabilite*100:.1f}% de risque)")
                st.write("Ce profil montre des **signaux stables**.")

        with res_col2:
            st.subheader("💡 Interprétation Métier")
            if prediction == 1:
                st.markdown("- **Alerte :** Le modèle a détecté des signaux forts de résiliation.")
                st.markdown("**Action recommandée :** Appel pro-actif avec remise de rétention.")
            else:
                st.markdown("- **Sécurisé :** Le client montre des signaux d'engagement stables.")
                st.markdown("**Action recommandée :** Campagne de fidélisation classique (Upsell).")
                
        # --- 8. INTERPRÉTABILITÉ ET GRAPHIQUE D'IMPORTANCE ---
        st.divider()
        st.subheader("🧠 Ce qui influence ce profil")
        
        try:
            # Vérification du type de modèle pour s'adapter dynamiquement
            is_linear = hasattr(logreg_model, 'coef_')
            is_tree = hasattr(logreg_model, 'feature_importances_')
            
            if is_linear:
                importances = logreg_model.coef_[0]
                st.write("*(Analyse des coefficients du modèle linéaire)*")
            elif is_tree:
                importances = logreg_model.feature_importances_
                st.write("*(Analyse de l'importance des variables du modèle)*")
            else:
                raise ValueError("Le modèle ne possède ni 'coef_' ni 'feature_importances_'.")
                
            feature_names = get_real_names(preprocessor)
            
            if len(importances) > 0 and len(feature_names) == len(importances):
                # On crée un dataframe avec la valeur et sa valeur absolue
                df_importance = pd.DataFrame({
                    'Critère': feature_names,
                    'Valeur': importances,
                    'Impact_Absolu': abs(importances)
                })
                
                # Nettoyage stylisé des noms (regex pour enlever les préfixes du preprocessor)
                df_importance['Critère'] = (df_importance['Critère']
                                            .astype(str)
                                            .str.replace(r'^(cat__|num__|num_log__)', '', regex=True)
                                            .str.replace('_', ' ')
                                            .str.title()) 
                
                # On prend les 10 plus gros impacts (positifs ou négatifs)
                df_top = df_importance.sort_values(by='Impact_Absolu', ascending=False).head(10)
                # On le trie dans l'ordre croissant pour l'affichage élégant avec Plotly
                df_top = df_top.sort_values(by='Impact_Absolu', ascending=True)
                
                # Construction du graphique avec des améliorations visuelles (valeurs sur les barres, nettoyage des axes)
                if is_linear:
                    df_top['Impact sur la rétention'] = np.where(df_top['Valeur'] > 0, 
                                                                 'Augmente le Risque (Churn)', 
                                                                 'Fidélise le Client (Rétention)')
                    color_discrete_map = {
                        'Augmente le Risque (Churn)': '#FF4B4B', 
                        'Fidélise le Client (Rétention)': '#0068C9'
                    }
                    
                    fig4 = px.bar(df_top, 
                                  x='Valeur', 
                                  y='Critère', 
                                  orientation='h',
                                  color='Impact sur la rétention', 
                                  color_discrete_map=color_discrete_map,
                                  text=df_top['Valeur'].apply(lambda x: f"{x:+.2f}"),
                                  title="<b>Top 10 des critères ayant influencé la décision</b>")
                                  
                    fig4.update_traces(textposition='outside')
                    st.caption("💡 **Comment lire ce graphique ?** Les variables poussant vers la droite (en rouge) augmentent la probabilité de résiliation. Celles poussant vers la gauche (en bleu) retiennent le client.")

                else:
                    fig4 = px.bar(df_top, 
                                  x='Valeur', 
                                  y='Critère', 
                                  orientation='h',
                                  text=df_top['Valeur'].apply(lambda x: f"{x:.3f}"),
                                  title="<b>Top 10 des critères les plus importants</b>",
                                  color='Valeur',
                                  color_continuous_scale='Reds')
                                  
                    fig4.update_traces(textposition='outside')
                    fig4.update_layout(coloraxis_showscale=False)
                    st.caption("💡 **Comment lire ce graphique ?** Il représente l'importance relative de chaque critère dans la décision du modèle (plus la barre est grande, plus le critère est décisif).")
                
                # Design global épuré
                fig4.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    margin=dict(l=0, r=40, t=40, b=0),
                    xaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(title=""),
                    showlegend=True if is_linear else False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig4, use_container_width=True)
                
            else:
                st.warning("Oups, les noms de colonnes n'ont pas pu être alignés avec les variables.")
                
        except Exception as e_graph:
            st.error(f"Le graphique d'importance n'a pas pu être généré : {e_graph}")

    except Exception as e:
        st.error(f"Erreur inattendue lors de la prédiction : {e}")
        
else:
    st.info("👈 Renseignez le profil client à gauche et cliquez sur 'Calculer le risque' pour lancer une simulation en direct.")