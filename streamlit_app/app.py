import zipfile
import os
import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go
import shap
import numpy as np
import joblib
import plotly.express as px

# 🟢 Décompression du fichier ZIP si le CSV n'existe pas
zip_path = "streamlit_app/app_test.csv.zip"
csv_path = "streamlit_app/app_test.csv"

if not os.path.exists(csv_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("streamlit_app")  # Extraction dans le même dossier
    st.success("✅ Fichier app_test.csv décompressé avec succès !")

# 🟢 Charger la base client (CORRECTION : suppression de la redondance)
@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

df_clients = load_data()

# 🟢 Vérification des données clients
st.write("### Vérification des clients chargés :")
st.write(df_clients.head())

# 🟢 URL de l'API
API_URL = "http://13.36.172.156:8000/predict/"

# 🟢 Titre de l'application
st.title("Dashboard de Crédit Scoring")

# 🟢 Sidebar : Sélection du client
st.sidebar.header("Sélection du Client")
client_id = st.sidebar.selectbox("Choisir un ID client", df_clients["SK_ID_CURR"])

# 🟢 Récupération des données du client sélectionné
client_data = df_clients[df_clients["SK_ID_CURR"] == client_id].drop(columns=["SK_ID_CURR"]).to_dict(orient="records")[0]

# 🟢 Affichage des données du client
st.header("Données du client sélectionné")
st.write(pd.DataFrame(client_data, index=["Valeur"]))

# 🟢 Bouton de prédiction
if st.button("Obtenir la prédiction"):

    # Requête API
    response = requests.post(API_URL, json={"features": client_data})

    if response.status_code == 200:
        result = response.json()
        
        # Affichage des résultats
        st.subheader("Résultat de la Prédiction")
        st.write(f"**Seuil utilisé** : {result['threshold']}")
        st.write(f"**Probabilité de défaut** : {result['probability']}")
        st.write(f"**Classe prédite** : {result['class']}")

        # 🟢 Ajouter une jauge pour visualiser le score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["probability"] * 100,  # Convertir en pourcentage
            title={"text": "Probabilité de Défaut (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, result["threshold"] * 100], "color": "green"},
                    {"range": [result["threshold"] * 100, 100], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": result["threshold"] * 100
                }
            }
        ))

        st.plotly_chart(fig)  # Affichage de la jauge

    else:
        st.error(f"Erreur API : {response.status_code} - {response.text}")

# 🟢 Charger et afficher la feature importance globale
@st.cache_data
def load_global_feature_importance():
    file_path = os.path.join(os.path.dirname(__file__), "../global_feature_importance.csv")  
    return pd.read_csv(file_path)

global_feature_importance = load_global_feature_importance()

# Affichage test
st.sidebar.subheader("🔎 Feature Importance Globale")
st.write(global_feature_importance.head())

# 🟢 Charger le modèle
@st.cache_data
def load_model():
    model_path = "model_pipeline/LightGBM_pipeline.pkl"
    model = joblib.load(model_path)  # Charger le modèle LightGBM
    return model

model = load_model()
st.write(f"Type du modèle après extraction : {type(model)}")  # Vérification

# 🟢 Calcul de la Feature Importance Locale avec SHAP
@st.cache_data
def compute_local_feature_importance(client_data):
    df_client = pd.DataFrame([client_data])  # Convertir en DataFrame

    explainer = shap.TreeExplainer(model)  # Utiliser le modèle extrait
    shap_values = explainer(df_client)  # Nouvelle méthode pour éviter l'erreur

    feature_importance_local = pd.DataFrame({
        "Feature": df_client.columns,
        "SHAP Value": shap_values.values[0]  # Récupérer les valeurs SHAP correctement
    }).sort_values(by="SHAP Value", ascending=False)

    return feature_importance_local

# Calcul et affichage de la Feature Importance Locale
feature_importance_local = compute_local_feature_importance(client_data)
st.subheader("🔍 Feature Importance Locale")
st.write(feature_importance_local.head())

# 🟢 Affichage du graphique SHAP (Feature Importance Locale vs Globale)
merged_importance = global_feature_importance.merge(
    feature_importance_local, on="Feature", suffixes=("_globale", "_locale")
)

fig = px.bar(
    merged_importance.melt(id_vars="Feature", var_name="Type", value_name="Valeur"),
    x="Valeur", y="Feature", color="Type", orientation="h",
    title="Comparaison Feature Importance : Locale vs Globale"
)

st.subheader("📊 Comparaison Feature Importance")
st.plotly_chart(fig)
