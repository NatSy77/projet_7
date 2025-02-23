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


# ==========================  CONFIGURATION DE LA PAGE (Accessibilité) ========================== #
st.set_page_config(
    page_title="Dashboard de Crédit Scoring",
    page_icon="📊",
    layout="wide"
)

# ==========================  DÉCOMPRESSION DES DONNÉES ========================== #

zip_path = "streamlit_app/app_test.csv.zip"
csv_path = "streamlit_app/app_test.csv"

if not os.path.exists(csv_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("streamlit_app")
    st.success("✅ Fichier app_test.csv décompressé avec succès !")

# ==========================  CHARGEMENT DES DONNÉES CLIENTS ========================== #

@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

df_clients = load_data()

# ==========================  CONFIGURATION API ========================== #

API_URL = "http://13.36.172.156:8000/predict/"

# ==========================  INTERFACE STREAMLIT ========================== #

st.title("Dashboard de Crédit Scoring")

#  🟢 AJOUTER UNE INTRODUCTION #
st.markdown("""
## 📖 Comment utiliser ce dashboard ?
Bienvenue dans l'outil de scoring de crédit !  
Ce dashboard vous permet de comprendre la décision du modèle pour chaque client sélectionné.

### 🔍 Parcours utilisateur :
1️⃣ **Sélectionnez un client** dans la barre latérale à gauche.  
2️⃣ **Consultez ses informations principales** dans la section affichée.  
3️⃣ **Cliquez sur "Obtenir la prédiction"** pour voir le score de crédit et la décision du modèle.  
4️⃣ **Analysez le graphique** pour comprendre **les variables ayant influencé la décision**.

**🎯 Objectif :** Aider les chargés de relation client à expliquer une décision et à mieux conseiller leurs clients.  

---
""")

st.sidebar.header("Sélection du Client")
client_id = st.sidebar.selectbox("Choisir un ID client", df_clients["SK_ID_CURR"])

client_data = df_clients[df_clients["SK_ID_CURR"] == client_id].drop(columns=["SK_ID_CURR"]).to_dict(orient="records")[0]

st.header("Données du client sélectionné")
st.write(pd.DataFrame(client_data, index=["Valeur"]))

# ==========================  PRÉDICTION VIA API ========================== #

result = None  # Initialisation pour éviter une erreur en cas d'échec API

if st.button("Obtenir la prédiction"):
    response = requests.post(API_URL, json={"features": client_data})

    if response.status_code == 200:
        result = response.json()
        
        st.subheader("Résultat de la Prédiction")
        st.write(f"**Seuil utilisé** : {result['threshold']}")
        st.write(f"**Probabilité de défaut** : {result['probability']}")
        st.write(f"**Classe prédite** : {result['class']}")

        #  🟢 MESSAGE D'INTERPRÉTATION #
        if result["class"] == "Accepté":
            st.success("✅ Félicitations, ce client a un bon score et est **éligible à un crédit**.")
        else:
            st.error("❌ Ce client est **refusé** pour un crédit.")

        # 📖 DESCRIPTION ACCESSIBLE DU GRAPHIQUE #
        st.markdown("📊 **Ce graphique représente la probabilité de défaut du client sous forme de jauge.** Il indique si le client est accepté ou refusé en fonction d'un seuil de décision.")

        #  🟢 GRAPHIQUE #
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["probability"] * 100,
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

        fig.update_layout(font=dict(size=14))  # 🔍 Augmenter la taille du texte pour une meilleure lisibilité
        st.plotly_chart(fig)
# ==========================  CHARGEMENT DE L'IMPORTANCE GLOBALE ========================== #

@st.cache_data
def load_global_feature_importance():
    file_path = os.path.join(os.path.dirname(__file__), "../global_feature_importance.csv")  
    return pd.read_csv(file_path)

# Charger les données d'importance globale
global_feature_importance = load_global_feature_importance()

# Renommer la colonne contenant l'importance globale pour éviter les erreurs
global_feature_importance.rename(columns={global_feature_importance.columns[1]: "Importance"}, inplace=True)

# ==========================  CHARGEMENT DU MODÈLE ========================== #
@st.cache_data
def load_model():
    model_path = "model_pipeline/LightGBM_pipeline.pkl"
    return joblib.load(model_path)

# Charger le modèle globalement
model = load_model()
st.write(f"✅ Modèle chargé : {type(model)}")

# ==========================  IMPORTANCE LOCALE (SHAP) ========================== #

@st.cache_data
def compute_local_feature_importance(client_data):
    df_client = pd.DataFrame([client_data])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df_client)

    feature_importance_local = pd.DataFrame({
        "Feature": df_client.columns,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", ascending=False)

    return feature_importance_local

feature_importance_local = compute_local_feature_importance(client_data)

st.subheader("🔍 Feature Importance Locale")
st.write(feature_importance_local.head())
    
# ==========================  FUSION DES IMPORTANCES GLOBALES & LOCALES ========================== #

# Vérifier que les colonnes ont bien les bons noms après le chargement
global_col = "Importance"  # Nom correct pour l'importance globale
shap_col = "SHAP Value"  # Nom correct pour les valeurs SHAP

# Fusionner l'importance globale et locale
merged_importance = global_feature_importance.merge(
    feature_importance_local, on="Feature", how="outer"
).fillna(0)

# 🛠️ Vérifier les colonnes après fusion
st.write("🛠️ Colonnes disponibles après fusion :", merged_importance.columns)

# Appliquer la transformation logarithmique pour harmoniser les échelles
merged_importance[global_col] = np.log1p(merged_importance[global_col])
merged_importance[shap_col] = np.sign(merged_importance[shap_col]) * np.log1p(abs(merged_importance[shap_col]))

# Ajouter une colonne pour le total impact
merged_importance["Total Impact"] = merged_importance[global_col].abs() + merged_importance[shap_col].abs()

# Trier et limiter aux 10 variables les plus influentes
merged_importance = merged_importance.sort_values(by="Total Impact", ascending=False).head(10)


# ==========================  AFFICHAGE DU GRAPHIQUE "Importance des variables" ========================== #

# 📖 DESCRIPTION ACCESSIBLE DU GRAPHIQUE #
st.markdown("📊 **Ce graphique montre quelles variables ont influencé la décision du modèle.** En bleu, l'importance moyenne des variables sur l'ensemble des clients. En rouge, leur impact pour CE client en particulier.")

color_map = {
    "Importance": "blue",  # 🔵 Importance Globale
    "SHAP Value": "red",  # 🔴 SHAP Value Locale
    "Total Impact": "green"  # 🟢 Total Impact
}

fig = px.bar(
    merged_importance.melt(id_vars="Feature", var_name="Type", value_name="Valeur"),
    x="Valeur", y="Feature", color="Type", orientation="h",
    title="📊 Explication de la décision : Importance Globale vs Locale",
    color_discrete_map=color_map,
    labels={"Feature": "Variable", "Valeur": "Importance (log)"}
)

fig.update_layout(
    title_font_size=18,
    xaxis_title="Importance (log)",
    yaxis_title="Feature",
    xaxis_tickfont_size=14,
    yaxis_tickfont_size=14
)

st.subheader("📊 Explication de la décision du modèle")
st.write("💡 **Ce graphique montre quelles variables ont influencé la décision, avec des couleurs plus contrastées et un texte plus lisible.**")
st.plotly_chart(fig)
