
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
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt


# ========================== CONFIG PAGE ========================== #
st.set_page_config(page_title="Dashboard de Crédit Scoring", page_icon="📊", layout="wide")

# ========================== SESSION STATE ========================== #
for key, default in {
    "score_clicked": False,
    "prediction_result": None,
    "lime_clicked": False,
    "lime_exp": None,
    "compare_clicked": False,
    "selected_feature": None,
    "current_client_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ========================== DATA LOADING ========================== #
# Détermine le chemin absolu du dossier courant (même dans Docker)
base_dir = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(base_dir, "donnees_clients.zip")
csv_path = os.path.join(base_dir, "app_test.csv")

# Décompression conditionnelle
if not os.path.exists(csv_path):
    if os.path.exists(zip_path) and zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(base_dir)
        st.success("Fichier app_test.csv décompressé avec succès !")
    else:
        st.error("Le fichier ZIP est manquant ou invalide.")

@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

df_clients = load_data()

@st.cache_data
def load_global_feature_importance():
    file_path = os.path.join(os.path.dirname(__file__), "../global_feature_importance.csv")  
    return pd.read_csv(file_path)

global_feature_importance = load_global_feature_importance()
global_feature_importance.rename(columns={global_feature_importance.columns[1]: "Importance"}, inplace=True)

@st.cache_data
def load_model():
    model_path = "model_pipeline/LightGBM_pipeline.pkl"
    return joblib.load(model_path)

model = load_model()

# ========================== UI ========================== #
API_URL = "http://13.38.80.175:8000/predict/"
st.title("Dashboard de Crédit Scoring")

st.markdown("""
## 📖 Comment utiliser ce dashboard ?
Bienvenue dans l'outil de scoring de crédit !

1️⃣ **Sélectionnez un client**  
2️⃣ **Consultez ses infos et prédisez**  
3️⃣ **Comprenez la décision grâce aux graphiques**  
""")

st.sidebar.header("Sélection du Client")
client_id = st.sidebar.selectbox("Choisir un ID client", df_clients["SK_ID_CURR"])
client_data = df_clients[df_clients["SK_ID_CURR"] == client_id].drop(columns=["SK_ID_CURR"]).to_dict(orient="records")[0]

if st.session_state["current_client_id"] != client_id:
    st.session_state["current_client_id"] = client_id
    st.session_state["score_clicked"] = False
    st.session_state["prediction_result"] = None
    st.session_state["lime_clicked"] = False
    st.session_state["lime_exp"] = None
    st.session_state["compare_clicked"] = False
    st.session_state["selected_feature"] = None

st.header("Données du client sélectionné")
st.write(pd.DataFrame(client_data, index=["Valeur"]))

# ========================== PREDICTION ========================== #
if st.button("Obtenir la prédiction"):
    response = requests.post(API_URL, json={"features": client_data})
    if response.status_code == 200:
        st.session_state["score_clicked"] = True
        st.session_state["prediction_result"] = response.json()

if st.session_state["score_clicked"] and st.session_state["prediction_result"] is not None:
    result = st.session_state["prediction_result"]
    st.subheader("Résultat de la Prédiction")
    st.write(f"**Seuil utilisé** : {result['threshold']}")
    st.write(f"**Probabilité de défaut** : {result['probability']}")
    st.write(f"**Classe prédite** : {result['class']}")
    if result["class"] == "Accepté":
        st.success("✅ Client éligible à un crédit.")
    else:
        st.error("❌ Client refusé pour un crédit.")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result["probability"] * 100,
        title="Probabilité de Défaut (%)",
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
    fig.update_layout(font=dict(size=14))
    st.plotly_chart(fig)

# ========================== LIME ========================== #
@st.cache_data
def compute_lime_explanation(client_data, _model, df_clients):
    client_values = np.array(list(client_data.values())).reshape(1, -1)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=df_clients.drop(columns=["SK_ID_CURR"]).values,
        feature_names=df_clients.drop(columns=["SK_ID_CURR"]).columns,
        class_names=["Refusé", "Accepté"],
        mode="classification"
    )
    return explainer.explain_instance(client_values[0], _model.predict_proba, num_features=10)

st.subheader("🔍 Explication avec LIME")
if st.button("Générer une explication LIME"):
    lime_exp = compute_lime_explanation(client_data, model, df_clients)
    st.session_state["lime_clicked"] = True
    st.session_state["lime_exp"] = lime_exp

if st.session_state["lime_clicked"] and st.session_state["lime_exp"] is not None:
    fig = st.session_state["lime_exp"].as_pyplot_figure()
    st.pyplot(fig)

# ========================== SHAP ========================== #
@st.cache_data
def compute_local_feature_importance(client_data):
    df_client = pd.DataFrame([client_data])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df_client)
    return pd.DataFrame({
        "Feature": df_client.columns,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", ascending=False)

feature_importance_local = compute_local_feature_importance(client_data)

# ========================== GRAPH: IMPORTANCE VARIABLES ========================== #
global_col = "Importance"
shap_col = "SHAP Value"

merged_importance = global_feature_importance.merge(
    feature_importance_local, on="Feature", how="outer"
).fillna(0)
merged_importance[global_col] = np.log1p(merged_importance[global_col])
merged_importance[shap_col] = np.sign(merged_importance[shap_col]) * np.log1p(abs(merged_importance[shap_col]))
merged_importance["Total Impact"] = merged_importance[global_col].abs() + merged_importance[shap_col].abs()
merged_importance = merged_importance.sort_values(by="Total Impact", ascending=False).head(10)

st.subheader("📊 Explication de la décision du modèle")

color_map = {
    "Importance": "blue",
    "SHAP Value": "red",
    "Total Impact": "green"
}

fig = px.bar(
    merged_importance.melt(id_vars="Feature", var_name="Type", value_name="Valeur"),
    x="Valeur", y="Feature", color="Type", orientation="h",
    title="📊 Importance Globale vs Locale",
    color_discrete_map=color_map,
    labels={"Feature": "Variable", "Valeur": "Importance (log)"}
)
fig.update_layout(font=dict(size=14))
st.plotly_chart(fig)

# ========================== COMPARAISON PAR VARIABLE ========================== #
st.subheader("📊 Comparaison à l'ensemble des clients")
st.markdown("**Choisissez une variable à comparer.**")

feature_list = global_feature_importance["Feature"].tolist()
selected_feature = st.selectbox("Choisir une variable", feature_list)

if st.button("Comparer cette variable"):
    st.session_state["compare_clicked"] = True
    st.session_state["selected_feature"] = selected_feature

if st.session_state["compare_clicked"] and st.session_state["selected_feature"] in df_clients.columns:
    feature = st.session_state["selected_feature"]
    
    st.markdown("""
    #### ℹ️ Légende du graphique :
    - **Axe horizontal** : valeurs possibles de la variable sélectionnée  
    - **Axe vertical** : nombre de clients ayant cette valeur  
    - **Barres bleues** : distribution dans l’ensemble des clients  
    - **Ligne noire pointillée** : valeur du client sélectionné
    """)

    fig = px.histogram(df_clients, x=feature, nbins=50,
                       title=f"Distribution de {feature} dans l'ensemble des clients",
                       labels={feature: feature})
    fig.add_vline(x=client_data[feature], line_dash="dash", line_color="black")
    fig.update_layout(font=dict(size=14))
    st.plotly_chart(fig)

