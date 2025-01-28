import streamlit as st
import requests
import pandas as pd

# URL de l'API (changer pour l'URL déployée si nécessaire)
API_URL = "http://127.0.0.1:8007/predict/"

# Titre de l'application
st.title("Interface de Scoring Crédit")

# Section : Saisie des caractéristiques du client
st.header("Entrer les caractéristiques du client")
form = st.form(key="client_form")

# Création des champs pour chaque feature
features = [
    "EXT_SOURCE_1", "EXT_SOURCE_3", "AMT_CREDIT", "DAYS_BIRTH", "EXT_SOURCE_2",
    "AMT_ANNUITY", "DAYS_EMPLOYED", "AMT_GOODS_PRICE", "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE", "AMT_INCOME_TOTAL", "DAYS_REGISTRATION"
]
client_data = {}
for feature in features:
    client_data[feature] = form.number_input(f"{feature}", value=0.0)

# Bouton de soumission
submit_button = form.form_submit_button(label="Obtenir la prédiction")

if submit_button:
    # Envoyer les données au serveur via l'API
    payload = {"features": list(client_data.values())}
    response = requests.post(API_URL, json=payload)

    # Gérer la réponse
    if response.status_code == 200:
        result = response.json()
        st.subheader("Résultats de la prédiction")
        st.write(f"**Seuil utilisé** : {result['threshold']}")
        st.write(f"**Probabilité de défaut** : {result['probability']}")
        st.write(f"**Classe prédite** : {result['class']}")
    else:
        st.error("Erreur lors de la requête à l'API.")
