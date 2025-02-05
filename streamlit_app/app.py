import streamlit as st
import requests
import pandas as pd

# URL de l'API (changer si nécessaire)
API_URL = "http://127.0.0.1:8000/predict/"

# Titre de l'application
st.title("Prédiction de Crédit")

# Section : Saisie des caractéristiques du client
st.header("Entrer les caractéristiques du client")
form = st.form(key="client_form")

# Liste des features importantes (12 affichées)
features = [
    "EXT_SOURCE_1", "EXT_SOURCE_3", "AMT_CREDIT", "DAYS_BIRTH", "AMT_ANNUITY",
    "EXT_SOURCE_2", "AMT_GOODS_PRICE", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE", "DAYS_REGISTRATION", "AMT_INCOME_TOTAL"
]

# Création des champs pour chaque feature
client_data = {}
for feature in features:
    client_data[feature] = form.number_input(f"{feature}", value=0.0)

# Bouton de soumission
submit_button = form.form_submit_button(label="Obtenir la prédiction")

if submit_button:
    # Construire un dictionnaire avec les valeurs par défaut pour toutes les features du modèle
    default_input = {f: 0 for f in features}  # Par défaut, tout à 0
    
    # Mettre à jour uniquement les features affichées
    for f in client_data:
        default_input[f] = client_data[f]
    
    try:
        # Création du payload JSON
        payload = {"features": default_input}
        
        # Envoi de la requête à l'API
        response = requests.post(API_URL, json=payload)
        
        # Vérification du statut de la réponse
        if response.status_code == 200:
            result = response.json()
            st.subheader("Résultats de la prédiction")
            st.write(f"**Seuil utilisé** : {result['threshold']}")
            st.write(f"**Probabilité de défaut** : {result['probability']}")
            st.write(f"**Classe prédite** : {result['class']}")
        else:
            st.error(f"Erreur lors de la requête à l'API : {response.status_code}")
            st.text(f"Message de l'API : {response.text}")  # Afficher le message d'erreur détaillé

    except Exception as e:
        st.error("Erreur lors de la connexion à l'API.")
        st.text(f"Détail de l'erreur : {e}")
