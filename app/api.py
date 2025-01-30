# === Imports nécessaires ===
import numpy as np
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Définir le chemin des fichiers
base_dir = os.path.dirname(os.path.abspath(__file__))  # Récupère le dossier où se trouve api.py
model_path = os.path.join(base_dir, "../model_pipeline/model.pkl")
csv_path = os.path.join(base_dir, "../global_feature_importance.csv")

# Charger le modèle
model = joblib.load(model_path)  # Directement le modèle

# Charger les 20 features importantes
important_features = pd.read_csv(csv_path)["Feature"].head(20).tolist()

# Seuil de décision
optimal_threshold = 0.55

# Initialisation de FastAPI
app = FastAPI()

# Modèle de données pour l'entrée API
class ClientData(BaseModel):
    features: dict  # Dictionnaire {nom_feature: valeur}

@app.post("/predict/")
async def predict(client_data: ClientData):
    try:
        # Vérifier que toutes les features nécessaires sont présentes
        missing_features = [f for f in important_features if f not in client_data.features]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Features manquantes : {missing_features}")

        # Filtrer uniquement les features importantes
        filtered_features = np.array([client_data.features[f] for f in important_features]).reshape(1, -1)

        # Prédiction
        y_prob = model.predict_proba(filtered_features)[0][1]
        decision = "Refusé" if y_prob >= optimal_threshold else "Accepté"

        return {
            "threshold": optimal_threshold,
            "probability": round(y_prob, 4),
            "class": decision
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))