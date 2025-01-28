# === Imports nécessaires ===
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Charger le modèle enregistré
model_path = "model_pipeline/model.pkl"  # Chemin relatif vers le fichier
model = joblib.load(model_path)  # Le modèle est directement chargé, pas besoin d'une clé "model"

# le seuil
optimal_threshold = 0.55  # Valeur fixe du seuil

# Initialisation de FastAPI
app = FastAPI()

# Modèle de données pour l'entrée API
class ClientData(BaseModel):
    features: list

@app.post("/predict/")
async def predict(client_data: ClientData):
    try:
        features = np.array(client_data.features).reshape(1, -1)  # Mise en forme des données
        y_prob = model.predict_proba(features)[0][1]  # Probabilité de défaut
        decision = "Refusé" if y_prob >= optimal_threshold else "Accepté"  # Décision basée sur le seuil

        return {
            "threshold": optimal_threshold,
            "probability": round(y_prob, 4),
            "class": decision
        }
    except Exception as e:
        return {"error": str(e)}