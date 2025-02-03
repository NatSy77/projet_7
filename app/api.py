from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Union  # Import de Union pour Python < 3.10
import pandas as pd
import joblib
from app.models import model_loader

# Initialisation de FastAPI
app = FastAPI()

# Définition plus souple des types avec Union
class ClientData(BaseModel):
    features: Dict[str, Union[float, int]]  # Accepte float et int

@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API de prédiction !"}

@app.post("/predict/")
def predict(client_data: ClientData):
    """ Endpoint pour effectuer une prédiction. """
    try:
        if model_loader.model is None:
            raise HTTPException(status_code=500, detail="Modèle non chargé.")

        # Récupérer les features du modèle
        feature_names = model_loader.model.feature_names_in_ if hasattr(model_loader.model, 'feature_names_in_') else [f'feature_{i}' for i in range(model_loader.model.n_features_in_)]

        full_input = {f: 0 for f in feature_names}  # Valeurs par défaut

        # Vérifier et convertir les valeurs en float (si int)
        for f in client_data.features:
            if f in full_input:
                if isinstance(client_data.features[f], (int, float)):  # Vérifie que c'est un nombre
                    full_input[f] = float(client_data.features[f])
                else:
                    raise ValueError(f"Valeur incorrecte pour {f}: {client_data.features[f]} (attendu: nombre)")

        input_df = pd.DataFrame([full_input])
        result = model_loader.predict(input_df)

        return {
            "threshold": model_loader.threshold if model_loader.threshold is not None else 0.55,
            "probability": round(result["probabilities"][0], 4),
            "class": result["classes"][0]
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
