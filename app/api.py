from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from app.models import model_loader  # Importer la classe ModelLoader

# Initialisation de FastAPI
app = FastAPI()

# Définir le schéma pour les données d'entrée
class ClientData(BaseModel):
    features: dict  # Dictionnaire {nom_feature: valeur}

@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API de prédiction !"}

@app.post("/predict/")
def predict(client_data: ClientData):
    """
    Endpoint pour effectuer une prédiction.
    """
    try:
        if model_loader.model is None:
            raise HTTPException(status_code=500, detail="Modèle non chargé.")
        
        # Récupérer les features du modèle (ordre correct)
        feature_names = model_loader.model.feature_names_in_ if hasattr(model_loader.model, 'feature_names_in_') else [f'feature_{i}' for i in range(model_loader.model.n_features_in_)]
        
        # Construire un dictionnaire avec toutes les features attendues
        full_input = {f: 0 for f in feature_names}  # Valeurs par défaut
        
        # Mettre à jour avec les valeurs envoyées
        for f in client_data.features:
            if f in full_input:  # Vérifier que la feature existe bien
                full_input[f] = client_data.features[f]
        
        # Convertir en DataFrame
        input_df = pd.DataFrame([full_input])
        
        # Prédiction
        result = model_loader.predict(input_df)
        
        return {
            "threshold": model_loader.threshold if model_loader.threshold is not None else 0.55,  # Valeur par défaut si absent
            "probability": round(result["probabilities"][0], 4),
            "class": result["classes"][0]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
