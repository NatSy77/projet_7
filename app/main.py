from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from app.models import model_loader

app = FastAPI()

# Définir le schéma pour les données d'entrée
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Ajoute toutes les colonnes nécessaires ici

@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API de prédiction !"}

@app.post("/predict/")
def predict(data: InputData):
    """
    Endpoint pour effectuer une prédiction.
    """
    try:
        # Convertir les données d'entrée en DataFrame
        input_df = pd.DataFrame([data.dict()])
        prediction = model_loader.predict(input_df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
