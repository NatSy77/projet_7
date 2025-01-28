from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Charger le modèle et le seuil optimal
model_data = joblib.load("path/to/your_model_with_threshold.joblib")
model = model_data["model"]
optimal_threshold = model_data["threshold"]

# Définir le schéma pour les données d'entrée
class InputData(BaseModel):
    EXT_SOURCE_1: float
    EXT_SOURCE_3: float
    AMT_CREDIT: float
    DAYS_BIRTH: float
    EXT_SOURCE_2: float
    AMT_ANNUITY: float
    DAYS_EMPLOYED: float
    AMT_GOODS_PRICE: float
    DAYS_ID_PUBLISH: float
    DAYS_LAST_PHONE_CHANGE: float
    AMT_INCOME_TOTAL: float
    DAYS_REGISTRATION: float

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
        
        # Calculer la probabilité de défaut
        y_prob = model.predict_proba(input_df)[0][1]  # Probabilité de la classe positive
        
        # Déterminer la classe en fonction du seuil optimal
        y_class = "Refusé" if y_prob >= optimal_threshold else "Accepté"
        
        return {
            "threshold": optimal_threshold,
            "probability": round(y_prob, 4),
            "class": y_class
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))