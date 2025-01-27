from pydantic import BaseModel

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Ajoute ici toutes les features nécessaires pour ton modèle

class PredictionResponse(BaseModel):
    prediction: float
