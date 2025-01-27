from fastapi import APIRouter
from app.models import predict
from app.schemas import PredictionRequest, PredictionResponse

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    """
    Endpoint pour effectuer une prédiction.
    """
    prediction = predict(request)
    return {"prediction": prediction}
