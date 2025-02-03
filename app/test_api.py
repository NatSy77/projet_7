import pytest
from fastapi.testclient import TestClient
from app.api import app

# Création du client de test pour FastAPI
client = TestClient(app)

def test_root():
    """ Teste si l'API renvoie un message d'accueil. """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue dans l'API de prédiction !"}

def test_predict_valid_input():
    """ Teste une requête valide avec des données correctes. """
    payload = {
        "features": {
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_3": 0.7,
            "AMT_CREDIT": 200000,
            "DAYS_BIRTH": -15000,
            "EXT_SOURCE_2": 0.6,
            "AMT_ANNUITY": 25000,
            "DAYS_EMPLOYED": -3000,
            "AMT_GOODS_PRICE": 180000,
            "DAYS_ID_PUBLISH": -1000,
            "DAYS_LAST_PHONE_CHANGE": -500,
            "AMT_INCOME_TOTAL": 150000,
            "DAYS_REGISTRATION": -4000
        }
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "threshold" in data
    assert "probability" in data
    assert "class" in data

def test_predict_missing_field():
    """ Teste une requête avec un champ manquant. """
    payload = {
        "features": {
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_3": 0.7,
            "AMT_CREDIT": 200000  # Manque d'autres champs
        }
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200  # L'API doit fonctionner en mettant des valeurs par défaut

def test_predict_invalid_input():
    """ Teste une requête avec des valeurs invalides. """
    payload = {
        "features": {
            "EXT_SOURCE_1": "invalid",  # Valeur texte au lieu de float
            "EXT_SOURCE_3": 0.7,
            "AMT_CREDIT": 200000
        }
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422  # Erreur attendue