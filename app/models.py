import joblib
from pathlib import Path

# Définir le chemin du modèle
MODEL_PATH = Path(__file__).parent / "../model_pipeline/model.pkl"

class ModelLoader:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.threshold = 0.55  # Définir un seuil par défaut (ajustable)
        self.load_model()

    def load_model(self):
        """ Charger le modèle """
        try:
            print(f"Chargement du modèle depuis : {self.model_path}")
            self.model = joblib.load(self.model_path)  # Charge directement un objet Pipeline
            print(f"Modèle chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            self.model = None

    def predict(self, input_data):
        """ Effectuer une prédiction """
        if self.model is None:
            raise ValueError("Le modèle n'est pas chargé.")

        try:
            # Calculer la probabilité de la classe positive
            probabilities = self.model.predict_proba(input_data)[:, 1]
            
            # Déterminer la classe en fonction du seuil
            classes = ["Refusé" if prob >= self.threshold else "Accepté" for prob in probabilities]
            
            return {"probabilities": probabilities, "classes": classes}
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            raise e

# Initialiser le modèle globalement
model_loader = ModelLoader()
