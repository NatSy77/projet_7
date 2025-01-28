import mlflow.pyfunc
from pathlib import Path
import joblib

# Définir le chemin du modèle
MODEL_PATH = Path(__file__).parent.parent / "modèle_pipeline"

# Classe pour charger et utiliser le modèle
class ModelLoader:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.threshold = None
        self.load_model()

    def load_model(self):
        """
        Charger le modèle MLflow depuis le dossier spécifié.
        """
        try:
            print(f"Chargement du modèle depuis : {self.model_path}")
            # Charger le modèle et le seuil depuis un fichier enregistré
            model_data = joblib.load(str(self.model_path))
            self.model = model_data["model"]
            self.threshold = model_data["threshold"]
            print(f"Modèle chargé avec succès. Seuil optimal : {self.threshold}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            self.model = None
            self.threshold = None

    def predict(self, input_data):
        """
        Effectuer une prédiction avec le modèle chargé.
        :param input_data: Données d'entrée (DataFrame ou dictionnaire).
        :return: Résultats des prédictions (probabilité et classe prédite).
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas chargé.")
        if self.threshold is None:
            raise ValueError("Le seuil optimal n'est pas défini.")
        
        try:
            # Calculer les probabilités de prédiction
            probabilities = self.model.predict_proba(input_data)[:, 1]  # Probabilité pour la classe positive
            
            # Déterminer les classes en fonction du seuil
            classes = ["Refusé" if prob >= self.threshold else "Accepté" for prob in probabilities]
            
            return {"probabilities": probabilities, "classes": classes}
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            raise e

# Initialiser le modèle globalement
model_loader = ModelLoader()

