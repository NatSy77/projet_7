import mlflow.pyfunc
from pathlib import Path

# Définir le chemin du modèle
MODEL_PATH = Path(__file__).parent.parent / "modèle_pipeline"

# Classe pour charger et utiliser le modèle
class ModelLoader:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Charger le modèle MLflow depuis le dossier spécifié.
        """
        try:
            print(f"Chargement du modèle depuis : {self.model_path}")
            self.model = mlflow.pyfunc.load_model(str(self.model_path))
            print("Modèle chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            self.model = None

    def predict(self, input_data):
        """
        Effectuer une prédiction avec le modèle chargé.
        :param input_data: Données d'entrée (DataFrame ou dictionnaire).
        :return: Résultats des prédictions.
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas chargé.")
        try:
            return self.model.predict(input_data)
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            raise e

# Initialiser le modèle globalement
model_loader = ModelLoader()
