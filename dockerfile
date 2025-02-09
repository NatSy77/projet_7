# Utiliser une image officielle Python
FROM python:3.9

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y cmake gcc g++ libgl1-mesa-glx

# Copier le bon fichier requirements.txt
COPY requirements.txt ./requirements.txt

# Copier les autres fichiers nécessaires
COPY model_pipeline model_pipeline/
COPY app app/
COPY streamlit_app streamlit_app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer les ports pour FastAPI et Streamlit
EXPOSE 8000 8501

# Lancer FastAPI et Streamlit en parallèle
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0"]