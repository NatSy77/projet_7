# Projet 7 - Implementer et déployer un modele de scoring via une API

## 📌 Objectif du projet
Ce projet vise à déployer un modèle de scoring de crédit sous forme d’API accessible via FastAPI et intégrée à une application Streamlit.

## 📁 Structure du projet
- `app/` : Contient l’API FastAPI (`api.py`)
- `streamlit_app/` : Contient l’application Streamlit (`app.py`)
- `model_pipeline/` : Contient le modèle entraîné et les scripts de préparation
- `tests/` : Contient les tests unitaires (`test_api.py`)
- `.github/workflows/` : Contient le workflow GitHub Actions pour le déploiement
- `requirements.txt` : Liste des dépendances

## 🚀 Déploiement & CI/CD
🔹 CI/CD : GitHub Actions automatise les tests ( pytest) et le déploiement sur AWS EC2 .
🔹 Docker : L'application est conteneurisée et stockée sur Docker Hub .
🔹 Déploiement : L'API et le tableau de bord Streamlit sont hébergés sur AWS EC2 .

## 📊 Fonctionnalités du Dashboard interactif
🔍 Visualisation du scoring client

- Affichage du score et de sa probabilité (est-il loin du seuil ?)

- Interprétation claire du score pour une personne non experte en data science

📌 Analyse des informations clients

- Affichage des principales informations descriptives d'un client

- Comparaison du client à l'ensemble des clients ou à un groupe similaire via un filtre

🎨 Accessibilité et ergonomie : Conformité aux critères WCAG pour améliorer l'accessibilité (couleurs, contraste, redimensionnement)

☁️ Hébergement et accessibilité : Déploiement sur AWS EC2 pour permettre un accès multi-utilisateurs

🛠️ Fonctionnalités avancées (optionnelles)
🔹 Ajout de LIME pour expliquer instantanément les décisions du modèle

## 🔧 Installation et utilisation
1. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
2. **Lancer l'API FastAPI** :
   uvicorn app.api:app --host 0.0.0.0 --port 8000
3. **Lancer l'application Streamlit** :
   streamlit run streamlit_app/app.py

##  Tests unitaires
Les tests pytestsont intégrés dans GitHub Actions : 
pytest app/test_api.py

## Liens
**API** : http://13.36.179.77:8000/docs

**APP Streamlit** : http://13.36.179.77:8501
