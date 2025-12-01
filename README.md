Détection Intelligente du Stress par Analyse Faciale
Application  pour détecter automatiquement trois états émotionnels (Normal, Fatigue, Stress) à partir d'expressions faciales.

🎯 Objectif
Classifier automatiquement les états de stress, fatigue et normalité à partir d'images faciales en temps réel.
Performances : 55.87% accuracy | F1-Score Normal: 70.35%
✨ Fonctionnalités

✅ Détection en temps réel via webcam
✅ Upload d'images pour analyse
✅ Interface web intuitive (Gradio)
✅ Visualisation des probabilités
✅ Conseils personnalisés
🚀 Installation

bash# Cloner le repository
git clone https://github.com/VOTRE-USERNAME/FaceStress-AI.git
cd FaceStress-AI

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset FER2013
python data/download_fer2013.py

# Lancer l'application
python app/app.py

